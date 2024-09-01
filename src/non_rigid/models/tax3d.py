from typing import Any, Dict

import lightning as L
import numpy as np
import omegaconf
import plotly.express as px
import rpad.pyg.nets.dgcnn as dgcnn
import rpad.visualize_3d.plots as vpl
import torch
import torch.nn.functional as F
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
import torchvision as tv
import wandb
from diffusers import get_cosine_schedule_with_warmup
from pytorch3d.transforms import Transform3d
from torch import nn, optim
from torch_geometric.nn import fps

from non_rigid.metrics.error_metrics import get_pred_pcd_rigid_errors
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.models.dit.models import DiT_PointCloud_Unc as DiT_pcu
from non_rigid.models.dit.models import (
    DiT_PointCloud_Unc_Cross,
    Rel3D_DiT_PointCloud_Unc_Cross,
    DiT_PointCloud_Cross,
    DiT_PointCloud
)
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd


def DiT_pcu_S(**kwargs):
    return DiT_pcu(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_pcu_xS(**kwargs):
    return DiT_pcu(depth=5, hidden_size=128, num_heads=4, **kwargs)


def DiT_pcu_cross_xS(**kwargs):
    return DiT_PointCloud_Unc_Cross(depth=5, hidden_size=128, num_heads=4, **kwargs)


def Rel3D_DiT_pcu_cross_xS(**kwargs):
    # Embed dim divisible by 3 for 3D positional encoding and divisible by num_heads for multi-head attention
    return Rel3D_DiT_PointCloud_Unc_Cross(
        depth=5, hidden_size=132, num_heads=4, **kwargs
    )

def DiT_PointCloud_Cross_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud_Cross(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)

def DiT_PointCloud_xS(use_rotary, **kwargs):
    # hidden size divisible by 3 for rotary embedding, and divisible by num_heads for multi-head attention
    hidden_size = 132 if use_rotary else 128
    return DiT_PointCloud(depth=5, hidden_size=hidden_size, num_heads=4, **kwargs)


DiT_models = {
    "DiT_pcu_S": DiT_pcu_S,
    "DiT_pcu_xS": DiT_pcu_xS,
    "DiT_pcu_cross_xS": DiT_pcu_cross_xS,
    "Rel3D_DiT_pcu_cross_xS": Rel3D_DiT_pcu_cross_xS,
    # there is no Rel3D_DiT_pcu_xS
    "DiT_PointCloud_Cross_xS": DiT_PointCloud_Cross_xS,
    # TODO: add the SD model here
    "DiT_PointCloud_xS": DiT_PointCloud_xS,
}


def get_model(model_cfg):
    #rotary = "Rel3D_" if model_cfg.rotary else ""
    cross = "Cross_" if model_cfg.name == "df_cross" else ""
    # model_name = f"{rotary}DiT_pcu_{cross}{model_cfg.size}"
    model_name = f"DiT_PointCloud_{cross}{model_cfg.size}"
    return DiT_models[model_name]


class DiffusionTransformerNetwork(nn.Module):
    """
    Network containing the specified Diffusion Transformer architecture.
    """
    def __init__(self, model_cfg=None):
        super().__init__()
        self.dit = get_model(model_cfg)(
            use_rotary=model_cfg.rotary,
            in_channels=model_cfg.in_channels,
            learn_sigma=model_cfg.learn_sigma,
            model_cfg=model_cfg,
        )
    
    def forward(self, x, t, **kwargs):
        return self.dit(x, t, **kwargs)
    

# TODO: do we need a separate training vs inference module? or can we combine them?
class SceneDisplacementTrainingModule(L.LightningModule):
    def __init__(self, network, cfg) -> None:
        super().__init__()
        self.network = network
        self.model_cfg = cfg.model
        self.prediction_type = self.model_cfg.type # flow or point
        self.mode = cfg.mode # train or eval

        # prediction type-specific processing
        if self.prediction_type == "flow":
            self.label_key = "flow"
        elif self.prediction_type == "point":
            self.label_key = "pc"
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

        # mode-specific processing
        if self.mode == "train":
            self.run_cfg = cfg.training 
            # training-specific params
            self.lr = self.run_cfg.lr
            self.weight_decay = self.run_cfg.weight_decay
            self.num_training_steps = self.run_cfg.num_training_steps
            self.lr_warmup_steps = self.run_cfg.lr_warmup_steps
            self.additional_train_logging_period = self.run_cfg.additional_train_logging_period
        elif self.mode == "eval":
            self.run_cfg = cfg.inference
            # inference-specific params
            self.num_trials = cfg.inference.num_trials
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


        # training params
        # self.lr = training_cfg.lr
        # self.weight_decay = training_cfg.weight_decay
        # self.num_training_steps = training_cfg.num_training_steps
        # self.lr_warmup_steps = training_cfg.lr_warmup_steps
        # self.additional_train_logging_period = training_cfg.additional_train_logging_period

        # data params
        self.batch_size = self.run_cfg.batch_size
        self.val_batch_size = self.run_cfg.val_batch_size
        # TODO: it is debatable if the module needs to know about the sample size
        self.sample_size = self.run_cfg.sample_size
        self.sample_size_anchor = self.run_cfg.sample_size_anchor

        # diffusion params
        # self.noise_schedule = model_cfg.diff_noise_schedule
        # self.noise_scale = model_cfg.diff_noise_scale
        self.diff_steps = self.model_cfg.diff_train_steps # TODO: rename to diff_steps
        self.num_wta_trials = self.run_cfg.num_wta_trials
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
            # noise_schedule=self.noise_schedule,
        )

    def configure_optimizers(self):
        assert self.mode == "train", "Can only configure optimizers in training mode."
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def forward(self, batch, t):
        """
        Forward pass to compute training loss.
        """
        pc_action = batch["pc_action"].permute(0, 2, 1) # channel first
        ground_truth = batch[self.label_key].permute(0, 2, 1) # channel first

        model_kwargs = dict(x0=pc_action)

        # run diffusion
        # noise = torch.randn_like(ground_truth) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            model=self.network, 
            x_start=ground_truth, 
            t=t, 
            model_kwargs=model_kwargs,
            # noise=noise,
        )
        loss = loss_dict["loss"].mean()
        return None, loss
    
    @torch.no_grad()
    def predict(self, batch, num_samples, unflatten=False, progress=True, full_prediction=True):
        """
        Predict the output for the given input batch.

        Args:
            full_prediction: if True, the function will return a full dictionary containing flow and 
                point predictions in goal and world frames (including intermediate diffusion steps). 
                If False, the function will only return the prediction type (flow or point) in the goal 
                frame. Set to False for training/validation.
        """
        pc_action = batch["pc_action"].to(self.device)

        # re-shaping and expanding for inference
        bs, sample_size = pc_action.shape[:2]
        pc_action = expand_pcd(pc_action, num_samples).transpose(-1, -2)
        model_kwargs = dict(x0=pc_action)

        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, sample_size, device=self.device)
        pred, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=progress,
            device=self.device,
        )
        # TODO: standardize all permute/transpose calls
        # TODO: is there any reason to return model_kwargs?
        # transform back to channel-last
        pred = pred.permute(0, 2, 1)
        pc_action = pc_action.permute(0, 2, 1)


        if not full_prediction:
            # only return the prediction type in the goal frame
            return {self.label_key: {"pred": pred}}
        else:
            # compute goal and world frame predictions, for flows and points
            T_goal2world = Transform3d(
                matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples)
            )

            if self.prediction_type == "flow":
                pred_flow = pred
                pred_point = pc_action + pred_flow
                # for flow predictions, convert results to point predictions
                results = [
                    pc_action + res.permute(0, 2, 1) for res in results
                ]
            elif self.prediction_type == "point":
                pred_point = pred
                pred_flow = pred_point - pc_action
                results = [
                    res.permute(0, 2, 1) for res in results
                ]

            # compute world frame predictions
            pred_point_world = T_goal2world.transform_points(pred_point)
            pc_action_world = T_goal2world.transform_points(pc_action)
            pred_flow_world = pred_point_world - pc_action_world
            results_world = [
                T_goal2world.transform_points(res) for res in results
            ]

        item = {
            # TODO: do we ened to return model kwargs? if not, unflatten is easy
            # "model_kwargs": ...,
            "flow": {
                "pred": pred_flow,
                "pred_world": pred_flow_world,
            },
            "point": {
                "pred": pred_point,
                "pred_world": pred_point_world,
            },
            "results": results,
            "results_world": results_world,
        }
        return item

    def predict_wta(self, batch, num_samples):
        """
        Predict WTA (winner-take-all) samples, and compute WTA metrics. Unlike predict, this 
        function assumes the ground truth is available.
        """
        ground_truth = batch[self.label_key].to(self.device)
        seg = batch["seg"].to(self.device)

        # re-shaping and expanding for winner-take-all
        bs = ground_truth.shape[0]
        ground_truth = expand_pcd(ground_truth, num_samples)
        seg = expand_pcd(seg, num_samples)

        # generating diffusion predictions
        pred_dict = self.predict(
            batch, num_samples, unflatten=False, progress=True
        )
        pred = pred_dict[self.prediction_type]["pred"]

        # computing error metrics
        cos_sim = flow_cos_sim(pred, ground_truth, mask=True, seg=seg).reshape(bs, num_samples)
        rmse = flow_rmse(pred, ground_truth, mask=True, seg=seg).reshape(bs, num_samples)
        pred = pred.reshape(bs, num_samples, -1, 3)

        # computing winner-take-all metrics
        winner = torch.argmin(rmse, dim=-1)
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_wta = pred[torch.arange(bs), winner]

        # return # TODO: this should return a dict

        return {
            "pred": pred,
            "pred_wta": pred_wta,
            "cos_sim": cos_sim,
            "cos_sim_wta": cos_sim_wta,
            "rmse": rmse,
            "rmse_wta": rmse_wta,
        }
        
    def training_step(self, batch):
        self.train()
        t = torch.randint(
            0, self.diff_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t)
        #########################################################
        # logging training metrics
        #########################################################
        self.log_dict(
            {"train/loss": loss},
            add_dataloader_idx=False,
            prog_bar=True,
        )

        # determine if additional logging should be done
        do_additional_logging = (
            self.global_step % self.additional_train_logging_period == 0
        )

        # additional logging
        if do_additional_logging:
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            pred = pred_wta_dict["pred"]
            pred_wta = pred_wta_dict["pred_wta"]
            cos_sim = pred_wta_dict["cos_sim"]
            cos_sim_wta = pred_wta_dict["cos_sim_wta"]
            rmse = pred_wta_dict["rmse"]
            rmse_wta = pred_wta_dict["rmse_wta"]

            ####################################################
            # logging training wta metrics
            ####################################################
            self.log_dict(
                {
                    "train/cos_sim": cos_sim.mean(),
                    "train/cos_sim_wta": cos_sim_wta.mean(),
                    "train/rmse": rmse.mean(),
                    "train/rmse_wta": rmse_wta.mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )


            ####################################################
            # logging visualizations
            ####################################################
            viz_idx = np.random.randint(0, batch["pc"].shape[0])
            pc_pos_viz = batch["pc"][viz_idx, :, :3]
            pc_action_viz = batch["pc_action"][viz_idx, :, :3]
            pred_viz = pred[viz_idx, 0, :, :3]
            pred_wta_viz = pred_wta[viz_idx, :, :3]
            viz_args = {
                "pc_pos_viz": pc_pos_viz,
                "pc_action_viz": pc_action_viz,
            }

            if self.prediction_type == "flow":
                pred_action_viz = pc_action_viz + pred_viz
                pred_action_wta_viz = pc_action_viz + pred_wta_viz
            elif self.prediction_type == "point":
                pred_action_viz = pred_viz
                pred_action_wta_viz = pred_wta_viz

            # logging predicted vs ground truth point cloud
            viz_args["pred_action_viz"] = pred_action_viz
            predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
            wandb.log({"train/predicted_vs_gt": predicted_vs_gt})

            # logging predicted vs ground truth point cloud (wta)
            viz_args["pred_action_viz"] = pred_action_wta_viz
            predicted_vs_gt_wta = viz_predicted_vs_gt(**viz_args)
            wandb.log({"train/predicted_vs_gt_wta": predicted_vs_gt_wta})
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            # winner-take-all predictions
            pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
            pred = pred_wta_dict["pred"]
            pred_wta = pred_wta_dict["pred_wta"]
            cos_sim = pred_wta_dict["cos_sim"]
            cos_sim_wta = pred_wta_dict["cos_sim_wta"]
            rmse = pred_wta_dict["rmse"]
            rmse_wta = pred_wta_dict["rmse_wta"]

        ####################################################
        # logging validation wta metrics
        ####################################################
        self.log_dict(
            {
                f"val_cos_sim_{dataloader_idx}": cos_sim.mean(),
                f"val_wta_cos_sim_{dataloader_idx}": cos_sim_wta.mean(),
                f"val_rmse_{dataloader_idx}": rmse.mean(),
                f"val_wta_rmse_{dataloader_idx}": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        ####################################################
        # logging visualizations
        ####################################################
        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pred_viz = pred[viz_idx, 0, :, :3]
        pred_wta_viz = pred_wta[viz_idx, :, :3]
        viz_args = {
            "pc_pos_viz": pc_pos_viz,
            "pc_action_viz": pc_action_viz,
        }

        if self.prediction_type == "flow":
            pred_action_viz = pc_action_viz + pred_viz
            pred_action_wta_viz = pc_action_viz + pred_wta_viz
        elif self.prediction_type == "point":
            pred_action_viz = pred_viz
            pred_action_wta_viz = pred_wta_viz

        # logging predicted vs ground truth point cloud
        viz_args["pred_action_viz"] = pred_action_viz
        predicted_vs_gt = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"val/predicted_vs_gt_{dataloader_idx}": predicted_vs_gt})

        # logging predicted vs ground truth point cloud (wta)
        viz_args["pred_action_viz"] = pred_action_wta_viz
        predicted_vs_gt_wta = viz_predicted_vs_gt(**viz_args)
        wandb.log({f"val_wta/predicted_vs_gt_{dataloader_idx}": predicted_vs_gt_wta})

    def predict_step(self, batch):
        # winner-take-all predictions
        pred_wta_dict = self.predict_wta(batch, self.num_wta_trials)
        return {
            "cos_sim": pred_wta_dict["cos_sim"],
            "cos_sim_wta": pred_wta_dict["cos_sim_wta"],
            "rmse": pred_wta_dict["rmse"],
            "rmse_wta": pred_wta_dict["rmse_wta"],
        }

class Tax3dTrainingModule(L.LightningModule):
    pass