from typing import Any, Dict

import numpy as np
import omegaconf
import plotly.express as px

import lightning as L
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision as tv

import rpad.pyg.nets.dgcnn as dgcnn
import torch_geometric.data as tgd
import torch_geometric.transforms as tgt
from torch_geometric.nn import fps
import wandb

from non_rigid.models.dit.models import (
    DiT_PointCloud_Unc as DiT_pcu,
    DiT_PointCloud_Unc_Cross,
)
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn
from non_rigid.utils.vis_utils import get_color

from diffusers import get_cosine_schedule_with_warmup


def DiT_pcu_S(**kwargs):
    return DiT_pcu(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_pcu_xS(**kwargs):
    return DiT_pcu(depth=5, hidden_size=128, num_heads=4, **kwargs)


def DiT_pcu_cross_xS(**kwargs):
    return DiT_PointCloud_Unc_Cross(depth=5, hidden_size=128, num_heads=4, **kwargs)


DiT_models = {
    "DiT_pcu_S": DiT_pcu_S,
    "DiT_pcu_xS": DiT_pcu_xS,
    "DiT_pcu_cross_xS": DiT_pcu_cross_xS,
}


class DiffusionFlowBase(nn.Module):
    # literally just unconditional DiT adapted for PC
    def __init__(self, in_channels=6, learn_sigma=False, model="DiT_pcu_S", model_cfg=None):
        super().__init__()
        # TODO: get in channels from params, and pass as kwargs
        # TODO: input needs to already be hidden size dim
        self.dit = DiT_models[model](in_channels=in_channels, learn_sigma=learn_sigma, model_cfg=model_cfg)

    def forward(self, x, t, **kwargs):
        # extract
        return self.dit(x, t, **kwargs)


class FlowPredictionTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg) -> None:
        super().__init__()
        self.network = network
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg
        
        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay  # 1e-5
        # self.mode, self.traj_len so far not needed
        # self.epochs = training_cfg.epochs
        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # TODO: organize these params by description
        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = training_cfg.training_sample_size
        self.diff_train_steps = model_cfg.diff_train_steps
        self.num_wta_trials = training_cfg.num_wta_trials
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_train_steps,
        )

    def forward(self, batch, t, mode="train"):
        if self.model_cfg.type == "flow":
            # get flow and pos
            pos = batch["pc_init"].permute(0, 2, 1)  # channel first
            flow = batch["flow"].permute(0, 2, 1)  # channel first

            # Setup additional data required by the model
            model_kwargs = dict(pos=pos)

        # If we are doing cross attention, we need to pass in additional data
        elif self.model_cfg.type == "flow_cross":
            pos = batch["pc_init"].permute(0, 2, 1)  # channel first
            pc_anchor = batch["pc_anchor"].permute(0, 2, 1)  # channel first
            flow = batch["flow"].permute(0, 2, 1)  # channel first
            
            model_kwargs = dict(
                y=pc_anchor,  # Pass original anchor point cloud
                x0=pos,  # Pass starting action point cloud
            )
        
        # run diffusion
        loss_dict = self.diffusion.training_losses(self.network, flow, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        self.log_dict(
            {
                f"{mode} loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar=mode == "train",
        )
        return None, loss

    # 'predict' should only be called for inference/evaluation
    @torch.no_grad()
    def predict(self, bs, model_kwargs, num_samples, unflatten=False):
        """
        unflatten: if True, unflatten all outputs to shape (batch_size, num_samples, ...); otherwise, return
            with shape (batch_size * num_samples, ...)
        """
        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, self.sample_size, device=self.device)
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )
        pred_flow = pred_flow.permute(0, 2, 1)
        if unflatten:
            pred_flow = pred_flow.reshape(bs, num_samples, self.sample_size, -1)
        
        for key in model_kwargs:
            if key in ["pos", "y", "x0"]:
                model_kwargs[key] = model_kwargs[key].permute(0, 2, 1)
                if unflatten:
                    model_kwargs[key] = model_kwargs[key].reshape(bs, num_samples, self.sample_size, -1)
        return model_kwargs, pred_flow, results

    def predict_wta(self, batch, mode):
        if self.model_cfg.type == "flow":
            pos = batch["pc_init"].to(self.device)
            gt_flow = batch["flow"].to(self.device)
            seg = batch["seg"].to(self.device)
            mask = True
            
            # reshaping and expanding for winner-take-all
            bs = pos.shape[0]
            gt_flow = (
                gt_flow.unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1, -1)
                .reshape(bs * self.num_wta_trials, self.sample_size, -1)
            )
            seg = (
                seg.unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1)
                .reshape(bs * self.num_wta_trials, -1)
            )
            pos = (
                pos.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1, -1)
                .reshape(bs * self.num_wta_trials, -1, self.sample_size)
            )
            
            model_kwargs = dict(pos=pos)
        elif self.model_cfg.type == "flow_cross":
            pos = batch["pc_init"].to(self.device)
            pc_anchor = batch["pc_anchor"].to(self.device)
            gt_flow = batch["flow"].to(self.device)
            seg = None # TODO Rigid dataset doesnt have this
            mask = False
            
            # reshaping and expanding for winner-take-all
            bs = pos.shape[0]
            gt_flow = (
                gt_flow.unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1, -1)
                .reshape(bs * self.num_wta_trials, self.sample_size, -1)
            )
            pos = (
                pos.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1, -1)
                .reshape(bs * self.num_wta_trials, -1, self.sample_size)
            )
            pc_anchor = (
                pc_anchor.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, self.num_wta_trials, -1, -1)
                .reshape(bs * self.num_wta_trials, -1, self.sample_size)
            )
            
            model_kwargs = dict(
                y=pc_anchor,  # Pass original anchor point cloud
                x0=pos,  # Pass starting action point cloud
            )
        
        # generating diffusion predictions
        model_kwargs, pred_flow, results = self.predict(bs, model_kwargs, self.num_wta_trials, unflatten=False)
        # computing wta errors
        cos_sim = flow_cos_sim(pred_flow, gt_flow, mask=mask, seg=seg).reshape(
            bs, self.num_wta_trials
        )
        rmse = flow_rmse(pred_flow, gt_flow, mask=False, seg=None).reshape(
            bs, self.num_wta_trials
        )
        pred_flow = pred_flow.reshape(bs, self.num_wta_trials, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        # logging
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_flows_wta = pred_flow[torch.arange(bs), winner]
        return pred_flows_wta, cos_sim_wta, rmse_wta

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.train()  # what is this line doing?
        t = torch.randint(
            0, self.diff_train_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            pred_flows_wta, cos_sim_wta, rmse_wta = self.predict_wta(
                batch, mode="val"
            )
            
        self.log_dict(
            {
                f"val_wta/cos_sim": cos_sim_wta.mean(),
                f"val_wta/rmse": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )
            
        if self.model_cfg.type == "flow_cross":
            # Visualize predicted vs ground truth
            viz_idx = np.random.randint(0, batch["pc_init"].shape[0])
            
            pc_pos_viz = batch["pc_init"][viz_idx, :, :3]
            pc_action_viz = batch["pc_action"][viz_idx, :, :3]
            pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
            
            pred_flows_viz = pred_flows_wta[viz_idx, :, :3]
            pred_action_wta_viz = pc_pos_viz + pred_flows_viz
            
            pc_comb_viz = torch.cat([pc_action_viz, pc_anchor_viz], dim=0)
            pc_comb_viz_min = pc_comb_viz.min(dim=0).values
            pc_comb_viz_max = pc_comb_viz.max(dim=0).values
            pc_comb_viz_extent = pc_comb_viz_max - pc_comb_viz_min
            pred_action_wta_viz = pred_action_wta_viz[
                (
                    pred_action_wta_viz[:, 0]
                    > pc_comb_viz_min[0] - 0.5 * pc_comb_viz_extent[0]
                )
                & (
                    pred_action_wta_viz[:, 0]
                    < pc_comb_viz_max[0] + 0.5 * pc_comb_viz_extent[0]
                )
                & (
                    pred_action_wta_viz[:, 1]
                    > pc_comb_viz_min[1] - 0.5 * pc_comb_viz_extent[1]
                )
                & (
                    pred_action_wta_viz[:, 1]
                    < pc_comb_viz_max[1] + 0.5 * pc_comb_viz_extent[1]
                )
                & (
                    pred_action_wta_viz[:, 2]
                    > pc_comb_viz_min[2] - 0.5 * pc_comb_viz_extent[2]
                )
                & (
                    pred_action_wta_viz[:, 2]
                    < pc_comb_viz_max[2] + 0.5 * pc_comb_viz_extent[2]
                )
            ]
            predicted_vs_gt_wta_tensors = [
                pc_action_viz,
                pc_anchor_viz,
                pred_action_wta_viz,
            ]
            predicted_vs_gt_wta_colors = ["green", "red", "blue"]
            predicted_vs_gt_wta = get_color(
                tensor_list=predicted_vs_gt_wta_tensors,
                color_list=predicted_vs_gt_wta_colors,
            )
            wandb.log({"val_wta/predicted_vs_gt": wandb.Object3D(predicted_vs_gt_wta)})
            
        return {
            "loss": rmse_wta,
            "cos_sim": cos_sim_wta,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # no test for now
        pass
    
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pred_actions_wta, cos_sim_wta, rmse_wta = self.predict_wta(
            batch, mode="val"
        )
        return {
            "cos_sim": cos_sim_wta,
            "rmse": rmse_wta,
        }


# TODO: inference module
class FlowPredictionInferenceModule(L.LightningModule):
    def __init__(self, network, inference_cfg, model_cfg) -> None:
        super().__init__()
        self.network = network
        self.batch_size = inference_cfg.batch_size
        self.val_batch_size = inference_cfg.val_batch_size
        self.num_wta_trials = inference_cfg.num_wta_trials
        self.sample_size = inference_cfg.sample_size

        self.diff_steps = model_cfg.diff_train_steps
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_steps,
        )

    def forward(self, data):
        raise NotImplementedError(
            "Inference module should not use forward method - use 'predict' instead."
        )

    @torch.no_grad()
    def predict(self, pos, num_samples, unflatten=False):
        """
        unflatten: if True, unflatten all outputs to shape (batch_size, num_samples, ...); otherwise, return
            with shape (batch_size * num_samples, ...)
        """
        bs = pos.shape[0]
        # reshaping and expanding
        pos = (
            pos.transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, num_samples, -1, -1)
            .reshape(bs * num_samples, -1, self.sample_size)
        )
        model_kwargs = dict(pos=pos)
        # generating latents and running diffusion
        z = torch.randn(bs * num_samples, 3, self.sample_size, device=self.device)
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )
        pred_flow = pred_flow.permute(0, 2, 1)
        pos = pos.permute(0, 2, 1)
        if unflatten:
            pos = pos.reshape(bs, num_samples, self.sample_size, -1)
            pred_flow = pred_flow.reshape(bs, num_samples, self.sample_size, -1)
        return pos, pred_flow

    def predict_wta(self, batch, mode):
        pos = batch["pc_init"]
        gt_flow = batch["flow"]
        seg = batch["seg"]
        # reshaping and expanding for winner-take-all
        bs = pos.shape[0]
        gt_flow = (
            gt_flow.unsqueeze(1)
            .expand(-1, self.num_wta_trials, -1, -1)
            .reshape(bs * self.num_wta_trials, self.sample_size, -1)
        )
        seg = (
            seg.unsqueeze(1)
            .expand(-1, self.num_wta_trials, -1)
            .reshape(bs * self.num_wta_trials, -1)
        )
        # generating diffusion predictions
        pos, pred_flow = self.predict(pos, self.num_wta_trials, unflatten=False)
        # computing wta errors
        cos_sim = flow_cos_sim(pred_flow, gt_flow, mask=True, seg=seg).reshape(
            bs, self.num_wta_trials
        )
        rmse = flow_rmse(pred_flow, gt_flow, mask=False, seg=None).reshape(
            bs, self.num_wta_trials
        )
        pred_flow = pred_flow.reshape(bs, self.num_wta_trials, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        # logging
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_flows_wta = pred_flow[torch.arange(bs), winner]
        # self.log_dict(
        #     {
        #         f"{mode}_wta/cos_sim": cos_sim_wta.mean(),
        #         f"{mode}_wta/rmse": rmse_wta.mean(),
        #     },
        #     add_dataloader_idx=False,
        #     prog_bar = True,
        # )
        return pred_flows_wta, cos_sim_wta, rmse_wta

    # don't need to use this yet
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pred_flows_wtas, cos_sim_wtas, rmse_wtas = self.predict_wta(
            batch, mode="predict"
        )
        return {
            "cos_sim": cos_sim_wtas,
            "rmse": rmse_wtas,
        }


class PointPredictionTrainingModule(L.LightningModule):
    def __init__(
        self,
        network: DiffusionFlowBase,
        training_cfg: omegaconf.DictConfig,
        model_cfg: omegaconf.DictConfig,
    ) -> None:
        super().__init__()
        self.network = network
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg
        
        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay  # 1e-5
        # self.mode, self.traj_len so far not needed
        # self.epochs = training_cfg.epochs
        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # TODO: organize these params by description
        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = training_cfg.training_sample_size
        self.diff_train_steps = model_cfg.diff_train_steps
        self.num_wta_trials = training_cfg.num_wta_trials

        self.noise_schedule = model_cfg.diff_noise_schedule
        self.noise_scale = model_cfg.diff_noise_scale

        self.diffusion = create_diffusion(
            noise_schedule=self.noise_schedule,
            timestep_respacing=None,
            diffusion_steps=self.diff_train_steps,
        )

    def forward(self, batch, t, mode="train"):
        # Extract point clouds from batch
        
        pos = batch["pc_init"].permute(0, 2, 1)  # B, C, N
        pc_action = batch["pc_action"].permute(0, 2, 1)  # B, C, N
        pc_anchor = batch["pc_anchor"].permute(0, 2, 1)  # B, C, N

        # Setup additional data required by the model
        model_kwargs = dict(
            y=pc_anchor,  # Pass original anchor point cloud
            x0=pc_action
        )

        # Run diffusion
        noise = torch.randn_like(pos) * self.noise_scale
        loss_dict = self.diffusion.training_losses(
            self.network, pos, t, model_kwargs, noise
        )
        loss = loss_dict["loss"].mean()

        # Logging
        self.log_dict(
            {
                f"{mode} loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar=mode == "train",
        )
        return None, loss

    @torch.no_grad()
    def predict(
        self,
        bs: int,
        model_kwargs: Dict[str, torch.Tensor],
        num_samples: int,
        unflatten: bool = False,
    ):
        """
        unflatten: if True, unflatten all outputs to shape (batch_size, num_samples, ...); otherwise, return
            with shape (batch_size * num_samples, ...)
        """
        # generating latents and running diffusion
        z = (
            torch.randn(bs * num_samples, 3, self.sample_size, device=self.device)
            * self.noise_scale
        )
        pred_action, results = self.diffusion.p_sample_loop(
            self.network,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=self.device,
        )
        pred_action = pred_action.permute(0, 2, 1)
        if unflatten:
            pred_action = pred_action.reshape(bs, num_samples, self.sample_size, -1)
            
        for key in model_kwargs:
            if key in ["x0", "y"]:
                model_kwargs[key] = model_kwargs[key].permute(0, 2, 1)
                if unflatten:
                    model_kwargs[key] = model_kwargs[key].reshape(bs, num_samples, self.sample_size, -1)

        return model_kwargs, pred_action, results

    def predict_wta(self, batch: Dict[str, torch.Tensor], mode: str):
        pos = batch["pc_init"].to(self.device)
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)

        # reshaping and expanding for winner-take-all
        bs = pc_action.shape[0]
        gt_action = (
            pos.unsqueeze(1)
            .expand(-1, self.num_wta_trials, -1, -1)
            .reshape(bs * self.num_wta_trials, self.sample_size, -1)
        )
        pc_action = (
            pc_action.transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, self.num_wta_trials, -1, -1)
            .reshape(bs * self.num_wta_trials, -1, self.sample_size)
        )
        pc_anchor = (
            pc_anchor.transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, self.num_wta_trials, -1, -1)
            .reshape(bs * self.num_wta_trials, -1, self.sample_size)
        )
        
        model_kwargs = dict(
            y=pc_anchor, 
            x0=pc_action,
        )
        
        # generating diffusion predictions
        model_kwargs, pred_action, results = self.predict(
            bs, model_kwargs, self.num_wta_trials, unflatten=False
        )

        # computing wta errors
        cos_sim = flow_cos_sim(pred_action, gt_action, mask=False, seg=None).reshape(
            bs, self.num_wta_trials
        )
        rmse = flow_rmse(pred_action, gt_action, mask=False, seg=None).reshape(
            bs, self.num_wta_trials
        )
        pred_action = pred_action.reshape(bs, self.num_wta_trials, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        # logging
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_actions_wta = pred_action[torch.arange(bs), winner]
        return pred_actions_wta, cos_sim_wta, rmse_wta

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.train()  # what is this line doing?
        t = torch.randint(
            0, self.diff_train_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            pred_actions_wta, cos_sim_wta, rmse_wta = self.predict_wta(
                batch, mode="val"
            )

        # Logging
        self.log_dict(
            {
                f"val_wta/cos_sim": cos_sim_wta.mean(),
                f"val_wta/rmse": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        # Visualize predicted vs ground truth
        viz_idx = np.random.randint(0, batch["pc_init"].shape[0])

        pc_pos_viz = batch["pc_init"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]

        pc_comb_viz = torch.cat([pc_action_viz, pc_anchor_viz], dim=0)
        pc_comb_viz_min = pc_comb_viz.min(dim=0).values
        pc_comb_viz_max = pc_comb_viz.max(dim=0).values
        pc_comb_viz_extent = pc_comb_viz_max - pc_comb_viz_min
        pred_action_wta_viz = pred_actions_wta[viz_idx, :, :3]
        pred_action_wta_viz = pred_action_wta_viz[
            (
                pred_action_wta_viz[:, 0]
                > pc_comb_viz_min[0] - 0.5 * pc_comb_viz_extent[0]
            )
            & (
                pred_action_wta_viz[:, 0]
                < pc_comb_viz_max[0] + 0.5 * pc_comb_viz_extent[0]
            )
            & (
                pred_action_wta_viz[:, 1]
                > pc_comb_viz_min[1] - 0.5 * pc_comb_viz_extent[1]
            )
            & (
                pred_action_wta_viz[:, 1]
                < pc_comb_viz_max[1] + 0.5 * pc_comb_viz_extent[1]
            )
            & (
                pred_action_wta_viz[:, 2]
                > pc_comb_viz_min[2] - 0.5 * pc_comb_viz_extent[2]
            )
            & (
                pred_action_wta_viz[:, 2]
                < pc_comb_viz_max[2] + 0.5 * pc_comb_viz_extent[2]
            )
        ]
        predicted_vs_gt_wta_tensors = [
            pc_pos_viz,
            pc_anchor_viz,
            pred_action_wta_viz,
        ]
        predicted_vs_gt_wta_colors = ["green", "red", "blue"]
        predicted_vs_gt_wta = get_color(
            tensor_list=predicted_vs_gt_wta_tensors,
            color_list=predicted_vs_gt_wta_colors,
        )
        wandb.log({"val_wta/predicted_vs_gt": wandb.Object3D(predicted_vs_gt_wta)})

        return {
            "loss": rmse_wta,
            "cos_sim": cos_sim_wta,
        }
        
    @torch.no_grad()
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        pred_actions_wta, cos_sim_wta, rmse_wta = self.predict_wta(
            batch, mode="val"
        )
        return {
            "cos_sim": cos_sim_wta,
            "rmse": rmse_wta,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # no test for now
        pass


if __name__ == "__main__":
    # test
    model = DiffusionFlowBase()
    print(model)

    input = torch.randn(2, 1867, 6)
    ts = torch.tensor([2, 3])
    output = model(input, ts)
    print("Output: ", output.shape)
