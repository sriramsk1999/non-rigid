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
from non_rigid.models.dit.models import LinearRegressionModel
from non_rigid.utils.logging_utils import viz_predicted_vs_gt
from non_rigid.utils.pointcloud_utils import expand_pcd

def LinearRegression_xS(**kwargs):
    return LinearRegressionModel(depth=5, hidden_size=132, num_heads=4, **kwargs)


models = {
    "linear_regression_xS": LinearRegression_xS,
}

def get_model(model_cfg):
    model_name = f"{model_cfg.name}_{model_cfg.size}"
    return models[model_name]

class LinearRegression(nn.Module):
    def __init__(
            self,
            model_cfg=None,
    ):
        super().__init__()
        self.model = get_model(model_cfg)(
            in_channels=model_cfg.in_channels,
            model_cfg=model_cfg,
        )

    def forward(self, x, y):
        return self.model(x=x, y=y)


class LinearRegressionTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg):
        super().__init__()
        self.network = network
        self.training_cfg = training_cfg
        self.model_cfg = model_cfg

        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay

        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = training_cfg.sample_size
        self.sample_size_anchor = training_cfg.sample_size_anchor

    def forward(self, batch, mode="train"):
        # Extract point clouds from batch
        pos = batch["pc"].permute(0, 2, 1) # (B, C, N)
        pc_action = batch["pc_action"].permute(0, 2, 1) # (B, C, N)
        pc_anchor = batch["pc_anchor"].permute(0, 2, 1) # (B, C, N)

        # Setup data required by the model
        model_kwargs = dict(
            x=pc_action, y=pc_anchor
        )
        pred_pos = self.network(**model_kwargs)

        loss = F.mse_loss(pred_pos, pos)
        return None, loss
    
    @torch.no_grad()
    def predict(
        self,
        batch,
    ):
        pos = batch["pc"].to(self.device)
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        # reshaping
        pc_action = pc_action.permute(0, 2, 1)
        pc_anchor = pc_anchor.permute(0, 2, 1)

        pred_pos = self.network(x=pc_action, y=pc_anchor)
        pred_pos = pred_pos.permute(0, 2, 1)

        cos_sim = flow_cos_sim(pred_pos, pos, mask=False, seg=None)
        rmse = flow_rmse(pred_pos, pos, mask=False, seg=None)

        return {
            "pred_action": pred_pos,
            "cos_sim": cos_sim,
            "rmse": rmse,
        }

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.lr_warmup_steps,
        )
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        self.train()
        _, loss = self(batch, "train")
        ##################################
        # Logging
        ##################################
        self.log_dict(
            {
                f"train/loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        do_additional_logging = (
            self.global_step % self.training_cfg.additional_train_logging_period == 0
        )

        # Additional logging
        if do_additional_logging:
            pred_dict = self.predict(batch)
            pred_action = pred_dict["pred_action"]
            cos_sim = pred_dict["cos_sim"]
            rmse = pred_dict["rmse"]

            self.log_dict(
                {
                    "train/cos_sim": cos_sim.mean(),
                    "train/rmse": rmse.mean(),
                },
                add_dataloader_idx=False,
                prog_bar=True,
            )

            viz_idx = np.random.randint(0, batch["pc"].shape[0])
            pc_pos_viz = batch["pc"][viz_idx, :, :3]
            pc_action_viz = batch["pc_action"][viz_idx, :, :3]
            pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
            pred_action_viz = pred_action[viz_idx, :, :3]

            # Get predicted vs ground truth visualization
            predicted_vs_gt_viz = viz_predicted_vs_gt(
                pc_pos_viz=pc_pos_viz,
                pc_action_viz=pc_action_viz,
                pc_anchor_viz=pc_anchor_viz,
                pred_action_viz=pred_action_viz,
            )
            wandb.log({"train/predicted_vs_gt": predicted_vs_gt_viz})

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            pred_dict = self.predict(batch)
            pred_action = pred_dict["pred_action"]
            cos_sim = pred_dict["cos_sim"]
            rmse = pred_dict["rmse"]

        # duplicatin wta metrics to prevent errors
        self.log_dict(
            {
                f"val_cos_sim_{dataloader_idx}": cos_sim.mean(),
                f"val_rmse_{dataloader_idx}": rmse.mean(),
                f"val_wta_cos_sim_{dataloader_idx}": cos_sim.mean(),
                f"val_wta_rmse_{dataloader_idx}": rmse.mean(),
            },
            add_dataloader_idx=False,
            prog_bar=True,
        )

        viz_idx = np.random.randint(0, batch["pc"].shape[0])
        pc_pos_viz = batch["pc"][viz_idx, :, :3]
        pc_action_viz = batch["pc_action"][viz_idx, :, :3]
        pc_anchor_viz = batch["pc_anchor"][viz_idx, :, :3]
        pred_action_viz = pred_action[viz_idx, :, :3]

        # Get predicted vs ground truth visualization
        predicted_vs_gt_viz = viz_predicted_vs_gt(
            pc_pos_viz=pc_pos_viz,
            pc_action_viz=pc_action_viz,
            pc_anchor_viz=pc_anchor_viz,
            pred_action_viz=pred_action_viz,
        )
        wandb.log({f"val_{dataloader_idx}/predicted_vs_gt": predicted_vs_gt_viz})

        return {
            "cos_sim": cos_sim,
            "loss": rmse,
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("Predict step not implemented for LinearRegressionTrainingModule")
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("Test step not implemented for LinearRegressionTrainingModule")
    

class LinearRegressionInferenceModule(L.LightningModule):
    def __init__(
        self, 
        network, 
        inference_cfg,
        model_cfg,
    ):
        super().__init__()
        self.network = network
        self.inference_cfg = inference_cfg
        self.model_cfg = model_cfg

        self.batch_size = inference_cfg.batch_size
        self.val_batch_size = inference_cfg.val_batch_size
        self.sample_size = inference_cfg.sample_size
        self.sample_size_anchor = inference_cfg.sample_size_anchor

    def forward(self, batch):
        raise NotImplementedError(
            "Inference module should not use forward method - use 'predict' instead."
        )

    @torch.no_grad()
    def predict(
        self, 
        batch,
        num_samples,
        progress=False,
    ):
        # TODO: add in num_samples as a dummy parameter
        pc_action = batch["pc_action"].to(self.device)
        pc_anchor = batch["pc_anchor"].to(self.device)
        # reshaping
        pc_action = pc_action.permute(0, 2, 1)
        pc_anchor = pc_anchor.permute(0, 2, 1)

        pred_action = self.network(x=pc_action, y=pc_anchor)
        pred_action = pred_action.permute(0, 2, 1)

        # expanding point clouds
        pc_action = expand_pcd(pc_action, num_samples)
        pred_action = expand_pcd(pred_action, num_samples)

        T_action2world = Transform3d(
            matrix=expand_pcd(batch["T_action2world"].to(self.device), num_samples),
        )
        T_goal2world = Transform3d(
            matrix=expand_pcd(batch["T_goal2world"].to(self.device), num_samples),
        )
        # computing pred flow in world frame
        pc_action = pc_action.permute(0, 2, 1)
        pred_flow = T_goal2world.transform_points(pred_action) - T_action2world.transform_points(pc_action)
        return {
            "pred_action": pred_action,
            "pred_world_flow": pred_flow,
            "pred_world_action": T_goal2world.transform_points(pred_action),
        }

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pos = batch["pc"].to(self.device)
        pred_dict = self.predict(batch, 1)
        pred_action = pred_dict["pred_action"]

        # computing errors
        cos_sim = flow_cos_sim(pred_action, pos, mask=False, seg=None)
        rmse = flow_rmse(pred_action, pos, mask=False, seg=None)
        # duplicating WTA metrics to prevent errors
        return {
            "cos_sim": cos_sim,
            "rmse": rmse,
            "cos_sim_wta": cos_sim,
            "rmse_wta": rmse,
        }