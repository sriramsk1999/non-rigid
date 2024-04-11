from typing import Any

import numpy as np
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

from non_rigid.models.dit.models import DiT_PointCloud_Unc as DiT_pcu
from non_rigid.models.dit.diffusion import create_diffusion
from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse, pc_nn

from diffusers import get_cosine_schedule_with_warmup

def DiT_pcu_S(**kwargs):
    return DiT_pcu(depth=12, hidden_size=384, num_heads=6, **kwargs)

def DiT_pcu_xS(**kwargs):
    return DiT_pcu(depth=5, hidden_size=128, num_heads=4, **kwargs)

DiT_models = {'DiT_pcu_S': DiT_pcu_S, 'DiT_pcu_xS': DiT_pcu_xS}

class DiffusionFlowBase(nn.Module):
    # literally just unconditional DiT adapted for PC
    def __init__(self, in_channels = 6, learn_sigma=False, model='DiT_pcu_S'):
        super().__init__()
        # TODO: get in channels from params, and pass as kwargs
        # TODO: input needs to already be hidden size dim
        self.dit = DiT_models[model](in_channels=in_channels, learn_sigma=learn_sigma)

    def forward(self, x, t, **kwargs):
        # extract
        return self.dit(x, t, **kwargs)
    

class ArticulatedFlowTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg) -> None:
        super().__init__()
        self.network = network
        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay # 1e-5
        # self.mode, self.traj_len so far not needed
        # self.epochs = training_cfg.epochs
        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # TODO: organize these params by description
        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = model_cfg.sample_size # TODO: should this come from model cfg?
        self.diff_train_steps = model_cfg.diff_train_steps
        self.num_wta_trials = training_cfg.num_wta_trials
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_train_steps,
        )
        # self.mode_distribution_cache = {"x": None, "y": None}

    def forward(self, x, t, mode="train"):
        # get flow and pos
        # pos, flow, _, _, _, *misc  = x
        pos = x['pc_init']
        flow = x['flow']

        # channel first
        pos = torch.transpose(pos, -1, -2)
        flow = torch.transpose(flow, -1, -2)
        model_kwargs = dict(pos=pos)
        # run diffusion
        loss_dict = self.diffusion.training_losses(
            self.network, flow, t, model_kwargs
        )
        loss = loss_dict["loss"].mean()
        self.log_dict(
            {
                f"{mode} loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar = mode == "train",
        )
        return None, loss
    
    # TODO:this is only for inference
    @torch.no_grad()
    def predict(self, batch):
        pass

    def predict_wta(self, batch, mode):
        # pos, gt_flow, seg, _, goal, *misc = batch
        pos = batch['pc_init']
        gt_flow = batch['flow']
        seg = batch['seg']
        goal = batch['goal']
        
        # reshaping and expanding for wta
        bs = pos.shape[0]
        pos = pos.transpose(-1, -2).unsqueeze(1).expand(-1, self.num_wta_trials, -1, -1).reshape(bs * self.num_wta_trials, -1, self.sample_size)
        gt_flow = gt_flow.unsqueeze(1).expand(-1, self.num_wta_trials, -1, -1).reshape(bs * self.num_wta_trials, self.sample_size, -1)
        seg = seg.unsqueeze(1).expand(-1, self.num_wta_trials, -1).reshape(bs * self.num_wta_trials, -1)
        # generating latents and running diffusion
        model_kwargs = dict(pos=pos)
        z = torch.randn(
            bs * self.num_wta_trials, 3, self.sample_size, device=self.device
        )
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=self.device
        )
        pred_flow = pred_flow.permute(0, 2, 1)
        # computing wta errors
        cos_sim = flow_cos_sim(pred_flow, gt_flow, mask=True, seg=seg).reshape(bs, self.num_wta_trials)
        rmse = flow_rmse(pred_flow, gt_flow, mask=False, seg=None).reshape(bs, self.num_wta_trials)
        pred_flow = pred_flow.reshape(bs, self.num_wta_trials, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        # logging
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_flows_wta = pred_flow[torch.arange(bs), winner]
        self.log_dict(
            {
                f"{mode}_wta/cos_sim": cos_sim_wta.mean(),
                f"{mode}_wta/rmse": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar = True,
        )
        return pred_flows_wta, cos_sim_wta, rmse_wta
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        self.train() # what is this line doing?
        t = torch.randint(
            0, self.diff_train_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        # if new epoch, clear cache
        # if batch_idx == 0:
        #     self.mode_distribution_cache = {"x": None, "y": None}
        with torch.no_grad():
            pred_flows_wtas, cos_sim_wtas, rmse_wtas = self.predict_wta(batch, mode="val")
        return {
            "loss": rmse_wtas,
            "cos_sim": cos_sim_wtas,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # no test for now
        pass



class ClothFlowTrainingModule(L.LightningModule):
    def __init__(self, network, training_cfg, model_cfg) -> None:
        super().__init__()
        self.network = network
        self.lr = training_cfg.lr
        self.weight_decay = training_cfg.weight_decay # 1e-5
        # self.mode, self.traj_len so far not needed
        # self.epochs = training_cfg.epochs
        self.num_training_steps = training_cfg.num_training_steps
        self.lr_warmup_steps = training_cfg.lr_warmup_steps

        # TODO: organize these params by description
        self.batch_size = training_cfg.batch_size
        self.val_batch_size = training_cfg.val_batch_size
        self.sample_size = model_cfg.sample_size # TODO: should this come from model cfg?
        self.diff_train_steps = model_cfg.diff_train_steps
        self.num_wta_trials = training_cfg.num_wta_trials
        self.diffusion = create_diffusion(
            timestep_respacing=None,
            diffusion_steps=self.diff_train_steps,
        )

    def forward(self, x, t, mode="train"):
        # get flow and pos
        pos, flow, _ = x

        # channel first
        pos = torch.transpose(pos, -1, -2)
        flow = torch.transpose(flow, -1, -2)
        model_kwargs = dict(pos=pos)
        # run diffusion
        loss_dict = self.diffusion.training_losses(
            self.network, flow, t, model_kwargs
        )
        loss = loss_dict["loss"].mean()
        self.log_dict(
            {
                f"{mode} loss": loss,
            },
            add_dataloader_idx=False,
            prog_bar = mode == "train",
        )
        return None, loss
    
    # 'predict' should only be called for inference/evaluation
    @torch.no_grad()
    def predict(self, pos, num_samples, unflatten=False):
        """
        unflatten: if True, unflatten all outputs to shape (batch_size, num_samples, ...); otherwise, return 
            with shape (batch_size * num_samples, ...)
        """
        bs = pos.shape[0]
        # reshaping and expanding
        pos = pos.transpose(-1, -2).unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(bs * num_samples, -1, self.sample_size)
        model_kwargs = dict(pos=pos)
        # generating latents and running diffusion
        z = torch.randn(
            bs * num_samples, 3, self.sample_size, device=self.device
        )
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=self.device
        )
        pred_flow = pred_flow.permute(0, 2, 1)
        pos = pos.permute(0, 2, 1)
        if unflatten:
            pos = pos.reshape(bs, num_samples, self.sample_size, -1)
            pred_flow = pred_flow.reshape(bs, num_samples, self.sample_size, -1)
        return pos, pred_flow

    def predict_wta(self, batch, mode):
        pos, gt_flow, seg = batch
        # reshaping and expanding for winner-take-all
        bs = pos.shape[0]
        gt_flow = gt_flow.unsqueeze(1).expand(-1, self.num_wta_trials, -1, -1).reshape(bs * self.num_wta_trials, self.sample_size, -1)
        seg = seg.unsqueeze(1).expand(-1, self.num_wta_trials, -1).reshape(bs * self.num_wta_trials, -1)
        # generating diffusion predictions
        pos, pred_flow = self.predict(pos, self.num_wta_trials, unflatten=False)
        # computing wta errors
        cos_sim = flow_cos_sim(pred_flow, gt_flow, mask=True, seg=seg).reshape(bs, self.num_wta_trials)
        rmse = flow_rmse(pred_flow, gt_flow, mask=False, seg=None).reshape(bs, self.num_wta_trials)
        pred_flow = pred_flow.reshape(bs, self.num_wta_trials, -1, 3)
        winner = torch.argmin(rmse, dim=-1)
        # logging
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_flows_wta = pred_flow[torch.arange(bs), winner]
        self.log_dict(
            {
                f"{mode}_wta/cos_sim": cos_sim_wta.mean(),
                f"{mode}_wta/rmse": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar = True,
        )
        return pred_flows_wta, cos_sim_wta, rmse_wta
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps,
            num_training_steps=self.num_training_steps,
        )
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        self.train() # what is this line doing?
        t = torch.randint(
            0, self.diff_train_steps, (self.batch_size,), device=self.device
        ).long()
        _, loss = self(batch, t, "train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        # if new epoch, clear cache
        # if batch_idx == 0:
        #     self.mode_distribution_cache = {"x": None, "y": None}
        with torch.no_grad():
            pred_flows_wtas, cos_sim_wtas, rmse_wtas = self.predict_wta(batch, mode="val")
        return {
            "loss": rmse_wtas,
            "cos_sim": cos_sim_wtas,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # no test for now
        pass


# TODO: inference module


if __name__ == "__main__":
    # test
    model = DiffusionFlowBase()
    print(model)

    input = torch.randn(2, 1867, 6)
    ts = torch.tensor([2, 3])
    output = model(input, ts)
    print('Output: ', output.shape)