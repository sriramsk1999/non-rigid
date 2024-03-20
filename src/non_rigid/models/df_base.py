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
    

class DiffusionFlowTrainingModule(L.LightningModule):
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

        self.mode_distribution_cache = {"x": None, "y": None}

    def forward(self, x, t, mode="train"):
        # get flow and pos
        pc, _ = x
        pos = pc[..., :3]
        flow = pc[..., 3:6]
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
    
    # this is only for inference?
    def predict(self):
        pass

    def predict_wta(self, batch, mode):
        # TODO: this will not work if batch size != 1
        assert batch[0].shape[0] == 1, "WTA requires batch size of 1"
        pc, goals, angles = batch
        angles = angles.squeeze(0)
        seg = pc[..., 3].bool().expand(self.num_wta_trials, -1)
        pos = pc[..., :3].expand(self.num_wta_trials, -1, -1).transpose(-1, -2)
        # generating latents and running diffusion
        model_kwargs = dict(pos=pos)
        z = torch.randn(
            self.num_wta_trials, 3, self.sample_size, device=self.device
        )
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=self.device
        )
        pred_flow = pred_flow.permute(0, 2, 1)

        cos_sim_wtas = []
        rmse_wtas = []
        pred_flows_wtas = []
        pairwise_dists = []
        # computing wta errors for each goal
        for i in range(goals.shape[1]):
            # extracting goal flow
            goal_i = goals[:, i, ...].expand(self.num_wta_trials, -1, -1)
            # computing errors
            cos_sim = flow_cos_sim(pred_flow, goal_i, mask=True, seg=seg)
            rmse = flow_rmse(pred_flow, goal_i, mask=False, seg=None)
            # wta based on rmse
            winner = torch.argmin(rmse, dim=-1)
            cos_sim_wtas.append(cos_sim[winner])
            rmse_wtas.append(rmse[winner])
            pred_flows_wtas.append(pred_flow[winner, ...])
            # storing rmse's for nn
            pairwise_dists.append(rmse)
        # caching mode distributions
        pairwise_dists = torch.stack(pairwise_dists, dim=0)
        _, closest_mode = pc_nn(pairwise_dists)
        mode_freqs = torch.bincount(closest_mode, minlength=goals.shape[1])
        if self.mode_distribution_cache["x"] is None:
            self.mode_distribution_cache["x"] = angles[1:]
            self.mode_distribution_cache["y"] = mode_freqs
        else:
            self.mode_distribution_cache["y"] += mode_freqs
        # logging
        cos_sim_wtas = torch.stack(cos_sim_wtas, dim=0)
        rmse_wtas = torch.stack(rmse_wtas, dim=0)
        pred_flows_wtas = torch.stack(pred_flows_wtas, dim=0)
        self.log_dict(
            {
                f"{mode}_wta/cos_sim": cos_sim_wtas.mean(),
                f"{mode}_wta/rmse": rmse_wtas.mean(),
            },
            add_dataloader_idx=False,
            prog_bar = True,
        )
        return pred_flows_wtas, cos_sim_wtas, rmse_wtas
        quit()
        return pred_flows_wtas, cos_sim_wtas, rmse_wtas

        bs = batch[0].shape[0]
        nt = self.num_wta_trials
        # TODO: for each val. run through dataset, compute angle-specific metrics
        pc, _ = batch
        seg = pc[..., 6].bool().unsqueeze(1).expand(-1, nt, -1)
        pos = pc[..., :3].unsqueeze(1).expand(-1, nt, -1, -1)
        flow = pc[..., 3:6].unsqueeze(1).expand(-1, nt, -1, -1)
        # shaping for diffusion
        seg = seg.reshape(bs*nt, -1)
        pos = pos.reshape(bs*nt, self.sample_size, -1).transpose(-1, -2)
        flow = flow.reshape(bs*nt, self.sample_size, -1)
        # running diffusion
        model_kwargs = dict(pos=pos)
        z = torch.randn(
            bs*nt, 3, self.sample_size, device=self.device
        )
        pred_flow, results = self.diffusion.p_sample_loop(
            self.network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=self.device
        )
        pred_flow = pred_flow.permute(0, 2, 1)
        # computing errors
        cos_sim = flow_cos_sim(pred_flow, flow, mask=True, seg=seg).reshape(bs, nt)
        rmse = flow_rmse(pred_flow, flow, mask=False, seg=None).reshape(bs, nt)
        pred_flow = pred_flow.reshape(bs, nt, self.sample_size, -1)
        # wta based on rmse
        winner = torch.argmin(rmse, dim=-1)
        cos_sim_wta = cos_sim[torch.arange(bs), winner]
        rmse_wta = rmse[torch.arange(bs), winner]
        pred_flow_wta = pred_flow[torch.arange(bs), winner, ...]

        # log
        self.log_dict(
            {
                f"{mode}_wta/cos_sim": cos_sim_wta.mean(),
                f"{mode}_wta/rmse": rmse_wta.mean(),
            },
            add_dataloader_idx=False,
            prog_bar = True,
        )
        # TODO: return best results for visualization purposes?
        return pred_flow_wta, cos_sim_wta, rmse_wta

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
        if batch_idx == 0:
            self.mode_distribution_cache = {"x": None, "y": None}
        with torch.no_grad():
            pred_flows_wtas, cos_sim_wtas, rmse_wtas = self.predict_wta(batch, mode="val")
        # TODO: viz metrics for pred_flow, cache joint angle specific distributions?
        return {
            "loss": rmse_wtas,
            "cos_sim": cos_sim_wtas,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # no test for now
        pass

    def make_plots(self):
        # plotting mode distributions as bar chart
        fig = px.bar(
            x=self.mode_distribution_cache["x"].cpu(),
            y=self.mode_distribution_cache["y"].cpu(),
            labels={"x": "Joint Angle", "y": "Frequency"},
            title="Mode Distribution",
        )
        return {"mode_distribution": fig}



if __name__ == "__main__":
    # test
    model = DiffusionFlowBase()
    print(model)

    input = torch.randn(2, 1867, 6)
    ts = torch.tensor([2, 3])
    output = model(input, ts)
    print('Output: ', output.shape)