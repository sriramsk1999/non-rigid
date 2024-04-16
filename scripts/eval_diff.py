import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from pathlib import Path
import os

from non_rigid.datasets.microwave_flow import MicrowaveFlowDataset, MicrowaveFlowDataModule, DATASET_FN
from non_rigid.models.df_base import DiffusionFlowBase
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    match_fn,
)

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
from non_rigid.models.dit.diffusion import create_diffusion
from tqdm import tqdm
import numpy as np


def predict(network, batch, device, diffusion, trials, sample_size):
    # pos, gt_flow, seg, t_wc, goal = batch
    pos = batch['pc_init']
    gt_flow = batch['flow']
    seg = batch['seg']
    t_wc = batch['t_wc']
    goal = batch['goal']
    
    pos = pos.to(device)
    gt_flow = gt_flow.to(device)
    seg = seg.to(device)
    # reshaping and expanding for wta
    bs = pos.shape[0]
    pos = pos.transpose(-1, -2).unsqueeze(1).expand(-1, trials, -1, -1).reshape(bs * trials, 3, sample_size)
    gt_flow = gt_flow.unsqueeze(1).expand(-1, trials, -1, -1).reshape(bs * trials, sample_size, 3)
    seg = seg.unsqueeze(1).expand(-1, trials, -1).reshape(bs * trials, -1)
    # generating latents and running diffusion
    model_kwargs = dict(pos=pos)
    z = torch.randn(
        bs * trials, 3, sample_size, device=device
    )
    pred_flow, results = diffusion.p_sample_loop(
        network, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    pred_flow = pred_flow.permute(0, 2, 1).reshape(bs, trials, -1, 3)
    pos = pos.permute(0, 2, 1).reshape(bs, trials, -1, 3)
    gt_flow = gt_flow.reshape(bs, trials, -1, 3)
    return pos, pred_flow, gt_flow, t_wc, goal


def predict_wta(network, batch, device, diffusion, num_wta_trials=50, sample_size=1200):
    # pos, gt_flow, seg, _, goal = batch
    pos = batch['pc_init']
    gt_flow = batch['flow']
    seg = batch['seg']
    goal = batch['goal']
    
    pos = pos.to(device)
    gt_flow = gt_flow.to(device)
    seg = seg.to(device)
    # reshaping and expanding for wta
    bs = pos.shape[0]
    pos = pos.transpose(-1, -2).unsqueeze(1).expand(-1, num_wta_trials, -1, -1).reshape(bs * num_wta_trials, -1, sample_size)
    gt_flow = gt_flow.unsqueeze(1).expand(-1, num_wta_trials, -1, -1).reshape(bs * num_wta_trials, sample_size, -1)
    seg = seg.unsqueeze(1).expand(-1, num_wta_trials, -1).reshape(bs * num_wta_trials, -1)
    # generating latents and running diffusion
    model_kwargs = dict(pos=pos)
    z = torch.randn(
        bs * num_wta_trials, 3, sample_size, device=device
    )
    pred_flow, results = diffusion.p_sample_loop(
        network, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    pred_flow = pred_flow.permute(0, 2, 1)
    # computing wta errors
    cos_sim = flow_cos_sim(pred_flow, gt_flow, mask=True, seg=seg).reshape(bs, num_wta_trials)
    rmse = flow_rmse(pred_flow, gt_flow, mask=False, seg=None).reshape(bs, num_wta_trials)
    pred_flow = pred_flow.reshape(bs, num_wta_trials, -1, 3)
    winner = torch.argmin(rmse, dim=-1)
    # logging
    cos_sim_wta = cos_sim[torch.arange(bs), winner]
    rmse_wta = rmse[torch.arange(bs), winner]
    pred_flows_wta = pred_flow[torch.arange(bs), winner]
    return pred_flows_wta, cos_sim_wta, rmse_wta

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    ######################################################################
    # Torch settings.
    ######################################################################

    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")

    # Global seed for reproducibility.
    L.seed_everything(42)

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################

    # data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    # datamodule = MicrowaveFlowDataModule(
    #     root=data_root,
    #     batch_size=cfg.training.batch_size,
    #     val_batch_size=cfg.training.val_batch_size,
    #     num_workers=cfg.resources.num_workers,
    # )

    # TODO: for now, don't log to wandb
    ######################################################################
    # Create the network(s) which will be evaluated (same as training).
    # You might want to put this into a "create_network" function
    # somewhere so train and eval can be the same.
    #
    # We'll also load the weights.
    ######################################################################

    # Model architecture is dataset-dependent, so we have a helper
    # function to create the model (while separating out relevant vals).
    network = DiffusionFlowBase(
        in_channels=cfg.model.in_channels,
        learn_sigma=cfg.model.learn_sigma,
        model=cfg.model.dit_arch,
    )

    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    if checkpoint_reference.startswith(cfg.wandb.entity):
        api = wandb.Api()
        artifact_dir = cfg.wandb.artifact_dir
        artifact = api.artifact(checkpoint_reference, type="model")
        ckpt_file = artifact.get_path("model.ckpt").download(root=artifact_dir)
    else:
        ckpt_file = checkpoint_reference

    # Load the network weights.
    ckpt = torch.load(ckpt_file)
    network.load_state_dict(
        {k.partition(".")[2]: v for k, v, in ckpt["state_dict"].items()}
    )
    # set model to eval mode
    network.eval()
    # move network to gpu for evaluation
    if torch.cuda.is_available():
        network.cuda()



    device = "cuda"
    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    train_dataset = DATASET_FN[cfg.dataset.type](data_root, "train")
    val_dataset = DATASET_FN[cfg.dataset.type](data_root, "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    num_wta_trials = 50
    diffusion = create_diffusion(timestep_respacing=None, diffusion_steps=cfg.model.diff_train_steps)

    get_train_dataset_metrics = True
    get_val_dataset_metrics = True
    visualize_all_predictions = False
    visualize_single_prediction = True
    visualize_single_prediction_idx = 10

    import plotly.express as px

    # train dataset metrics
    if get_train_dataset_metrics:
        cos_wtas = []
        rmse_wtas = []
        goals = []
        for batch in tqdm(train_loader):
            goals.append(batch['goal'])
            pred_flows_wta, cos_sim_wta, rmse_wta = predict_wta(network, batch, device, diffusion, num_wta_trials=num_wta_trials)
            cos_wtas.append(cos_sim_wta.cpu())
            rmse_wtas.append(rmse_wta.cpu())
        cos_wtas = torch.cat(cos_wtas)
        rmse_wtas = torch.cat(rmse_wtas)
        goals = torch.cat(goals)
        print("Train Cosine Similarity: ", torch.mean(cos_wtas))
        print("Train RMSE: ", torch.mean(rmse_wtas))
        fig = px.scatter(x=goals, y=cos_wtas,
                        labels={"x": "Goal", "y": "Cosine Similarity"},
                        title="Train Cosine Similarity Distribution")
        fig.show()
        fig = px.scatter(x=goals, y=rmse_wtas,
                        labels={"x": "Goal", "y": "RMSE"},
                        title="Train RMSE Distribution")
        fig.show()


    # val dataset metrics
    if get_val_dataset_metrics:
        cos_wtas = []
        rmse_wtas = []
        goals = []
        for batch in tqdm(val_loader):
            goals.append(batch['goal'])
            pred_flows_wta, cos_sim_wta, rmse_wta = predict_wta(network, batch, device, diffusion, num_wta_trials=num_wta_trials)
            cos_wtas.append(cos_sim_wta.cpu())
            rmse_wtas.append(rmse_wta.cpu())
        cos_wtas = torch.cat(cos_wtas)
        rmse_wtas = torch.cat(rmse_wtas)
        goals = torch.cat(goals)
        print("Val Cosine Similarity: ", torch.mean(cos_wtas))
        print("Val RMSE: ", torch.mean(rmse_wtas))
        fig = px.scatter(x=goals, y=cos_wtas, 
                        labels={"x": "Goal", "y": "Cosine Similarity"},
                        title="Val. Cosine Similarity Distribution")
        fig.show()
        fig = px.scatter(x=goals, y=rmse_wtas,
                        labels={"x": "Goal", "y": "RMSE"},
                        title="Val. RMSE Distribution")
        fig.show()


    import rpad.visualize_3d.plots as vpl
    if visualize_all_predictions:
        pred_pcs_t = []
        gt_pcs_t = []
        # visualizing predictions
        for batch in tqdm(val_loader):
            pos, pred_flow, gt_flow, t_wc, goal = predict(network, batch, device, diffusion, 10, 1200)
            pred_pc = pos + pred_flow
            gt_pc = pos[:, 0, ...] + gt_flow[:, 0, ...]

            # transform gt to world frame
            gt_pc_t = torch.cat([gt_pc.cpu(), torch.ones((gt_pc.shape[0], gt_pc.shape[1], 1))], axis=-1) @ t_wc.permute(0, 2, 1)[..., :3]
            gt_pc_t = torch.flatten(gt_pc_t, end_dim=-2)
            gt_pcs_t.append(gt_pc_t.numpy())

            # transform pred to world frame
            t_wc = t_wc.unsqueeze(1).expand(-1, 10, -1, -1).reshape(-1, 4, 4)
            pred_pc = pred_pc.reshape(-1, 1200, 3)
            pred_pc_t = torch.cat([pred_pc.cpu(), torch.ones((pred_pc.shape[0], pred_pc.shape[1], 1))], axis=-1) @ t_wc.permute(0, 2, 1)[..., :3]
            pred_pc_t = torch.flatten(pred_pc_t, end_dim=-2)        
            pred_pcs_t.append(pred_pc_t.numpy())
        
        pred_pcs_t = np.concatenate(pred_pcs_t)
        gt_pcs_t = np.concatenate(gt_pcs_t)
        pred_seg = np.ones((pred_pcs_t.shape[0],), dtype=np.int64)
        gt_seg = np.zeros((gt_pcs_t.shape[0],), dtype=np.int64)
        
        fig = vpl.segmentation_fig(
            np.concatenate((pred_pcs_t, gt_pcs_t)), np.concatenate((pred_seg, gt_seg))
        )
        fig.show()

    # plot single diffusion chain
    from non_rigid.utils.vis_utils import FlowNetAnimation
    if visualize_single_prediction:
        animation = FlowNetAnimation()
        # pos, _, _, t_wc, _  = val_dataset[1]
        pos = torch.tensor(val_dataset[visualize_single_prediction_idx]['pc_init'])
        t_wc = torch.tensor(val_dataset[visualize_single_prediction_idx]['t_wc'])

        pos = pos.unsqueeze(0).cuda().transpose(-1, -2)
        z = torch.randn(1, 1200, 3).cuda().transpose(-1, -2)
        model_kwargs = dict(pos=pos)
        # denoise
        pred_flow, results = diffusion.p_sample_loop(
            network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
        pred_flow = pred_flow.squeeze(0).permute(1, 0)
        pcd = pos.squeeze().permute(1, 0).cpu().numpy()
        for noise_step in tqdm(results[0:]):
            pred_flow_step = noise_step.squeeze(0).permute(1, 0)
            animation.add_trace(
                torch.as_tensor(pcd),
                torch.as_tensor([pcd]),
                torch.as_tensor([pred_flow_step.cpu().numpy()]),
                "red",
            )
        fig = animation.animate()
        fig.show()

    quit()


if __name__ == "__main__":
    main()