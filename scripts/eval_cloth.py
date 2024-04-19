import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from pathlib import Path
import os

import plotly.express as px

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    match_fn,
    flatten_outputs
)

from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
from non_rigid.models.dit.diffusion import create_diffusion
from tqdm import tqdm
import numpy as np

def predict(network, batch, device, diffusion, trials, sample_size):
    # pos, gt_flow, seg, t_wc, goal = batch
    # pos, gt_flow, seg = batch
    pos = batch["pc_init"]
    gt_flow = batch["flow"]
    seg = batch["seg"]
    # seg = torch.ones_like(gt_flow[..., 0], device=device)
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
    return pos, pred_flow, gt_flow



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

    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    # datamodule = MicrowaveFlowDataModule(
    #     root=data_root,
    #     batch_size=cfg.training.batch_size,
    #     val_batch_size=cfg.training.val_batch_size,
    #     num_workers=cfg.resources.num_workers,
    # )
    # if cfg.dataset.type in ["articulated", "articulated_multi"]:
    #     dm = MicrowaveFlowDataModule
    if cfg.dataset.type == "cloth":
        dm = ProcClothFlowDataModule

    datamodule = dm(
        root=data_root,
        batch_size=cfg.inference.batch_size,
        val_batch_size=cfg.inference.val_batch_size,
        num_workers=cfg.resources.num_workers,
        type=cfg.dataset.type,
    )
    datamodule.setup(stage="predict")

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
    # if checkpoint_reference.startswith(cfg.wandb.entity):
    if True:
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
    cfg.inference.sample_size = cfg.dataset.sample_size
    model = FlowPredictionInferenceModule(network, inference_cfg=cfg.inference, model_cfg=cfg.model)

    device = "cuda"
    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    sample_size = cfg.dataset.sample_size
    # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    train_dataset = ProcClothFlowDataset(data_root, "train")
    val_dataset = ProcClothFlowDataset(data_root, "val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    num_wta_trials = 50
    diffusion = create_diffusion(timestep_respacing=None, diffusion_steps=cfg.model.diff_train_steps)
    

    GET_TRAIN_METRICS = True
    GET_VAL_METRICS = True
    VISUALIZE_ALL = True
    VISUALIZE_SINGLE = True
    VISUALIZE_SINGLE_IDX = 0

    MMD_METRICS = True
    PRECISION_METRICS = True



    ######################################################################
    # Create the trainer.
    # Bit of a misnomer here, we're not doing training. But we are gonna
    # use it to set up the model appropriately and do all the batching
    # etc.
    #
    # If this is a different kind of downstream eval, chuck this block.
    ######################################################################

    trainer = L.Trainer(
        accelerator="gpu",
        devices=cfg.resources.gpus,
        # precision="16-mixed",
        precision="32-true",
        logger=False,
    )

    ######################################################################
    # Run the model on the train/val/test sets.
    # This outputs a list of dictionaries, one for each batch. This
    # is annoying to work with, so later we'll flatten.
    #
    # If a downstream eval, you can swap it out with whatever the eval
    # function is.
    ######################################################################

    
    if MMD_METRICS:
        train_outputs, val_outputs = trainer.predict(
            model,
            dataloaders=[
                datamodule.train_dataloader(),
                datamodule.val_dataloader(),
            ]
            )

        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            # plot histogram
            fig = px.histogram(outputs["rmse"], nbins=100, title=f"{name} MMD RMSE")
            fig.show()
            # Compute the metrics.
            cos_sim = torch.mean(outputs["cos_sim"])
            rmse = torch.mean(outputs["rmse"])
            print(f"{name} cos sim: {cos_sim}, rmse: {rmse}")
            # TODO: THIS SHOULD ALSO LOG HISTOGRAMS FROM BEFORE MEANS


    # TODO: for now, all action inputs are the same, so just grab the first one
    
    if PRECISION_METRICS:
        device = "cuda"
        data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
        train_dataset = ProcClothFlowDataset(data_root, "train")
        val_dataset = ProcClothFlowDataset(data_root, "val")
        num_samples = cfg.inference.num_wta_trials
        # generate predictions
        model.to(device)
        action_input = train_dataset[0]["pc_init"].unsqueeze(0).to(device)
        pos, pred_flows = model.predict(action_input, num_samples, False)

        # top - down heat map
        preds = (pos + pred_flows).cpu().numpy().reshape(-1, 3)
        fig = px.density_heatmap(x=preds[:, 0], y=preds[:, 1], 
                                 nbinsx=100, nbinsy=100,
                                 title="Predicted Flows XY Heatmap")
        fig.show()

        # load val dataset at once
        val_flows = []
        for i in range(len(val_dataset)):
            val_flows.append(val_dataset[i]["flow"])
        val_flows = torch.stack(val_flows).to(device)
 
        precision_rmses = []
        for i in tqdm(range(pred_flows.shape[0])):
            pf = pred_flows[[i]].expand(val_flows.shape[0], -1, -1)
            rmse = flow_rmse(pf, val_flows, mask=False, seg=None)
            rmse_match = torch.min(rmse)
            precision_rmses.append(rmse_match)
        precision_rmses = torch.stack(precision_rmses)
        fig = px.histogram(precision_rmses.cpu().numpy(), 
                           nbins=20, title="Precision RMSE")
        fig.show()
        print(precision_rmses.mean())
        quit()



    if VISUALIZE_ALL:
        anchor_pcs = []
        for i in range(cfg.dataset.num_anchors):
            anchor_pcs.append(np.load(data_root / f"anchor_{i}.npz")["pc"])
        anchor_pc = np.concatenate(anchor_pcs)


        import rpad.visualize_3d.plots as vpl
        pred_pcs_t = []
        gt_pcs_t = []
        # visualizing predictions
        for batch in tqdm(val_loader):
            pos, pred_flow, gt_flow = predict(network, batch, device, diffusion, 10, 213)
            
            pred_pc = pos + pred_flow
            gt_pc = pos[:, 0, ...] + gt_flow[:, 0, ...]

            gt_pc_t = gt_pc.flatten(end_dim=-2).cpu().numpy()
            pred_pc_t = pred_pc.flatten(end_dim=-2).cpu().numpy()
            gt_pcs_t.append(gt_pc_t)
            pred_pcs_t.append(pred_pc_t)
        
        pred_pcs_t = np.concatenate(pred_pcs_t)
        gt_pcs_t = np.concatenate(gt_pcs_t)
        # pred_seg = np.ones(pred_pcs_t.shape[0], dtype=np.int64)
        pred_seg = np.array([np.arange(2, 322)] * 213).T.flatten()

        gt_seg = np.zeros(gt_pcs_t.shape[0], dtype=np.int64)
        anchor_seg = np.ones(anchor_pc.shape[0], dtype=np.int64)*1

        action_pc = train_dataset[0]["pc_init"]
        action_seg = np.ones(action_pc.shape[0], dtype=np.int64)*1

        fig = vpl.segmentation_fig(
            # np.concatenate((pred_pcs_t, gt_pcs_t, anchor_pc, action_pc)), 
            # np.concatenate((pred_seg, gt_seg, anchor_seg, action_seg)),
            np.concatenate((pred_pcs_t, anchor_pc, action_pc)), 
            np.concatenate((pred_seg, anchor_seg, action_seg))
        )
        fig.show()


    # plot single diffusion chain
    from non_rigid.utils.vis_utils import FlowNetAnimation
    if VISUALIZE_SINGLE:
        animation = FlowNetAnimation()
        pos  = val_dataset[0]["pc_init"]
        pos = pos.unsqueeze(0).cuda().transpose(-1, -2)
        z = torch.randn(1, 213, 3).cuda().transpose(-1, -2)
        model_kwargs = dict(pos=pos)
        # denoise
        pred_flow, results = diffusion.p_sample_loop(
            network, z.shape, z, clip_denoised=False,
            model_kwargs=model_kwargs, progress=True, device=device
        )
        pred_flow = pred_flow.squeeze(0).permute(1, 0)
        pcd = pos.squeeze().permute(1, 0).cpu()

        combined_pcd = torch.cat([pcd, torch.as_tensor(anchor_pc)], dim=0)

        for noise_step in tqdm(results[0:]):
            pred_flow_step = noise_step.squeeze(0).permute(1, 0).cpu()
            animation.add_trace(
                combined_pcd,
                [pcd],
                [pred_flow_step],#combined_flow,
                "red",
            )

        fig = animation.animate()
        fig.show()


if __name__ == "__main__":
    main()