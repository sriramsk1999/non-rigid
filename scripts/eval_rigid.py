from functools import partial
import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from pathlib import Path
import os

import plotly.express as px
import plotly.graph_objects as go

from rpad.visualize_3d.plots import flow_fig, _flow_traces, pointcloud, _3d_scene
from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
from non_rigid.datasets.rigid import RigidPointDataset, RigidFlowDataset, RigidDataModule
from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionTrainingModule
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

    print(F'Config:\n{cfg}')


    ######################################################################
    # Load the datamodule
    ######################################################################
    
    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    if cfg.dataset.type == "cloth":
        dm = ProcClothFlowDataModule
    elif cfg.dataset.type in ["rigid_point", "rigid_flow"]:
        dm = partial(RigidDataModule, dataset_cfg=cfg.dataset) # TODO: Remove the need to use partial

    datamodule = dm(
        root=data_root,
        batch_size=cfg.inference.batch_size,
        val_batch_size=cfg.inference.val_batch_size,
        num_workers=cfg.resources.num_workers,
        type=cfg.dataset.type,
    )
    datamodule.setup(stage="predict")

    
    ######################################################################
    # Set up the network.
    ######################################################################
    
    # function to create the model (while separating out relevant vals).
    network = DiffusionFlowBase(
        in_channels=cfg.model.in_channels,
        learn_sigma=cfg.model.learn_sigma,
        model=cfg.model.dit_arch,
        model_cfg=cfg.model,
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
    
    
    ######################################################################
    # Set up the model
    ######################################################################
    
    if cfg.model.type in ["flow", "flow_cross"]:
        model = FlowPredictionInferenceModule(network, inference_cfg=cfg.inference, model_cfg=cfg.model)
    elif cfg.model.type == "point_cross":
        model = PointPredictionTrainingModule(network, training_cfg=cfg.inference, model_cfg=cfg.model)
    else:
        raise ValueError(f"Model type {cfg.model.type} not recognized.")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    device = model.device

    ######################################################################
    # Set up the individual datasets
    ######################################################################

    # data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    # sample_size = cfg.dataset.sample_size
    # # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    # if cfg.dataset.type == "cloth":
    #     train_dataset = ProcClothFlowDataset(data_root, "train")
    #     val_dataset = ProcClothFlowDataset(data_root, "val")
    # elif cfg.dataset.type == "rigid_point":
    #     train_dataset = RigidPointDataset(data_root, "train", dataset_cfg=cfg.dataset)
    #     val_dataset = RigidPointDataset(data_root, "val", dataset_cfg=cfg.dataset)
    # elif cfg.dataset.type == "rigid_flow":
    #     train_dataset = RigidFlowDataset(data_root, "train", dataset_cfg=cfg.dataset)
    #     val_dataset = RigidFlowDataset(data_root, "val", dataset_cfg=cfg.dataset)

    MMD_METRICS = True
    PRECISION_METRICS = True
    VISUALIZE_ALL = True
    VISUALIZE_SINGLE = True
    VISUALIZE_SINGLE_IDX = 0


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
        print(f'Calculating MMD metrics')
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
        print(f'Calculating precision metrics')
        device = "cuda"
        data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
        num_samples = cfg.inference.num_wta_trials
        # generate predictions
        model.to(device)

        if cfg.model.type == "point_cross":
            data = datamodule.val_dataset[0]
            pos = data["pc_init"].unsqueeze(0).to(device)
            pc_action = data["pc_action"].unsqueeze(0).to(device)
            pc_anchor = data["pc_anchor"].unsqueeze(0).to(device)
            
            bs = 1
            pc_action = (
                pc_action.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            pc_anchor = (
                pc_anchor.transpose(-1, -2)
                .unsqueeze(1)
                .expand(-1, num_samples, -1, -1)
                .reshape(bs * num_samples, -1, cfg.dataset.sample_size)
            )
            
            model_kwargs = dict(
                y=pc_anchor,
                x0=pc_action
            )
            model_kwargs, pred_actions, results = model.predict(bs, model_kwargs, num_samples, False)
            
            preds = pred_actions.cpu().numpy().reshape(-1, 3)

            # top - down heat map
            fig = px.density_heatmap(x=preds[:, 0], y=preds[:, 1], 
                                        nbinsx=100, nbinsy=100,
                                        title="Predicted Flows XY Heatmap")
            fig.show()

            # load val dataset at once
            val_action = []
            for i in range(len(datamodule.val_dataset)):
                val_action.append(datamodule.val_dataset[i]["pc_init"])
            val_action = torch.stack(val_action).to(device)

            fig = go.Figure()
            fig.add_trace(pointcloud(pred_actions[0].detach().cpu(), downsample=1, scene="scene1", name="Predicted PCD"))
            for i in range(val_action.shape[0]):
                fig.add_trace(pointcloud(val_action[i].detach().cpu(), downsample=1, scene="scene1", name=f"Goal Action PCD {i}"))
            fig.update_layout(scene1=_3d_scene(pred_actions[0].detach().cpu()))
            fig.show()

            precision_rmses = []
            for i in tqdm(range(pred_actions.shape[0])):
                pa = pred_actions[[i]].expand(val_action.shape[0], -1, -1)
                rmse = flow_rmse(pa, val_action, mask=False, seg=None)
                rmse_match = torch.min(rmse)
                precision_rmses.append(rmse_match)
            precision_rmses = torch.stack(precision_rmses)
            fig = px.histogram(precision_rmses.cpu().numpy(), 
                                nbins=20, title="Precision RMSE")
            fig.show()
            print(precision_rmses.mean())
        else:
            raise ValueError(f"Model type {cfg.model.type} not recognized.")


    if VISUALIZE_ALL:
        import rpad.visualize_3d.plots as vpl
        pred_pcs_t = []
        gt_pcs_t = []
        model.to(device)
        
        # visualizing predictions
        for batch in tqdm(datamodule.val_dataloader()):
            pred_actions, _, _ = model.predict_wta(batch, "val")
            
            pred_pc = pred_actions
            gt_pc = batch["pc_init"]

            gt_pc_t = gt_pc.flatten(end_dim=-2).cpu().numpy()
            pred_pc_t = pred_pc.flatten(end_dim=-2).cpu().numpy()
            gt_pcs_t.append(gt_pc_t)
            pred_pcs_t.append(pred_pc_t)
        
        pred_pcs_t = np.concatenate(pred_pcs_t)
        pred_seg = np.array([np.arange(3, 19)] * cfg.dataset.sample_size).T.flatten()

        gt_pcs_t = np.concatenate(gt_pcs_t)
        gt_seg = np.array([np.arange(19, 35)] * cfg.dataset.sample_size).T.flatten()
        
        # Get other pcds from single example. TODO: These change across examples, change this to something better
        data = datamodule.val_dataset[0]
        
        anchor_pc = data["pc_anchor"]
        anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)*1

        pos = data["pc_init"]
        pos_seg = np.ones(pos.shape[0], dtype=np.int64)*1
        
        action_pc = data["pc_action"]
        action_seg = np.full(action_pc.shape[0], 2, dtype=np.int64)

        fig = vpl.segmentation_fig(
            # np.concatenate((pred_pcs_t, gt_pcs_t, anchor_pc, action_pc)), 
            # np.concatenate((pred_seg, gt_seg, anchor_seg, action_seg)),
            np.concatenate((pred_pcs_t, anchor_pc, action_pc, pos)), 
            np.concatenate((pred_seg, anchor_seg, action_seg, pos_seg)),
        )
        fig.show()


    # plot single diffusion chain
    from non_rigid.utils.vis_utils import FlowNetAnimation
    if VISUALIZE_SINGLE:
        model.to(device)
        animation = FlowNetAnimation()
        
        data = datamodule.val_dataset[VISUALIZE_SINGLE_IDX]
        pos = data["pc_init"]
        pc_action = data["pc_action"]
        pc_anchor = data["pc_anchor"]
        
        pc_action = pc_action.unsqueeze(0).permute(0, 2, 1).to(device)
        pc_anchor = pc_anchor.unsqueeze(0).permute(0, 2, 1).to(device)
        
        # z = torch.randn(1, cfg.dataset.sample_size, 3).cuda().transpose(-1, -2) * model.noise_scale
        model_kwargs = dict(
            y=pc_anchor,
            x0=pc_action, 
        )
        # # denoise
        # pred_pos, results = diffusion.p_sample_loop(
        #     network, z.shape, z, clip_denoised=False,
        #     model_kwargs=model_kwargs, progress=True, device=device
        # )
        model_kwargs, pred_pos, results = model.predict(1, model_kwargs, 1, False)
        
        pred_pos = pred_pos[0].cpu()
        pos = pos.cpu()
        pcd = pc_action[0].permute(1, 0).cpu()
        anchor_pc = pc_anchor[0].permute(1, 0).cpu()

        # combined_pcd = torch.cat([torch.as_tensor(anchor_pc), torch.as_tensor(pos)], dim=0)

        # for noise_step in tqdm(results[0:]):
        #     pred_pos_step = noise_step[viz_idx].permute(1, 0).cpu()
        #     animation.add_trace(
        #         combined_pcd,
        #         [pcd],
        #         [pred_pos_step - pos],#combined_flow,
        #         "red",
        #     )

        # fig = animation.animate()
        # fig.show()
        
        fig = go.Figure()
        
        fig.add_trace(pointcloud(pos, downsample=1, scene="scene1", name="Goal Action PCD"))
        fig.add_trace(pointcloud(anchor_pc, downsample=1, scene="scene1", name="Anchor PCD"))
        fig.add_trace(pointcloud(pcd, downsample=1, scene="scene1", name="Context Action PCD"))
        fig.add_trace(pointcloud(pred_pos, downsample=1, scene="scene1", name="Predicted PCD"))
        
        fig.update_layout(scene1=_3d_scene(torch.cat([pred_pos, pos, anchor_pc], dim=0).detach().cpu()))
        fig.show()
        
        
        # fig.write_html("/home/odonca/workspace/rpad/non-rigid/animation.html")
        


if __name__ == "__main__":
    main()