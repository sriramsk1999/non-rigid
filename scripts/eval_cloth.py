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
from plotly.subplots import make_subplots

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.utils.vis_utils import FlowNetAnimation
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

import rpad.visualize_3d.plots as vpl



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
    if cfg.dataset.type in ["cloth", "cloth_point"]:
        dm = ProcClothFlowDataModule
    else:
        raise NotImplementedError('This script is only for cloth evaluations.')

    datamodule = dm(
        root=data_root,
        batch_size=cfg.inference.batch_size,
        val_batch_size=cfg.inference.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    datamodule.setup(stage="predict")


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
    # if torch.cuda.is_available():
    #     network.to(f"cuda:{cfg.resources.gpus[0]}")
    
    # setting sample sizes
    if "scene" in cfg.dataset and cfg.dataset.scene:
        if cfg.model.type != "flow":
            raise NotImplementedError("Scene inputs cannot be used with cross-type models.")
        cfg.inference.sample_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    else:
        cfg.inference.sample_size = cfg.dataset.sample_size_action
        cfg.inference.sample_size_anchor = cfg.dataset.sample_size_anchor

    # override the task type here based on the dataset
    if "cloth" in cfg.dataset.type:
        cfg.task_type = "cloth"
    elif "rigid" in cfg.dataset.type:
        cfg.task_type = "rigid"
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset.type}")
    # create model
    if cfg.model.type in ["flow", "flow_cross"]:
        model = FlowPredictionInferenceModule(network, inference_cfg=cfg.inference, model_cfg=cfg.model)
    elif cfg.model.type in ["point_cross"]:
        model = PointPredictionInferenceModule(
            network, task_type=cfg.task_type, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    model.eval()
    # model.to(f'cuda:{cfg.resources.gpus[0]}')




    # data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    # # sample_size = cfg.dataset.sample_size
    # # data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    # train_dataset = ProcClothFlowDataset(data_root, "train", **cfg.dataset.type_args)
    # val_dataset = ProcClothFlowDataset(data_root, "val", **cfg.dataset.type_args)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
    # # num_wta_trials = 50
    # diffusion = create_diffusion(timestep_respacing=None, diffusion_steps=cfg.model.diff_train_steps)
    
    device = f"cuda:{cfg.resources.gpus[0]}"
    MMD_METRICS = True
    PRECISION_METRICS = False

    VISUALIZE_PREDS = True
    VISUALIZE_SINGLE = True



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
        train_outputs, val_outputs, val_ood_outputs = trainer.predict(
            model,
            dataloaders=[
                datamodule.train_dataloader(),
                *datamodule.val_dataloader(),
            ]
            )
    

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_wta = make_subplots(rows=2, cols=1, shared_xaxes=True)
        color_dict = {
            "train": "blue",
            "val": "red",
            "val_ood": "green",
        }
        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
            (val_ood_outputs, "val_ood")
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            # plot histogram
            fig.add_trace(go.Histogram(
                x=outputs["rmse"].flatten(), 
                nbinsx=100, 
                name=f"{name} RMSE",
                legendgroup=f"{name} RMSE",
                marker=dict(
                    color=color_dict[name],
                ),
                # color=name,
                ), row=1, col=1,
            )
            fig.add_trace(go.Box(
                x=outputs["rmse"].flatten(),
                marker_symbol='line-ns-open',
                marker=dict(
                    color=color_dict[name],
                ),
                boxpoints='all',
                #fillcolor='rgba(0,0,0,0)',
                #line_color='rgba(0,0,0,0)',
                pointpos=0,
                hoveron='points',
                name=f"{name} RMSE",
                showlegend=False,
                legendgroup=f"{name} RMSE",           
                ), row=2, col=1
            )
            # plot wta histogram
            fig_wta.add_trace(go.Histogram(
                x=outputs["rmse_wta"].flatten(), 
                nbinsx=100, 
                name=f"{name} RMSE WTA",
                legendgroup=f"{name} RMSE WTA",
                marker=dict(
                    color=color_dict[name],
                ),
                # color=name,
                ), row=1, col=1,
            )
            fig_wta.add_trace(go.Box(
                x=outputs["rmse_wta"].flatten(),
                marker_symbol='line-ns-open',
                marker=dict(
                    color=color_dict[name],
                ),
                boxpoints='all',
                #fillcolor='rgba(0,0,0,0)',
                #line_color='rgba(0,0,0,0)',
                pointpos=0,
                hoveron='points',
                name=f"{name} RMSE WTA",
                showlegend=False,
                legendgroup=f"{name} RMSE WTA",           
                ), row=2, col=1
            )

            # Compute the metrics.
            cos_sim = torch.mean(outputs["cos_sim"])
            rmse = torch.mean(outputs["rmse"])
            cos_sim_wta = torch.mean(outputs["cos_sim_wta"])
            rmse_wta = torch.mean(outputs["rmse_wta"])
            print(f"{name} cos sim: {cos_sim}, rmse: {rmse}")
            print(f"{name} cos sim wta: {cos_sim_wta}, rmse wta: {rmse_wta}")
        fig.show()
        fig_wta.show()


    if PRECISION_METRICS:
        model.to(device)
        # data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
        #train_dataset = ProcClothFlowDataset(data_root, "train")
        #val_dataset = ProcClothFlowDataset(data_root, "val")
        num_samples = cfg.inference.num_wta_trials
        # generate predictions
        action_input = train_dataset[0]["pc"].unsqueeze(0).to(device)
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


    if VISUALIZE_PREDS:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_ood_dataset, batch_size=1, shuffle=False
        )
        batch = next(iter(dataloader))
        pred_dict = model.predict(batch, 10)
        # extracting anchor point cloud depending on model type
        if cfg.model.type == "flow":
            scene_pc = batch["pc"].flatten(0, 1).cpu().numpy()
            seg = batch["seg"].flatten(0, 1).cpu().numpy()
            anchor_pc = scene_pc[~seg.astype(bool)]
        else:
            anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()

        pred_action = pred_dict["pred_action"].flatten(0, 1).cpu().numpy()
        # color-coded segmentations
        anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
        if cfg.model.type == "flow":
            pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
        else:
            pred_action_size = cfg.dataset.sample_size_action
        pred_action_seg = np.array([np.arange(1, 11)] * pred_action_size).T.flatten()
        # visualize
        fig = vpl.segmentation_fig(
            np.concatenate((anchor_pc, pred_action)),
            np.concatenate((anchor_seg, pred_action_seg)),
        )
        fig.show()


    if VISUALIZE_SINGLE:
        model.to(device)
        dataloader = torch.utils.data.DataLoader(
            datamodule.val_dataset, batch_size=1, shuffle=False
        )
        batch = next(iter(dataloader))
        pred_dict = model.predict(batch, 1)

        results = pred_dict["results"]
        action_pc = batch["pc_action"].flatten(0, 1).cpu()
        # pred_action = .cpu()
        if cfg.model.type == "flow":
            # pcd = batch["pc_action"].flatten(0, 1).cpu()
            pcd = torch.cat([
                batch["pc_action"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu(),
            ]).cpu()
        elif cfg.model.type == "flow_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                batch["pc_action"].flatten(0, 1),
                pred_dict['pred_action'].flatten(0, 1).cpu(),
            ], dim=0).cpu()
        elif cfg.model.type == "point_cross":
            pcd = torch.cat([
                batch["pc_anchor"].flatten(0, 1),
                pred_dict["pred_action"].flatten(0, 1).cpu()
            ], dim=0).cpu()    
        
        # visualize
        animation = FlowNetAnimation()
        for noise_step in tqdm(results):
            pred_step = noise_step[0].permute(1, 0).cpu()
            if cfg.model.type == "point_cross":
                flows = torch.zeros_like(pred_step)
                animation.add_trace(
                    pcd,
                    [flows],
                    [pred_step],
                    "red",
                )
            else:
                animation.add_trace(
                    pcd,
                    [action_pc if cfg.model.type == "flow_cross" else pcd],
                    [pred_step],
                    "red",
                )
        fig = animation.animate()
        fig.show()

if __name__ == "__main__":
    main()