import hydra
import lightning as L
import json
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
from non_rigid.utils.pointcloud_utils import expand_pcd
from tqdm import tqdm
import numpy as np

from pytorch3d.transforms import Transform3d, Rotate, Translate, euler_angles_to_matrix
import rpad.visualize_3d.plots as vpl



def visualize_batched_point_clouds(point_clouds):
    """
    Helper function to visualize a list of batched point clouds. This is meant to be used 
    when visualizing action/anchor/prediction point clouds, without having to add 

    point_clouds: list of point clouds, each of shape (B, N, 3)
    """
    pcs = [pc.cpu().flatten(0, 1) for pc in point_clouds]
    segs = []
    for i, pc in enumerate(pcs):
        segs.append(torch.ones(pc.shape[0]).int() * i)

    return vpl.segmentation_fig(
        torch.cat(pcs),
        torch.cat(segs),
    )




@torch.no_grad()
@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg):
    print(
        json.dumps(
            omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            sort_keys=True,
            indent=4,
        )
    )
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

    device = f"cuda:{cfg.resources.gpus[0]}"

    ######################################################################
    # Create the datamodule.
    # Should be the same one as in training, but we're gonna use val+test
    # dataloaders.
    ######################################################################

    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))

    # based on multi cloth dataset, update data root
    if "multi_cloth" in cfg.dataset:
        if cfg.dataset.multi_cloth.hole == "single":
            print("Running metric evals on single-hole dataset.")
            data_root = data_root / "multi_cloth_1/"
        elif cfg.dataset.multi_cloth.hole == "double":
            print("Running metric evals on double-hole dataset.")
            data_root = data_root / "multi_cloth_2/"
        elif cfg.dataset.multi_cloth.hole == "all":
            print("Running metric evals on single- and double-hole dataset.")
            data_root = data_root / "multi_cloth_all/"
        else:
            raise ValueError(f"Unknown multi-cloth dataset type: {cfg.dataset.multi_cloth.hole}")


    # check that dataset and model types are compatible
    if cfg.model.type != cfg.dataset.type:
        raise ValueError(f"Model type: '{cfg.model.type}' and dataset type: '{cfg.dataset.type}' are incompatible.")
    elif cfg.model.type not in ["flow", "point"]:
        raise ValueError(f"Unsupported model type: {cfg.model.type}.")


    # if cfg.dataset.type in ["cloth", "cloth_point"]:
    if cfg.dataset.name in ["proc_cloth"]:
        dm = ProcClothFlowDataModule
    else:
        raise NotImplementedError('This script is only for cloth evaluations.')
    
    if cfg.viz:
        cfg.dataset.sample_size_action = -1
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
        # model=cfg.model.dit_arch,
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
    ckpt = torch.load(ckpt_file, map_location=device)
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

    # # override the task type here based on the dataset
    # if "cloth" in cfg.dataset.type:
    #     cfg.task_type = "cloth"
    # elif "rigid" in cfg.dataset.type:
    #     cfg.task_type = "rigid"
    # else:
    #     raise ValueError(f"Unsupported dataset type: {cfg.dataset.type}")

    # create model
    # if cfg.model.type in ["flow", "flow_cross"]:
    if cfg.model.type == "flow":
        model = FlowPredictionInferenceModule(
            network, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    # elif cfg.model.type in ["point_cross"]:
    elif cfg.model.type == "point":
        model = PointPredictionInferenceModule(
            network, task_type=cfg.task_type, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.type}")
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
    
    # MMD_METRICS = False
    # PRECISION_METRICS = False

    VISUALIZE_DEMOS = True
    VISUALIZE_PREDS = False
    VISUALIZE_SINGLE = False
    VISUALIZE_PULL = False



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
    
    if cfg.coverage:
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


    if cfg.precision:
        model.to(device)
        # precision_dm = 2

        # creating precision-specific datamodule - needs specific batch size
        # if "multi_cloth" in cfg.dataset:
        #     bs = int(400 / cfg.dataset.multi_cloth.size)
        # else:
        #     raise NotImplementedError("Precision metrics only supported for multi-cloth datasets.")
        
        if int(cfg.dataset.multi_cloth.size) == 1:
            num_samples = 1
            train_bs = 400
            val_bs = 40
        else:
            num_samples = 20
            train_bs = 4
            val_bs = 4


        # cfg.dataset.sample_size_action = -1
        datamodule = dm(
            root=data_root,
            batch_size=train_bs,
            val_batch_size=val_bs,
            num_workers=cfg.resources.num_workers,
            dataset_cfg=cfg.dataset,
        )
        # just need the dataloaders
        datamodule.setup(stage="predict")
        train_loader = datamodule.train_dataloader()
        val_loader, val_ood_loader = datamodule.val_dataloader()

        # TODO: PRED FLOW IS NOT IN THE SAME FRAME AS GT FLOW...
        # ALSO WHY DO THEY GET SO MUCH WORSE FOR VAL AND VAL_OOD


        def precision_eval(dataloader, model, bs):
            precision_rmses = []
            for batch in tqdm(dataloader):
                # generate predictions
                seg = batch["seg"].to(device)
                pred_dict = model.predict(batch, num_samples, progress=False)
                pred_action = pred_dict["pred_action"]#.cpu() # in goal/anchor frame

                rot = batch['rot'].to(device)
                trans = batch['trans'].to(device)
                pc = batch["pc"].to(device)
                # reverting transforms
                T_world2origin = Translate(trans).inverse().compose(
                    Rotate(euler_angles_to_matrix(rot, 'XYZ'))
                )
                T_goal2world = Transform3d(
                    matrix=batch["T_goal2world"].to(device)
                )
                # goal to world, then world to origin
                pc = T_goal2world.transform_points(pc)
                pc = T_world2origin.transform_points(pc)

                # expanding transform to handle pred_action
                T_goal2world = Transform3d(
                    matrix=expand_pcd(T_goal2world.get_matrix(), num_samples)
                )
                T_world2origin = Transform3d(
                    matrix=expand_pcd(T_world2origin.get_matrix(), num_samples)
                )
                pred_action = T_goal2world.transform_points(pred_action)
                pred_action = T_world2origin.transform_points(pred_action)
            
                #fig = visualize_batched_point_clouds([pc_anchor, pc, pred_action])
                #fig.show()

                batch_rmses = []

                for p in pred_action:
                    p = expand_pcd(p.unsqueeze(0), bs)
                    rmse = flow_rmse(p, pc, mask=True, seg=seg)
                    rmse_min = torch.min(rmse)
                    batch_rmses.append(rmse_min)
                precision_rmses.append(torch.tensor(batch_rmses).mean())
            return torch.stack(precision_rmses).mean()

        train_precision_rmse = precision_eval(train_loader, model, train_bs)
        val_precision_rmse = precision_eval(val_loader, model, val_bs)
        val_ood_precision_rmse = precision_eval(val_ood_loader, model, val_bs)
        print("Train Precision RMSE: ", train_precision_rmse)
        print("Val Precision RMSE: ", val_precision_rmse)
        print("Val OOD Precision RMSE: ", val_ood_precision_rmse)


    if cfg.viz:
        if VISUALIZE_DEMOS:
            model.to(device)
            bs = 12
            train_dataloader = torch.utils.data.DataLoader(
                datamodule.train_dataset, batch_size=400, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=40, shuffle=True
            )
            val_ood_loader = torch.utils.data.DataLoader(
                datamodule.val_ood_dataset, batch_size=40, shuffle=True
            )

            train_batch = next(iter(train_dataloader))
            val_batch = next(iter(val_dataloader))
            val_ood_batch = next(iter(val_ood_loader))


            # train_dict = model.predict_wta(train_batch, 'train')
            # val_dict = model.predict_wta(val_batch, 'val')
            # val_ood_dict = model.predict_wta(val_ood_batch, 'val_ood')

            # val_errors = val_dict['rmse']
            # val_ood_errors = val_ood_dict['rmse']


            cdw_errs = np.load('/home/eycai/datasets/nrp/cd-w.npz')
            cd_errs = np.load('/home/eycai/datasets/nrp/tax3dcd.npz')

            vem_cdw = cdw_errs['vem']
            voem_cdw = cdw_errs['voem']
            vem_cd = cd_errs['vem']
            voem_cd = cd_errs['voem']



            val_errors = np.random.rand(40)
            val_ood_errors = np.random.rand(40) * 4

            train_pc = train_batch["pc_anchor"]
            val_pc = val_batch["pc_anchor"]
            val_ood_pc = val_ood_batch["pc_anchor"]


            train_locs = torch.mean(train_pc, dim=1)
            val_locs = torch.mean(val_pc, dim=1)
            val_ood_locs = torch.mean(val_ood_pc, dim=1)
            # plotly go to scatter plot locs
            fig = go.Figure()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                row_heights=[0.5, 0.5],
                                subplot_titles=("CD-W", "TAX3D-CP (Ours)"))

            # ----------- PLOTTING X VS Y ----------------
            fig.add_trace(go.Scatter(
                x=train_locs[:, 0].cpu(),
                y=train_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='x-thin',
                marker=dict(
                    size=20,
                    color='rgb(38,133,249)',
                    line=dict(
                        width=4,
                        color='rgb(38,133,249)'
                    ),
                ),
                name='Train',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=val_locs[:, 0].cpu(),
                y=val_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='square',
                marker=dict(
                    size=20,
                    color=vem_cdw,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=val_ood_locs[:, 0].cpu(),
                y=val_ood_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='diamond',
                marker=dict(
                    size=20,
                    color=voem_cdw,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen (OOD)',
            ), row=1, col=1)


            # ----------- PLOTTING X VS Z ----------------
            fig.add_trace(go.Scatter(
                x=train_locs[:, 0].cpu(),
                y=train_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='x-thin',
                marker=dict(
                    size=20,
                    color='rgb(38,133,249)',
                    line=dict(
                        width=4,
                        color='rgb(38,133,249)'
                    ),
                ),
                name='Train',
                showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=val_locs[:, 0].cpu(),
                y=val_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='square',
                marker=dict(
                    size=20,
                    color=vem_cd,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen',
                showlegend=False,
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=val_ood_locs[:, 0].cpu(),
                y=val_ood_locs[:, 1].cpu(),
                mode='markers',
                marker_symbol='diamond',
                marker=dict(
                    size=20,
                    color=voem_cd,
                    coloraxis='coloraxis',
                    line=dict(
                        width=2,
                        color='Black'
                    ),
                ),
                name='Unseen (OOD)',
                showlegend=False,
            ), row=2, col=1)


            fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                'font': dict(
                    family="Arial",
                    size=52,
                    color="Black"
                ),
                })
            fig.update_annotations(font_size=72)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
            fig.update_yaxes(title_text="Y", row=1, col=1)
            fig.update_yaxes(title_text="Y", row=2, col=1)
            fig.update_xaxes(title_text="X", row=2, col=1)

            fig.update_layout(legend=dict(
                # yanchor="top",
                # y=0.65,
                xanchor="left",
                x=0.68,
                orientation="h",
            ))
            fig.update_layout(
                coloraxis_colorbar=dict(
                    title="RMSE",
                ),
                coloraxis=dict(
                    colorscale=['green', 'red'],
                    #cmin=0,
                    #cmax=4,
                )
            )

            fig.show()
            quit()
            visualize_batched_point_clouds([train_pc, val_pc, val_ood_pc]).show()

            pass

        if VISUALIZE_PREDS:
            model.to(device)
            dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=1, shuffle=False
            )
            iterator = iter(dataloader)
            for _ in range(32):
                batch = next(iterator)
            pred_dict = model.predict(batch, 50)
            # extracting anchor point cloud depending on model type
            if cfg.model.type == "flow":
                scene_pc = batch["pc"].flatten(0, 1).cpu().numpy()
                seg = batch["seg"].flatten(0, 1).cpu().numpy()
                anchor_pc = scene_pc[~seg.astype(bool)]
            else:
                anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()

            pred_action = pred_dict["pred_action"][[8]] # 0,8
            pred_action_size = pred_action.shape[1]
            pred_action = pred_action.flatten(0, 1).cpu().numpy()
            # color-coded segmentations
            anchor_seg = np.zeros(anchor_pc.shape[0], dtype=np.int64)
            # if cfg.model.type == "flow":
            #     pred_action_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
            # else:
            #     pred_action_size = cfg.dataset.sample_size_action
            pred_action_seg = np.array([np.arange(1, 2)] * pred_action_size).T.flatten()
            # visualize
            fig = vpl.segmentation_fig(
                np.concatenate((anchor_pc, pred_action)),
                np.concatenate((anchor_seg, pred_action_seg)),
            )
            fig.show()

        if VISUALIZE_PULL:
            model.to(device)
            dataloader = torch.utils.data.DataLoader(
                datamodule.val_dataset, batch_size=1, shuffle=False
            )
            iterator = iter(dataloader)
            for _ in range(11):
                batch = next(iterator)
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
                    # pred_dict['pred_action'].flatten(0, 1).cpu(),
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
                        [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                        [pred_step],
                        "red",
                    )
            fig = animation.animate()
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
                # pred_dict['pred_action'].flatten(0, 1).cpu(),
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
                    [action_pc],# if cfg.model.type == "flow_cross" else pcd],
                    [pred_step],
                    "red",
                )
        fig = animation.animate()
        fig.show()

if __name__ == "__main__":
    main()