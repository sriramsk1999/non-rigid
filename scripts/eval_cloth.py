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
from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
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

    # GET_TRAIN_METRICS = True
    # GET_VAL_METRICS = True
    VISUALIZE_PREDS = False
    VISUALIZE_PRED_IDS = [0, 1]
    PREDS_PER_SAMPLE = 10


    VISUALIZE_SINGLE = True
    VISUALIZE_SINGLE_IDX = 0

    MMD_METRICS = True
    PRECISION_METRICS = False



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
        for outputs_list, name in [
            (train_outputs, "train"),
            (val_outputs, "val"),
            (val_ood_outputs, "val_ood")
        ]:
            # Put everything on CPU, and flatten a list of dicts into one dict.
            out_cpu = [pytree.tree_map(lambda x: x.cpu(), o) for o in outputs_list]
            outputs = flatten_outputs(out_cpu)
            # plot histogram
            fig = px.histogram(outputs["rmse"], nbins=100, title=f"{name} MMD RMSE")
            # fig.show()
            # Compute the metrics.
            cos_sim = torch.mean(outputs["cos_sim"])
            rmse = torch.mean(outputs["rmse"])
            cos_sim_wta = torch.mean(outputs["cos_sim_wta"])
            rmse_wta = torch.mean(outputs["rmse_wta"])
            print(f"{name} cos sim: {cos_sim}, rmse: {rmse}")
            print(f"{name} cos sim wta: {cos_sim_wta}, rmse wta: {rmse_wta}")
        quit()
        # TODO: THIS SHOULD ALSO LOG HISTOGRAMS FROM BEFORE MEANS


    # TODO: for now, all action inputs are the same, so just grab the first one
    # actions inputs no longer the same..grab val, sample a bunch, and eval all of them...
    
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
        
        for i in VISUALIZE_PRED_IDS:
            pred_pcs = []
            gt_pcs = []
            # sampling predictions
            sample = val_dataset[i]
            pos = sample["pc"].to(device).unsqueeze(0)
            gt_flow = sample["flow"].to(device).unsqueeze(0)
            seg = sample["seg"].to(device).bool()
            
            # ground truth flow
            gt_pc = (pos + gt_flow).flatten(end_dim=-2).cpu().numpy()
            gt_seg = np.zeros(gt_pc.shape[0], dtype=np.int64)
            # predicted flow
            pred_pos, pred_flows = model.predict(pos, PREDS_PER_SAMPLE, False)
            pred_pcs = (pred_pos[..., seg, :] + pred_flows[..., seg, :]).flatten(end_dim=-2).cpu().numpy()
            # plot
            # color_seg = np.array([np.arange(PREDS_PER_SAMPLE + 1)] * cfg.dataset.sample_size).T.flatten()
            color_seg = np.array([np.arange(PREDS_PER_SAMPLE)] * seg.cpu().sum()).T.flatten()

            fig = vpl.segmentation_fig(
                np.concatenate((pos.cpu().squeeze(), gt_pc, pred_pcs)), 
                np.concatenate((np.zeros(pos.shape[-2], dtype=np.int64), 
                                gt_seg, color_seg)),
            )
            fig.show()
        # pred_pcs = []
        # gt_pcs = []
        # # visualizing predictions
        # for batch in tqdm(val_loader):
        #     pos = batch["pc"].to(device)
        #     gt_flow = batch["flow"].to(device)
        #     gt_pc = (pos + gt_flow).flatten(end_dim=-2).cpu().numpy()
        #     pos, pred_flows = model.predict(pos, 10, False)
        #     pred_pc = (pos + pred_flows).flatten(end_dim=-2).cpu().numpy()

        #     gt_pcs.append(gt_pc)
        #     pred_pcs.append(pred_pc)
        #     # TODO: clean this up; better variable names, remove predict function
        #     # remove all unnecssary variables
        
        # pred_pcs = np.concatenate(pred_pcs)
        # gt_pcs = np.concatenate(gt_pcs)
        # # pred_seg = np.ones(pred_pcs_t.shape[0], dtype=np.int64)
        # pred_seg = np.array([np.arange(2, 322)] * 213).T.flatten()

        # gt_seg = np.zeros(gt_pcs.shape[0], dtype=np.int64)
        # # anchor_seg = np.ones(anchor_pc.shape[0], dtype=np.int64)*1

        # action_pc = train_dataset[0]["pc"]
        # action_seg = np.ones(action_pc.shape[0], dtype=np.int64)*1

        # fig = vpl.segmentation_fig(
        #     # np.concatenate((pred_pcs, gt_pcs, anchor_pc, action_pc)), 
        #     # np.concatenate((pred_seg, gt_seg, anchor_seg, action_seg)),
        #     np.concatenate((pred_pcs, gt_pcs, action_pc)), 
        #     np.concatenate((pred_seg, gt_seg, action_seg)),
        # )
        # fig.show()


    # plot single diffusion chain
    # TODO: have model.predict return results so this can just call that instead
    # of having to create separate diffusion object
    
    if VISUALIZE_SINGLE:
        from non_rigid.utils.vis_utils import FlowNetAnimation
        animation = FlowNetAnimation()
        pos = val_dataset[VISUALIZE_SINGLE_IDX]["pc"].to(device).unsqueeze(0)
        seg = val_dataset[VISUALIZE_SINGLE_IDX]["seg"].to(device).bool()

        pos[:, ~seg, :] += torch.tensor([[[0.0, 0, 5.0]]]).to(device)

        pred_pos, pred_flow, results = model.predict(pos, 1, False, True)
        print(pred_pos.shape, pred_flow.shape, len(results), results[0].shape)

        # pos = pos.unsqueeze(0).cuda().transpose(-1, -2)
        # z = torch.randn(1, 213, 3).cuda().transpose(-1, -2)
        # model_kwargs = dict(pos=pos)
        # # denoise
        # pred_flow, results = diffusion.p_sample_loop(
        #     network, z.shape, z, clip_denoised=False,
        #     model_kwargs=model_kwargs, progress=True, device=device
        # )
        pred_flow = pred_flow.squeeze(0)
        pcd = pred_pos.squeeze().cpu()

        # combined_pcd = torch.cat([pcd, torch.as_tensor(anchor_pc)], dim=0)

        for noise_step in tqdm(results[0:]):
            pred_flow_step = noise_step.squeeze(0).cpu()
            animation.add_trace(
                pcd,
                [pcd],
                [pred_flow_step],
                "red",
            )

        fig = animation.animate()
        fig.show()


if __name__ == "__main__":
    main()