import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from pathlib import Path
import os

from non_rigid.datasets.microwave_flow import MicrowaveFlowDataset, MicrowaveFlowDataModule
from non_rigid.models.df_base import DiffusionFlowBase, DiffusionFlowTrainingModule
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    match_fn,
)

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

    from non_rigid.metrics.flow_metrics import flow_cos_sim, flow_rmse
    from non_rigid.models.dit.diffusion import create_diffusion
    from tqdm import tqdm
    import numpy as np

    device = "cuda"
    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    data_root = data_root / f"{cfg.dataset.obj_id}_flow_{cfg.dataset.type}"
    dataset = MicrowaveFlowDataset(data_root, "train")
    n_train_demos = 10
    n_train_modes = 2
    diffusion = create_diffusion(timestep_respacing=None, diffusion_steps=cfg.model.diff_train_steps)

    best_modes = []
    best_flow_errors = []
    best_cos_sims = []
    init_pos = []
    pred_flows = []

    for i in tqdm(range(n_train_demos)):
        # get initial pos observation
        train_sample = dataset[i*n_train_modes][0]
        pos = train_sample[..., :3].unsqueeze(0)
        pos = torch.transpose(pos, -1, -2).cuda()
        model_kwargs = dict(pos=pos)
        for j in tqdm(range(10)):
            # sample random latent
            z = torch.randn(1, 1867, 3).cuda()
            z = torch.transpose(z, -1, -2)
            # denoise
            pred_flow, results = diffusion.p_sample_loop(
                network, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
            pred_flow = pred_flow.permute(0, 2, 1)
            pred_flows.append(pred_flow)
            init_pos.append(pos)

            cos_sims = []
            flow_errs = []
            for k in range(n_train_modes):
                gt_flow_k = dataset[i*n_train_modes+k][0][..., 3:6]
                seg_k = torch.as_tensor(dataset[i*n_train_modes+k][0][..., 6], dtype=torch.bool).unsqueeze(0).cuda()
                gt_flow_k = gt_flow_k.unsqueeze(0).cuda()
                # computing cosine similarity
                cos_sim = flow_cos_sim(pred_flow, gt_flow_k, True, seg_k).mean()
                flow_err = flow_rmse(pred_flow, gt_flow_k, True, seg_k).mean()
                cos_sims.append(cos_sim.cpu().numpy())
                flow_errs.append(flow_err.cpu().numpy())
            
            # getting best mode
            best_mode = np.argmin(flow_errs)
            best_modes.append(best_mode)
            best_flow_errors.append(flow_errs[best_mode])
            best_cos_sims.append(cos_sims[best_mode])

    best_modes = np.array(best_modes)
    best_flow_errors = np.array(best_flow_errors)
    best_cos_sims = np.array(best_cos_sims)
    print((best_modes == 0).sum())
    print((best_modes == 1).sum())

    from matplotlib import pyplot as plt

    # Plotting error distributions
    flow_range = (best_flow_errors.min(), best_flow_errors.max())
    plt.hist(best_flow_errors[best_modes == 0], bins=100, color="blue", range=flow_range)
    plt.hist(best_flow_errors[best_modes == 1], bins=100, color="red", range=flow_range)
    # set x label
    plt.xlabel('Flow Error')
    plt.title('Flow Error Histogram')
    # plt.savefig(f'plots/{config.model_name}/flow_error_histogram.png')
    plt.show()
    cos_range = (best_cos_sims.min(), best_cos_sims.max())
    plt.hist(best_cos_sims[best_modes == 0], bins=100, color="blue", range=cos_range)
    plt.hist(best_cos_sims[best_modes == 1], bins=100, color="red", range=cos_range)
    plt.xlabel('Cosine Similarity')
    plt.title('Cosine Similarity Histogram')
    # plt.savefig(f'plots/{config.model_name}/cosine_similarity.png')
    plt.show()

    # plotting flow predictions
    pcs = [flow.squeeze() + pos.squeeze().transpose(1, 0) for flow, pos in zip(pred_flows, init_pos)]
    pcs = torch.concat(pcs, dim=0).cpu()
    seg = torch.ones((pcs.shape[0],), dtype=torch.int64)

    sample30 = dataset[0][0]
    pc30 = sample30[..., :3] + sample30[..., 3:6]
    sample60 = dataset[1][0]
    pc60 = sample60[..., :3] + sample60[..., 3:6]

    gt_pc = torch.cat([pc30, pc60], dim=0).cpu()
    gt_seg = torch.zeros((gt_pc.shape[0],), dtype=torch.int64)

    import rpad.visualize_3d.plots as vpl
    fig = vpl.segmentation_fig(
        torch.concat([pcs, gt_pc], dim=0),
        torch.cat([seg, gt_seg], dim=0),
        )
    fig.show()


    from non_rigid.utils.vis_utils import FlowNetAnimation
    animation = FlowNetAnimation()

    sample = dataset[1][0]
    pos = sample[..., :3].unsqueeze(0).cuda()
    pos = torch.transpose(pos, -1, -2)
    z = torch.randn(1, 1867, 3).cuda()
    z = torch.transpose(z, -1, -2)
    gt_flow = sample[..., 3:6]
    seg = torch.as_tensor(sample[..., 6], dtype=torch.bool)


    model_kwargs = dict(pos=pos)
    # denoise
    pred_flow, results = diffusion.p_sample_loop(
        network, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device
    )
    pred_flow = pred_flow.squeeze(0).permute(1, 0)
    # compute errors
    #cos_sim = flow_cosine_similarity(pred_flow, gt_flow, True, seg).mean()
    #flow_err = flow_correspondence(pred_flow, gt_flow, True, seg).mean()
    #print('Cosine Similarity: ', cos_sim, ' Flow Error: ', flow_err)

    pcd = pos.squeeze().permute(1, 0).cpu().numpy()


    for noise_step in tqdm(results[0:]):
        pred_flow_step = noise_step.squeeze(0).permute(1, 0).unsqueeze(1)
        animation.add_trace(
            torch.as_tensor(pcd),
            torch.as_tensor([pcd]),
            torch.as_tensor([pred_flow_step.squeeze().cpu().numpy()]),
            "red",
        )

    fig = animation.animate()
    fig.show()

if __name__ == "__main__":
    main()