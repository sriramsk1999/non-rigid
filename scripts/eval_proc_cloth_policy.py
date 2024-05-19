import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataModule

import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib import interactive

# interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args, args_postprocess
from dedo.utils.pcd_utils import visualize_data, render_video

import pybullet as p
from tqdm import tqdm


def policy_simple(obs, act, task, step, speed_factor=1.0):
    """A very simple default policy."""
    act = act.reshape(2, 3)
    obs = obs.reshape(-1, 3)
    if task == 'Button':
        act[:, :] = 0.0
        if obs[0, 0] < 0.10:
            act[:, 0] = 0.10  # increase x
    elif task in ['HangGarment', 'HangProcCloth']:
        act[:, 1] = -0.2
        act[:, 2] += 0.01
    elif task in ['HangBag']:
        # Dragging T Shirt
        act[:, 1] = -0.5
        act[:, 2] = 0.6
    elif task in ['Dress']:
        act[:, 1] = -0.2
        act[:, 2] = 0.1
    elif task in ['Lasso', 'Hoop']:
        if obs[0, 1] > 0.0:
            act[:, 1] = -0.25  # decrease y
            act[:, 2] = -0.25  # decrease z
    elif obs[0, 2] > 0.50:
        act[:, 1] = -0.30  # decrease y
        act[:, 2] = 0.1  # decrease z
    return act.reshape(-1) * speed_factor

def get_action(model, obs):
    return

# def play(env, model, num_episodes, args):
def play(env, pred_flows, rot, trans, deform_params=None):
    num_episodes = pred_flows.shape[0]
    num_successes = 0

    if deform_params is None:
        deform_params = {
            'num_holes': 1,
            'node_density': 15,
            'w': 1.0,
            'h': 1.0,
            'holes': [{'x0': 5, 'x1': 8, 'y0': 5, 'y1': 7}]
        }
    obs = env.reset(rigid_trans=trans, rigid_rot=rot, deform_params=deform_params)
    # input('Press Enter to start playing...')
    for epsd in tqdm(range(num_episodes)):
        # print('------------ Play episode ', epsd, '------------------')
        if epsd > 0:
            obs = env.reset(rigid_trans=trans, rigid_rot=rot, deform_params=deform_params)
        pred_flow = pred_flows[epsd]
        a1_act = pred_flow[deform_params['node_density'] - 1] * 0.2 / 6.36
        a2_act = pred_flow[0] * 0.2 / 6.36
        act = torch.cat([a1_act, a2_act], dim=0).cpu().numpy()

        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            next_obs, rwd, done, info = env.step(act)
            if done:
                centroid_check = env.check_centroid()
                # success = env.check_success(debug=False)
                info = env.end_episode(rwd)
                polygon_check = env.check_polygon()
                success = np.any(centroid_check * polygon_check)
                #success = env.check_success2(debug=False)
                if success:
                    num_successes += 1
                # input('Episode ended; press Enter to continue...')
                # print('step: ', step, 'Task Successful: ', success)
                break
            obs = next_obs
            step += 1
        
        

        # input('Episode ended; press Enter to continue...')
        # env.close()
    return num_successes


def model_predict(cfg, model, batch):
    pred_dict = model.predict(batch, cfg.inference.num_trials)
    pred_flow = pred_dict["pred_world_flow"]

    # computing final predicted goal pose
    if cfg.model.type == "flow":
        seg = batch["seg"].to(f'cuda:{cfg.resources.gpus[0]}')
        pred_flow = pred_flow[:, seg.squeeze(0) == 1, :]
    return pred_flow


@torch.no_grad()
@hydra.main(config_path='../configs', config_name='eval_policy', version_base="1.3")
def main(cfg):
    ### CREATING THE ENVIRONMENT ###
    args = get_args()
    args.env = cfg.sim.env
    args.viz = cfg.sim.viz
    args_postprocess(args)
    assert ('Robot' not in args.env), 'This is a simple demo for anchors only'
    np.set_printoptions(precision=4, linewidth=150, suppress=True)
    kwargs = {'args': args}

    env = gym.make(args.env, **kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)

    ### LOAD THE DATASET ###
    # Make deterministic + reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Since most of us are training on 3090s+, we can use mixed precision.
    torch.set_float32_matmul_precision("medium")
    # Global seed for reproducibility.
    L.seed_everything(42)
    data_root = Path(os.path.expanduser(cfg.dataset.data_dir))
    if cfg.dataset.type not in ["cloth", "cloth_point"]:
        raise NotImplementedError('This script is only for cloth evaluations.')
    
    # TODO: if inference is full action, set sample size to -1
    if cfg.inference.action_full:
        cfg.dataset.sample_size_action = -1
    else:
        raise NotImplementedError('This script is only for full action inference.')

    datamodule = ProcClothFlowDataModule(
        root=data_root,
        batch_size=1, # cfg.inference.batch_size,
        val_batch_size=1, # cfg.inference.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    # We don't actually need the module, we just need to setup for val + val_ood datasets
    datamodule.setup(stage="val")


    # LOAD THE MODEL ###
    network = DiffusionFlowBase(
        in_channels=cfg.model.in_channels,
        learn_sigma=cfg.model.learn_sigma,
        model=cfg.model.dit_arch,
        model_cfg=cfg.model,
    )
    # get checkpoint file (for now, this does not log a run)
    checkpoint_reference = cfg.checkpoint.reference
    # TODO: load a proper check point here
    # checkpoint_reference = f"/home/eycai/Documents/non-rigid/scripts/logs/train_{cfg.dataset.name}_{cfg.model.name}/{cfg.date}/{cfg.time}/checkpoints/last.ckpt"
    if checkpoint_reference.startswith(cfg.wandb.entity):
    # if True:
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


    if cfg.model.type in ["flow", "flow_cross"]:
        model = FlowPredictionInferenceModule(network, inference_cfg=cfg.inference, model_cfg=cfg.model)
    elif cfg.model.type in ["point_cross"]:
        model = PointPredictionInferenceModule(
            network, task_type=cfg.task_type, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    model.eval()
    model.to(f'cuda:{cfg.resources.gpus[0]}')

    ### RUNNING EVALS ON EACH DATASET ###
    train_dataloader = datamodule.train_dataloader()
    val_dataloader, val_ood_dataloader = datamodule.val_dataloader()

    EVAL_TRAIN = True
    EVAL_VAL = True
    EVAL_VAL_OOD = True

    if EVAL_TRAIN:
        print('Evaluating on training data...')
        total_successes = 0
        for batch in tqdm(train_dataloader):
            # batch = {k: v.to(f'cuda:{cfg.resources.gpus[0]}') for k, v in batch.items()}
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            if "deform_params" in batch:
                deform_params = batch["deform_params"][0]
            else:
                deform_params = None

            pred_flow = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flow, rot, trans, deform_params)
        print('Total successes on training data: ', total_successes, '/', len(train_dataloader))

    if EVAL_VAL:
        print('Evaluating on validation data...')
        total_successes = 0
        for batch in tqdm(val_dataloader):
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            if "deform_params" in batch:
                deform_params = batch["deform_params"][0]
            else:
                deform_params = None

            pred_flows = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flows, rot, trans)
        print('Total successes on validation data: ', total_successes, '/', len(val_dataloader))

    if EVAL_VAL_OOD:
        print('Evaluating on ood validation data...')
        total_successes = 0
        for batch in tqdm(val_ood_dataloader):
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            if "deform_params" in batch:
                deform_params = batch["deform_params"][0]
            else:
                deform_params = None

            pred_flows = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flows, rot, trans)
        print('Total successes on ood validation data: ', total_successes, '/', len(val_ood_dataloader))


if __name__ == '__main__':
    main()