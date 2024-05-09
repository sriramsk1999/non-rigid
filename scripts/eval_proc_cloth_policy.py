import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionTrainingModule
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
def play(env, pred_flows, rot, trans):
    num_episodes = pred_flows.shape[0]
    num_successes = 0
    obs = env.reset(rigid_trans=trans, rigid_rot=rot)
    # input('Press Enter to start playing...')
    for epsd in tqdm(range(num_episodes)):
        # print('------------ Play episode ', epsd, '------------------')
        if epsd > 0:
            obs = env.reset(rigid_trans=trans, rigid_rot=rot)
        pred_flow = pred_flows[epsd]
        a1_act = pred_flow[14] * 0.2 / 6.36
        a2_act = pred_flow[0] * 0.2 / 6.36
        act = torch.cat([a1_act, a2_act], dim=0).cpu().numpy()

        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            next_obs, rwd, done, info = env.step(act)
            if done:
                success = env.check_success(debug=False)
                info = env.end_episode(rwd)
                if success:
                    num_successes += 1
                # print('step: ', step, 'Task Successful: ', success)
                break
            obs = next_obs
            step += 1

        # input('Episode ended; press Enter to continue...')
        # env.close()
    return num_successes


def model_predict(cfg, model, batch):
    pos = batch["pc"].to(f'cuda:{cfg.resources.gpus[0]}')
    bs = pos.shape[0]
    pos = (
        pos.transpose(-1, -2)
        .unsqueeze(1)
        .expand(-1, cfg.inference.num_trials, -1, -1)
        .reshape(bs * cfg.inference.num_trials, -1, cfg.inference.sample_size)
    )
    if cfg.model.type == "flow":
        model_kwargs = dict(pos=pos)
    elif cfg.model.type == "flow_cross":
        pc_anchor = batch["pc_anchor"].to(f'cuda:{cfg.resources.gpus[0]}')
        pc_anchor = (
            pc_anchor.transpose(-1, -2)
            .unsqueeze(1)
            .expand(-1, cfg.inference.num_trials, -1, -1)
            .reshape(bs * cfg.inference.num_trials, -1, cfg.inference.sample_size_anchor)
        )
        model_kwargs = dict(
            y=pc_anchor,
            x0=pos,
        )
    model_kwargs, pred_flow, results = model.predict(bs, model_kwargs, cfg.inference.num_trials, unflatten=False, progress=False)
    
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
    if cfg.dataset.type != "cloth":
        raise NotImplementedError('This script is only for cloth evaluations.')
    datamodule = ProcClothFlowDataModule(
        root=data_root,
        batch_size=1, # cfg.inference.batch_size,
        val_batch_size=1, # cfg.inference.val_batch_size,
        num_workers=cfg.resources.num_workers,
        dataset_cfg=cfg.dataset,
    )
    # We don't actually the the module, we just need to setup for val + val_ood datasets
    datamodule.setup(stage="val")


    # LOAD THE MODEL ###
    network = DiffusionFlowBase(
        in_channels=cfg.model.in_channels,
        learn_sigma=cfg.model.learn_sigma,
        model=cfg.model.dit_arch,
        model_cfg=cfg.model,
    )
    # get checkpoint file (for now, this does not log a run)
    # checkpoint_reference = cfg.checkpoint.reference
    checkpoint_reference = f"/home/eycai/Documents/non-rigid/scripts/logs/train_{cfg.dataset.name}_{cfg.model.name}/{cfg.date}/{cfg.time}/checkpoints/last.ckpt"
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
    # training module is for train and eval
    if cfg.dataset.type_args.scene:
        if cfg.model.type != "flow":
            raise NotImplementedError("Scene inputs cannot be used with cross-type models.")
        cfg.inference.sample_size = cfg.dataset.sample_size_action + cfg.dataset.sample_size_anchor
    else:
        cfg.inference.sample_size = cfg.dataset.sample_size_action
        cfg.inference.sample_size_anchor = cfg.dataset.sample_size_anchor
    # for now, this does not support ghost point prediction
    model = FlowPredictionTrainingModule(network, training_cfg=cfg.inference, model_cfg=cfg.model)
    model.eval()
    model = model.to(f'cuda:{cfg.resources.gpus[0]}')

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
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            # breakpoint()
            pred_flows = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flows, rot, trans)
        print('Total successes on training data: ', total_successes, '/', len(train_dataloader))

    if EVAL_VAL:
        print('Evaluating on validation data...')
        total_successes = 0
        for batch in tqdm(val_dataloader):
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            pred_flows = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flows, rot, trans)
        print('Total successes on validation data: ', total_successes, '/', len(val_dataloader))

    if EVAL_VAL_OOD:
        print('Evaluating on ood validation data...')
        total_successes = 0
        for batch in tqdm(val_ood_dataloader):
            rot = batch["rot"].squeeze().numpy()
            trans = batch["trans"].squeeze().numpy()
            pred_flows = model_predict(cfg, model, batch)
            total_successes += play(env, pred_flows, rot, trans)
        print('Total successes on ood validation data: ', total_successes, '/', len(val_ood_dataloader))


if __name__ == '__main__':
    main()