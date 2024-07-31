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

import rpad.visualize_3d.plots as vpl
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import Transform3d
from PIL import Image

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

def get_action(env, step, goal1, goal2):
    _, verts = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
    verts = np.array(verts)
    flow1 = goal1 - verts[env.deform_params['node_density'] - 1]
    flow2 = goal2 - verts[0]

    a1 = flow1 * 0.2 / 5
    scale = ( env.max_episode_len  - step ) / env.max_episode_len
    if np.linalg.norm(a1) < 0.3:
        a1 = a1 / np.linalg.norm(a1) * 0.3
    a2 = flow2 * 0.2 / 5
    if np.linalg.norm(a2) < 0.3:
        a2 = a2 / np.linalg.norm(a2) * 0.3
    act = np.concatenate([a1, a2], dtype=np.float32)
    return act

# def play(env, model, num_episodes, args):
def play(env, pred_flows, rot, trans, deform_params=None):
    num_episodes = pred_flows.shape[0]
    num_successes = 0
    env.max_episode_len = 400
    centroid_dist = 0.0

    if deform_params is None:
        deform_params = {
            'num_holes': 1,
            'node_density': 15,
            'w': 1.0,
            'h': 1.0,
            'holes': [{'x0': 5, 'x1': 8, 'y0': 5, 'y1': 7}]
        }
    obs = env.reset(rigid_trans=trans, rigid_rot=rot, deform_params=deform_params)
    # breakpoint()
    # input('Press Enter to start playing...')
    for epsd in tqdm(range(num_episodes)):
        # print('------------ Play episode ', epsd, '------------------')
        if epsd > 0:
            obs = env.reset(rigid_trans=trans, rigid_rot=rot, deform_params=deform_params)
        pred_flow = pred_flows[epsd].cpu().numpy()
        # a1_act = pred_flow[deform_params['node_density'] - 1]
        # a2_act = pred_flow[0]
        # a1_act = a1_act * 0.2 / 6.36
        # a2_act = a2_act * 0.2 / 6.36
        # act = np.concatenate([a1_act, a2_act], dtype=np.float32)

        # get the ground truth mesh
        _, verts = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = np.array(verts)
        goal1 = verts[deform_params['node_density'] - 1] + pred_flow[deform_params['node_density'] - 1]
        goal2 = verts[0] + pred_flow[0]

        goal = np.concatenate([goal1, goal2], dtype=np.float32)

        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            # act = get_action(env, step, goal1, goal2)
            # print('action: ', act)
            # next_obs, rwd, done, info = env.step(act)
            next_obs, rwd, done, info = env.step(goal, unscaled=True)
            if done:
                centroid_check, centroid_dists = env.check_centroid()
                centroid_dist += np.min(centroid_dists)
                # success = env.check_success(debug=False)
                # input('Press Enter to start playing...')
                info = env.end_episode(rwd)
                polygon_check = env.check_polygon()
                success = np.any(centroid_check * polygon_check)
                # print(success, centroid_check, polygon_check, centroid_dist)
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
    return num_successes, np.mean(centroid_dist)


def model_predict(cfg, model, batch):
    # cfg.inference.num_trials = 10
    pred_dict = model.predict(batch, cfg.inference.num_trials, progress=False)
    pred_flow = pred_dict["pred_world_flow"]

    # computing final predicted goal pose
    if cfg.model.type == "flow":
        seg = batch["seg"].to(f'cuda:{cfg.resources.gpus[0]}')
        pred_flow = pred_flow[:, seg.squeeze(0) == 1, :]

    # pc_anchor = batch["pc_anchor"].cpu().squeeze().numpy()
    # pc = batch["pc"].squeeze().cpu().numpy()
    # pred_action = pred_dict["pred_action"][[1]].cpu().squeeze().numpy()
    
    # pred_action_size = pred_action.shape[1]
    # pred_action = pred_action.reshape(-1, 3)
    # pred_action_seg = np.array([np.arange(2, 12)] * pred_action_size).T.flatten()
    # vpl.segmentation_fig(
    #     np.concatenate([pc_anchor, 
    #                     pc, 
    #                     pred_action], axis=0),
    #     np.concatenate([np.ones(pc_anchor.shape[0]) * 0, 
    #                     np.ones(pc.shape[0]) * 5,
    #                     np.ones(pred_action.shape[0]) * 1,
    #                     # pred_action_seg,
    #                     ], axis=0).astype(np.int16),
    # ).show()
    # quit()
    return pred_flow


def eval_dataset(cfg, env, dataloader, model):
    total_successes = 0
    centroid_dists = []
    for batch in tqdm(dataloader):
        #if i % 4 != 0:
        # if i < 17:
        #     continue
        rot = batch["rot"].squeeze().numpy()
        trans = batch["trans"].squeeze().numpy()
        if "deform_params" in batch:
            deform_params = batch["deform_params"][0]
        else:
            deform_params = None


        # import rpad.visualize_3d.plots as vpl
        # from pytorch3d.transforms import Transform3d

        # action = batch["pc_action"]
        # anchor = batch["pc_anchor"]
        # T_goal2world = Transform3d(matrix=batch["T_goal2world"])
        # T_action2world = Transform3d(matrix=batch["T_action2world"])

        # action = T_action2world.transform_points(action).squeeze().cpu().numpy()
        # anchor = T_goal2world.transform_points(anchor).squeeze().cpu().numpy()

        # breakpoint()

        pred_flow = model_predict(cfg, model, batch)
        successes, centroid_dist = play(env, pred_flow, rot, trans, deform_params)
        total_successes += successes
        centroid_dists.append(centroid_dist)
    return total_successes, np.mean(centroid_dists)

@torch.no_grad()
@hydra.main(config_path='../configs', config_name='eval_policy', version_base="1.3")
def main(cfg):
    ### CREATING THE ENVIRONMENT ###
    args = get_args()
    args.env = cfg.sim.env
    args.viz = cfg.sim.viz

    # args.tax3d = True
    # args.pcd = True
    # args.logdir = 'rendered'
    # args.cam_config_path = '/home/eycai/Documents/dedo/dedo/utils/cam_configs/camview_0.json'


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


    # if cfg.dataset.type not in ["cloth", "cloth_point"]:
    if cfg.dataset.name not in ["proc_cloth"]:
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
    datamodule.setup(stage="predict")


    # LOAD THE MODEL ###
    network = DiffusionFlowBase(
        # in_channels=cfg.model.in_channels,
        # learn_sigma=cfg.model.learn_sigma,
        # model=cfg.model.dit_arch,
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
    ckpt = torch.load(ckpt_file, map_location="cuda:0")
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
    
    # # override the task type here based on the dataset
    # if "cloth" in cfg.dataset.type:
    #     cfg.task_type = "cloth"
    # elif "rigid" in cfg.dataset.type:
    #     cfg.task_type = "rigid"
    # else:
    #     raise ValueError(f"Unsupported dataset type: {cfg.dataset.type}")


    if cfg.model.type == "flow":
        model = FlowPredictionInferenceModule(
            network, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    elif cfg.model.type == "point":
        model = PointPredictionInferenceModule(
            network, task_type=cfg.task_type, inference_cfg=cfg.inference, model_cfg=cfg.model
        )
    model.eval()
    model.to(f'cuda:{cfg.resources.gpus[0]}')

    ### RUNNING EVALS ON EACH DATASET ###
    train_dataloader = datamodule.train_dataloader()
    val_dataloader, val_ood_dataloader = datamodule.val_dataloader()


    if cfg.eval.train:
        print('Evaluating on training data...')
        total_successes, centroid_dist = eval_dataset(cfg, env, train_dataloader, model)
        print('Total successes on training data: ', total_successes, '/', len(train_dataloader))
        print('Mean centroid distance: ', centroid_dist)

    if cfg.eval.val:
        print('Evaluating on validation data...')
        total_successes, centroid_dist = eval_dataset(cfg, env, val_dataloader, model)
        print('Total successes on validation data: ', total_successes, '/', len(val_dataloader))
        print('Mean centroid distance: ', centroid_dist)

    if cfg.eval.val_ood:
        print('Evaluating on ood validation data...')
        total_successes, centroid_dist = eval_dataset(cfg, env, val_ood_dataloader, model)
        print('Total successes on ood validation data: ', total_successes, '/', len(val_ood_dataloader))
        print('Mean centroid distance: ', centroid_dist)

if __name__ == '__main__':
    main()