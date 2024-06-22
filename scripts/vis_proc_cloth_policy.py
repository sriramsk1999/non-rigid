import hydra
import lightning as L
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataModule

import matplotlib.pyplot as plt
from matplotlib import rcParams
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

CAM_PROJECTION = (1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, -1.0000200271606445, -1.0,
                    0.0, 0.0, -0.02000020071864128, 0.0)
CAM_VIEW = (0.9396926164627075, 0.14454397559165955, -0.3099755346775055, 0.0, 
            -0.342020183801651, 0.3971312642097473, -0.8516507148742676, 0.0, 
            7.450580596923828e-09, 0.9063077569007874, 0.4226182699203491, 0.0, 
            0.5810889005661011, -4.983892917633057, -22.852874755859375, 1.0)

CAM_WIDTH = 500
CAM_HEIGHT = 500

LOG_DIR = '/home/eycai/Documents/non_rigid_logs/'
LOG = True

def camera_project(point,
                   projectionMatrix=CAM_PROJECTION,
                   viewMatrix=CAM_VIEW,
                   height=CAM_WIDTH,
                   width=CAM_HEIGHT):
    """
    Projects a world point in homogeneous coordinates to pixel coordinates
    Args
        point: np.array of shape (N, 3); indicates the desired point to project
    Output
        (x, y): np.array of shape (N, 2); pixel coordinates of the projected point
    """
    point = np.concatenate([point, np.ones_like(point[:, :1])], axis=-1) # N x 4

    # reshape to get homogeneus transform
    persp_m = np.array(projectionMatrix).reshape((4, 4)).T
    view_m = np.array(viewMatrix).reshape((4, 4)).T

    # Perspective proj matrix
    world_pix_tran = persp_m @ view_m @ point.T
    world_pix_tran = world_pix_tran / world_pix_tran[-1]  # divide by w
    world_pix_tran[:3] = (world_pix_tran[:3] + 1) / 2
    x, y = world_pix_tran[0] * width, (1 - world_pix_tran[1]) * height
    x, y = np.floor(x).astype(int), np.floor(y).astype(int)

    return np.stack([x, y], axis=1)


def plot_diffusion(img, pred, color_key):
    # https://stackoverflow.com/questions/34768717/matplotlib-unable-to-save-image-in-same-resolution-as-original-image
    
    pred_cam = camera_project(pred)

    dpi = rcParams['figure.dpi']
    img = np.array(img)
    height, width, _ = img.shape
    figsize = width / dpi, height / dpi

    # creating figure of exact image size
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    ax.imshow(img)
    # breakpoint()
    # quit()

    # clipping points that are outside of camera frame
    filter = (pred_cam[:, 0] >= 0) & (pred_cam[:, 0] < CAM_WIDTH) & (pred_cam[:, 1] >= 0) & (pred_cam[:, 1] < CAM_HEIGHT)
    pred_cam = pred_cam[filter]
    color_key = color_key[filter]

    ax.scatter(pred_cam[:, 0], pred_cam[:, 1], 
                cmap='inferno', c=color_key, s=5, 
                marker='o')
    fig.canvas.draw()
    frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    # plt.imshow(img)
    # plt.scatter(pred_cam[:, 0], pred_cam[:, 1], 
    #             cmap='inferno', c=color_key, s=5, 
    #             marker='o')#, edgecolors='none')

    # fig = plt.gcf()
    # plt.axis('off')
    # fig.tight_layout(pad=0)
    # # plt.savefig(log_dir + 'frame2.png')
    # fig.canvas.draw()
    # frame = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return frame




# def play(env, model, num_episodes, args):
def play(env, log_dir, pred_flows, results, rot, trans, deform_params=None):
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
        vid_frames = []
        diffusion_frames = []

        if epsd > 0:
            obs = env.reset(rigid_trans=trans, rigid_rot=rot, deform_params=deform_params)
        
        if LOG:
            vid_frames.append(Image.fromarray(env.render(mode='rgb_array', width=CAM_WIDTH, height=CAM_HEIGHT)))

        pred_flow = pred_flows[epsd].cpu().numpy()
        results = [r[epsd].cpu().numpy() for r in results]

        

        # get the ground truth mesh
        _, verts = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
        verts = np.array(verts)
        goal1 = verts[deform_params['node_density'] - 1] + pred_flow[deform_params['node_density'] - 1]
        goal2 = verts[0] + pred_flow[0]

        goal = np.concatenate([goal1, goal2], dtype=np.float32)

        # plot_diffusion(vid_frames[0], results[-1], log_dir)

        if LOG:
            # this is hacky, get color map from initial verts
            color_key = verts[:, 0] + verts[:, 2]
            for r in tqdm(results):
                diffusion_frames.append(plot_diffusion(vid_frames[0], r, color_key))

            diffusion_frames[0].save(log_dir + 'diffusion.gif', save_all=True,
                                    append_images=diffusion_frames[1:], duration=33, loop=0)

        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            next_obs, rwd, done, info = env.step(goal, unscaled=True)
            if LOG:
                vid_frames.append(Image.fromarray(env.render(mode='rgb_array', width=CAM_WIDTH, height=CAM_HEIGHT)))
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
        
        if LOG:
            # save video
            vid_frames[0].save(log_dir + 'video.gif', save_all=True, 
                            append_images=vid_frames[1:], duration=33, loop=0)
            # save just last frame
            vid_frames[-1].save(log_dir + 'last_frame.png')
    quit()
    return num_successes, np.mean(centroid_dist)


def model_predict(cfg, model, batch):
    # cfg.inference.num_trials = 10
    pred_dict = model.predict(batch, cfg.inference.num_trials, progress=False)
    pred_flow = pred_dict["pred_world_flow"]
    results = pred_dict["results_world"]

    # computing final predicted goal pose
    if cfg.model.type == "flow":
        seg = batch["seg"].to(f'cuda:{cfg.resources.gpus[0]}')
        pred_flow = pred_flow[:, seg.squeeze(0) == 1, :]
        # TODO: ALSO SEG THE RESULTS
        results = [r[:, seg.squeeze(0) == 1, :] for r in results]

    # plotting point clouds
    gt_action = batch["pc"].flatten(0, 1).cpu().numpy()
    if cfg.model.type == "flow":
        scene_pc = batch["pc_action"].flatten(0, 1).cpu().numpy()
        seg = batch["seg"].flatten(0, 1).cpu().numpy()
        anchor_pc = scene_pc[~seg.astype(bool)]
        gt_action = gt_action[seg.astype(bool)]
    else:
        anchor_pc = batch["pc_anchor"].flatten(0, 1).cpu().numpy()
    pred_action = pred_dict["pred_action"].flatten(0, 1).cpu().numpy()
    

    fig = vpl.segmentation_fig(
        np.concatenate([
            anchor_pc,
            # gt_action,
            pred_action,
        ], axis=0),
        np.concatenate([
            np.ones(anchor_pc.shape[0]) * 0,
            # np.ones(gt_action.shape[0]) * 5,
            np.ones(pred_action.shape[0]) * 1,
        ], axis=0).astype(np.int16),
    )
    fig.show()


    return pred_flow, results


def eval_dataset(cfg, env, dataloader, model, log_dir):
    index = 21
    log_dir = log_dir + f'{index}/'

    os.makedirs(log_dir, exist_ok=True)
    total_successes = 0
    centroid_dists = []
    for i, batch in enumerate(tqdm(dataloader)):
        #if i % 4 != 0:
        if i < index:
            continue
        rot = batch["rot"].squeeze().numpy()
        trans = batch["trans"].squeeze().numpy()
        if "deform_params" in batch:
            deform_params = batch["deform_params"][0]
        else:
            deform_params = None

        pred_flow, results = model_predict(cfg, model, batch)
        successes, centroid_dist = play(env, log_dir, pred_flow, results, rot, trans, deform_params)
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
    datamodule.setup(stage="predict")


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

    
    if cfg.model.type == "point_cross":
        if cfg.dataset.world_frame:
            model_type = "cross point world"
        else:
            model_type = "cross point"
    elif cfg.model.type == "flow_cross":
        if cfg.dataset.world_frame:
            model_type = "cross flow world"
        else:
            if cfg.model.x0_encoder == "mlp":
                model_type = "cross flow"
            else:
                model_type = "cross flow nac"
    elif cfg.model.type == "flow":
        model_type = "scene_flow"
    else:
        raise NotImplementedError(f"Unsupported model type: {cfg.model.type}")

    if cfg.eval.train:
        print('Evaluating on training data...')
        log_dir = LOG_DIR + f'{cfg.dataset.multi_cloth.hole}_{cfg.dataset.multi_cloth.size}/train/{model_type}/'
        total_successes, centroid_dist = eval_dataset(cfg, env, train_dataloader, model, log_dir)
        print('Total successes on training data: ', total_successes, '/', len(train_dataloader))
        print('Mean centroid distance: ', centroid_dist)

    if cfg.eval.val:
        print('Evaluating on validation data...')
        log_dir = LOG_DIR + f'{cfg.dataset.multi_cloth.hole}_{cfg.dataset.multi_cloth.size}/val/{model_type}/'
        total_successes, centroid_dist = eval_dataset(cfg, env, val_dataloader, model, log_dir)
        print('Total successes on validation data: ', total_successes, '/', len(val_dataloader))
        print('Mean centroid distance: ', centroid_dist)

    if cfg.eval.val_ood:
        print('Evaluating on ood validation data...')
        log_dir = LOG_DIR + f'{cfg.dataset.multi_cloth.hole}_{cfg.dataset.multi_cloth.size}/val_ood/{model_type}/'
        total_successes, centroid_dist = eval_dataset(cfg, env, val_ood_dataloader, model, log_dir)
        print('Total successes on ood validation data: ', total_successes, '/', len(val_ood_dataloader))
        print('Mean centroid distance: ', centroid_dist)

if __name__ == '__main__':
    main()