import hydra
import lightning as L
import json
import omegaconf
import torch
import torch.utils._pytree as pytree
import wandb

from functools import partial
from pathlib import Path
import os
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from non_rigid.datasets.proc_cloth_flow import ProcClothFlowDataset, ProcClothFlowDataModule
from non_rigid.models.df_base import (
    DiffusionFlowBase, 
    FlowPredictionInferenceModule, 
    PointPredictionInferenceModule
)
from non_rigid.models.regression import (
    LinearRegression,
    LinearRegressionInferenceModule
)
from non_rigid.utils.vis_utils import FlowNetAnimation
from non_rigid.utils.script_utils import (
    PROJECT_ROOT,
    LogPredictionSamplesCallback,
    create_model,
    match_fn,
    flatten_outputs
)
from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import expand_pcd, downsample_pcd
from non_rigid.utils.augmentation_utils import ball_occlusion, plane_occlusion

from pytorch3d.transforms import Transform3d, Translate
from scipy.spatial.transform import Rotation as R
import rpad.visualize_3d.plots as vpl

def rigid_transform_3D(A, B):
    """
    https://nghiaho.com/?page_id=671
    """
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t, centroid_B - centroid_A

@torch.no_grad()
@hydra.main(config_path="../configs", config_name="real_world", version_base="1.3")
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
    L.seed_everything(420)

    ######################################################################
    # Load model.
    ######################################################################
    device = f"cuda:{cfg.resources.gpus[0]}"
    network, model = create_model(cfg)
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

    ######################################################################
    # Load the scene observation.
    ######################################################################
    data = np.load(os.path.expanduser(cfg.data_path), allow_pickle=True)


    # num_samples = 10
    # for i in range(num_samples):
    #     # downsample
    #     # batchify
    #     # model
    #     # store all predictions

    # # plot all predictions

    # # pick one, and compute the transformation

    num_samples = 10
    action_pcs = []
    pred_pcs = []
    for i in range(num_samples):
        points_action = torch.from_numpy(data['action_pc']).float()
        points_anchor = torch.from_numpy(data['anchor_pc']).float()
        seg_action = torch.from_numpy(data['action_seg']).int()
        seg_anchor = torch.from_numpy(data['anchor_seg']).int()

        # points_anchor = points_anchor[points_anchor[:, 2] > 0.05]
        

        # downsample
        points_action, action_indices = downsample_pcd(points_action.unsqueeze(0), 512)
        points_anchor, anchor_indices = downsample_pcd(points_anchor.unsqueeze(0), 512)
        seg_action = seg_action[action_indices.squeeze(0)]
        seg_anchor = seg_anchor[anchor_indices.squeeze(0)]
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)
        # store action point cloud in word frame, before mean-centering
        action_pcs.append(points_action.clone())


        # mean-centering
        action_center = points_action.mean(axis=0)
        anchor_center = points_anchor.mean(axis=0)
        points_action -= action_center
        points_anchor -= anchor_center


        # random rotation?
        T1 = random_se3(
            N=1,
            rot_var=180,
            trans_var=0.0,
            rot_sample_method='random_flat_upright',
        )
        points_anchor = T1.transform_points(points_anchor.unsqueeze(0)).squeeze(0)

        # occlusions
        # points_anchor = plane_occlusion(points_anchor)
        # points_anchor = ball_occlusion(points_anchor)


        # saving mean-centering transforms
        T_action2world = Translate(action_center.unsqueeze(0))
        # T_goal2world = Translate(anchor_center.unsqueeze(0))
        T_goal2world = T1.inverse().compose(
            Translate(anchor_center.unsqueeze(0))
        )

        batch = {
            'pc_action': points_action.unsqueeze(0),
            'pc_anchor': points_anchor.unsqueeze(0),
            # 'seg': torch.from_numpy(data['action_seg']).unsqueeze(0),
            # 'seg_anchor': torch.from_numpy(data['anchor_seg']).unsqueeze(0),
            'seg': seg_action.unsqueeze(0),
            'seg_anchor': seg_anchor.unsqueeze(0),
            'T_action2world': T_action2world.get_matrix(),
            'T_goal2world': T_goal2world.get_matrix(),
        }
        pred_dict = model.predict(batch, 1, progress=True)
        pred_world_action = pred_dict['pred_world_action']

        pred_pcs.append(pred_world_action)

    # visualize all predictions
    pred_pc_viz = torch.cat(pred_pcs, dim=0).flatten(0, 1).cpu().numpy()
    action_pcd = data['action_pc']
    anchor_pcd = data['anchor_pc']
    # color coded segmentations
    anchor_seg = np.zeros(anchor_pcd.shape[0])
    action_seg = np.ones(action_pcd.shape[0])
    pred_pc_seg = np.array([np.arange(2, num_samples)] * 512).T.flatten()
    fig = vpl.segmentation_fig(
        np.concatenate([anchor_pcd, action_pcd, pred_pc_viz], axis=0),
        np.concatenate([anchor_seg, action_seg, pred_pc_seg], axis=0).astype(int),
    )
    fig.show()


    # compute transformations
    for i in range(num_samples):
        action_pc_i = action_pcs[i].cpu().numpy()
        pred_pc_i = pred_pcs[i].squeeze(0).cpu().numpy()

        R_i, t_i, t_world_i = rigid_transform_3D(action_pc_i.T, pred_pc_i.T)

        # visualize optimized transformation
        R_i = R.from_matrix(R_i)
        action_pc_transformed = R_i.apply(action_pc_i) + t_i.T
        # action_pc_transformed = action_pc_i + t_world_i.T + np.array([0.02, 0, 0])
        fig = vpl.segmentation_fig(
            np.concatenate([anchor_pcd, action_pc_i, action_pc_transformed], axis=0),#, pred_pc_i], axis=0),
            np.concatenate([
                anchor_seg,
                np.ones(action_pc_i.shape[0]),
                np.ones(action_pc_i.shape[0]) * 2,
                # np.ones(action_pc_i.shape[0]) * 3,
            ]).astype(int)
        )
        # set titel of fig
        fig.update_layout(
            title=i
        )
        fig.show()

        # computing intermediate waypoint
        offset = action_pc_transformed.mean(axis=0) - anchor_pcd.mean(axis=0)
        offset = offset[0:2]
        offset = (offset / np.linalg.norm(offset)) * 0.12
        print(f'i{i}: ', R_i.as_matrix(), t_world_i.reshape(3), offset)

  

    quit()
    g = 2
    for i in range(g, g+1):
        pred_world_action_i = pred_world_action[i].cpu().numpy()
        print(pred_world_action_i.shape, action_pc.shape)
        R_gripper, t_gripper = rigid_transform_3D(action_pc.T, pred_world_action_i.T)
        
        # visualize optimized transformation
        R_gripper = R.from_matrix(R_gripper)
        action_pc_transformed = R_gripper.apply(action_pc) + t_gripper.T
        fig = vpl.segmentation_fig(
            np.concatenate([action_pc, anchor_pc, action_pc_transformed, pred_world_action_i], axis=0),
            np.concatenate([action_seg, anchor_seg, action_seg * 2, action_seg * 3], axis=0).astype(int),
        )
        fig.show()
        print(R_gripper.as_rotvec(), t_gripper)
        quit()

if __name__ == "__main__":
    main()

    """
    0.454, 0.309

    0.269, 0.257

    0.185    0.052
    """