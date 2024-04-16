import numpy as np
from pathlib import Path
import os
import time
import copy

import torch 
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import torch_geometric.transforms as tgt
from torch_geometric.nn import fps

from rpad.pybullet_envs.pm_suction import PMObjectEnv

if __name__ == "__main__":
    OBJ_ID = "7273"
    NUM_DEMOS = 8
    NUM_VAL_DEMOS = 2
    # PM_DIR = Path(os.path.expanduser("~/datasets/partnet-mobility/dataset"))
    PM_DIR = Path(os.path.expanduser("~/workspace/rpad/data/partnet-mobility/dataset"))
    # NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/{OBJ_ID}_flow"))
    # NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/{OBJ_ID}_flow_multi"))
    # NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/{OBJ_ID}_flow_uniform"))
    NR_DIR = Path(os.path.expanduser(f"~/workspace/rpad/data/non_rigid/dataset/demos/{OBJ_ID}_flow_uniform"))
    GUI = True
    USE_EGL = not GUI
    GENERATE_TRAIN = False
    GENERATE_VAL = True
    GENERATE_TEST = False
    # angles = [np.pi/6, np.pi/3]
    # 16 angles even spaced from 0 to pi/2
    angles = np.linspace(0, np.pi/2, 16)

    env = PMObjectEnv(OBJ_ID, PM_DIR, GUI, use_egl=USE_EGL)

    env.reset()
    env.set_axis_of_rotation()
    obs = env.get_obs()
    # obs_rot = env.rotate_link_pc(obs, np.pi/4)
    # obs_rot1 = env.rotate_link_pc(obs, angles[0])
    # obs_rot2 = env.rotate_link_pc(obs, angles[1])

    obs_rots = [env.rotate_link_pc(obs, angle) for angle in angles]

    if GENERATE_TRAIN:
        os.makedirs(NR_DIR / "train", exist_ok=True)
        for i in range(NUM_DEMOS):
            print(f'Generating training demo {i}...')
            # fps sampling
            fps_idx = fps(torch.as_tensor(obs['pc']).to(device="cuda"), random_start=True, ratio=0.025)
            fps_idx = fps_idx.cpu().numpy()

            # get all downsampled obs, obs_rot, seg, start_angle, and goal_angle
            pc_start_i = obs['pc'][fps_idx]
            seg_i = obs['seg'][fps_idx][:, np.newaxis]

            for j, angle in enumerate(angles):
                goal_angle_i = obs_rots[j]['pc'][fps_idx]
                # computing flow
                flow_angle_i = goal_angle_i - pc_start_i
                # saving to file
                pc_angle_i = np.concatenate((pc_start_i, flow_angle_i, seg_i), axis=-1)
                angles_i = np.array([obs["joint_angle"], angle])
                np.savez(NR_DIR / "train" / f"demo_{i*len(angles) + j}.npz", pc=pc_angle_i, angles=angles_i)

    if GENERATE_VAL:
        os.makedirs(NR_DIR / "val", exist_ok=True)
        for i in range(NUM_VAL_DEMOS):
            print(f'Generating validation demo {i}...')
            # fps sampling
            fps_idx = fps(torch.as_tensor(obs['pc']).to(device="cuda"), random_start=True, ratio=0.025)
            fps_idx = fps_idx.cpu().numpy()

            # get all downsampled obs, obs_rot, seg, start_angle, and goal_angle
            pc_start_i = obs['pc'][fps_idx]
            seg_i = obs['seg'][fps_idx][:, np.newaxis]

            goal_flows_i = [obs['pc'][fps_idx] - pc_start_i for obs in obs_rots]
            goal_flows_i = np.stack(goal_flows_i, axis=0)
            pc_angle_i = np.concatenate((pc_start_i, seg_i), axis=-1)
            angles_i = np.array([obs["joint_angle"], *angles])
            np.savez(NR_DIR / "val" / f"demo_{i}.npz", pc=pc_angle_i, goals=goal_flows_i, angles=angles_i)

    if GENERATE_TEST:
        os.makedirs(NR_DIR / "test", exist_ok=True)
        # generate separate test set
        for i in range(NUM_DEMOS):
            print(f'Generating test demo {i}...')
            # fps sampling
            fps_idx = fps(torch.as_tensor(obs['pc']).to(device="cuda"), random_start=True, ratio=0.025)
            fps_idx = fps_idx.cpu().numpy()

            # get all downsampled obs, obs_rot, seg, start_angle, and goal_angle
            pc_start_i = obs['pc'][fps_idx]
            seg_i = obs['seg'][fps_idx][:, np.newaxis]
            pc_i = np.concatenate((pc_start_i, seg_i), axis=-1)
            # test set does not have goal flow - just save the point cloud
            np.save(NR_DIR / "test" / f"test_{i}.npy", pc_i)