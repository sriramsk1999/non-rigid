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

import rpad.visualize_3d.plots as vpl

from rpad.pybullet_envs.pm_suction import PMObjectEnv
from rpad.partnet_mobility_utils.dataset import read_ids, get_ids_by_class

if __name__ == "__main__":
    # OBJ_ID = "7273"
    # NUM_TRAINING_DEMOS = 128
    # NUM_VAL_DEMOS = 24
    OBJ_IDs = ['7310', '7167', '7292', '7273']
    NUM_TRAINING_DEMOS_PER_OBJ = 128
    NUM_VAL_DEMOS_PER_OBJ = 48
    PM_DIR = Path(os.path.expanduser("~/datasets/partnet-mobility/dataset"))
    # NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/{OBJ_ID}_flow"))
    # NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/{OBJ_ID}_flow_multi"))
    NR_DIR = Path(os.path.expanduser(f"~/datasets/nrp/dataset/demos/microwave-4_flow"))
    GUI = False
    # USE_EGL = not GUI
    USE_EGL = False
    GENERATE_TRAIN = True
    GENERATE_VAL = True
    GENERATE_TEST = False
    DEMO_PC_SIZE = 1200.0

    # TRAIN_MICROWAVES = ["7304", "7128", "7236", "7310", "7366", "7263", "7273", "7167", "7265", "7292"]
    # VAL_MICROWAVES = ["7349", "7296", "7119"]

    from matplotlib import pyplot as plt

    # env = PMObjectEnv(OBJ_ID, PM_DIR, GUI, use_egl=USE_EGL)
    # envs = [PMObjectEnv(OBJ_ID, PM_DIR, GUI, use_egl=USE_EGL) for OBJ_ID in OBJ_IDs]

    if GENERATE_TRAIN:
        os.makedirs(NR_DIR / "train", exist_ok=True)
        for i in range(len(OBJ_IDs)):
            env = PMObjectEnv(OBJ_IDs[i], PM_DIR, GUI, use_egl=USE_EGL)
            for j in range(NUM_TRAINING_DEMOS_PER_OBJ):
                print(f'Generating OBJ {OBJ_IDs[i]} training demo {j}...')
                # sampling random goal
                goal = np.random.rand() * np.pi / 2
                print(f'Goal: {goal * 180 / np.pi} degrees')
                pc_init, pc_goal, seg, rgb, t_wc = env.random_demo(goal, randomize_camera="random")
                fps_idx = fps(torch.as_tensor(pc_init).to(device="cuda:1"), random_start=True, ratio=DEMO_PC_SIZE / pc_init.shape[0])
                fps_idx = fps_idx.cpu().numpy()
                if fps_idx.shape[0] != DEMO_PC_SIZE:
                    # brute force to get the right number of points
                    j -= 1
                    continue
                pc_init = pc_init[fps_idx]
                pc_goal = pc_goal[fps_idx]
                seg = seg[fps_idx]
                
                # saving demo
                np.savez(
                    NR_DIR / "train" / f"obj_{i}_demo_{j}.npz", 
                    pc_init=pc_init, 
                    flow=pc_goal - pc_init, 
                    seg=seg, 
                    rgb=rgb,
                    t_wc=t_wc, 
                    goal=goal,
                    obj_id=OBJ_IDs[i]
                )

    if GENERATE_VAL:
        os.makedirs(NR_DIR / "val", exist_ok=True)
        for i in range(len(OBJ_IDs)):
            env = PMObjectEnv(OBJ_IDs[i], PM_DIR, GUI, use_egl=USE_EGL)
            for j in range(NUM_VAL_DEMOS_PER_OBJ):
                print(f'Generating OBJ {OBJ_IDs[i]} validation demo {j}...')
                # sampling random goal
                goal = np.random.rand() * np.pi / 2
                # save the camera transformation as well?
                pc_init, pc_goal, seg, rgb, t_wc = env.random_demo(goal, randomize_camera="random")
                fps_idx = fps(torch.as_tensor(pc_init).to(device="cuda:1"), random_start=True, ratio=DEMO_PC_SIZE / pc_init.shape[0])
                fps_idx = fps_idx.cpu().numpy()
                if fps_idx.shape[0] != DEMO_PC_SIZE:
                    # brute force to get the right number of points
                    j -= 1
                    continue
                pc_init = pc_init[fps_idx]
                pc_goal = pc_goal[fps_idx]
                seg = seg[fps_idx]
                # saving demo
                np.savez(
                    NR_DIR / "val" / f"obj_{i}_demo_{j}.npz", 
                    pc_init=pc_init, 
                    flow=pc_goal - pc_init, 
                    seg=seg, 
                    rgb=rgb,
                    t_wc=t_wc, 
                    goal=goal,
                    obj_id=OBJ_IDs[i]
                )
