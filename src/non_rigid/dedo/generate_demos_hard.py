import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib import interactive

# interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args
from dedo.utils.pcd_utils import visualize_data, render_video

from generate_demos import get_transform_scma_easy, get_transform_scma_easy_ood

import pybullet as p


import rpad.visualize_3d.plots as vpl
from scipy.spatial.transform import Rotation as R
from PIL import Image
import PIL


def demo_policy(env, hole_id, speed_factor=1.0):
    _, vertex_positions = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
    true_loop_vertices = env.args.deform_true_loop_vertices[hole_id]
    
    goal_pos = env.goal_pos[hole_id]
    pts = np.array(vertex_positions)
    cent_pts = pts[true_loop_vertices]
    cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]
    cent_pos = cent_pts.mean(axis=0)

    flow = goal_pos - cent_pos
    print(flow)
    a1_act = flow * 0.2 / 5 #6.36
    a2_act = flow * 0.2 / 5 # 6.36
    a1_act[2] += 0.01 / speed_factor
    a2_act[2] += 0.01 / speed_factor
    act = np.concatenate([a1_act, a2_act], axis=0).astype(np.float32)
    return act * speed_factor


def generate_demos(env, args, demo_path, mode, num_cloths, num_demos, num_holes):
    if num_demos % num_holes != 0:
        raise ValueError('Number of demos should be divisible by number of holes')
    os.makedirs(demo_path / mode, exist_ok=True)


    i = 0
    while i < num_cloths:
    # for i in range(num_cloths):
        print(f'----Generating demos for {num_holes}-hole cloth {i + 1} / {num_cloths}----')
        reset_counter = 0
        # initializing cloth geometry
        deform_params = {
            'node_density': 25,
            'num_holes': num_holes
        }

        deform_params = { # for single-cloth datasets
            'num_holes': num_holes,
            'node_density': 25,
            'w': 1.0,
            'h': 1.0,
            #'holes': [{'x0': 8 + j, 'x1': 16, 'y0': 9, 'y1': 13}]
            'holes': [{'x0': 8, 'x1': 16, 'y0': 9, 'y1': 13}]
        }
        # deform_params = {
        #     'num_holes': 1,
        #     'node_density': 15,
        #     'w': 1.0,
        #     'h': 1.0,
        #     'holes': [{'x0': 5, 'x1': 8, 'y0': 5, 'y1': 7}]
        # }

        obs = env.reset(deform_params=deform_params)
        deform_params = env.deform_params
        input('press Enter')

        print(deform_params)
        # generating demos per cloth
        j = 0

        while j < num_demos:
        # for j in range(num_demos):
            # if too many tries, skip to next cloth (deduct i accordingly)
            if reset_counter > 5:
                print('Too many retries; skipping to next cloth...')
                i -= 1
                break
            print(f'----Generating demo {j + 1} / {num_demos}----')
            # if j > 0:
            obs = env.reset(deform_params=deform_params)
            # input("Press Enter to continue...")
            # obs = env.reset(deform_params=deform_params)
            # updating policy velocity
            speed_factor = np.exp(np.random.uniform(0.0, 0.7))
            print(speed_factor)
            env.max_episode_len = args.max_episode_len * speed_factor

            # initial mesh/point cloud observation
            _, verts_init = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
            pcd_obs = env.get_pcd_obs()
            img_init, pcd_init, ids_init = pcd_obs.values()
            ids_rigid = ids_init[ids_init > 0]
            pcd_rigid = pcd_init[ids_init > 0]
            anchors = env.anchors

            # visualizing init for figure
            # action_pc = np.array(verts_init)
            # anchor_pc = np.array(pcd_rigid)
            # fig = vpl.segmentation_fig(
            #     np.concatenate([action_pc, anchor_pc]),
            #     np.concatenate([
            #         np.ones(action_pc.shape[0]), 
            #         np.zeros(anchor_pc.shape[0])
            #     ]).astype(np.int16),
            # )
            # fig.show()

            step = 0
            act = demo_policy(env, j % num_holes, 1.0 / speed_factor)
            # print(act)
            while True:
                assert (not isinstance(env.action_space, gym.spaces.Discrete))
                next_obs, rwd, done, info = env.step(act)

                if done:
                    centroid_check = env.check_centroid()
                    _, verts_goal = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
                    
                    # visualizing goal for figure
                    # action_pc = np.array(verts_goal)
                    # fig = vpl.segmentation_fig(
                    #     np.concatenate([action_pc, anchor_pc]),
                    #     np.concatenate([
                    #         np.ones(action_pc.shape[0]), 
                    #         np.zeros(anchor_pc.shape[0])
                    #     ]).astype(np.int16),
                    # )
                    # #fig.show()
                    # quit()
                    input('Enter')
                    info = env.end_episode(rwd)
                    input('Enter')
                    polygon_check = env.check_polygon()
                    success = np.any(centroid_check * polygon_check)
                    print('Task Successful: ', success)
                    break
                obs = next_obs
                step += 1
            # re-try demo if failed (deduct j accordingly)
            if not success:
                print('Demo failed; retrying...')
                reset_counter += 1
                continue
            else: # THIS IS JSUT FOR VIZ PURPOSES, GET RID OF LATER
                j = num_demos
                continue

            # processing demo
            verts_init = np.array(verts_init)
            verts_goal = np.array(verts_goal)
            # random transformation of goal
            # rot, trans = get_transform_scma_easy_ood() if mode == "val_ood" else get_transform_scma_easy()
            
            if mode == "val_ood" or mode == "val_ood_1":
                rot, trans = get_transform_scma_easy_ood()
            else:
                rot, trans = get_transform_scma_easy()
            
            verts_goal = (verts_goal @ rot.as_matrix().T) + trans
            pcd_rigid = (pcd_rigid @ rot.as_matrix().T) + trans

            # # random transformation of initial
            # if DATA_SPLIT == "val_ood2":
            #     cloth_rot = np.random.uniform(0, 2 * np.pi)
            #     cloth_rot = R.from_euler('z', cloth_rot)
            #     verts_init = (verts_init - env.args.deform_init_pos) @ cloth_rot.as_matrix().T + env.args.deform_init_pos
            # else:
            #     cloth_rot = [0, 0, 0]

            # saving demo
            flow = verts_goal - verts_init
            seg = np.ones(verts_init.shape[0])
            ids_rigid = ids_rigid * 0
            demo_id = i * num_demos + j
            # np.savez(demo_path / mode / f'demo_{demo_id}.npz',
            #          action_pc=verts_init, action_seg=seg,
            #          anchor_pc=pcd_rigid, anchor_seg=ids_rigid,
            #          flow=flow, speed_factor=speed_factor,
            #          rot=rot.as_euler('xyz'), trans=trans,
            #          # cloth_rot=cloth_rot.as_euler('xyz'),
            #          deform_params=deform_params, anchors=anchors,
            # )
            # increment demo counter
            j += 1
        # increment cloth counter
        i += 1

# TODO: delete the last demo
NUM_DEMOS = 4
NUM_HOLES = 1 # 1 or 2
DATA_SPLIT = "train" # train, val, or val_ood
NUM_CLOTHS = {
    "train": 100,
    "train_10": 10,
    "train_1": 1,
    "val": 10,
    "val_1": 1,
    "val_ood": 10,
    "val_ood_1": 1,
}
SEED = {
    "train": 0,
    "train_10": 0,
    "train_1": 50,
    "val": 100,
    "val_1": 50, # should be same cloth as train_1 
    "val_ood": 200,
    "val_ood_1": 50 # should be same cloth as train_1
}

if DATA_SPLIT == "train_10":
    NUM_DEMOS = 40
elif DATA_SPLIT == "train_1":
    NUM_DEMOS = 400
elif DATA_SPLIT == "val_1":
    NUM_DEMOS = 40
elif DATA_SPLIT == "val_ood_1":
    NUM_DEMOS = 40

def main(args):
    assert ('Robot' not in args.env), 'This is a simple demo for anchors only'
    np.set_printoptions(precision=4, linewidth=150, suppress=True)

    # TODO: pcd should always be true
    kwargs = {'args': args}
    env = gym.make(args.env, **kwargs)
    env.args.seed = SEED[DATA_SPLIT] + NUM_HOLES
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)
    # TODO: just pass a ton of episodes in 

    exp_name = f"multi_cloth_{NUM_HOLES}"
    demo_path = Path(os.path.expanduser(f"~/datasets/nrp/ProcCloth/{exp_name}/"))
    
    generate_demos(env, args, demo_path, 
                   DATA_SPLIT, num_cloths=NUM_CLOTHS[DATA_SPLIT], 
                   num_demos=NUM_DEMOS, num_holes=NUM_HOLES,
    )

    # generate_demos(env, args, demo_path, 'val', num_cloths=NUM_CLOTHS, num_demos=8, num_holes=2)

if __name__ == '__main__':
    args = get_args()
    main(args)