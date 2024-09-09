import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib import interactive

# interactive(True)
import numpy as np

import gym

from dedo.utils.args import get_args
from dedo.utils.pcd_utils import visualize_data, render_video

import pybullet as p


import rpad.visualize_3d.plots as vpl
from scipy.spatial.transform import Rotation as R

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



def policy(env, speed_factor=1.0):
    _, vertex_positions = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
    true_loop_vertices = env.args.deform_true_loop_vertices[1]
    
    goal_pos = env.goal_pos[1]
    pts = np.array(vertex_positions)
    cent_pts = pts[true_loop_vertices]
    cent_pts = cent_pts[~np.isnan(cent_pts).any(axis=1)]
    cent_pos = cent_pts.mean(axis=0)

    flow = goal_pos - cent_pos
    a1_act = flow * 0.2 / 5 #6.36
    a2_act = flow * 0.2 / 5 # 6.36
    a1_act[2] += 0.01 / speed_factor
    a2_act[2] += 0.01 / speed_factor
    act = np.concatenate([a1_act, a2_act], axis=0).astype(np.float32)
    return act * speed_factor




def generate_demo(env, num_episodes, args, demo_path, mode):
    # demo_path = Path(os.path.expanduser("~/datasets/nrp/ProcCloth/single_cloth/demos/train/"))
    # demo_path = demo_path / mode
    os.makedirs(demo_path / mode, exist_ok=True)
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        speed_factor = np.exp(np.random.uniform(-0.7, 0.7))
        env.max_episode_len = args.max_episode_len * speed_factor
        obs = env.reset()

        input('Press enter to start episode')

        # TODO: need camera pcd for rigid object...
        _, verts_init = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)

        pcd_obs = env.get_pcd_obs()
        img_init, pcd_init, ids_init = pcd_obs.values()

        ids_rigid = ids_init[ids_init > 0]
        pcd_rigid = pcd_init[ids_init > 0]

        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            # print('step', step)
            act = env.action_space.sample()  # in [-1,1]
            noise_act = 0.1 * act
            act = policy_simple(obs, noise_act, args.task, step, 1.0 / speed_factor)

            next_obs, rwd, done, info = env.step(act)

            if done:
                _, verts_goal = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
                info = env.end_episode(rwd)
                print('step', step)
                print('Task Successful: ', info['is_success'])
                break
            obs = next_obs
            step += 1

        # processing demo
        verts_init = np.array(verts_init)
        verts_goal = np.array(verts_goal)
        # every other episode, transform goal
        if epsd % 2 == 1:
            rot = R.from_euler('xyz', [0, 0, np.pi/4]).as_matrix()
            trans = np.array([5, -3, 0])
            verts_goal = (verts_goal @ rot.T) + trans
            pcd_rigid = (pcd_rigid @ rot.T) + trans


        # saving anchor
        if mode == "train" and (epsd == 0 or epsd == 1):
            np.savez(
                demo_path / f"anchor_{epsd}.npz",
                pc=pcd_rigid,
                id=ids_rigid,
            )

        flow = verts_goal - verts_init
        seg = np.ones(verts_init.shape[0])
        np.savez(demo_path / mode / f'demo_{epsd}.npz', pc_init=verts_init, flow=flow, seg=seg, speed_factor=speed_factor)
        # input('Episode ended; press enter to go on')
    pass



def generate_demos_multi_cloth(env, num_episodes, args):
    demo_path = Path(os.path.expanduser("~/datasets/nrp/ProcCloth/multi_cloth/demos/"))
    os.makedirs(demo_path, exist_ok=True)
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        speed_factor = np.exp(np.random.uniform(-0.7, 0.7))
        env.max_episode_len = args.max_episode_len * speed_factor
        obs = env.reset()


def generate_demos_single_cloth_multi_anchor(env, num_episodes, args, demo_path, mode):
    os.makedirs(demo_path / mode, exist_ok=True)
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        speed_factor = np.exp(np.random.uniform(-0.7, 0.7))
        env.max_episode_len = args.max_episode_len * speed_factor
        # obs = env.reset(rigid_trans=[0, 0, 10.0], rigid_rot=[0, np.pi / 2, 0])
        # obs = env.reset(rigid_trans=[0, 0, 0], rigid_rot=[0, 0, 0])

        # rot, transform = get_transform_scma_easy()
        # breakpoint()
        # obs = env.reset(rigid_trans=transform, rigid_rot=rot.as_euler('xyz'))

        # deform_params = {
        #     'num_holes': 1,
        #     'node_density': 15,
        #     'w': 1.0,
        #     'h': 1.0,
        #     'holes': [{'x0': 5, 'x1': 8, 'y0': 5, 'y1': 7}]
        # }
        # obs = env.reset(deform_params=deform_params)
        
        # TODO: procedural hang cloth should check each of these individually?
        deform_params = {
            'num_holes': 2, 
            'node_density': 25, 
            'w': 0.4621052219323072, 
            'h': 0.3401474209098767, 
            'holes': [
                {'x0': 3, 'x1': 8, 'y0': 12, 'y1': 19}, 
                {'x0': 13, 'x1': 14, 'y0': 13, 'y1': 16}
            ]
        }
        obs = env.reset(deform_params=deform_params)



        # obs = env.reset()
        print(env.deform_params)

        # input('Press enter to start episode')
        _, verts_init = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
        pcd_obs = env.get_pcd_obs()
        img_init, pcd_init, ids_init = pcd_obs.values()
        ids_rigid = ids_init[ids_init > 0]
        pcd_rigid = pcd_init[ids_init > 0]

        act = policy(env, 1.0 / speed_factor)
        step = 0
        while True:
            assert (not isinstance(env.action_space, gym.spaces.Discrete))
            # act = env.action_space.sample()
            # noise_act = 0.1 * act
            # act = policy_simple(obs, noise_act, args.task, step, 1.0 / speed_factor)

            next_obs, rwd, done, info = env.step(act)

            if done:
                centroid_check = env.check_centroid()
                _, verts_goal = p.getMeshData(env.deform_id, flags=p.MESH_DATA_SIMULATION_MESH)
                # env.check_success2()
                # success = env.check_success(debug=True)
                info = env.end_episode(rwd)
                polygon_check = env.check_polygon()
                success = np.any(centroid_check * polygon_check)
                # print('Task Successful: ', info['is_success'])
                print('step: ', step, 'Task Successful: ', success)
                break
            obs = next_obs
            step += 1
        
        # processing demo
        verts_init = np.array(verts_init)
        verts_goal = np.array(verts_goal)
        # random transformation
        rot, trans = get_transform_scma_easy_ood() if mode == "val_ood" else get_transform_scma_easy()
        print(rot.as_euler('xyz'), trans)
        verts_goal = (verts_goal @ rot.as_matrix().T) + trans
        pcd_rigid = (pcd_rigid @ rot.as_matrix().T) + trans


        # saving demo
        flow = verts_goal - verts_init
        seg = np.ones(verts_init.shape[0])
        ids_rigid = ids_rigid * 0
        # np.savez(demo_path / mode / f'demo_{epsd}.npz',
        #          action_pc=verts_init, action_seg=seg,
        #          anchor_pc=pcd_rigid, anchor_seg=ids_rigid,
        #          flow=flow, speed_factor=speed_factor,
        #          rot=rot.as_euler('xyz'), trans=trans)


def get_transform_scma_easy():
    z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
    rot = R.from_euler('z', 
        z_rot
    )#.as_matrix()
    transform = np.array([
        np.random.uniform() * 5 * np.power(-1, z_rot < 0),
        np.random.uniform() * -10,
        0.0
    ])
    return rot, transform

def get_transform_scma_easy_ood():
    z_rot = np.random.uniform(-np.pi / 3, np.pi / 3)
    rot = R.from_euler('z', 
        z_rot
    )#.as_matrix()
    transform = np.array([
        np.random.uniform(5, 10) * np.power(-1, z_rot < 0),
        np.random.uniform() * -10,
        np.random.uniform(1, 5)
    ])
    return rot, transform


GENERATE_TRAIN = True
GENERATE_VAL = True
GENERATE_VAL_OOD = True

def main(args):
    assert ('Robot' not in args.env), 'This is a simple demo for anchors only'
    np.set_printoptions(precision=4, linewidth=150, suppress=True)

    # TODO: pcd should always be true
    kwargs = {'args': args}
    env = gym.make(args.env, **kwargs)
    env.seed(env.args.seed)
    print('Created', args.task, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape)
    # TODO: just pass a ton of episodes in 

    exp_name = "single_cloth_multi_anchor"
    demo_path = Path(os.path.expanduser(f"~/datasets/nrp/ProcCloth/{exp_name}/demos/"))
    
    if GENERATE_TRAIN:
        mode = "train"
        num_demos = 128
        generate_demos_single_cloth_multi_anchor(env, num_demos, args, demo_path, mode)
        
    if GENERATE_VAL:
        mode = "val"
        num_demos = 32
        generate_demos_single_cloth_multi_anchor(env, num_demos, args, demo_path, mode)

    if GENERATE_VAL_OOD:
        mode = "val_ood"
        num_demos = 32
        generate_demos_single_cloth_multi_anchor(env, num_demos, args, demo_path, mode)
    env.close()


if __name__ == "__main__":
    main(get_args())
