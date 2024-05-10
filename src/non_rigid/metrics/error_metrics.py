import numpy as np
from pytorch3d.transforms import Transform3d, Rotate, Translate
from scipy.spatial.transform import Rotation as R
import torch
from typing import Dict, List

from non_rigid.utils.transform_utils import (
    flow_to_tf,
    get_degree_angle,
    get_translation,
    get_transform_list_min_rotation_errors,
    get_transform_list_min_translation_errors,
    matrix_from_list,
)


def get_rigid_errors(
    T_pred: Transform3d,
    T_gt: Transform3d,
    T_action2distractor_list: List[Transform3d] = None,
    error_type: str = "distractor_min",
) -> Dict[str, float]:
    """
    Compute rigid errors given a predicted rigid transform and ground truth rigid transforms.

    Args:
        T_pred: predicted rigid transform from action to predicted action
        T_gt: ground truth rigid transform from action to gt anchor goal
        T_action2distractor_list: List of rigid transforms from action to distractor anchor goals
        error_type: type of error to compute

    Returns:
        dict: dictionary of rigid errors
    """
    if error_type == "demo":
        T_pred_diff = T_gt.compose(T_pred.inverse())

        error_t_max, error_t_min, error_t_mean = get_translation(T_pred_diff)
        error_R_max, error_R_min, error_R_mean = get_degree_angle(T_pred_diff)

        return {
            "error_t_max": error_t_max,
            "error_t_min": error_t_min,
            "error_t_mean": error_t_mean,
            "error_R_max": error_R_max,
            "error_R_min": error_R_min,
            "error_R_mean": error_R_mean,
        }
    elif error_type == "distractor_min":
        assert (
            T_action2distractor_list is not None
        ), "T_action2distactor_list must be provided for distractor_min error"

        T_pred_diff = T_gt.compose(T_pred.inverse())

        T_action2distractor_diff_list = [
            T_action2distractor.compose(T_pred.inverse())
            for T_action2distractor in T_action2distractor_list
        ]

        error_t_max, error_t_min, error_t_mean = (
            get_transform_list_min_translation_errors(
                T_pred_diff, T_action2distractor_diff_list
            )
        )
        error_R_max, error_R_min, error_R_mean = get_transform_list_min_rotation_errors(
            T_pred_diff, T_action2distractor_diff_list
        )

        return {
            "error_t_max": error_t_max,
            "error_t_min": error_t_min,
            "error_t_mean": error_t_mean,
            "error_R_max": error_R_max,
            "error_R_min": error_R_min,
            "error_R_mean": error_R_mean,
        }
    else:
        raise ValueError(f"Invalid error type: {error_type}")


def get_rigid_available_pose_errors(
    T_pred: Transform3d,
    batch: Dict[str, torch.Tensor],
    trans_thresh: float = 0.02,
) -> Dict[str, float]:
    """
    Compute rigid errors given a predicted rigid transform and a set of saved available poses.

    Args:
        T_pred: predicted rigid transform from action to predicted action
        batch: dictionary of batch
        trans_thresh: translation threshold for available pose

    Returns:
        dict: dictionary of min. rigid errors
    """
    required_keys = [
        "pc_action_mean",
        "pc_anchor_mean",
        "rpdiff_obj_mesh_file",
        "rpdiff_saved_poses_path",
        "rpdiff_obj_final_obj_pose",
        "rpdiff_pcd_scale_factor",
    ]
    assert all(
        key in batch for key in required_keys
    ), f"Missing some required keys in batch. Required = {required_keys}"

    T0 = Transform3d(matrix=batch['T0'])
    T1 = Transform3d(matrix=batch['T1'])

    multi_obj_mesh_file = batch["rpdiff_obj_mesh_file"]
    parent_fnames = multi_obj_mesh_file["parent"][0]
    child_fnames = multi_obj_mesh_file["child"][0]

    # Get the final poses of the parent and child
    multi_obj_final_obj_poses = batch["rpdiff_obj_final_obj_pose"]

    parent_final_poses = multi_obj_final_obj_poses["parent"]
    parent_final_poses = torch.stack(parent_final_poses).permute(1, 0, 2)

    child_final_poses = multi_obj_final_obj_poses["child"]
    child_final_poses = torch.stack(child_final_poses).permute(1, 0, 2)

    # Get the mean of the original points
    points_action_means = batch["pc_action_mean"]
    points_anchor_means = batch["pc_anchor_mean"]

    batch_min_dists = []
    for batch_idx in range(len(parent_fnames)):
        # Get parent final poses (parent in world frame) as matrices
        parent_final_pose = parent_final_poses[batch_idx]
        parent_final_pose_mat = matrix_from_list(
            parent_final_pose.squeeze(-1).detach().cpu().numpy()
        )

        # Get child final poses (child in world frame) as matrices
        child_final_pose = child_final_poses[batch_idx]
        child_final_pose_mat = matrix_from_list(
            child_final_pose.squeeze(-1).detach().cpu().numpy()
        )

        # Extract rotation and translation, by default mat. rotation component is the inverse of the actual rotation
        parent_final_pose_rot = parent_final_pose_mat[:3, :3]
        parent_final_pose_rot_tf = Rotate(torch.Tensor(parent_final_pose_rot)).to(
            batch["pc_action"].device
        )
        parent_final_pose_translation = parent_final_pose_mat[:3, 3]
        parent_final_pose_translation_tf = Translate(
            torch.Tensor(-parent_final_pose_translation).unsqueeze(0)
        ).to(batch["pc_action"].device)

        child_final_pose_rot = child_final_pose_mat[:3, :3]
        child_final_pose_rot_tf = Rotate(torch.Tensor(child_final_pose_rot)).to(
            batch["pc_action"].device
        )
        child_final_pose_translation = child_final_pose_mat[:3, 3]
        child_final_pose_translation_tf = Translate(
            torch.Tensor(-child_final_pose_translation).unsqueeze(0)
        ).to(batch["pc_action"].device)

        # Compose the transform from the parent final frame (parent in world frame) to the parent frame
        parent_final_pose_inv_tf = parent_final_pose_translation_tf.compose(
            parent_final_pose_rot_tf
        )

        # Compose the transform from the child final frame (child in world frame) to the child frame
        child_final_pose_inv_tf = child_final_pose_translation_tf.compose(
            child_final_pose_rot_tf
        )

        # By default the action/anchor points are centered about the anchor mean
        translate_to_action_mean = Translate(
            -points_anchor_means[batch_idx].unsqueeze(0)
        ).to(batch["pc_action"].device)

        # Transform pose from parent frame to parent final frame (parent in world frame), then to anchor frame (centered about action mean)
        parent_pose_to_anchor_frame = parent_final_pose_inv_tf.inverse().compose(
            translate_to_action_mean
        )

        # Transform pose from child frame to child final frame (child in world frame), then to anchor frame (centered about action mean)
        child_pose_to_anchor_frame = child_final_pose_inv_tf.inverse().compose(
            translate_to_action_mean
        )

        # Transform pose from anchor frame to anchor trans frame
        parent_pose_to_trans_anchor_frame = parent_pose_to_anchor_frame.compose(
            T1[batch_idx]
        )

        # Get the predicted child pose in the trans anchor frame
        child_pred_pose = child_pose_to_anchor_frame.compose(T0[batch_idx]).compose(
            T_pred[batch_idx]
        )

        if (
            "book" in child_fnames[batch_idx]
            and "bookshelf" in parent_fnames[batch_idx]
        ):
            saved_available_poses_fname = batch["rpdiff_saved_poses_path"][batch_idx]

            loaded_poses = np.loadtxt(saved_available_poses_fname)
            loaded_poses = [matrix_from_list(pose) for pose in loaded_poses]

            # get avail poses in the trans anchor frame
            avail_poses_trans_anchor_frame_base = [
                np.matmul(
                    parent_pose_to_trans_anchor_frame.get_matrix()
                    .squeeze(0)
                    .T.detach()
                    .cpu()
                    .numpy(),
                    pose,
                )
                for pose in loaded_poses
            ]

            avail_poses_trans_anchor_frame = []
            for p_idx, pose in enumerate(avail_poses_trans_anchor_frame_base):
                # get all four orientations that work
                r1 = R.from_euler("xyz", [0, 0, 0]).as_matrix()
                r2 = R.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
                # r3 = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                # r4 = R.from_euler('xyz', [np.pi, 0, np.pi]).as_matrix()
                r5 = R.from_euler("xyz", [0, np.pi, 0]).as_matrix()
                r6 = R.from_euler("xyz", [np.pi, np.pi, 0]).as_matrix()
                # r7 = R.from_euler('xyz', [0, np.pi, np.pi]).as_matrix()
                # r8 = R.from_euler('xyz', [np.pi, np.pi, np.pi]).as_matrix()

                tf1 = np.eye(4)
                tf1[:-1, :-1] = r1
                tf2 = np.eye(4)
                tf2[:-1, :-1] = r2
                # tf3 = np.eye(4); tf3[:-1, :-1] = r3
                # tf4 = np.eye(4); tf4[:-1, :-1] = r4
                tf5 = np.eye(4)
                tf5[:-1, :-1] = r5
                tf6 = np.eye(4)
                tf6[:-1, :-1] = r6
                # tf7 = np.eye(4); tf7[:-1, :-1] = r7
                # tf8 = np.eye(4); tf8[:-1, :-1] = r8

                p1 = np.matmul(pose, tf1)
                p2 = np.matmul(pose, tf2)
                # p3 = np.matmul(pose, tf3)
                # p4 = np.matmul(pose, tf4)
                p5 = np.matmul(pose, tf5)
                p6 = np.matmul(pose, tf6)
                # p7 = np.matmul(pose, tf7)
                # p8 = np.matmul(pose, tf8)

                # all_poses_to_save = [p1, p2, p3, p4, p5, p6, p7, p8]
                all_poses_to_save = [p1, p2, p5, p6]

                # Don't save poses that are too close to existing ones
                for p_to_save in all_poses_to_save:

                    a_rotmat = p_to_save[:-1, :-1]
                    close_to_existing = False
                    for p2_idx, pose2 in enumerate(avail_poses_trans_anchor_frame):
                        trans_ = np.linalg.norm(
                            p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1
                        )

                        b_rotmat = pose2[:-1, :-1]
                        qa = R.from_matrix(a_rotmat).as_quat()
                        qb = R.from_matrix(b_rotmat).as_quat()

                        quat_scalar_prod = np.sum(qa * qb)
                        rot_ = 1 - quat_scalar_prod**2

                        if trans_ < 0.02 and rot_ < np.deg2rad(5):
                            close_to_existing = True
                            break

                    if not close_to_existing:
                        avail_poses_trans_anchor_frame.append(p_to_save)
                # avail_poses_trans_anchor_frame.extend(all_poses_to_save)

        elif "can" in child_fnames[batch_idx] and "cabinet" in parent_fnames[batch_idx]:
            saved_available_poses_fname = batch["rpdiff_saved_poses_path"][batch_idx]

            loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
            avail_pose_info_all = loaded_poses["avail_top_poses"]

            points_action = batch["pc_action"][batch_idx, :, :3]
            # Get the extents of the action points
            action_min = points_action.min(dim=0).values
            action_max = points_action.max(dim=0).values
            action_extents = action_max - action_min

            action_h = action_extents[-1]

            top_poses = [pose_info["pose"] for pose_info in avail_pose_info_all]
            base_poses = []
            for pose in top_poses:
                base_pose = pose.copy()
                base_pose[2, -1] += action_h / 2
                base_poses.append(base_pose)

            # get avail poses in the trans anchor frame
            avail_poses_trans_anchor_frame_base = [
                np.matmul(
                    parent_pose_to_trans_anchor_frame.get_matrix()
                    .squeeze(0)
                    .T.detach()
                    .cpu()
                    .numpy(),
                    pose,
                )
                for pose in base_poses
            ]

            avail_poses_trans_anchor_frame = []
            for p_idx, pose in enumerate(avail_poses_trans_anchor_frame_base):
                # get all orientations that work
                r1 = R.from_euler("xyz", [0, 0, 0]).as_matrix()
                r2 = R.from_euler("xyz", [np.pi, 0, 0]).as_matrix()
                # r3 = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                # r4 = R.from_euler('xyz', [np.pi, 0, np.pi]).as_matrix()
                # r5 = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()
                # r6 = R.from_euler('xyz', [np.pi, np.pi, 0]).as_matrix()
                # r7 = R.from_euler('xyz', [0, np.pi, np.pi]).as_matrix()
                # r8 = R.from_euler('xyz', [np.pi, np.pi, np.pi]).as_matrix()

                tf1 = np.eye(4)
                tf1[:-1, :-1] = r1
                tf2 = np.eye(4)
                tf2[:-1, :-1] = r2
                # tf3 = np.eye(4); tf3[:-1, :-1] = r3
                # tf4 = np.eye(4); tf4[:-1, :-1] = r4
                # tf5 = np.eye(4); tf5[:-1, :-1] = r5
                # tf6 = np.eye(4); tf6[:-1, :-1] = r6
                # tf7 = np.eye(4); tf7[:-1, :-1] = r7
                # tf8 = np.eye(4); tf8[:-1, :-1] = r8

                p1 = np.matmul(pose, tf1)
                p2 = np.matmul(pose, tf2)
                # p3 = np.matmul(pose, tf3)
                # p4 = np.matmul(pose, tf4)
                # p5 = np.matmul(pose, tf5)
                # p6 = np.matmul(pose, tf6)
                # p7 = np.matmul(pose, tf7)
                # p8 = np.matmul(pose, tf8)

                # avail_poses_trans_anchor_frame.append(p1)
                # avail_poses_trans_anchor_frame.append(p2)

                # all_poses_to_save = [p1, p2, p3, p4, p5, p6, p7, p8]
                all_poses_to_save = [p1, p2]

                for p_to_save in all_poses_to_save:

                    a_rotmat = p_to_save[:-1, :-1]
                    close_to_existing = False
                    for p2_idx, pose2 in enumerate(avail_poses_trans_anchor_frame):
                        trans_ = np.linalg.norm(
                            p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1
                        )

                        b_rotmat = pose2[:-1, :-1]
                        qa = R.from_matrix(a_rotmat).as_quat()
                        qb = R.from_matrix(b_rotmat).as_quat()

                        quat_scalar_prod = np.sum(qa * qb)
                        rot_ = 1 - quat_scalar_prod**2

                        if trans_ < 0.02 and rot_ < np.deg2rad(5):
                            close_to_existing = True
                            break

                    if not close_to_existing:
                        avail_poses_trans_anchor_frame.append(p_to_save)

        child_pred_pose_mat = (
            child_pred_pose.get_matrix().squeeze(0).T.detach().cpu().numpy()
        )
        child_pred_pose_trans = child_pred_pose_mat[:-1, -1]
        child_pred_pose_rot = child_pred_pose_mat[:-1, :-1]

        min_trans_dist = float("inf")
        min_rot_dist = float("inf")
        close_trans_list = []
        for pose in avail_poses_trans_anchor_frame:
            pose_trans = pose[:-1, -1]
            pose_rot = pose[:-1, :-1]

            trans_ = np.linalg.norm(child_pred_pose_trans - pose_trans, axis=-1)

            # q_child_pred = R.from_matrix(child_pred_pose_rot).as_quat()
            # q_pose = R.from_matrix(pose_rot).as_quat()

            # quat_scalar_prod = np.sum(q_child_pred * q_pose)
            # rot_ = 1 - quat_scalar_prod**2

            rot_child_pred = Rotate(torch.Tensor(child_pred_pose_rot.T)).to(
                batch["pc_action"].device
            )
            rot_pose = Rotate(torch.Tensor(pose_rot.T)).to(
                batch["pc_action"].device
            )
            _, _, rot_ = get_degree_angle(rot_pose.compose(rot_child_pred.inverse()))

            # Find the available pose thats closest to the T_pred (in terms of translation)
            if trans_ < min_trans_dist:
                min_trans_dist = trans_
                min_rot_dist = rot_

            # If T_pred is close to many available poses, need to find the one with best translation AND rotation error
            if trans_ < min_trans_dist + trans_thresh:
                close_trans_list.append((trans_, rot_))

        min_trans_rot_dist = float("inf")
        for dist_pair in close_trans_list:
            trans_, rot_ = dist_pair

            if trans_ + rot_ < min_trans_rot_dist:
                min_trans_rot_dist = trans_ + rot_
                min_trans_dist = trans_
                min_rot_dist = rot_

        batch_min_dists.append([min_trans_dist, min_rot_dist])

    # Calculate the batch mean of the min translation and rotation errors
    batch_min_dists = np.array(batch_min_dists)
    batch_min_dists = np.mean(batch_min_dists, axis=0)

    return {
        "error_t_mean": batch_min_dists[0],
        "error_R_mean": batch_min_dists[1],
    }


def get_pred_pcd_rigid_errors(
    batch: Dict[str, torch.Tensor],
    pred_xyz: torch.Tensor,
    error_type: str = "demo",
) -> Dict[str, float]:
    """
    Compute rigid errors given a predicted point cloud and ground truth point cloud.

    Args:
        batch: dictionary of batch data
        pred_xyz: [B, N, 3] tensor of predicted point positions
        error_type: type of error to compute

    Returns:
        dict: dictionary of rigid errors
    """
    start_xyz = batch["pc_action"]
    T_gt = batch["T_action2goal"]
    T_action2distractor_list = (
        batch["T_action2distractor_list"]
        if "T_action2distractor_list" in batch
        else None
    )

    T_gt_ = Transform3d(matrix=T_gt.permute(0, 2, 1))
    T_action2distractor_list_ = None
    if T_action2distractor_list is not None:
        T_action2distractor_list_ = [
            Transform3d(matrix=T_action2distractor.permute(0, 2, 1))
            for T_action2distractor in T_action2distractor_list
        ]

    pred_flows = pred_xyz - start_xyz
    T_pred = flow_to_tf(start_xyz, pred_flows)

    if error_type in ["demo", "distractor_min"]:
        errors = get_rigid_errors(T_pred, T_gt_, T_action2distractor_list_, error_type)
    elif error_type == "rpdiff_precision_wta":
        errors = get_rigid_available_pose_errors(T_pred, batch)
    return errors
