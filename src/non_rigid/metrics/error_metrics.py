from pytorch3d.transforms import Transform3d
import torch
from typing import Dict, List

from non_rigid.utils.transform_utils import (
    flow_to_tf,
    get_degree_angle,
    get_translation,
    get_transform_list_min_rotation_errors,
    get_transform_list_min_translation_errors,
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


def get_pred_pcd_rigid_errors(
    start_xyz: torch.Tensor,
    pred_xyz: torch.Tensor,
    T_gt: torch.Tensor,
    T_action2distractor_list: List[torch.Tensor] = None,
    error_type: str = "distractor_min",
) -> Dict[str, float]:
    """
    Compute rigid errors given a predicted point cloud and ground truth point cloud.

    Args:
        start_xyz: [B, N, 3] tensor of initial point positions
        pred_xyz: [B, N, 3] tensor of predicted point positions
        T_gt: rigid transform from action to anchor goal
        T_action2distractor_list: List of rigid transforms from action to distractor anchor goals
        error_type: type of error to compute

    Returns:
        dict: dictionary of rigid errors
    """
    T_gt_ = Transform3d(matrix=T_gt.permute(0, 2, 1))
    T_action2distractor_list_ = None
    if T_action2distractor_list is not None:
        T_action2distractor_list_ = [
            Transform3d(matrix=T_action2distractor.permute(0, 2, 1))
            for T_action2distractor in T_action2distractor_list
        ]

    pred_flows = pred_xyz - start_xyz
    T_pred = flow_to_tf(start_xyz, pred_flows)

    errors = get_rigid_errors(T_pred, T_gt_, T_action2distractor_list_, error_type)
    return errors
