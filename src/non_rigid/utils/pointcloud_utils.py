import numpy as np
from pytorch3d.ops import sample_farthest_points
import re
import torch
from typing import Tuple


def downsample_pcd(
    points: torch.Tensor, num_points: int = 1024, type: str = "fps"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Downsamples a pointcloud using a specified method.

    Args:
        points (torch.Tensor): [B, N, 3] Pointcloud to downsample.
        num_points (int): Number of points to downsample to.
        type (str): Method of downsampling the point cloud.

    Returns:
        (torch.Tensor): Downsampled pointcloud.
        (torch.Tensor): Indices of the downsampled points in the original pointcloud.
    """

    if re.match(r"^fps$", type) is not None:
        return sample_farthest_points(points, K=num_points, random_start_point=True)
    elif re.match(r"^random$", type) is not None:
        random_idx = torch.randperm(points.shape[1])[:num_points]
        return points[:, random_idx], random_idx
    elif re.match(r"^random_0\.[0-9]$", type) is not None:
        prob = float(re.match(r"^random_(0\.[0-9])$", type).group(1))
        if np.random.random() > prob:
            return sample_farthest_points(points, K=num_points, random_start_point=True)
        else:
            random_idx = torch.randperm(points.shape[1])[:num_points]
            return points[:, random_idx], random_idx
    elif re.match(r"^[0-9]+N_random_fps$", type) is not None:
        random_num_points = (
            int(re.match(r"^([0-9]+)N_random_fps$", type).group(1)) * num_points
        )
        random_idx = torch.randperm(points.shape[1])[:random_num_points]
        random_points = points[:, random_idx]
        return sample_farthest_points(
            random_points, K=num_points, random_start_point=True
        )
    else:
        raise NotImplementedError(f"Downsample type {type} not implemented")
