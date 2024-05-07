import numpy as np
from pytorch3d.transforms import (
    Transform3d,
    Rotate,
    axis_angle_to_matrix,
    quaternion_to_matrix,
)
import torch
import torch.nn.functional as F


def random_se3(
    N: int = 1,
    rot_var: float = np.pi / 180 * 5,
    trans_var: float = 0.1,
    rot_sample_method: str = "axis_angle",
    device: str = None,
) -> Transform3d:
    """
    Generates random transforms in SE(3) space.

    Args:
        N (int): Number of transforms to generate.
        rot_var (float): Maximum rotation in radians
        trans_var (float): Maximum translation
        rot_sample_method (str): Method to sample the rotation. Options are:
            - "axis_angle": Random axis angle sampling
            - "axis_angle_uniform_z": Random axis angle sampling with uniform z axis rotation
            - "quat_uniform": Uniform SE(3) sampling
            - "random_flat_upright": Random rotation around z axis and xy translation (no z translation)
            - "random_upright": Random rotation around z axis and xyz translation
        device: Device to put the transform on.

    Returns:
        (Transform3d): Random SE(3) transform(s).
    """

    assert rot_sample_method in [
        "axis_angle",
        "axis_angle_uniform_z",
        "quat_uniform",
        "random_flat_upright",
        "random_upright",
        "identity",
    ]

    if rot_sample_method == "axis_angle":
        # this is random axis angle sampling (rot_sample_method == "axis_angle")
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = (
            torch.rand(1).item()
            * rot_var
            / torch.norm(axis_angle_random, dim=1).max().item()
        )
        constrained_axix_angle = rot_ratio * axis_angle_random  # max angle is rot_var
        R = axis_angle_to_matrix(constrained_axix_angle)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "axis_angle_uniform_z":
        # this is random axis angle sampling
        axis_angle_random = torch.randn(N, 3, device=device)
        rot_ratio = (
            torch.rand(1).item()
            * rot_var
            / torch.norm(axis_angle_random, dim=1).max().item()
        )
        constrained_axix_angle = rot_ratio * axis_angle_random  # max angle is rot_var
        R_random = axis_angle_to_matrix(constrained_axix_angle)

        # this is uniform z axis rotation sampling
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R_z = axis_angle_to_matrix(axis_angle_z)

        R = torch.bmm(R_z, R_random)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    if rot_sample_method == "quat_uniform":
        # This true uniform SE(3) sampling tends to make it hard to train the models
        # In contrast, the axis angle sampling tends to leave the objects close to upright
        quat = torch.randn(N, 4, device=device)
        quat = quat / torch.linalg.norm(quat)
        R = quaternion_to_matrix(quat)
        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "random_upright":
        # Random rotation around z axis and xy translation (no z translation)
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)

        random_translation = torch.randn(N, 3, device=device)
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "random_flat_upright":
        # Random rotation around z axis and xy translation (no z translation)
        theta = torch.rand(N, 1, device=device) * 2 * np.pi
        axis_angle_z = torch.cat([torch.zeros(N, 2, device=device), theta], dim=1)
        R = axis_angle_to_matrix(axis_angle_z)

        random_translation = torch.randn(N, 3, device=device)
        random_translation[:, 2] = 0
        translation_ratio = (
            trans_var / torch.norm(random_translation, dim=1).max().item()
        )
        t = torch.rand(1).item() * translation_ratio * random_translation
    elif rot_sample_method == "identity":
        R = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)
        t = torch.zeros(N, 3, device=device)
    else:
        raise ValueError(f"Unknown rot_sample_method: {rot_sample_method}")
    return Rotate(R, device=device).translate(t)


def symmetric_orthogonalization(M):
    """Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)

    M: should have size [batch_size, 3, 3]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    return R


def flow_to_tf(
    start_xyz: torch.Tensor,
    flow: torch.Tensor,
) -> Transform3d:
    """
    Converts point-wise flows into a rigid transform.
    
    Args:
        start_xyz: (B, N, 3) tensor of initial point positions
        flow: (B, N, 3) tensor of point-wise flows
        
    Returns:
        Transform3d: rigid transform
    """
    assert start_xyz.shape == flow.shape
    B, N, _ = start_xyz.shape
    
    start_xyz_mean = start_xyz.mean(dim=1, keepdim=True)
    start_xyz_demean = start_xyz - start_xyz_mean
    
    target_xyz = start_xyz + flow
    target_xyz_mean = target_xyz.mean(dim=1, keepdim=True)
    target_xyz_demean = target_xyz - target_xyz_mean
    
    X = torch.bmm(start_xyz_demean.transpose(-2, -1), target_xyz_demean)
    
    R = symmetric_orthogonalization(X)
    t = target_xyz_mean - torch.bmm(start_xyz_mean, R)
    
    return Rotate(R).translate(t.squeeze(1))    
