from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import os
from pathlib import Path
import lightning as L
import omegaconf
import torch
import torch.utils.data as data
from typing import Dict

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd, get_multi_anchor_scene


@dataclass
class RigidDatasetCfg:
    name: str = "rigid"
    data_dir: str = "data/rigid"
    type: str = "train"
    
    ###################################################
    # General Dataset Parameters
    ###################################################
    # Number of demos to load
    num_demos: int = None
    # Length of the train dataset
    train_dataset_size: int = 256
    # Length of the validation dataset
    val_dataset_size: int = 16
    
    ###################################################
    # Point cloud augmentation parameters
    ###################################################
    # [action, anchor, none], centers the point clouds w.r.t. the action, anchor, or no centering
    center_type: str = "anchor"
    # Number of points to downsample to
    sample_size: int = 1024
    # Method of downsampling the point cloud
    downsample_type: str = "fps"
    # Scale factor to apply to the point clouds
    pcd_scale_factor: float = 1.0
    # Demonstration transformation parameters
    action_transform_type: str = "quat_uniform"
    anchor_transform_type: str = "quat_uniform"
    # Translation and rotation variance for the action and anchor transformations
    translation_variance: float = 0.5
    rotation_variance: float = 180
    
    ###################################################
    # Distractor anchor point cloud parameters
    ###################################################
    # Number of distractor anchor point clouds to load
    distractor_anchor_pcds: int = 0
    # Transformation type to apply when generating distractor pcds
    distractor_transform_type: str = "random_flat_upright"
    # Translation and rotation variance for the distractor transformations
    distractor_translation_variance: float = 0.5
    distractor_rotation_variance: float = 180
    
class RigidPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action -= center
        points_anchor -= center

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Downsample the point clouds
        points_action, _ = downsample_pcd(
            points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        points_anchor, _ = downsample_pcd(
            points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)
        
        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor
        
        # Transform the point clouds
        # ransform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )
        
        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)
        
        # Get starting action point cloud
        # Transform the action point cloud
        points_action = goal_points_action - goal_points_action.mean(dim=0)
        points_action = T0.transform_points(points_action)
        
        # Center the action point cloud
        points_action = points_action - points_action.mean(dim=0)
        
        return {
            "pc": goal_points_action, # Action points in goal position
            "pc_anchor": goal_points_anchor, # Anchor points in goal position
            "pc_action": points_action, # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
        }


class RigidFlowDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center

        goal_points_action = torch.as_tensor(goal_points_action).float()
        goal_points_anchor = torch.as_tensor(goal_points_anchor).float()

        # Downsample the point clouds
        goal_points_action, _ = downsample_pcd(
            goal_points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_anchor, _ = downsample_pcd(
            goal_points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_action = goal_points_action.squeeze(0)
        goal_points_anchor = goal_points_anchor.squeeze(0)
        
        # Apply scale factor
        goal_points_action *= self.dataset_cfg.pcd_scale_factor
        goal_points_anchor *= self.dataset_cfg.pcd_scale_factor
        
        # Transform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )
        
        goal_points_action = T1.transform_points(goal_points_action)
        goal_points_anchor = T1.transform_points(goal_points_anchor)
        
        # Get starting action point cloud
        # Transform the action point cloud
        points_action = goal_points_action.clone() - goal_points_action.mean(dim=0)
        points_action = T0.transform_points(points_action)
        
        # Center the action point cloud
        points_action = points_action - points_action.mean(dim=0)
        
        # Calculate goal flow
        flow = goal_points_action - points_action
        
        return {
            "pc": points_action, # Action points in starting position
            "pc_anchor": goal_points_anchor, # Anchor points in goal position
            "pc_action": goal_points_action, # Action points in goal position
            "flow": flow,
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
        }


class NDFPointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: RigidDatasetCfg = RigidDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        
        dir_type = self.type if self.type == "train" else "test"
        self.dataset_dir = self.root / f'{dir_type}_data/renders'
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None and self.type == "train":
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} {self.type} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        points_action = torch.as_tensor(points_action).float()
        points_anchor = torch.as_tensor(points_anchor).float()

        # Add distractor anchor point clouds
        if self.dataset_cfg.distractor_anchor_pcds > 0:
            _, points_action, points_anchor_base, distractor_anchor_pcd_list, T_distractor_list, debug = get_multi_anchor_scene(
                points_gripper=None,
                points_action=points_action.unsqueeze(0),
                points_anchor_base=points_anchor.unsqueeze(0),
                rot_var=self.dataset_cfg.distractor_rotation_variance,
                trans_var=self.dataset_cfg.distractor_translation_variance,
                rot_sample_method=self.dataset_cfg.distractor_transform_type,
                num_anchors_to_add=self.dataset_cfg.distractor_anchor_pcds,
            )
            points_anchor = torch.cat([points_anchor_base] + distractor_anchor_pcd_list, dim=1).squeeze(0)
            points_action = points_action.squeeze(0)
            
        # Center the point clouds
        if self.dataset_cfg.center_type == "action_center":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_center":
            center = points_anchor.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor_random":
            center = points_anchor[np.random.choice(len(points_anchor))]
        elif self.dataset_cfg.center_type == "none":
            center = np.zeros(3)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        points_action -= center
        points_anchor -= center


        # Downsample the point clouds
        points_action, _ = downsample_pcd(
            points_action.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        points_anchor, _ = downsample_pcd(
            points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size,
            type=self.dataset_cfg.downsample_type,
        )
        points_action = points_action.squeeze(0)
        points_anchor = points_anchor.squeeze(0)
        
        # Apply scale factor
        points_action *= self.dataset_cfg.pcd_scale_factor
        points_anchor *= self.dataset_cfg.pcd_scale_factor
        
        # Transform the point clouds
        # ransform the point clouds
        T0 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.action_transform_type,
        )
        T1 = random_se3(
            N=1,
            rot_var=self.dataset_cfg.rotation_variance,
            trans_var=self.dataset_cfg.translation_variance,
            rot_sample_method=self.dataset_cfg.anchor_transform_type,
        )
        
        goal_points_action = T1.transform_points(points_action)
        goal_points_anchor = T1.transform_points(points_anchor)
        
        # Get starting action point cloud
        # Transform the action point cloud
        points_action = goal_points_action - goal_points_action.mean(dim=0)
        points_action = T0.transform_points(points_action)
        
        # Center the action point cloud
        points_action = points_action - points_action.mean(dim=0)
        
        return {
            "pc": goal_points_action, # Action points in goal position
            "pc_anchor": goal_points_anchor, # Anchor points in goal position
            "pc_action": points_action, # Action points for context
            "T0": T0.get_matrix().squeeze(0).T,
            "T1": T1.get_matrix().squeeze(0).T,
        }


DATASET_FN = {
    "rigid_point": RigidPointDataset,
    "rigid_flow": RigidFlowDataset,
    "ndf_point": NDFPointDataset,
}


class RigidDataModule(L.LightningModule):
    def __init__(
        self,
        root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        type: str = "rigid",
        dataset_cfg: omegaconf.DictConfig = None,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.type = type
        self.dataset_cfg = dataset_cfg

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        self.train_dataset = DATASET_FN[self.type](
            self.root,
            type="train",
            dataset_cfg=RigidDatasetCfg(
                **omegaconf.OmegaConf.to_container(self.dataset_cfg)
            ),
        )
        self.val_dataset = DATASET_FN[self.type](
            self.root,
            type="val",
            dataset_cfg=RigidDatasetCfg(
                **omegaconf.OmegaConf.to_container(self.dataset_cfg)
            ),
        )

    def train_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )
