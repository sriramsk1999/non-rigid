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
from non_rigid.utils.pointcloud_utils import downsample_pcd


@dataclass
class PointDatasetCfg:
    name: str = "point"
    data_dir: str = "data/point"
    type: str = "train"
    
    # Number of demos to load
    num_demos: int = None
    # Length of the train dataset
    train_dataset_size: int = 256
    # Length of the validation dataset
    val_dataset_size: int = 16
    
    # [action, anchor, none], centers the point clouds w.r.t. the action, anchor, or no centering
    center_type: str = "anchor"
    # Number of points to downsample to
    sample_size: int = 1024
    # Method of downsampling the point cloud
    downsample_type: str = "fps"


class PointDataset(data.Dataset):
    def __init__(
        self,
        root: Path,
        type: str = "train",
        dataset_cfg: PointDatasetCfg = PointDatasetCfg(),
    ):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_cfg = dataset_cfg
        
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.demo_files = list(self.dataset_dir.glob("*_teleport_obj_points.npz"))
        if self.dataset_cfg.num_demos is not None:
            self.demo_files = self.demo_files[: self.dataset_cfg.num_demos]
            self.num_demos = len(self.demo_files)
        print(f"Loaded {self.num_demos} demos from {self.dataset_dir}")

    def __len__(self):
        if self.type == "train":
            return self.dataset_cfg.train_dataset_size
        elif self.type == "val":
            return self.dataset_cfg.val_dataset_size
        else:
            raise ValueError(f"Unknown dataset type: {self.type}")

    @lru_cache(maxsize=1000)
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        demo = np.load(self.demo_files[index % self.num_demos])

        # Extract point clouds
        points_raw = demo["clouds"]
        classes_raw = demo["classes"]

        # TODO: Remove this hardcoded class selection
        points_action = points_raw[classes_raw == 0]
        points_anchor = points_raw[classes_raw == 1]

        # Center the point clouds
        if self.dataset_cfg.center_type == "action":
            center = points_action.mean(axis=0)
        elif self.dataset_cfg.center_type == "anchor":
            center = points_anchor.mean(axis=0)
        else:
            center = np.zeros(3)
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

        return {
            "pc_action": points_action,
            "pc_anchor": points_anchor,
        }


class PointDataModule(L.LightningModule):
    def __init__(
        self,
        root: Path,
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        type: str = "point",
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
        self.train_dataset = PointDataset(
            self.root,
            type="train",
            dataset_cfg=PointDatasetCfg(
                **omegaconf.OmegaConf.to_container(self.dataset_cfg)
            ),
        )
        self.val_dataset = PointDataset(
            self.root,
            type="val",
            dataset_cfg=PointDatasetCfg(
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
