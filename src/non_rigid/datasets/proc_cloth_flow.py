import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.utils.data as data
import torchvision as tv
from torchvision import transforms as T

import rpad.visualize_3d.plots as vpl

import numpy as np
import torch_geometric.data as tgd
import torch_geometric.loader as tgl
import torch_geometric.transforms as tgt

from pathlib import Path
import os
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.utils.transform_utils import random_se3
from non_rigid.utils.pointcloud_utils import downsample_pcd


class ProcClothFlowDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type="train"):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        # self.dataset_type = "flow"

        self.dataset_cfg = dataset_cfg
        self.scene = self.dataset_cfg.scene
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor
        
    
    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, index):
        # load data
        demo = np.load(self.dataset_dir / f"demo_{index}.npz")
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        speed_factor = torch.as_tensor(demo["speed_factor"]).float()
        flow = torch.as_tensor(demo["flow"]).float()
        rot = demo["rot"]
        trans = demo["trans"]

        # downsampling anchor point cloud
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        if self.scene:
            scene_pc = torch.cat([action_pc, anchor_pc], dim=0)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=0)
            anchor_flow = torch.zeros_like(anchor_pc)
            scene_flow = torch.cat([flow, anchor_flow], dim=0)
            item = {
                "pc": scene_pc + scene_flow, # Scene points in goal position
                "pc_action": scene_pc, # Scene points in starting position
                "seg": scene_seg, 
                "flow": scene_flow,
                "speed_factor": speed_factor,
                "rot": rot,
                "trans": trans,
            }
        else:
            item = {
                "pc": action_pc + flow, # Action points in goal position
                "pc_action": action_pc, # Action points in starting position
                "pc_anchor": anchor_pc, # Anchor points in goal position
                "seg": action_seg,
                "seg_anchor": anchor_seg,
                "speed_factor": speed_factor,
                "flow": flow,
                "rot": rot,
                "trans": trans,
            }
        return item



class ProcClothPointDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type="train"):
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))

        self.dataset_cfg = dataset_cfg
        self.sample_size_action = self.dataset_cfg.sample_size_action
        self.sample_size_anchor = self.dataset_cfg.sample_size_anchor

    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, index):
        # load data
        demo = np.load(self.dataset_dir / f"demo_{index}.npz")
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        speed_factor = torch.as_tensor(demo["speed_factor"]).float()
        flow = torch.as_tensor(demo["flow"]).float()
        rot = demo["rot"]
        trans=demo["trans"]

        # kinda confusing variables names for now, just for consistency
        points_action = action_pc + flow
        points_anchor = anchor_pc

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

        # Downsample the point clouds (for now, this is only for anchor)
        goal_points_anchor, _ = downsample_pcd(
            goal_points_anchor.unsqueeze(0),
            num_points=self.dataset_cfg.sample_size_anchor,
            type=self.dataset_cfg.downsample_type,
        )
        goal_points_anchor = goal_points_anchor.squeeze(0)

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

        # Get starting action point cloud (for now, don't rotate)
        points_action = action_pc - action_pc.mean(dim=0)

        return {
            "pc": goal_points_action, # Action points in goal position
            "pc_action": points_action, # Action points for context
            "pc_anchor": goal_points_anchor, # Anchor points in goal position
            "seg": action_seg,
            "seg_anchor": anchor_seg,
            "speed_factor": speed_factor,
            "flow": flow,
            "rot": rot,
            "trans": trans,
        }

DATASET_FN = {
    "cloth": ProcClothFlowDataset,
    "cloth_point": ProcClothPointDataset,
}




class ProcClothFlowDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, val_batch_size, num_workers, dataset_cfg):# type, scene):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.stage = None
        self.dataset_cfg = dataset_cfg

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "train"):
        self.stage = stage
        # self.train_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "train")
        # self.val_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "val")
        # self.val_ood_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "val_ood")
        
        self.train_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "train"
        )
        self.val_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val"
        )
        self.val_ood_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val_ood"
        )
        
        # generator = torch.Generator().manual_seed(42)
        # self.train_set, self.val_set = torch.utils.data.random_split(
        #     dataset, [0.9, 0.1], generator=generator
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "train" else False,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return val_dataloader, val_ood_dataloader