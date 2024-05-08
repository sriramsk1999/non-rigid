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

from non_rigid.utils.pointcloud_utils import downsample_pcd


class ProcClothFlowDataset(data.Dataset):
    def __init__(self, root, dataset_cfg, type="train"):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
        self.dataset_type = "flow"

        self.dataset_cfg = dataset_cfg
        self.scene = self.dataset_cfg.type_args.scene
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
                "pc": scene_pc, # Scene points in starting position
                "pc_action": scene_pc + scene_flow, # Scene points in goal position
                "seg": scene_seg, 
                "flow": scene_flow,
                "speed_factor": speed_factor,
                "rot": rot,
                "trans": trans,
            }
        else:
            item = {
                "pc": action_pc, # Action points in starting position
                "pc_action": action_pc + flow, # Action points in goal position
                "pc_anchor": anchor_pc, # Anchor points in goal position
                "seg": action_seg,
                "seg_anchor": anchor_seg,
                "speed_factor": speed_factor,
                "flow": flow,
                "rot": rot,
                "trans": trans,
            }
        return item
    
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
        self.train_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "train")
        self.val_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "val")
        self.val_ood_dataset = ProcClothFlowDataset(self.root, self.dataset_cfg, "val_ood")
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