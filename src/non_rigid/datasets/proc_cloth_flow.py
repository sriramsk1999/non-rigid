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


class ProcClothFlowDataset(data.Dataset):
    def __init__(self, root, type="train"):
        # This is a toy dataset - no need to normalize or otherwise process point cloud with torch geometric
        super().__init__()
        self.root = root
        self.type = type
        self.dataset_dir = self.root / self.type
        self.num_demos = int(len(os.listdir(self.dataset_dir)))
    
    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, index):
        # load data
        demo = np.load(self.dataset_dir / f"demo_{index}.npz")
        pc_init = torch.as_tensor(demo["pc_init"]).float()
        flow = torch.as_tensor(demo["flow"]).float()
        seg = torch.as_tensor(demo["seg"]).int()
        # return pc_init, flow, seg
        return {
            "pc_init": pc_init,
            "flow": flow,
            "seg": seg,
        
        }
    
class ProcClothFlowDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, val_batch_size, num_workers):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str):
        self.train_dataset = ProcClothFlowDataset(self.root, "train")
        self.val_dataset = ProcClothFlowDataset(self.root, "val")
        # generator = torch.Generator().manual_seed(42)
        # self.train_set, self.val_set = torch.utils.data.random_split(
        #     dataset, [0.9, 0.1], generator=generator
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    

if __name__ == "__main__":
    root = Path(os.path.expanduser("~/datasets/nrp/ProcCloth/single_cloth/demos/"))
    type = "val"

    dataset = ProcClothFlowDataset(root, type)

    points = []
    segs = []

    for i in range(len(dataset)):
        pc_init, flow, seg = dataset[i]
        points.append(pc_init + flow)
        segs.append(seg * i)
    points = np.concatenate(points, axis=0)
    segs = np.concatenate(segs, axis=0)
    fig = vpl.segmentation_fig(points, segs)
    fig.show()