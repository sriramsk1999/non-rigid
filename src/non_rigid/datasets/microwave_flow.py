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



class MicrowaveFlowDataset(data.Dataset):
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
        t_wc = torch.as_tensor(demo["t_wc"]).float()
        goal = torch.as_tensor(demo["goal"]).float()
        # return pc_init, flow, seg, t_wc, goal
        return {
            "pc_init": pc_init,
            "flow": flow,
            "seg": seg,
            "t_wc": t_wc,
            "goal": goal
        }



class MicrowaveFlowDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, val_batch_size, num_workers):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        self.train_dataset = MicrowaveFlowDataset(self.root, "train")
        self.val_dataset = MicrowaveFlowDataset(self.root, "val")
        # generator = torch.Generator().manual_seed(42)
        # self.train_set, self.val_set = torch.utils.data.random_split(
        #     dataset, [0.9, 0.1], generator=generator
        # )

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
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    # testing the dataset and datamodule
    root = Path(os.path.expanduser("~/datasets/nrp/dataset/demos/microwave_flow/"))
    type = "val"

    dataset = MicrowaveFlowDataset(root, type)
    goals = []
    for i in range(len(dataset)):
        pc, flow, seg, t, goal = dataset[i]
        print(pc.shape, flow.shape, seg.shape)


    # x, y = dataset[0]
    # pos = x[:, :3]
    # flow = x[:, 3:6]
    # fig1 = vpl.segmentation_fig(pos, torch.ones((pos.shape[0],), dtype=torch.int64))
    # fig1.show()
    # fig2 = vpl.segmentation_fig(pos + flow, torch.ones((pos.shape[0],), dtype=torch.int64))
    # fig2.show()

    # x, y = dataset[1]
    # pos = x[:, :3]
    # flow = x[:, 3:6]
    # fig1 = vpl.segmentation_fig(pos, torch.ones((pos.shape[0],), dtype=torch.int64))
    # fig1.show()
    # fig2 = vpl.segmentation_fig(pos + flow, torch.ones((pos.shape[0],), dtype=torch.int64))
    # fig2.show()
    # quit()
    