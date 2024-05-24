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
        self.world_frame = self.dataset_cfg.world_frame
        if self.scene and not self.world_frame:
            raise ValueError("Scene inputs require a world frame.")
        # if world frame, manually override data pre-processing
        if self.world_frame:
            print("-------Overriding data pre-processing for world frame.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0
        
    
    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, index):
        # load data
        demo = np.load(self.dataset_dir / f"demo_{index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        speed_factor = torch.as_tensor(demo["speed_factor"]).float()
        flow = torch.as_tensor(demo["flow"]).float()
        rot = torch.as_tensor(demo["rot"]).float()
        trans = torch.as_tensor(demo["trans"]).float()

        # initializing item
        item = {
            "speed_factor": speed_factor,
            "rot": rot,
            "trans": trans,
        }
        # legacy, because some old demos don't have this field
        if "deform_params" in demo:
            item["deform_params"] = demo["deform_params"].item()

        # downsampling action point cloud
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            flow = flow[action_pc_indices.squeeze(0)]

        # downsampling anchor point cloud
        anchor_pc, anchor_pc_indices = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        anchor_pc = anchor_pc.squeeze(0)
        anchor_seg = anchor_seg[anchor_pc_indices.squeeze(0)]


        if self.scene:
            scene_pc = torch.cat([action_pc, anchor_pc], dim=0)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=0)
            anchor_flow = torch.zeros_like(anchor_pc)
            scene_flow = torch.cat([flow, anchor_flow], dim=0)
            # item = {
            #     "pc": scene_pc + scene_flow, # Scene points in goal position
            #     "pc_action": scene_pc, # Scene points in starting position
            #     "seg": scene_seg, 
            #     "flow": scene_flow,
            #     "speed_factor": speed_factor,
            #     "rot": rot,
            #     "trans": trans,
            # }
            item["pc"] = scene_pc + scene_flow # Scene points in goal position
            item["pc_action"] = scene_pc # Scene points in starting position
            item["seg"] = scene_seg
            item["flow"] = scene_flow
        else:
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
                center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
            goal_points_action = points_action - center
            goal_points_anchor = points_anchor - center

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
            T_goal2world = T1.inverse().compose(
                Translate(center.unsqueeze(0))
            )

            # Get starting action point cloud (TODO: eventually, include T0)

            if self.dataset_cfg.action_context_center_type == "center":
                action_center = action_pc.mean(axis=0)
            elif self.dataset_cfg.action_context_center_type == "random":
                action_center = action_pc[np.random.choice(len(action_pc))]
            elif self.dataset_cfg.action_context_center_type == "none":
                action_center = torch.zeros(3, dtype=torch.float32)
            else:
                raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")

            points_action = action_pc - action_center
            T_action2world = Translate(action_center.unsqueeze(0))

            # Get the flow
            gt_flow = goal_points_action - points_action
            # item = {
            #     "pc": goal_points_action, # Action points in goal position
            #     "pc_action": points_action, # Action points for context
            #     "pc_anchor": goal_points_anchor, # Anchor points in goal position
            #     "seg": action_seg,
            #     "seg_anchor": anchor_seg,
            #     "speed_factor": speed_factor,
            #     "flow": gt_flow, # flow in goal position
            #     "rot": rot,
            #     "trans": trans,
            #     "T_goal2world": T_goal2world.get_matrix().squeeze(0),
            #     "T_action2world": T_action2world.get_matrix().squeeze(0),
            # }
            item["pc"] = goal_points_action # Action points in goal position
            item["pc_action"] = points_action # Action points for context
            item["pc_anchor"] = goal_points_anchor # Anchor points in goal position
            item["seg"] = action_seg
            item["seg_anchor"] = anchor_seg
            item["flow"] = gt_flow
            item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
            item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
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
        self.world_frame = self.dataset_cfg.world_frame
        # if world frame, manually override data pre-processing
        if self.world_frame:
            print("-------Overriding data pre-processing for world frame.-------")
            self.dataset_cfg.center_type = "none"
            self.dataset_cfg.action_context_center_type = "none"
            self.dataset_cfg.action_transform_type = "identity"
            self.dataset_cfg.anchor_transform_type = "identity"
            self.dataset_cfg.rotation_variance = 0.0
            self.dataset_cfg.translation_variance = 0.0

    def __len__(self):
        return self.num_demos
    
    def __getitem__(self, index):
        # load data
        demo = np.load(self.dataset_dir / f"demo_{index}.npz", allow_pickle=True)
        action_pc = torch.as_tensor(demo["action_pc"]).float()
        action_seg = torch.as_tensor(demo["action_seg"]).int()
        anchor_pc = torch.as_tensor(demo["anchor_pc"]).float()
        anchor_seg = torch.as_tensor(demo["anchor_seg"]).int()
        speed_factor = torch.as_tensor(demo["speed_factor"]).float()
        flow = torch.as_tensor(demo["flow"]).float()
        rot = torch.as_tensor(demo["rot"]).float()
        trans = torch.as_tensor(demo["trans"]).float()

        # initializing item
        item = {
            "speed_factor": speed_factor,
            "rot": rot,
            "trans": trans,
        }
        # legacy, because some old demos don't have this field
        if "deform_params" in demo:
            item["deform_params"] = demo["deform_params"].item()

        # downsample action
        if self.sample_size_action > 0 and action_pc.shape[0] > self.sample_size_action:
            action_pc, action_pc_indices = downsample_pcd(action_pc.unsqueeze(0), self.sample_size_action, type=self.dataset_cfg.downsample_type)
            action_pc = action_pc.squeeze(0)
            action_seg = action_seg[action_pc_indices.squeeze(0)]
            flow = flow[action_pc_indices.squeeze(0)]
        # downsample anchor
        anchor_pc, _ = downsample_pcd(anchor_pc.unsqueeze(0), self.sample_size_anchor, type=self.dataset_cfg.downsample_type)
        anchor_pc = anchor_pc.squeeze(0)

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
            center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown center type: {self.dataset_cfg.center_type}")
        goal_points_action = points_action - center
        goal_points_anchor = points_anchor - center

        # Downsample the point clouds (for now, this is only for anchor)
        # goal_points_anchor, _ = downsample_pcd(
        #     goal_points_anchor.unsqueeze(0),
        #     num_points=self.dataset_cfg.sample_size_anchor,
        #     type=self.dataset_cfg.downsample_type,
        # )
        # goal_points_anchor = goal_points_anchor.squeeze(0)

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
        T_goal2world = T1.inverse().compose(
            Translate(center.unsqueeze(0))
        )

        # Get starting action point cloud (TODO: eventually, include T0)
        # action_center = action_pc.mean(axis=0)
        # points_action = action_pc - action_center
        # T_action2world = Translate(action_center.unsqueeze(0))

        if self.dataset_cfg.action_context_center_type == "center":
            action_center = action_pc.mean(axis=0)
        elif self.dataset_cfg.action_context_center_type == "random":
            action_center = action_pc[np.random.choice(len(action_pc))]
        elif self.dataset_cfg.action_context_center_type == "none":
            action_center = torch.zeros(3, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown action context center type: {self.dataset_cfg.action_context_center_type}")

        points_action = action_pc - action_center
        T_action2world = Translate(action_center.unsqueeze(0))

        # return {
        #     "pc": goal_points_action, # Action points in goal position
        #     "pc_action": points_action, # Action points for context
        #     "pc_anchor": goal_points_anchor, # Anchor points in goal position
        #     "seg": action_seg,
        #     "seg_anchor": anchor_seg,
        #     "speed_factor": speed_factor,
        #     "flow": flow,
        #     "rot": rot,
        #     "trans": trans,
        #     "T_goal2world": T_goal2world.get_matrix().squeeze(0),
        #     "T_action2world": T_action2world.get_matrix().squeeze(0),
        # }
        item["pc"] = goal_points_action # Action points in goal position
        item["pc_action"] = points_action # Action points for context
        item["pc_anchor"] = goal_points_anchor # Anchor points in goal position
        item["seg"] = action_seg 
        item["seg_anchor"] = anchor_seg
        item["flow"] = flow
        item["T_goal2world"] = T_goal2world.get_matrix().squeeze(0)
        item["T_action2world"] = T_action2world.get_matrix().squeeze(0)
        return item

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
        # TODO: this needs to be fixed later
        if "multi_cloth" in self.dataset_cfg and self.dataset_cfg.multi_cloth.size == 10:
            train_type = f"train_{self.dataset_cfg.multi_cloth.size}"
        else:
            train_type = "train"
        self.train_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, train_type
        )
        self.val_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val"
        )
        self.val_ood_dataset = DATASET_FN[self.dataset_cfg.type](
            self.root, self.dataset_cfg, "val_ood"
        )
    
    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True if self.stage == "train" else False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
    
    def val_dataloader(self):
        val_dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        val_ood_dataloader = data.DataLoader(
            self.val_ood_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=cloth_collate_fn,
        )
        return val_dataloader, val_ood_dataloader


# custom collate function to handle deform params
def cloth_collate_fn(batch):
    # batch is a list of dictionaries
    # we need to convert it to a dictionary of lists
    keys = batch[0].keys()
    out = {k: None for k in keys}

    for k in keys:
        if k == "deform_params":
            out[k] = [item[k] for item in batch]
        else:
            out[k] = torch.stack([item[k] for item in batch])
    return out


if __name__ == "__main__":
    dir = Path("/home/eycai/datasets/nrp/ProcCloth/multi_cloth_1/train_10")
    import rpad.visualize_3d.plots as vpl
    for i in range(16):
        demo = np.load(dir / f"demo_{i}.npz", allow_pickle=True)
        action = demo["action_pc"]
        anchor = demo["anchor_pc"]
        action_seg = demo["action_seg"]
        anchor_seg = demo["anchor_seg"]
        flow = demo["flow"]
        goal = action + flow

        # action = torch.tensor(action).float()
        # action, indices = downsample_pcd(action.unsqueeze(0), 512, type="fps")
        # action_seg = action_seg[indices.squeeze(0)]
        # vpl.segmentation_fig(action.squeeze(0), action_seg.astype(int)).show()



        action_seg = np.ones_like(action_seg).astype(np.int32)
        anchor_seg = np.zeros_like(anchor_seg).astype(np.int32)
        goal_seg = np.ones_like(action_seg).astype(np.int32) * 5

        fig = vpl.segmentation_fig(
            np.concatenate([action, anchor, goal], axis=0),
            np.concatenate([action_seg, anchor_seg, goal_seg], axis=0),
        )
        fig.show()
    print('done')