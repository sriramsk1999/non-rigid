import torch
import numpy as np
from pytorch3d.transforms import Transform3d, Translate

from non_rigid.models.df_base import DiffusionFlowBase, FlowPredictionInferenceModule, PointPredictionInferenceModule
from non_rigid.utils.script_utils import create_model

from diffusion_policy_3d.policy.base_policy import BasePolicy

import rpad.visualize_3d.plots as vpl


class TAX3D(BasePolicy):
    """
    This is simple TAX3D wrapper exclusively for policy rollouts.
    """
    def __init__(
            self,
            ckpt_file,
            device,
            eval_cfg,
            run_cfg,
    ):
        super().__init__()
        self.run_cfg = run_cfg
        self.eval_cfg = eval_cfg

        # switch mode to eval
        self.run_cfg.mode = "eval"
        self.run_cfg.inference = self.eval_cfg.inference

        network, model = create_model(self.run_cfg)
        self.network = network
        self.model = model

        # load network weights from checkpoint
        checkpoint = torch.load(ckpt_file, map_location=device)
        self.network.load_state_dict(
            {k.partition(".")[2]: v for k, v, in checkpoint["state_dict"].items()}
        )

        # self.network = DiffusionFlowBase(
        #     model_cfg=self.run_cfg.model
        # )
        
        # # Load the network weights.
        # checkpoint = torch.load(ckpt_file, map_location=self.device)
        # self.network.load_state_dict(
        #     {k.partition(".")[2]: v for k, v, in checkpoint["state_dict"].items()}
        # )
        # self.network.eval()

        # # Setting sample sizes.
        # if self.eval_cfg.inference.action_full:
        #     self.run_cfg.dataset.sample_size_action = -1
        # self.eval_cfg.inference.sample_size = self.run_cfg.dataset.sample_size_action    
        # self.eval_cfg.inference.sample_size_anchor = self.run_cfg.dataset.sample_size_anchor

        # # Load the model.
        # if self.run_cfg.model.type == "flow":
        #     self.model = FlowPredictionInferenceModule(
        #         self.network,
        #         inference_cfg=self.eval_cfg.inference,
        #         model_cfg=self.run_cfg.model,
        #     )
        # elif self.run_cfg.model.type == "point":
        #     self.model = PointPredictionInferenceModule(
        #         self.network,
        #         task_type=self.run_cfg.task_type,
        #         inference_cfg=self.eval_cfg.inference,
        #         model_cfg=self.run_cfg.model,
        #     )
        # else:
        #     raise ValueError(f"Unknown model type: {self.run_cfg.type}")
        self.network.eval()
        self.model.eval()
        self.model.to(device)
        self.to(device)

        # Initializing current goal position. This is set during policy reset.
        self.goal_position = None
        self.results_world = None



    def reset(self):
        """
        Since this is open loop, this function will set the goal position to None.
        """
        self.goal_position = None

    def predict_action(self, obs_dict, deform_params):
        """
        Predict the action.
        """
        # if goal_position is unset (after policy reset), predict the goal position.
        if self.goal_position == None:
            # pred_dict = self.model_predict(obs_dict)
            # pred_action = pred_dict["pred_world_action"]
            # pred_flow = pred_dict["pred_world_flow"]

            pred_action, results_world = self.model_predict(obs_dict)

            # pred_action = pred_dict["point"]["pred_world"]

            if self.eval_cfg.task.env_runner.task_name == "proccloth":
                # TODO: this is is missing segmentation logic for SD models
                goal1 = pred_action[:, deform_params['node_density'] - 1, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 0, :]# + torch.tensor([0, -0.5, 1.0], device=self.device)
            elif self.eval_cfg.task.env_runner.task_name == "hangbag":
                goal1 = pred_action[:, 209, :] + torch.tensor([0, -0.5, 1.0], device=self.device)
                goal2 = pred_action[:, 297, :] + torch.tensor([0, -0.5, 1.0], device=self.device)

                # adding hard-coded offset
                # flow1 = pred_flow[:, 209, :]
                # flow2 = pred_flow[:, 297, :]
                # flow1 = flow1 / torch.norm(flow1, dim=1, keepdim=True) * 1.18
                # flow2 = flow2 / torch.norm(flow2, dim=1, keepdim=True) * 1.18

                # goal1 = goal1 + flow1
                # goal2 = goal2 + flow2

            else:
                raise ValueError(f"Unknown task name: {self.eval_cfg.env_runner.task_name}")
            self.goal_position = torch.cat([goal1, goal2], dim=1).unsqueeze(0)
            # self.results_world = [res.squeeze().cpu().numpy() for res in pred_dict["results_world"]]
            self.results_world = [res.squeeze().cpu().numpy() for res in results_world]

        action_dict = {
            'action': self.goal_position,
        }
        return action_dict
    
    def model_predict(self, obs_dict):
        """
        TAX3D inference.
        """

        # TODO: this needs to be updated for SD models, but low priority
        # points_action = obs_dict["pc_action"]
        # points_anchor = obs_dict["pc_anchor"]

        action_pc = obs_dict["pc_action"]
        anchor_pc = obs_dict["pc_anchor"]
        action_seg = obs_dict["seg"]
        anchor_seg = obs_dict["seg_anchor"]


        if self.run_cfg.dataset.scene:
            # scene-level processing
            scene_pc = torch.cat([action_pc, anchor_pc], dim=1)
            scene_seg = torch.cat([action_seg, anchor_seg], dim=1)

            # center the point cloud
            scene_center = scene_pc.mean(dim=1)
            scene_pc = scene_pc - scene_center
            T_goal2world = Translate(scene_center).get_matrix()

            item = {
                "pc_action": scene_pc,
                "seg": scene_seg,
                "T_goal2world": T_goal2world,
            }
        else:
            # object-centric processing
            if self.run_cfg.dataset.world_frame:
                action_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
                anchor_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                action_center = action_pc.mean(dim=1)
                anchor_center = anchor_pc.mean(dim=1)

            # center the point clouds
            action_pc = action_pc - action_center
            anchor_pc = anchor_pc - anchor_center
            T_action2world = Translate(action_center).get_matrix()
            T_goal2world = Translate(anchor_center).get_matrix()

            item = {
                "pc_action": action_pc,
                "pc_anchor": anchor_pc,
                "seg": action_seg,
                "seg_anchor": anchor_seg,
                "T_action2world": T_action2world,
                "T_goal2world": T_goal2world,
            }

        pred_dict = self.model.predict(item, self.eval_cfg.inference.num_trials, progress=False)
        pred_action = pred_dict["point"]["pred_world"]
        results_world = pred_dict["results_world"]

        if self.run_cfg.dataset.scene:
            # masking out action object in scene-level processing
            pred_action = pred_action[:, scene_seg.squeeze(0).bool(), :]
            results_world = [res[:, scene_seg.squeeze(0).bool(), :] for res in results_world]

        return pred_action, results_world

        # if self.run_cfg.dataset.world_frame:
        #     action_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        #     anchor_center = torch.zeros(3, dtype=torch.float32, device=self.device).unsqueeze(0)
        # else:
        #     action_center = points_action.mean(dim=1)
        #     anchor_center = points_anchor.mean(dim=1)

        # points_action = points_action - action_center
        # points_anchor = points_anchor - anchor_center


        # T_action2world = Translate(action_center).get_matrix()
        # T_goal2world = Translate(anchor_center).get_matrix()

        # item = {
        #     'pc_action': points_action,
        #     'pc_anchor': points_anchor,
        #     'seg': obs_dict['seg'],
        #     'seg_anchor': obs_dict['seg_anchor'],
        #     'T_action2world': T_action2world,
        #     'T_goal2world': T_goal2world,
        # }
        
        return self.model.predict(item, self.eval_cfg.inference.num_trials, progress=False)
