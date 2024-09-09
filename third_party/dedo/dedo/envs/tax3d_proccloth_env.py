import os
import time

from matplotlib import cm  # for colors
import numpy as np
import gym
import pybullet
import pybullet_utils.bullet_client as bclient

from ..utils.anchor_utils import (
    attach_anchor, command_anchor_velocity, command_anchor_position, create_anchor, create_anchor_geom,
    pin_fixed, change_anchor_color_gray)
from ..utils.init_utils import (
    load_deform_object, load_rigid_object, reset_bullet, load_deformable, 
    load_floor, get_preset_properties, apply_anchor_params)
from ..utils.mesh_utils import get_mesh_data
from ..utils.task_info import (
    DEFAULT_CAM_PROJECTION, DEFORM_INFO, SCENE_INFO, TASK_INFO,
    TOTE_MAJOR_VERSIONS, TOTE_VARS_PER_VERSION)
from ..utils.procedural_utils import (
    gen_procedural_hang_cloth, gen_procedural_button_cloth)
from ..utils.args import preset_override_util
from ..utils.process_camera import ProcessCamera, cameraConfig

from scipy.spatial.transform import Rotation as R

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import plotly.graph_objects as go
from PIL import Image
import copy

from tax3d_env import Tax3DEnv

class Tax3DProcClothEnv(Tax3DEnv):
    """
    Tax3d environment for HangProcCloth task.
    """

    def __init__(self, args):
        super().__init__(args)

    def load_objects(self, sim, args, debug,
                     deform_pos = {}, rigid_pos = {},
                     deform_params = {}, anchor_params = {}):
        # Make v0 the random version
        if args.version == 0:
            args.use_random_textures = True

        data_path = os.path.join(os.path.split(__file__)[0], '..', 'data')
        args.data_path = data_path
        sim.setAdditionalSearchPath(data_path)

    def check_centroid(self):
        pass

    def check_polygon(self):
        pass

    def pseudo_expert_action(self):
        pass