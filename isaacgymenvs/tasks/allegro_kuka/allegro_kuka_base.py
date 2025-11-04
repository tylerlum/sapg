# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import time
import datetime
from pathlib import Path
import io
import math
import os
import random
import tempfile
from copy import copy, deepcopy
from os.path import join
from typing import List, Tuple, Optional

from isaacgym import gymapi, gymtorch, gymutil
from torch import Tensor

from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import DofParameters, populate_dof_properties
from isaacgymenvs.utils.observation_action_utils import compute_observation, OBS_NAMES, compute_joint_pos_targets, create_chain_and_serial_chain
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.allegro_kuka.generate_cuboids import (
    generate_big_cuboids,
    generate_default_cube,
    generate_small_cuboids,
    generate_sticks,
)
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.allegro_kuka.object_trajectories import (
    get_hammer_trajectory, get_screwdriver_trajectory, get_marker_trajectory, get_eraser_trajectory, get_phone_trajectory)

DATETIME_STR = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# VISUALIZE_PD_TARGET_AS_BLUE_ROBOT = False is default
# Set to True to visualize the PD target as a blue robot
VISUALIZE_PD_TARGET_AS_BLUE_ROBOT = True


class AllegroKukaBase(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.frame_since_restart: int = 0  # number of control steps since last restart across all actors

        self.hand_arm_asset_file: str = self.cfg["env"]["asset"]["kukaAllegro"]

        self.clamp_abs_observations: float = self.cfg["env"]["clampAbsObservations"]

        self.privileged_actions = self.cfg["env"]["privilegedActions"]
        self.privileged_actions_torque = self.cfg["env"]["privilegedActionsTorque"]

        # 4 joints for index, middle, ring, and thumb and 7 for kuka arm
        self.num_arm_dofs = 7
        self.num_finger_dofs = 4
        self.num_allegro_fingertips = 4
        self.num_hand_dofs = self.num_finger_dofs * self.num_allegro_fingertips
        self.num_hand_arm_dofs = self.num_hand_dofs + self.num_arm_dofs

        self.num_allegro_kuka_actions = self.num_hand_arm_dofs
        if self.privileged_actions:
            self.num_allegro_kuka_actions += 3

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.allegro_actions_penalty_scale = self.cfg["env"]["allegroActionsPenaltyScale"]

        self.dof_params: DofParameters = DofParameters.from_cfg(self.cfg)

        self.initial_tolerance = self.cfg["env"]["successTolerance"]
        self.target_tolerance = self.cfg["env"]["targetSuccessTolerance"]
        self.success_tolerance = self.initial_tolerance
        self.tolerance_curriculum_increment = self.cfg["env"]["toleranceCurriculumIncrement"]
        self.tolerance_curriculum_interval = self.cfg["env"]["toleranceCurriculumInterval"]

        self.save_states = self.cfg["env"]["saveStates"]
        self.save_states_filename = self.cfg["env"]["saveStatesFile"]

        self.should_load_initial_states = self.cfg["env"]["loadInitialStates"]
        self.load_states_filename = self.cfg["env"]["loadStatesFile"]
        self.initial_root_state_tensors = self.initial_dof_state_tensors = None
        self.initial_state_idx = self.num_initial_states = 0

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

        self.reset_position_noise_x = self.cfg["env"]["resetPositionNoiseX"]
        self.reset_position_noise_y = self.cfg["env"]["resetPositionNoiseY"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise_fingers = self.cfg["env"]["resetDofPosRandomIntervalFingers"]
        self.reset_dof_pos_noise_arm = self.cfg["env"]["resetDofPosRandomIntervalArm"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.use_relative_control = self.cfg["env"]["useRelativeControl"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.success_steps: int = self.cfg["env"]["successSteps"]

        # 1.0 means keypoints correspond to the corners of the object
        # larger values help the agent to prioritize rotation matching
        self.keypoint_scale = self.cfg["env"]["keypointScale"]

        # size of the object (i.e. cube) before scaling
        self.object_base_size = self.cfg["env"]["objectBaseSize"]

        # whether to sample random object dimensions
        self.randomize_object_dimensions = self.cfg["env"]["randomizeObjectDimensions"]
        self.with_small_cuboids = self.cfg["env"]["withSmallCuboids"]
        self.with_big_cuboids = self.cfg["env"]["withBigCuboids"]
        self.with_sticks = self.cfg["env"]["withSticks"]

        self.with_dof_force_sensors = False
        # create fingertip force-torque sensors
        self.with_fingertip_force_sensors = False

        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (self.control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",  # 0.05m box
            "table": "urdf/table_narrow.urdf",
            "bucket": "urdf/objects/bucket.urdf",
            "lightbulb": "lightbulb/A60_E27_SI.urdf",
            "socket": "E27SocketSimple.urdf",
            "ball": "urdf/objects/ball.urdf",
        }

        self.keypoints_offsets = self._object_keypoint_offsets()

        self.num_keypoints = len(self.keypoints_offsets)

        self.allegro_fingertips = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        self.fingertip_offsets = np.array(
            [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]], dtype=np.float32
        )
        self.palm_offset = np.array([-0.00, -0.02, 0.16], dtype=np.float32)

        assert self.num_allegro_fingertips == len(self.allegro_fingertips)

        # can be only "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_state"]):
            raise Exception("Unknown type of observations!")

        print("Obs type:", self.obs_type)

        num_dof_pos = self.num_hand_arm_dofs
        num_dof_vel = self.num_hand_arm_dofs
        num_dof_forces = self.num_hand_arm_dofs if self.with_dof_force_sensors else 0

        palm_pos_size = 3
        palm_rot_vel_angvel_size = 10

        obj_rot_vel_angvel_size = 10

        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips

        keypoint_info_size = self.num_keypoints * 3 + self.num_keypoints * 3
        object_scales_size = 3
        max_keypoint_dist_size = 1
        lifted_object_flag_size = 1
        progress_obs_size = 1 + 1
        closest_fingertip_distance_size = self.num_allegro_fingertips
        reward_obs_size = 1

        self.full_state_size = (
            num_dof_pos
            + num_dof_vel
            + num_dof_forces
            + palm_pos_size
            + palm_rot_vel_angvel_size
            + obj_rot_vel_angvel_size
            + fingertip_rel_pos_size
            + keypoint_info_size
            + object_scales_size
            + max_keypoint_dist_size
            + lifted_object_flag_size
            + progress_obs_size
            + closest_fingertip_distance_size
            + reward_obs_size
            # + self.num_allegro_actions
        )

        num_states = self.full_state_size

        self.num_obs_dict = {
            "full_state": self.full_state_size,
        }

        self.up_axis = "z"

        self.fingertip_obs = True

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.num_allegro_kuka_actions

        self.cfg["device_type"] = sim_device.split(":")[0] if sim_device.find(":") != -1 else sim_device
        self.cfg["device_id"] = int(sim_device.split(":")[1]) if sim_device.find(":") != -1 else 0
        self.cfg["headless"] = headless

        # Must subscribe to keyboard events before calling super().__init__()
        self._subscribe_to_keyboard_events()

        super().__init__(
            config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        # Index of environment to view in viewer and camera
        self.index_to_view = 0

        # Camera position and target for viewer
        cam_target = gymapi.Vec3(0.0, 0.0, 0.53)
        cam_pos = cam_target + gymapi.Vec3(0.0, -1.0, 0.5)
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(self.viewer, self.envs[self.index_to_view], cam_pos, cam_target)

        # Init camera for wandb logging
        self._initialize_camera_sensor(cam_pos=cam_pos, cam_target=cam_target)
        self._modify_render_settings_if_headless()

        # volume to sample target position from
        target_volume_origin = np.array([0, 0.05, 0.8], dtype=np.float32)
        target_volume_extent = np.array([[-0.4, 0.4], [-0.05, 0.3], [-0.12, 0.25]], dtype=np.float32)
        
        self.target_volume_origin = torch.from_numpy(target_volume_origin).to(self.device).float()
        self.target_volume_extent = torch.from_numpy(target_volume_extent).to(self.device).float()

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state":
            if self.with_fingertip_force_sensors:
                sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
                self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(
                    self.num_envs, self.num_allegro_fingertips * 6
                )

            if self.with_dof_force_sensors:
                dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
                self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(
                    self.num_envs, self.num_hand_arm_dofs
                )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)

        self.hand_arm_default_dof_pos = torch.zeros(self.num_hand_arm_dofs, dtype=torch.float, device=self.device)

        desired_kuka_pos = torch.tensor([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])  # pose v1
        # desired_kuka_pos = torch.tensor([-2.135, 0.843, 1.786, -0.903, -2.262, 1.301, -2.791])  # pose v2
        self.hand_arm_default_dof_pos[:7] = desired_kuka_pos

        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]
        if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_arm_dofs:]
            self.blue_robot_arm_hand_dof_pos = self.blue_robot_arm_hand_dof_state[..., 0]
            self.blue_robot_arm_hand_dof_vel = self.blue_robot_arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.set_actor_root_state_object_indices: List[Tensor] = []
        self.set_dof_state_object_indices: List[Tensor] = []

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(
            self.num_envs, -1
        )
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_successes = torch.zeros_like(self.successes)

        # true objective value for the whole episode, plus saving values for the previous episode
        self.true_objective = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_true_objective = torch.zeros_like(self.true_objective)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(self.num_envs, device=self.device)
            + torch.log(self.force_prob_range[1])
        )

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.action_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.obj_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )
        self.goal_keypoint_pos = torch.zeros(
            (self.num_envs, self.num_keypoints, 3), dtype=torch.float, device=self.device
        )

        # how many steps we were within the goal tolerance
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_total_episode_closest_keypoint_max_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_episode_closest_keypoint_max_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.prev_episode_closest_keypoint_max_dist = 1000*torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.closest_fingertip_dist = -torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )
        self.furthest_hand_dist = -torch.ones([self.num_envs], dtype=torch.float, device=self.device)

        self.finger_rew_coeffs = torch.ones(
            [self.num_envs, self.num_allegro_fingertips], dtype=torch.float, device=self.device
        )

        reward_keys = [
            "raw_fingertip_delta_rew",
            "raw_hand_delta_penalty",
            "raw_lifting_rew",
            "raw_keypoint_rew",
            "fingertip_delta_rew",
            "hand_delta_penalty",
            "lifting_rew",
            "lift_bonus_rew",
            "keypoint_rew",
            "bonus_rew",
            "kuka_actions_penalty",
            "allegro_actions_penalty",
        ]

        self.rewards_episode = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys
        }

        self.last_curriculum_update = 0

        self.episode_root_state_tensors = [[] for _ in range(self.num_envs)]
        self.episode_dof_states = [[] for _ in range(self.num_envs)]

        self.eval_stats: bool = self.cfg["env"]["evalStats"]

        self.good_reset_boundary = self.cfg["env"].get("goodResetBoundary", 0) # Max number of envs that can be reset with good states
        
        if self.good_reset_boundary > 0:
            self.max_buffer_size = self.cfg["env"].get("maxBufferSize", self.max_episode_length * self.num_envs * 2 // max(self.max_consecutive_successes, 20)) # Max number of states that can be stored in the buffer
            self.max_temp_buffer_size = self.max_episode_length # Max number of states that can be stored in the buffer
            self.temp_root_states_buf = torch.empty((self.num_envs, self.max_temp_buffer_size, self.root_state_tensor.shape[0] // self.num_envs, *self.root_state_tensor.shape[1:]), dtype=self.root_state_tensor.dtype, device='cpu')
            self.temp_dof_states_buf = torch.empty((self.num_envs, self.max_temp_buffer_size, self.dof_state.shape[0] // self.num_envs, *self.dof_state.shape[1:]), dtype=self.dof_state.dtype, device='cpu')
            self.temp_buffer_index = torch.zeros(self.num_envs, dtype=torch.int, device='cpu')
            self.root_state_resets = torch.empty((self.max_buffer_size, self.root_state_tensor.shape[0] // self.num_envs, *self.root_state_tensor.shape[1:]), dtype=self.root_state_tensor.dtype, device='cpu')
            self.dof_resets = torch.empty((self.max_buffer_size, self.dof_state.shape[0] // self.num_envs, *self.dof_state.shape[1:]), dtype=self.dof_state.dtype, device='cpu')
            self.buffer_index = 0
            self.buffer_length = 0
        
        if self.eval_stats:
            self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.successes_count = torch.zeros(
                self.max_consecutive_successes + 1, dtype=torch.float, device=self.device
            )
            from tensorboardX import SummaryWriter

            self.eval_summary_dir = "./eval_summaries"
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil

                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)

    ##### KEYBOARD START #####
    def _subscribe_to_keyboard_events(self) -> None:
        from dataclasses import dataclass
        from typing import Callable

        @dataclass
        class KeyboardShortcut:
            name: str
            key: int
            function: Callable

        keyboard_shortcuts = [
            KeyboardShortcut(
                name="breakpoint",
                key=gymapi.KEY_B,
                function=self._breakpoint_callback,
            ),
            KeyboardShortcut(
                name="reset",
                key=gymapi.KEY_E,
                function=self._reset_callback,
            ),
            KeyboardShortcut(
                name="toggle_do_not_move",
                key=gymapi.KEY_T,
                function=self._toggle_do_not_move_callback,
            ),
        ]
        self.name_to_keyboard_shortcut_dict = {
            keyboard_shortcut.name: keyboard_shortcut
            for keyboard_shortcut in keyboard_shortcuts
        }
        self._DO_NOT_MOVE = False

    def _breakpoint_callback(self) -> None:
        print("Breakpoint")
        breakpoint()

    def _reset_callback(self) -> None:
        print("Resetting...")
        # Easiest way to reset without other issues (reset being overwritten)
        # Is to set the progress buffer to the max episode length - 2 so it will reset shortly
        self.progress_buf[:] = self.max_episode_length - 2

    def _toggle_do_not_move_callback(self) -> None:
        print("Toggling do not move...")
        self._DO_NOT_MOVE = not self._DO_NOT_MOVE
        print(f"Do not move is now {self._DO_NOT_MOVE}")

    ##### KEYBOARD END #####

    # AllegroKukaBase abstract interface - to be overriden in derived classes
    def change_on_restart(self, cfg):
        self.frame_since_restart = 0
        self.last_curriculum_update = 0
        
        self.cfg["env"]["distanceDeltaRewScale"] = cfg["env"]["distanceDeltaRewScale"]
        self.cfg["env"]["liftingRewScale"] = cfg["env"]["liftingRewScale"]
        self.cfg["env"]["liftingBonus"] = cfg["env"]["liftingBonus"]
        self.cfg["env"]["liftingBonusThreshold"] = cfg["env"]["liftingBonusThreshold"]
        self.cfg["env"]["keypointRewScale"] = cfg["env"]["keypointRewScale"]
        self.cfg["env"]["kukaActionsPenaltyScale"] = cfg["env"]["kukaActionsPenaltyScale"]
        self.cfg["env"]["allegroActionsPenaltyScale"] = cfg["env"]["allegroActionsPenaltyScale"]
        self.cfg["env"]["reachGoalBonus"] = cfg["env"]["reachGoalBonus"]
        self.cfg["env"]["fallDistance"] = cfg["env"]["fallDistance"]
        self.cfg["env"]["fallPenalty"] = cfg["env"]["fallPenalty"]


        self.distance_delta_rew_scale = self.cfg["env"]["distanceDeltaRewScale"]
        self.lifting_rew_scale = self.cfg["env"]["liftingRewScale"]
        self.lifting_bonus = self.cfg["env"]["liftingBonus"]
        self.lifting_bonus_threshold = self.cfg["env"]["liftingBonusThreshold"]
        self.keypoint_rew_scale = self.cfg["env"]["keypointRewScale"]
        self.kuka_actions_penalty_scale = self.cfg["env"]["kukaActionsPenaltyScale"]
        self.allegro_actions_penalty_scale = self.cfg["env"]["allegroActionsPenaltyScale"]        

        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]

    def _object_keypoint_offsets(self):
        raise NotImplementedError()

    def _object_start_pose(self, allegro_pose, table_pose_dy, table_pose_dz):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = allegro_pose.p.x

        pose_dy, pose_dz = table_pose_dy, table_pose_dz + 0.25

        object_start_pose.p.y = allegro_pose.p.y + pose_dy
        object_start_pose.p.z = allegro_pose.p.z + pose_dz

        return object_start_pose

    def _main_object_assets_and_scales(self, object_asset_root, tmp_assets_dir):
        object_asset_files, object_asset_scales = self._box_asset_files_and_scales(object_asset_root, tmp_assets_dir)
        if not self.randomize_object_dimensions:
            object_asset_files = object_asset_files[:1]
            object_asset_scales = object_asset_scales[:1]

        # randomize order
        files_and_scales = list(zip(object_asset_files, object_asset_scales))

        # use fixed seed here to make sure when we restart from checkpoint the distribution of object types is the same
        rng = np.random.default_rng(42)
        rng.shuffle(files_and_scales)

        object_asset_files, object_asset_scales = zip(*files_and_scales)
        need_vhacds = [False] * len(object_asset_files)

        # Hammers
        from dataclasses import dataclass
        @dataclass
        class Hammer:
            file: str
            scale: List[float]
            need_vhacd: bool
            fixed_trajectory: torch.Tensor

        this_dir = Path(__file__).parent
        root_dir = this_dir.parent.parent.parent
        init_state = [0.0, 0, 0.65, 1, 0, 0, 0]
        name_to_hammer_dict = {
            "scanned_hammer_1": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/hammer_1/hammer_1.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "scanned_hammer_2": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/hammer_2/hammer_2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "YcbHammer": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/YcbHammer/model.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cuboidal_hammer": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-3_0-03_0-02_0-03_0-1_0-02_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cylindrical_hammer": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-3_0-015_0-015_0-1_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cuboidal_hammer_1-25x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-375_0-0375_0-025_0-0375_0-125_0-025_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cuboidal_hammer_1-5x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-44999999999999996_0-045_0-03_0-045_0-15000000000000002_0-03_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cuboidal_hammer_1-75x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-525_0-0525_0-035_0-0525_0-17500000000000002_0-035_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cuboidal_hammer_2x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-6_0-06_0-04_0-06_0-2_0-04_0-1_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            # "cuboidal_hammer_4x": Hammer(
            #     file=str(root_dir / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_1-2_0-12_0-08_0-12_0-4_0-08_0-1_0-2.urdf"),
            #     scale=[3.0, 0.5, 0.5],
            #     need_vhacd=False,
            # ),
            "cylindrical_hammer_1-25x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-375_0-01875_0-01875_0-125_0-125_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cylindrical_hammer_1-5x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-44999999999999996_0-0225_0-0225_0-15000000000000002_0-15000000000000002_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cylindrical_hammer_1-75x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-525_0-02625_0-02625_0-17500000000000002_0-17500000000000002_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            "cylindrical_hammer_2x": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-6_0-03_0-03_0-2_0-2_0-2.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=False,
                fixed_trajectory=get_hammer_trajectory(init_state, device=self.device),
            ),
            # "cylindrical_hammer_4x": Hammer(
            #     file=str(root_dir / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_1-2_0-06_0-06_0-4_0-4_0-2.urdf"),
            #     scale=[3.0, 0.5, 0.5],
            #     need_vhacd=False,
            # ),
            "040_large_marker": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/040_large_marker/040_large_marker.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_marker_trajectory(init_state, device=self.device),
            ),
            "whiteboard_eraser": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/whiteboard_eraser/source/model.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_eraser_trajectory(init_state, device=self.device),
            ),
            "phone": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/phone/model.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_phone_trajectory(init_state, device=self.device),
            ),
            "044_flat_screwdriver": Hammer(
                file=str(root_dir / "assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf"),
                scale=[3.0, 0.5, 0.5],
                need_vhacd=True,
                fixed_trajectory=get_screwdriver_trajectory(init_state, device=self.device),
            ),
        }
        for hammer in name_to_hammer_dict.values():
            assert Path(hammer.file).exists(), f"Hammer file {hammer.file} does not exist"

        object_type = self.cfg["env"]["object_type"]
        USE_FIXED_SET_OF_GOAL_STATES = self.cfg["env"]["use_fixed_set_of_goal_states"]
        if object_type == "scanned_hammer_1":
            object_asset_files = [name_to_hammer_dict["scanned_hammer_1"].file]
            object_asset_scales = [name_to_hammer_dict["scanned_hammer_1"].scale]
            need_vhacds = [name_to_hammer_dict["scanned_hammer_1"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["scanned_hammer_1"].fixed_trajectory
        elif object_type == "scanned_hammer_2":
            object_asset_files = [name_to_hammer_dict["scanned_hammer_2"].file]
            object_asset_scales = [name_to_hammer_dict["scanned_hammer_2"].scale]
            need_vhacds = [name_to_hammer_dict["scanned_hammer_2"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["scanned_hammer_2"].fixed_trajectory
        elif object_type == "YcbHammer":
            object_asset_files = [name_to_hammer_dict["YcbHammer"].file]
            object_asset_scales = [name_to_hammer_dict["YcbHammer"].scale]
            need_vhacds = [name_to_hammer_dict["YcbHammer"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["YcbHammer"].fixed_trajectory
        elif object_type == "cuboidal_hammer":
            object_asset_files = [name_to_hammer_dict["cuboidal_hammer"].file]
            object_asset_scales = [name_to_hammer_dict["cuboidal_hammer"].scale]
            need_vhacds = [name_to_hammer_dict["cuboidal_hammer"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cuboidal_hammer"].fixed_trajectory
        elif object_type == "cylindrical_hammer":
            object_asset_files = [name_to_hammer_dict["cylindrical_hammer"].file]
            object_asset_scales = [name_to_hammer_dict["cylindrical_hammer"].scale]
            need_vhacds = [name_to_hammer_dict["cylindrical_hammer"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cylindrical_hammer"].fixed_trajectory
        elif object_type == "cuboidal_hammer_2x":
            object_asset_files = [name_to_hammer_dict["cuboidal_hammer_2x"].file]
            object_asset_scales = [name_to_hammer_dict["cuboidal_hammer_2x"].scale]
            need_vhacds = [name_to_hammer_dict["cuboidal_hammer_2x"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cuboidal_hammer_2x"].fixed_trajectory
        elif object_type == "cuboidal_hammer_4x":
            object_asset_files = [name_to_hammer_dict["cuboidal_hammer_4x"].file]
            object_asset_scales = [name_to_hammer_dict["cuboidal_hammer_4x"].scale]
            need_vhacds = [name_to_hammer_dict["cuboidal_hammer_4x"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cuboidal_hammer_4x"].fixed_trajectory
        elif object_type == "cylindrical_hammer_2x":
            object_asset_files = [name_to_hammer_dict["cylindrical_hammer_2x"].file]
            object_asset_scales = [name_to_hammer_dict["cylindrical_hammer_2x"].scale]
            need_vhacds = [name_to_hammer_dict["cylindrical_hammer_2x"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cylindrical_hammer_2x"].fixed_trajectory
        elif object_type == "cylindrical_hammer_4x":
            object_asset_files = [name_to_hammer_dict["cylindrical_hammer_4x"].file]
            object_asset_scales = [name_to_hammer_dict["cylindrical_hammer_4x"].scale]
            need_vhacds = [name_to_hammer_dict["cylindrical_hammer_4x"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cylindrical_hammer_4x"].fixed_trajectory
        elif object_type == "040_large_marker":
            object_asset_files = [name_to_hammer_dict["040_large_marker"].file]
            object_asset_scales = [name_to_hammer_dict["040_large_marker"].scale]
            need_vhacds = [name_to_hammer_dict["040_large_marker"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["040_large_marker"].fixed_trajectory
        elif object_type == "whiteboard_eraser":
            object_asset_files = [name_to_hammer_dict["whiteboard_eraser"].file]
            object_asset_scales = [name_to_hammer_dict["whiteboard_eraser"].scale]
            need_vhacds = [name_to_hammer_dict["whiteboard_eraser"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["whiteboard_eraser"].fixed_trajectory
        elif object_type == "phone":
            object_asset_files = [name_to_hammer_dict["phone"].file]
            object_asset_scales = [name_to_hammer_dict["phone"].scale]
            need_vhacds = [name_to_hammer_dict["phone"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["phone"].fixed_trajectory
        elif object_type == "screwdriver":
            object_asset_files = [name_to_hammer_dict["screwdriver"].file]
            object_asset_scales = [name_to_hammer_dict["screwdriver"].scale]
            need_vhacds = [name_to_hammer_dict["screwdriver"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["screwdriver"].fixed_trajectory
        elif object_type == "044_flat_screwdriver":
            object_asset_files = [name_to_hammer_dict["044_flat_screwdriver"].file]
            object_asset_scales = [name_to_hammer_dict["044_flat_screwdriver"].scale]
            need_vhacds = [name_to_hammer_dict["044_flat_screwdriver"].need_vhacd]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["044_flat_screwdriver"].fixed_trajectory
        elif object_type == "all_hammers":
            object_asset_files = [hammer.file for hammer in name_to_hammer_dict.values()]
            object_asset_scales = [hammer.scale for hammer in name_to_hammer_dict.values()]
            need_vhacds = [hammer.need_vhacd for hammer in name_to_hammer_dict.values()]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = [hammer.fixed_trajectory for hammer in name_to_hammer_dict.values()]
        elif object_type == "all_cuboidal_hammers":
            cuboidal_hammer_names = ["cuboidal_hammer", "cuboidal_hammer_1-25x", "cuboidal_hammer_1-5x", "cuboidal_hammer_1-75x", "cuboidal_hammer_2x"]
            object_asset_files = [name_to_hammer_dict[name].file for name in cuboidal_hammer_names]
            object_asset_scales = [name_to_hammer_dict[name].scale for name in cuboidal_hammer_names]
            need_vhacds = [name_to_hammer_dict[name].need_vhacd for name in cuboidal_hammer_names]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cuboidal_hammer"].fixed_trajectory
        elif object_type == "all_cylindrical_hammers":
            cylindrical_hammer_names = ["cylindrical_hammer", "cylindrical_hammer_1-25x", "cylindrical_hammer_1-5x", "cylindrical_hammer_1-75x", "cylindrical_hammer_2x"]
            object_asset_files = [name_to_hammer_dict[name].file for name in cylindrical_hammer_names]
            object_asset_scales = [name_to_hammer_dict[name].scale for name in cylindrical_hammer_names]
            need_vhacds = [name_to_hammer_dict[name].need_vhacd for name in cylindrical_hammer_names]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cylindrical_hammer"].fixed_trajectory
        elif object_type == "all_cuboidal_and_cylindrical_hammers":
            cuboidal_and_cylindrical_hammer_names = ["cuboidal_hammer", "cuboidal_hammer_1-25x", "cuboidal_hammer_1-5x", "cuboidal_hammer_1-75x", "cuboidal_hammer_2x", "cylindrical_hammer", "cylindrical_hammer_1-25x", "cylindrical_hammer_1-5x", "cylindrical_hammer_1-75x", "cylindrical_hammer_2x"]
            object_asset_files = [name_to_hammer_dict[name].file for name in cuboidal_and_cylindrical_hammer_names]
            object_asset_scales = [name_to_hammer_dict[name].scale for name in cuboidal_and_cylindrical_hammer_names]
            need_vhacds = [name_to_hammer_dict[name].need_vhacd for name in cuboidal_and_cylindrical_hammer_names]
            if USE_FIXED_SET_OF_GOAL_STATES:
                self.trajectory_states = name_to_hammer_dict["cuboidal_hammer"].fixed_trajectory
        elif object_type == "cuboid":
            # Use what was already used before
            pass
        elif object_type == "tyler_cuboid_cylinder":
            object_asset_files, object_asset_scales, need_vhacds = self._tyler_cuboid_cylinder(
                str(Path(tmp_assets_dir) / "tyler_cuboid_cylinder"),
            )
        else:
            raise ValueError(f"Unknown object type: {object_type}")
        if USE_FIXED_SET_OF_GOAL_STATES:
            self.max_consecutive_successes = len(self.trajectory_states)

        return object_asset_files, object_asset_scales, need_vhacds

    def _load_main_object_asset(self):
        """Load manipulated object and goal assets."""
        object_assets = []
        for object_asset_file, need_vhacd in zip(self.object_asset_files, self.object_need_vhacds):
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.vhacd_enabled = need_vhacd

            object_asset_dir = os.path.dirname(object_asset_file)
            object_asset_fname = os.path.basename(object_asset_file)

            object_asset_ = self.gym.load_asset(self.sim, object_asset_dir, object_asset_fname, object_asset_options)
            object_assets.append(object_asset_)
        object_rb_count = self.gym.get_asset_rigid_body_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        object_shapes_count = self.gym.get_asset_rigid_shape_count(
            object_assets[0]
        )  # assuming all of them have the same rb count
        return object_assets, object_rb_count, object_shapes_count

    def _load_additional_assets(self, object_asset_root, arm_pose):
        """
        returns: tuple (num_rigid_bodies, num_shapes)
        """
        return 0, 0

    def _create_additional_objects(self, env_ptr, env_idx, object_asset_idx):
        pass

    def _after_envs_created(self):
        pass

    def _extra_reset_rules(self, resets):
        return resets

    def _reset_target(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        raise NotImplementedError()

    def _extra_object_indices(self, env_ids: Tensor) -> List[Tensor]:
        return []

    def _extra_curriculum(self):
        pass

    # AllegroKukaBase implementation
    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return dict(
            success_tolerance=self.success_tolerance,
            prev_episode_successes=self.prev_episode_successes,
            prev_episode_true_objective=self.prev_episode_true_objective,
            dof_state=self.dof_state,
            root_state_tensor=self.root_state_tensor,
            rigid_body_states=self.rigid_body_states,
            successes=self.successes,
            true_objective=self.true_objective,
            near_goal_steps=self.near_goal_steps,
            lifted_object=self.lifted_object,
            closest_keypoint_max_dist=self.closest_keypoint_max_dist,
            closest_fingertip_dist=self.closest_fingertip_dist,
            furthest_hand_dist=self.furthest_hand_dist,
            prev_targets=self.prev_targets,
            cur_targets=self.cur_targets,
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            reset_goal_buf=self.reset_goal_buf,
            obj_keypoint_pos=self.obj_keypoint_pos,
            goal_keypoint_pos=self.goal_keypoint_pos,
            rewards_episode=self.rewards_episode,
            last_curriculum_update=self.last_curriculum_update,
            rb_forces=self.rb_forces,
            random_force_prob=self.random_force_prob,
            goal_states=self.goal_states,
            goal_init_state=self.goal_init_state,
            object_init_state=self.object_init_state,
            prev_total_episode_closest_keypoint_max_dist=self.prev_total_episode_closest_keypoint_max_dist,
            total_episode_closest_keypoint_max_dist=self.total_episode_closest_keypoint_max_dist,
            prev_episode_closest_keypoint_max_dist=self.prev_episode_closest_keypoint_max_dist,
            frame_since_restart=self.frame_since_restart,
        )

    def set_env_state(self, env_state):
        if env_state is None:
            return
        
        rewards_episode = env_state.get("rewards_episode", None)
        if rewards_episode is not None:
            for key in rewards_episode.keys():
                if key in self.rewards_episode and self.rewards_episode[key].shape == rewards_episode[key].shape:
                    self.rewards_episode[key].copy_(rewards_episode[key])
        del env_state["rewards_episode"]

        for key in self.get_env_state().keys():
            value = env_state.get(key, None)
            if value is None:
                continue

            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
            if isinstance(value, torch.Tensor) and self.__dict__[key].shape != value.shape:
                print("Skipping loading env state value", key, "because of shape mismatch")
                continue
            
            if isinstance(value, torch.Tensor):
                self.__dict__[key].copy_(value)
            else:
                self.__dict__[key] = value
            print(f"Loaded env state value {key}:{value}")
        
        print(self._extra_object_indices(None)[0][0].shape)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, : self.num_hand_arm_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]
        
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device), tensor_reset=False)
        self.set_actor_root_state_tensor_indexed()
        print(f"Success tolerance value after loading from checkpoint: {self.success_tolerance}")

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.control_dt = self.dt * self.control_freq_inv
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2 (same as in allegro_hand.py)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _box_asset_files_and_scales(self, object_assets_root, generated_assets_dir):
        files = []
        scales = []

        try:
            filenames = os.listdir(generated_assets_dir)
            for fname in filenames:
                if fname.endswith(".urdf"):
                    os.remove(join(generated_assets_dir, fname))
        except Exception as exc:
            print(f"Exception {exc} while removing older procedurally-generated urdf assets")

        objects_rel_path = os.path.dirname(self.asset_files_dict[self.object_type])
        objects_dir = join(object_assets_root, objects_rel_path)
        base_mesh = join(objects_dir, "meshes", "cube_multicolor.obj")

        generate_default_cube(generated_assets_dir, base_mesh, self.object_base_size)

        if self.with_small_cuboids:
            generate_small_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if self.with_big_cuboids:
            generate_big_cuboids(generated_assets_dir, base_mesh, self.object_base_size)
        if self.with_sticks:
            generate_sticks(generated_assets_dir, base_mesh, self.object_base_size)

        filenames = os.listdir(generated_assets_dir)
        filenames = sorted(filenames)

        for fname in filenames:
            if fname.endswith(".urdf"):
                scale_tokens = os.path.splitext(fname)[0].split("_")[2:]
                files.append(join(generated_assets_dir, fname))
                scales.append([float(scale_token) / 100 for scale_token in scale_tokens])

        return files, scales

    def _tyler_cuboid_cylinder(self, generated_assets_dir):
        if not os.path.exists(generated_assets_dir):
            os.makedirs(generated_assets_dir)

        try:
            filenames = os.listdir(generated_assets_dir)
            for fname in filenames:
                if fname.endswith(".urdf"):
                    os.remove(join(generated_assets_dir, fname))
        except Exception as exc:
            print(f"Exception {exc} while removing older procedurally-generated urdf assets")

        NUM_CUBOIDS = self.cfg["env"]["tyler_num_cuboids"]
        CUBOID_X_MIN, CUBOID_X_MAX = 0.1, 0.4  # Length
        CUBOID_Y_MIN, CUBOID_Y_MAX = 0.02, 0.1  # Width
        CUBOID_Z_MIN, CUBOID_Z_MAX = 0.02, 0.1  # Thickness
        cuboid_x_lengths = np.random.uniform(CUBOID_X_MIN, CUBOID_X_MAX, size=NUM_CUBOIDS)
        cuboid_y_lengths = np.random.uniform(CUBOID_Y_MIN, CUBOID_Y_MAX, size=NUM_CUBOIDS)
        cuboid_z_lengths = np.random.uniform(CUBOID_Z_MIN, CUBOID_Z_MAX, size=NUM_CUBOIDS)
        cuboid_scales = np.stack([cuboid_x_lengths, cuboid_y_lengths, cuboid_z_lengths], axis=1).tolist()
        if self.cfg["env"]["tyler_randomize_com"]:
            CUBOID_COM_X_RANGE = 0.1
            CUBOID_COM_Y_RANGE = 0.02
            CUBOID_COM_Z_RANGE = 0.02
        else:
            CUBOID_COM_X_RANGE = 0
            CUBOID_COM_Y_RANGE = 0
            CUBOID_COM_Z_RANGE = 0
        cuboid_coms = np.stack([
            np.random.uniform(-CUBOID_COM_X_RANGE, CUBOID_COM_X_RANGE, size=NUM_CUBOIDS),
            np.random.uniform(-CUBOID_COM_Y_RANGE, CUBOID_COM_Y_RANGE, size=NUM_CUBOIDS),
            np.random.uniform(-CUBOID_COM_Z_RANGE, CUBOID_COM_Z_RANGE, size=NUM_CUBOIDS),
        ], axis=1).tolist()

        cuboid_files = [
            self.generate_cuboid_urdf(
                filepath=join(generated_assets_dir, f"{idx:03d}_cuboid_{x_scale}_{y_scale}_{z_scale}".replace(".", "-") + ".urdf"),
                x_scale=x_scale,
                y_scale=y_scale,
                z_scale=z_scale,
                com_x=cuboid_coms[idx][0],
                com_y=cuboid_coms[idx][1],
                com_z=cuboid_coms[idx][2],
            )
            for idx, (x_scale, y_scale, z_scale) in enumerate(cuboid_scales)
        ]

        NUM_CYLINDERS = self.cfg["env"]["tyler_num_cylinders"]
        CYLINDER_HEIGHT_MIN, CYLINDER_HEIGHT_MAX = 0.1, 0.4  # Length
        CYLINDER_DIAMETER_MIN, CYLINDER_DIAMETER_MAX = 0.02, 0.1  # Diameter
        cylinder_diameters = np.random.uniform(CYLINDER_DIAMETER_MIN, CYLINDER_DIAMETER_MAX, size=NUM_CYLINDERS)
        cylinder_heights = np.random.uniform(CYLINDER_HEIGHT_MIN, CYLINDER_HEIGHT_MAX, size=NUM_CYLINDERS)
        cylinder_scales = np.stack([cylinder_heights, cylinder_diameters, cylinder_diameters], axis=1).tolist()
        if self.cfg["env"]["tyler_randomize_com"]:
            CYLINDER_COM_X_RANGE = 0.1
            CYLINDER_COM_Y_RANGE = 0.02
            CYLINDER_COM_Z_RANGE = 0.02
        else:
            CYLINDER_COM_X_RANGE = 0
            CYLINDER_COM_Y_RANGE = 0
            CYLINDER_COM_Z_RANGE = 0
        cylinder_coms = np.stack([
            np.random.uniform(-CYLINDER_COM_X_RANGE, CYLINDER_COM_X_RANGE, size=NUM_CYLINDERS),
            np.random.uniform(-CYLINDER_COM_Y_RANGE, CYLINDER_COM_Y_RANGE, size=NUM_CYLINDERS),
            np.random.uniform(-CYLINDER_COM_Z_RANGE, CYLINDER_COM_Z_RANGE, size=NUM_CYLINDERS),
        ], axis=1).tolist()
        cylinder_files = [
            self.generate_cylinder_urdf(
                filepath=join(generated_assets_dir, f"{idx:03d}_cylinder_{height}_{diameter}_{diameter}".replace(".", "-") + ".urdf"),
                height=height,
                radius=diameter / 2,
                com_x=cylinder_coms[idx][0],
                com_y=cylinder_coms[idx][1],
                com_z=cylinder_coms[idx][2],
            )
            for idx, (height, diameter, _) in enumerate(cylinder_scales)
        ]

        RESCALE_FACTOR = 10  # Rescale to try to stay close to 1.0 in magnitude so roughly 0.1 to 4.0
        files = cuboid_files + cylinder_files
        scales = (
            [(x * RESCALE_FACTOR, y * RESCALE_FACTOR, z * RESCALE_FACTOR) for (x, y, z) in cuboid_scales]
            + [(x * RESCALE_FACTOR, y * RESCALE_FACTOR, z * RESCALE_FACTOR) for (x, y, z) in cylinder_scales]
        )
        need_vhacds = [False] * len(files)
        return files, scales, need_vhacds

    @staticmethod
    def generate_cuboid_urdf(filepath, x_scale, y_scale, z_scale, com_x, com_y, com_z):
        urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">

  <link name="cuboid">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{x_scale} {y_scale} {z_scale}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{x_scale} {y_scale} {z_scale}"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>
"""
        with open(filepath, "w") as f:
            f.write(urdf)
        print(f"✅ URDF written to {filepath}")
        return filepath

    @staticmethod
    def generate_cylinder_urdf(filepath, height, radius, com_x, com_y, com_z):
        urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">

  <link name="cylinder">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder radius="{radius}" length="{height}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder radius="{radius}" length="{height}"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="{com_x} {com_y} {com_z}" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>
"""
        with open(filepath, "w") as f:
            f.write(urdf)
        print(f"✅ URDF written to {filepath}")
        return filepath

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")

        object_asset_root = asset_root
        tmp_assets_dir = tempfile.TemporaryDirectory()
        self.object_asset_files, self.object_asset_scales, self.object_need_vhacds = self._main_object_assets_and_scales(
            object_asset_root, tmp_assets_dir.name
        )

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        print(f"Loading asset {self.hand_arm_asset_file} from {asset_root}")
        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)
        print(f"Loaded asset {allegro_kuka_asset}")

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Number of DOFs in asset {allegro_kuka_asset} is {num_hand_arm_dofs}, but {self.num_hand_arm_dofs} was expected"

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes

        allegro_rigid_body_names = [
            self.gym.get_asset_rigid_body_name(allegro_kuka_asset, i) for i in range(self.num_hand_arm_bodies)
        ]
        print(f"Allegro num rigid bodies: {self.num_hand_arm_bodies}")
        print(f"Allegro rigid bodies: {allegro_rigid_body_names}")

        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.allegro_sensors = []
        allegro_sensor_pose = gymapi.Transform()

        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])

        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        allegro_pose = gymapi.Transform()
        allegro_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        allegro_pose.r = gymapi.Quat(0, 0, 0, 1)

        object_assets, object_rb_count, object_shapes_count = self._load_main_object_asset()
        max_agg_bodies += object_rb_count
        max_agg_shapes += object_shapes_count

        # load auxiliary objects
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = False
        table_asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, self.asset_files_dict["table"], table_asset_options)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3()
        table_pose.p.x = allegro_pose.p.x
        table_pose_dy, table_pose_dz = -0.8, 0.38
        table_pose.p.y = allegro_pose.p.y + table_pose_dy
        table_pose.p.z = allegro_pose.p.z + table_pose_dz

        table_rb_count = self.gym.get_asset_rigid_body_count(table_asset)
        table_shapes_count = self.gym.get_asset_rigid_shape_count(table_asset)
        max_agg_bodies += table_rb_count
        max_agg_shapes += table_shapes_count

        additional_rb, additional_shapes = self._load_additional_assets(object_asset_root, allegro_pose)
        max_agg_bodies += additional_rb
        max_agg_shapes += additional_shapes

        # set up object and goal positions
        self.object_start_pose = self._object_start_pose(allegro_pose, table_pose_dy, table_pose_dz)

        self.allegro_hands = []
        self.envs = []
        if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robots = []

        object_init_state = []
        
        self.rigid_body_name_to_idx = {}

        self.allegro_hand_indices = []
        if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = []
        object_indices = []
        table_indices = []
        object_scales = []
        object_keypoint_offsets = []

        # Sanity checks
        body_names = self.gym.get_asset_rigid_body_names(allegro_kuka_asset)
        for name in self.allegro_fingertips:
            assert name in body_names, f"Finger {name} not found in asset {allegro_kuka_asset}"
        has_iiwa14 = "iiwa14_link_7" in body_names
        has_iiwa7 = "iiwa7_link_7" in body_names
        if (has_iiwa14 and has_iiwa7) or (not has_iiwa14 and not has_iiwa7):
            raise ValueError(f"Either iiwa14 or iiwa7 must be in the asset {allegro_kuka_asset}, but not both, has_iiwa14: {has_iiwa14}, has_iiwa7: {has_iiwa7}, body_names: {body_names}")

        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        if has_iiwa14:
            self.robot_name = "iiwa14"
            self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa14_link_7")
        elif has_iiwa7:
            self.robot_name = "iiwa7"
            self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")
        else:
            raise ValueError(f"Either iiwa14 or iiwa7 must be in the asset {allegro_kuka_asset}, but not both, has_iiwa14: {has_iiwa14}, has_iiwa7: {has_iiwa7}, body_names: {body_names}")

        # this rely on the fact that objects are added right after the arms in terms of create_actor()
        self.object_rb_handles = list(range(self.num_hand_arm_bodies, self.num_hand_arm_bodies + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            allegro_actor = self.gym.create_actor(env_ptr, allegro_kuka_asset, allegro_pose, "allegro", i, -1, 0)

            # HACK: Ovewrite
            # self.dof_params.allegro_stiffness = 80
            # self.dof_params.allegro_stiffness = 160
            self.dof_params.allegro_stiffness = 4000
            self.dof_params.allegro_damping = 200
            # self.dof_params.allegro_effort = 350
            # self.dof_params.kuka_stiffness = [600, 600, 500, 400, 200, 200, 200]
            self.dof_params.kuka_stiffness = [300, 300, 300, 300, 300, 300, 300]
            self.dof_params.kuka_damping = [20, 20, 20, 20, 20, 20, 20]
            # self.dof_params.kuka_damping = [20, 20, 17, 14, 7, 7, 7]
            # self.dof_params.kuka_damping = [20, 20, 17, 10, 5, 5, 5]
            populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)
            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, allegro_actor):
                self.rigid_body_name_to_idx["allegro/" + name] = self.gym.find_actor_rigid_body_index(env_ptr, allegro_actor, name, gymapi.DOMAIN_ENV)

            if self.obs_type == "full_state":
                if self.with_fingertip_force_sensors:
                    for ft_handle in self.allegro_fingertip_handles:
                        env_sensors = [self.gym.create_force_sensor(env_ptr, ft_handle, allegro_sensor_pose)]
                        self.allegro_sensors.append(env_sensors)

                if self.with_dof_force_sensors:
                    self.gym.enable_actor_dof_force_sensors(env_ptr, allegro_actor)

            if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                blue_robot_actor = self.gym.create_actor(
                    env_ptr,
                    allegro_kuka_asset,
                    allegro_pose,
                    "blue_robot",
                    i + self.num_envs * 2,
                    -1,
                    0,
                )
                self.gym.set_actor_dof_properties(
                    env_ptr,
                    blue_robot_actor,
                    allegro_hand_dof_props,
                )
                self.blue_robots.append(blue_robot_actor)
                BLUE = (0, 0, 1)
                self._set_actor_color(env_ptr, blue_robot_actor, BLUE)

                blue_robot_idx = self.gym.get_actor_index(env_ptr, blue_robot_actor, gymapi.DOMAIN_SIM)
                self.blue_robot_indices.append(blue_robot_idx)

            # add object
            object_asset_idx = i % len(object_assets)
            object_asset = object_assets[object_asset_idx]

            object_handle = self.gym.create_actor(env_ptr, object_asset, self.object_start_pose, "object", i, 0, 0)
            object_init_state.append(
                [
                    self.object_start_pose.p.x,
                    self.object_start_pose.p.y,
                    self.object_start_pose.p.z,
                    self.object_start_pose.r.x,
                    self.object_start_pose.r.y,
                    self.object_start_pose.r.z,
                    self.object_start_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            object_indices.append(object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, object_handle):
                self.rigid_body_name_to_idx["object/" + name] = self.gym.find_actor_rigid_body_index(env_ptr, object_handle, name, gymapi.DOMAIN_ENV)

            object_scale = self.object_asset_scales[object_asset_idx]
            object_scales.append(object_scale)
            object_offsets = []
            for keypoint in self.keypoints_offsets:
                keypoint = copy(keypoint)
                for coord_idx in range(3):
                    keypoint[coord_idx] *= object_scale[coord_idx] * self.object_base_size * self.keypoint_scale / 2
                object_offsets.append(keypoint)

            object_keypoint_offsets.append(object_offsets)

            # table object
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table_object", i, 0, 0)
            table_object_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            table_indices.append(table_object_idx)
            for name in self.gym.get_actor_rigid_body_names(env_ptr, table_handle):
                self.rigid_body_name_to_idx["table/" + name] = self.gym.find_actor_rigid_body_index(env_ptr, table_handle, name, gymapi.DOMAIN_ENV)

            # task-specific objects (i.e. goal object for reorientation task)
            self._create_additional_objects(env_ptr, env_idx=i, object_asset_idx=object_asset_idx)

            self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13
        )
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()

        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(table_indices, dtype=torch.long, device=self.device)
        if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.blue_robot_indices = to_torch(self.blue_robot_indices, dtype=torch.long, device=self.device)

        self.object_scales = to_torch(object_scales, dtype=torch.float, device=self.device)
        self.object_keypoint_offsets = to_torch(object_keypoint_offsets, dtype=torch.float, device=self.device)

        self.joint_names = self.gym.get_actor_joint_names(env_ptr, allegro_actor)
        props = self.gym.get_actor_dof_properties(env_ptr, allegro_actor)
        self.joint_lower_limits = props["lower"]
        self.joint_upper_limits = props["upper"]

        print(f"Allegro joint names: {self.joint_names}")

        self._after_envs_created()

        try:
            # by this point we don't need the temporary folder for procedurally generated assets
            tmp_assets_dir.cleanup()
        except Exception:
            pass

    def _set_actor_color(self, env, actor, color: Tuple[float, float, float]) -> None:
        for rigid_body_idx in range(self.gym.get_actor_rigid_body_count(env, actor)):
            self.gym.set_rigid_body_color(
                env,
                actor,
                rigid_body_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(*color),
            )

    def _distance_delta_rewards(self, lifted_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Rewards for fingertips approaching the object or penalty for hand getting further away from the object."""
        # this is positive if we got closer, negative if we're further away than the closest we've gotten
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        # update the values if finger tips got closer to the object
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        # again, positive is closer, negative is further away
        # here we use index of the 1st finger, when the distance is large it doesn't matter which one we use
        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        # update the values if finger tips got further away from the object
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])

        # clip between zero and +inf to turn deltas into rewards
        fingertip_deltas = torch.clip(fingertip_deltas_closest, 0, 10)
        fingertip_deltas *= self.finger_rew_coeffs
        fingertip_delta_rew = torch.sum(fingertip_deltas, dim=-1)
        # add this reward only before the object is lifted off the table
        # after this, we should be guided only by keypoint and bonus rewards
        fingertip_delta_rew *= ~lifted_object

        # clip between zero and -inf to turn deltas into penalties
        hand_delta_penalty = torch.clip(hand_deltas_furthest, -10, 0)
        hand_delta_penalty *= ~lifted_object
        # multiply by the number of fingers so two rewards are on the same scale
        hand_delta_penalty *= self.num_allegro_fingertips

        return fingertip_delta_rew, hand_delta_penalty

    def _lifting_reward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""

        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_state[:, 2]
        lifting_rew = torch.clip(z_lift, 0, 0.5)

        # this flag tells us if we lifted an object above a certain height compared to the initial position
        lifted_object = (z_lift > self.lifting_bonus_threshold) | self.lifted_object

        # Since we stop rewarding the agent for height after the object is lifted, we should give it large positive reward
        # to compensate for "lost" opportunity to get more lifting reward for sitting just below the threshold.
        # This bonus depends on the max lifting reward (lifting reward coeff * threshold) and the discount factor
        # (i.e. the effective future horizon for the agent)
        # For threshold 0.15, lifting reward coeff = 3 and gamma 0.995 (effective horizon ~500 steps)
        # a value of 300 for the bonus reward seems reasonable
        just_lifted_above_threshold = lifted_object & ~self.lifted_object
        lift_bonus_rew = self.lifting_bonus * just_lifted_above_threshold

        # stop giving lifting reward once we crossed the threshold - now the agent can focus entirely on the
        # keypoint reward
        lifting_rew *= ~lifted_object

        # update the flag that describes whether we lifted an object above the table or not
        self.lifted_object = lifted_object
        return lifting_rew, lift_bonus_rew, lifted_object

    def _keypoint_reward(self, lifted_object: Tensor) -> Tensor:
        # this is positive if we got closer, negative if we're further away
        max_keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist

        # update the values if we got closer to the target
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)

        # clip between zero and +inf to turn deltas into rewards
        max_keypoint_deltas = torch.clip(max_keypoint_deltas, 0, 100)

        # administer reward only when we already lifted an object from the table
        # to prevent the situation where the agent just rolls it around the table
        keypoint_rew = max_keypoint_deltas * lifted_object

        return keypoint_rew

    def _action_penalties(self) -> Tuple[Tensor, Tensor]:
        kuka_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 0:7]), dim=-1) * self.kuka_actions_penalty_scale
        )
        allegro_actions_penalty = (
            torch.sum(torch.abs(self.arm_hand_dof_vel[..., 7 : self.num_hand_arm_dofs]), dim=-1)
            * self.allegro_actions_penalty_scale
        )

        return -1 * kuka_actions_penalty, -1 * allegro_actions_penalty

    def _compute_resets(self, is_success):
        resets = torch.where(self.object_pos[:, 2] < 0.1, torch.ones_like(self.reset_buf), self.reset_buf)  # fall
        if self.max_consecutive_successes > 0:
            # Reset progress buffer if max_consecutive_successes > 0
            self.progress_buf = torch.where(is_success > 0, torch.zeros_like(self.progress_buf), self.progress_buf)
            resets = torch.where(self.successes >= self.max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(resets), resets)
        resets = self._extra_reset_rules(resets)
        return resets

    def _true_objective(self):
        raise NotImplementedError()

    def compute_kuka_reward(self) -> Tuple[Tensor, Tensor]:
        lifting_rew, lift_bonus_rew, lifted_object = self._lifting_reward()
        fingertip_delta_rew, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_rew = self._keypoint_reward(lifted_object)

        keypoint_success_tolerance = self.success_tolerance * self.keypoint_scale

        # noinspection PyTypeChecker
        near_goal: Tensor = self.keypoints_max_dist <= keypoint_success_tolerance
        self.near_goal_steps += near_goal

        is_success = self.near_goal_steps >= self.success_steps
        goal_resets = is_success
        self.successes += is_success

        self.reset_goal_buf[:] = goal_resets

        self.rewards_episode["raw_fingertip_delta_rew"] += fingertip_delta_rew
        self.rewards_episode["raw_hand_delta_penalty"] += hand_delta_penalty
        self.rewards_episode["raw_lifting_rew"] += lifting_rew
        self.rewards_episode["raw_keypoint_rew"] += keypoint_rew

        fingertip_delta_rew *= self.distance_delta_rew_scale
        hand_delta_penalty *= self.distance_delta_rew_scale * 0  # currently disabled
        lifting_rew *= self.lifting_rew_scale
        keypoint_rew *= self.keypoint_rew_scale

        kuka_actions_penalty, allegro_actions_penalty = self._action_penalties()

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        # We spread out the reward over "success_steps"
        bonus_rew = near_goal * (self.reach_goal_bonus / self.success_steps)

        reward = (
            fingertip_delta_rew
            + hand_delta_penalty  # + sign here because hand_delta_penalty is negative
            + lifting_rew
            + lift_bonus_rew
            + keypoint_rew
            + kuka_actions_penalty
            + allegro_actions_penalty
            + bonus_rew
        )

        self.rew_buf[:] = reward

        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        self.extras["successes"] = self.prev_episode_successes
        self.extras["success_ratio"] = self.prev_episode_successes.mean().item() / self.max_consecutive_successes
        self.extras["closest_keypoint_max_dist"] = self.prev_episode_closest_keypoint_max_dist
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective

        # scalars for logging
        # self.extras["true_objective_mean"] = self.true_objective.mean()
        # self.extras["true_objective_min"] = self.true_objective.min()
        # self.extras["true_objective_max"] = self.true_objective.max()

        rewards = [
            (fingertip_delta_rew, "fingertip_delta_rew"),
            (hand_delta_penalty, "hand_delta_penalty"),
            (lifting_rew, "lifting_rew"),
            (lift_bonus_rew, "lift_bonus_rew"),
            (keypoint_rew, "keypoint_rew"),
            (kuka_actions_penalty, "kuka_actions_penalty"),
            (allegro_actions_penalty, "allegro_actions_penalty"),
            (bonus_rew, "bonus_rew"),
        ]

        episode_cumulative = dict()
        for rew_value, rew_name in rewards:
            self.rewards_episode[rew_name] += rew_value
            episode_cumulative[rew_name] = rew_value
        self.extras["rewards_episode"] = self.rewards_episode
        self.extras["episode_cumulative"] = episode_cumulative

        return self.rew_buf, is_success

    def _eval_stats(self, is_success: Tensor) -> None:
        if self.eval_stats:
            frame: int = self.frame_since_restart
            n_frames = torch.empty_like(self.last_success_step).fill_(frame)
            self.success_time = torch.where(is_success, n_frames - self.last_success_step, self.success_time)
            self.last_success_step = torch.where(is_success, n_frames, self.last_success_step)
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
            else:
                avg_time_mean = math.nan

            self.total_resets = self.total_resets + self.reset_buf.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            self.total_num_resets += self.reset_buf

            reset_ids = self.reset_buf.nonzero().squeeze()
            last_successes = self.successes[reset_ids].long()
            self.successes_count[last_successes] += 1

            if frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Average consecutive successes: {self.prev_episode_successes.mean().item():.2f}")
                print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
                print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")
                print(f"Last ep successes: {self.prev_episode_successes.mean().item():.2f}")
                print(f"Last ep true objective: {self.prev_episode_true_objective.mean().item():.2f}")

                self.eval_summaries.add_scalar("last_ep_successes", self.prev_episode_successes.mean().item(), frame)
                self.eval_summaries.add_scalar(
                    "last_ep_true_objective", self.prev_episode_true_objective.mean().item(), frame
                )
                self.eval_summaries.add_scalar(
                    "reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, frame
                )
                self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), frame)

                self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, frame)
                frame_time = self.control_freq_inv * self.dt
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, frame
                )
                self.eval_summaries.add_scalar(
                    "policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), frame
                )
                print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt

                plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()

    def compute_observations(self) -> Tuple[Tensor, int]:
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state":
            if self.with_fingertip_force_sensors:
                self.gym.refresh_force_sensor_tensor(self.sim)
            if self.with_dof_force_sensors:
                self.gym.refresh_dof_force_tensor(self.sim)

        self.object_state = self.root_state_tensor[self.object_indices, 0:13]
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]

        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # HACK: Move offsets down by X along x axis to grab hammer on bottom of handle
        use_hack_object_pos_offset = self.cfg["env"]["use_hack_object_pos_offset"]
        hack_object_pos_offset = self.cfg["env"]["hack_object_pos_offset"]
        if use_hack_object_pos_offset:
            self.object_pos_offset = torch.tensor([
                -hack_object_pos_offset, 0.0, 0.0
            ], device=self.device)[None].repeat_interleave(self.num_envs, dim=0)
            self.object_pos = self.object_pos + quat_rotate(self.object_rot, self.object_pos_offset)
            self.goal_pos = self.goal_pos + quat_rotate(self.goal_rot, self.object_pos_offset)

        self.palm_center_offset = torch.from_numpy(self.palm_offset).to(self.device).repeat((self.num_envs, 1))
        self._palm_state = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:13]
        self._palm_pos = self.rigid_body_states[:, self.allegro_palm_handle][:, 0:3]
        self._palm_rot = self.rigid_body_states[:, self.allegro_palm_handle][:, 3:7]
        self.palm_center_pos = self._palm_pos + quat_rotate(self._palm_rot, self.palm_center_offset)

        self.fingertip_state = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 0:3]
        self.fingertip_rot = self.rigid_body_states[:, self.allegro_fingertip_handles][:, :, 3:7]

        if not isinstance(self.fingertip_offsets, torch.Tensor):
            self.fingertip_offsets = (
                torch.from_numpy(self.fingertip_offsets).to(self.device).repeat((self.num_envs, 1, 1))
            )

        if hasattr(self, "fingertip_pos_rel_object"):
            self.fingertip_pos_rel_object_prev[:, :, :] = self.fingertip_pos_rel_object
        else:
            self.fingertip_pos_rel_object_prev = None

        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos).to(self.device)
        for i in range(self.num_allegro_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets[:, i]
            )

        obj_pos_repeat = self.object_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_pos_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)

        # when episode ends or target changes we reset this to -1, this will initialize it to the actual distance on the 1st frame of the episode
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_allegro_fingertips, 1)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_center_repeat

        if self.fingertip_pos_rel_object_prev is None:
            self.fingertip_pos_rel_object_prev = self.fingertip_pos_rel_object.clone()

        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets[:, i]
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos

        palm_center_repeat = self.palm_center_pos.unsqueeze(1).repeat(1, self.num_keypoints, 1)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_center_repeat

        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)

        # furthest keypoint from the goal
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values

        # this is the closest the keypoint had been to the target in the current episode (for the furthest keypoint of all)
        # make sure we initialize this value before using it for obs or rewards
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )

        if self.obs_type == "full_state":
            full_state_size, reward_obs_ofs = self.compute_full_state(self.obs_buf)
            assert (
                full_state_size == self.full_state_size
            ), f"Expected full state size {self.full_state_size}, actual: {full_state_size}"

            return self.obs_buf, reward_obs_ofs
        else:
            raise ValueError("Unkown observations type!")

    def compute_full_state(self, buf: Tensor) -> Tuple[int, int]:
        num_dofs = self.num_hand_arm_dofs
        ofs = 0

        # dof positions
        buf[:, ofs : ofs + num_dofs] = unscale(
            self.arm_hand_dof_pos[:, :num_dofs],
            self.arm_hand_dof_lower_limits[:num_dofs],
            self.arm_hand_dof_upper_limits[:num_dofs],
        )
        ofs += num_dofs

        # dof velocities
        buf[:, ofs : ofs + num_dofs] = self.arm_hand_dof_vel[:, :num_dofs]
        ofs += num_dofs

        if self.with_dof_force_sensors:
            # dof forces
            buf[:, ofs : ofs + num_dofs] = self.dof_force_tensor[:, :num_dofs]
            ofs += num_dofs

        # palm pos
        buf[:, ofs : ofs + 3] = self.palm_center_pos
        ofs += 3

        # palm rot, linvel, ang vel
        buf[:, ofs : ofs + 4] = self._palm_state[:, 3:7]
        ofs += 4
        buf[:, ofs : ofs + 3] = self._palm_state[:, 7:10] * self.turn_off_palm_vel_obs_scale
        ofs += 3
        buf[:, ofs : ofs + 3] = self._palm_state[:, 10:13] * self.turn_off_palm_vel_obs_scale
        ofs += 3

        # object rot, linvel, ang vel
        buf[:, ofs : ofs + 4] = self.object_state[:, 3:7]
        ofs += 4
        buf[:, ofs : ofs + 3] = self.object_state[:, 7:10] * self.turn_off_object_vel_obs_scale
        ofs += 3
        buf[:, ofs : ofs + 3] = self.object_state[:, 10:13] * self.turn_off_object_vel_obs_scale
        ofs += 3

        # fingertip pos relative to the palm of the hand
        fingertip_rel_pos_size = 3 * self.num_allegro_fingertips
        buf[:, ofs : ofs + fingertip_rel_pos_size] = self.fingertip_pos_rel_palm.reshape(
            self.num_envs, fingertip_rel_pos_size
        )
        ofs += fingertip_rel_pos_size

        # keypoint distances relative to the palm of the hand
        keypoint_rel_pos_size = 3 * self.num_keypoints
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_palm.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        # keypoint distances relative to the goal
        buf[:, ofs : ofs + keypoint_rel_pos_size] = self.keypoints_rel_goal.reshape(
            self.num_envs, keypoint_rel_pos_size
        )
        ofs += keypoint_rel_pos_size

        # object scales
        buf[:, ofs : ofs + 3] = self.object_scales
        ofs += 3

        # closest distance to the furthest keypoint, achieved so far in this episode
        buf[:, ofs : ofs + 1] = self.closest_keypoint_max_dist.unsqueeze(-1) * self.turn_off_extra_obs_scale
        ofs += 1

        # closest distance between a fingertip and an object achieved since last target reset
        # this should help the critic predict the anticipated fingertip reward
        buf[:, ofs : ofs + self.num_allegro_fingertips] = self.closest_fingertip_dist * self.turn_off_extra_obs_scale
        ofs += self.num_allegro_fingertips

        # indicates whether we already lifted the object from the table or not, should help the critic be more accurate
        buf[:, ofs : ofs + 1] = self.lifted_object.unsqueeze(-1) * self.turn_off_extra_obs_scale
        ofs += 1

        # this should help the critic predict the future rewards better and anticipate the episode termination
        buf[:, ofs : ofs + 1] = torch.log(self.progress_buf / 10 + 1).unsqueeze(-1) * self.turn_off_extra_obs_scale
        ofs += 1
        buf[:, ofs : ofs + 1] = torch.log(self.successes + 1).unsqueeze(-1) * self.turn_off_extra_obs_scale
        ofs += 1

        # this is where we will add the reward observation
        reward_obs_ofs = ofs
        ofs += 1

        # Default CHECK_WITH_COMPUTED_OBS = False
        # Set to True to check if the observations are computed correctly
        CHECK_WITH_COMPUTED_OBS = False
        if CHECK_WITH_COMPUTED_OBS:
            import pytorch_kinematics as pk
            # Create chain and palm_serial_chain from URDF
            if not hasattr(self, "chain") or not hasattr(self, "palm_serial_chain"):
                self.chain, self.palm_serial_chain = create_chain_and_serial_chain(device=self.device, robot_name=self.robot_name)

            computed_obs = compute_observation(
                q=self.arm_hand_dof_pos,
                qd=self.arm_hand_dof_vel,
                object_pose=self.object_pose,
                goal_object_pose=self.goal_pose,
                object_scales=self.object_scales,
                chain=self.chain,
                palm_serial_chain=self.palm_serial_chain,
            )

            # Validate
            assert computed_obs.shape == (self.num_envs, len(OBS_NAMES)), f"computed_obs.shape: {computed_obs.shape}, expected: ({self.num_envs}, {len(OBS_NAMES)})"
            assert buf.shape == computed_obs.shape, f"buf.shape: {buf.shape}, expected: {computed_obs.shape}"
            num_errors = 0
            for i, name in enumerate(OBS_NAMES):
                val_orig = buf[0, i].item()
                val_computed = computed_obs[0, i].item()
                print(f"{name}: original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}")
                # Note that there are some reasonably large 2e-3 differences in the palm vel computation
                # Maybe from Jacobian computation being different from the maximal sim computation
                if abs(val_orig - val_computed) > 1e-2:
                    num_errors += 1
                    print("--------------------------------")
                    print(f"Error: {name}: original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}")
                    print("--------------------------------")
            print("="*100)
            print(f"num_errors: {num_errors}")
            print("="*100)
            breakpoint()

        assert ofs == self.full_state_size
        return ofs, reward_obs_ofs

    @property
    def turn_off_palm_vel_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_palm_vel_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_palm_vel_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_palm_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_object_vel_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_object_vel_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_object_vel_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0
        self.extras["turn_off_object_vel_obs_scale"] = scale
        return scale

    @property
    def turn_off_extra_obs_scale(self) -> float:
        # 1 means not turned off
        # 0.5 means half turned off
        # 0 means turned off
        if self.cfg["env"]["turn_off_extra_obs"]:
            scale = 0.0
        elif self.cfg["env"]["turn_off_extra_obs_slowly"]:
            if self.cfg["env"]["use_obs_dropout"]:
                # When curriculum_scale is 0.0, turn_off_extra_obs_scale is 1.0, which means no extra obs are turned off
                # When curriculum_scale is 1.0, turn_off_extra_obs_scale is 0.0, which means all extra obs are turned off
                # Smoothly transition between 1.0 and 0.0
                prob_of_turn_off = self._tyler_curriculum_scale
                scale = 0.0 if random.random() < prob_of_turn_off else 1.0
            else:
                # When curriculum_scale is 0.0, turn_off_extra_obs_scale is 1.0, which means no extra obs are turned off
                # When curriculum_scale is 1.0, turn_off_extra_obs_scale is 0.0, which means all extra obs are turned off
                # Smoothly transition between 1.0 and 0.0
                scale = 1.0 - self._tyler_curriculum_scale
        else:
            scale = 1.0

        self.extras["turn_off_extra_obs_scale"] = scale
        return scale

    def clamp_obs(self, obs_buf: Tensor) -> None:
        if self.clamp_abs_observations > 0:
            obs_buf.clamp_(-self.clamp_abs_observations, self.clamp_abs_observations)

    def get_random_quat(self, env_ids):
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)

        return new_rot

    def reset_target_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        self._reset_target(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)
        
        if tensor_reset:
            self.reset_goal_buf[env_ids] = 0
            self.near_goal_steps[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = self.total_episode_closest_keypoint_max_dist[env_ids]
            self.total_episode_closest_keypoint_max_dist[env_ids] += torch.where(self.closest_keypoint_max_dist[env_ids] > 0, self.closest_keypoint_max_dist[env_ids], torch.zeros_like(self.closest_keypoint_max_dist[env_ids]))
            self.closest_keypoint_max_dist[env_ids] = -1

    def reset_object_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True):
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            obj_indices = self.object_indices[env_ids]

            USE_FIXED_INIT_OBJECT_POSE = self.cfg["env"]["use_fixed_init_object_pose"]

            # reset object
            rand_pos_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 3), device=self.device)
            if USE_FIXED_INIT_OBJECT_POSE:
                rand_pos_floats[:] = 0.0 #HACK
            self.root_state_tensor[obj_indices] = self.object_init_state[env_ids].clone()

            # indices 0..2 correspond to the object position
            self.root_state_tensor[obj_indices, 0:1] = (
                self.object_init_state[env_ids, 0:1] + self.reset_position_noise_x * rand_pos_floats[:, 0:1]
            )
            self.root_state_tensor[obj_indices, 1:2] = (
                self.object_init_state[env_ids, 1:2] + self.reset_position_noise_y * rand_pos_floats[:, 1:2]
            )
            self.root_state_tensor[obj_indices, 2:3] = (
                self.object_init_state[env_ids, 2:3] + self.reset_position_noise_z * rand_pos_floats[:, 2:3]
            )
            new_object_rot = self.get_random_quat(env_ids)
            if USE_FIXED_INIT_OBJECT_POSE:
                new_object_rot[:] = 0.0 #HACK
                new_object_rot[:, -1] = 1.0 #HACK  xyzw

            # indices 3,4,5,6 correspond to the rotation quaternion
            self.root_state_tensor[obj_indices, 3:7] = new_object_rot

            self.root_state_tensor[obj_indices, 7:13] = torch.zeros_like(self.root_state_tensor[obj_indices, 7:13])
        
        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            obj_indices = self.object_indices[env_ids]
            # TODO: Check if last 6 indices are 0 
            rs_ofs = self.root_state_resets.shape[1]
            self.root_state_tensor[obj_indices, :] = self.root_state_resets[reset_buf_idxs[env_ids].cpu(), obj_indices.cpu() % rs_ofs, :].to(self.device)

        # since we reset the object, we also should update distances between fingers and the object
        if tensor_reset:
            self.closest_fingertip_dist[env_ids] = -1
            self.furthest_hand_dist[env_ids] = -1
            self.lifted_object[env_ids] = False
        self.deferred_set_actor_root_state_tensor_indexed([self.object_indices[env_ids]])

    def deferred_set_actor_root_state_tensor_indexed(self, obj_indices: List[Tensor]) -> None:
        self.set_actor_root_state_object_indices.extend(obj_indices)

    def set_actor_root_state_tensor_indexed(self) -> None:
        object_indices: List[Tensor] = self.set_actor_root_state_object_indices
        if not object_indices:
            # nothing to set
            return

        unique_object_indices = torch.unique(torch.cat(object_indices).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(unique_object_indices),
            len(unique_object_indices),
        )

        self.set_actor_root_state_object_indices = []

    def deferred_set_dof_state_tensor_indexed(self, dof_indices: List[Tensor]) -> None:
        self.set_dof_state_object_indices.extend(dof_indices)

    def set_dof_state_tensor_indexed(self) -> None:
        dof_indices: List[Tensor] = self.set_dof_state_object_indices
        if not dof_indices:
            # nothing to set
            return

        unique_dof_indices = torch.unique(torch.cat(dof_indices).to(torch.int32))
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(unique_dof_indices),
            len(unique_dof_indices),
        )

        self.set_dof_state_object_indices = []

    def reset_idx(self, env_ids: Tensor, reset_buf_idxs=None, episode_reset=True, tensor_reset=True) -> None:
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if len(env_ids) == 0:
            return

        if self.randomize and episode_reset:
            self.apply_randomizations(self.randomization_params)

        # randomize start object poses
        self.reset_target_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        # reset rigid body forces
        if tensor_reset:
            self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.reset_object_pose(env_ids, reset_buf_idxs, tensor_reset=tensor_reset)

        hand_indices = self.allegro_hand_indices[env_ids].to(torch.int32)

        # reset random force probabilities
        if tensor_reset:
            self.random_force_prob[env_ids] = torch.exp(
                (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                * torch.rand(len(env_ids), device=self.device)
                + torch.log(self.force_prob_range[1])
            )

        # reset allegro hand
        if len(env_ids) > 0 and reset_buf_idxs is None and tensor_reset:
            delta_max = self.arm_hand_dof_upper_limits - self.hand_arm_default_dof_pos
            delta_min = self.arm_hand_dof_lower_limits - self.hand_arm_default_dof_pos

            rand_dof_floats = torch_rand_float(0.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)

            rand_delta = delta_min + (delta_max - delta_min) * rand_dof_floats

            noise_coeff = torch.zeros_like(self.hand_arm_default_dof_pos, device=self.device)

            noise_coeff[0:7] = self.reset_dof_pos_noise_arm
            noise_coeff[7 : self.num_hand_arm_dofs] = self.reset_dof_pos_noise_fingers

            allegro_pos = self.hand_arm_default_dof_pos + noise_coeff * rand_delta

            self.arm_hand_dof_pos[env_ids, :] = allegro_pos
            if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
                self.blue_robot_arm_hand_dof_pos[env_ids, :] = allegro_pos.clone()
                self.blue_robot_arm_hand_dof_vel[env_ids, :] = 0.0

            rand_vel_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)
            self.arm_hand_dof_vel[env_ids, :] = self.reset_dof_vel_noise * rand_vel_floats
            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
        
        if len(env_ids) > 0 and reset_buf_idxs is not None and tensor_reset:
            self.arm_hand_dof_pos[env_ids, :] = self.dof_resets[reset_buf_idxs[env_ids].cpu(), :, 0].to(self.device)
            self.arm_hand_dof_vel[env_ids, :] = self.dof_resets[reset_buf_idxs[env_ids].cpu(), :, 1].to(self.device)
            allegro_pos = self.arm_hand_dof_pos[env_ids, : self.num_hand_arm_dofs]

            self.prev_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos
            self.cur_targets[env_ids, : self.num_hand_arm_dofs] = allegro_pos


        if self.should_load_initial_states:
            if len(env_ids) > self.num_initial_states:
                print(f"Not enough initial states to load {len(env_ids)}/{self.num_initial_states}...")
            else:
                if self.initial_state_idx + len(env_ids) > self.num_initial_states:
                    self.initial_state_idx = 0

                dof_states_to_load = self.initial_dof_state_tensors[
                    self.initial_state_idx : self.initial_state_idx + len(env_ids)
                ]
                self.dof_state.reshape([self.num_envs, -1, *self.dof_state.shape[1:]])[
                    env_ids
                ] = dof_states_to_load.clone()
                root_state_tensors_to_load = self.initial_root_state_tensors[
                    self.initial_state_idx : self.initial_state_idx + len(env_ids)
                ]
                cube_object_idx = self.object_indices[0]
                self.root_state_tensor.reshape([self.num_envs, -1, *self.root_state_tensor.shape[1:]])[
                    env_ids, cube_object_idx
                ] = root_state_tensors_to_load[:, cube_object_idx].clone()

                self.initial_state_idx += len(env_ids)

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.prev_targets), gymtorch.unwrap_tensor(hand_indices), len(env_ids)
        )

        self.deferred_set_dof_state_tensor_indexed([hand_indices])
        self.deferred_set_actor_root_state_tensor_indexed(self._extra_object_indices(env_ids))

        if episode_reset and tensor_reset:
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

            self.prev_episode_successes[env_ids] = self.successes[env_ids]
            self.successes[env_ids] = 0

            self.prev_episode_true_objective[env_ids] = self.true_objective[env_ids]
            self.true_objective[env_ids] = 0
            
            self.prev_episode_closest_keypoint_max_dist[env_ids] = torch.where(self.prev_episode_successes[env_ids] > 0, self.prev_total_episode_closest_keypoint_max_dist[env_ids]/self.prev_episode_successes[env_ids], self.total_episode_closest_keypoint_max_dist[env_ids])
            self.total_episode_closest_keypoint_max_dist[env_ids] = 0
            self.prev_total_episode_closest_keypoint_max_dist[env_ids] = 0

            for key in self.rewards_episode.keys():
                self.rewards_episode[key][env_ids] = 0

            if self.save_states:
                self.dump_env_states(env_ids)

            self.extras["scalars"] = dict()
            self.extras["scalars"]["success_tolerance"] = self.success_tolerance

    def pre_physics_step(self, actions, joint_pos_targets: Optional[torch.Tensor] = None):
        PRINT_TIME_SINCE_LAST_STEP = False
        if PRINT_TIME_SINCE_LAST_STEP:
            if not hasattr(self, "last_time"):
                self.last_time = time.time()
            print(f"Time since last step: {time.time() - self.last_time:.3f} s, {1.0 / (time.time() - self.last_time):.1f} Hz")
            self.last_time = time.time()

        actions = actions.to(self.device)

        self.actions = actions.clone()

        if self.privileged_actions:
            torque_actions = actions[:, :3]
            actions = actions[:, 3:]

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        reset_goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.good_reset_boundary > 0 and self.buffer_length > 0:
            good_reset_env_ids = reset_env_ids[reset_env_ids < self.good_reset_boundary]
            random_reset_env_ids = reset_env_ids[reset_env_ids >= self.good_reset_boundary]
            good_reset_goal_env_ids = reset_goal_env_ids[reset_goal_env_ids < self.good_reset_boundary]
            random_reset_goal_env_ids = reset_goal_env_ids[reset_goal_env_ids >= self.good_reset_boundary]
            reset_buf_idxs = torch.randint(0, self.buffer_length, (self.num_envs,), device=self.device)
        else:
            good_reset_env_ids = torch.tensor([], device=self.device, dtype=reset_env_ids.dtype)
            random_reset_env_ids = reset_env_ids
            good_reset_goal_env_ids = torch.tensor([], device=self.device, dtype=reset_goal_env_ids.dtype)
            random_reset_goal_env_ids = reset_goal_env_ids
            reset_buf_idxs = None
        
        combined_random_env_ids = torch.cat([random_reset_env_ids, random_reset_goal_env_ids, random_reset_goal_env_ids])
        uniques, counts = combined_random_env_ids.unique(return_counts=True)
        random_reset_goal_env_ids = uniques[counts == 2]
        self.reset_target_pose(random_reset_goal_env_ids, None)
        self.reset_idx(good_reset_goal_env_ids, reset_buf_idxs, False)

        if len(reset_env_ids) > 0:
            self.reset_idx(random_reset_env_ids, None)
            self.reset_idx(good_reset_env_ids, reset_buf_idxs)
            

        self.set_actor_root_state_tensor_indexed()

        if self.use_relative_control:
            # hand
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
                actions[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
                self.act_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
                self.cur_targets[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )

            # arm
            targets = self.arm_hand_dof_pos[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            self.cur_targets[:, :7] = tensor_clamp(
                targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
            )
        else:
            # target position control for the hand DOFs

            # hand
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
                actions[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
                self.act_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
            )
            self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
                self.cur_targets[:, 7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
                self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
            )

            # Arm
            targets = self.prev_targets[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            self.cur_targets[:, :7] = tensor_clamp(
                targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
            )

        # Smooth arm
        SMOOTH_ARM = True
        if SMOOTH_ARM:
            self.cur_targets[:, :7] = (
                self.act_moving_average * self.cur_targets[:, :7]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, :7]
            )

        # Default CHECK_WITH_COMPUTED_JOINT_POS_TARGETS = False
        # Set to True to check if the computed joint pos targets are correct
        CHECK_WITH_COMPUTED_JOINT_POS_TARGETS = False
        if CHECK_WITH_COMPUTED_JOINT_POS_TARGETS:
            computed_joint_pos_targets = compute_joint_pos_targets(
                actions=self.actions,
                prev_targets=self.prev_targets,
                act_moving_average=self.act_moving_average,
                hand_dof_speed_scale=self.hand_dof_speed_scale,
                dt=self.dt,
            )
            assert computed_joint_pos_targets.shape == (self.num_envs, self.num_hand_arm_dofs), f"computed_joint_pos_targets.shape: {computed_joint_pos_targets.shape}, expected: ({self.num_envs}, {self.num_hand_arm_dofs})"
            assert self.cur_targets.shape == computed_joint_pos_targets.shape, f"self.cur_targets.shape: {self.cur_targets.shape}, expected: {computed_joint_pos_targets.shape}"

            num_errors = 0
            for i, name in enumerate(self.joint_names):
                val_orig = self.cur_targets[0, i].item()
                val_computed = computed_joint_pos_targets[0, i].item()
                print(f"{name} (idx {i}): original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}")
                if abs(val_orig - val_computed) > 1e-2:
                    num_errors += 1
                    print("--------------------------------")
                    print(f"Error: {name} (idx {i}): original: {val_orig}, computed: {val_computed}, diff: {val_orig - val_computed}")
                    print("--------------------------------")
            print("="*100)
            print(f"num_errors: {num_errors}")
            print("="*100)
            breakpoint()

        if joint_pos_targets is not None:
            self.cur_targets[:, :self.num_hand_arm_dofs] = joint_pos_targets.clone()

        if self._DO_NOT_MOVE:
            self.cur_targets[:, :] = self.prev_targets[:, :]

        self.prev_targets[:, :] = self.cur_targets[:, :]

        if VISUALIZE_PD_TARGET_AS_BLUE_ROBOT:
            self.cur_targets[:, self.num_hand_arm_dofs:] = self.cur_targets[:, :self.num_hand_arm_dofs].clone()
            self.blue_robot_arm_hand_dof_pos[:] = self.cur_targets[:, self.num_hand_arm_dofs:].clone()
            self.blue_robot_arm_hand_dof_vel[:] = 0.0

            blue_robot_indices = self.blue_robot_indices.to(torch.int32)
            self.deferred_set_dof_state_tensor_indexed([blue_robot_indices])

        self.set_dof_state_tensor_indexed()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device)
                * self.object_rb_masses
                * self.force_scale
            )

            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )
        
        if self.good_reset_boundary > 0:
            self.temp_root_states_buf[:, self.temp_buffer_index] = self.root_state_tensor.reshape(self.num_envs, -1, self.root_state_tensor.shape[1:]).cpu()
            self.temp_dof_states_buf[:, self.temp_buffer_index] = self.dof_state.reshape(self.num_envs, -1, self.dof_state.shape[1:]).cpu()
            self.temp_buffer_index += 1
        # apply torques
        if self.privileged_actions:
            torque_actions = torque_actions.unsqueeze(1)
            torque_amount = self.privileged_actions_torque
            torque_actions *= torque_amount
            self.action_torques[:, self.object_rb_handles, :] = torque_actions
            self.gym.apply_rigid_body_force_tensors(
                self.sim, None, gymtorch.unwrap_tensor(self.action_torques), gymapi.ENV_SPACE
            )

        USE_LIVE_PLOTTER = False
        if USE_LIVE_PLOTTER:
            if not hasattr(self, "live_plotter"):
                from live_plotter import FastLivePlotter
                self.live_plotter = FastLivePlotter(
                    n_plots=len(self.joint_names),
                    titles=self.joint_names,
                    xlabels=["idx"] * len(self.joint_names),
                    ylabels=["joint pos"] * len(self.joint_names),
                    ylims=[(self.joint_lower_limits[i], self.joint_upper_limits[i]) for i in range(len(self.joint_names))],
                    legends=[["pos", "target"]] * len(self.joint_names),
                )
                self.joint_pos_history = []
                self.joint_target_history = []

            ENV_IDX = 0
            joint_pos = self.arm_hand_dof_pos[ENV_IDX].cpu().numpy().copy()
            joint_target = self.cur_targets[ENV_IDX].cpu().numpy().copy()
            assert joint_pos.shape == joint_target.shape == (len(self.joint_names),), f"{joint_pos.shape} != {joint_target.shape} != {len(self.joint_names)}"
            self.joint_pos_history.append(joint_pos)
            self.joint_target_history.append(joint_target)
            joint_pos_history = np.stack(self.joint_pos_history, axis=0)
            joint_target_history = np.stack(self.joint_target_history, axis=0)
            joint_pos_and_target_history = np.stack([joint_pos_history, joint_target_history], axis=-1)
            assert joint_pos_and_target_history.shape == (len(self.joint_pos_history), len(self.joint_names), 2), f"{joint_pos_and_target_history.shape} != ({len(self.joint_pos_history)}, {len(self.joint_names)}, 2)"

            # Should be (N, 2)
            self.live_plotter.plot(
                y_data_list=[
                    joint_pos_and_target_history[:, i, :] for i in range(len(self.joint_names))
                ]
            )

        RECORD_DATA = self.cfg["env"]["record_data"]
        if RECORD_DATA:
            from recorded_data_scripts.recorded_data import RecordedData
            N_TIMESTEPS = self.cfg["env"]["record_data_num_steps"]

            # Get data from sim
            robot_root_state = self.root_state_tensor[self.allegro_hand_indices, :13].cpu().numpy()
            object_root_state = self.root_state_tensor[self.object_indices, :13].cpu().numpy()
            robot_joint_position = self.arm_hand_dof_pos.cpu().numpy()
            table_root_state = self.root_state_tensor[self.table_indices, :13].cpu().numpy()
            if hasattr(self, "goal_object_indices"):
                goal_root_state = self.root_state_tensor[self.goal_object_indices, :13].cpu().numpy()
            robot_joint_velocity = self.arm_hand_dof_vel.cpu().numpy()
            robot_joint_pos_target = self.cur_targets[:, :self.num_hand_arm_dofs].cpu().numpy()
            observations = self.obs_buf.cpu().numpy()
            actions = self.actions.cpu().numpy()

            # Initialize arrays if not already initialized
            if not hasattr(self, "robot_root_states_array"):
                self.robot_root_states_array = []
                self.object_root_states_array = []
                self.robot_joint_positions_array = []
                self.robot_joint_names = self.joint_names

                self.table_root_states_array = []
                if hasattr(self, "goal_object_indices"):
                    self.goal_root_states_array = []
                self.robot_joint_velocities_array = []
                self.robot_joint_pos_targets_array = []

                self.observations_array = []
                self.actions_array = []

            # Append data to arrays
            self.robot_root_states_array.append(robot_root_state)
            self.object_root_states_array.append(object_root_state)
            self.robot_joint_positions_array.append(robot_joint_position)
            self.table_root_states_array.append(table_root_state)
            if hasattr(self, "goal_object_indices"):
                self.goal_root_states_array.append(goal_root_state)
            self.robot_joint_velocities_array.append(robot_joint_velocity)
            self.robot_joint_pos_targets_array.append(robot_joint_pos_target)
            self.observations_array.append(observations)
            self.actions_array.append(actions)
            print(f"Recorded {len(self.robot_root_states_array)} / {N_TIMESTEPS} steps")

            # Save data to file
            if len(self.robot_root_states_array) >= N_TIMESTEPS:
                datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                this_dir = Path(__file__).parent
                root_dir = this_dir.parent.parent.parent
                recorded_data_path = root_dir / "recorded_data" / f"{datetime_str}.npz"
                recorded_data_path.parent.mkdir(parents=True, exist_ok=True)

                self.robot_root_states_array = np.stack(self.robot_root_states_array, axis=0)
                self.object_root_states_array = np.stack(self.object_root_states_array, axis=0)
                self.robot_joint_positions_array = np.stack(self.robot_joint_positions_array, axis=0)
                self.table_root_states_array = np.stack(self.table_root_states_array, axis=0)
                if hasattr(self, "goal_object_indices"):
                    self.goal_root_states_array = np.stack(self.goal_root_states_array, axis=0)
                self.robot_joint_velocities_array = np.stack(self.robot_joint_velocities_array, axis=0)
                self.robot_joint_pos_targets_array = np.stack(self.robot_joint_pos_targets_array, axis=0)
                self.observations_array = np.stack(self.observations_array, axis=0)
                self.actions_array = np.stack(self.actions_array, axis=0)

                assert self.robot_root_states_array.shape == (N_TIMESTEPS, self.num_envs, 13), f"{self.robot_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
                assert self.object_root_states_array.shape == (N_TIMESTEPS, self.num_envs, 13), f"{self.object_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
                assert self.robot_joint_positions_array.shape == (N_TIMESTEPS, self.num_envs, len(self.robot_joint_names)), f"{self.robot_joint_positions_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
                assert self.table_root_states_array.shape == (N_TIMESTEPS, self.num_envs, 13), f"{self.table_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
                if hasattr(self, "goal_object_indices"):
                    assert self.goal_root_states_array.shape == (N_TIMESTEPS, self.num_envs, 13), f"{self.goal_root_states_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, 13)"
                assert self.robot_joint_velocities_array.shape == (N_TIMESTEPS, self.num_envs, len(self.robot_joint_names)), f"{self.robot_joint_velocities_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
                assert self.robot_joint_pos_targets_array.shape == (N_TIMESTEPS, self.num_envs, len(self.robot_joint_names)), f"{self.robot_joint_pos_targets_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {len(self.robot_joint_names)})"
                assert self.observations_array.shape == (N_TIMESTEPS, self.num_envs, self.obs_buf.shape[1]), f"{self.observations_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {self.obs_buf.shape[1]})"
                assert self.actions_array.shape == (N_TIMESTEPS, self.num_envs, self.actions.shape[1]), f"{self.actions_array.shape} != ({N_TIMESTEPS}, {self.num_envs}, {self.actions.shape[1]})"

                time_array = np.arange(N_TIMESTEPS) * self.dt

                ENV_IDX = 0
                recorded_data = RecordedData(
                    robot_root_states_array=self.robot_root_states_array[:, ENV_IDX],
                    object_root_states_array=self.object_root_states_array[:, ENV_IDX],
                    robot_joint_positions_array=self.robot_joint_positions_array[:, ENV_IDX],
                    time_array=time_array,
                    robot_joint_names=self.robot_joint_names,
                    table_root_states_array=self.table_root_states_array[:, ENV_IDX],
                    goal_root_states_array=self.goal_root_states_array[:, ENV_IDX] if hasattr(self, "goal_object_indices") else None,
                    robot_joint_velocities_array=self.robot_joint_velocities_array[:, ENV_IDX],
                    robot_joint_pos_targets_array=self.robot_joint_pos_targets_array[:, ENV_IDX],
                    observations_array=self.observations_array[:, ENV_IDX],
                    actions_array=self.actions_array[:, ENV_IDX],
                )
                recorded_data.to_file(recorded_data_path)
                print(f"Saved recorded data to {recorded_data_path}")
                breakpoint()

                # Reset arrays
                self.robot_root_states_array = []
                self.object_root_states_array = []
                self.robot_joint_positions_array = []
                self.robot_joint_names = self.joint_names

                self.table_root_states_array = []
                if hasattr(self, "goal_object_indices"):
                    self.goal_root_states_array = []
                self.robot_joint_velocities_array = []
                self.robot_joint_pos_targets_array = []

                self.observations_array = []
                self.actions_array = []


    @property
    def act_moving_average(self) -> float:
        if self.cfg["env"]["actionsMovingAverageFinal"] is None or not hasattr(self, "_tyler_curriculum_scale"):
            return self.cfg["env"]["actionsMovingAverage"]
        else:
            return self.interpolate(
                init=self.cfg["env"]["actionsMovingAverage"],
                final=self.cfg["env"]["actionsMovingAverageFinal"],
                alpha=self._tyler_curriculum_scale,
            )

    @property
    def hand_dof_speed_scale(self) -> float:
        if self.cfg["env"]["dofSpeedScaleFinal"] is None or not hasattr(self, "_tyler_curriculum_scale"):
            return self.cfg["env"]["dofSpeedScale"]
        else:
            return self.interpolate(
                init=self.cfg["env"]["dofSpeedScale"],
                final=self.cfg["env"]["dofSpeedScaleFinal"],
                alpha=self._tyler_curriculum_scale,
            )

    @staticmethod
    def interpolate(init, final, alpha: float) -> float:
        assert 0 <= alpha <= 1, f"alpha must be between 0 and 1, got {alpha}"
        return init + (final - init) * alpha

    def post_physics_step(self):
        self.frame_since_restart += 1

        self.progress_buf += 1
        self.randomize_buf += 1

        self._extra_curriculum()

        self._update_tyler_curriculum()

        obs_buf, reward_obs_ofs = self.compute_observations()
        rewards, is_success = self.compute_kuka_reward()
        
        if self.good_reset_boundary > 0:
            add_indices = torch.where(is_success)[0]
            add_indices = add_indices[add_indices >= self.good_reset_boundary]
            add_indices = add_indices[self.temp_buffer_index[add_indices] > self.success_steps]
            
            if len(add_indices) > 0:
                rs_to_add = torch.stack([self.temp_root_states_buf[idx, torch.arange(self.temp_buffer_index[idx]-self.success_steps)] for idx in add_indices])
                dof_to_add = torch.stack([self.temp_dof_states_buf[idx, torch.arange(self.temp_buffer_index[idx]-self.success_steps)] for idx in add_indices])
                
                num_to_add = len(rs_to_add)
                
                next_index = self.buffer_index + num_to_add
                self.buffer_length = min(self.buffer_length + num_to_add, self.max_buffer_size)
                
                if next_index >= self.max_buffer_size:
                    num_to_add -= (self.max_buffer_size - self.buffer_index)
                    self.root_state_resets[self.buffer_index:] = rs_to_add[:self.max_buffer_size-self.buffer_index]
                    self.root_state_resets[:num_to_add] = rs_to_add[self.max_buffer_size-self.buffer_index:]
                    self.dof_resets[self.buffer_index:] = dof_to_add[:self.max_buffer_size-self.buffer_index]
                    self.dof_resets[:num_to_add] = dof_to_add[self.max_buffer_size-self.buffer_index:]
                else:
                    self.root_state_resets[self.buffer_index:next_index] = rs_to_add
                    self.dof_resets[self.buffer_index:next_index] = dof_to_add
                
                self.buffer_index = next_index % self.max_buffer_size
                
                print(f"Added {len(rs_to_add)} states, lifted {self.lifted_object[add_indices].sum().item()}/{len(add_indices)} objects")
            
            self.temp_buffer_index[torch.where(is_success)[0]] = 0
            self.temp_buffer_index[torch.where(self.reset_buf)[0]] = 0

        # add rewards to observations
        reward_obs_scale = 0.01
        obs_buf[:, reward_obs_ofs : reward_obs_ofs + 1] = rewards.unsqueeze(-1) * reward_obs_scale * self.turn_off_extra_obs_scale
        # print(f"obs_buf: {obs_buf[0]}")

        self.clamp_obs(obs_buf)

        self._eval_stats(is_success)

        if self.save_states:
            self.accumulate_env_states()

        self._capture_video_if_needed()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            axes_geom = gymutil.AxesGeometry(0.1)

            sphere_pose = gymapi.Transform()
            sphere_pose.r = gymapi.Quat(0, 0, 0, 1)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, sphere_pose, color=(1, 1, 0))
            sphere_geom_white = gymutil.WireframeSphereGeometry(0.02, 8, 8, sphere_pose, color=(1, 1, 1))

            palm_center_pos_cpu = self.palm_center_pos.cpu().numpy()
            palm_rot_cpu = self._palm_rot.cpu().numpy()

            # for i in range(self.num_envs):
            #     palm_center_transform = gymapi.Transform()
            #     palm_center_transform.p = gymapi.Vec3(*palm_center_pos_cpu[i])
            #     palm_center_transform.r = gymapi.Quat(*palm_rot_cpu[i])
            #     gymutil.draw_lines(sphere_geom_white, self.gym, self.viewer, self.envs[i], palm_center_transform)

            # for j in range(self.num_allegro_fingertips):
            #     fingertip_pos_cpu = self.fingertip_pos_offset[:, j].cpu().numpy()
            #     fingertip_rot_cpu = self.fingertip_rot[:, j].cpu().numpy()

            #     for i in range(self.num_envs):
            #         fingertip_transform = gymapi.Transform()
            #         fingertip_transform.p = gymapi.Vec3(*fingertip_pos_cpu[i])
            #         fingertip_transform.r = gymapi.Quat(*fingertip_rot_cpu[i])

            #         gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], fingertip_transform)

            # for j in range(self.num_keypoints):
            #     keypoint_pos_cpu = self.obj_keypoint_pos[:, j].cpu().numpy()
                # goal_keypoint_pos_cpu = self.goal_keypoint_pos[:, j].cpu().numpy()

                # for i in range(self.num_envs):
                #     keypoint_transform = gymapi.Transform()
                #     keypoint_transform.p = gymapi.Vec3(*keypoint_pos_cpu[i])
                #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], keypoint_transform)

                    # goal_keypoint_transform = gymapi.Transform()
                    # goal_keypoint_transform.p = gymapi.Vec3(*goal_keypoint_pos_cpu[i])
                    # gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], goal_keypoint_transform)

            # Visualize object and goal pose
            for i in range(self.num_envs):
                object_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.object_pos[i]),
                    r=gymapi.Quat(*self.object_rot[i]),
                )
                goal_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self.goal_pos[i]),
                    r=gymapi.Quat(*self.goal_rot[i]),
                )
                self._draw_transform(transform=object_transform, env_idx=i)
                # self._draw_transform(transform=goal_transform, env_idx=i)

    def _update_tyler_curriculum(self):
        # Vary _tyler_curriculum_scale from 0.0 to 1.0 over time
        # 0.0 means easy and 1.0 means hard
        if not hasattr(self, "_tyler_curriculum_scale"):
            self._tyler_curriculum_scale = 0.0
            self._last_tyler_curriculum_update = time.time()

        # If gets at least 50% of max consecutive successes and been at least 5 minutes since last update, turn off extra obs more
        mean_successes = self.prev_episode_successes.mean().item()
        minutes_elapsed_since_last_update = (time.time() - self._last_tyler_curriculum_update) / 60
        doing_well = mean_successes > self.max_consecutive_successes * 0.6
        enough_time_since_last_update = minutes_elapsed_since_last_update > 5
        if doing_well and enough_time_since_last_update:
            self._tyler_curriculum_scale += 0.01
            if self._tyler_curriculum_scale > 1.0:
                self._tyler_curriculum_scale = 1.0
            self._last_tyler_curriculum_update = time.time()

        self.extras["tyler_curriculum_scale"] = self._tyler_curriculum_scale
        self.extras["mean_successes"] = mean_successes
        self.extras["mean_success_ratio"] = mean_successes / self.max_consecutive_successes
        self.extras["minutes_elapsed_since_last_update"] = minutes_elapsed_since_last_update

    def _initialize_camera_sensor(self, cam_pos, cam_target) -> None:
        self.camera_properties = gymapi.CameraProperties()
        RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE = 4
        self.camera_properties.width = int(
            self.camera_properties.width / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        self.camera_properties.height = int(
            self.camera_properties.height / RESOLUTION_REDUCTION_FACTOR_TO_SAVE_SPACE
        )
        self.camera_handle = self.gym.create_camera_sensor(
            self.envs[self.index_to_view],
            self.camera_properties,
        )

        # self.video_frames is important for understanding the state of video recording
        #   Case 1: self.video_frames is None:
        #     * This means that we are not recording video
        #   Case 2: self.video_frames = []
        #     * This means that we should start recording video
        #     * BUT, we want our videos to start at the first frame of an episode
        #     * So, we are waiting for this
        #   Case 3: self.video_frames = [np.array(frame) for frame in ...]
        #     * These are image frames that will be assembled into a video when enough frames are capture
        self.video_frames: Optional[List[np.ndarray]] = None
        self.gym.set_camera_location(
            self.camera_handle, self.envs[self.index_to_view], cam_pos, cam_target
        )

    def _modify_render_settings_if_headless(self) -> None:
        # If not headless, leave things as they are
        if self.viewer is not None:
            return

        # If headless, we should default to having self.enable_viewer_sync=False to speed up env stepping
        self.enable_viewer_sync = False

    def _capture_video_if_needed(self) -> None:
        # If capture_video is False, we don't need to capture video
        if not self.cfg["env"]["capture_video"]:
            return

        # If enableCameraSensors is False, we can't capture video
        assert self.cfg["env"]["enableCameraSensors"], "capture_video is only supported if enableCameraSensors is True"

        should_start_video_capture_at_start_of_next_episode = (
            self.video_frames is None
            and self.control_steps % self.cfg["env"]["capture_video_freq"] == 0
            # and (self.control_steps > 0)  # Don't record video on first step
        )
        if should_start_video_capture_at_start_of_next_episode:
            print("-" * 80)
            print(
                f"At self.control_steps = {self.control_steps}, should start video capture at start of next episode"
            )
            print("-" * 80)
            self.video_frames = []
            return

        should_start_video_capture_now = (
            self.video_frames is not None
            and len(self.video_frames) == 0
            # and self.progress_buf[self.index_to_view].item() <= 1  # Only start video capture on first step of episode so that videos don't start in the middle of an episode
                                                                     # Actually doesn't work because progress_buf gets reset to 0 not only at start of episode but on success
            and self.reset_buf[self.index_to_view].item() == 1       # Only start video capture after reset of an env
        )
        video_capture_in_progress = (
            self.video_frames is not None and len(self.video_frames) > 0
        )
        if should_start_video_capture_now or video_capture_in_progress:
            self._capture_video(video_capture_in_progress)

    def _capture_video(self, video_capture_in_progress: bool) -> None:
        assert self.video_frames is not None
        if not video_capture_in_progress:
            print("-" * 80)
            print("Starting to capture video frames...")
            print("-" * 80)
            # Video capture requires that self.enable_viewer_sync=True
            # If there is a viewer, we need to save the previous value of self.enable_viewer_sync so we can restore it later
            if self.viewer is not None:
                self.enable_viewer_sync_before = self.enable_viewer_sync
            # If there is no viewer, we always want self.enable_viewer_sync=False to speed up env stepping
            else:
                self.enable_viewer_sync_before = False

        # Store image
        self.enable_viewer_sync = True
        self.gym.render_all_camera_sensors(self.sim)
        color_image = self.gym.get_camera_image(
            self.sim,
            self.envs[self.index_to_view],
            self.camera_handle,
            gymapi.IMAGE_COLOR,
        )
        if color_image.size == 0:
            print(f"Warning: color_image is empty on {self.control_steps}th step, make sure you have this change to vec_task.py")
            print("https://github.com/tylerlum/human2sim2robot/blob/a5fd55baf83fbd04c585e2d596967ba08a38d540/human2sim2robot/sim_training/tasks/base/vec_task.py#L544")
            return
        NUM_RGBA = 4
        color_image = color_image.reshape(
            self.camera_properties.height, self.camera_properties.width, NUM_RGBA
        )
        self.video_frames.append(color_image)

        if len(self.video_frames) == self.cfg["env"]["capture_video_len"]:
            video_filename = f"{DATETIME_STR}_video_{self.control_steps}.mp4"
            videos_dir = Path("videos")
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_path = videos_dir / video_filename
            print("-" * 80)
            print(f"Saving video to {video_path} ...")

            if not self.enable_viewer_sync_before:
                self.video_frames.pop(0)  # Remove first frame because it was not synced

            import imageio
            import wandb

            imageio.mimsave(video_path, self.video_frames, fps=int(1.0 / self.control_dt))
            if wandb.run is not None:
                wandb_video = wandb.Video(
                    str(video_path), fps=int(1.0 / self.control_dt)
                )
                wandb.log({"video": wandb_video})
                # self.wandb_dict["video"] = wandb.Video(
                #     str(video_path), fps=int(1.0 / self.control_dt)
                # )
            print("DONE")
            print("-" * 80)

            # Reset variables
            self.video_frames = None
            self.enable_viewer_sync = self.enable_viewer_sync_before


    def _draw_transform(
        self, transform: gymapi.Transform, line_length: float = 0.2, env_idx: int = 0
    ) -> None:
        env = self.envs[env_idx]

        origin = transform.transform_point(gymapi.Vec3(0, 0, 0))
        x_dir = transform.transform_point(gymapi.Vec3(line_length, 0, 0))
        y_dir = transform.transform_point(gymapi.Vec3(0, line_length, 0))
        z_dir = transform.transform_point(gymapi.Vec3(0, 0, line_length))

        RED = (1, 0, 0)
        GREEN = (0, 1, 0)
        BLUE = (0, 0, 1)

        for color, dir in zip([RED, GREEN, BLUE], [x_dir, y_dir, z_dir]):
            gymutil.draw_line(
                p1=origin,
                p2=dir,
                color=gymapi.Vec3(*color),  # type: ignore
                gym=self.gym,
                viewer=self.viewer,
                env=env,
            )
            self._draw_debug_line_of_spheres(
                env=env,
                start_pos=origin,
                end_pos=dir,
                color=color,
            )
    def _draw_debug_line_of_spheres(
        self,
        env,
        start_pos: gymapi.Vec3,
        end_pos: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.01,
        num_spheres: int = 10,
    ) -> None:
        for i in range(num_spheres):
            fraction = (i + 1) / (num_spheres + 1)
            pos = start_pos + ((end_pos - start_pos) * fraction)
            self._draw_debug_sphere(
                env=env,
                position=pos,
                color=color,
                radius=radius,
            )
    def _draw_debug_sphere(
        self,
        env,
        position: gymapi.Vec3,
        color: Tuple[float, float, float],
        radius: float = 0.005,
        num_lats: int = 2,
        num_lons: int = 2,
    ) -> None:
        sphere_geom = gymutil.WireframeSphereGeometry(radius, num_lats, num_lons, color=color)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, gymapi.Transform(p=position))
    
    def accumulate_env_states(self):
        root_state_tensor = self.root_state_tensor.reshape(
            [self.num_envs, -1, *self.root_state_tensor.shape[1:]]
        ).clone()
        dof_state = self.dof_state.reshape([self.num_envs, -1, *self.dof_state.shape[1:]]).clone()

        for env_idx in range(self.num_envs):
            env_root_state_tensor = root_state_tensor[env_idx]
            self.episode_root_state_tensors[env_idx].append(env_root_state_tensor)

            env_dof_state = dof_state[env_idx]
            self.episode_dof_states[env_idx].append(env_dof_state)

    def dump_env_states(self, env_ids):
        def write_tensor_to_bin_stream(tensor, stream):
            bin_buff = io.BytesIO()
            torch.save(tensor, bin_buff)
            bin_buff = bin_buff.getbuffer()
            stream.write(int(len(bin_buff)).to_bytes(4, "big"))
            stream.write(bin_buff)

        with open(self.save_states_filename, "ab") as save_states_file:
            bin_stream = io.BytesIO()

            for env_idx in env_ids:
                ep_len = len(self.episode_root_state_tensors[env_idx])
                if ep_len <= 20:
                    continue

                states_to_save = min(ep_len // 10, 50)
                state_indices = random.sample(range(ep_len), states_to_save)

                print(f"Adding {states_to_save} states {state_indices}")
                bin_stream.write(int(states_to_save).to_bytes(4, "big"))

                root_states = [self.episode_root_state_tensors[env_idx][si] for si in state_indices]
                dof_states = [self.episode_dof_states[env_idx][si] for si in state_indices]

                root_states = torch.stack(root_states)
                dof_states = torch.stack(dof_states)

                write_tensor_to_bin_stream(root_states, bin_stream)
                write_tensor_to_bin_stream(dof_states, bin_stream)

                self.episode_root_state_tensors[env_idx] = []
                self.episode_dof_states[env_idx] = []

            bin_data = bin_stream.getbuffer()
            if bin_data.nbytes > 0:
                print(f"Writing {len(bin_data)} to file {self.save_states_filename}")
                save_states_file.write(bin_data)

    def load_initial_states(self):
        loaded_root_states = []
        loaded_dof_states = []

        with open(self.load_states_filename, "rb") as states_file:

            def read_nbytes(n_):
                res = states_file.read(n_)
                if len(res) < n_:
                    raise RuntimeError(
                        f"Could not read {n_} bytes from the binary file. Perhaps reached the end of file"
                    )
                return res

            while True:
                try:
                    num_states = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"num_states_chunk {num_states}")

                    root_states_len = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"root tensors len {root_states_len}")
                    root_states_bytes = read_nbytes(root_states_len)

                    dof_states_len = int.from_bytes(read_nbytes(4), byteorder="big")
                    print(f"dof_states_len {dof_states_len}")
                    dof_states_bytes = read_nbytes(dof_states_len)

                except Exception as exc:
                    print(exc)
                    break
                finally:
                    # parse binary buffers
                    def parse_tensors(bin_data):
                        with io.BytesIO(bin_data) as buffer:
                            tensors = torch.load(buffer)
                            return tensors

                    root_state_tensors = parse_tensors(root_states_bytes)
                    dof_state_tensors = parse_tensors(dof_states_bytes)
                    loaded_root_states.append(root_state_tensors)
                    loaded_dof_states.append(dof_state_tensors)

        self.initial_root_state_tensors = torch.cat(loaded_root_states)
        self.initial_dof_state_tensors = torch.cat(loaded_dof_states)
        assert self.initial_dof_state_tensors.shape[0] == self.initial_root_state_tensors.shape[0]
        self.num_initial_states = len(self.initial_root_state_tensors)

        print(f"{self.num_initial_states} states loaded from file {self.load_states_filename}!")
