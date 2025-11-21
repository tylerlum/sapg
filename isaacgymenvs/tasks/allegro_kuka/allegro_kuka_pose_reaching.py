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

import os
from typing import List, Optional

import torch
from isaacgym import gymapi
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import get_axis_params, to_torch, torch_rand_float
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_utils import populate_dof_properties

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgym import gymapi, gymtorch, gymutil


class AllegroKukaPoseReaching(AllegroKukaBase):
    """
    Simple reaching task with no manipulated objects or table.
    The policy is rewarded for matching desired joint angles.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.use_green_robot = cfg["env"]["use_green_robot"]
        self.sanity_check_controls = cfg["env"]["sanity_check_controls"]
        self.goal_object_indices: List[int] = []

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.pose_obs_size = self.full_state_size
        self.reward_obs_offset = self.full_state_size - 1
        self._dof_obs_dim = self.num_hand_arm_dofs * 4

        self.nominal_joint_target = self._build_nominal_joint_target()
        self.joint_targets = torch.zeros(
            (self.num_envs, self.num_hand_arm_dofs), dtype=torch.float, device=self.device
        )
        self.current_joint_error = torch.zeros_like(self.joint_targets)
        self.current_joint_abs_error = torch.zeros_like(self.joint_targets)

        self.joint_target_noise = self.cfg["env"].get("targetJointNoise", 0.5)
        self.joint_success_tolerance = self.cfg["env"].get("jointSuccessTolerance", 0.05)
        self.kuka_joint_success_tolerance = self.cfg["env"].get("kukaJointSuccessTolerance", 0.01)
        self.hand_joint_success_tolerance = self.cfg["env"].get("handJointSuccessTolerance", 0.05)
        self.joint_error_scale = self.cfg["env"].get("jointErrorRewScale", 1.0)
        self.joint_velocity_penalty_scale = self.cfg["env"].get("jointVelocityPenaltyScale", 0.0)
        
        # Dummy buffers for base-class debug drawing helpers
        self.object_state = torch.zeros((self.num_envs, 13), dtype=torch.float, device=self.device)
        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos = torch.zeros_like(self.object_pos)
        self.goal_rot = torch.zeros_like(self.object_rot)
        if self.use_green_robot:
            self.green_robot_arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_hand_arm_dofs:]
            self.green_robot_arm_hand_dof_pos = self.green_robot_arm_hand_dof_state[..., 0]
            self.green_robot_arm_hand_dof_vel = self.green_robot_arm_hand_dof_state[..., 1]

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._sample_joint_targets(env_ids)

    def _build_nominal_joint_target(self) -> Tensor:
        target_pose_cfg = self.cfg["env"].get("targetJointPose", None)
        if target_pose_cfg is None:
            return self.hand_arm_default_dof_pos.clone().to(self.device)

        if len(target_pose_cfg) != self.num_hand_arm_dofs:
            raise ValueError(
                f"targetJointPose must have {self.num_hand_arm_dofs} entries, got {len(target_pose_cfg)}"
            )
        return torch.tensor(target_pose_cfg, dtype=torch.float, device=self.device)

    # ------------------------------------------------------------------ #
    # Scene creation
    # ------------------------------------------------------------------ #
    def _object_keypoint_offsets(self):
        return []

    def _create_envs(self, num_envs, spacing, num_per_row):
        if self.should_load_initial_states:
            self.load_initial_states()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
        allegro_pose = gymapi.Transform()
        allegro_pose.p = gymapi.Vec3(*get_axis_params(0.0, self.up_axis_idx)) + gymapi.Vec3(0.0, 0.8, 0)
        allegro_pose.r = gymapi.Quat(0, 0, 0, 1)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        allegro_kuka_asset = self.gym.load_asset(self.sim, asset_root, self.hand_arm_asset_file, asset_options)

        self.num_hand_arm_bodies = self.gym.get_asset_rigid_body_count(allegro_kuka_asset)
        self.num_hand_arm_shapes = self.gym.get_asset_rigid_shape_count(allegro_kuka_asset)
        num_hand_arm_dofs = self.gym.get_asset_dof_count(allegro_kuka_asset)
        assert (
            self.num_hand_arm_dofs == num_hand_arm_dofs
        ), f"Asset DOF count {num_hand_arm_dofs} != expected {self.num_hand_arm_dofs}"

        max_agg_bodies = self.num_hand_arm_bodies
        max_agg_shapes = self.num_hand_arm_shapes
        if self.use_green_robot:
            max_agg_bodies += self.num_hand_arm_bodies
            max_agg_shapes += self.num_hand_arm_shapes

        allegro_hand_dof_props = self.gym.get_asset_dof_properties(allegro_kuka_asset)
        populate_dof_properties(allegro_hand_dof_props, self.dof_params, self.num_arm_dofs, self.num_hand_dofs)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        for i in range(self.num_hand_arm_dofs):
            self.arm_hand_dof_lower_limits.append(allegro_hand_dof_props["lower"][i])
            self.arm_hand_dof_upper_limits.append(allegro_hand_dof_props["upper"][i])
        self.arm_hand_dof_lower_limits = to_torch(self.arm_hand_dof_lower_limits, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(self.arm_hand_dof_upper_limits, device=self.device)

        self.set_allegro_kuka_asset_rigid_shape_properties(
            allegro_kuka_asset=allegro_kuka_asset,
            friction=self.cfg["env"].get("assetFriction", 0.5),
            fingertip_friction=self.cfg["env"].get("fingertipFriction", 1.5),
        )

        self.envs = []
        self.allegro_hands = []
        self.objects = [None] * num_envs
        self.allegro_hand_indices = []
        self.rigid_body_name_to_idx = {}
        if self.use_green_robot:
            self.green_robots = []
            self.green_robot_indices = []

        body_names = self.gym.get_asset_rigid_body_names(allegro_kuka_asset)
        self.allegro_fingertip_handles = [
            self.gym.find_asset_rigid_body_index(allegro_kuka_asset, name) for name in self.allegro_fingertips
        ]

        has_iiwa14 = "iiwa14_link_7" in body_names
        has_iiwa7 = "iiwa7_link_7" in body_names
        if (has_iiwa14 and has_iiwa7) or (not has_iiwa14 and not has_iiwa7):
            raise ValueError(
                f"Expected either iiwa14 or iiwa7 in asset {self.hand_arm_asset_file}, got body names {body_names}"
            )
        if has_iiwa14:
            self.robot_name = "iiwa14"
            self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa14_link_7")
        else:
            self.robot_name = "iiwa7"
            self.allegro_palm_handle = self.gym.find_asset_rigid_body_index(allegro_kuka_asset, "iiwa7_link_7")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            collision_group = i
            collision_filter = -1
            allegro_actor = self.gym.create_actor(
                env_ptr, allegro_kuka_asset, allegro_pose, "allegro", collision_group, collision_filter, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, allegro_actor, allegro_hand_dof_props)

            allegro_hand_idx = self.gym.get_actor_index(env_ptr, allegro_actor, gymapi.DOMAIN_SIM)
            self.allegro_hand_indices.append(allegro_hand_idx)

            for name in self.gym.get_actor_rigid_body_names(env_ptr, allegro_actor):
                rb_idx = self.gym.find_actor_rigid_body_index(env_ptr, allegro_actor, name, gymapi.DOMAIN_ENV)
                self.rigid_body_name_to_idx["allegro/" + name] = rb_idx

            if self.use_green_robot:
                green_robot_actor = self.gym.create_actor(
                    env_ptr,
                    allegro_kuka_asset,
                    allegro_pose,
                    "green_robot",
                    i + self.num_envs * 2,
                    -1,
                    0,
                )
                self.gym.set_actor_dof_properties(env_ptr, green_robot_actor, allegro_hand_dof_props)
                self._set_actor_color(env_ptr, green_robot_actor, color=(0, 1, 0))
                self.green_robots.append(green_robot_actor)
                green_robot_idx = self.gym.get_actor_index(env_ptr, green_robot_actor, gymapi.DOMAIN_SIM)
                self.green_robot_indices.append(green_robot_idx)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.allegro_hands.append(allegro_actor)

        self.allegro_hand_indices = to_torch(self.allegro_hand_indices, dtype=torch.long, device=self.device)
        self.allegro_fingertip_handles = to_torch(self.allegro_fingertip_handles, dtype=torch.long, device=self.device)
        # self.object_indices = self.allegro_hand_indices.clone()
        # self.table_indices = self.allegro_hand_indices.clone()
        # self.object_rb_handles = to_torch([self.allegro_palm_handle], dtype=torch.long, device=self.device)
        # self.object_rb_masses = torch.ones_like(self.object_rb_handles, dtype=torch.float)

        self.object_init_state = torch.zeros((self.num_envs, 13), dtype=torch.float, device=self.device)
        self.goal_states = self.object_init_state.clone()
        self.goal_init_state = self.goal_states.clone()

        if self.use_green_robot:
            self.green_robot_indices = to_torch(self.green_robot_indices, dtype=torch.long, device=self.device)

        self._after_envs_created()

    # ------------------------------------------------------------------ #
    # Goal sampling
    # ------------------------------------------------------------------ #
    def _sample_joint_targets(self, env_ids: Tensor) -> None:
        if len(env_ids) == 0:
            return

        # lower = self.arm_hand_dof_lower_limits[: self.num_hand_arm_dofs].unsqueeze(0)
        # upper = self.arm_hand_dof_upper_limits[: self.num_hand_arm_dofs].unsqueeze(0)
        # rand = torch.rand((len(env_ids), self.num_hand_arm_dofs), device=self.device)
        # target = lower + rand * (upper - lower)
        # sample around nominal joint target
        target = self.nominal_joint_target.unsqueeze(0).repeat(len(env_ids), 1)
        noise = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_arm_dofs), device=self.device)
        target = target + noise * self.joint_target_noise
        target = torch.clamp(target, self.arm_hand_dof_lower_limits[: self.num_hand_arm_dofs], self.arm_hand_dof_upper_limits[: self.num_hand_arm_dofs])

        self.joint_targets[env_ids] = target

    def _reset_target(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True) -> None:
        if len(env_ids) == 0:
            return
        self._sample_joint_targets(env_ids)

    def reset_object_pose(self, env_ids: Tensor, reset_buf_idxs=None, tensor_reset=True):
        return

    # ------------------------------------------------------------------ #
    # Observations and rewards
    # ------------------------------------------------------------------ #
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.obs_type == "full_state":
            if self.with_fingertip_force_sensors or self.with_table_force_sensor:
                self.gym.refresh_force_sensor_tensor(self.sim)
            if self.with_dof_force_sensors:
                self.gym.refresh_dof_force_tensor(self.sim)

        joint_pos = self.arm_hand_dof_pos[:, : self.num_hand_arm_dofs]
        joint_vel = self.arm_hand_dof_vel[:, : self.num_hand_arm_dofs]

        error = self.joint_targets - joint_pos
        obs = torch.cat([joint_pos, joint_vel, self.joint_targets], dim=-1)

        self.obs_buf.zero_()
        cols = obs.shape[1]
        self.obs_buf[:, :cols] = obs[:, :cols]

        self.current_joint_error = error
        self.current_joint_abs_error = torch.abs(error)

        return self.obs_buf, self.reward_obs_offset

    def compute_kuka_reward(self):
        joint_error = self.current_joint_error
        joint_error_mse = torch.mean(joint_error ** 2, dim=1)
        joint_vel_penalty = torch.mean(
            self.arm_hand_dof_vel[:, : self.num_hand_arm_dofs] ** 2,
            dim=1,
        )
        abs_error = torch.abs(joint_error)

        max_abs_error = self.current_joint_abs_error.max(dim=1).values
        reward = -torch.mean(abs_error, dim=1)

        # old threshold
        kuka_joint_mse = torch.mean(joint_error[:, :self.num_arm_dofs] ** 2, dim=1)
        hand_joint_mse = torch.mean(joint_error[:, self.num_arm_dofs:] ** 2, dim=1)
        kuka_near_goal = kuka_joint_mse <= self.kuka_joint_success_tolerance
        hand_near_goal = hand_joint_mse <= self.hand_joint_success_tolerance

        # new threshold
        # kuka_abs_error = torch.mean(abs_error[:, :self.num_arm_dofs], dim=1)
        # hand_abs_error = torch.mean(abs_error[:, self.num_arm_dofs:], dim=1)
        # kuka_near_goal = kuka_abs_error <= self.kuka_joint_success_tolerance
        # hand_near_goal = hand_abs_error <= self.hand_joint_success_tolerance
        near_goal = kuka_near_goal & hand_near_goal
        print(f"near_goal: {near_goal[0].item()}")
        print(f"joint_error_mse: {joint_error_mse[0].item()}")
        self.near_goal_steps += near_goal
        is_success = self.near_goal_steps >= self.success_steps
        self.successes += is_success
        self.reset_goal_buf[:] = is_success

        self.rewards_episode["keypoint_rew"] += -self.joint_error_scale * joint_error_mse
        self.rewards_episode["allegro_actions_penalty"] += -self.joint_velocity_penalty_scale * joint_vel_penalty

        bonus = near_goal * (self.reach_goal_bonus / max(self.success_steps, 1))
        reward = reward + bonus
        self.rew_buf[:] = reward
        resets = self._compute_resets(is_success)
        self.reset_buf[:] = resets

        self.joint_error_mse = joint_error_mse
        self.extras["joint_error_mse"] = joint_error_mse
        self.extras["joint_max_abs_error"] = max_abs_error
        self.true_objective = self._true_objective()
        self.extras["true_objective"] = self.true_objective
        self.extras["successes"] = self.prev_episode_successes
        if self.max_consecutive_successes > 0:
            denom = max(1, self.max_consecutive_successes)
            self.extras["success_ratio"] = self.prev_episode_successes.mean().item() / denom

        return self.rew_buf, is_success

    def _compute_resets(self, is_success: Tensor):
        ones = torch.ones_like(self.reset_buf)
        zeros = torch.zeros_like(self.reset_buf)

        max_episode = torch.where(self.progress_buf >= self.max_episode_length - 1, ones, zeros)

        resets = self.reset_buf | max_episode
        return resets

    def _true_objective(self) -> Tensor:
        return -self.joint_error_mse

    # ------------------------------------------------------------------ #
    # Control overrides
    # ------------------------------------------------------------------ #
    def pre_physics_step(self, actions, joint_pos_targets: Optional[Tensor] = None):
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
        # if env 0 is reset, then print a message
        if 0 in reset_env_ids:
            print("Environment 0 was reset--------------------------------")

        if len(reset_env_ids) > 0:
            self.reset_idx(random_reset_env_ids, None)
            self.reset_idx(good_reset_env_ids, reset_buf_idxs)
            
        self.set_actor_root_state_tensor_indexed()

        if self.use_relative_control:
            # arm relative to current position
            targets = self.arm_hand_dof_pos[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            self.cur_targets[:, :7] = tensor_clamp(
                targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
            )
        else:
            # arm relative to previous target
            targets = self.prev_targets[:, :7] + self.hand_dof_speed_scale * self.dt * self.actions[:, :7]
            self.cur_targets[:, :7] = tensor_clamp(
                targets, self.arm_hand_dof_lower_limits[:7], self.arm_hand_dof_upper_limits[:7]
            )

        # Smooth arm
        self.cur_targets[:, :7] = (
            self.arm_moving_average * self.cur_targets[:, :7]
            + (1.0 - self.arm_moving_average) * self.prev_targets[:, :7]
        )

        # hand
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = scale(
            actions[:, 7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
        )
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = (
            self.hand_moving_average * self.cur_targets[:, 7 : self.num_hand_arm_dofs]
            + (1.0 - self.hand_moving_average) * self.prev_targets[:, 7 : self.num_hand_arm_dofs]
        )
        self.cur_targets[:, 7 : self.num_hand_arm_dofs] = tensor_clamp(
            self.cur_targets[:, 7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_lower_limits[7 : self.num_hand_arm_dofs],
            self.arm_hand_dof_upper_limits[7 : self.num_hand_arm_dofs],
        )

        self.prev_targets[:, :] = self.cur_targets[:, :].clone()
        desired_pose = self.joint_targets.clone()
        if self.sanity_check_controls:
            self.cur_targets[:, :self.num_hand_arm_dofs] = desired_pose[:]
        self.cur_targets[:, self.num_hand_arm_dofs:] = desired_pose.clone()
        self.green_robot_arm_hand_dof_pos[:] = desired_pose
        self.green_robot_arm_hand_dof_vel.zero_()

        green_robot_indices = self.green_robot_indices.to(torch.int32)
        self.deferred_set_dof_state_tensor_indexed([green_robot_indices])
        self.set_dof_state_tensor_indexed()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        

