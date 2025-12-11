from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
import time
from pathlib import Path
from typing import Tuple

from sim2real.rl_player import RlPlayer

import torch  # isort:skip
import pytorch_kinematics as pk
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    compute_joint_pos_targets,
    compute_observation,
    create_chain_and_serial_chain,
)
from sim2sim.isaac_sim.isaac_env import create_env

N_OBS = 140
N_ACT = 29

HAND_MOVING_AVERAGE = 0.1
ARM_MOVING_AVERAGE = 0.1
HAND_DOF_SPEED_SCALE = 2.5


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacEnvNoRosJointPosTargets:
    def __init__(
        self,
        env: AllegroKukaBase,
        control_dt: float,
        device: str,
        chain: pk.Chain,
    ):
        self.env = env
        self.control_dt = control_dt
        self.device = device
        self.chain = chain

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((1, N_ACT), device=self.device))
        return obs["obs"]

    def step_with_joint_pos_targets(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action,
            prev_targets=self.env.prev_targets,
            hand_moving_average=HAND_MOVING_AVERAGE,
            arm_moving_average=ARM_MOVING_AVERAGE,
            hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
            dt=self.control_dt,
        )

        self.env.root_state_tensor[self.env.goal_object_indices, 0:7] = torch.tensor([0.,  0.,  0.88, 0.,  0.,  0.,  1.], device=self.device)[None]
        self.env.deferred_set_actor_root_state_tensor_indexed([self.env.goal_object_indices])
        obs, reward, done, info = self.env.step(
            action, joint_pos_targets=joint_pos_targets
        )
        q = self.env.arm_hand_dof_pos
        qd = self.env.arm_hand_dof_vel
        object_pose = self.env.object_pose
        goal_object_pose = self.env.goal_pose
        object_scales = self.env.object_scales

        DEBUG = False
        if DEBUG:
            print(f"q = {q}")
            print(f"qd = {qd}")
            print(f"object_pose = {object_pose}")
            print(f"goal_object_pose = {goal_object_pose}")
            print(f"object_scales = {object_scales}")
            breakpoint()

        # HACK: Overwrite
        goal_object_pose[:] = torch.tensor([0.,  0.,  0.88, 0.,  0.,  0.,  1.], device=self.device)[None]

        new_obs = compute_observation(
            q=q,
            qd=qd,
            prev_action_targets=self.env.prev_targets,
            object_pose=object_pose,
            goal_object_pose=goal_object_pose,
            object_scales=object_scales,
            chain=self.chain,
            obs_list=self.env.obs_list,
        )
        # print(f"q = {q}")
        # print(f"qd = {qd}")
        # print(f"prev_action_targets = {self.env.prev_targets}")
        # print(f"object_pose = {object_pose}")
        # print(f"goal_object_pose = {goal_object_pose}")
        # print(f"object_scales = {object_scales}")
        # print(f"new_obs = {new_obs}")
        # breakpoint()

        DEBUG = False
        if DEBUG:
            diff= (obs['obs'] - new_obs).abs()[0]
            print(f"diff = {diff}")
            print(f"diff.max() = {diff.max()}")
            print(f"diff.argsort() = {diff.argsort()}")

            from isaacgymenvs.utils.observation_action_utils_sharpa import OBS_NAMES
            idxs = diff.argsort()
            for idx in idxs:
                print(f"OBS_NAMES[{idx}] = {OBS_NAMES[idx]}")
                print(f"obs['obs'][{idx}] = {obs['obs'][0, idx]}")
                print(f"new_obs[{idx}] = {new_obs[0, idx]}")
                print(f"diff[{idx}] = {diff[idx]}")
                print(f"--------------------------------")

            breakpoint()
        return new_obs, reward, done, info


def main():
    CONTROL_DT = 1.0 / 60.0
    CONFIG_PATH = Path(
        "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/config.yaml"
    )
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path(
        "/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/newGains.pth"
    )
    assert CHECKPOINT_PATH.exists()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # DEVICE = "cpu"  # "cpu" faster for single env, but some bugs with cpu like force sensors not working
    env = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=DEVICE,
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.resetRotationNoise": 0.0,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
        },
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(CHECKPOINT_PATH)
    env_state = checkpoint[0]["env_state"]
    env.set_env_state(env_state)

    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )

    chain, _ = create_chain_and_serial_chain(
        device=DEVICE, robot_name="iiwa14_left_sharpa_adjusted_restricted"
    )

    isaac_env_no_ros_joint_pos_targets = IsaacEnvNoRosJointPosTargets(
        env=env,
        control_dt=CONTROL_DT,
        device=DEVICE,
        chain=chain,
    )
    observation = isaac_env_no_ros_joint_pos_targets.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, done, _ = (
            isaac_env_no_ros_joint_pos_targets.step_with_joint_pos_targets(action)
        )
        if done.item():
            observation = isaac_env_no_ros_joint_pos_targets.reset()
        end_time = time.time()
        sleep_time = CONTROL_DT - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(
                f"Control loop too slow! Desired FPS: {1.0 / CONTROL_DT:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}"
            )


if __name__ == "__main__":
    main()
