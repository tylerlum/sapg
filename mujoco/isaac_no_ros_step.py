import sys
sys.path.append("/home/tylerlum/github_repos/sapg/sim2real")
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
from typing import Tuple
import time
from pathlib import Path
from sim2real.rl_player import RlPlayer
import numpy as np
import torch  # isort:skip
from isaac import create_env
from termcolor import colored
import pytorch_kinematics as pk

from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
)

def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class IsaacNoRos:
    def __init__(self, sim: AllegroKukaBase, control_dt: float, device: str, chain: pk.Chain, palm_serial_chain: pk.SerialChain):
        self.sim = sim
        self.device = device
        self.chain = chain
        self.palm_serial_chain = palm_serial_chain
        # self.prev_targets = self.sim.prev_targets.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.sim.step(action)
        q = self.sim.arm_hand_dof_pos
        qd = self.sim.arm_hand_dof_vel
        object_pose = self.sim.object_pose
        goal_object_pose = self.sim.goal_pose
        object_scales = self.sim.object_scales
        print("In step:")
        print(f"q = {q}")
        print(f"qd = {qd}")
        print(f"object_pose = {object_pose}")
        print(f"goal_object_pose = {goal_object_pose}")
        print(f"object_scales = {object_scales}")
        new_obs = compute_observation(
            q=q,
            qd=qd,
            object_pose=object_pose,
            goal_object_pose=goal_object_pose,
            object_scales=object_scales,
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        # self.prev_targets = joint_pos_targets.clone()
        diff = new_obs - obs["obs"]
        max_diff = diff.abs().max()
        print(f"in step, max_diff: {max_diff}")
        return new_obs, reward, done, info
        # return obs["obs"], reward, done, info

    def reset(self) -> torch.Tensor:
        obs, reward, done, info = self.sim.step(torch.zeros((1, 23), device=self.device))
        return obs["obs"]

    def step_with_joint_pos_targets(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        # breakpoint()
        # print(f"self.sim.prev_targets: {self.sim.prev_targets}")
        # print(f"self.sim.arm_hand_dof_pos: {self.sim.arm_hand_dof_pos}")
        joint_pos_targets = compute_joint_pos_targets(
            actions=action,
            prev_targets=self.sim.prev_targets,
            act_moving_average=0.1,
            hand_dof_speed_scale=1.0,
            dt=1 / 60,
        )

        obs, reward, done, info = self.sim.step(action, joint_pos_targets=joint_pos_targets)
        q = self.sim.arm_hand_dof_pos
        qd = self.sim.arm_hand_dof_vel
        object_pose = self.sim.object_pose
        goal_object_pose = self.sim.goal_pose
        object_scales = self.sim.object_scales
        print("In step_with_joint_pos_targets:")
        print(f"q = {q}")
        print(f"qd = {qd}")
        print(f"object_pose = {object_pose}")
        print(f"goal_object_pose = {goal_object_pose}")
        print(f"object_scales = {object_scales}")
        breakpoint()
        new_obs = compute_observation(
            q=q,
            qd=qd,
            object_pose=object_pose,
            goal_object_pose=goal_object_pose,
            object_scales=object_scales,
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        # self.prev_targets = joint_pos_targets.clone()
        diff = new_obs - obs["obs"]
        max_diff = diff.abs().max()
        print(f"max_diff: {max_diff}")
        return new_obs, reward, done, info


def main():
    control_dt = 1.0 / 60.0
    CONFIG_PATH = Path("/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml")
    assert Path(CONFIG_PATH).exists()
    CHECKPOINT_PATH = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-22_slow-action-obs-randomize-all_slower-curriculum/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/runs/00_slowarmhand_slowobs_hammer_2025-10-23_00-48-56/last/model.pth")
    assert CHECKPOINT_PATH.exists()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=device,
    )

    # Set env state from checkpoint to match things like success_tolerance
    checkpoint = torch.load(CHECKPOINT_PATH)
    env_state = checkpoint[0]["env_state"]
    sim.set_env_state(env_state)
    # print(f"sim.success_tolerance: {sim.success_tolerance}")

    policy = RlPlayer(
        num_observations=117,
        num_actions=23,
        config_path=CONFIG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        device="cuda",
    )

    asset_root = Path(__file__).parent / "../assets"
    urdf_path = (
        asset_root / "urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
    )
    assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"
    chain = pk.build_chain_from_urdf(
        open(urdf_path).read(),
    ).to(device=device)
    palm_serial_chain = pk.SerialChain(chain, "iiwa7_link_7").to(
        device=device
    )

    sim_no_ros = IsaacNoRos(sim=sim, control_dt=control_dt, device=device, chain=chain, palm_serial_chain=palm_serial_chain)
    observation = sim_no_ros.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)
        observation, _, done, _ = sim_no_ros.step_with_joint_pos_targets(action)
        if done.item():
            observation = sim_no_ros.reset()
        end_time = time.time()
        sleep_time = control_dt - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Control loop too slow! Desired FPS: {1.0 / control_dt:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}")


if __name__ == "__main__":
    main()


