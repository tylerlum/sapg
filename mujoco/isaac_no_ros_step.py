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
        self.prev_targets = self.sim.prev_targets.clone()

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        joint_pos_targets = compute_joint_pos_targets(
            actions=action,
            prev_targets=self.prev_targets,
            act_moving_average=0.1,
            hand_dof_speed_scale=1.0,
            dt=1 / 60,
        )

        obs, reward, done, info = self.sim.step(action, joint_pos_targets=joint_pos_targets)
        new_obs = compute_observation(
            q=self.sim.arm_hand_dof_pos,
            qd=self.sim.arm_hand_dof_vel,
            object_pose=self.sim.object_pose,
            goal_object_pose=self.sim.goal_pose,
            object_scales=self.sim.object_scales,
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        self.prev_targets = joint_pos_targets.clone()
        diff = new_obs - obs["obs"]
        max_diff = diff.abs().max()
        print(f"max_diff: {max_diff}")
        return new_obs, reward, done, info


def main():
    sim_dt = 1.0 / 60.0
    control_dt = 1.0 / 60.0
    CONFIG_PATH = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-20_slow-action-obs-randomize-all/00_slowarmhand_slowobs_hammer_2025-10-21_02-39-06/runs/00_slowarmhand_slowobs_hammer_2025-10-21_02-39-06/config.yaml")
    assert Path(CONFIG_PATH).exists()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = create_env(
        config_path=str(CONFIG_PATH),
        headless=False,
        device=device,
    )
    policy = RlPlayer(
        num_observations=117,
        num_actions=23,
        config_path=Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-17_slow-action_randomize_turn-off-obs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/runs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/config.yaml"),
        checkpoint_path=Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-17_slow-action_randomize_turn-off-obs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/runs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/last/model.pth"),
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
    observation, _, _, _ = sim_no_ros.step(torch.zeros((1, 23), device=device))

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation)
        observation, _, _, _ = sim_no_ros.step(action)
        end_time = time.time()
        sleep_time = control_dt - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Control loop too slow! Desired FPS: {1.0 / control_dt:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}")


if __name__ == "__main__":
    main()


