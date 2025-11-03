import sys
sys.path.append("/home/tylerlum/github_repos/sapg")
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


class IsaacEnvNoRos:
    def __init__(self, sim: AllegroKukaBase, control_dt: float, device: str):
        self.sim = sim
        self.device = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.sim.step(action)
        return obs["obs"], reward, done, info

    def reset(self) -> torch.Tensor:
        obs, reward, done, info = self.sim.step(torch.zeros((1, 23), device=self.device))
        return obs["obs"]

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

    isaac_env_no_ros = IsaacEnvNoRos(sim=sim, control_dt=control_dt, device=device)
    observation = isaac_env_no_ros.reset()

    while True:
        start_time = time.time()
        action = policy.get_normalized_action(observation, deterministic_actions=True)  # Careful about deterministic_actions=True here!
        observation, _, _, _ = isaac_env_no_ros.step(action)
        end_time = time.time()
        sleep_time = control_dt - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Control loop too slow! Desired FPS: {1.0 / control_dt:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}")


if __name__ == "__main__":
    main()


