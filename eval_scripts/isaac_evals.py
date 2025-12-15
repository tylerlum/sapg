from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip
import time
from pathlib import Path
from typing import Tuple

# from eval_scripts.rl_player import RlPlayer
from sim2real.rl_player import RlPlayer

import torch  # isort:skip
from termcolor import colored

from eval_scripts.isaac_utils import create_env_from_cfg
from eval_scripts.isaac_utils import load_cfg
import argparse


class IsaacEnvNoRos:
    def __init__(self, env: AllegroKukaBase, device: str):
        self.env = env
        self.device = device

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return obs["obs"], reward, done, info

    def reset(self) -> torch.Tensor:
        obs, _, _, _ = self.env.step(torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device))
        return obs["obs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="/share/portal/kk837/sapg/train_dir/FINAL_ASYMMETRIC_RUNS/NEW_GAINS/2.5_Speed_controlFreqInv_1_successSteps_10_delta_2025-12-09_19-47-41")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    cfg = load_cfg(checkpoint_dir)
    env = create_env_from_cfg(cfg, 
        headless=True,
        episode_length=600,
        num_envs=10,
    )

    policy = RlPlayer(
        num_observations=cfg['task']['env']['numObservations'],
        num_actions=cfg['task']['env']['numActions'],
        config_path=cfg['train']['config_path'],
        checkpoint_path=cfg['train']['load_path'],
        device="cuda",
    )

    isaac_env_no_ros = IsaacEnvNoRos(env=env, device="cuda")
    observation = isaac_env_no_ros.reset()

    num_total_steps = int(1e4)
    for num_step in range(num_total_steps):
        print(f"Step {num_step} of {num_total_steps}")
        start_time = time.time()
        action = policy.get_normalized_action(
            observation, deterministic_actions=True
        )  # Careful about deterministic_actions=True here!
        breakpoint()
        observation, reward, done, info = isaac_env_no_ros.step(action)
        print("successes: ", info["successes"][0])
        if done[0]:
            breakpoint()
            print(info["successes"][0])
        end_time = time.time()
        control_hz = 1.0 / (end_time - start_time)
        print(f"Control Hz: {control_hz:.1f}")

    print("Done")

if __name__ == "__main__":
    main()
