import os
from typing import Optional

import numpy as np
import torch
from gym import spaces
from rl_games.torch_runner import Runner, players

from sim2real.rl_player_utils import (
    read_cfg,
)


def assert_equals(a, b):
    assert a == b, f"{a} != {b}"


class RlPlayer:
    def __init__(
        self,
        cfg: dict,
        device: str,
    ) -> None:
        self.device = device
        self.set_env_state = lambda *args, **kwargs: None

        self.cfg = cfg
        self.num_observations = cfg["task"]["env"]["numObservations"]
        self.num_actions = cfg["task"]["env"]["numActions"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_actions,), dtype=np.float32
        )
        self.num_envs = cfg["task"]["env"]["numEnvs"]
        self.player = self.create_rl_player()

    def create_rl_player(self) -> players.PpoPlayerContinuous:
        from rl_games.common import env_configurations

        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kwargs: self, "vecenv_type": "RLGPU"}
        )
        runner = Runner()
        runner.load(self.cfg['train'])

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        player = runner.create_player()
        player.init_rnn()
        player.has_batch_dimension = True
        player.restore(self.cfg['train']['load_path'])
        return player

    def get_normalized_action(
        self, obs: torch.Tensor, deterministic_actions: bool = True
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        assert_equals(obs.shape, (batch_size, self.num_observations))

        # SAPG HACK: Need to idx to end of observation
        obs = torch.cat(
            [obs, 50.0 + torch.zeros((batch_size, 1), device=self.device)], dim=1
        )

        normalized_action = self.player.get_action(
            obs=obs, is_deterministic=deterministic_actions
        )
        normalized_action = normalized_action.reshape(-1, self.num_actions)
        assert_equals(normalized_action.shape, (batch_size, self.num_actions))
        return normalized_action


def main() -> None:
    from pathlib import Path

    device = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIG_PATH = Path(
        "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-17_slow-action_randomize_turn-off-obs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/runs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/config.yaml"
    )
    CHECKPOINT_PATH = Path(
        "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-10-17_slow-action_randomize_turn-off-obs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/runs/00_slow-arm-hand-slowly_marker_2025-10-18_14-37-58/last/model.pth"
    )
    NUM_OBSERVATIONS = 117
    NUM_ACTIONS = 23

    player = RlPlayer(
        num_observations=NUM_OBSERVATIONS,
        num_actions=NUM_ACTIONS,
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
        device=device,
    )

    batch_size = 1
    obs = torch.zeros(batch_size, NUM_OBSERVATIONS).to(device)
    normalized_action = player.get_normalized_action(
        obs=obs, deterministic_actions=True
    )  # Careful about deterministic_actions=True here!
    print(f"Using player with config: {CONFIG_PATH} and checkpoint: {CHECKPOINT_PATH}")
    print(f"And num_observations: {NUM_OBSERVATIONS} and num_actions: {NUM_ACTIONS}")
    print(f"Sampled obs: {obs} with shape: {obs.shape}")
    print(
        f"Got normalized_action: {normalized_action} with shape: {normalized_action.shape}"
    )
    print(f"player: {player.player.model}")


if __name__ == "__main__":
    main()
