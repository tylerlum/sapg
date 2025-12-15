# gymtorch must be imported before torch
from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip

from pathlib import Path
from typing import Optional, Dict, Any

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict


def read_cfg(config_path: str) -> dict:
    omegaconf_cfg = OmegaConf.load(config_path)
    return omegaconf_to_dict(omegaconf_cfg)

def load_cfg(checkpoint_dir: Path) -> dict:
    model_name = checkpoint_dir.name
    checkpoint_path = checkpoint_dir / "runs" / model_name / "best" / "model.pth"
    config_path = checkpoint_dir / "runs" / f"00_{model_name}" / "config.yaml"
    cfg = read_cfg(config_path)
    cfg["train"]["load_path"] = checkpoint_path
    return cfg

def create_env_from_cfg(
    cfg: DictConfig,
    num_envs: int = 1,
    headless: bool = False,
    enable_viewer_sync_at_start: bool = True,
    episode_length: Optional[int] = 600,
    overrides: Optional[Dict[str, Any]] = None,
) -> AllegroKukaBase:
    # Modify the config
    cfg["headless"] = headless
    cfg["task"]["sim"]["enable_viewer_sync_at_start"] = enable_viewer_sync_at_start
    cfg["task"]["env"]["numEnvs"] = num_envs
    cfg["graphics_device_id"] = 0
    # Modify the config for the task
    if overrides is not None:
        # Example: overrides = {"task.env.asset.kukaAllegro": "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted.urdf"}
        for key, value in overrides.items():
            if isinstance(value, str):
                value = f'"{value}"'
            else:
                value = str(value)
            eval_str = f"cfg.{key} = {value}"
            print(f"Evaluating: {eval_str}")
            exec(eval_str)

    print_dict(cfg)

    env = isaacgym_task_map[cfg["task_name"]](
        cfg=cfg["task"],
        sim_device=cfg["sim_device"],
        rl_device=cfg["rl_device"],
        graphics_device_id=cfg["graphics_device_id"],
        headless=cfg["headless"],
        virtual_screen_capture=False,
        force_render=True,
    )
    return env

