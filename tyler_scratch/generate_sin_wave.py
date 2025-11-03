from recorded_data_scripts.recorded_data import RecordedData
from typing import Literal
from pathlib import Path
from dataclasses import dataclass
import tyro
import numpy as np


# Home joint positions
HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])
HOME_JOINT_POS_ALLEGRO = np.zeros(16)
HOME_JOINT_POS_ALLEGRO[12] = 0.6
HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_ALLEGRO])

JOINT_NAMES = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
    "joint_0.0",
    "joint_1.0",
    "joint_2.0",
    "joint_3.0",
    "joint_4.0",
    "joint_5.0",
    "joint_6.0",
    "joint_7.0",
    "joint_8.0",
    "joint_9.0",
    "joint_10.0",
    "joint_11.0",
    "joint_12.0",
    "joint_13.0",
    "joint_14.0",
    "joint_15.0",
]


@dataclass
class SinWaveCfg:
    dt: float = 1 / 60  # 60 Hz
    total_time: float = 10.0  # 10 seconds
    period: float = 2.0  # 2 seconds
    amplitude: float = 0.1  # 0.1 radians

    @property
    def T(self) -> int:
        return int(self.total_time / self.dt)


def generate_sin_wave(cfg: SinWaveCfg, output_dir: Path, mode: Literal["hand", "arm"]) -> np.ndarray:
    # Validate inputs
    assert mode in ["hand", "arm"], f"Invalid mode: {mode}"

    # Generate time array and sin wave
    time_array = np.linspace(0, cfg.total_time, cfg.T)
    sin_wave = cfg.amplitude * np.sin(2 * np.pi * time_array / cfg.period)
    assert time_array.shape == sin_wave.shape == (cfg.T,), f"Expected time_array.shape and sin_wave.shape to be (cfg.T,), got {time_array.shape} and {sin_wave.shape}"

    # Add to home position
    J = len(JOINT_NAMES)
    robot_joint_positions_array = HOME_JOINT_POS.copy()[None].repeat(cfg.T, axis=0)
    assert robot_joint_positions_array.shape == (cfg.T, J), f"Expected robot_joint_positions_array.shape to be (cfg.T, {J}), got {robot_joint_positions_array.shape}"

    if mode == "hand":
        robot_joint_positions_array[:, 7:] += sin_wave[..., None]
    elif mode == "arm":
        robot_joint_positions_array[:, :7] += sin_wave[..., None]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    robot_root_states_array = np.zeros((cfg.T, 13))
    robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
    object_root_states_array = np.zeros((cfg.T, 13))
    object_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
    recorded_data = RecordedData(
        robot_root_states_array=robot_root_states_array,
        object_root_states_array=object_root_states_array,
        robot_joint_positions_array=robot_joint_positions_array,
        time_array=time_array,
        robot_joint_names=JOINT_NAMES,
    )
    output_path = output_dir / (f"sin_wave_{mode}_{cfg.total_time}s_{cfg.period}s_{cfg.amplitude}rad".replace(".", "-") + ".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving recorded data to {output_path}")
    recorded_data.to_file(output_path)
    return recorded_data


def main():
    tyro.cli(generate_sin_wave)

if __name__ == "__main__":
    main()

