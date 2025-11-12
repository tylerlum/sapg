from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from recorded_data_scripts.recorded_data import RecordedData

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
    "joint_16.0",
    "joint_17.0",
    "joint_18.0",
    "joint_19.0",
    "joint_20.0",
    "joint_21.0",
]

ANGLE_RANGES = [
    (0, 50),   # Thumb CMC Flexion/Extension
    (0, 10),     # Thumb CMC Abduction/Adduction
    (0, 30),   # Thumb MCP Flexion/Extension
    (0, 10),     # Thumb MCP Abduction/Adduction
    (0, 40),     # Thumb DIP Flexion/Extension
    (0, 20),   # Index MCP Flexion/Extension
    (-20, 20),   # Index MCP Abduction/Adduction
    (0, 20),     # Index PIP Flexion/Extension
    (0, 20),     # Index DIP Flexion/Extension
    (0, 20),   # Middle MCP Flexion/Extension
    (-20, 20),   # Middle MCP Abduction/Adduction
    (0, 20),     # Middle PIP Flexion/Extension
    (0, 20),     # Middle DIP Flexion/Extension
    (0, 20),   # Ring MCP Flexion/Extension
    (-20, 20),   # Ring MCP Abduction/Adduction
    (0, 20),     # Ring PIP Flexion/Extension
    (0, 20),     # Ring DIP Flexion/Extension
    (0, 10),     # Pinky CMC Flexion/Extension
    (0, 20),   # Pinky MCP Flexion/Extension
    (-20, 20),   # Pinky MCP Abduction/Adduction
    (0, 20),     # Pinky PIP Flexion/Extension
    (0, 20),     # Pinky DIP Flexion/Extension
]


# Home joint positions
HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 1.309])
HOME_JOINT_POS_SHARPA = np.deg2rad(np.array([(high + low) / 2.0 for low, high in ANGLE_RANGES]))
HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])


@dataclass
class SinWaveCfg:
    dt: float = 1 / 60  # 60 Hz
    total_time: float = 10.0  # 10 seconds
    period: float = 2.0  # 2 seconds

    @property
    def T(self) -> int:
        return int(self.total_time / self.dt)


def generate_sin_wave(
    cfg: SinWaveCfg, output_dir: Path, mode: Literal["hand", "arm"]
) -> np.ndarray:
    # Validate inputs
    assert mode in ["hand", "arm"], f"Invalid mode: {mode}"

    # Generate time array and sin wave
    time_array = np.linspace(0, cfg.total_time, cfg.T)
    sin_wave = np.sin(2 * np.pi * time_array / cfg.period)
    assert time_array.shape == sin_wave.shape == (cfg.T,), (
        f"Expected time_array.shape and sin_wave.shape to be (cfg.T,), got {time_array.shape} and {sin_wave.shape}"
    )

    # Add to home position
    NUM_JOINTS = len(JOINT_NAMES)
    NUM_SHARPA_JOINTS = len(JOINT_NAMES) - 7
    print(f"len(JOINT_NAMES): {len(JOINT_NAMES)}, NUM_SHARPA_JOINTS: {NUM_SHARPA_JOINTS}")
    print(f"len(ANGLE_RANGES): {len(ANGLE_RANGES)}")
    robot_joint_positions_list = []
    for t in range(cfg.T):
        robot_joint_positions = HOME_JOINT_POS.copy()
        for j in range(NUM_SHARPA_JOINTS):
            low, high = ANGLE_RANGES[j]
            low_rad, high_rad = np.deg2rad([low, high])
            amplitude = (high_rad - low_rad) / 2.0
            robot_joint_positions[7+j] = HOME_JOINT_POS_SHARPA[j] + (amplitude * sin_wave[t])
        robot_joint_positions_list.append(robot_joint_positions)
    robot_joint_positions_array = np.array(robot_joint_positions_list)
    assert robot_joint_positions_array.shape == (cfg.T, NUM_JOINTS), (
        f"Expected robot_joint_positions_array.shape to be (cfg.T, {NUM_JOINTS}), got {robot_joint_positions_array.shape}"
    )

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
    output_path = output_dir / (
        f"sharpa_sin_wave_{mode}_{cfg.total_time}s_{cfg.period}s".replace(
            ".", "-"
        )
        + ".npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving recorded data to {output_path}")
    recorded_data.to_file(output_path)
    return recorded_data


def main():
    tyro.cli(generate_sin_wave)


if __name__ == "__main__":
    main()
