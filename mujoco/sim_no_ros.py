import sys
sys.path.append("/home/tylerlum/github_repos/sapg/sim2real")
import time
import torch
from pathlib import Path
from sim2real.rl_player import RlPlayer
import numpy as np
from sim import JOINT_NAMES, N_IIWA_JOINTS, Simulator, SimulatorConfig
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


class SimNoRos:
    def __init__(self, sim: Simulator, object_scales: np.ndarray, chain: pk.Chain, palm_serial_chain: pk.SerialChain, act_moving_average: float, hand_dof_speed_scale: float, control_dt: float, device: str):
        self.sim = sim
        self.object_scales = object_scales
        self.chain = chain
        self.palm_serial_chain = palm_serial_chain
        self.act_moving_average = act_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device

    def compute_observation(self) -> torch.Tensor:
        sim_state = self.sim.get_sim_state()

        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_pose_R = np.concatenate([object_pos, object_quat_wxyz])

        table_pos = sim_state["table_pos"]
        table_quat_wxyz = sim_state["table_quat_wxyz"]
        goal_object_pos = table_pos + np.array([0.0, 0.0, 0.5])
        goal_object_quat_wxyz = table_quat_wxyz
        goal_object_pose_R = np.concatenate([goal_object_pos, goal_object_quat_wxyz])

        q = sim_state["joint_positions"]
        qd = sim_state["joint_velocities"]

        observation = compute_observation(
            q=torch.from_numpy(q).float().to(self.device)[None],
            qd=torch.from_numpy(qd).float().to(self.device)[None],
            object_pose=torch.from_numpy(object_pose_R).float().to(self.device)[None],
            goal_object_pose=torch.from_numpy(goal_object_pose_R)
            .float()
            .to(self.device)[None],
            object_scales=torch.from_numpy(self.object_scales).float().to(self.device)[None],
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        assert observation.shape == (1, 117,), f"observation.shape: {observation.shape}, expected: (1, 117,)"
        print(f"observation: {observation[0]}")
        return observation

    def step(self, action: torch.Tensor) -> None:
        joint_cmd = compute_joint_pos_targets(
            actions=action,
            prev_targets=torch.from_numpy(self.sim.robot_joint_pos_targets).float().to(self.device)[None],
            act_moving_average=self.act_moving_average,
            hand_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
        )
        self.sim.set_robot_joint_pos_targets(joint_cmd.squeeze(dim=0).cpu().numpy())

        for _ in range(self.sim_steps_per_control_step):
            self.sim.sim_step()

            if self.sim.config.enable_viewer:
                self.sim.viewer.sync()

        return

    @property
    def sim_steps_per_control_step(self) -> int:
        return int(self.control_dt / self.sim.config.sim_dt)


def main():
    sim_dt = 1.0 / 600.0
    control_dt = 1.0 / 60.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = Simulator(SimulatorConfig(enable_viewer=True, sim_dt=sim_dt))
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

    sim_no_ros = SimNoRos(sim=sim, object_scales=np.array([4.0, 1.0, 1.0]), chain=chain, palm_serial_chain=palm_serial_chain, act_moving_average=0.1, hand_dof_speed_scale=1.0, control_dt=control_dt, device=device)

    while True:
        start_time = time.time()
        observation = sim_no_ros.compute_observation()
        action = policy.get_normalized_action(observation)
        sim_no_ros.step(action)
        end_time = time.time()
        sleep_time = control_dt - (end_time - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print(f"Control loop too slow! Desired FPS: {1.0 / control_dt:.1f}, Actual FPS: {1.0 / (end_time - start_time):.1f}")


if __name__ == "__main__":
    main()

