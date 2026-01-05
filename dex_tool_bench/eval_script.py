from isaacgymenvs.tasks.allegro_kuka.allegro_kuka_base import AllegroKukaBase  # isort:skip

import json
import time
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import torch  # isort:skip
import viser
from termcolor import colored
from viser.extras import ViserUrdf

from isaacgymenvs.utils.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir
from sim2real.rl_player import RlPlayer
from sim2sim.isaac_sim.isaac_env import create_env


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


class ViserServer:
    def __init__(self, object_name, trajectory_name, port=8080):
        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup_scene(object_name, trajectory_name)

    def _setup_scene(self, object_name, trajectory_name):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -1.0, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.5)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        robot_urdf = get_repo_root_dir() / "assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        self.server.scene.add_frame("/robot", position=(0, 0.8, 0), wxyz=(1, 0, 0, 0), show_axes=False)
        self.robot = ViserUrdf(self.server, robot_urdf, root_node_name="/robot")
        self.robot.update_cfg(np.zeros(29))

        table_urdf = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
        self.server.scene.add_frame("/table", position=(0, 0, 0.38), wxyz=(1, 0, 0, 0), show_axes=False)
        ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 0.5))

        object_urdf = NAME_TO_OBJECT[object_name].filepath
        self.object_frame = self.server.scene.add_frame("/object", show_axes=True, axes_length=0.1, axes_radius=0.001)
        ViserUrdf(self.server, object_urdf, root_node_name="/object")
        self.goal_frame = self.server.scene.add_frame("/goal", show_axes=True, axes_length=0.1, axes_radius=0.001)
        ViserUrdf(self.server, object_urdf, root_node_name="/goal", mesh_color_override=(0, 255, 0, 0.5))

        self.server.gui.add_markdown(f"**Task:** {trajectory_name}")
        self.server.gui.add_markdown(f"**Object:** {object_name}")
        self.server.gui.add_markdown("---")
        self.progress_text = self.server.gui.add_markdown("**Progress:** --")
        self.stats_text = self.server.gui.add_markdown("**Stats:** No episodes completed")

    def update_progress(self, current, total, timestep, control_hz=60.0):
        pct = 100 * current / total if total > 0 else 0
        self.progress_text.content = f"**Time:** {timestep / control_hz:.1f}s | **Goal:** {current}/{total} ({pct:.0f}%)"

    def update_stats(self, num_episodes, avg_goal_pct, avg_time_sec):
        self.stats_text.content = f"**Episodes:** {num_episodes} | **Avg Goal:** {avg_goal_pct:.1f}% | **Avg Time:** {avg_time_sec:.1f}s"

    def update(self, joint_pos, object_pose, goal_pose):
        self.robot.update_cfg(joint_pos)
        self.object_frame.position = object_pose[:3]
        self.object_frame.wxyz = quat_xyzw_to_wxyz(object_pose[3:7])
        self.goal_frame.position = goal_pose[:3]
        self.goal_frame.wxyz = quat_xyzw_to_wxyz(goal_pose[3:7])

    def get_frame(self):
        clients = list(self.server.get_clients().values())
        return clients[0].camera.get_render(height=480, width=640) if clients else np.zeros((480, 640, 3), dtype=np.uint8)

    def add_run_button(self, callback):
        self.server.gui.add_button("Run Episode").on_click(lambda _: callback())


class IsaacEnvNoRos:
    def __init__(self, env, config_path, checkpoint_path, object_name, trajectory_name, output_dir=None):
        self.env = env
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_act = 29
        self.control_hz = 60.0
        self.control_dt = 1.0 / self.control_hz
        self.record_fps = 10
        self.record_interval = int(self.control_hz / self.record_fps)

        self.joint_lower = env.arm_hand_dof_lower_limits[:self.n_act].cpu().numpy()
        self.joint_upper = env.arm_hand_dof_upper_limits[:self.n_act].cpu().numpy()

        self.env.set_env_state(torch.load(checkpoint_path)[0]["env_state"])
        self.policy = RlPlayer(140, self.n_act, config_path, checkpoint_path, self.device, env.num_envs)

        self.record_video = output_dir is not None
        self.episode_count, self.episode_goal_pcts, self.episode_lengths = 0, [], []
        if self.record_video:
            self.session_dir = output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
            print(colored(f"Recording to: {self.session_dir}", "cyan"))

        self.viser = ViserServer(object_name, trajectory_name)
        print(colored(f"Viser: http://localhost:{self.viser.port}", "green"))
        self.obs = self._reset()

    def _reset(self):
        obs, _, _, _ = self.env.step(torch.zeros((self.env.num_envs, self.n_act), device=self.device))
        return obs["obs"]

    def _step(self, action):
        obs, _, done, _ = self.env.step(action)
        return obs["obs"], done[0].item()

    def _get_state(self):
        obs_np = self.obs[0].cpu().numpy()
        joint_pos = 0.5 * (obs_np[:29] + 1.0) * (self.joint_upper - self.joint_lower) + self.joint_lower
        return joint_pos, self.env.object_state[0, :7].cpu().numpy(), self.env.goal_pose[0].cpu().numpy()

    def _sim_step(self, timestep):
        t0 = time.time()
        self.viser.update(*self._get_state())
        self.obs, done = self._step(self.policy.get_normalized_action(self.obs, deterministic_actions=True))
        self.viser.update_progress(int(self.env.successes[0].item()), self.env.max_consecutive_successes, timestep, self.control_hz)
        if (sleep := self.control_dt - (time.time() - t0)) > 0:
            time.sleep(sleep)
        return done

    def _render_video(self, states, path):
        print(colored(f"Rendering {len(states)} frames...", "cyan"))
        frames = []
        for i, state in enumerate(states):
            self.viser.update(*state)
            time.sleep(0.05)
            frames.append(self.viser.get_frame())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(states)}")
        imageio.mimsave(str(path), frames, fps=self.record_fps)
        print(colored(f"Saved: {path}", "green"))

    def _run_episode(self):
        self.policy.reset()
        print(colored("Reset...", "cyan"))
        self.obs = self._reset()
        self.viser.update(*self._get_state())

        print(colored(f"Running{' (recording)' if self.record_video else ''}...", "green"))
        states, step, done = [], 0, False
        while not done:
            if self.record_video and step % self.record_interval == 0:
                states.append(tuple(x.copy() for x in self._get_state()))
            done = self._sim_step(step)
            step += 1

        goal_pct = 100 * int(self.env.successes[0].item()) / self.env.max_consecutive_successes
        self.episode_goal_pcts.append(goal_pct)
        self.episode_lengths.append(step)
        self.episode_count += 1
        avg_goal_pct = sum(self.episode_goal_pcts) / len(self.episode_goal_pcts)
        avg_time_sec = sum(self.episode_lengths) / len(self.episode_lengths) / self.control_hz
        self.viser.update_stats(self.episode_count, avg_goal_pct, avg_time_sec)
        print(colored(f"Done: {step / self.control_hz:.1f}s, {goal_pct:.0f}% goals", "green"))

        if states and self.record_video:
            self._render_video(states, self.session_dir / f"{self.episode_count}.mp4")

    def run(self):
        self.viser.add_run_button(self._run_episode)
        print(colored(f"Open http://localhost:{self.viser.port}", "cyan"))
        print(colored("Click 'Run Episode' to start.", "cyan"))
        while True:
            time.sleep(1.0)


def parse_checkpoint_dir(path):
    run_dir = sorted(d for d in (path / "runs").iterdir() if d.is_dir())[0]
    return run_dir / "config.yaml", run_dir / "last" / "model.pth"


def main():
    checkpoint_dir = Path("/share/portal/kk837/sapg/train_dir/customPretraining/FINETUNE_4x/FINETUNE_4x_SLOWSPEED_ADD_ACTION_DELAY_2026-01-03_01-32-46")
    object_type = "hammer"
    object_name = "scanned_hammer_2"
    trajectory_name = "horizontal_swing"
    output_dir = None  # Set to Path("videos") to enable recording

    trajectory_path = get_repo_root_dir() / "dex_tool_bench/evaluation_trajectories" / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory file not found: {trajectory_path}"
    with open(trajectory_path) as f:
        traj_data = json.load(f)

    config_path, checkpoint_path = parse_checkpoint_dir(checkpoint_dir)
    env = create_env(
        config_path=str(config_path),
        headless=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        overrides={
            "task.env.resetPositionNoiseX": 0.0,
            "task.env.resetPositionNoiseY": 0.0,
            "task.env.resetPositionNoiseZ": 0.0,
            "task.env.resetRotationNoise": 0.0,
            "task.env.resetDofPosRandomIntervalFingers": 0.0,
            "task.env.resetDofPosRandomIntervalArm": 0.0,
            "task.env.resetDofVelRandomInterval": 0.0,
            "task.env.object_type": object_name,
            "task.env.randomizeObjectRotation": False,
            "task.env.forceScale": 0.0,
            "task.env.numEnvs": 1,
            "task.env.envSpacing": 0.4,
            "task.env.tableResetRange": 0.0,
            "task.env.capture_video": False,
            "task.env.use_fixed_set_of_goal_states": True,
            "task.env.fixedGoalStates": traj_data["goals"],
            "task.env.objectStartPose": traj_data["start_pose"],
        },
    )
    IsaacEnvNoRos(env, config_path, checkpoint_path, object_name, trajectory_name, output_dir).run()


if __name__ == "__main__":
    main()
