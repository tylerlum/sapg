import json
import time

import numpy as np
import viser
from scipy.spatial.transform import Rotation as R
from viser.extras import ViserUrdf

from isaacgymenvs.utils.objects import NAME_TO_OBJECT
from isaacgymenvs.utils.utils import get_repo_root_dir


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    return (q[3], q[0], q[1], q[2])


class GoalTrajectoryCreator:
    def __init__(self, object_name, port=8080):
        self.port = port
        self.object_name = object_name
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.start_pose = None
        self.goals = []
        self._setup_scene()

    def _setup_scene(self):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.0, -0.8, 1.0)
            client.camera.look_at = (0.0, 0.0, 0.7)

        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
        table_urdf = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
        self.server.scene.add_frame("/table", position=(0, 0, 0.38), wxyz=(1, 0, 0, 0), show_axes=False)
        ViserUrdf(self.server, table_urdf, root_node_name="/table", mesh_color_override=(0, 0, 0, 0.5))

        object_urdf = NAME_TO_OBJECT[self.object_name].filepath
        self.object_control = self.server.scene.add_transform_controls(
            "/object_control", position=(0.0, 0.0, 0.85), wxyz=(1, 0, 0, 0), scale=0.15,
        )
        ViserUrdf(self.server, object_urdf, root_node_name="/object_control")

        self.server.gui.add_markdown(f"**Object:** {self.object_name}")
        self.server.gui.add_markdown("*Goal min Z >= 0.63 (table 0.38 + offset 0.25)*")
        self.current_pose_text = self.server.gui.add_markdown("**Current:** --")
        self.server.gui.add_markdown("---")

        # Position sliders
        self.slider_x = self.server.gui.add_slider("X", min=-0.3, max=0.3, step=0.01, initial_value=0.0)
        self.slider_y = self.server.gui.add_slider("Y", min=-0.3, max=0.3, step=0.01, initial_value=0.0)
        self.slider_z = self.server.gui.add_slider("Z", min=0.5, max=1.2, step=0.01, initial_value=0.85)
        # Rotation sliders (euler angles in degrees)
        self.slider_roll = self.server.gui.add_slider("Roll", min=-180, max=180, step=1, initial_value=0)
        self.slider_pitch = self.server.gui.add_slider("Pitch", min=-180, max=180, step=1, initial_value=0)
        self.slider_yaw = self.server.gui.add_slider("Yaw", min=-180, max=180, step=1, initial_value=0)

        self.slider_x.on_update(lambda _: self._update_from_sliders())
        self.slider_y.on_update(lambda _: self._update_from_sliders())
        self.slider_z.on_update(lambda _: self._update_from_sliders())
        self.slider_roll.on_update(lambda _: self._update_from_sliders())
        self.slider_pitch.on_update(lambda _: self._update_from_sliders())
        self.slider_yaw.on_update(lambda _: self._update_from_sliders())

        self.server.gui.add_markdown("---")
        self.start_pose_text = self.server.gui.add_markdown("**Start:** --")
        self.last_goal_text = self.server.gui.add_markdown("**Last goal:** --")
        self.status = self.server.gui.add_markdown("**Goals count:** 0")
        self.server.gui.add_markdown("---")
        self.trajectory_name = self.server.gui.add_text("Trajectory name", initial_value="horizontal_swing")
        self.server.gui.add_button("Set Start Pose").on_click(lambda _: self._set_start())
        self.server.gui.add_button("Add Goal").on_click(lambda _: self._add_goal())
        self.server.gui.add_button("Undo Last Goal").on_click(lambda _: self._undo_goal())
        self.server.gui.add_button("Save Trajectory").on_click(lambda _: self._save())

        self._updating_from_sliders = False

    def _get_pose(self):
        return np.concatenate([np.array(self.object_control.position), quat_wxyz_to_xyzw(self.object_control.wxyz)])

    def _update_from_sliders(self):
        self._updating_from_sliders = True
        pos = (self.slider_x.value, self.slider_y.value, self.slider_z.value)
        euler = np.deg2rad([self.slider_roll.value, self.slider_pitch.value, self.slider_yaw.value])
        quat_xyzw = R.from_euler("xyz", euler).as_quat()
        self.object_control.position = pos
        self.object_control.wxyz = quat_xyzw_to_wxyz(quat_xyzw)
        self._updating_from_sliders = False

    def _update_sliders_from_control(self):
        if self._updating_from_sliders:
            return
        pos = self.object_control.position
        quat_xyzw = quat_wxyz_to_xyzw(self.object_control.wxyz)
        euler = np.rad2deg(R.from_quat(quat_xyzw).as_euler("xyz"))
        self.slider_x.value = pos[0]
        self.slider_y.value = pos[1]
        self.slider_z.value = pos[2]
        self.slider_roll.value = euler[0]
        self.slider_pitch.value = euler[1]
        self.slider_yaw.value = euler[2]

    def _set_start(self):
        self.start_pose = self._get_pose()
        self.start_pose_text.content = f"**Start:** {self._format_pose(self.start_pose)}"
        self._update_status()

    def _add_goal(self):
        self.goals.append(self._get_pose())
        self._update_status()

    def _undo_goal(self):
        if self.goals:
            self.goals.pop()
            self._update_status()

    def _format_pose(self, pose):
        xyz = pose[:3].round(3).tolist()
        quat = pose[3:7].round(3).tolist()
        return f"xyz={xyz} quat={quat}"

    def _update_status(self):
        self.status.content = f"**Goals count:** {len(self.goals)}"
        if self.goals:
            self.last_goal_text.content = f"**Last goal:** {self._format_pose(self.goals[-1])}"
        else:
            self.last_goal_text.content = "**Last goal:** --"

    def _save(self):
        if self.start_pose is None or not self.goals:
            return
        output_dir = get_repo_root_dir() / "dex_tool_bench/evaluation_trajectories/hammer" / self.object_name
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self.trajectory_name.value.strip().replace(" ", "_") or "default"
        output_path = output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump({"start_pose": self.start_pose.tolist(), "goals": [g.tolist() for g in self.goals]}, f, indent=2)
        print(f"Saved: {output_path}")

    def run(self):
        print(f"Viser: http://localhost:{self.port}")
        while True:
            self._update_sliders_from_control()
            self.current_pose_text.content = f"**Current:** {self._format_pose(self._get_pose())}"
            time.sleep(0.1)


def main():
    GoalTrajectoryCreator("scanned_hammer_2").run()


if __name__ == "__main__":
    main()
