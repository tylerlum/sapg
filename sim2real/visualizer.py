#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
import trimesh
import viser
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from termcolor import colored
from viser.extras import ViserUrdf


def get_ros_loop_rate_str(
    start_time: rospy.Time,
    before_sleep_time: rospy.Time,
    after_sleep_time: rospy.Time,
    node_name: Optional[str] = None,
) -> str:
    max_rate_dt = (before_sleep_time - start_time).to_sec()
    max_rate_hz = 1 / max_rate_dt
    actual_rate_dt = (after_sleep_time - start_time).to_sec()
    actual_rate_hz = 1 / actual_rate_dt
    loop_rate_str = f"Max rate: {np.round(max_rate_hz, 1)} Hz ({np.round(max_rate_dt * 1000, 1)} ms), Actual rate: {np.round(actual_rate_hz, 1)} Hz"
    return f"{node_name} {loop_rate_str}" if node_name is not None else loop_rate_str


NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

BLUE_RGB = (0, 0, 255)
RED_RGB = (255, 0, 0)
GREEN_RGB = (0, 255, 0)
YELLOW_RGB = (255, 255, 0)
CYAN_RGB = (0, 255, 255)
MAGENTA_RGB = (255, 0, 255)
WHITE_RGB = (255, 255, 255)
BLACK_RGB = (0, 0, 0)

AXES_LENGTH = 0.1
AXES_RADIUS = 0.001

NUM_HAND_KEYPOINTS = 12

# Viser Server global variable
SERVER = viser.ViserServer()
IMAGE_CREATED = False
FRUSTUM_CREATED = False


@SERVER.on_client_connect
def _(client: viser.ClientHandle) -> None:
    """For each client that connects, set the camera pose."""
    with client.atomic():
        client.camera.position = (0.0, -1.0, 1.03)
        client.camera.look_at = (0.0, 0.0, 0.53)
        # client.camera.wxyz = (w, x, y, z)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), T.shape
    n_pts = points.shape[0]
    assert points.shape == (n_pts, 3), points.shape

    return (T[:3, :3] @ points.T + T[:3, 3][:, None]).T


@dataclass
class RosSnapshot:
    iiwa_joint_cmd: Optional[np.ndarray]
    allegro_joint_cmd: Optional[np.ndarray]
    iiwa_joint_state: Optional[np.ndarray]
    allegro_joint_state: Optional[np.ndarray]
    object_pose: Optional[np.ndarray]
    goal_object_pose: Optional[np.ndarray]

    @classmethod
    def make_with_nones(cls) -> RosSnapshot:
        return cls(
            iiwa_joint_cmd=None,
            allegro_joint_cmd=None,
            iiwa_joint_state=None,
            allegro_joint_state=None,
            object_pose=None,
            goal_object_pose=None,
        )

    def make_copy_with_defaults(self) -> RosSnapshot:
        if self.iiwa_joint_cmd is None:
            print(colored("iiwa_joint_cmd is None", "yellow"))
            iiwa_joint_cmd = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_cmd = self.iiwa_joint_cmd

        if self.allegro_joint_cmd is None:
            print(colored("allegro_joint_cmd is None", "yellow"))
            allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_cmd = self.allegro_joint_cmd

        if self.iiwa_joint_state is None:
            print(colored("iiwa_joint_state is None", "yellow"))
            iiwa_joint_state = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_state = self.iiwa_joint_state

        if self.allegro_joint_state is None:
            print(colored("allegro_joint_state is None", "yellow"))
            allegro_joint_state = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_state = self.allegro_joint_state

        if self.object_pose is None:
            print(colored("object_pose is None", "yellow"))
            object_pose = np.eye(4)
            object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose = self.object_pose

        if self.goal_object_pose is None:
            print(colored("goal_object_pose is None", "yellow"))
            goal_object_pose = np.eye(4)
            goal_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose = self.goal_object_pose

        return RosSnapshot(
            iiwa_joint_cmd=iiwa_joint_cmd,
            allegro_joint_cmd=allegro_joint_cmd,
            iiwa_joint_state=iiwa_joint_state,
            allegro_joint_state=allegro_joint_state,
            object_pose=object_pose,
            goal_object_pose=goal_object_pose,
        )


class ViserVisualizationNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("viser_visualization_ros_node")

        # Store snapshot
        self.ros_snapshot = RosSnapshot.make_with_nones()

        # Subscribers
        self.initialize_ros_subscribers()

        # Initialize Viser
        self.initialize_viser()

        # Set update rate to 10Hz
        self.rate_hz = 10
        self.rate = rospy.Rate(self.rate_hz)

    def initialize_ros_subscribers(self):
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )

    def initialize_viser(self):
        SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        # Load robot URDF with a fixed base
        robot_urdf_path = Path(
            "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
        )
        assert robot_urdf_path.exists(), f"robot_urdf_path not found: {robot_urdf_path}"

        SERVER.scene.add_frame(
            "/robot/state",
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_frame(
            "/robot/cmd",
            position=(0, 0, 0),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        self.robot_viser = ViserUrdf(
            SERVER, robot_urdf_path, root_node_name="/robot/state"
        )
        self.robot_cmd_viser = ViserUrdf(
            SERVER,
            robot_urdf_path,
            root_node_name="/robot/cmd",
            mesh_color_override=BLUE_RGB,
        )

        # Set the cmd robot to be translucent
        # NOTE: To change opacity, you must create ViserUrdf with mesh_color_override
        for robot_cmd_mesh in self.robot_cmd_viser._meshes:
            assert isinstance(robot_cmd_mesh, viser.MeshHandle), (
                f"robot_cmd_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(robot_cmd_mesh)}"
            )
            robot_cmd_mesh.opacity = 0.5

        LOAD_TABLE = True
        if LOAD_TABLE:
            table_urdf_path = Path(
                "/home/tylerlum/github_repos/sapg/assets/urdf/table_narrow.urdf"
            )
            assert table_urdf_path.exists(), (
                f"table_urdf_path not found: {table_urdf_path}"
            )

            SERVER.scene.add_frame(
                "/table",
                position=(0, -0.8, 0.38),
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            table_viser = ViserUrdf(
                SERVER,
                table_urdf_path,
                root_node_name="/table",
                mesh_color_override=BLACK_RGB,
            )

            TRANSPARENT_TABLE = False
            if TRANSPARENT_TABLE:
                # NOTE: To change opacity, you must create ViserUrdf with mesh_color_override
                # Make the table transparent
                # Change the color of each link (including the base)
                for table_mesh in table_viser._meshes:
                    assert isinstance(table_mesh, viser.MeshHandle), (
                        f"table_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(table_mesh)}"
                    )
                    table_mesh.color = (0, 0, 0)
                    table_mesh.opacity = 0.5

        # Load the object mesh
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_path = rospy.get_param("/mesh_file", None)
        if object_mesh_path is None:
            DEFAULT_MESH_PATH = Path(
                "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/040_large_marker/040_large_marker/google_16k/textured_vhacd.obj"
            )
            object_mesh_path = str(DEFAULT_MESH_PATH)
            print(colored(f"Using default object mesh: {object_mesh_path}", "yellow"))
        assert isinstance(object_mesh_path, str), (
            f"object_mesh_path: {object_mesh_path}"
        )
        print("~" * 80)
        print(f"object_mesh_path: {object_mesh_path}")
        print("~" * 80 + "\n")

        goal_object_mesh_path = object_mesh_path

        object_mesh = trimesh.load(object_mesh_path)
        goal_object_mesh = trimesh.load(goal_object_mesh_path)
        self.object_viser = SERVER.scene.add_frame(
            "/object",
            position=FAR_AWAY_OBJECT_POSITION,
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        SERVER.scene.add_mesh_trimesh(name="/object/mesh", mesh=object_mesh)
        self.goal_object_viser = SERVER.scene.add_frame(
            "/goal_object",
            position=FAR_AWAY_OBJECT_POSITION + np.array([0.2, 0.2, 0.2]),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        SERVER.scene.add_mesh_simple(
            name="/goal_object/mesh",
            vertices=goal_object_mesh.vertices,
            faces=goal_object_mesh.faces,
            color=GREEN_RGB,
            opacity=0.5,
        )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.robot_viser.update_cfg(DEFAULT_Q)
        self.robot_cmd_viser.update_cfg(DEFAULT_Q)

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.ros_snapshot.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.ros_snapshot.allegro_joint_cmd = np.array(msg.position)

    def iiwa_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.ros_snapshot.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.ros_snapshot.allegro_joint_state = np.array(msg.position)

    def object_pose_callback(self, msg: Pose):
        """ "Callback to update the current object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.ros_snapshot.object_pose = latest_pose

    def goal_object_pose_callback(self, msg: Pose):
        """ "Callback to update the goal object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.ros_snapshot.goal_object_pose = latest_pose

    def update_viser(self):
        """Update the viser simulation with the commanded joint positions."""
        ros_snapshot = self.ros_snapshot.make_copy_with_defaults()
        iiwa_joint_cmd = ros_snapshot.iiwa_joint_cmd
        allegro_joint_cmd = ros_snapshot.allegro_joint_cmd
        iiwa_joint_state = ros_snapshot.iiwa_joint_state
        allegro_joint_state = ros_snapshot.allegro_joint_state
        object_pose = ros_snapshot.object_pose
        goal_object_pose = ros_snapshot.goal_object_pose

        assert iiwa_joint_cmd is not None
        assert allegro_joint_cmd is not None
        assert iiwa_joint_state is not None
        assert allegro_joint_state is not None
        assert object_pose is not None
        assert goal_object_pose is not None

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.robot_viser.update_cfg(q_state)
        self.robot_cmd_viser.update_cfg(q_cmd)

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_O = object_pose
        T_R_O = self.T_R_C @ T_C_O
        object_pos = T_R_O[:3, 3]
        object_quat_xyzw = R.from_matrix(T_R_O[:3, :3]).as_quat()
        self.object_viser.position = object_pos
        self.object_viser.wxyz = object_quat_xyzw[[3, 0, 1, 2]]

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_G = goal_object_pose
        T_R_G = self.goal_T_R_C @ T_C_G
        goal_object_pos = T_R_G[:3, 3]
        goal_object_quat_xyzw = R.from_matrix(T_R_G[:3, :3]).as_quat()
        self.goal_object_viser.position = goal_object_pos
        self.goal_object_viser.wxyz = goal_object_quat_xyzw[[3, 0, 1, 2]]

    @property
    def T_R_C(self) -> np.ndarray:
        # HACK
        return np.eye(4)

    @property
    def goal_T_R_C(self) -> np.ndarray:
        # HACK
        return np.eye(4)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""

        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the viser simulation with the current joint commands
            self.update_viser()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()
            print(
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )


def main():
    try:
        # Create and run the ViserVisualizationNode
        node = ViserVisualizationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
