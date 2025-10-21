#!/usr/bin/env python

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import rospy
import trimesh
import tyro
import viser
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray
from termcolor import colored
from viser.extras import ViserUrdf

from human2sim2robot.hardware_deployment.utils.print_utils import get_ros_loop_rate_str
from human2sim2robot.sim_training import get_asset_root
from human2sim2robot.sim_training.utils.cross_embodiment.camera_extrinsics import (
    ZED_CAMERA_T_R_C,
)
from human2sim2robot.sim_training.utils.cross_embodiment.table_constants import (
    TABLE_QW,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
)

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
        client.camera.position = (1.9319, 0.0, 0.5176)
        client.camera.look_at = (0, 0, 0)
        # client.camera.wxyz = (w, x, y, z)


@dataclass
class Args:
    load_point_cloud: bool = False
    load_rgb_image: bool = False
    add_labels: bool = False
    rate_hz: float = 10
    load_scene_mesh: bool = False

    def __post_init__(self):
        if self.load_point_cloud and not self.load_rgb_image:
            print(
                colored(
                    "When loading point cloud, you must also load the RGB image, setting load_rgb_image=True",
                    "yellow",
                )
            )
            self.load_rgb_image = True


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), T.shape
    n_pts = points.shape[0]
    assert points.shape == (n_pts, 3), points.shape

    return (T[:3, :3] @ points.T + T[:3, 3][:, None]).T


def get_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = np.array(depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)
    assert points.shape == (height, width, 3), f"points.shape: {points.shape}"
    return points


@dataclass
class RosSnapshot:
    iiwa_joint_cmd: Optional[np.ndarray]
    allegro_joint_cmd: Optional[np.ndarray]
    iiwa_joint_state: Optional[np.ndarray]
    allegro_joint_state: Optional[np.ndarray]
    palm_target: Optional[np.ndarray]
    object_pose: Optional[np.ndarray]
    object_pose_2: Optional[np.ndarray]
    goal_object_pose: Optional[np.ndarray]
    goal_object_pose_2: Optional[np.ndarray]
    keypoint_3d: Optional[np.ndarray]
    keypoint_3d_2: Optional[np.ndarray]
    target_keypoint_3d: Optional[np.ndarray]
    target_keypoint_3d_2: Optional[np.ndarray]
    right_hand_keypoints: Optional[np.ndarray]
    left_hand_keypoints: Optional[np.ndarray]
    rgb_image: Optional[np.ndarray]
    depth_image: Optional[np.ndarray]
    cam_K: Optional[np.ndarray]

    @classmethod
    def make_with_nones(cls) -> RosSnapshot:
        return cls(
            iiwa_joint_cmd=None,
            allegro_joint_cmd=None,
            iiwa_joint_state=None,
            allegro_joint_state=None,
            palm_target=None,
            object_pose=None,
            object_pose_2=None,
            goal_object_pose=None,
            goal_object_pose_2=None,
            keypoint_3d=None,
            keypoint_3d_2=None,
            target_keypoint_3d=None,
            target_keypoint_3d_2=None,
            right_hand_keypoints=None,
            left_hand_keypoints=None,
            rgb_image=None,
            depth_image=None,
            cam_K=None,
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

        if self.palm_target is None:
            print(colored("palm_target is None", "yellow"))
            palm_target = np.zeros(6) + 100  # Far away
        else:
            palm_target = self.palm_target

        if self.object_pose is None:
            print(colored("object_pose is None", "yellow"))
            object_pose = np.eye(4)
            object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose = self.object_pose

        if self.object_pose_2 is None:
            print(colored("object_pose_2 is None", "yellow"))
            object_pose_2 = np.eye(4)
            object_pose_2[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose_2 = self.object_pose_2

        if self.keypoint_3d is None:
            print(colored("keypoint_3d is None", "yellow"))
            keypoint_3d = np.zeros(3) + 100  # Far away
        else:
            keypoint_3d = self.keypoint_3d

        if self.keypoint_3d_2 is None:
            print(colored("keypoint_3d_2 is None", "yellow"))
            keypoint_3d_2 = np.zeros(3) + 100  # Far away
        else:
            keypoint_3d_2 = self.keypoint_3d_2

        if self.target_keypoint_3d is None:
            print(colored("target_keypoint_3d is None", "yellow"))
            target_keypoint_3d = np.zeros(3) + 100  # Far away
        else:
            target_keypoint_3d = self.target_keypoint_3d

        if self.target_keypoint_3d_2 is None:
            print(colored("target_keypoint_3d_2 is None", "yellow"))
            target_keypoint_3d_2 = np.zeros(3) + 100  # Far away
        else:
            target_keypoint_3d_2 = self.target_keypoint_3d_2

        if self.right_hand_keypoints is None:
            print(colored("right_hand_keypoints is None", "yellow"))
            right_hand_keypoints = np.zeros((NUM_HAND_KEYPOINTS, 3)) + 100  # Far away
        else:
            right_hand_keypoints = self.right_hand_keypoints

        if self.left_hand_keypoints is None:
            print(colored("left_hand_keypoints is None", "yellow"))
            left_hand_keypoints = np.zeros((NUM_HAND_KEYPOINTS, 3)) + 100  # Far away
        else:
            left_hand_keypoints = self.left_hand_keypoints

        if self.goal_object_pose is None:
            print(colored("goal_object_pose is None", "yellow"))
            goal_object_pose = np.eye(4)
            goal_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose = self.goal_object_pose

        if self.goal_object_pose_2 is None:
            print(colored("goal_object_pose_2 is None", "yellow"))
            goal_object_pose_2 = np.eye(4)
            goal_object_pose_2[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose_2 = self.goal_object_pose_2

        if self.rgb_image is None:
            print(colored("rgb_image is None", "yellow"))
            # No good default for this, just set to None
            rgb_image = None
        else:
            rgb_image = self.rgb_image

        if self.depth_image is None:
            print(colored("depth_image is None", "yellow"))
            # No good default for this, just set to None
            depth_image = None
        else:
            depth_image = self.depth_image

        if self.cam_K is None:
            print(colored("cam_K is None", "yellow"))
            # No good default for this, just set to None
            cam_K = None
        else:
            cam_K = self.cam_K

        return RosSnapshot(
            iiwa_joint_cmd=iiwa_joint_cmd,
            allegro_joint_cmd=allegro_joint_cmd,
            iiwa_joint_state=iiwa_joint_state,
            allegro_joint_state=allegro_joint_state,
            palm_target=palm_target,
            object_pose=object_pose,
            object_pose_2=object_pose_2,
            goal_object_pose=goal_object_pose,
            goal_object_pose_2=goal_object_pose_2,
            keypoint_3d=keypoint_3d,
            keypoint_3d_2=keypoint_3d_2,
            target_keypoint_3d=target_keypoint_3d,
            target_keypoint_3d_2=target_keypoint_3d_2,
            right_hand_keypoints=right_hand_keypoints,
            left_hand_keypoints=left_hand_keypoints,
            rgb_image=rgb_image,
            depth_image=depth_image,
            cam_K=cam_K,
        )


class ViserVisualizationNode:
    def __init__(self, args: Args):
        # ROS setup
        rospy.init_node("viser_visualization_ros_node")
        self.args = args

        # Store snapshot
        self.ros_snapshot = RosSnapshot.make_with_nones()

        # Subscribers
        self.initialize_ros_subscribers()

        # Initialize Viser
        self.initialize_viser()

        # Set control rate to 60Hz
        self.rate_hz = args.rate_hz
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
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.object_pose_2_sub = rospy.Subscriber(
            "/object_pose_2", Pose, self.object_pose_2_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.goal_object_pose_2_sub = rospy.Subscriber(
            "/goal_object_pose_2", Pose, self.goal_object_pose_2_callback
        )
        self.keypoint_3d_sub = rospy.Subscriber(
            "/keypoint_3d", Point, self.keypoint_3d_callback, queue_size=1
        )
        self.keypoint_3d_2_sub = rospy.Subscriber(
            "/keypoint_3d_2", Point, self.keypoint_3d_2_callback, queue_size=1
        )
        self.target_keypoint_3d_sub = rospy.Subscriber(
            "/target_keypoint_3d", Point, self.target_keypoint_3d_callback, queue_size=1
        )
        self.target_keypoint_3d_2_sub = rospy.Subscriber(
            "/target_keypoint_3d_2",
            Point,
            self.target_keypoint_3d_2_callback,
            queue_size=1,
        )
        self.right_hand_keypoints_sub = rospy.Subscriber(
            "/right_hand_keypoints", Float64MultiArray, self.right_hand_keypoints_callback, queue_size=1
        )
        self.left_hand_keypoints_sub = rospy.Subscriber(
            "/left_hand_keypoints", Float64MultiArray, self.left_hand_keypoints_callback, queue_size=1
        )

        if self.camera == "zed":
            rgb_sub_topic = "/zed/zed_node/rgb/image_rect_color"
            depth_sub_topic = "/zed/zed_node/depth/depth_registered"
            camera_info_sub_topic = "/zed/zed_node/rgb/camera_info"
        elif self.camera == "realsense":
            rgb_sub_topic = "/camera/color/image_raw"
            depth_sub_topic = "/camera/aligned_depth_to_color/image_raw"
            camera_info_sub_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Invalid camera: {self.camera}")

        if self.args.load_point_cloud:
            self.depth_image_sub = rospy.Subscriber(
                depth_sub_topic, Image, self.depth_image_callback
            )

        if self.args.load_rgb_image:
            self.bridge = CvBridge()
            self.rgb_image_sub = rospy.Subscriber(
                rgb_sub_topic, Image, self.rgb_image_callback
            )
            self.cam_K_sub = rospy.Subscriber(
                camera_info_sub_topic, CameraInfo, self.cam_K_callback
            )

    def initialize_viser(self):
        SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        # Load robot URDF with a fixed base
        robot_urdf_path = get_asset_root() / "kuka_allegro/kuka_allegro.urdf"
        assert robot_urdf_path.exists(), f"robot_urdf_path not found: {robot_urdf_path}"

        # WARNING: After extensive testing, we find that the Allegro hand robot in the real world
        #          is about 1.2cm lower than the simulated Allegro hand for most joint angles.
        #          This difference is severe enough to cause low-profile manipulation tasks to fail
        #          Thus, we manually offset the robot base by 1.2cm in the z-direction.
        # MANUAL_OFFSET_ROBOT_Z = -0.007
        MANUAL_OFFSET_ROBOT_Z = -0.012
        SERVER.scene.add_frame(
            "/robot/state",
            position=(0, 0, MANUAL_OFFSET_ROBOT_Z),
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_frame(
            "/robot/cmd",
            position=(0, 0, MANUAL_OFFSET_ROBOT_Z),
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
            assert isinstance(
                robot_cmd_mesh, viser.MeshHandle
            ), f"robot_cmd_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(robot_cmd_mesh)}"
            robot_cmd_mesh.opacity = 0.5

        # Load the scene mesh
        LOAD_SCENE_MESH = self.args.load_scene_mesh
        if LOAD_SCENE_MESH:
            scene_urdf_path = (
                get_asset_root() / "scene_mesh_cropped/scene_mesh_cropped.urdf"
            )
            assert (
                scene_urdf_path.exists()
            ), f"scene_urdf_path not found: {scene_urdf_path}"
            T = np.linalg.inv(
                np.array(
                    [
                        [
                            -9.87544368e-01,
                            -1.57333070e-01,
                            -1.55753395e-03,
                            7.91730212e-02,
                        ],
                        [
                            -9.08047145e-04,
                            -4.19989728e-03,
                            9.99990768e-01,
                            -3.65614006e-01,
                        ],
                        [
                            -1.57338159e-01,
                            9.87536666e-01,
                            4.00471907e-03,
                            5.94016453e-01,
                        ],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                )
            )
            x, y, z = T[:3, 3]
            qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()

            SERVER.scene.add_frame(
                "/scene", position=(x, y, z), wxyz=(qw, qx, qy, qz), show_axes=False
            )
            _scene_viser = ViserUrdf(SERVER, scene_urdf_path, root_node_name="/scene")

        LOAD_TABLE = True
        if LOAD_TABLE:
            table_urdf_path = get_asset_root() / "table/table.urdf"
            assert (
                table_urdf_path.exists()
            ), f"table_urdf_path not found: {table_urdf_path}"

            SERVER.scene.add_frame(
                "/table",
                position=(TABLE_X, TABLE_Y, TABLE_Z),
                wxyz=(TABLE_QW, TABLE_QX, TABLE_QY, TABLE_QZ),
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
                    assert isinstance(
                        table_mesh, viser.MeshHandle
                    ), f"table_mesh is not a MeshHandle, you must create ViserUrdf with mesh_color_override: {type(table_mesh)}"
                    table_mesh.color = (0, 0, 0)
                    table_mesh.opacity = 0.5

        # Load the object mesh
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_path = rospy.get_param("/mesh_file", None)
        if object_mesh_path is None:
            DEFAULT_MESH_PATH = get_asset_root() / "kiri/snackbox/snackbox.obj"
            object_mesh_path = str(DEFAULT_MESH_PATH)
            print(colored(f"Using default object mesh: {object_mesh_path}", "yellow"))
        assert isinstance(
            object_mesh_path, str
        ), f"object_mesh_path: {object_mesh_path}"
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
        if self.args.add_labels:
            self.object_viser_label = SERVER.scene.add_label(
                name="/object/label",
                text="object",
            )
            self.goal_object_viser_label = SERVER.scene.add_label(
                name="/goal_object/label",
                text="goal_object",
            )

        # Load the object mesh 2
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_2_path = rospy.get_param("/mesh_file_2", None)
        if object_mesh_2_path is None:
            DEFAULT_MESH_PATH = get_asset_root() / "kiri/snackbox/snackbox.obj"
            object_mesh_2_path = str(DEFAULT_MESH_PATH)
            print(
                colored(f"Using default object mesh 2: {object_mesh_2_path}", "yellow")
            )
        assert isinstance(
            object_mesh_2_path, str
        ), f"object_mesh_2_path: {object_mesh_2_path}"
        print("~" * 80)
        print(f"object_mesh_2_path: {object_mesh_2_path}")
        print("~" * 80 + "\n")

        goal_object_mesh_2_path = object_mesh_2_path

        object_mesh_2 = trimesh.load(object_mesh_2_path)
        goal_object_mesh_2 = trimesh.load(goal_object_mesh_2_path)
        self.object_viser_2 = SERVER.scene.add_frame(
            "/object_2",
            position=FAR_AWAY_OBJECT_POSITION,
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        SERVER.scene.add_mesh_trimesh(name="/object_2/mesh", mesh=object_mesh_2)
        self.goal_object_viser_2 = SERVER.scene.add_frame(
            "/goal_object_2",
            position=FAR_AWAY_OBJECT_POSITION + np.array([0.2, 0.2, 0.2]),
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        SERVER.scene.add_mesh_simple(
            name="/goal_object_2/mesh",
            vertices=goal_object_mesh_2.vertices,
            faces=goal_object_mesh_2.faces,
            color=GREEN_RGB,
            opacity=0.5,
        )
        if self.args.add_labels:
            self.object_viser_2_label = SERVER.scene.add_label(
                name="/object_2/label",
                text="object_2",
            )
            self.goal_object_viser_2_label = SERVER.scene.add_label(
                name="/goal_object_2/label",
                text="goal_object_2",
            )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.robot_viser.update_cfg(DEFAULT_Q)
        self.robot_cmd_viser.update_cfg(DEFAULT_Q)

        # Keep track of the link names and IDs
        # Create the hand target
        FAR_AWAY_PALM_TARGET = np.concatenate([np.ones(3), np.zeros(3)])
        self.palm_target_viser_frame = SERVER.scene.add_frame(
            "/palm_target",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        self.palm_viser_frame = SERVER.scene.add_frame(
            "/palm",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )
        self.palm_cmd_viser_frame = SERVER.scene.add_frame(
            "/palm_cmd",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )

        # Create the keypoint 3d
        sphere_mesh = trimesh.creation.icosphere(subdivisions=4, radius=0.03)

        self.keypoint_3d_viser_frame = SERVER.scene.add_frame(
            "/keypoint_3d",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_mesh_simple(
            name="/keypoint_3d/point",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            color=(255, 0, 0),
            opacity=0.5,
        )

        self.keypoint_3d_2_viser_frame = SERVER.scene.add_frame(
            "/keypoint_3d_2",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_mesh_simple(
            name="/keypoint_3d_2/point",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            color=(0, 0, 255),
            opacity=0.5,
        )
        if self.args.add_labels:
            self.keypoint_3d_viser_label = SERVER.scene.add_label(
                name="/keypoint_3d/label",
                text="keypoint_3d",
            )
            self.keypoint_3d_2_viser_label = SERVER.scene.add_label(
                name="/keypoint_3d_2/label",
                text="keypoint_3d_2",
            )

        self.target_keypoint_3d_viser_frame = SERVER.scene.add_frame(
            "/target_keypoint_3d",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_mesh_simple(
            name="/target_keypoint_3d/point",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            color=(255, 0, 0),
            opacity=0.5,
        )

        self.target_keypoint_3d_2_viser_frame = SERVER.scene.add_frame(
            "/target_keypoint_3d_2",
            position=FAR_AWAY_PALM_TARGET[:3],
            wxyz=(1, 0, 0, 0),
            show_axes=False,
        )
        SERVER.scene.add_mesh_simple(
            name="/target_keypoint_3d_2/point",
            vertices=sphere_mesh.vertices,
            faces=sphere_mesh.faces,
            color=(0, 0, 255),
            opacity=0.5,
        )
        if self.args.add_labels:
            self.target_keypoint_3d_viser_label = SERVER.scene.add_label(
                name="/target_keypoint_3d/label",
                text="target_keypoint_3d",
            )
            self.target_keypoint_3d_2_viser_label = SERVER.scene.add_label(
                name="/target_keypoint_3d_2/label",
                text="target_keypoint_3d_2",
            )

        # Hand keypoints
        self.right_hand_keypoint_viser_frames = [
            SERVER.scene.add_frame(
                f"/right_hand_keypoint_{i}",
                position=FAR_AWAY_PALM_TARGET[:3],
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            for i in range(NUM_HAND_KEYPOINTS)
        ]
        for i, x in enumerate(self.right_hand_keypoint_viser_frames):
            SERVER.scene.add_mesh_simple(
                name=f"/right_hand_keypoint_{i}/point",
                vertices=sphere_mesh.vertices,
                faces=sphere_mesh.faces,
                color=(0, 0, 255),
                opacity=0.5,
            )
        self.left_hand_keypoint_viser_frames = [
            SERVER.scene.add_frame(
                f"/left_hand_keypoint_{i}",
                position=FAR_AWAY_PALM_TARGET[:3],
                wxyz=(1, 0, 0, 0),
                show_axes=False,
            )
            for i in range(NUM_HAND_KEYPOINTS)
        ]
        for i, x in enumerate(self.left_hand_keypoint_viser_frames):
            SERVER.scene.add_mesh_simple(
                name=f"/left_hand_keypoint_{i}/point",
                vertices=sphere_mesh.vertices,
                faces=sphere_mesh.faces,
                color=(0, 0, 255),
                opacity=0.5,
            )

        # Create the camera lines
        cam_pos = self.T_R_C[:3, 3]
        cam_quat_xyzw = R.from_matrix(self.T_R_C[:3, :3]).as_quat()
        SERVER.scene.add_frame(
            "/camera",
            position=cam_pos,
            wxyz=(
                cam_quat_xyzw[3],
                cam_quat_xyzw[0],
                cam_quat_xyzw[1],
                cam_quat_xyzw[2],
            ),
            show_axes=True,
            axes_length=AXES_LENGTH,
            axes_radius=AXES_RADIUS,
        )

        if self.args.load_point_cloud:
            self.point_cloud_viser: Optional[viser.PointCloudHandle] = None

        if self.args.load_rgb_image:
            cam_pos = self.T_R_C[:3, 3]
            cam_quat_xyzw = R.from_matrix(self.T_R_C[:3, :3]).as_quat()
            DUMMY_WIDTH, DUMMY_HEIGHT, DUMMY_FY = 640, 360, 338.8556823730469
            DUMMY_RGB_IMAGE = np.zeros((DUMMY_HEIGHT, DUMMY_WIDTH, 3), dtype=np.uint8)
            self.camera_frustum_viser = SERVER.scene.add_camera_frustum(
                name="/camera_frustum",
                fov=float(2 * np.arctan(DUMMY_HEIGHT / (2 * DUMMY_FY))),
                aspect=float(DUMMY_WIDTH / DUMMY_HEIGHT),
                scale=0.1,
                line_width=2.0,
                color=BLACK_RGB,
                image=DUMMY_RGB_IMAGE,
                wxyz=(
                    cam_quat_xyzw[3],
                    cam_quat_xyzw[0],
                    cam_quat_xyzw[1],
                    cam_quat_xyzw[2],
                ),
                position=cam_pos,
            )

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

    def palm_target_callback(self, msg: Float64MultiArray):
        """Callback to update the current hand target."""
        self.ros_snapshot.palm_target = np.array(msg.data)

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

    def object_pose_2_callback(self, msg: Pose):
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
        self.ros_snapshot.object_pose_2 = latest_pose

    def goal_object_pose_2_callback(self, msg: Pose):
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
        self.ros_snapshot.goal_object_pose_2 = latest_pose

    def keypoint_3d_callback(self, msg: Point):
        self.ros_snapshot.keypoint_3d = np.array([msg.x, msg.y, msg.z])

    def keypoint_3d_2_callback(self, msg: Point):
        self.ros_snapshot.keypoint_3d_2 = np.array([msg.x, msg.y, msg.z])

    def target_keypoint_3d_callback(self, msg: Point):
        self.ros_snapshot.target_keypoint_3d = np.array([msg.x, msg.y, msg.z])

    def target_keypoint_3d_2_callback(self, msg: Point):
        self.ros_snapshot.target_keypoint_3d_2 = np.array([msg.x, msg.y, msg.z])

    def right_hand_keypoints_callback(self, msg: Float64MultiArray):
        self.ros_snapshot.right_hand_keypoints = np.array(msg.data).reshape(NUM_HAND_KEYPOINTS, 3)

    def left_hand_keypoints_callback(self, msg: Float64MultiArray):
        self.ros_snapshot.left_hand_keypoints = np.array(msg.data).reshape(NUM_HAND_KEYPOINTS, 3)

    def rgb_image_callback(self, msg: Image):
        self.ros_snapshot.rgb_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

    def depth_image_callback(self, msg: Image):
        self.ros_snapshot.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def cam_K_callback(self, msg: CameraInfo):
        self.ros_snapshot.cam_K = np.array(msg.K).reshape(3, 3)

    def update_viser(self):
        """Update the viser simulation with the commanded joint positions."""
        ros_snapshot = self.ros_snapshot.make_copy_with_defaults()
        iiwa_joint_cmd = ros_snapshot.iiwa_joint_cmd
        allegro_joint_cmd = ros_snapshot.allegro_joint_cmd
        iiwa_joint_state = ros_snapshot.iiwa_joint_state
        allegro_joint_state = ros_snapshot.allegro_joint_state
        palm_target = ros_snapshot.palm_target
        object_pose = ros_snapshot.object_pose
        goal_object_pose = ros_snapshot.goal_object_pose
        object_pose_2 = ros_snapshot.object_pose_2
        goal_object_pose_2 = ros_snapshot.goal_object_pose_2
        keypoint_3d = ros_snapshot.keypoint_3d
        keypoint_3d_2 = ros_snapshot.keypoint_3d_2
        target_keypoint_3d = ros_snapshot.target_keypoint_3d
        target_keypoint_3d_2 = ros_snapshot.target_keypoint_3d_2
        right_hand_keypoints = ros_snapshot.right_hand_keypoints
        left_hand_keypoints = ros_snapshot.left_hand_keypoints
        rgb_image = ros_snapshot.rgb_image
        depth_image = ros_snapshot.depth_image
        cam_K = ros_snapshot.cam_K

        assert iiwa_joint_cmd is not None
        assert allegro_joint_cmd is not None
        assert iiwa_joint_state is not None
        assert allegro_joint_state is not None
        assert palm_target is not None
        assert object_pose is not None
        assert goal_object_pose is not None
        assert object_pose_2 is not None
        assert goal_object_pose_2 is not None
        assert keypoint_3d is not None
        assert keypoint_3d_2 is not None
        assert target_keypoint_3d is not None
        assert target_keypoint_3d_2 is not None
        assert right_hand_keypoints is not None
        assert left_hand_keypoints is not None
        # assert rgb_image is not None
        # assert depth_image is not None
        # assert cam_K is not None

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.robot_viser.update_cfg(q_state)
        self.robot_cmd_viser.update_cfg(q_cmd)

        # Update the hand target
        self.palm_target_viser_frame.position = palm_target[:3]
        self.palm_target_viser_frame.wxyz = R.from_euler(
            "ZYX", palm_target[3:]
        ).as_quat()[[3, 0, 1, 2]]

        # Visualize the palm of the robot
        palm_pose = self.robot_viser._urdf.get_transform(frame_to="palm_link").copy()
        assert palm_pose.shape == (
            4,
            4,
        ), f"palm_pose.shape: {palm_pose.shape}"
        palm_xyz = palm_pose[:3, 3]
        palm_quat_xyzw = R.from_matrix(palm_pose[:3, :3]).as_quat()
        self.palm_viser_frame.position = palm_xyz
        self.palm_viser_frame.wxyz = palm_quat_xyzw[[3, 0, 1, 2]]

        cmd_palm_pose = self.robot_cmd_viser._urdf.get_transform(
            frame_to="palm_link"
        ).copy()
        assert cmd_palm_pose.shape == (
            4,
            4,
        ), f"cmd_palm_pose.shape: {cmd_palm_pose.shape}"
        cmd_palm_xyz = cmd_palm_pose[:3, 3]
        cmd_palm_quat_xyzw = R.from_matrix(cmd_palm_pose[:3, :3]).as_quat()
        self.palm_cmd_viser_frame.position = cmd_palm_xyz
        self.palm_cmd_viser_frame.wxyz = cmd_palm_quat_xyzw[[3, 0, 1, 2]]

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

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_O_2 = object_pose_2
        T_R_O_2 = self.T_R_C @ T_C_O_2
        object_pos_2 = T_R_O_2[:3, 3]
        object_quat_xyzw_2 = R.from_matrix(T_R_O_2[:3, :3]).as_quat()
        self.object_viser_2.position = object_pos_2
        self.object_viser_2.wxyz = object_quat_xyzw_2[[3, 0, 1, 2]]

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_G_2 = goal_object_pose_2
        T_R_G_2 = self.goal_T_R_C @ T_C_G_2
        goal_object_pos_2 = T_R_G_2[:3, 3]
        goal_object_quat_xyzw_2 = R.from_matrix(T_R_G_2[:3, :3]).as_quat()
        self.goal_object_viser_2.position = goal_object_pos_2
        self.goal_object_viser_2.wxyz = goal_object_quat_xyzw_2[[3, 0, 1, 2]]

        # Update the keypoint
        # Keypoint is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        keypoint_3d_R = transform_points(self.T_R_C, keypoint_3d[None]).squeeze(axis=0)
        self.keypoint_3d_viser_frame.position = keypoint_3d_R

        keypoint_3d_2_R = transform_points(self.T_R_C, keypoint_3d_2[None]).squeeze(
            axis=0
        )
        self.keypoint_3d_2_viser_frame.position = keypoint_3d_2_R

        # target_keypoint_3d_R = transform_points(self.T_R_C, target_keypoint_3d[None]).squeeze(axis=0)
        # NOTE: Target keypoint is already in robot frame
        target_keypoint_3d_R = target_keypoint_3d
        self.target_keypoint_3d_viser_frame.position = target_keypoint_3d_R

        target_keypoint_3d_2_R = transform_points(
            self.T_R_C, target_keypoint_3d_2[None]
        ).squeeze(axis=0)
        self.target_keypoint_3d_2_viser_frame.position = target_keypoint_3d_2_R

        # Update the hand keypoints
        # These are in camera frame = C frame
        # We want them in world frame = robot frame = R frame
        for i in range(NUM_HAND_KEYPOINTS):
            self.right_hand_keypoint_viser_frames[i].position = transform_points(self.T_R_C, right_hand_keypoints[i][None]).squeeze(axis=0)
            self.left_hand_keypoint_viser_frames[i].position = transform_points(self.T_R_C, left_hand_keypoints[i][None]).squeeze(axis=0)

        # Update the point cloud
        if depth_image is not None and cam_K is not None and rgb_image is not None:
            self.draw_colored_point_cloud_from_depth_image(
                depth_image=depth_image,
                cam_K=cam_K,
                rgb_image=rgb_image,
                T_R_C=self.T_R_C,
            )

        if self.args.add_labels:
            self.object_viser_label.text = f"object xyz: {np.round(object_pos, decimals=2)}, xyzw: {np.round(object_quat_xyzw, decimals=2)}"
            self.goal_object_viser_label.text = f"goal_object xyz: {np.round(goal_object_pos, decimals=2)}, xyzw: {np.round(goal_object_quat_xyzw, decimals=2)}"
            self.object_viser_2_label.text = f"object_2 xyz: {np.round(object_pos_2, decimals=2)}, xyzw: {np.round(object_quat_xyzw_2, decimals=2)}"
            self.goal_object_viser_2_label.text = f"goal_object_2 xyz: {np.round(goal_object_pos_2, decimals=2)}, xyzw: {np.round(goal_object_quat_xyzw_2, decimals=2)}"
            self.keypoint_3d_viser_label.text = (
                f"keypoint_3d xyz: {np.round(keypoint_3d_R, decimals=2)}"
            )
            self.keypoint_3d_2_viser_label.text = (
                f"keypoint_3d_2 xyz: {np.round(keypoint_3d_2_R, decimals=2)}"
            )
            self.target_keypoint_3d_viser_label.text = (
                f"target_keypoint_3d xyz: {np.round(target_keypoint_3d_R, decimals=2)}"
            )
            self.target_keypoint_3d_2_viser_label.text = f"target_keypoint_3d_2 xyz: {np.round(target_keypoint_3d_2_R, decimals=2)}"

        if cam_K is not None and rgb_image is not None:
            fy = cam_K[1, 1]
            width, height = rgb_image.shape[1], rgb_image.shape[0]
            self.camera_frustum_viser.image = rgb_image
            self.camera_frustum_viser.fov = float(2 * np.arctan(height / (2 * fy)))
            self.camera_frustum_viser.aspect = float(width / height)

    def draw_colored_point_cloud_from_depth_image(
        self,
        depth_image: np.ndarray,
        cam_K: np.ndarray,
        rgb_image: np.ndarray,
        T_R_C: np.ndarray,
    ):
        H, W, C = rgb_image.shape
        assert depth_image.shape == (H, W), f"depth_image.shape: {depth_image.shape}"
        assert cam_K.shape == (3, 3), f"cam_K.shape: {cam_K.shape}"
        assert T_R_C.shape == (4, 4), f"T_R_C.shape: {T_R_C.shape}"

        # Depth is in mm
        depth_image = depth_image / 1000

        # Convert depth image to point cloud
        point_cloud_C = get_points(depth=depth_image, cam_K=cam_K).reshape(-1, 3)
        point_cloud_R = transform_points(T=T_R_C, points=point_cloud_C)
        point_cloud_colors = rgb_image.reshape(-1, 3)

        FILTER_POINT_CLOUD = False
        if FILTER_POINT_CLOUD:
            # idxs = (point_cloud_R[:, 0] > 0) & (point_cloud_R[:, 1] < 0)
            idxs = (point_cloud_R[:, 0] > 0) & (point_cloud_R[:, 1] < -0.2)
            point_cloud_R = point_cloud_R[idxs]
            point_cloud_colors = point_cloud_colors[idxs]

        # Use debug points instead of spheres for faster rendering
        if self.point_cloud_viser is None:
            print(f"Creating new point cloud with {len(point_cloud_R)} points")
            self.point_cloud_viser = SERVER.scene.add_point_cloud(
                "/point_cloud",
                points=point_cloud_R,
                colors=point_cloud_colors,
                point_size=0.001,
            )
        else:
            self.point_cloud_viser.points = point_cloud_R
            self.point_cloud_viser.colors = point_cloud_colors

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

    @property
    @functools.lru_cache()
    def camera(self) -> Literal["zed", "realsense"]:
        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            camera = DEFAULT_CAMERA
        print(f"Using camera: {camera}")
        assert camera in ["zed", "realsense"], f"camera: {camera}"
        return camera

    @property
    @functools.lru_cache()
    def goal_camera(self) -> Literal["zed", "realsense"]:
        # Check goal_camera parameter
        goal_camera = rospy.get_param("/goal_camera", None)
        if goal_camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /goal_camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            goal_camera = DEFAULT_CAMERA
        print(f"Using goal_camera: {goal_camera}")
        assert goal_camera in ["zed", "realsense"], f"goal_camera: {goal_camera}"
        return goal_camera

    @property
    @functools.lru_cache()
    def T_R_C(self) -> np.ndarray:
        if self.camera == "zed":
            return ZED_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown camera: {self.camera}")

    @property
    @functools.lru_cache()
    def goal_T_R_C(self) -> np.ndarray:
        # Check goal_camera parameter
        if self.goal_camera == "zed":
            return ZED_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown goal_camera: {self.goal_camera}")


def main():
    args = tyro.cli(Args)
    try:
        # Create and run the ViserVisualizationNode
        node = ViserVisualizationNode(args)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()