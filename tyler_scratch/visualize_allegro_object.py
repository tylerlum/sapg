import viser
import time
from viser.extras import ViserUrdf
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_T(pose: np.ndarray) -> np.ndarray:
    assert pose.shape == (7,), f"Expected pose to be (7,), got {pose.shape}"
    xyz = pose[:3]
    xyzw = pose[3:7]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T

def T_to_pose(T: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), f"Expected T to be (4, 4), got {T.shape}"
    xyz = T[:3, 3]
    xyzw = R.from_matrix(T[:3, :3]).as_quat()
    pose = np.concatenate([xyz, xyzw])
    return pose

# Constants
# AXES_LENGTH = 0.1
# AXES_RADIUS = 0.001
AXES_LENGTH = 0.2
AXES_RADIUS = 0.01
DT = 1/60


# Load data
recorded_data_path = Path("/home/tylerlum/github_repos/sapg/recorded_data/2025-10-15_15-21-48.npy")
assert recorded_data_path.exists(), f"Recorded data file {recorded_data_path} does not exist"
recorded_data = np.load(recorded_data_path, allow_pickle=True).item()

robot_root_states = recorded_data["robot_root_states"]
object_root_states = recorded_data["object_root_states"]
robot_joint_positions = recorded_data["robot_joint_positions"]
robot_joint_names = recorded_data["robot_joint_names"]

T, N = robot_root_states.shape[0], robot_root_states.shape[1]
assert robot_root_states.shape == (T, N, 13), f"Expected robot root states to be (T, N, 13), got {robot_root_states.shape}"
assert object_root_states.shape == (T, N, 13), f"Expected object root states to be (T, N, 13), got {object_root_states.shape}"
assert robot_joint_positions.shape == (T, N, len(robot_joint_names)), f"Expected robot joint positions to be (T, N, {len(robot_joint_names)}), got {robot_joint_positions.shape}"
assert len(robot_joint_names) == 23, f"Expected 23 robot joint names, got {len(robot_joint_names)}"

# Create server
SERVER = viser.ViserServer()
SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

# Load assets
KUKA_ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf")
assert KUKA_ALLEGRO_URDF_PATH.exists(), f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
OBJECT_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf")
assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf")
assert ALLEGRO_URDF_PATH.exists(), f"ALLEGRO_URDF_PATH not found: {ALLEGRO_URDF_PATH}"

kuka_allegro_frame = SERVER.scene.add_frame("/robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
kuka_allegro_viser = ViserUrdf(
    SERVER, KUKA_ALLEGRO_URDF_PATH, root_node_name="/robot/state"
)
object_frame = SERVER.scene.add_frame("/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
object_viser = ViserUrdf(
    SERVER, OBJECT_URDF_PATH, root_node_name="/object"
)

palm_frame = SERVER.scene.add_frame("/robot_palm", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)

# Main loop
T_IDX = 0
E_IDX = 0

# Keep floating allegro hand in place on the right
allegro_frame = SERVER.scene.add_frame("/allegro", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
allegro_viser = ViserUrdf(
    SERVER, ALLEGRO_URDF_PATH, root_node_name="/allegro"
)
allegro_frame.position = robot_root_states[T_IDX, E_IDX, :3] + np.array([0.5, 0, 0])
allegro_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

object_in_allegro_frame = SERVER.scene.add_frame("/allegro/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
object_in_allegro_viser = ViserUrdf(
    SERVER, OBJECT_URDF_PATH, root_node_name="/allegro/object"
)

# Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
kuka_allegro_viser_joint_names = kuka_allegro_viser._urdf.actuated_joint_names
allegro_viser_joint_names = allegro_viser._urdf.actuated_joint_names

while True:
    # Get data
    robot_root_state = robot_root_states[T_IDX, E_IDX]
    object_root_state = object_root_states[T_IDX, E_IDX]
    robot_joint_position = robot_joint_positions[T_IDX, E_IDX]
    robot_joint_position_dict = {
        name: pos for name, pos in zip(robot_joint_names, robot_joint_position)
    }

    # Update state
    kuka_allegro_frame.position = robot_root_state[:3]
    kuka_allegro_frame.wxyz = robot_root_state[3:7][[3, 0, 1, 2]]
    object_frame.position = object_root_state[:3]
    object_frame.wxyz = object_root_state[3:7][[3, 0, 1, 2]]
    kuka_allegro_viser.update_cfg(
        np.array([robot_joint_position_dict[name] for name in kuka_allegro_viser_joint_names])
    )
    allegro_viser.update_cfg(
        np.array([robot_joint_position_dict[name] for name in allegro_viser_joint_names])
    )

    # Visualize the palm of the robot
    palm_pose_R = kuka_allegro_viser._urdf.get_transform(frame_to="allegro_mount").copy()
    assert palm_pose_R.shape == (
        4,
        4,
    ), f"palm_pose_R.shape: {palm_pose_R.shape}"
    palm_xyz_R = palm_pose_R[:3, 3]
    palm_quat_xyzw_R = R.from_matrix(palm_pose_R[:3, :3]).as_quat()
    T_R_P = pose_to_T(np.concatenate([palm_xyz_R, palm_quat_xyzw_R]))
    T_W_R = pose_to_T(robot_root_state[:7])
    T_W_P = T_W_R @ T_R_P
    palm_xyz_W = T_W_P[:3, 3]
    palm_quat_xyzw_W = R.from_matrix(T_W_P[:3, :3]).as_quat()
    palm_frame.position = palm_xyz_W
    palm_frame.wxyz = palm_quat_xyzw_W[[3, 0, 1, 2]]

    # By default MOVE_FLOATING_ALLEGRO_HAND = False so we can see how the object is moving wrt a fixed allegro hand
    # Can set to True to debug and make sure that everything aligns
    MOVE_FLOATING_ALLEGRO_HAND = False
    if MOVE_FLOATING_ALLEGRO_HAND:
        allegro_frame.position = palm_xyz_W
        allegro_frame.wxyz = palm_quat_xyzw_W[[3, 0, 1, 2]]

    # # Get object pose wrt palm pose
    T_W_O = pose_to_T(object_root_state[:7])
    T_P_W = np.linalg.inv(T_W_P)
    T_P_O = T_P_W @ T_W_O
    object_xyz_P = T_P_O[:3, 3]
    object_quat_xyzw_P = R.from_matrix(T_P_O[:3, :3]).as_quat()
    object_in_allegro_frame.position = object_xyz_P
    object_in_allegro_frame.wxyz = object_quat_xyzw_P[[3, 0, 1, 2]]

    time.sleep(DT)
    T_IDX += 1
    if T_IDX >= T:
        # T_IDX = T - 1
        T_IDX = 0

breakpoint()