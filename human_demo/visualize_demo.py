import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from isaacgymenvs.utils.utils import get_repo_root_dir
from typing import Literal
import pytorch_kinematics as pk
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_axis_angle
import torch

import numpy as np
import tyro
import yaml
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import viser
from viser.extras import ViserUrdf

T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])
T_R_C = np.array([
    [0.95527630647288930, -0.17920451516639435, 0.23522950502752071, -0.50020504226664309],
    [-0.28890230754832508, -0.39580744250644329, 0.87170632964878869, -1.43857156913606077],
    [-0.06310812138518884, -0.90067874972183481, -0.42987806970668574, 1.02018932829980047],
    [0.00000000000000000, 0.00000000000000000, 0.00000000000000000, 1.00000000000000000]
])
T_W_C = T_W_R @ T_R_C

AXES_LENGTH = 0.0
AXES_RADIUS = 0.0

def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])

def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([x, y, z, w])

def pose_to_T(pose: np.ndarray) -> np.ndarray:
    assert pose.shape == (7,), f"Expected pose to be (7,), got {pose.shape}"
    xyz = pose[:3]
    xyzw = pose[3:7]
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzw).as_matrix()
    T[:3, 3] = xyz
    return T

def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize 1D vector
    """
    assert v.ndim == 1, f"v.shape: {v.shape}"
    norm = np.linalg.norm(v)
    assert norm > 0, f"norm: {norm}"
    return v / norm


def transform_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Transform point by transform T
    """
    assert point.shape == (3,)
    assert T.shape == (4, 4)
    point = np.concatenate([point, [1]])
    transformed_point = T @ point
    return transformed_point[:3]


def create_transform(
    pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    """
    Create transform T from position and rotation
    """
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def create_urdf(
    obj_filepath: Path,
    mass: float = 0.066,
    ixx: float = 1e-3,
    iyy: float = 1e-3,
    izz: float = 1e-3,
    color: Optional[Literal["white"]] = None,
) -> Path:
    """
    Create URDF file for new object from path to object mesh
    """
    if color == "white":
        color_material = (
            """<material name="white"> <color rgba="1. 1. 1. 1."/> </material>"""
        )
    elif color is None:
        color_material = ""
    else:
        raise ValueError(f"Invalid color {color}")

    assert obj_filepath.suffix == ".obj"
    urdf_filepath = obj_filepath.with_suffix(".urdf")
    urdf_text = f"""<?xml version="1.0" ?>
        <robot name="model.urdf">
        <link name="baseLink">
            <contact>
                <lateral_friction value="0.8"/>
                <rolling_friction value="0.001"/>g
                <contact_cfm value="0.0"/>
                <contact_erp value="1.0"/>
            </contact>
            <inertial>
                <mass value="{mass}"/>
                <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
            </inertial>
            <visual>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            {color_material}
            </visual>
            <collision>
            <geometry>
                <mesh filename="{obj_filepath.name}" scale="1 1 1"/>
            </geometry>
            </collision>
        </link>
        </robot>"""
    with urdf_filepath.open("w") as f:
        f.write(urdf_text)
    return urdf_filepath


def control_ik(
    j_eef: torch.Tensor, dpose: torch.Tensor, damping: float = 0.5
) -> torch.Tensor:
    # solve damped least squares

    # Set controller parameters
    NUM_ENVS, NUM_EE_DIMS, NUM_JOINTS = j_eef.shape
    assert dpose.shape == (NUM_ENVS, NUM_EE_DIMS), f"dpose.shape: {dpose.shape}"

    j_eef_T = torch.transpose(j_eef, 1, 2)
    assert j_eef_T.shape == (
        NUM_ENVS,
        NUM_JOINTS,
        NUM_EE_DIMS,
    ), f"j_eef_T.shape: {j_eef_T.shape}"

    lmbda = torch.eye(NUM_EE_DIMS, device=j_eef.device) * (damping**2)
    assert lmbda.shape == (NUM_EE_DIMS, NUM_EE_DIMS), f"lmbda.shape: {lmbda.shape}"

    # u = J.T @ (J @ J.T + lambda)^-1 @ dpose
    u = torch.bmm(
        j_eef_T,
        torch.bmm(
            torch.inverse(torch.bmm(j_eef, j_eef_T) + lmbda[None]),
            dpose.unsqueeze(dim=-1),
        ),
    ).squeeze(dim=-1)
    assert u.shape == (NUM_ENVS, NUM_JOINTS), f"u.shape: {u.shape}"

    return u

def compute_new_q_arm(arm_pk_chain: pk.SerialChain, target_wrist_pose: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
    NUM_ARM_JOINTS = 7
    assert target_wrist_pose.shape == (4, 4), f"target_wrist_pose.shape: {target_wrist_pose.shape}"
    assert q_arm.shape == (NUM_ARM_JOINTS,), f"q_arm.shape: {q_arm.shape}"

    device = arm_pk_chain.device
    q_arm_torch = torch.from_numpy(q_arm).float().to(device)
    target_wrist_pose_torch = torch.from_numpy(target_wrist_pose).float().to(device)

    # Compute current wrist pose
    wrist_pose = arm_pk_chain.forward_kinematics(q_arm_torch).get_matrix()[0]
    assert wrist_pose.shape == (
        4,
        4,
    ), f"wrist_pose.shape: {wrist_pose.shape}"

    # Compute wrist error
    wrist_pos_error = (target_wrist_pose_torch[:3, 3] - wrist_pose[:3, 3])[None]
    wrist_rot_error = (target_wrist_pose_torch[:3, :3] @ wrist_pose[:3, :3].T)[None]
    wrist_rot_error = matrix_to_axis_angle(wrist_rot_error)
    wrist_error = torch.cat([wrist_pos_error, wrist_rot_error], dim=-1)
    NUM_XYZRPY = 6
    assert wrist_error.shape == (
        1,
        NUM_XYZRPY,
    ), f"wrist_error.shape: {wrist_error.shape}"

    # Compute jacobian
    jacobian = arm_pk_chain.jacobian(q_arm_torch[None])
    assert jacobian.shape == (
        1,
        NUM_XYZRPY,
        NUM_ARM_JOINTS,
    ), f"jacobian.shape: {jacobian.shape}"
    pos_only = False
    if pos_only:
        jacobian = jacobian[:, 0:3]
        NUM_XYZ = 3
        assert jacobian.shape == (
            1,
            NUM_XYZ,
            NUM_ARM_JOINTS,
        ), f"jacobian.shape: {jacobian.shape}"

    # Compute delta arm joint position
    delta_q_arm = control_ik(
        j_eef=jacobian,
        dpose=wrist_error,
    )

    new_q_arm = q_arm_torch + delta_q_arm
    assert new_q_arm.shape == (
        1,
        NUM_ARM_JOINTS,
    ), f"new_q_arm.shape: {new_q_arm.shape}"
    return new_q_arm.cpu().numpy()[0]


@dataclass
class Args:
    object_path: Path
    object_poses_json_path: Path
    hand_poses_dir: Path
    visualize_hand_meshes: bool = False
    retarget_robot: bool = False
    dt: float = 1.0 / 30
    start_idx: int = 0


def set_keypoint_sphere_positions(hand_keypoint_to_xyz: dict, server: viser.ViserServer) -> None:
    from human_demo.colors import RED_TRANSLUCENT_RGBA, RED_RGBA, GREEN_TRANSLUCENT_RGBA, GREEN_RGBA, BLUE_TRANSLUCENT_RGBA, BLUE_RGBA, YELLOW_TRANSLUCENT_RGBA, YELLOW_RGBA, MAGENTA_RGBA, CYAN_RGBA
    keypoint_to_rgba = {
        "wrist_back": RED_TRANSLUCENT_RGBA,
        "wrist_front": RED_RGBA,
        "index_0_back": GREEN_TRANSLUCENT_RGBA,
        "index_0_front": GREEN_RGBA,
        "middle_0_back": BLUE_TRANSLUCENT_RGBA,
        "middle_0_front": BLUE_RGBA,
        "ring_0_back": YELLOW_TRANSLUCENT_RGBA,
        "ring_0_front": YELLOW_RGBA,
        "index_3": GREEN_RGBA,
        "middle_3": BLUE_RGBA,
        "ring_3": YELLOW_RGBA,
        "thumb_3": MAGENTA_RGBA,
        "PALM_TARGET": CYAN_RGBA,
    }
    keypoints = keypoint_to_rgba.keys()

    if not hasattr(set_keypoint_sphere_positions, "spheres"):
        set_keypoint_sphere_positions.spheres = [
            server.scene.add_icosphere(f"/hand/keypoint_{keypoint}", radius=0.02, color=keypoint_to_rgba[keypoint][:3], position=hand_keypoint_to_xyz[keypoint], opacity=keypoint_to_rgba[keypoint][3])
            for keypoint in keypoints
        ]
    else:
        for keypoint, sphere in zip(keypoints, set_keypoint_sphere_positions.spheres):
            sphere.position = hand_keypoint_to_xyz[keypoint]


def create_transformed_keypoint_to_xyz(hand_json: dict, T_W_C: np.ndarray) -> dict:
    keypoint_to_xyz = hand_json

    keypoints = [
        "wrist_back",
        "wrist_front",
        "index_0_back",
        "index_0_front",
        "middle_0_back",
        "middle_0_front",
        "ring_0_back",
        "ring_0_front",
        "index_3",
        "middle_3",
        "ring_3",
        "thumb_3",
    ]
    for keypoint in keypoints:
        assert keypoint in keypoint_to_xyz, (
            f"{keypoint} not in {keypoint_to_xyz.keys()}"
        )
        keypoint_to_xyz[keypoint] = np.array(keypoint_to_xyz[keypoint])

    # Shorthand for next computations
    kpt_map = keypoint_to_xyz

    # Palm target
    mean_middle_0 = np.mean(
        [
            kpt_map["middle_0_back"],
            kpt_map["middle_0_front"],
        ],
        axis=0,
    )
    mean_wrist = np.mean(
        [
            kpt_map["wrist_back"],
            kpt_map["wrist_front"],
        ],
        axis=0,
    )
    palm_normal = normalize(
        np.cross(
            normalize(kpt_map["index_0_front"] - kpt_map["ring_0_front"]),
            normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]),
        )
    )
    kpt_map["PALM_TARGET"] = (
        # VERSION 0
        mean_wrist
        # mean_middle_0
        # # VERSION 1
        # - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03
        # - palm_normal * 0.03
        #
        # VERSION 2
        # - palm_normal * 0.03 * np.sqrt(2)
        #
        # VERSION 3
        # - normalize(kpt_map["middle_0_front"] - kpt_map["wrist_front"]) * 0.03 * np.sqrt(2)
    )

    transformed_keypoint_to_xyz = {
        keypoint: transform_point(T=T_W_C, point=kpt_map[keypoint])
        for keypoint in keypoints + ["PALM_TARGET"]
    }

    # WARNING: After extensive testing, we find that the Allegro hand robot in the real world
    #          is about 1.2cm lower than the simulated Allegro hand for most joint angles.
    #          This difference is severe enough to cause low-profile manipulation tasks to fail
    #          Thus, we manually offset the robot base by 1.2cm in the z-direction.
    MANUAL_OFFSET_ROBOT_Z = 0.012
    NEW_transformed_keypoint_to_xyz = {
        keypoint: transformed_keypoint_to_xyz[keypoint]
        + np.array([0, 0, MANUAL_OFFSET_ROBOT_Z])
        for keypoint in keypoints + ["PALM_TARGET"]
    }
    transformed_keypoint_to_xyz = NEW_transformed_keypoint_to_xyz

    # HACK: add global_orient
    transformed_keypoint_to_xyz["global_orient"] = kpt_map["global_orient"]
    return transformed_keypoint_to_xyz


def compute_r_R_P(keypoint_to_xyz: dict) -> np.ndarray:
    # Z = palm to middle finger
    # Y = thumb to palm
    # X = palm normal
    kpt_map = keypoint_to_xyz
    palm_to_middle_finger = normalize(
        kpt_map["middle_0_front"] - kpt_map["wrist_front"]
    )
    thumb_to_palm = normalize(kpt_map["ring_0_front"] - kpt_map["index_0_front"])
    _palm_normal = normalize(np.cross(thumb_to_palm, palm_to_middle_finger))

    Z = palm_to_middle_finger
    Y_not_orthogonal = thumb_to_palm
    Y = normalize(Y_not_orthogonal - np.dot(Y_not_orthogonal, Z) * Z)
    X = normalize(
        np.cross(
            Y,
            Z,
        )
    )
    r_R_P = np.stack(
        [X, Y, Z],
        axis=1,
    )
    return r_R_P


def main():
    args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80)

    # Start visualizer
    SERVER = viser.ViserServer()

    @SERVER.on_client_connect
    def _(client):
        client.camera.position = T_W_C[:3, 3]
        client.camera.wxyz = xyzw_to_wxyz(R.from_matrix(T_W_C[:3, :3]).as_quat())

    # Load table
    TABLE_URDF_PATH = get_repo_root_dir() / "assets/urdf/table_narrow.urdf"
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    table_frame = SERVER.scene.add_frame(
        "/table", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS, position=(0, 0, 0.38), wxyz=(1, 0, 0, 0),
    )
    table_viser = ViserUrdf(SERVER, TABLE_URDF_PATH, root_node_name="/table")

    # Load robot
    KUKA_SHARPA_URDF_PATH = get_repo_root_dir() / "assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    assert KUKA_SHARPA_URDF_PATH.exists(), (
        f"KUKA_SHARPA_URDF_PATH not found: {KUKA_SHARPA_URDF_PATH}"
    )
    kuka_sharpa_frame = SERVER.scene.add_frame(
        "/robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS, position=(0, 0.8, 0), wxyz=(1, 0, 0, 0),
    )
    kuka_sharpa_viser = ViserUrdf(
        SERVER, KUKA_SHARPA_URDF_PATH, root_node_name="/robot/state"
    )
    HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571 - np.deg2rad(10), -0.000, 1.376 + np.deg2rad(10), -0.000, 1.485, 1.308])
    HOME_JOINT_POS_SHARPA = np.zeros(22)
    HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])
    kuka_sharpa_viser.update_cfg(HOME_JOINT_POS)

    # Load object poses
    assert args.object_poses_json_path.exists(), (
        f"Object poses json path {args.object_poses_json_path} does not exist"
    )
    with open(args.object_poses_json_path, "r") as f:
        object_poses_data = json.load(f)
    T_W_O_start = pose_to_T(np.array(object_poses_data["start_pose"]))
    T_W_Os = [pose_to_T(np.array(pose)) for pose in object_poses_data["goals"]]

    # Load object
    assert args.object_path.exists(), f"Object path {args.object_path} does not exist"
    if args.object_path.suffix == ".obj":
        object_urdf_path = create_urdf(args.obj_path)
    elif args.object_path.suffix == ".urdf":
        object_urdf_path = args.object_path
    else:
        raise ValueError(f"Invalid object path: {args.object_path}")
    print(f"Loading object from {object_urdf_path}")
    object_frame_viser = SERVER.scene.add_frame(
        "/object",
        position=T_W_O_start[:3, 3],
        wxyz=R.from_matrix(T_W_O_start[:3, :3]).as_quat(),
        show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS,
    )
    object_viser = ViserUrdf(SERVER, object_urdf_path, root_node_name=object_frame_viser.name)

    # Load hand poses
    assert args.hand_poses_dir.exists(), (
        f"Hand poses dir {args.hand_poses_dir} does not exist"
    )

    hand_json_files = sorted(list(args.hand_poses_dir.glob("*.json")))
    assert len(hand_json_files) > 0, f"No hand poses found in {args.hand_poses_dir}"
    hand_jsons = []
    for filename in tqdm(hand_json_files, desc="Loading hand poses"):
        with open(filename, "r") as f:
            hand_jsons.append(json.load(f))

    hand_keypoint_to_xyzs = [
        create_transformed_keypoint_to_xyz(hand_json, T_W_C) for hand_json in hand_jsons
    ]

    FAR_AWAY_POSITION = np.ones(3) * 100
    # Load hand meshes
    if args.visualize_hand_meshes:
        # Each timestep has a different hand mesh because they can change shape
        # So this is slow to load
        hand_urdf_files = [
            create_urdf(hand_json_file.with_suffix(".obj"))
            for hand_json_file in hand_json_files
        ]

        SPAWN_HANDS_AT_FAR_AWAY_POSITION = False
        if SPAWN_HANDS_AT_FAR_AWAY_POSITION:
            hand_xyz, hand_quat_xyzw = FAR_AWAY_POSITION, [0, 0, 0, 1]
        else:
            hand_xyz, hand_quat_xyzw = (
                T_W_C[:3, 3],
                R.from_matrix(T_W_C[:3, :3]).as_quat(),
            )

        hand_frames = []
        hand_visers = []
        for i, hand_urdf_file in tqdm(enumerate(hand_urdf_files), desc="Loading hands", total=len(hand_urdf_files)):
            hand_frame = SERVER.scene.add_frame(
                f"/hand/{i}",
                position=hand_xyz,
                wxyz=hand_quat_xyzw,
                show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS,
            )
            hand_viser = ViserUrdf(SERVER, hand_urdf_file, root_node_name=hand_frame.name)
            if not SPAWN_HANDS_AT_FAR_AWAY_POSITION:
                # Move hand to far away position after spawning
                hand_frame.position = FAR_AWAY_POSITION

            hand_frames.append(hand_frame)
            hand_visers.append(hand_viser)

    N_TIMESTEPS = min(len(T_W_Os), len(hand_keypoint_to_xyzs))
    print(f"len(T_W_Os): {len(T_W_Os)}, len(hand_keypoint_to_xyzs): {len(hand_keypoint_to_xyzs)}, N_TIMESTEPS: {N_TIMESTEPS}")
    T_W_Os = T_W_Os[:N_TIMESTEPS]
    hand_keypoint_to_xyzs = hand_keypoint_to_xyzs[:N_TIMESTEPS]
    hand_frames = hand_frames[:N_TIMESTEPS]
    hand_visers = hand_visers[:N_TIMESTEPS]

    if args.retarget_robot:
        with open(KUKA_SHARPA_URDF_PATH, "rb") as f:
            urdf_str = f.read()
        DEVICE = "cpu"
        arm_pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="left_hand_C_MC",
        ).to(device=DEVICE)

    # Visualization loop
    while True:
        for i, (T_W_O, hand_keypoint_to_xyz) in tqdm(
            enumerate(zip(T_W_Os, hand_keypoint_to_xyzs)),
            total=N_TIMESTEPS,
            desc="Visualizing trajectory",
        ):
            if i < args.start_idx:
                continue

            start_time = time.time()

            # Object
            obj_xyz, obj_quat_xyzw = (
                T_W_O[:3, 3],
                R.from_matrix(T_W_O[:3, :3]).as_quat(),
            )
            object_frame_viser.position = obj_xyz
            object_frame_viser.wxyz = xyzw_to_wxyz(obj_quat_xyzw)

            # Hand keypoints
            set_keypoint_sphere_positions(hand_keypoint_to_xyz, SERVER)

            # Hand meshes
            if args.visualize_hand_meshes:
                # Move previous hand to far away position
                # Works when i = 0 because it just moves the last one
                prev_hand_frame = hand_frames[i - 1]
                prev_hand_frame.position = FAR_AWAY_POSITION

                hand_frame = hand_frames[i]
                hand_xyz, hand_quat_xyzw = (
                    T_W_C[:3, 3],
                    R.from_matrix(T_W_C[:3, :3]).as_quat(),
                )
                hand_frame.position = hand_xyz
                hand_frame.wxyz = xyzw_to_wxyz(hand_quat_xyzw)

            # Retarget robot
            if args.retarget_robot:
                T_W_P = np.eye(4)
                T_W_P[:3, 3] = hand_keypoint_to_xyz["PALM_TARGET"]
                r_R_P = compute_r_R_P(keypoint_to_xyz=hand_keypoint_to_xyz)
                r_W_R = T_W_R[:3, :3]
                r_W_P = r_W_R @ r_R_P
                T_W_P[:3, :3] = r_W_P

                T_R_W = np.linalg.inv(T_W_R)
                T_R_P = T_R_W @ T_W_P

                q = np.array(kuka_sharpa_viser._urdf.cfg)
                q_arm = q[:7]
                new_q_arm = compute_new_q_arm(
                    arm_pk_chain=arm_pk_chain,
                    target_wrist_pose=T_R_P,
                    q_arm=q_arm,
                )
                new_q = np.concatenate([new_q_arm, q[7:]])
                kuka_sharpa_viser.update_cfg(new_q)

            end_time = time.time()
            extra_dt = args.dt - (end_time - start_time)
            if extra_dt > 0:
                time.sleep(extra_dt)
            else:
                print(
                    f"Visualization is running slow, late by {-extra_dt * 1000:.2f} ms"
                )

        print("=" * 80)
        print("Setting breakpoint. Continue to start over")
        print("=" * 80 + "\n")
        breakpoint()


if __name__ == "__main__":
    main()