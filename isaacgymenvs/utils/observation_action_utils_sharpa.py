from __future__ import annotations

from scipy.spatial.transform import Rotation as R
from typing import Literal, Tuple
from collections import defaultdict
import yourdfpy
import numpy as np
from pathlib import Path
import torch
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import tensor_clamp

def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)

def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0)[..., None]
    b = np.cross(q_vec, v, axis=-1) * q_w[..., None] * 2.0
    c = q_vec * \
        (q_vec.reshape(shape[0], 1, 3) @ v.reshape(shape[0], 3, 1))[..., 0] * 2.0
    return a + b + c


# Constants
# JOINT_NAMES_ISAACGYM = [
#     'iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7',
#     'left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
#     'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
#     'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP',
#     'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
#     'left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
# ]
JOINT_NAMES_ISAACGYM = [
    'iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7',
    'left_1_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
    'left_2_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
    'left_3_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
    'left_4_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
    'left_5_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP',
]

def matrix_to_quaternion_xyzw_scipy(matrix: np.ndarray) -> np.ndarray:
    return R.from_matrix(matrix).as_quat()

assert len(JOINT_NAMES_ISAACGYM) == 29, (
    f"len(JOINT_NAMES_ISAACGYM): {len(JOINT_NAMES_ISAACGYM)}, expected: 29"
)

# Q_LOWER_LIMITS_np = np.array( [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543, -0.1745,
#         -0.0349,  0.0000,  0.0000, -0.1745, -0.0349,  0.0000,  0.0000,  0.0000,
#         -0.1745, -0.0349,  0.0000,  0.0000, -0.1745, -0.0349,  0.0000,  0.0000,
#         -0.1745, -0.3491, -0.5236, -0.3491,  0.0000],)
# 
# Q_UPPER_LIMITS_np = np.array( [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543, 1.5708, 0.0349,
#         1.7453, 1.3963, 1.5708, 0.0349, 1.7453, 1.3963, 0.2618, 1.5708, 0.0349,
#         1.7453, 1.3963, 1.5708, 0.0349, 1.7453, 1.3963, 1.9199, 0.1309, 1.3963,
#         0.3491, 1.7453],)

Q_LOWER_LIMITS_np = np.array( [
        -2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543, -0.1745,
        -0.3491, -0.5236, -0.3491,  0.0000, -0.1745, -0.0349,  0.0000,  0.0000,
        -0.1745, -0.0349,  0.0000,  0.0000, -0.1745, -0.0349,  0.0000,  0.0000,
         0.0000, -0.1745, -0.0349,  0.0000,  0.0000
])
Q_UPPER_LIMITS_np = np.array( [
        2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543, 1.9199, 0.1309,
        1.3963, 0.3491, 1.7453, 1.5708, 0.0349, 1.7453, 1.3963, 1.5708, 0.0349,
        1.7453, 1.3963, 1.5708, 0.0349, 1.7453, 1.3963, 0.2618, 1.5708, 0.0349,
        1.7453, 1.3963
])
assert Q_LOWER_LIMITS_np.shape == (29,), f"Q_LOWER_LIMITS_np.shape: {Q_LOWER_LIMITS_np.shape}, expected: (29,)"
assert Q_UPPER_LIMITS_np.shape == (29,), f"Q_UPPER_LIMITS_np.shape: {Q_UPPER_LIMITS_np.shape}, expected: (29,)"

Q_LOWER_LIMITS_restricted_np = Q_LOWER_LIMITS_np.copy()
Q_LOWER_LIMITS_restricted_np[:7] += np.deg2rad(10.0)

Q_UPPER_LIMITS_restricted_np = Q_UPPER_LIMITS_np.copy()
Q_UPPER_LIMITS_restricted_np[:7] -= np.deg2rad(10.0)
assert Q_LOWER_LIMITS_restricted_np.shape == (29,), f"Q_LOWER_LIMITS_restricted_np.shape: {Q_LOWER_LIMITS_restricted_np.shape}, expected: (29,)"
assert Q_UPPER_LIMITS_restricted_np.shape == (29,), f"Q_UPPER_LIMITS_restricted_np.shape: {Q_UPPER_LIMITS_restricted_np.shape}, expected: (29,)"

OBS_NAME_TO_NAMES = {
    "joint_pos": [f"{name}_q" for name in JOINT_NAMES_ISAACGYM],
    "joint_vel": [f"{name}_qd" for name in JOINT_NAMES_ISAACGYM],
    "prev_action_targets": [f"{name}_prev_action_target" for name in JOINT_NAMES_ISAACGYM],
    "palm_pos": [f"palm_center_pos_{x}" for x in "xyz"],
    "palm_rot": [f"palm_rot_{x}" for x in "xyzw"],
    "object_rot": [f"object_rot_{x}" for x in "xyzw"],
    "keypoints_rel_palm": [
        f"keypoints_rel_palm_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "keypoints_rel_goal": [
        f"keypoints_rel_goal_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "fingertip_pos_rel_palm": [
        f"fingertip_rel_pos_{finger}_{x}"
        for finger in ["index", "middle", "ring", "thumb", "pinky"]
        for x in "xyz"
    ],
    "object_scales": [f"object_scales_{x}" for x in "xyz"],
}
OBS_NAMES = sum(OBS_NAME_TO_NAMES.values(), [])
N_OBS = 140
assert len(OBS_NAMES) == N_OBS, f"len(OBS_NAMES): {len(OBS_NAMES)}, expected: {N_OBS}"

T_W_R_np = np.eye(4)
T_W_R_np[:3, 3] = np.array([0.0, 0.8, 0.0])

PALM_OFFSET_np = np.array([-0.00, -0.02, 0.16])

FINGERTIP_OFFSETS_np = np.array(
    [[0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0], [0.02, 0.002, 0]]
)
OBJECT_KEYPOINT_OFFSETS_np = np.array(
    [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
)


def create_urdf_object(robot_name: Literal["iiwa14", "iiwa7", "iiwa14_left_sharpa_between", "iiwa14_left_sharpa_adjusted_restricted"]) -> yourdfpy.URDF:
    asset_root = Path(__file__).parent / "../../assets"
    assert asset_root.exists(), f"Asset root {asset_root} does not exist"
    if robot_name == "iiwa14":
        urdf_path = asset_root / "urdf/kuka_allegro_description/iiwa14_real.urdf"
    elif robot_name == "iiwa7":
        urdf_path = asset_root / "urdf/kuka_allegro_description/iiwa7_real.urdf"
    elif robot_name == "iiwa14_left_sharpa_between":
        urdf_path = asset_root / "urdf/kuka_allegro_description/iiwa14_left_sharpa_between.urdf"
    elif robot_name == "iiwa14_left_sharpa_adjusted_restricted":
        urdf_path = asset_root / "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")
    assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"
    return yourdfpy.URDF.load(urdf_path)


def compute_fk_dict(urdf: yourdfpy.URDF, q: np.ndarray, link_names: list[str]) -> dict[str, np.ndarray]:
    N = q.shape[0]
    assert q.shape == (N, 29), f"q.shape: {q.shape}, expected: (N, 29)"
    fk_dict = defaultdict(list)
    for i in range(N):
        urdf.update_cfg(q[i])
        for link_name in link_names:
            fk_dict[link_name].append(urdf.get_transform(frame_to=link_name))
    for link_name in link_names:
        fk_dict[link_name] = np.stack(fk_dict[link_name], axis=0)
        assert fk_dict[link_name].shape == (N, 4, 4), f"fk_dict[link_name].shape: {fk_dict[link_name].shape}, expected: (N, 4, 4)"
    return fk_dict


def compute_observation(
    q: np.ndarray,
    qd: np.ndarray,
    prev_action_targets: np.ndarray,
    object_pose: np.ndarray,
    goal_object_pose: np.ndarray,
    object_scales: np.ndarray,
    urdf: yourdfpy.URDF,
    obs_list: list[str],
) -> np.ndarray:
    # Assume q and qd are in the order of JOINT_NAMES_ISAACGYM
    # object_pose, goal_object_pose are the pose of the object and goal in world frame (xyz_xyzw)
    # object_scales is the scale of the object [x, y, z]
    # chain is to compute fk of robot across all links
    # palm_serial_chain is to compute the jacobian of the palm link
    import time
    t0 = time.time()

    N = q.shape[0]
    J = 29
    assert q.shape == (N, J), f"q.shape: {q.shape}, expected: (N, J)"
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, J)"
    assert prev_action_targets.shape == (N, J), (
        f"prev_action_targets.shape: {prev_action_targets.shape}, expected: (N, J)"
    )
    q_lower_limits = Q_LOWER_LIMITS_np
    q_upper_limits = Q_UPPER_LIMITS_np
    assert q_lower_limits.shape == (J,), (
        f"q_lower_limits.shape: {q_lower_limits.shape}, expected: (J,)"
    )
    assert q_upper_limits.shape == (J,), (
        f"q_upper_limits.shape: {q_upper_limits.shape}, expected: (J,)"
    )
    assert object_pose.shape == (N, 7), (
        f"object_pose.shape: {object_pose.shape}, expected: (N, 7)"
    )
    assert goal_object_pose.shape == (N, 7), (
        f"goal_object_pose.shape: {goal_object_pose.shape}, expected: (N, 7)"
    )
    assert object_scales.shape == (N, 3), (
        f"object_scales.shape: {object_scales.shape}, expected: (N, 3)"
    )
    assert set(obs_list).issubset(set(OBS_NAME_TO_NAMES.keys())), f"obs_list: {obs_list} is not a subset of OBS_NAME_TO_NAMES.keys(): {OBS_NAME_TO_NAMES.keys()}"
    t1 = time.time()

    # q unscaled
    q_unscaled = unscale(
        x=q,
        lower=q_lower_limits,
        upper=q_upper_limits,
    )
    t2 = time.time()

    # FK to get link poses
    N_FINGERTIPS = 5
    assert JOINT_NAMES_ISAACGYM == urdf.actuated_joint_names, f"JOINT_NAMES_ISAACGYM: {JOINT_NAMES_ISAACGYM} != urdf.actuated_joint_names: {urdf.actuated_joint_names}"
    LINK_NAMES = ["iiwa14_link_7"] + ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
    fk_dict = compute_fk_dict(urdf=urdf, q=q, link_names=LINK_NAMES)
    t3 = time.time()
    palm_center_pos, palm_rot = _compute_palm_center_pos_and_rot(fk_dict=fk_dict)
    t4 = time.time()
    fingertip_positions_with_offsets = _compute_fingertip_positions_with_offsets(
        fk_dict=fk_dict
    )
    t5 = time.time()
    fingertip_rel_pos = fingertip_positions_with_offsets - palm_center_pos[:, None]
    t6 = time.time()
    assert palm_center_pos.shape == (N, 3), (
        f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    )
    assert fingertip_rel_pos.shape == (N, N_FINGERTIPS, 3), (
        f"fingertip_rel_pos.shape: {fingertip_rel_pos.shape}, expected: (N, N_FINGERTIPS, 3)"
    )

    # keypoint positions
    N_KEYPOINTS = 4
    object_keypoint_positions = _compute_keypoint_positions(
        pose=object_pose, scales=object_scales
    )
    t7 = time.time()
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=goal_object_pose, scales=object_scales
    )
    t8 = time.time()
    keypoints_rel_palm = object_keypoint_positions - palm_center_pos[:, None]
    t9 = time.time()
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    t10 = time.time()
    assert keypoints_rel_palm.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_palm.shape: {keypoints_rel_palm.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )

    # Object rot
    object_rot = object_pose[:, 3:7]
    t11 = time.time()
    assert object_rot.shape == (N, 4), (
        f"object_rot.shape: {object_rot.shape}, expected: (N, 4)"
    )

    obs_dict = {
        "joint_pos": q_unscaled,
        "joint_vel": qd,
        "prev_action_targets": prev_action_targets,
        "palm_pos": palm_center_pos,
        "palm_rot": palm_rot,
        "object_rot": object_rot,
        "keypoints_rel_palm": keypoints_rel_palm.reshape(N, -1),
        "keypoints_rel_goal": keypoints_rel_goal.reshape(N, -1),
        "fingertip_pos_rel_palm": fingertip_rel_pos.reshape(N, -1),
        "object_scales": object_scales,
    }
    t12 = time.time()
    for k, v in obs_dict.items():
        assert v.ndim == 2, f"v.ndim: {v.ndim}, expected: 2 for key: {k}: {v.shape}"
        assert v.shape[0] == N, f"v.shape[0]: {v.shape[0]}, expected: {N} for key: {k}"
    for name, names in OBS_NAME_TO_NAMES.items():
        assert name in obs_dict, f"name: {name} not in obs_dict"
        assert obs_dict[name].shape[1] == len(names), (
            f"obs_dict[name].shape[1]: {obs_dict[name].shape[1]}, expected: {len(names)} for name: {name}"
        )
    t13 = time.time()

    obs = np.concatenate(
        [obs_dict[key] for key in obs_list],
        axis=-1,
    )
    t14 = time.time()

    print("IN COMPUTE OBS")
    print("=" * 100)
    total_dt = t14 - t0
    # Compute each dt in ms and as a fraction of the total and print as a percentage and absolute value
    print(f"total_dt: {total_dt:.6f} s")
    print(f"t1 - t0: {(t1 - t0) * 1000:.1f} ms, {((t1 - t0)) / total_dt * 100:.1f}%")
    print(f"t2 - t1: {(t2 - t1) * 1000:.1f} ms, {((t2 - t1)) / total_dt * 100:.1f}%")
    print(f"t3 - t2: {(t3 - t2) * 1000:.1f} ms, {((t3 - t2)) / total_dt * 100:.1f}%")
    print(f"t4 - t3: {(t4 - t3) * 1000:.1f} ms, {((t4 - t3)) / total_dt * 100:.1f}%")
    print(f"t5 - t4: {(t5 - t4) * 1000:.1f} ms, {((t5 - t4)) / total_dt * 100:.1f}%")
    print(f"t6 - t5: {(t6 - t5) * 1000:.1f} ms, {((t6 - t5)) / total_dt * 100:.1f}%")
    print(f"t7 - t6: {(t7 - t6) * 1000:.1f} ms, {((t7 - t6)) / total_dt * 100:.1f}%")
    print(f"t8 - t7: {(t8 - t7) * 1000:.1f} ms, {((t8 - t7)) / total_dt * 100:.1f}%")
    print(f"t9 - t8: {(t9 - t8) * 1000:.1f} ms, {((t9 - t8)) / total_dt * 100:.1f}%")
    print(f"t10 - t9: {(t10 - t9) * 1000:.1f} ms, {((t10 - t9)) / total_dt * 100:.1f}%")
    print(f"t11 - t10: {(t11 - t10) * 1000:.1f} ms, {((t11 - t10)) / total_dt * 100:.1f}%")
    print(f"t12 - t11: {(t12 - t11) * 1000:.1f} ms, {((t12 - t11)) / total_dt * 100:.1f}%")
    print(f"t13 - t12: {(t13 - t12) * 1000:.1f} ms, {((t13 - t12)) / total_dt * 100:.1f}%")
    print(f"t14 - t13: {(t14 - t13) * 1000:.1f} ms, {((t14 - t13)) / total_dt * 100:.1f}%")
    print("=" * 100)

    assert obs.shape == (N, N_OBS), f"obs.shape: {obs.shape}, expected: (N, {N_OBS})"
    return obs


def compute_joint_pos_targets(
    actions: Tensor,
    prev_targets: Tensor,
    hand_moving_average: float,
    arm_moving_average: float,
    hand_dof_speed_scale: float,
    dt: float,
) -> Tensor:
    N = actions.shape[0]
    J = 29
    assert actions.shape == (N, J), f"actions.shape: {actions.shape}, expected: (N, J)"
    assert prev_targets.shape == (N, J), (
        f"prev_targets.shape: {prev_targets.shape}, expected: (N, J)"
    )
    q_lower_limits = torch.from_numpy(Q_LOWER_LIMITS_np).float().to(actions.device)
    q_upper_limits = torch.from_numpy(Q_UPPER_LIMITS_np).float().to(actions.device)
    assert q_lower_limits.shape == (J,), (
        f"q_lower_limits.shape: {q_lower_limits.shape}, expected: (J,)"
    )
    assert q_upper_limits.shape == (J,), (
        f"q_upper_limits.shape: {q_upper_limits.shape}, expected: (J,)"
    )
    assert 0.0 <= hand_moving_average <= 1.0, (
        f"hand_moving_average: {hand_moving_average}, expected: (0.0, 1.0)"
    )
    assert 0.0 <= arm_moving_average <= 1.0, (
        f"arm_moving_average: {arm_moving_average}, expected: (0.0, 1.0)"
    )

    # hand
    cur_targets = prev_targets.clone()
    cur_targets[:, 7:29] = scale(
        actions[:, 7:29],
        q_lower_limits[7:29],
        q_upper_limits[7:29],
    )
    cur_targets[:, 7:29] = (
        hand_moving_average * cur_targets[:, 7:29]
        + (1.0 - hand_moving_average) * prev_targets[:, 7:29]
    )
    cur_targets[:, 7:29] = tensor_clamp(
        cur_targets[:, 7:29],
        q_lower_limits[7:29],
        q_upper_limits[7:29],
    )

    # arm
    cur_targets[:, :7] = (
        prev_targets[:, :7] + hand_dof_speed_scale * dt * actions[:, :7]
    )
    cur_targets[:, :7] = tensor_clamp(
        cur_targets[:, :7],
        q_lower_limits[:7],
        q_upper_limits[:7],
    )
    cur_targets[:, :7] = (
        arm_moving_average * cur_targets[:, :7]
        + (1.0 - arm_moving_average) * prev_targets[:, :7]
    )
    return cur_targets


def _compute_palm_center_pos_and_rot(
    fk_dict: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    import time
    t00 = time.time()
    T_R_Ps = fk_dict["iiwa14_link_7"]
    N = T_R_Ps.shape[0]
    t01 = time.time()
    T_W_Rs = T_W_R_np[None]
    assert T_W_Rs.shape == (1, 4, 4), (
        f"T_W_Rs.shape: {T_W_Rs.shape}, expected: (1, 4, 4)"
    )
    t02 = time.time()

    T_W_Ps = T_W_Rs @ T_R_Ps
    t03 = time.time()

    palm_offset = PALM_OFFSET_np[None].repeat(N, axis=0)
    t04 = time.time()
    assert palm_offset.shape == (N, 3), (
        f"palm_offset.shape: {palm_offset.shape}, expected: (N, 3)"
    )
    t05 = time.time()
    palm_pos = T_W_Ps[:, :3, 3]
    t06 = time.time()
    palm_rot = T_W_Ps[:, :3, :3]
    t07 = time.time()
    # palm_quat_wxyz = matrix_to_quaternion(palm_rot)
    # palm_quat_xyzw = palm_quat_wxyz[:, [1, 2, 3, 0]]

    t07_5 = time.time()
    palm_quat_xyzw = matrix_to_quaternion_xyzw_scipy(palm_rot)
    t08 = time.time()
    t09 = time.time()

    palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, palm_offset)
    t10 = time.time()
    assert palm_center_pos.shape == (N, 3), (
        f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    )
    t11 = time.time()
    total_dt = t11 - t00
    # print("IN _compute_palm_center_pos_and_rot")
    # print("=" * 100)
    # print(f"total_dt: {total_dt:.6f} s")
    # print(f"t01 - t00: {(t01 - t00) * 1000:.1f} ms, {((t01 - t00)) / total_dt * 100:.1f}%")
    # print(f"t02 - t01: {(t02 - t01) * 1000:.1f} ms, {((t02 - t01)) / total_dt * 100:.1f}%")
    # print(f"t03 - t02: {(t03 - t02) * 1000:.1f} ms, {((t03 - t02)) / total_dt * 100:.1f}%")
    # print(f"t04 - t03: {(t04 - t03) * 1000:.1f} ms, {((t04 - t03)) / total_dt * 100:.1f}%")
    # print(f"t05 - t04: {(t05 - t04) * 1000:.1f} ms, {((t05 - t04)) / total_dt * 100:.1f}%")
    # print(f"t06 - t05: {(t06 - t05) * 1000:.1f} ms, {((t06 - t05)) / total_dt * 100:.1f}%")
    # print(f"t07 - t06: {(t07 - t06) * 1000:.1f} ms, {((t07 - t06)) / total_dt * 100:.1f}%")
    # print(f"t07_5 - t07: {(t07_5 - t07) * 1000:.1f} ms, {((t07_5 - t07)) / total_dt * 100:.1f}%")
    # print(f"t08 - t07_5: {(t08 - t07_5) * 1000:.1f} ms, {((t08 - t07_5)) / total_dt * 100:.1f}%")
    # print(f"t09 - t08: {(t09 - t08) * 1000:.1f} ms, {((t09 - t08)) / total_dt * 100:.1f}%")
    # print(f"t10 - t09: {(t10 - t09) * 1000:.1f} ms, {((t10 - t09)) / total_dt * 100:.1f}%")
    # print(f"t11 - t10: {(t11 - t10) * 1000:.1f} ms, {((t11 - t10)) / total_dt * 100:.1f}%")
    # print("=" * 100)
    return palm_center_pos, palm_quat_xyzw



def _compute_fingertip_positions_with_offsets(
    fk_dict: dict[str, np.ndarray],
) -> np.ndarray:
    N_FINGERTIPS = 5
    T_R_F_list = [
        fk_dict[name]
        for name in ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
    ]
    T_R_Fs = np.stack(T_R_F_list, axis=1)
    N = T_R_Fs.shape[0]
    assert T_R_Fs.shape == (N, N_FINGERTIPS, 4, 4), (
        f"T_R_Fs.shape: {T_R_Fs.shape}, expected: (N, N_FINGERTIPS, 4, 4)"
    )

    T_W_Rs = T_W_R_np[None, None]
    assert T_W_Rs.shape == (1, 1, 4, 4), (
        f"T_W_Rs.shape: {T_W_Rs.shape}, expected: (1, 1, 4, 4)"
    )

    T_W_Fs = T_W_Rs @ T_R_Fs
    fingertip_positions = T_W_Fs[:, :, :3, 3]
    fingertip_rots = T_W_Fs[:, :, :3, :3]
    fingertip_quat_xyzw = matrix_to_quaternion_xyzw_scipy(fingertip_rots.reshape(-1, 3, 3)).reshape(N, N_FINGERTIPS, 4)

    fingertip_offsets = FINGERTIP_OFFSETS_np[None].repeat(N, axis=0)
    assert fingertip_offsets.shape == (N, N_FINGERTIPS, 3), (
        f"fingertip_offsets.shape: {fingertip_offsets.shape}, expected: (N, N_FINGERTIPS, 3)"
    )
    fingertip_positions_with_offsets = np.zeros(
        (N, N_FINGERTIPS, 3), dtype=np.float32
    )
    for i in range(N_FINGERTIPS):
        fingertip_positions_with_offsets[:, i] = fingertip_positions[
            :, i
        ] + quat_rotate(fingertip_quat_xyzw[:, i], fingertip_offsets[:, i])
    return fingertip_positions_with_offsets


def _compute_keypoint_positions(
    pose: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    N = pose.shape[0]
    assert pose.shape == (N, 7), f"pose.shape: {pose.shape}, expected: (N, 7)"
    assert scales.shape == (N, 3), f"scales.shape: {scales.shape}, expected: (N, 3)"

    OBJECT_BASE_SIZE = 0.04
    KEYPOINT_SCALE = 1.5
    object_keypoint_offsets = OBJECT_KEYPOINT_OFFSETS_np[None] * OBJECT_BASE_SIZE * KEYPOINT_SCALE / 2 * scales[:, None]
    N_KEYPOINTS = 4
    assert object_keypoint_offsets.shape == (N, N_KEYPOINTS, 3), (
        f"object_keypoint_offsets.shape: {object_keypoint_offsets.shape}, expected: (N, N_KEYPOINTS, 3)"
    )

    pos = pose[:, :3]
    quat_xyzw = pose[:, 3:7]

    keypoint_positions = np.zeros(
        (N, N_KEYPOINTS, 3), dtype=np.float32
    )
    for i in range(N_KEYPOINTS):
        keypoint_positions[:, i] = pos + quat_rotate(
            quat_xyzw, object_keypoint_offsets[:, i]
        )
    return keypoint_positions


"""
Frames:
* Let W = world, R = robot base, P = palm, O = object, G = goal, F = fingertip, C = camera
* Currently, W != R because the table is placed at (x=0, y=0, z=0.38) such that the tabletop is at x=0.53
  * Can consider moving W = R to simplify things
* T_W_R = constant (x=0, y=0.8, z=0)
* T_R_P = fk(q)
* T_R_F = fk(q)
* T_R_C = constant from camera calibration
* T_C_O = FoundationPose(image, mesh)
* T_?_G = Human video or SORA video or manually specified trajectory

Considerations:
* Likely need a stateful ObservationComputer class to store a fk function (yourdfpy or pytorch_kinematics)
* Need to have some hardcoded values in observation computation like keypoint offsets and link pose offsets
* Because of above, we may want a separate node to compute these so we can visualize them without publishing them from here so policy can run as fast as possible
^ decided to keep them all in this file for now to simplify things
"""

"""
Pseudocode for how real world code will look:

@dataclass
class Config:
    act_moving_average: float
    hand_dof_speed_scale: float
    dt: float

@dataclass
class FixedRawObservation:
    object_scales: Tensor

@dataclass
class RawObservation:
    q: Tensor
    qd: Tensor
    object_pose: Tensor
    goal_object_pose: Tensor

@dataclass
class PolicyState:
    prev_targets: Tensor

# Load config
cfg = Config(
    act_moving_average=0.1,
    hand_dof_speed_scale=1.0,
    dt=1/60,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load chain and palm_serial_chain
chain, palm_serial_chain = create_chain_and_serial_chain(device=device, robot_name="iiwa14")

# Load policy
wandb_run_path = "tylerlum/sapg_allegro_kuka_reorientation/uid_00_default_marker_2025-10-18_01-43-44"
policy = load_policy(wandb_run_path)

# Load fixed raw observation
fixed_raw_obs = FixedRawObservation(
    object_scales=[...],
)

# Get initial raw observation to initialize policy state
raw_obs = get_raw_obs()
policy_state = PolicyState(
    prev_targets=raw_obs.q,
)

# Main loop
loop_dts, loop_without_sleep_dts = [], []
while True:
    start_time = time.time()

    # Get latest raw observation
    raw_obs = get_raw_obs()

    # Compute policy observation
    obs = compute_observation(
        q=raw_obs.q,
        qd=raw_obs.qd,
        object_pose=raw_obs.object_pose,
        goal_object_pose=raw_obs.goal_object_pose,
        object_scales=fixed_raw_obs.object_scales,
        chain=chain,
        palm_serial_chain=palm_serial_chain,
    )

    # Get policy action
    actions = policy.get_action(obs)

    # Compute joint position targets
    joint_pos_targets = compute_joint_pos_targets(
        actions=actions,
        prev_targets=policy_state.prev_targets,
        act_moving_average=cfg.act_moving_average,
        hand_dof_speed_scale=cfg.hand_dof_speed_scale,
        dt=cfg.dt,
    )

    # Update policy state
    policy_state.prev_targets = joint_pos_targets

    # Send joint position targets to robot
    send_joint_pos_targets(joint_pos_targets)

    # Sleep and log
    end_time = time.time()
    loop_without_sleep_dt = end_time - start_time
    sleep_time = cfg.dt - loop_without_sleep_dt
    if sleep_time > 0:
        time.sleep(sleep_time)
        loop_dt = loop_without_sleep_dt + sleep_time
    else:
        print(f"Loop too slow! Desired FPS: {1.0 / cfg.dt:.1f}, Actual FPS: {1.0 / loop_without_sleep_dt:.1f}")
        loop_dt = loop_without_sleep_dt
    loop_dts.append(loop_dt)
    loop_without_sleep_dts.append(loop_without_sleep_dt)
    if len(loop_dts) > 100:
        avg_loop_dt = sum(loop_dts) / len(loop_dts)
        avg_loop_without_sleep_dt = sum(loop_without_sleep_dts) / len(loop_without_sleep_dts)
        print(f'FPS: {1.0 / avg_loop_dt:.1f}, Max FPS: {1.0 / avg_loop_without_sleep_dt:.1f}")
"""
