from __future__ import annotations

from typing import Literal, Tuple
import numpy as np
import pytorch_kinematics as pk
import torch
from pathlib import Path
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import (
    matrix_to_quaternion,
    quat_rotate,
    scale,
    tensor_clamp,
    unscale,
)

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


def create_chain_and_serial_chain(device: str, robot_name: Literal["iiwa14", "iiwa7", "iiwa14_left_sharpa_between", "iiwa14_left_sharpa_adjusted_restricted"]) -> Tuple[pk.Chain, pk.SerialChain]:
    asset_root = Path(__file__).parent / "../../assets"
    assert asset_root.exists(), f"Asset root {asset_root} does not exist"

    if robot_name == "iiwa14":
        urdf_path = (
            asset_root / "urdf/kuka_allegro_description/iiwa14_real.urdf"
        )
        palm_link_name = "iiwa14_link_7"
    elif robot_name == "iiwa7":
        urdf_path = (
            asset_root / "urdf/kuka_allegro_description/iiwa7_real.urdf"
        )
        palm_link_name = "iiwa7_link_7"
    elif robot_name == "iiwa14_left_sharpa_between":
        urdf_path = (
            asset_root / "urdf/kuka_allegro_description/iiwa14_left_sharpa_between.urdf"
        )
        palm_link_name = "iiwa14_link_7"
    elif robot_name == "iiwa14_left_sharpa_adjusted_restricted":
        urdf_path = (
            asset_root / "urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        palm_link_name = "iiwa14_link_7"
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")
    assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"

    chain = pk.build_chain_from_urdf(
        open(urdf_path).read(),
    ).to(device=device)
    palm_serial_chain = pk.SerialChain(chain, palm_link_name).to(
        device=device
    )
    return chain, palm_serial_chain


def compute_observation(
    q: Tensor,
    qd: Tensor,
    prev_action_targets: Tensor,
    object_pose: Tensor,
    goal_object_pose: Tensor,
    object_scales: Tensor,
    chain: pk.Chain,
    obs_list: list[str],
) -> Tensor:
    # Assume q and qd are in the order of JOINT_NAMES_ISAACGYM
    # object_pose, goal_object_pose are the pose of the object and goal in world frame (xyz_xyzw)
    # object_scales is the scale of the object [x, y, z]
    # chain is to compute fk of robot across all links
    # palm_serial_chain is to compute the jacobian of the palm link

    N = q.shape[0]
    J = 29
    assert q.shape == (N, J), f"q.shape: {q.shape}, expected: (N, J)"
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, J)"
    assert prev_action_targets.shape == (N, J), (
        f"prev_action_targets.shape: {prev_action_targets.shape}, expected: (N, J)"
    )
    q_lower_limits = torch.from_numpy(Q_LOWER_LIMITS_np).float().to(q.device)
    q_upper_limits = torch.from_numpy(Q_UPPER_LIMITS_np).float().to(q.device)
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

    # q unscaled
    q_unscaled = unscale(
        x=q,
        lower=q_lower_limits,
        upper=q_upper_limits,
    )

    # FK to get link poses
    N_FINGERTIPS = 5
    fk_dict = chain.forward_kinematics(
        _change_joint_order(
            q=q,
            from_order=JOINT_NAMES_ISAACGYM,
            to_order=chain.get_joint_parameter_names(),
        )
    )
    palm_center_pos, palm_rot = _compute_palm_center_pos_and_rot(fk_dict=fk_dict)
    fingertip_positions_with_offsets = _compute_fingertip_positions_with_offsets(
        fk_dict=fk_dict
    )
    fingertip_rel_pos = fingertip_positions_with_offsets - palm_center_pos.unsqueeze(
        dim=1
    )
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
    goal_keypoint_positions = _compute_keypoint_positions(
        pose=goal_object_pose, scales=object_scales
    )
    keypoints_rel_palm = object_keypoint_positions - palm_center_pos.unsqueeze(dim=1)
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    assert keypoints_rel_palm.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_palm.shape: {keypoints_rel_palm.shape}, expected: (N, N_KEYPOINTS, 3)"
    )
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), (
        f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"
    )

    # Object rot
    object_rot = object_pose[:, 3:7]
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
    for k, v in obs_dict.items():
        assert v.ndim == 2, f"v.ndim: {v.ndim}, expected: 2 for key: {k}: {v.shape}"
        assert v.shape[0] == N, f"v.shape[0]: {v.shape[0]}, expected: {N} for key: {k}"
    for name, names in OBS_NAME_TO_NAMES.items():
        assert name in obs_dict, f"name: {name} not in obs_dict"
        assert obs_dict[name].shape[1] == len(names), (
            f"obs_dict[name].shape[1]: {obs_dict[name].shape[1]}, expected: {len(names)} for name: {name}"
        )

    obs = torch.cat(
        [obs_dict[key] for key in obs_list],
        dim=-1,
    )

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


def _change_joint_order(
    q: torch.Tensor,
    from_order: list[str],
    to_order: list[str],
    require_all_joints: bool = True,
) -> torch.Tensor:
    J = len(from_order)
    assert q.ndim in [1, 2], f"Expected q to be either (N,) or (N, J), got {q.shape}"
    assert q.shape[-1] == J, (
        f"Expected q to have the same length as from_order, got {q.shape[-1]} and {J}"
    )

    if require_all_joints:
        assert len(to_order) == J, (
            f"Expected to_order to have the same length as from_order, got {len(to_order)} and {len(from_order)}. If you don't want to require all joints, set require_all_joints to False."
        )

    # q is given in the from_order
    joint_name_to_value = {from_order[i]: q[..., i] for i in range(J)}
    new_q = torch.stack([joint_name_to_value[name] for name in to_order], dim=-1)

    assert new_q.shape == (q.shape[:-1] + (len(to_order),)), (
        f"Expected new_q to be {q.shape[:-1] + (len(to_order),)}, got {new_q.shape}"
    )
    if require_all_joints:
        assert new_q.shape == q.shape, (
            f"Expected new_q to be {q.shape}, got {new_q.shape}"
        )
    return new_q


def _compute_palm_center_pos_and_rot(
    fk_dict: dict[str, pk.Transform3d],
) -> tuple[Tensor, Tensor]:
    # T_R_Ps = fk_dict["iiwa7_link_7"].get_matrix()
    T_R_Ps = fk_dict["iiwa14_link_7"].get_matrix()
    N = T_R_Ps.shape[0]

    T_W_Rs = (
        torch.from_numpy(T_W_R_np)
        .float()
        .to(T_R_Ps.device)[None]
        .repeat_interleave(N, dim=0)
    )
    assert T_W_Rs.shape == (N, 4, 4), (
        f"T_W_Rs.shape: {T_W_Rs.shape}, expected: (N, 4, 4)"
    )

    T_W_Ps = T_W_Rs @ T_R_Ps

    palm_offset = (
        torch.from_numpy(PALM_OFFSET_np)
        .float()
        .to(T_W_Ps.device)[None]
        .repeat_interleave(N, dim=0)
    )
    assert palm_offset.shape == (N, 3), (
        f"palm_offset.shape: {palm_offset.shape}, expected: (N, 3)"
    )

    palm_pos = T_W_Ps[:, :3, 3]
    palm_rot = T_W_Ps[:, :3, :3]
    palm_quat_wxyz = matrix_to_quaternion(palm_rot)
    palm_quat_xyzw = palm_quat_wxyz[:, [1, 2, 3, 0]]

    palm_center_pos = palm_pos + quat_rotate(palm_quat_xyzw, palm_offset)
    assert palm_center_pos.shape == (N, 3), (
        f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    )
    return palm_center_pos, palm_quat_xyzw


def _compute_palm_linvel_and_angvel(
    qd: Tensor,
    jacobian: Tensor,
) -> tuple[Tensor, Tensor]:
    N = qd.shape[0]
    J = qd.shape[1]
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, J)"
    assert jacobian.shape == (N, 6, J), (
        f"jacobian.shape: {jacobian.shape}, expected: (N, 6, J)"
    )

    # (N, 6, 1) = (N, 6, J) @ (N, J, 1)
    v_omega = torch.bmm(jacobian, qd.unsqueeze(dim=-1)).squeeze(dim=-1)
    assert v_omega.shape == (N, 6), f"v_omega.shape: {v_omega.shape}, expected: (N, 6)"

    v, omega = v_omega[:, :3], v_omega[:, 3:]

    # Frames for velocity
    # wrt = with respect to = velocity relative to what frame (this choice is NOT simply a rotation, but changes the magnitude if the other frame is moving)
    # in = in the frame = the frame in which the velocity is expressed (this choice is simply a rotation, no change to magnitude)
    lin_vel_wrt_R_in_R = v
    ang_vel_wrt_R_in_R = omega
    return lin_vel_wrt_R_in_R, ang_vel_wrt_R_in_R


def _compute_fingertip_positions_with_offsets(
    fk_dict: dict[str, pk.Transform3d],
) -> Tensor:
    N_FINGERTIPS = 5
    T_R_F_list = [
        fk_dict[name].get_matrix()
        for name in ["left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP"]
    ]
    T_R_Fs = torch.stack(T_R_F_list, dim=1)
    N = T_R_Fs.shape[0]
    assert T_R_Fs.shape == (N, N_FINGERTIPS, 4, 4), (
        f"T_R_Fs.shape: {T_R_Fs.shape}, expected: (N, N_FINGERTIPS, 4, 4)"
    )

    T_W_Rs = (
        torch.from_numpy(T_W_R_np)
        .float()
        .to(T_R_Fs.device)[None]
        .repeat_interleave(N, dim=0)[:, None]
        .repeat_interleave(N_FINGERTIPS, dim=1)
    )
    assert T_W_Rs.shape == (N, N_FINGERTIPS, 4, 4), (
        f"T_W_Rs.shape: {T_W_Rs.shape}, expected: (N, N_FINGERTIPS, 4, 4)"
    )

    T_W_Fs = T_W_Rs @ T_R_Fs
    fingertip_positions = T_W_Fs[:, :, :3, 3]
    fingertip_rots = T_W_Fs[:, :, :3, :3]
    fingertip_quat_wxyz = matrix_to_quaternion(fingertip_rots)
    fingertip_quat_xyzw = fingertip_quat_wxyz[..., [1, 2, 3, 0]]
    fingertip_offsets = (
        torch.from_numpy(FINGERTIP_OFFSETS_np)
        .float()
        .to(fingertip_positions.device)[None]
        .repeat_interleave(N, dim=0)
    )
    assert fingertip_offsets.shape == (N, N_FINGERTIPS, 3), (
        f"fingertip_offsets.shape: {fingertip_offsets.shape}, expected: (N, N_FINGERTIPS, 3)"
    )
    fingertip_positions_with_offsets = torch.zeros(
        (N, N_FINGERTIPS, 3), dtype=torch.float, device=fingertip_positions.device
    )
    for i in range(N_FINGERTIPS):
        fingertip_positions_with_offsets[:, i] = fingertip_positions[
            :, i
        ] + quat_rotate(fingertip_quat_xyzw[:, i], fingertip_offsets[:, i])
    return fingertip_positions_with_offsets


def _compute_keypoint_positions(
    pose: Tensor,
    scales: Tensor,
) -> Tensor:
    N = pose.shape[0]
    assert pose.shape == (N, 7), f"pose.shape: {pose.shape}, expected: (N, 7)"
    assert scales.shape == (N, 3), f"scales.shape: {scales.shape}, expected: (N, 3)"

    OBJECT_BASE_SIZE = 0.04
    KEYPOINT_SCALE = 1.5
    object_keypoint_offsets = (
        torch.from_numpy(OBJECT_KEYPOINT_OFFSETS_np)
        .float()
        .to(pose.device)[None]
        .repeat_interleave(N, dim=0)
        * OBJECT_BASE_SIZE
        * KEYPOINT_SCALE
        / 2
        * scales.unsqueeze(dim=1)
    )
    N_KEYPOINTS = 4
    assert object_keypoint_offsets.shape == (N, N_KEYPOINTS, 3), (
        f"object_keypoint_offsets.shape: {object_keypoint_offsets.shape}, expected: (N, N_KEYPOINTS, 3)"
    )

    pos = pose[:, :3]
    quat_xyzw = pose[:, 3:7]

    keypoint_positions = torch.zeros(
        (N, N_KEYPOINTS, 3), dtype=torch.float, device=pose.device
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
