from __future__ import annotations

import numpy as np
import pytorch_kinematics as pk
import torch
from torch import Tensor

from isaacgymenvs.utils.torch_jit_utils import (
    matrix_to_quaternion,
    quat_rotate,
    scale,
    tensor_clamp,
    unscale,
)

# Constants
JOINT_NAMES_ISAACGYM = [
    "iiwa7_joint_1",
    "iiwa7_joint_2",
    "iiwa7_joint_3",
    "iiwa7_joint_4",
    "iiwa7_joint_5",
    "iiwa7_joint_6",
    "iiwa7_joint_7",
    "index_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_0",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_0",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_0",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
assert len(JOINT_NAMES_ISAACGYM) == 23, (
    f"len(JOINT_NAMES_ISAACGYM): {len(JOINT_NAMES_ISAACGYM)}, expected: 23"
)

OBS_NAME_TO_NAMES = {
    "q": [f"{name}_q" for name in JOINT_NAMES_ISAACGYM],
    "qd": [f"{name}_qd" for name in JOINT_NAMES_ISAACGYM],
    "palm_center_pos": [f"palm_center_pos_{x}" for x in "xyz"],
    "palm_rot": [f"palm_rot_{x}" for x in "xyzw"],
    "palm_linvel": [f"palm_linvel_{x}" for x in "xyz"],
    "palm_angvel": [f"palm_angvel_{x}" for x in "xyz"],
    "object_rot": [f"object_rot_{x}" for x in "xyzw"],
    "object_linvel": [f"object_linvel_{x}" for x in "xyz"],
    "object_angvel": [f"object_angvel_{x}" for x in "xyz"],
    "fingertip_rel_pos": [
        f"fingertip_rel_pos_{finger}_{x}"
        for finger in ["index", "middle", "ring", "thumb"]
        for x in "xyz"
    ],
    "keypoints_rel_palm": [
        f"keypoints_rel_palm_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "keypoints_rel_goal": [
        f"keypoints_rel_goal_{i}_{x}" for i in range(4) for x in "xyz"
    ],
    "object_scales": [f"object_scales_{x}" for x in "xyz"],
    "closest_keypoint_max_dist": ["closest_keypoint_max_dist"],
    "closest_fingertip_dist": [
        f"closest_fingertip_dist_{finger}"
        for finger in ["index", "middle", "ring", "thumb"]
    ],
    "lifted_object": ["lifted_object"],
    "progress_obs": ["progress_obs"],
    "successes": ["successes_obs"],
    "reward_obs": ["reward_obs"],
}
OBS_NAMES = sum(OBS_NAME_TO_NAMES.values(), [])
N_OBS = 117
assert len(OBS_NAMES) == N_OBS, f"len(OBS_NAMES): {len(OBS_NAMES)}, expected: {N_OBS}"

T_W_R_np = np.eye(4)
T_W_R_np[:3, 3] = np.array([0.0, 0.8, 0.0])

PALM_OFFSET_np = np.array([-0.00, -0.02, 0.16])
FINGERTIP_OFFSETS_np = np.array(
    [[0.05, 0.005, 0], [0.05, 0.005, 0], [0.05, 0.005, 0], [0.06, 0.005, 0]]
)
OBJECT_KEYPOINT_OFFSETS_np = np.array(
    [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]
)


def compute_observation(
    q: Tensor,
    qd: Tensor,
    q_lower_limits: Tensor,
    q_upper_limits: Tensor,
    object_pose: Tensor,
    goal_object_pose: Tensor,
    object_scales: Tensor,
    chain: pk.Chain,
    palm_serial_chain: pk.SerialChain,
) -> Tensor:
    # Assume q, qd, q_lower_limits, q_upper_limits are in the order of JOINT_NAMES_ISAACGYM
    # object_pose, goal_object_pose are the pose of the object and goal in world frame (xyz_xyzw)
    # object_scales is the scale of the object [x, y, z]
    # chain is to compute fk of robot across all links
    # palm_serial_chain is to compute the jacobian of the palm link

    N = q.shape[0]
    J = 23
    assert q.shape == (N, J), f"q.shape: {q.shape}, expected: (N, J)"
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, J)"
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

    # q unscaled
    q_unscaled = unscale(
        x=q,
        lower=q_lower_limits,
        upper=q_upper_limits,
    )

    # FK to get link poses
    N_FINGERTIPS = 4
    fk_dict = chain.forward_kinematics(
        _change_joint_order(
            q=q,
            from_order=JOINT_NAMES_ISAACGYM,
            to_order=chain.get_joint_parameter_names(),
        )
    )
    jacobian = palm_serial_chain.jacobian(
        _change_joint_order(
            q=q,
            from_order=JOINT_NAMES_ISAACGYM,
            to_order=palm_serial_chain.get_joint_parameter_names(),
            require_all_joints=False,
        )
    )
    palm_center_pos, palm_rot = _compute_palm_center_pos_and_rot(fk_dict=fk_dict)
    palm_linvel, palm_angvel = _compute_palm_linvel_and_angvel(
        qd=_change_joint_order(
            q=qd,
            from_order=JOINT_NAMES_ISAACGYM,
            to_order=palm_serial_chain.get_joint_parameter_names(),
            require_all_joints=False,
        ),
        jacobian=jacobian,
    )
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

    # Object velocity (0'd out)
    object_linvel = torch.zeros((N, 3), dtype=torch.float, device=q.device)
    object_angvel = torch.zeros((N, 3), dtype=torch.float, device=q.device)

    # Extra observations (0'd out)
    closest_keypoint_max_dist = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    N_FINGERTIPS = 4
    closest_fingertip_dist = torch.zeros(
        (N, N_FINGERTIPS), dtype=torch.float, device=q.device
    )
    lifted_object = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    progress_obs = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    successes = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    reward_obs = torch.zeros((N, 1), dtype=torch.float, device=q.device)

    obs_dict = {
        "q": q_unscaled,
        "qd": qd,
        "palm_center_pos": palm_center_pos,
        "palm_rot": palm_rot,
        "palm_linvel": palm_linvel,
        "palm_angvel": palm_angvel,
        "object_rot": object_rot,
        "object_linvel": object_linvel,
        "object_angvel": object_angvel,
        "fingertip_rel_pos": fingertip_rel_pos.reshape(N, -1),
        "keypoints_rel_palm": keypoints_rel_palm.reshape(N, -1),
        "keypoints_rel_goal": keypoints_rel_goal.reshape(N, -1),
        "object_scales": object_scales,
        "closest_keypoint_max_dist": closest_keypoint_max_dist,
        "closest_fingertip_dist": closest_fingertip_dist,
        "lifted_object": lifted_object,
        "progress_obs": progress_obs,
        "successes": successes,
        "reward_obs": reward_obs,
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
        [obs_dict[key] for key in obs_dict.keys()],
        dim=-1,
    )

    assert obs.shape == (N, N_OBS), f"obs.shape: {obs.shape}, expected: (N, {N_OBS})"
    return obs


def compute_joint_pos_targets(
    actions: Tensor,
    prev_targets: Tensor,
    q_lower_limits: Tensor,
    q_upper_limits: Tensor,
    act_moving_average: float,
    hand_dof_speed_scale: float,
    dt: float,
) -> Tensor:
    N = actions.shape[0]
    J = 23
    assert actions.shape == (N, J), f"actions.shape: {actions.shape}, expected: (N, J)"
    assert prev_targets.shape == (N, J), (
        f"prev_targets.shape: {prev_targets.shape}, expected: (N, J)"
    )
    assert q_lower_limits.shape == (J,), (
        f"q_lower_limits.shape: {q_lower_limits.shape}, expected: (J,)"
    )
    assert q_upper_limits.shape == (J,), (
        f"q_upper_limits.shape: {q_upper_limits.shape}, expected: (J,)"
    )
    assert 0.0 <= act_moving_average <= 1.0, (
        f"act_moving_average: {act_moving_average}, expected: (0.0, 1.0)"
    )

    # hand
    cur_targets = prev_targets.clone()
    cur_targets[:, 7:23] = scale(
        actions[:, 7:23],
        q_lower_limits[7:23],
        q_upper_limits[7:23],
    )
    cur_targets[:, 7:23] = (
        act_moving_average * cur_targets[:, 7:23]
        + (1.0 - act_moving_average) * prev_targets[:, 7:23]
    )
    cur_targets[:, 7:23] = tensor_clamp(
        cur_targets[:, 7:23],
        q_lower_limits[7:23],
        q_upper_limits[7:23],
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
    T_R_Ps = fk_dict["iiwa7_link_7"].get_matrix()
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
    N_FINGERTIPS = 4
    T_R_F_list = [
        fk_dict[name].get_matrix()
        for name in ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
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

    OBJECT_BASE_SIZE = 0.05
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
"""

"""
Pseudocode for how real world code will look:

@dataclass
class Config:
    q_lower_limits: Tensor
    q_upper_limits: Tensor
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
    q_lower_limits=[...],
    q_upper_limits=[...],
    act_moving_average=0.1,
    hand_dof_speed_scale=1.0,
    dt=1/60,
)

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
        q_lower_limits=cfg.q_lower_limits,
        q_upper_limits=cfg.q_upper_limits,
        object_pose=raw_obs.object_pose,
        goal_object_pose=raw_obs.goal_object_pose,
        object_scales=fixed_raw_obs.object_scales,
    )

    # Get policy action
    actions = policy.get_action(obs)

    # Compute joint position targets
    joint_pos_targets = compute_joint_pos_targets(
        actions=actions,
        prev_targets=policy_state.prev_targets,
        q_lower_limits=cfg.q_lower_limits,
        q_upper_limits=cfg.q_upper_limits,
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
