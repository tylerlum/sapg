from torch import Tensor
import torch
from isaacgymenvs.utils.torch_jit_utils import unscale, scale, tensor_clamp


def compute_observation(
    q: Tensor,
    qd: Tensor,
    q_lower_limits: Tensor,
    q_upper_limits: Tensor,
    object_pose: Tensor,
    goal_object_pose: Tensor,
    object_scales: Tensor,
) -> Tensor:
    # Assume q, qd, q_lower_limits, q_upper_limits are in the order of IsaacGym joint names
    # object_pose, goal_object_pose are the pose of the object and goal in world frame (xyz_xyzw)
    # object_scales is the scale of the object [x, y, z]

    N = q.shape[0]
    J = 23
    assert q.shape == (N, J), f"q.shape: {q.shape}, expected: (N, J)"
    assert qd.shape == (N, J), f"qd.shape: {qd.shape}, expected: (N, J)"
    assert q_lower_limits.shape == (J,), f"q_lower_limits.shape: {q_lower_limits.shape}, expected: (J,)"
    assert q_upper_limits.shape == (J,), f"q_upper_limits.shape: {q_upper_limits.shape}, expected: (J,)"
    assert object_pose.shape == (N, 7), f"object_pose.shape: {object_pose.shape}, expected: (N, 7)"
    assert goal_object_pose.shape == (N, 7), f"goal_object_pose.shape: {goal_object_pose.shape}, expected: (N, 7)"
    assert object_scales.shape == (N, 3), f"object_scales.shape: {object_scales.shape}, expected: (N, 3)"

    # q unscaled
    q_unscaled = unscale(
        x=q,
        lower=q_lower_limits,
        upper=q_upper_limits,
    )

    # FK to get link poses
    N_FINGERTIPS = 4
    palm_center_pos = compute_palm_center_pos(q)
    palm_rot = compute_palm_rot(q)
    palm_linvel = compute_palm_linvel(q, qd)
    palm_angvel = compute_palm_angvel(q, qd)
    fingertip_positions = compute_fingertip_positions(q)
    fingertip_rel_pos = fingertip_positions - palm_center_pos.unsqueeze(dim=1)
    assert palm_center_pos.shape == (N, 3), f"palm_center_pos.shape: {palm_center_pos.shape}, expected: (N, 3)"
    assert fingertip_rel_pos.shape == (N, N_FINGERTIPS, 3), f"fingertip_rel_pos.shape: {fingertip_rel_pos.shape}, expected: (N, N_FINGERTIPS, 3)"

    # keypoint positions
    N_KEYPOINTS = 4
    object_keypoint_positions = compute_keypoint_positions(object_pose)
    goal_keypoint_positions = compute_keypoint_positions(goal_object_pose)
    keypoints_rel_palm = object_keypoint_positions - palm_center_pos.unsqueeze(dim=1)
    keypoints_rel_goal = object_keypoint_positions - goal_keypoint_positions
    assert keypoints_rel_palm.shape == (N, N_KEYPOINTS, 3), f"keypoints_rel_palm.shape: {keypoints_rel_palm.shape}, expected: (N, N_KEYPOINTS, 3)"
    assert keypoints_rel_goal.shape == (N, N_KEYPOINTS, 3), f"keypoints_rel_goal.shape: {keypoints_rel_goal.shape}, expected: (N, N_KEYPOINTS, 3)"

    # Object rot
    object_rot = object_pose[:, 3:7]
    assert object_rot.shape == (N, 4), f"object_rot.shape: {object_rot.shape}, expected: (N, 4)"

    # Object velocity (0'd out)
    object_linvel = torch.zeros((N, 3), dtype=torch.float, device=q.device)
    object_angvel = torch.zeros((N, 3), dtype=torch.float, device=q.device)

    # Extra observations (0'd out)
    closest_keypoint_max_dist = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    closest_fingertip_dist = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    lifted_object = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    progress_obs = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    successes = torch.zeros((N, 1), dtype=torch.float, device=q.device)
    reward_obs = torch.zeros((N, 1), dtype=torch.float, device=q.device)

    obs = torch.cat([
        q_unscaled,
        qd,
        palm_center_pos,
        palm_rot,
        palm_linvel,
        palm_angvel,
        object_rot,
        object_linvel,
        object_angvel,
        fingertip_rel_pos,
        keypoints_rel_palm,
        keypoints_rel_goal,
        object_scales,
        closest_keypoint_max_dist,
        closest_fingertip_dist,
        lifted_object,
        progress_obs,
        successes,
        reward_obs,
    ], dim=-1)

    N_OBS = 100
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
    assert prev_targets.shape == (N, J), f"prev_targets.shape: {prev_targets.shape}, expected: (N, J)"
    assert q_lower_limits.shape == (J,), f"q_lower_limits.shape: {q_lower_limits.shape}, expected: (J,)"
    assert q_upper_limits.shape == (J,), f"q_upper_limits.shape: {q_upper_limits.shape}, expected: (J,)"
    assert 0.0 <= act_moving_average <= 1.0, f"act_moving_average: {act_moving_average}, expected: (0.0, 1.0)"

    # hand
    cur_targets = prev_targets.clone()
    cur_targets[:, 7 : 23] = scale(
        actions[:, 7 : 23],
        q_lower_limits[7 : 23],
        q_upper_limits[7 : 23],
    )
    cur_targets[:, 7 : 23] = (
        act_moving_average * cur_targets[:, 7 : 23]
        + (1.0 - act_moving_average) * prev_targets[:, 7 : 23]
    )
    cur_targets[:, 7 : 23] = tensor_clamp(
        cur_targets[:, 7 : 23],
        q_lower_limits[7 : 23],
        q_upper_limits[7 : 23],
    )

    # arm
    cur_targets[:, :7] = prev_targets[:, :7] + hand_dof_speed_scale * dt * actions[:, :7]
    cur_targets[:, :7] = tensor_clamp(
        cur_targets[:, :7],
        q_lower_limits[:7],
        q_upper_limits[:7],
    )
    return cur_targets

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