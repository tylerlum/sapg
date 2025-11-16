#!/usr/bin/env python

import copy
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pytorch_kinematics as pk
import rospy
import torch
from geometry_msgs.msg import Pose, PoseStamped
from rl_player import RlPlayer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from termcolor import colored

from isaacgymenvs.utils.observation_action_utils_sharpa import (
    compute_joint_pos_targets,
    compute_observation,
    Q_LOWER_LIMITS_restricted_np as Q_LOWER_LIMITS_np_between,
    Q_UPPER_LIMITS_restricted_np as Q_UPPER_LIMITS_np_between,
)

T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])

BETWEEN_JOINT_ORDER = [
    'iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7',
    'left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
    'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
    'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP',
    'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
    'left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
]

ADJUSTED_JOINT_ORDER = [
    'iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7',
    'left_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
    'left_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
    'left_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
    'left_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
    'left_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP',
]
def change_joint_order(
    q: np.ndarray,
    from_order: list[str],
    to_order: list[str],
) -> np.ndarray:
    J = len(from_order)
    assert len(to_order) == J, (
        f"Expected to_order to have the same length as from_order, got {len(to_order)} and {len(from_order)}"
    )
    assert q.shape == (J,), (
        f"Expected q to have length {J}, got {q.shape}"
    )

    assert set(to_order) == set(from_order), (
        f"Expected to_order to be the same as from_order, got to_order: {to_order} and from_order: {from_order}. Only in to_order: {set(to_order) - set(from_order)}"
    )

    # q is given in the from_order
    joint_name_to_value = {from_order[i]: q[i] for i in range(J)}
    new_q = np.array([joint_name_to_value[name] for name in to_order])

    assert new_q.shape == (len(to_order),), (
        f"Expected new_q to be {len(to_order)}, got {new_q.shape}"
    )
    return new_q



def adjusted_to_between(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q=q,
        from_order=ADJUSTED_JOINT_ORDER,
        to_order=BETWEEN_JOINT_ORDER,
    )


def between_to_adjusted(q: np.ndarray) -> np.ndarray:
    return change_joint_order(
        q=q,
        from_order=BETWEEN_JOINT_ORDER,
        to_order=ADJUSTED_JOINT_ORDER,
    )



def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float, key=None):
    """
    Print a warning message at most once every n_seconds per unique key.
    Stores state inside the function itself (no globals).
    """
    if not hasattr(warn_every, "_last_times"):
        warn_every._last_times = {}  # create on first call

    key = key or message
    last_times = warn_every._last_times
    last_time = last_times.get(key, 0)

    if time.time() - last_time > n_seconds:
        warn(message)
        last_times[key] = time.time()


def info(message: str):
    print(colored(message, "green"))


def assert_equals(a, b):
    assert a == b, f"a: {a}, b: {b}"


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


def var_to_is_none_str(var) -> str:
    if var is None:
        return "None"
    return "Not None"


def pose_msg_to_T(msg: Pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.array([msg.position.x, msg.position.y, msg.position.z])
    T[:3, :3] = R.from_quat(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    ).as_matrix()
    return T


def T_to_pos_quat_xyzw(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = T[:3, 3]
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return pos, quat_xyzw


class RLPolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("rl_policy_node_sharpa")

        # Publisher for iiwa and sharpa joint commands
        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.sharpa_joint_cmd_pub = rospy.Publisher(
            "/sharpa/joint_cmd", JointState, queue_size=10
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.sharpa_joint_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose", PoseStamped, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/robot_frame/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.sharpa_joint_state_sub = rospy.Subscriber(
            "/sharpa/joint_states", JointState, self.sharpa_joint_state_callback
        )

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 133  # Update this number based on actual dimensions
        self.num_actions = 29

        CONFIG_PATH = Path(
            # "/home/tylerlum/github_repos/sapg/closed_loop_testing/config.yaml"
            # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-05_hairbrush/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/runs/00_smooth-arm-hand_speed-10_dropout-obs_2025-11-05_05-20-24/config.yaml"
            "/home/tylerlum/github_repos/sapg/closed_loop_testing_sharpa/config.yaml"
        )
        assert Path(CONFIG_PATH).exists()
        CHECKPOINT_PATH = Path(
            # Fast
            # "/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/2025-11-12_sharpa_hammer_2_coacd/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/runs/00_CUBOID_obs-curriculum_thresh0-1_local_2025-11-14_00-04-24/last/model.pth"
            # Slow
            "/juno/u/kedia/sapg/train_dir/checkpoints/SLOW_CUBOID/model.pth"
        )
        assert CHECKPOINT_PATH.exists()

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            device=self.device,
        )

        # ROS rate
        self.rate_hz = 60
        self.dt = 1.0 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # Set up chain and palm_serial_chain
        asset_root = Path(__file__).parent / "../assets"
        urdf_path = asset_root / "urdf/kuka_allegro_description/iiwa14_left_sharpa_between.urdf"
        assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"
        self.chain = pk.build_chain_from_urdf(
            open(urdf_path).read(),
        ).to(device=self.device)
        self.palm_serial_chain = pk.SerialChain(self.chain, "iiwa14_link_7").to(
            device=self.device
        )

        # State: prev_targets
        self.prev_targets = None

    def object_pose_callback(self, msg: PoseStamped):
        self.object_pose_msg = msg.pose

    def goal_object_pose_callback(self, msg: Pose):
        self.goal_object_pose_msg = msg

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def sharpa_joint_state_callback(self, msg: JointState):
        self.sharpa_joint_state_msg = msg

    def create_observation(self) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.sharpa_joint_state_msg is None
            or self.object_pose_msg is None
            or self.goal_object_pose_msg is None
        ):
            warn_every(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, sharpa_joint_state_msg: {var_to_is_none_str(self.sharpa_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, goal_object_pose_msg: {var_to_is_none_str(self.goal_object_pose_msg)}",
                n_seconds=1.0,
            )
            return None, None

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        sharpa_joint_state_msg = copy.copy(self.sharpa_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)
        goal_object_pose_msg = copy.copy(self.goal_object_pose_msg)

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        sharpa_position = np.array(sharpa_joint_state_msg.position)
        sharpa_velocity = np.array(sharpa_joint_state_msg.velocity)

        T_R_O = pose_msg_to_T(object_pose_msg)
        T_R_G = pose_msg_to_T(goal_object_pose_msg)

        T_W_O = T_W_R @ T_R_O
        T_W_G = T_W_R @ T_R_G

        object_position_W, object_quat_xyzw_W = T_to_pos_quat_xyzw(T_W_O)
        object_pose_W = np.concatenate([object_position_W, object_quat_xyzw_W])

        goal_object_pos_W, goal_object_quat_xyzw_W = T_to_pos_quat_xyzw(T_W_G)
        goal_object_pose_W = np.concatenate(
            [goal_object_pos_W, goal_object_quat_xyzw_W]
        )

        q = np.concatenate([iiwa_position, sharpa_position])
        qd = np.concatenate([iiwa_velocity, sharpa_velocity])

        # HACK: Rearrange joint order
        q_between = adjusted_to_between(
            q=q,
        )
        qd_between = adjusted_to_between(
            q=qd,
        )

        observation = compute_observation(
            q=torch.from_numpy(q_between).float().to(self.device)[None],
            qd=torch.from_numpy(qd_between).float().to(self.device)[None],
            object_pose=torch.from_numpy(object_pose_W).float().to(self.device)[None],
            goal_object_pose=torch.from_numpy(goal_object_pose_W)
            .float()
            .to(self.device)[None],
            object_scales=torch.from_numpy(self.object_scales)
            .float()
            .to(self.device)[None],
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        assert_equals(
            observation.shape,
            (
                1,
                self.num_observations,
            ),
        )

        DEBUG = False
        if DEBUG:
            print(f"q: {q}")
            print(f"qd: {qd}")
            print(f"object_pose_W: {object_pose_W}")
            print(f"goal_object_pose_W: {goal_object_pose_W}")
            print(f"object_scales: {self.object_scales}")
            print(f"q_between: {q_between}")
            print(f"qd_between: {qd_between}")
            breakpoint()

        return observation, q_between

    def publish_targets(self, joint_pos_targets: torch.Tensor):
        assert_equals(joint_pos_targets.shape, (1, self.num_actions))
        joint_pos_targets = joint_pos_targets.squeeze(dim=0)
        joint_pos_targets = joint_pos_targets.cpu().numpy()

        q_adjusted = between_to_adjusted(
            q=joint_pos_targets,
        )

        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.header.frame_id = ""
        iiwa_msg.name = [
            "iiwa_joint_1",
            "iiwa_joint_2",
            "iiwa_joint_3",
            "iiwa_joint_4",
            "iiwa_joint_5",
            "iiwa_joint_6",
            "iiwa_joint_7",
        ]
        iiwa_msg.position = q_adjusted[:7].tolist()
        self.iiwa_joint_cmd_pub.publish(iiwa_msg)
        sharpa_msg = JointState()
        sharpa_msg.header.stamp = rospy.Time.now()
        sharpa_msg.header.frame_id = ""
        sharpa_msg.name = [
            "joint_0.0",
            "joint_1.0",
            "joint_2.0",
            "joint_3.0",
            "joint_4.0",
            "joint_5.0",
            "joint_6.0",
            "joint_7.0",
            "joint_8.0",
            "joint_9.0",
            "joint_10.0",
            "joint_11.0",
            "joint_12.0",
            "joint_13.0",
            "joint_14.0",
            "joint_15.0",
            "joint_16.0",
            "joint_17.0",
            "joint_18.0",
            "joint_19.0",
            "joint_20.0",
            "joint_21.0",
        ]
        sharpa_msg.position = q_adjusted[7:].tolist()
        self.sharpa_joint_cmd_pub.publish(sharpa_msg)

    def run(self):
        first_observations_received = False

        loop_no_sleep_dts, loop_dts = [], []

        # CURRENT_STEP = 0
        while not rospy.is_shutdown():
            # print(f"Current step: {CURRENT_STEP}")
            # if CURRENT_STEP > 1500:
            #     print("Exiting")
            #     import sys

            #     sys.exit(0)
            # CURRENT_STEP += 1

            start_time = rospy.Time.now()

            # Create observation from the latest messages
            obs, q_between = self.create_observation()

            if obs is not None and q_between is not None:
                if not first_observations_received:
                    info("=" * 100)
                    info("First observations received, starting to publish sim state")
                    info("=" * 100)
                    first_observations_received = True

                if self.prev_targets is None:
                    self.prev_targets = torch.from_numpy(q_between).float().to(self.device)[None]

                assert_equals(obs.shape, (1, self.num_observations))

                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(
                    obs=obs,
                    deterministic_actions=True,
                    # obs=obs, deterministic_actions=True
                )
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                HAND_MOVING_AVERAGE = 0.1
                ARM_MOVING_AVERAGE = 0.01
                HAND_DOF_SPEED_SCALE = 5.0
                DT = 1 / 60
                joint_pos_targets = compute_joint_pos_targets(
                    actions=normalized_action,
                    prev_targets=self.prev_targets,
                    hand_moving_average=HAND_MOVING_AVERAGE,
                    arm_moving_average=ARM_MOVING_AVERAGE,
                    hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
                    dt=DT,
                )
                assert_equals(joint_pos_targets.shape, (1, self.num_actions))

                # Clamp
                joint_pos_targets = torch.clip(
                    joint_pos_targets,
                    min=torch.from_numpy(Q_LOWER_LIMITS_np_between).float().to(self.device)[None],
                    max=torch.from_numpy(Q_UPPER_LIMITS_np_between).float().to(self.device)[None],
                )

                # Publish the targets
                self.publish_targets(joint_pos_targets)
                # print(f"CURRENT_STEP: {CURRENT_STEP}")
                # print(f"joint_pos_targets: {joint_pos_targets}")
                # print()
                self.prev_targets = joint_pos_targets.clone()

            # Sleep to maintain loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            loop_no_sleep_dt = (before_sleep_time - start_time).to_sec()
            loop_no_sleep_dts.append(loop_no_sleep_dt)
            loop_dt = (after_sleep_time - start_time).to_sec()
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
            # if True:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []

    @property
    def object_scales(self) -> np.ndarray:
        # object_scales = np.array([0.1, 0.035, 0.025]) * 20
        # object_scales = np.array([3.0, 0.5, 0.5])
        object_scales = np.array([4.0, 0.75, 1.0])
        # object_scales = np.array([4.0, 0.75, 1.0]) * 1.25
        return object_scales


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
