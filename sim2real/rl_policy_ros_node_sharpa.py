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
    create_chain_and_serial_chain,
    Q_LOWER_LIMITS_restricted_np as Q_LOWER_LIMITS_np,
    Q_UPPER_LIMITS_restricted_np as Q_UPPER_LIMITS_np,
)

SAVE_INPUTS_TO_FILE = True


T_W_R = np.eye(4)
T_W_R[:3, 3] = np.array([0.0, 0.8, 0.0])

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
    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        hand_moving_average: float,
        arm_moving_average: float,
        overwrite_targets_filepath: Optional[Path] = None,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.overwrite_targets_filepath = overwrite_targets_filepath

        # Initialize the ROS node
        rospy.init_node("rl_policy_node_sharpa")

        if SAVE_INPUTS_TO_FILE:
            # ##############################################################################
            # Signal handling to save on shutdown
            # When in progress saving to file, stop updating latest joint states and commands
            # ##############################################################################
            import signal
            # Signal handling to save on shutdown
            # When in progress saving to file, stop updating latest joint states and commands
            self._is_in_progress_saving_to_file = False
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Store history of joint states and commands
            self.time_history: list[float] = []
            self.q_history: list[np.ndarray] = []
            self.qd_history: list[np.ndarray] = []
            self.q_target_history: list[np.ndarray] = []
            self.object_pose_history: list[np.ndarray] = []
            self.goal_object_pose_history: list[np.ndarray] = []

        # Publisher for iiwa and sharpa joint commands
        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=1
        )
        self.sharpa_joint_cmd_pub = rospy.Publisher(
            "/sharpa/joint_cmd", JointState, queue_size=1
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.sharpa_joint_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/robot_frame/current_object_pose", PoseStamped, self.object_pose_callback,
            queue_size=1
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/robot_frame/goal_object_pose", Pose, self.goal_object_pose_callback,
            queue_size=1
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback,
            queue_size=1
        )
        self.sharpa_joint_state_sub = rospy.Subscriber(
            "/sharpa/joint_states", JointState, self.sharpa_joint_state_callback,
            queue_size=1
        )

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 140  # Update this number based on actual dimensions
        self.num_actions = 29

        assert self.config_path.exists(), f"config_path: {self.config_path} does not exist"
        assert self.checkpoint_path.exists(), f"checkpoint_path: {self.checkpoint_path} does not exist"

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=str(self.config_path),
            checkpoint_path=str(self.checkpoint_path),
            device=self.device,
        )
        self.obs_list = self.player.cfg["task"]["env"]["obsList"]

        # ROS rate
        self.control_dt = 1.0 / 60

        # Set up chain
        robot_name = "iiwa14_left_sharpa_adjusted_restricted"
        self.chain, _ = create_chain_and_serial_chain(
            device=self.device, robot_name=robot_name
        )

        # State: prev_targets
        self.prev_targets = None

        if self.overwrite_targets_filepath is not None:
            info(f"Overwriting targets from file: {self.overwrite_targets_filepath}")
            from recorded_data_scripts.recorded_data_sharpa import RecordedData
            data_path = self.overwrite_targets_filepath
            assert data_path.exists(), f"File {data_path} does not exist"
            data = RecordedData.from_file(data_path)
            self.q_targets_from_file = torch.from_numpy(data.robot_joint_pos_targets_array).float().to(self.device)
            T, D = self.q_targets_from_file.shape
            print(f"T: {T}, D: {D}")
            assert D == 29, f"D: {D}, expected: 29"
            self.current_step = 0

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

        prev_action_targets = self.prev_targets if self.prev_targets is not None else torch.from_numpy(q).float().to(self.device)[None]
        observation = compute_observation(
            q=torch.from_numpy(q).float().to(self.device)[None],
            qd=torch.from_numpy(qd).float().to(self.device)[None],
            prev_action_targets=prev_action_targets,
            object_pose=torch.from_numpy(object_pose_W).float().to(self.device)[None],
            goal_object_pose=torch.from_numpy(goal_object_pose_W)
            .float()
            .to(self.device)[None],
            object_scales=torch.from_numpy(self.object_scales)
            .float()
            .to(self.device)[None],
            chain=self.chain,
            obs_list=self.obs_list,
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
            breakpoint()

        # ##############################################################################
        # Record time and joint states and commands and object pose and goal object pose
        # ##############################################################################
        if SAVE_INPUTS_TO_FILE:
            if not hasattr(self, "start_run_time"):
                self.start_run_time = time.time()
            current_time = time.time()
            dt = current_time - self.start_run_time

            self.time_history.append(dt)
            self.q_history.append(q)
            self.qd_history.append(qd)
            self.q_target_history.append(prev_action_targets.cpu().numpy()[0])
            self.object_pose_history.append(object_pose_W)
            self.goal_object_pose_history.append(goal_object_pose_W)

        return observation, q

    def publish_targets(self, joint_pos_targets: torch.Tensor):
        assert_equals(joint_pos_targets.shape, (1, self.num_actions))
        joint_pos_targets = joint_pos_targets.squeeze(dim=0)
        joint_pos_targets = joint_pos_targets.cpu().numpy()

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
        iiwa_msg.position = joint_pos_targets[:7].tolist()
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
        sharpa_msg.position = joint_pos_targets[7:].tolist()
        self.sharpa_joint_cmd_pub.publish(sharpa_msg)

    def _wait_and_warmup(self):
        # Wait
        while not rospy.is_shutdown():
            obs, q = self.create_observation()
            if obs is not None and q is not None:
                break
            time.sleep(self.control_dt)

        # Done waiting
        info("=" * 100)
        info("First observations received, starting to publish sim state")
        info("=" * 100)

        self.prev_targets = torch.from_numpy(q).float().to(self.device)[None]

        # Warm up the policy and publishing
        info("=" * 100)
        info("Warming up policy and publishing current targets")
        info("=" * 100)
        # THIS IS NOT THE REAL LOOP, DON'T CARE ABOUT THESE NUMBERs
        num_steps = 0
        NUM_WARMUP_STEPS = 100
        while not rospy.is_shutdown():
            num_steps += 1
            info(f"Warmup step {num_steps} of {NUM_WARMUP_STEPS}")
            if num_steps > NUM_WARMUP_STEPS:
                info(f"Reached {NUM_WARMUP_STEPS} steps, stopping warmup")
                break

            # Create observation from the latest messages
            obs, q = self.create_observation()
            assert obs is not None and q is not None, f"obs: {obs}, q: {q}"
            assert_equals(obs.shape, (1, self.num_observations))

            # Get the normalized action from the RL player
            normalized_action = self.player.get_normalized_action(
                obs=obs,
                deterministic_actions=True,
            )
            # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
            assert_equals(normalized_action.shape, (1, self.num_actions))

            _ = compute_joint_pos_targets(
                actions=normalized_action,
                prev_targets=self.prev_targets,
                hand_moving_average=0.1,
                arm_moving_average=0.1,
                hand_dof_speed_scale=2.5,
                dt=1/60,
            )

            # We do not actually use the joint pos targets computed by the policy, we use the actual joint states so it doesn't move
            joint_pos_targets = torch.clip(
                torch.from_numpy(q).float().to(self.device)[None],
                min=torch.from_numpy(Q_LOWER_LIMITS_np).float().to(self.device)[None],
                max=torch.from_numpy(Q_UPPER_LIMITS_np).float().to(self.device)[None],
            )

            # Publish the targets
            self.publish_targets(joint_pos_targets)
            self.prev_targets = joint_pos_targets.clone()
            time.sleep(self.control_dt)

        # Reset rnn state
        self.player.reset()

        # Done warming up
        info("=" * 100)
        info("Warmup complete")
        info("=" * 100)

    def run(self):
        self._wait_and_warmup()

        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_loop_no_sleep_time = time.time()

            # Create observation from the latest messages
            obs, q = self.create_observation()
            assert obs is not None and q is not None, f"obs: {obs}, q: {q}"

            assert_equals(obs.shape, (1, self.num_observations))

            # Get the normalized action from the RL player
            normalized_action = self.player.get_normalized_action(
                obs=obs,
                deterministic_actions=True,
            )
            assert_equals(normalized_action.shape, (1, self.num_actions))

            HAND_DOF_SPEED_SCALE = 2.5
            DT = 1 / 60
            joint_pos_targets = compute_joint_pos_targets(
                actions=normalized_action,
                prev_targets=self.prev_targets,
                hand_moving_average=self.hand_moving_average,
                arm_moving_average=self.arm_moving_average,
                hand_dof_speed_scale=HAND_DOF_SPEED_SCALE,
                dt=DT,
            )
            assert_equals(joint_pos_targets.shape, (1, self.num_actions))

            # Clamp
            joint_pos_targets = torch.clip(
                joint_pos_targets,
                min=torch.from_numpy(Q_LOWER_LIMITS_np).float().to(self.device)[None],
                max=torch.from_numpy(Q_UPPER_LIMITS_np).float().to(self.device)[None],
            )

            if self.overwrite_targets_filepath is not None:
                if self.current_step >= self.q_targets_from_file.shape[0]:
                    self.current_step = self.q_targets_from_file.shape[0] - 1
                    info("Reached end of targets, holding last target")
                assert self.current_step < self.q_targets_from_file.shape[0], f"current_step: {self.current_step}, expected: < {self.q_targets_from_file.shape[0]}"
                joint_pos_targets = self.q_targets_from_file[self.current_step].unsqueeze(0)
                self.current_step += 1

            # Publish the targets
            self.publish_targets(joint_pos_targets)
            self.prev_targets = joint_pos_targets.clone()

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.control_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                warn(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.control_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(PRINT_FPS_EVERY_N_SECONDS / self.control_dt)
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
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
        # Hammer 2
        # object_scales = np.array([3.0, 0.25, 0.2])

        # blue_cuboid (rearrange)
        # object_scales = np.array([4.0, 1.0, 0.75])

        # blue_cuboid (not rearrange) and different scale
        object_scales = np.array([5.0, 0.9375, 1.25])

        # blue_cuboid_real_iphone
        # object_scales = np.array([3.0, 1.4, 0.2])

        # # blue_cuboid_fake_iphone
        # object_scales = np.array([2.0, 1.25, 0.5])

        # # blue_cuboid_real_hammer
        # object_scales = np.array([2.0, 0.55, 0.35])

        # # blue_cuboid_fake_hammer
        # object_scales = np.array([2.5, 0.75, 0.65])

        # # blue_cuboid_real_screwdriver
        # object_scales = np.array([1.3, 0.7, 0.5])

        return object_scales

    def _signal_handler(self, signum, frame):
        assert SAVE_INPUTS_TO_FILE, "SAVE_INPUTS_TO_FILE must be True to save to file"

        import datetime
        if self._is_in_progress_saving_to_file:
            warn("Already in progress of saving to file, skipping")
            return

        self._is_in_progress_saving_to_file = True
        if len(self.time_history) == 0:
            warn("No data recorded, skipping")
        else:
            info(f"Received signal {signum}, saving to file")
            datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # filename = datetime_str
            filename = f"{datetime_str}_{self.checkpoint_path.stem}_arm{self.arm_moving_average}"
            output_path = Path("recorded_robot_inputs") / "isaac" /f"{filename}.npz"
            self.save_to_file(output_path)
            info(f"Saved to file: {output_path}")

        rospy.signal_shutdown("Shutting down")

    def save_to_file(self, file_path: Path):
        assert SAVE_INPUTS_TO_FILE, "SAVE_INPUTS_TO_FILE must be True to save to file"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        info(f"Saving to file: {file_path}")

        T = len(self.time_history)
        robot_root_states_array = np.zeros((T, 13))
        robot_root_states_array[:, 1] = 0.8
        robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
        object_root_states_array = np.zeros((T, 13))
        object_root_states_array[:, :7] = np.array(self.object_pose_history)
        table_root_states_array = np.zeros((T, 13))
        table_root_states_array[:, :3] = np.array([0.0, 0.0, 0.38])[None]
        goal_root_states_array = np.zeros((T, 13))
        goal_root_states_array[:, :7] = np.array(self.goal_object_pose_history)

        robot_joint_positions = np.array(self.q_history)
        robot_joint_velocities = np.array(self.qd_history)

        robot_joint_pos_targets = np.array(self.q_target_history)
        time_array = np.array(self.time_history)

        assert robot_joint_positions.shape == (T, 29), (
            f"robot_joint_positions.shape: {robot_joint_positions.shape}, expected: (T, 29)"
        )
        assert robot_joint_velocities.shape == (T, 29), (
            f"robot_joint_velocities.shape: {robot_joint_velocities.shape}, expected: (T, 29)"
        )
        assert robot_joint_pos_targets.shape == (T, 29), (
            f"robot_joint_pos_targets.shape: {robot_joint_pos_targets.shape}, expected: (T, 29)"
        )
        assert object_root_states_array.shape == (T, 13), (
            f"object_root_states_array.shape: {object_root_states_array.shape}, expected: (T, 13)"
        )
        assert time_array.shape == (T,), (
            f"time_array.shape: {time_array.shape}, expected: (T,)"
        )

        JOINT_NAMES = [
            'iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7',
            'left_1_thumb_CMC_FE', 'left_thumb_CMC_AA', 'left_thumb_MCP_FE', 'left_thumb_MCP_AA', 'left_thumb_IP',
            'left_2_index_MCP_FE', 'left_index_MCP_AA', 'left_index_PIP', 'left_index_DIP',
            'left_3_middle_MCP_FE', 'left_middle_MCP_AA', 'left_middle_PIP', 'left_middle_DIP',
            'left_4_ring_MCP_FE', 'left_ring_MCP_AA', 'left_ring_PIP', 'left_ring_DIP',
            'left_5_pinky_CMC', 'left_pinky_MCP_FE', 'left_pinky_MCP_AA', 'left_pinky_PIP', 'left_pinky_DIP',
        ]

        from recorded_data_scripts.recorded_data_sharpa import RecordedData

        recorded_data = RecordedData(
            robot_root_states_array=robot_root_states_array,
            object_root_states_array=object_root_states_array,
            robot_joint_positions_array=robot_joint_positions,
            time_array=time_array,
            robot_joint_names=JOINT_NAMES,
            robot_joint_velocities_array=robot_joint_velocities,
            robot_joint_pos_targets_array=robot_joint_pos_targets,
            goal_root_states_array=goal_root_states_array,
        )
        recorded_data.to_file(file_path)


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode(
            config_path=Path("/juno/u/kedia/sapg/train_dir/checkpoints/asymmetric/newGains_2.5speed/config.yaml"),
            # checkpoint_path=Path("/juno/u/kedia/sapg/train_dir/checkpoints/2025-12-11_newGains/cleanInputs.pth"),
            checkpoint_path=Path("/juno/u/kedia/sapg/train_dir/checkpoints/2025-12-11_newGains/noisyInputs.pth"),
            hand_moving_average=0.1,
            arm_moving_average=0.05,
            overwrite_targets_filepath=None,
            # overwrite_targets_filepath=Path("recorded_robot_inputs/isaac/2025-12-12_19-34-25_noisyInputs_arm0.05.npz"),
        )
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
