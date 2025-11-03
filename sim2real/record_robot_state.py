import copy
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

from recorded_data_scripts.recorded_data import RecordedData


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


JOINT_NAMES = [
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


class RecordRobotState:
    def __init__(self):
        rospy.init_node("record_robot_state")

        # Store latest joint states and commands
        self.latest_iiwa_joint_state: Optional[JointState] = None
        self.latest_allegro_joint_state: Optional[JointState] = None
        self.latest_iiwa_joint_cmd: Optional[JointState] = None
        self.latest_allegro_joint_cmd: Optional[JointState] = None

        # Store history of joint states and commands
        self.time_history: list[float] = []
        self.iiwa_joint_position_history: list[np.ndarray] = []
        self.allegro_joint_position_history: list[np.ndarray] = []
        self.iiwa_joint_velocity_history: list[np.ndarray] = []
        self.allegro_joint_velocity_history: list[np.ndarray] = []
        self.iiwa_joint_pos_target_history: list[np.ndarray] = []
        self.allegro_joint_pos_target_history: list[np.ndarray] = []

        # Subscribers
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self._iiwa_joint_state_callback
        )
        self.allegro_joint_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states",
            JointState,
            self._allegro_joint_state_callback,
        )
        self.iiwa_joint_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self._iiwa_joint_cmd_callback
        )
        self.allegro_joint_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self._allegro_joint_cmd_callback
        )

        # ROS rate
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def _iiwa_joint_state_callback(self, msg: JointState):
        self.latest_iiwa_joint_state = msg

    def _allegro_joint_state_callback(self, msg: JointState):
        self.latest_allegro_joint_state = msg

    def _iiwa_joint_cmd_callback(self, msg: JointState):
        self.latest_iiwa_joint_cmd = msg

    def _allegro_joint_cmd_callback(self, msg: JointState):
        self.latest_allegro_joint_cmd = msg

    def run(self):
        while not rospy.is_shutdown():
            if (
                self.latest_iiwa_joint_state is None
                or self.latest_allegro_joint_state is None
                or self.latest_iiwa_joint_cmd is None
                or self.latest_allegro_joint_cmd is None
            ):
                warn_every(
                    f"Waiting: latest_iiwa_joint_state = {self.latest_iiwa_joint_state}, latest_allegro_joint_state = {self.latest_allegro_joint_state}, latest_iiwa_joint_cmd = {self.latest_iiwa_joint_cmd}, latest_allegro_joint_cmd = {self.latest_allegro_joint_cmd}",
                    n_seconds=1.0,
                )
                self.rate.sleep()
                continue

        info("All messages received, starting to record robot state")

        start_time = time.time()
        while not rospy.is_shutdown():
            # Record time
            current_time = time.time()
            dt = current_time - start_time

            # Create copy of latest joint states and commands
            iiwa_joint_state = copy.copy(self.latest_iiwa_joint_state)
            allegro_joint_state = copy.copy(self.latest_allegro_joint_state)
            iiwa_joint_cmd = copy.copy(self.latest_iiwa_joint_cmd)
            allegro_joint_cmd = copy.copy(self.latest_allegro_joint_cmd)

            # Convert to numpy arrays
            iiwa_joint_position = np.array(iiwa_joint_state.position)
            iiwa_joint_velocity = np.array(iiwa_joint_state.velocity)
            allegro_joint_position = np.array(allegro_joint_state.position)
            allegro_joint_velocity = np.array(allegro_joint_state.velocity)
            iiwa_joint_cmd_position = np.array(iiwa_joint_cmd.position)
            allegro_joint_cmd_position = np.array(allegro_joint_cmd.position)

            # Store
            self.iiwa_joint_position_history.append(iiwa_joint_position)
            self.allegro_joint_position_history.append(allegro_joint_position)
            self.iiwa_joint_velocity_history.append(iiwa_joint_velocity)
            self.allegro_joint_velocity_history.append(allegro_joint_velocity)
            self.iiwa_joint_pos_target_history.append(iiwa_joint_cmd_position)
            self.allegro_joint_pos_target_history.append(allegro_joint_cmd_position)
            self.time_history.append(dt)

            self.rate.sleep()

    def save_to_file(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to file: {file_path}")

        T = len(self.time_history)
        robot_root_states_array = np.zeros((T, 13))
        robot_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1
        object_root_states_array = np.zeros((T, 13))
        object_root_states_array[:, 6] = 1.0  # quaternion xyzw has w=1

        iiwa_joint_positions = np.array(self.iiwa_joint_position_history)
        allegro_joint_positions = np.array(self.allegro_joint_position_history)
        robot_joint_positions = np.concatenate(
            [iiwa_joint_positions, allegro_joint_positions], axis=1
        )

        iiwa_joint_velocities = np.array(self.iiwa_joint_velocity_history)
        allegro_joint_velocities = np.array(self.allegro_joint_velocity_history)
        robot_joint_velocities = np.concatenate(
            [iiwa_joint_velocities, allegro_joint_velocities], axis=1
        )

        iiwa_joint_pos_targets = np.array(self.iiwa_joint_pos_target_history)
        allegro_joint_pos_targets = np.array(self.allegro_joint_pos_target_history)
        robot_joint_pos_targets = np.concatenate(
            [iiwa_joint_pos_targets, allegro_joint_pos_targets], axis=1
        )

        time_array = np.array(self.time_history)

        assert robot_joint_positions.shape == (T, 23), (
            f"robot_joint_positions.shape: {robot_joint_positions.shape}, expected: (T, 23)"
        )
        assert robot_joint_velocities.shape == (T, 23), (
            f"robot_joint_velocities.shape: {robot_joint_velocities.shape}, expected: (T, 23)"
        )
        assert robot_joint_pos_targets.shape == (T, 23), (
            f"robot_joint_pos_targets.shape: {robot_joint_pos_targets.shape}, expected: (T, 23)"
        )
        assert time_array.shape == (T,), (
            f"time_array.shape: {time_array.shape}, expected: (T,)"
        )

        recorded_data = RecordedData(
            robot_root_states_array=robot_root_states_array,
            object_root_states_array=object_root_states_array,
            robot_joint_positions_array=robot_joint_positions,
            time_array=np.array(self.time_history),
            robot_joint_names=JOINT_NAMES,
            robot_joint_velocities_array=robot_joint_velocities,
            robot_joint_pos_targets_array=robot_joint_pos_targets,
        )
        recorded_data.to_file(file_path)


def main():
    pass


if __name__ == "__main__":
    main()
