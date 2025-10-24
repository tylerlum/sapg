#!/usr/bin/env python
import time
from typing import Literal, Optional

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from termcolor import colored

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

DEFAULT_ARM_Q = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])

DEFAULT_HAND_Q = np.zeros(16)
DEFAULT_HAND_Q[12] = 0.3


def warn(message: str):
    print(colored(message, "yellow"))


def warn_every(message: str, n_seconds: float):
    first_time = not hasattr(warn_every, "last_warn_time")
    enough_time_has_passed = (
        hasattr(warn_every, "last_warn_time")
        and time.time() - warn_every.last_warn_time > n_seconds
    )
    if first_time or enough_time_has_passed:
        warn(message)
        warn_every.last_warn_time = time.time()


class FakeRobotNode:
    def __init__(self):
        # ROS setup
        rospy.init_node("fake_robot_ros_node")

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None

        # Publisher and subscriber
        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=10)
        self.allegro_pub = rospy.Publisher(
            "/allegroHand_0/joint_states", JointState, queue_size=10
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )

        # State
        self.iiwa_joint_q = DEFAULT_ARM_Q
        self.allegro_joint_q = DEFAULT_HAND_Q
        self.iiwa_joint_qd = np.zeros(NUM_ARM_JOINTS)
        self.allegro_joint_qd = np.zeros(NUM_HAND_JOINTS)

        # Set control rate to 60Hz
        self.rate_hz = 60
        self.dt = 1 / self.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

        # When only testing the arm, set this to False to ignore the Allegro hand
        self.WAIT_FOR_ALLEGRO_CMD = True
        if not self.WAIT_FOR_ALLEGRO_CMD:
            warn("NOT WAITING FOR ALLEGRO CMD")
            self.allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.allegro_joint_cmd = np.array(msg.position)

    def update_joint_states(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        if self.iiwa_joint_cmd is None or self.allegro_joint_cmd is None:
            warn_every(
                f"Waiting: iiwa_joint_cmd: {self.iiwa_joint_cmd}, allegro_joint_cmd: {self.allegro_joint_cmd}",
                n_seconds=1.0,
            )
            return

        delta_iiwa = self.iiwa_joint_cmd - self.iiwa_joint_q
        delta_allegro = self.allegro_joint_cmd - self.allegro_joint_q

        MODE: Literal["INTERPOLATE", "PD_CONTROL"] = "INTERPOLATE"
        if MODE == "INTERPOLATE":
            delta_iiwa_norm = np.linalg.norm(delta_iiwa)
            delta_allegro_norm = np.linalg.norm(delta_allegro)

            MAX_DELTA_IIWA = 0.1
            MAX_DELTA_ALLEGRO = 0.1
            if delta_iiwa_norm > MAX_DELTA_IIWA:
                delta_iiwa = MAX_DELTA_IIWA * delta_iiwa / delta_iiwa_norm
            if delta_allegro_norm > MAX_DELTA_ALLEGRO:
                delta_allegro = MAX_DELTA_ALLEGRO * delta_allegro / delta_allegro_norm

            self.iiwa_joint_q += delta_iiwa
            self.allegro_joint_q += delta_allegro
            self.iiwa_joint_qd = delta_iiwa / self.dt
            self.allegro_joint_qd = np.zeros(NUM_HAND_JOINTS)
        elif MODE == "PD_CONTROL":
            P = 10
            D = 0
            iiwa_qd_cmd = 0
            allegro_qd_cmd = 0
            delta_iiwa_qd = iiwa_qd_cmd - self.iiwa_joint_qd
            delta_allegro_qd = allegro_qd_cmd - self.allegro_joint_qd

            iiwa_qdd = P * delta_iiwa + D * delta_iiwa_qd
            allegro_qdd = P * delta_allegro + D * delta_allegro_qd
            self.iiwa_joint_qd += iiwa_qdd * self.dt
            self.allegro_joint_qd += allegro_qdd * self.dt
            self.iiwa_joint_q += self.iiwa_joint_qd * self.dt
            self.allegro_joint_q += self.allegro_joint_qd * self.dt
        else:
            raise ValueError(f"Invalid mode: {MODE}")

    def publish_joint_states(self):
        """Publish the current joint states from PyBullet."""
        iiwa_msg = JointState()
        iiwa_msg.header.stamp = rospy.Time.now()
        iiwa_msg.name = ["iiwa_joint_" + str(i) for i in range(NUM_ARM_JOINTS)]
        iiwa_msg.position = self.iiwa_joint_q.tolist()
        iiwa_msg.velocity = self.iiwa_joint_qd.tolist()
        self.iiwa_pub.publish(iiwa_msg)

        allegro_msg = JointState()
        allegro_msg.header.stamp = rospy.Time.now()
        allegro_msg.name = ["allegro_joint_" + str(i) for i in range(NUM_HAND_JOINTS)]
        allegro_msg.position = self.allegro_joint_q.tolist()
        allegro_msg.velocity = self.allegro_joint_qd.tolist()
        self.allegro_pub.publish(allegro_msg)

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        loop_no_sleep_dts, loop_dts = [], []
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the joint states
            self.update_joint_states()

            # Publish the current joint states to ROS
            self.publish_joint_states()

            # Sleep to maintain the loop rate
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


if __name__ == "__main__":
    try:
        # Create and run the FakeRobotNode
        node = FakeRobotNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
