import copy
import sys
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState

# Global variables for current joint positions
CURRENT_JOINT_POS_IIWA = None
CURRENT_JOINT_POS_ALLEGRO = None

# Home joint positions
HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])
HOME_JOINT_POS_ALLEGRO = np.zeros(16)
HOME_JOINT_POS_ALLEGRO[12] = 0.3
HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_ALLEGRO])


def current_joint_pos_iiwa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_IIWA
    CURRENT_JOINT_POS_IIWA = np.array(msg.position)


def current_joint_pos_allegro_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_ALLEGRO
    CURRENT_JOINT_POS_ALLEGRO = np.array(msg.position)


def interpolate_joint_pos(
    init_joint_pos: np.ndarray, final_joint_pos: np.ndarray, num_steps: int
) -> np.ndarray:
    assert init_joint_pos.shape == final_joint_pos.shape, (
        f"init_joint_pos.shape: {init_joint_pos.shape}, final_joint_pos.shape: {final_joint_pos.shape}"
    )
    joint_positions_list = []
    for i in range(num_steps):
        joint_positions_list.append(
            init_joint_pos + (final_joint_pos - init_joint_pos) * (i + 1) / num_steps
        )
    joint_positions_array = np.array(joint_positions_list)
    assert joint_positions_array.shape == (num_steps, init_joint_pos.shape[0]), (
        f"joint_positions_array.shape: {joint_positions_array.shape}, expected: ({num_steps}, {init_joint_pos.shape[0]})"
    )
    return np.array(joint_positions_list)


def publish_joint_pos_targets(
    joint_pos_targets: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_allegro: rospy.Publisher,
) -> None:
    assert joint_pos_targets.shape == (23,), (
        f"joint_pos_targets.shape: {joint_pos_targets.shape}, expected: ({23},)"
    )
    iiwa_joint_pos = joint_pos_targets[:7]
    allegro_joint_pos = joint_pos_targets[7:]

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
    allegro_msg = JointState()
    allegro_msg.header.stamp = rospy.Time.now()
    allegro_msg.header.frame_id = ""
    allegro_msg.name = [
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
    ]

    iiwa_msg.position = copy.deepcopy(iiwa_joint_pos.tolist())
    allegro_msg.position = copy.deepcopy(allegro_joint_pos.tolist())
    pub_iiwa.publish(iiwa_msg)
    pub_allegro.publish(allegro_msg)


def move_to_pose(
    target_pos: np.ndarray,
    pub_iiwa: rospy.Publisher,
    pub_allegro: rospy.Publisher,
    move_time: float = 10.0,
    control_hz: int = 60,
) -> None:
    assert target_pos.shape == (23,), (
        f"target_pos.shape: {target_pos.shape}, expected: ({23},)"
    )
    current_allegro_pos = CURRENT_JOINT_POS_ALLEGRO.copy()
    current_iiwa_pos = CURRENT_JOINT_POS_IIWA.copy()
    current_pos = np.concatenate([current_iiwa_pos, current_allegro_pos])

    SECONDS_TO_MOVE = move_time
    CONTROL_HZ = control_hz
    interpolated_targets = interpolate_joint_pos(
        init_joint_pos=current_pos,
        final_joint_pos=target_pos,
        num_steps=int(CONTROL_HZ * SECONDS_TO_MOVE),
    )
    for target_pos in interpolated_targets:
        if rospy.is_shutdown():
            print("ROS shutdown, exiting")
            sys.exit(0)

        start_time = rospy.Time.now()
        publish_joint_pos_targets(
            target_pos, pub_iiwa=pub_iiwa, pub_allegro=pub_allegro
        )
        end_time = rospy.Time.now()

        loop_without_sleep_dt = (end_time - start_time).to_sec()
        sleep_dt = 1 / CONTROL_HZ - loop_without_sleep_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            print(
                f"Loop too slow! Desired FPS: {CONTROL_HZ}, Actual FPS: {1.0 / loop_without_sleep_dt:.1f}"
            )


def main():
    # Initialize ROS node
    rospy.init_node("iiwa_allegro_joint_publisher", anonymous=True)

    # Create subscribers and publishers
    _sub_iiwa = rospy.Subscriber(
        "/iiwa/joint_states", JointState, current_joint_pos_iiwa_callback
    )
    _sub_allegro = rospy.Subscriber(
        "/allegroHand_0/joint_states",
        JointState,
        current_joint_pos_allegro_callback,
    )
    pub_iiwa = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
    pub_allegro = rospy.Publisher("/allegroHand_0/joint_cmd", JointState, queue_size=10)

    # Wait for current joint positions to be available
    while not rospy.is_shutdown():
        if CURRENT_JOINT_POS_IIWA is None or CURRENT_JOINT_POS_ALLEGRO is None:
            print(
                f"Waiting: CURRENT_JOINT_POS_IIWA = {CURRENT_JOINT_POS_IIWA}, CURRENT_JOINT_POS_ALLEGRO = {CURRENT_JOINT_POS_ALLEGRO}"
            )
            rospy.sleep(0.1)
        else:
            print("=" * 100)
            print("Got CURRENT_JOINT_POS_IIWA and CURRENT_JOINT_POS_ALLEGRO")
            print("=" * 100)
            break

    # Move to home pose
    print("Moving to home pose")
    move_to_pose(
        HOME_JOINT_POS,
        pub_iiwa=pub_iiwa,
        pub_allegro=pub_allegro,
        move_time=10.0,
    )
    print("Reached home pose")


if __name__ == "__main__":
    main()
