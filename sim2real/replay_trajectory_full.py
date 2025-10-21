# For RecordedData import issue
import sys
sys.path.append("/home/tylerlum/github_repos/sapg")

import rospy
import copy
import numpy as np
import time
from sensor_msgs.msg import JointState
from recorded_data_scripts.recorded_data import RecordedData
from pathlib import Path



INIT_JOINT_POS_IIWA = None
INIT_JOINT_POS_ALLEGRO = None
CURRENT_JOINT_POS_IIWA = None
CURRENT_JOINT_POS_ALLEGRO = None

# HOME_JOINT_POS_IIWA = np.deg2rad(np.array([0, 0, 0, -90, 0, 90, 0]))

HOME_JOINT_POS_IIWA = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])

HOME_JOINT_POS_ALLEGRO = np.zeros(16)
HOME_JOINT_POS_ALLEGRO[12] = 0.3

# HOME_JOINT_POS_ALLEGRO = np.array([
#     -0.00, 1.37, 0.07, 0.00, 0.00, 1.5,
#      0.06, 0.00, 0.00, 1.51, 0.05, 0.0,
#      1.47, 0.00, 0.038, 0.0
# ])


def joint_state_callback_iiwa(msg: JointState) -> None:
    global INIT_JOINT_POS_IIWA
    if INIT_JOINT_POS_IIWA is not None:
        return
    INIT_JOINT_POS_IIWA = np.array(msg.position)
    rospy.loginfo(f"Initial IIWA joint positions: {INIT_JOINT_POS_IIWA}")


def current_joint_pos_iiwa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_IIWA
    CURRENT_JOINT_POS_IIWA = np.array(msg.position)

def current_joint_pos_allegro_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_ALLEGRO
    CURRENT_JOINT_POS_ALLEGRO = np.array(msg.position)

def interpolate_joint_pos(joint_pos1, joint_pos2, num_steps):
    interpolated_joint_pos = []
    for i in range(num_steps):
        interpolated_joint_pos.append(joint_pos1 + (joint_pos2 - joint_pos1) * (i+1)/num_steps)
    return np.array(interpolated_joint_pos)


def publish_joint_pos_targets(joint_pos_targets: np.ndarray, pub_iiwa: rospy.Publisher, pub_allegro: rospy.Publisher) -> None:
    assert joint_pos_targets.shape == (23,), f"joint_pos_targets.shape: {joint_pos_targets.shape}, expected: ({23},)"
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
    # print(f"Published joint positions: {iiwa_msg.position}, {allegro_msg.position}")


def move_to_pose(target_pos: np.ndarray, pub_iiwa: rospy.Publisher, pub_allegro: rospy.Publisher, move_time: float = 10.0) -> None:
    assert target_pos.shape == (23,), f"target_pos.shape: {target_pos.shape}, expected: ({23},)"
    current_allegro_pos = CURRENT_JOINT_POS_ALLEGRO.copy()
    current_iiwa_pos = CURRENT_JOINT_POS_IIWA.copy()
    current_pos = np.concatenate([current_iiwa_pos, current_allegro_pos])

    SECONDS_TO_MOVE = move_time
    CONTROL_HZ = 60
    interpolated_targets = interpolate_joint_pos(joint_pos1=current_pos, joint_pos2=target_pos, num_steps=int(CONTROL_HZ * SECONDS_TO_MOVE))
    for target_pos in interpolated_targets:
        start_time = rospy.Time.now()
        publish_joint_pos_targets(target_pos, pub_iiwa=pub_iiwa, pub_allegro=pub_allegro)
        end_time = rospy.Time.now()

        loop_without_sleep_dt = (end_time - start_time).to_sec()
        sleep_dt = 1/CONTROL_HZ - loop_without_sleep_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            print(f"Loop too slow! Desired FPS: {CONTROL_HZ}, Actual FPS: {1.0 / loop_without_sleep_dt:.1f}")
 

if __name__ == "__main__":
    file_path = Path(
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-43-04.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-42-41.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-30-37.npz"
        "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"
    )
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)
    joint_positions_array = recorded_data.robot_joint_positions_array
    joint_pos_targets_array = recorded_data.robot_joint_pos_targets_array
    joint_names = recorded_data.robot_joint_names
    T = joint_positions_array.shape[0]
    J = len(joint_names)
    assert joint_positions_array.shape == (T, J), f"joint_positions_array.shape: {joint_positions_array.shape}, expected: ({T}, {J})"
    assert joint_pos_targets_array.shape == (T, J), f"joint_pos_targets_array.shape: {joint_pos_targets_array.shape}, expected: ({T}, {J})"

    try:
        rospy.init_node("iiwa_allegro_joint_publisher", anonymous=True)

        sub_iiwa = rospy.Subscriber("/iiwa/joint_states", JointState, current_joint_pos_iiwa_callback)
        sub_allegro = rospy.Subscriber("/allegroHand_0/joint_states", JointState, current_joint_pos_allegro_callback)
        pub_iiwa = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
        pub_allegro = rospy.Publisher("/allegroHand_0/joint_cmd", JointState, queue_size=10)

        while CURRENT_JOINT_POS_IIWA is None or CURRENT_JOINT_POS_ALLEGRO is None:
            print("Waiting for CURRENT_JOINT_POS_IIWA and CURRENT_JOINT_POS_ALLEGRO")
            rospy.sleep(0.1)
            if rospy.is_shutdown():
                raise Exception("rospy shutdown")
        print("Got CURRENT_JOINT_POS_IIWA and CURRENT_JOINT_POS_ALLEGRO")
        
        # print(f"joint_positions_array[0]: {joint_positions_array[0]}")
        # print(f"joint_positions_array[0, :7]: {joint_positions_array[0, :7]}")
        # print(f"joint_positions_array[0, 7:]: {joint_positions_array[0, 7:]}")
        # print(f"CURRENT_JOINT_POS_IIWA: {CURRENT_JOINT_POS_IIWA}")
        # print(f"CURRENT_JOINT_POS_ALLEGRO: {CURRENT_JOINT_POS_ALLEGRO}")
        print("Moving to initial pose")
        # breakpoint()
        move_to_pose(joint_positions_array[0], pub_iiwa=pub_iiwa, pub_allegro=pub_allegro, move_time=10.0)
        # for timestep in range(300):
        #     print(f"timestep: {timestep}")
        #     # print(f"joint_positions_array[timestep]: {joint_positions_array[timestep]}")
        #     # print(f"joint_positions_array[timestep, :7]: {joint_positions_array[timestep, :7]}")
        #     # print(f"joint_positions_array[timestep, 7:]: {joint_positions_array[timestep, 7:]}")
        #     # print(f"CURRENT_JOINT_POS_IIWA: {CURRENT_JOINT_POS_IIWA}")
        #     # print(f"CURRENT_JOINT_POS_ALLEGRO: {CURRENT_JOINT_POS_ALLEGRO}")
        #     # move_to_pose(joint_positions_array[timestep], pub_iiwa=pub_iiwa, pub_allegro=pub_allegro, move_time=1.0)
        #     move_to_pose(joint_positions_array[timestep], pub_iiwa=pub_iiwa, pub_allegro=pub_allegro, move_time=0.2)

        # publish_joint_cmds(iiwa_init_joint_pos=iiwa_init, allegro_init_joint_pos=allegro_init, joint_pos_history=interpolated_joint_pos_history)
    except rospy.ROSInterruptException:
        pass 
