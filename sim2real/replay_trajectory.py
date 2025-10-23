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


def joint_state_callback_allegro(msg: JointState) -> None:
    global INIT_JOINT_POS_ALLEGRO
    if INIT_JOINT_POS_ALLEGRO is not None:
        return
    INIT_JOINT_POS_ALLEGRO = np.array(msg.position)
    rospy.loginfo(f"Initial Allegro joint positions: {INIT_JOINT_POS_ALLEGRO}")

def current_joint_pos_iiwa_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_IIWA
    CURRENT_JOINT_POS_IIWA = np.array(msg.position)

def current_joint_pos_allegro_callback(msg: JointState) -> None:
    global CURRENT_JOINT_POS_ALLEGRO
    CURRENT_JOINT_POS_ALLEGRO = np.array(msg.position)

def get_initial_joint_pos_iiwa() -> np.ndarray:
    sub = rospy.Subscriber("/iiwa/joint_states", JointState, joint_state_callback_iiwa)
    while INIT_JOINT_POS_IIWA is None:
        rospy.loginfo("Waiting for INIT_JOINT_POS_IIWA")
        rospy.sleep(0.1)
        if rospy.is_shutdown():
            raise Exception("rospy shutdown")
    rospy.loginfo("Got INIT_JOINT_POS_IIWA")
    return INIT_JOINT_POS_IIWA


def get_initial_joint_pos_allegro() -> np.ndarray:
    sub = rospy.Subscriber("/allegroHand_0/joint_states", JointState, joint_state_callback_allegro)
    while INIT_JOINT_POS_ALLEGRO is None:
        rospy.loginfo("Waiting for INIT_JOINT_POS_ALLEGRO")
        rospy.sleep(0.1)
        if rospy.is_shutdown():
            raise Exception("rospy shutdown")
    rospy.loginfo("Got INIT_JOINT_POS_ALLEGRO")
    return INIT_JOINT_POS_ALLEGRO

def publish_joint_cmds(iiwa_init_joint_pos: np.ndarray, allegro_init_joint_pos: np.ndarray, joint_pos_history: np.ndarray) -> None:
    pub_iiwa = rospy.Publisher("/iiwa/joint_cmd", JointState, queue_size=10)
    pub_allegro = rospy.Publisher("/allegroHand_0/joint_cmd", JointState, queue_size=10)
    sub_iiwa = rospy.Subscriber("/iiwa/joint_states", JointState, current_joint_pos_iiwa_callback)
    sub_allegro = rospy.Subscriber("/allegroHand_0/joint_states", JointState, current_joint_pos_allegro_callback)
    rate = rospy.Rate(60)

    # IIWA message setup
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

    # Allegro message setup
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

    assert (
        len(iiwa_init_joint_pos) == 7
    ), f"Initial IIWA joint state must have 7 elements, has {len(iiwa_init_joint_pos)}"
    assert (
        len(allegro_init_joint_pos) == 16
    ), f"Initial Allegro joint state must have 16 elements, has {len(allegro_init_joint_pos)}"

    iiwa_msg.position = copy.deepcopy(iiwa_init_joint_pos.tolist())
    iiwa_msg.velocity = [0.0] * 7
    iiwa_msg.effort = []

    allegro_msg.position = copy.deepcopy(allegro_init_joint_pos.tolist())
    allegro_msg.velocity = [0.0] * 16
    allegro_msg.effort = []

    start_time = rospy.Time.now()
    DURATION = 4
    STATIONARY_TIME = 2
    last_publish_time = rospy.Time.now()

    current_idx = 0
    interpolation_alpha = 0
    step_size = 0.0001
    while not rospy.is_shutdown():
        elapsed_time = (rospy.Time.now() - start_time).to_sec()

        iiwa_msg.position = copy.deepcopy(iiwa_init_joint_pos.tolist())
        allegro_msg.position = copy.deepcopy(allegro_init_joint_pos.tolist())
        current_iiwa_target_pos = joint_pos_history[current_idx, :7]
        current_allegro_target_pos = joint_pos_history[current_idx, 7:]
        if elapsed_time > STATIONARY_TIME:
            if interpolation_alpha >=1:
                step_size = 0.05
                interpolation_alpha = 0
                current_idx += 1
                print("Current IDX", current_idx)
                if current_idx >= 200: current_idx = 200
                rospy.loginfo(f"DONE {current_idx}")
                iiwa_init_joint_pos = new_pos_iiwa
                allegro_init_joint_pos = new_pos_allegro
            else:
                interpolation_alpha += step_size
            new_pos_iiwa = (
                iiwa_init_joint_pos * (1 - interpolation_alpha)
                + current_iiwa_target_pos * interpolation_alpha
            )
            iiwa_msg.position = new_pos_iiwa.tolist()

            new_pos_allegro = (
                allegro_init_joint_pos * (1 - interpolation_alpha)
                + current_allegro_target_pos * interpolation_alpha
            )
            allegro_msg.position = new_pos_allegro.tolist()
        else:
            rospy.loginfo(
                f"Holding at stationary position for {STATIONARY_TIME - elapsed_time} seconds more"
            )

        # Update timestamps
        now = rospy.Time.now()
        iiwa_msg.header.stamp = now
        allegro_msg.header.stamp = now

        # Publish both commands
        # breakpoint()
        pub_iiwa.publish(iiwa_msg)
        pub_allegro.publish(allegro_msg)
        


        time_since_last_publish = (rospy.Time.now() - last_publish_time).to_sec()
        if time_since_last_publish > 0.2:
            rospy.loginfo("\n" + "=" * 80)
            rospy.loginfo("SLOW")
        # rospy.loginfo(
        #     f"Publishing {np.round(time_since_last_publish * 1000)} ms since last publish, {np.round(1./time_since_last_publish)} Hz)"
        # )
        if time_since_last_publish > 0.2:
            rospy.loginfo("SLOW")
            rospy.loginfo("\n" + "=" * 80 + "\n")
        last_publish_time = rospy.Time.now()
        rate.sleep()

def interpolate_joint_pos_history(joint_pos_history, start_idx, end_idx, slow_down_factor=10):
    interpolated_joint_pos_history = [] # (slow_down_factor*(end_idx-start_idx), 23)
    for i in range(start_idx, end_idx):
        for j in range(slow_down_factor):
            interpolated_joint_pos = joint_pos_history[i] + (joint_pos_history[i+1] - joint_pos_history[i]) * (j+1)/slow_down_factor
            interpolated_joint_pos_history.append(interpolated_joint_pos)
        
    return np.array(interpolated_joint_pos_history)

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
    print(f"Published joint positions: {iiwa_msg.position}, {allegro_msg.position}")


def move_to_pose(target_pos: np.ndarray, pub_iiwa: rospy.Publisher, pub_allegro: rospy.Publisher) -> None:
    assert target_pos.shape == (23,), f"target_pos.shape: {target_pos.shape}, expected: ({23},)"
    current_allegro_pos = CURRENT_JOINT_POS_ALLEGRO.copy()
    current_iiwa_pos = CURRENT_JOINT_POS_IIWA.copy()
    current_pos = np.concatenate([current_iiwa_pos, current_allegro_pos])

    SECONDS_TO_MOVE = 10.0
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
        
        timestep = 0
        print(f"joint_positions_array[timestep]: {joint_positions_array[timestep]}")
        print(f"joint_positions_array[timestep, :7]: {joint_positions_array[timestep, :7]}")
        print(f"joint_positions_array[timestep, 7:]: {joint_positions_array[timestep, 7:]}")
        print(f"CURRENT_JOINT_POS_IIWA: {CURRENT_JOINT_POS_IIWA}")
        print(f"CURRENT_JOINT_POS_ALLEGRO: {CURRENT_JOINT_POS_ALLEGRO}")
        breakpoint()
        move_to_pose(joint_positions_array[timestep], pub_iiwa=pub_iiwa, pub_allegro=pub_allegro)

        # publish_joint_cmds(iiwa_init_joint_pos=iiwa_init, allegro_init_joint_pos=allegro_init, joint_pos_history=interpolated_joint_pos_history)
    except rospy.ROSInterruptException:
        pass 
