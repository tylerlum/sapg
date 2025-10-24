#!/usr/bin/env python

import copy
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pytorch_kinematics as pk
import rospy
import torch
from geometry_msgs.msg import Pose
from rl_player import RlPlayer
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState

from isaacgymenvs.utils.observation_action_utils import (
    compute_joint_pos_targets,
    compute_observation,
)


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


FABRIC_MODE: Literal["PCA", "ALL"] = "PCA"


class RLPolicyNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("rl_policy_node")

        # Publisher for iiwa and allegro joint commands
        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=10
        )
        self.allegro_joint_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=10
        )

        # Variables to store the latest messages
        self.object_pose_msg = None
        self.goal_object_pose_msg = None
        self.iiwa_joint_state_msg = None
        self.allegro_joint_state_msg = None

        # Subscribers
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_joint_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )

        # RL Player setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_observations = 117  # Update this number based on actual dimensions
        self.num_actions = 23

        # HACK
        # self.config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/config.yaml")
        # self.checkpoint_path = Path(__file__).parent / "checkpoint.pt"
        self.checkpoint_path = Path("/juno/u/tylerlum/github_repos/sapg/train_dir/allegro_kuka_reorientation/test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/runs/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s/nn/00_test_24576envs_mixed_expl_learn_param_lf_1p_09_10_00h10m32s.pth")

        # Create the RL player
        self.player = RlPlayer(
            num_observations=self.num_observations,
            num_actions=self.num_actions,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # ROS rate
        self.rate_hz = 60
        self.rate = rospy.Rate(self.rate_hz)

        # Set up chain and palm_serial_chain
        asset_root = Path(__file__).parent / "../assets"
        urdf_path = (
            asset_root / "urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
        )
        assert urdf_path.exists(), f"URDF file {urdf_path} does not exist"
        self.chain = pk.build_chain_from_urdf(
            open(urdf_path).read(),
        ).to(device=self.device)
        self.palm_serial_chain = pk.SerialChain(self.chain, "iiwa7_link_7").to(
            device=self.device
        )

        # State: prev_targets
        self.prev_targets = None

    def object_pose_callback(self, msg: Pose):
        self.object_pose_msg = msg

    def goal_object_pose_callback(self, msg: Pose):
        self.goal_object_pose_msg = msg

    def iiwa_joint_state_callback(self, msg: JointState):
        self.iiwa_joint_state_msg = msg

    def allegro_joint_state_callback(self, msg: JointState):
        self.allegro_joint_state_msg = msg

    def create_observation(self) -> Optional[torch.Tensor]:
        # Ensure all messages are received before processing
        if (
            self.iiwa_joint_state_msg is None
            or self.allegro_joint_state_msg is None
            or self.object_pose_msg is None
            or self.goal_object_pose_msg is None
        ):
            rospy.logwarn(
                f"Waiting for all messages to be received... iiwa_joint_state_msg: {var_to_is_none_str(self.iiwa_joint_state_msg)}, allegro_joint_state_msg: {var_to_is_none_str(self.allegro_joint_state_msg)}, object_pose_msg: {var_to_is_none_str(self.object_pose_msg)}, goal_object_pose_msg: {var_to_is_none_str(self.goal_object_pose_msg)}"
            )
            return None

        iiwa_joint_state_msg = copy.copy(self.iiwa_joint_state_msg)
        allegro_joint_state_msg = copy.copy(self.allegro_joint_state_msg)
        object_pose_msg = copy.copy(self.object_pose_msg)
        goal_object_pose_msg = copy.copy(self.goal_object_pose_msg)

        # Concatenate the data from joint states and object pose
        iiwa_position = np.array(iiwa_joint_state_msg.position)
        iiwa_velocity = np.array(iiwa_joint_state_msg.velocity)

        allegro_position = np.array(allegro_joint_state_msg.position)
        allegro_velocity = np.array(allegro_joint_state_msg.velocity)

        T_C_O = pose_msg_to_T(object_pose_msg)
        T_C_G = pose_msg_to_T(goal_object_pose_msg)

        T_R_O = self.T_R_C @ T_C_O
        object_position_R, object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_O)

        T_R_G = self.goal_T_R_C @ T_C_G
        goal_object_pos_R, goal_object_quat_xyzw_R = T_to_pos_quat_xyzw(T_R_G)

        q = np.concatenate([iiwa_position, allegro_position])
        qd = np.concatenate([iiwa_velocity, allegro_velocity])

        observation = compute_observation(
            q=torch.from_numpy(q).float().to(self.device),
            qd=torch.from_numpy(qd).float().to(self.device),
            object_pose=torch.from_numpy(object_position_R).float().to(self.device),
            goal_object_pose=torch.from_numpy(goal_object_pos_R)
            .float()
            .to(self.device),
            object_scales=torch.from_numpy(self.object_scales).float().to(self.device),
            chain=self.chain,
            palm_serial_chain=self.palm_serial_chain,
        )
        assert_equals(observation.shape, (self.num_observations,))

        return torch.from_numpy(observation).float().unsqueeze(0).to(self.device)

    def publish_targets(self, joint_pos_targets: torch.Tensor):
        assert_equals(joint_pos_targets.shape, (1, self.num_actions))
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
        iiwa_msg.position = joint_pos_targets[:, :7].cpu().numpy().tolist()
        self.iiwa_joint_cmd_pub.publish(iiwa_msg)
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
        allegro_msg.position = joint_pos_targets[:, 7:].cpu().numpy().tolist()
        self.allegro_joint_cmd_pub.publish(allegro_msg)

    def run(self):
        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Create observation from the latest messages
            obs = self.create_observation()

            if obs is not None:
                if self.prev_targets is None:
                    self.prev_targets = (
                        torch.from_numpy(
                            np.concatenate(
                                [
                                    self.iiwa_joint_state_msg.position,
                                    self.allegro_joint_state_msg.position,
                                ]
                            )
                        )
                        .float()
                        .to(self.device)
                    )

                assert_equals(obs.shape, (1, self.num_observations))

                # Get the normalized action from the RL player
                normalized_action = self.player.get_normalized_action(
                    obs=obs,
                    deterministic_actions=False,
                    # obs=obs, deterministic_actions=True
                )
                # normalized_action = torch.zeros(1, self.num_actions, device=self.device)
                assert_equals(normalized_action.shape, (1, self.num_actions))

                joint_pos_targets = compute_joint_pos_targets(
                    actions=normalized_action,
                    prev_targets=self.prev_targets,
                    act_moving_average=0.1,
                    hand_dof_speed_scale=1.0,
                    dt=1 / 60,
                )
                assert_equals(joint_pos_targets.shape, (1, self.num_actions))

                # Publish the targets
                self.publish_targets(joint_pos_targets)
                self.prev_targets = joint_pos_targets.clone()

            # Sleep to maintain 15 loop rate
            before_sleep_time = rospy.Time.now()
            self.rate.sleep()
            after_sleep_time = rospy.Time.now()

            rospy.loginfo(
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )

    @property
    def T_R_C(self) -> np.ndarray:
        # HACK
        return np.eye(4)

    @property
    def goal_T_R_C(self) -> np.ndarray:
        # HACK
        return np.eye(4)

    @property
    def object_scales(self) -> np.ndarray:
        # HACK
        return np.array([1.0, 1.0, 1.0])


if __name__ == "__main__":
    try:
        rl_policy_node = RLPolicyNode()
        rl_policy_node.run()
    except rospy.ROSInterruptException:
        pass
