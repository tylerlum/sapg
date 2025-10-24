import time

import numpy as np
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from sim import JOINT_NAMES, N_IIWA_JOINTS, Simulator, SimulatorConfig
from termcolor import colored

# Goal object pose doesn't exist in the simulation
# But we can just publish the goal object pose above the table
PUBLISH_GOAL_OBJECT_POSE = True


def warn(message: str):
    print(colored(message, "yellow"))


def info(message: str):
    print(colored(message, "green"))


class SimRos:
    def __init__(self, sim: Simulator, update_and_publish_dt: float = 1.0 / 60):
        self.sim = sim
        self._update_and_publish_dt = update_and_publish_dt
        self._last_update_and_publish_time = time.time()
        self._init_ros()

    def _init_ros(self):
        rospy.init_node("sim_ros_node")

        self.latest_iiwa_joint_cmd = None
        self.latest_allegro_joint_cmd = None

        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self._iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self._allegro_joint_cmd_callback
        )

        self.iiwa_pub = rospy.Publisher("/iiwa/joint_states", JointState, queue_size=10)
        self.allegro_pub = rospy.Publisher(
            "/allegroHand_0/joint_states", JointState, queue_size=10
        )
        self.object_pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=10)
        if PUBLISH_GOAL_OBJECT_POSE:
            self.goal_object_pose_pub = rospy.Publisher("/goal_object_pose", Pose, queue_size=10)

    def _iiwa_joint_cmd_callback(self, msg: JointState):
        self.latest_iiwa_joint_cmd = np.array(msg.position)

    def _allegro_joint_cmd_callback(self, msg: JointState):
        self.latest_allegro_joint_cmd = np.array(msg.position)

    def _publish(self, sim_state: dict[str, np.ndarray]):
        object_pos = sim_state["object_pos"]
        object_quat_wxyz = sim_state["object_quat_wxyz"]
        object_pose_msg = Pose()
        object_pose_msg.position.x = object_pos[0]
        object_pose_msg.position.y = object_pos[1]
        object_pose_msg.position.z = object_pos[2]
        object_pose_msg.orientation.w = object_quat_wxyz[0]
        object_pose_msg.orientation.x = object_quat_wxyz[1]
        object_pose_msg.orientation.y = object_quat_wxyz[2]
        object_pose_msg.orientation.z = object_quat_wxyz[3]
        self.object_pose_pub.publish(object_pose_msg)

        if PUBLISH_GOAL_OBJECT_POSE:
            goal_object_pos = sim_state["table_pos"]
            goal_object_quat_wxyz = sim_state["table_quat_wxyz"]
            goal_object_pose_msg = Pose()
            goal_object_pose_msg.position.x = goal_object_pos[0]
            goal_object_pose_msg.position.y = goal_object_pos[1]
            goal_object_pose_msg.position.z = goal_object_pos[2] + 0.5  # Above the table
            goal_object_pose_msg.orientation.w = goal_object_quat_wxyz[0]
            goal_object_pose_msg.orientation.x = goal_object_quat_wxyz[1]
            goal_object_pose_msg.orientation.y = goal_object_quat_wxyz[2]
            goal_object_pose_msg.orientation.z = goal_object_quat_wxyz[3]
            self.goal_object_pose_pub.publish(goal_object_pose_msg)

        joint_positions = sim_state["joint_positions"]
        joint_velocities = sim_state["joint_velocities"]
        joint_names = JOINT_NAMES

        iiwa_joint_msg = JointState(
            name=joint_names[:N_IIWA_JOINTS],
            position=joint_positions[:N_IIWA_JOINTS],
            velocity=joint_velocities[:N_IIWA_JOINTS],
        )
        self.iiwa_pub.publish(iiwa_joint_msg)

        allegro_joint_msg = JointState(
            name=joint_names[N_IIWA_JOINTS:],
            position=joint_positions[N_IIWA_JOINTS:],
            velocity=joint_velocities[N_IIWA_JOINTS:],
        )
        self.allegro_pub.publish(allegro_joint_msg)

    def _continue_running(self) -> bool:
        return not rospy.is_shutdown() and self.sim._continue_running()

    def run(self):
        first_commands_received = False

        loop_no_sleep_dts, loop_dts = [], []

        # Loop will try to run at self.sim.config.sim_dt
        # But will only update joint PD targets and publish sim state at self._update_and_publish_dt
        while self._continue_running():
            start_loop_no_sleep_time = time.time()

            update_and_publish = False
            if (
                time.time() - self._last_update_and_publish_time
                > self._update_and_publish_dt
            ):
                update_and_publish = True
                self._last_update_and_publish_time = time.time()

            if (
                self.latest_iiwa_joint_cmd is None
                or self.latest_allegro_joint_cmd is None
            ):
                # Still run loop while waiting to start publishing sim state
                # Print waiting message every 1000 loops
                if len(loop_no_sleep_dts) % 1000 == 0:
                    warn(
                        f"Waiting: latest_iiwa_joint_cmd = {self.latest_iiwa_joint_cmd}, latest_allegro_joint_cmd = {self.latest_allegro_joint_cmd}"
                    )
            elif update_and_publish:
                if not first_commands_received:
                    info("=" * 100)
                    info("First commands received, starting to publish sim state")
                    info("=" * 100)
                    first_commands_received = True

                # Get latest joint commands
                iiwa_joint_cmd = self.latest_iiwa_joint_cmd.copy()
                allegro_joint_cmd = self.latest_allegro_joint_cmd.copy()
                joint_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
                self.sim.set_robot_joint_pos_targets(joint_cmd)

            # Step simulation
            self.sim.sim_step()

            if self.sim.config.enable_viewer:
                self.sim.viewer.sync()

            # Get simulation state
            if update_and_publish:
                sim_state_dict = self.sim.get_sim_state()
                self._publish(sim_state_dict)

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.sim.config.sim_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                warn(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.sim.config.sim_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            # Get FPS
            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(
                PRINT_FPS_EVERY_N_SECONDS / self.sim.config.sim_dt
            )
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


def main():
    sim = Simulator(SimulatorConfig(enable_viewer=False, sim_dt=1.0 / 500.0))
    sim_ros = SimRos(sim)
    sim_ros.run()


if __name__ == "__main__":
    main()
