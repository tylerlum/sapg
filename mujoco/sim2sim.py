from __future__ import annotations
import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

ENABLE_VIEWER = True

SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION = 1.0, 0.005, 0.0001
INIT_JOINT_POS_IIWA = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])
IIWA_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',]
IIWA_ACTUATOR_NAMES = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7',]

class BaseSimulator:
    def __init__(self):
        self.sim_hz = 1000  # Need a high enough frequency to get stable physics
        self.sim_dt = 1 / self.sim_hz
        self.init_scene()
        self.set_init_robot_joint_positions()

    def init_scene(self):
        robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml")
        assert robot_path.exists(), f"Robot path does not exist: {robot_path}"

        # Load mjspec from robot path
        spec = mujoco.MjSpec.from_file(str(robot_path))
        spec.option.timestep = self.sim_dt

        # Table
        WHITE_RGBA = np.array([1.0, 1.0, 1.0, 1.0])
        TABLE_LEN_X, TABLE_LEN_Y, TABLE_LEN_Z = 0.475, 0.4, 0.3
        TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z = 0.0, -0.8, 0.38
        table_body = spec.worldbody.add_body()
        table_body.name = "table"
        table_body.pos = np.array([TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z])
        table_geom = table_body.add_geom()
        table_geom.name = "table_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = np.array([TABLE_LEN_X / 2, TABLE_LEN_Y / 2, TABLE_LEN_Z / 2])  # Half extents
        table_geom.pos = np.array([0.0, 0.0, 0.0])
        table_geom.rgba = WHITE_RGBA
        table_geom.friction = np.array([SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION])

        # Object
        OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z = 0.0, -0.8, 0.38 + 0.8
        mesh = spec.add_mesh()
        mesh.name = "object_mesh"
        # mesh.file = "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured_vhacd.obj"
        mesh.file = "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/040_large_marker/040_large_marker/google_16k/textured_vhacd.obj"
        assert Path(mesh.file).exists(), f"Mesh file does not exist: {mesh.file}"
        mesh.scale = np.array([1.0, 1.0, 1.0])
        object_body = spec.worldbody.add_body()
        object_body.name = "object"
        object_body.pos = np.array([OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z])
        object_geom = object_body.add_geom()
        object_geom.name = "object_geom"
        object_geom.type = mujoco.mjtGeom.mjGEOM_MESH
        object_geom.meshname = mesh.name
        object_geom.rgba = np.array([0.0, 0.0, 0.0, 1.0])
        object_geom.friction = np.array([SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION])
        object_free_joint = object_body.add_joint()
        object_free_joint.name = "object_free_joint"
        object_free_joint.type = mujoco.mjtJoint.mjJNT_FREE

        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        if ENABLE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def set_init_robot_joint_positions(self):
        iiwa_joint_ids = [self.mj_model.joint(name=name).id for name in IIWA_JOINT_NAMES]
        for i, iiwa_joint_id in enumerate(iiwa_joint_ids):
            self.mj_data.qpos[iiwa_joint_id] = INIT_JOINT_POS_IIWA[i]

    def get_body_pose(self, body_name: str):
        body_id = self.mj_model.body(name=body_name).id
        pos = self.mj_data.xpos[body_id]   # (3,) world position of body frame
        quat = self.mj_data.xquat[body_id] # (4,) world orientation quaternion (w, x, y, z)
        return pos, quat

    def get_sim_state(self):
        # Usage:
        table_pos, table_quat = self.get_body_pose("table")
        object_pos, object_quat = self.get_body_pose("object")
        robot_base_pos, robot_base_quat = self.get_body_pose("base")  # replace with actual base body name
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        joint_ids = [self.mj_model.joint(name=name).id for name in joint_names]
        joint_positions = self.mj_data.qpos[joint_ids]
        joint_velocities = self.mj_data.qvel[joint_ids]
        actuator_names = [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

        PRINT = False
        if PRINT:
            print(f"Table pos: {table_pos}, Table quat: {table_quat}")
            print(f"Object pos: {object_pos}, Object quat: {object_quat}")
            print(f"Robot base pos: {robot_base_pos}, Robot base quat: {robot_base_quat}")
            print(f"Joint names: {joint_names}")
            print(f"Joint positions: {joint_positions}")
            print(f"Joint velocities: {joint_velocities}")
            print("Actuator names:", actuator_names)

    def sim_step(self):
        iiwa_actuator_ids = [self.mj_model.actuator(name=name).id for name in IIWA_ACTUATOR_NAMES]
        for i, iiwa_actuator_id in enumerate(iiwa_actuator_ids):
            self.mj_data.ctrl[iiwa_actuator_id] = INIT_JOINT_POS_IIWA[i]
        mujoco.mj_step(self.mj_model, self.mj_data)

    def continue_running(self) -> bool:
        if ENABLE_VIEWER:
            return self.viewer.is_running()
        else:
            return True
    
    def run(self):
        loop_no_sleep_dts, loop_dts = [], []

        while self.continue_running():
            start_loop_no_sleep_time = time.time()

            # Step simulation
            self.sim_step()

            # Get simulation state
            self.get_sim_state()

            # Update viewer
            if ENABLE_VIEWER:
                self.viewer.sync()

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.sim_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                print(f"Simulation is running slower than real time, desired FPS = {1.0 / self.sim_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}")
            loop_dts.append(loop_dt)

            # Get FPS
            if len(loop_dts) == 100:
                total_loop_no_sleep_dt = np.sum(loop_no_sleep_dts)
                total_loop_dt = np.sum(loop_dts)
                print(f"Max FPS: {100 / total_loop_no_sleep_dt:.1f}")
                print(f"FPS: {100 / total_loop_dt:.1f}")
                loop_no_sleep_dts, loop_dts = [], []

if __name__ == "__main__":
    simulation = BaseSimulator()
    simulation.run()
