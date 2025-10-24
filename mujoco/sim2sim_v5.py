import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

ENABLE_VIEWER = True

class BaseSimulator:
    def __init__(self):
        self.sim_hz = 60
        self.sim_dt = 1 / self.sim_hz
        self.init_scene()

    def init_scene(self):
        this_dir = Path(__file__).parent
        output_dir = this_dir / "scenes"
        output_dir.mkdir(parents=True, exist_ok=True)

        robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml")
        # robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml")
        assert robot_path.exists(), f"Robot path does not exist: {robot_path}"

        spec = mujoco.MjSpec.from_file(str(robot_path))

        # Table
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
        table_geom.rgba = np.array([1.0, 1.0, 1.0, 1.0])

        # Object
        OBJECT_LEN_X, OBJECT_LEN_Y, OBJECT_LEN_Z = 0.3, 0.05, 0.05
        OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z = 0.0, -0.8, 0.38 + 0.3
        object_body = spec.worldbody.add_body()
        object_body.name = "object"
        object_body.pos = np.array([OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z])
        object_geom = object_body.add_geom()
        object_geom.name = "object_geom"
        object_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        object_geom.size = np.array([OBJECT_LEN_X / 2, OBJECT_LEN_Y / 2, OBJECT_LEN_Z / 2])  # Half extents
        object_geom.pos = np.array([0.0, 0.0, 0.0])
        object_geom.rgba = np.array([0.0, 0.0, 0.0, 1.0])
        object_free_joint = object_body.add_joint()
        object_free_joint.name = "object_free_joint"
        object_free_joint.type = mujoco.mjtJoint.mjJNT_FREE

        self.mj_model = spec.compile()
        # self.mj_model = mujoco.MjModel.from_xml_path(str(robot_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        if ENABLE_VIEWER:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def sim_step(self):
        # self.mj_data.ctrl = np.zeros(self.mj_model.nu)
        if not hasattr(self, "counter"):
            self.counter = 0
        self.counter += 1

        if self.counter % 20 < 10:
            self.mj_data.ctrl = np.ones(self.mj_model.nu)
        else:
            self.mj_data.ctrl = np.zeros(self.mj_model.nu)
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
