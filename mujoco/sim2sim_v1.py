import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

class BaseSimulator:
    def __init__(self):
        self.sim_hz = 120
        self.sim_dt = 1 / self.sim_hz
        self.init_scene()

    def init_scene(self):
        robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml")
        # robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml")
        assert robot_path.exists(), f"Robot path does not exist: {robot_path}"
        self.mj_model = mujoco.MjModel.from_xml_path(str(robot_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def sim_step(self):
        self.mj_data.ctrl = np.zeros(self.mj_model.nu)
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def run(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()

            # End of loop timekeeping
            sim_cnt += 1
            end_time = time.time()
            time_elapsed = end_time - start_time
            sleep_dt = self.sim_dt - time_elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                print(f"Simulation is running slower than real time, desired FPS = {1.0 / self.sim_dt:.1f}, actual FPS = {1.0 / time_elapsed:.1f}")

            # Get FPS
            if sim_cnt % 100 == 0:
                print(f"FPS: {100 / (end_time - start_time)}")

            start_time = end_time

if __name__ == "__main__":
    simulation = BaseSimulator()
    simulation.run()
