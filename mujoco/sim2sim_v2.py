import mujoco
import numpy as np
import time
from pathlib import Path

class BaseSimulator:
    def __init__(self):
        self.sim_hz = 60
        self.sim_dt = 1 / self.sim_hz
        self.init_scene()

    def init_scene(self):
        robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml")
        # robot_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/iiwa14.xml")
        assert robot_path.exists(), f"Robot path does not exist: {robot_path}"
        self.mj_model = mujoco.MjModel.from_xml_path(str(robot_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

    def sim_step(self):
        self.mj_data.ctrl = np.zeros(self.mj_model.nu)
        mujoco.mj_step(self.mj_model, self.mj_data)
    
    def run(self):
        sim_step_times = []
        sleep_times = []

        while True:
            start_time = time.time()
            self.sim_step()
            end_time = time.time()

            # End of loop timekeeping
            time_elapsed = end_time - start_time
            sim_step_times.append(time_elapsed)
            sleep_dt = self.sim_dt - time_elapsed
            sleep_times.append(sleep_dt)
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                print(f"Simulation is running slower than real time, desired FPS = {1.0 / self.sim_dt:.1f}, actual FPS = {1.0 / time_elapsed:.1f}")

            # Get FPS
            if len(sim_step_times) == 100:
                total_sim_step_time = np.sum(sim_step_times)
                total_sleep_time = np.array(sleep_times).clip(min=0.0, max=None).sum()
                total_time = total_sim_step_time + total_sleep_time
                print(f"total_sim_step_time: {total_sim_step_time}")
                print(f"total_sleep_time: {total_sleep_time}")
                print(f"total_time: {total_time}")
                breakpoint()
                print(f"Max FPS: {100 / total_sim_step_time:.1f}")
                print(f"FPS: {100 / total_time:.1f}")
                sim_step_times = []

if __name__ == "__main__":
    simulation = BaseSimulator()
    simulation.run()
