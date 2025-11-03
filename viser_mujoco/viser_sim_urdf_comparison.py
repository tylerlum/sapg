import time
from pathlib import Path

import numpy as np
import viser
from sim import INIT_JOINT_POS, Simulator, SimulatorConfig
from viser.extras import ViserUrdf
from viser_sim import ViserSimulator


def main():
    server = viser.ViserServer()
    sim = Simulator(SimulatorConfig(enable_viewer=False))
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(
            "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
        ),
        load_meshes=True,
    )
    viser_urdf.update_cfg(INIT_JOINT_POS)
    viser_sim = ViserSimulator(server, sim)
    viser_sim.sim.sim_step()
    viser_sim._update_viser()
    while True:
        print("Sleeping for 1.0 seconds")
        time.sleep(1.0)
        print(f"viser urdf qpos: {np.round(viser_urdf._urdf._cfg, 3)}")
        print(
            f"viser sim qpos: {np.round(viser_sim.sim.get_sim_state()['joint_positions'], 3)}"
        )
        breakpoint()


if __name__ == "__main__":
    main()
