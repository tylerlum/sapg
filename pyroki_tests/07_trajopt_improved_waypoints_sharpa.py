"""Trajectory Optimization with Waypoints

Demonstrates avoiding a Box obstacle while hitting specific waypoints.
"""

import json
import time
from pathlib import Path
from typing import Literal

# Enable caching for faster re-runs
import jax
import numpy as np
import pyroki as pk
import pyroki_snippets as pks
import trimesh
import tyro
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


def main(robot_name: Literal["ur5", "panda", "sharpa"] = "sharpa"):
    # Load robot
    HOME_JOINT_POS_IIWA = np.array(
        [
            -1.571,
            1.571 - np.deg2rad(10),
            -0.000,
            1.376 + np.deg2rad(10),
            -0.000,
            1.485,
            1.308,
        ]
    )
    HOME_JOINT_POS_SHARPA = np.zeros(22)
    HOME_JOINT_POS = np.concatenate([HOME_JOINT_POS_IIWA, HOME_JOINT_POS_SHARPA])

    if robot_name == "ur5":
        urdf = load_robot_description("ur5_description")
        down_wxyz = np.array([0.707, 0, 0.707, 0])
        target_link_name = "ee_link"
        sphere_json_path = Path(__file__).parent / "assets" / "ur5_spheres.json"

        default_cfg = np.zeros(6)
        default_cfg[1] = -1.308
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=default_cfg)

    elif robot_name == "panda":
        urdf = load_robot_description("panda_description")
        target_link_name = "panda_hand"
        down_wxyz = np.array([0, 0, 1, 0])
        sphere_json_path = Path(__file__).parent / "assets" / "panda_spheres.json"
        robot = pk.Robot.from_urdf(urdf)

    elif robot_name == "sharpa":
        import yourdfpy

        # Sharpa
        urdf = yourdfpy.URDF.load(
            "assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        target_link_name = "left_hand_C_MC"
        down_wxyz = np.array([0.5, 0.5, 0.5, -0.5])
        sphere_json_path = Path(__file__).parent / "assets" / "sharpa_spheres.json"
        robot = pk.Robot.from_urdf(urdf, default_joint_cfg=HOME_JOINT_POS)
    else:
        raise ValueError(f"Invalid robot: {robot_name}")

    # Load collision spheres
    with open(sphere_json_path, "r") as f:
        sphere_decomposition = json.load(f)
    robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomposition=sphere_decomposition,
        urdf=urdf,
    )

    # Problem Setup
    timesteps, dt = 50, 0.05

    # Implicit start config based on current robot state (usually 0 unless set)
    # We will use the robot's default config as the "seed" for the start pose IK
    # but to pass a specific start_cfg to the solver, we can run a quick IK first
    # or just use zeros if compatible. Let's use zeros for Panda.
    start_cfg = HOME_JOINT_POS
    assert (
        len(start_cfg) == robot.joints.num_actuated_joints
    ), f"start_cfg.shape: {start_cfg.shape}, expected: ({robot.joints.num_actuated_joints},)"

    # Define Obstacles
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )

    table_size = np.array([0.475, 0.4, 0.3])
    table_center = np.array([0.0, -0.8, 0.38])

    table_coll = pk.collision.Box.from_extent(
        extent=table_size,
        position=table_center,
    )
    world_coll = [ground_coll, table_coll]

    # Define Waypoints
    # 1. Start is implicitly t=0 (enforced by start_cfg)
    # 2. Midpoint (High over the table)

    mid_pos = np.array([0.0, -0.6, 0.8])
    right_pos = np.array([-0.5, -0.6, 0.8])
    left_pos = np.array([0.5, -0.6, 0.8])

    waypoints = {}
    waypoints.update({
        5 + i: (mid_pos, down_wxyz)
        for i in range(5)
    })
    waypoints.update({
        25 + i: (right_pos, down_wxyz)
        for i in range(5)
    })
    waypoints.update({
        49 - i: (left_pos, down_wxyz)
        for i in range(5)
    })

    # Visualize problem setup
    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # Draw robot
    urdf_vis = ViserUrdf(server, urdf)
    urdf_vis.update_cfg(start_cfg)

    # Draw robot collision model ghost
    server.scene.add_mesh_trimesh(
        "robot_coll_ghost",
        robot_coll.at_config(robot, start_cfg).to_trimesh(),
    )

    # Draw Table
    server.scene.add_mesh_trimesh(
        "table_box",
        trimesh.creation.box(
            extents=table_size,
            transform=trimesh.transformations.translation_matrix(table_center),
        ),
    )

    # Draw Waypoints
    for t, (pos, wxyz) in waypoints.items():
        server.scene.add_frame(
            f"/waypoint_t{t}",
            position=pos,
            wxyz=wxyz,
            axes_length=0.1,
            axes_radius=0.01,
        )

    # Solve trajectory
    print("Solving trajectory with waypoints...")
    start_time = time.time()

    traj = pks.solve_waypoint_trajopt(
        robot,
        robot_coll,
        world_coll,
        target_link_name,
        start_cfg,
        waypoints,
        timesteps,
        dt,
    )

    traj = np.array(traj)
    print(f"Solved in {time.time() - start_time:.4f}s")

    # Visualize trajectory
    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        cfg = traj[slider.value]
        urdf_vis.update_cfg(cfg)

        # Update collision ghost occasionally to see fit
        # (Updating every frame might be slow for complex meshes)
        if slider.value % 5 == 0:
            server.scene.add_mesh_trimesh(
                "robot_coll_ghost",
                robot_coll.at_config(robot, cfg).to_trimesh(),
            )

        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
