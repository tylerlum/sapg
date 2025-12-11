from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from recorded_data_scripts.recorded_data_sharpa import RecordedData

# ###########
# Constants
# ###########
AXES_LENGTH = 0.2
AXES_RADIUS = 0.01

DISABLE_AXES = False
if DISABLE_AXES:
    AXES_LENGTH = 0.00001
    AXES_RADIUS = 0.00001


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    assert xyzw.shape[-1] == 4, f"Expected xyzw to be (..., 4), got {xyzw.shape}"
    return xyzw[..., [3, 0, 1, 2]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()
    file_path = Path(args.file_path)

    # ###########
    # Load recorded data
    # ###########
    # file_path = Path(
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-43-04.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-42-41.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-30-37.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-09.npz"
    #     # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-31.npz"
    #     "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_17-18-32.npz"
    # )
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)

    # ###########
    # Create viser server and create viser objects
    # ###########
    # Create server
    SERVER = viser.ViserServer()
    SERVER.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Set initial camera pose
    @SERVER.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, -1.0, 1.03)
        # client.camera.wxyz = (0, 0, 0, 1)
        client.camera.look_at = (0, 0, 0.53)

    # Load assets into viser
    KUKA_ALLEGRO_URDF_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
        "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_left_sharpa_between.urdf"
    )
    assert KUKA_ALLEGRO_URDF_PATH.exists(), (
        f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
    )
    from isaacgymenvs.utils.objects import NAME_TO_OBJECT
    DEFAULT_OBJECT_NAME = "blue_cuboid"
    object_name = DEFAULT_OBJECT_NAME
    if recorded_data.object_name is None:
        print(f"Using default object name: {DEFAULT_OBJECT_NAME}")
        object_name = DEFAULT_OBJECT_NAME
    elif recorded_data.object_name not in NAME_TO_OBJECT:
        print(f"Object name {recorded_data.object_name} not found in NAME_TO_OBJECT, using default object name: {DEFAULT_OBJECT_NAME}")
        object_name = DEFAULT_OBJECT_NAME
    else:
        object_name = recorded_data.object_name
    OBJECT_URDF_PATH = NAME_TO_OBJECT[object_name].filepath
    # OBJECT_URDF_PATH = Path(
    #     "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf"
    # )
    assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
    ALLEGRO_URDF_PATH = Path(
        "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf"
    )
    assert ALLEGRO_URDF_PATH.exists(), (
        f"ALLEGRO_URDF_PATH not found: {ALLEGRO_URDF_PATH}"
    )
    TABLE_URDF_PATH = Path(
        "/home/tylerlum/github_repos/sapg/assets/urdf/table_narrow.urdf"
    )
    assert TABLE_URDF_PATH.exists(), f"TABLE_URDF_PATH not found: {TABLE_URDF_PATH}"

    # Robot
    kuka_allegro_frame = SERVER.scene.add_frame(
        "/robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS,
    )
    kuka_allegro_viser = ViserUrdf(
        SERVER, KUKA_ALLEGRO_URDF_PATH, root_node_name="/robot/state"
    )

    # Target robot
    if recorded_data.robot_joint_pos_targets_array is not None:
        target_kuka_allegro_frame = SERVER.scene.add_frame(
            "/target_robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS,
        )
        BLUE_RGBA = (0, 0, 255, 0.5)
        target_kuka_allegro_viser = ViserUrdf(
            SERVER, KUKA_ALLEGRO_URDF_PATH, root_node_name="/target_robot/state", mesh_color_override=BLUE_RGBA
        )

    # Object
    object_frame = SERVER.scene.add_frame(
        "/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )
    _object_viser = ViserUrdf(SERVER, OBJECT_URDF_PATH, root_node_name="/object")

    # Table
    if recorded_data.table_root_states_array is not None:
        table_frame = SERVER.scene.add_frame(
            "/table", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
        )
        _table_viser = ViserUrdf(SERVER, TABLE_URDF_PATH, root_node_name="/table")

    # Goal
    if recorded_data.goal_root_states_array is not None:
        goal_frame = SERVER.scene.add_frame(
            "/goal", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
        )
        INCLUDE_GOAL_OBJECT = True
        if INCLUDE_GOAL_OBJECT:
            _goal_object_viser = ViserUrdf(SERVER, OBJECT_URDF_PATH, root_node_name="/goal")

    # Palm
    palm_frame = SERVER.scene.add_frame(
        "/robot_palm", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )

    # Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
    kuka_allegro_viser_joint_names = kuka_allegro_viser._urdf.actuated_joint_names

    # ###########
    # Add controls
    # ###########
    def get_frame_idx_slider_text(
        recorded_data: RecordedData,
        idx: int,
    ) -> str:
        fps = 1 / recorded_data.dt
        current_time = recorded_data.time_array[idx] - recorded_data.time_array[0]
        total_time = recorded_data.total_time
        return f"{current_time:.2f}s/{total_time:.2f}s.Frame {idx}/{len(recorded_data)}. ({fps:.0f}fps)"

    with SERVER.gui.add_folder("Frame Controls"):
        frame_idx_slider = SERVER.gui.add_slider(
            label=get_frame_idx_slider_text(recorded_data=recorded_data, idx=0),
            min=0,
            max=len(recorded_data) - 1,
            step=1,
            initial_value=0,
        )
        pause_toggle_button = SERVER.gui.add_button(
            label="Pause",
        )
        increment_button = SERVER.gui.add_button(
            label="Increment",
        )
        decrement_button = SERVER.gui.add_button(
            label="Decrement",
        )
        reset_button = SERVER.gui.add_button(
            label="Reset",
        )

    # Loop state
    FRAME_IDX = frame_idx_slider.value
    PAUSED = False

    @frame_idx_slider.on_update
    def _(_) -> None:
        nonlocal FRAME_IDX, frame_idx_slider
        FRAME_IDX = int(
            np.clip(frame_idx_slider.value, a_min=0, a_max=len(recorded_data) - 1)
        )
        frame_idx_slider.label = get_frame_idx_slider_text(
            recorded_data=recorded_data, idx=FRAME_IDX
        )

    @pause_toggle_button.on_click
    def _(_) -> None:
        nonlocal PAUSED
        PAUSED = not PAUSED
        if PAUSED:
            pause_toggle_button.label = "Play"
        else:
            pause_toggle_button.label = "Pause"

    @increment_button.on_click
    def _(_) -> None:
        nonlocal PAUSED, frame_idx_slider, pause_toggle_button
        if not PAUSED:
            pause_toggle_button.value = True

        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1)
        )

    @decrement_button.on_click
    def _(_) -> None:
        nonlocal PAUSED, frame_idx_slider, pause_toggle_button
        if not PAUSED:
            pause_toggle_button.value = True

        frame_idx_slider.value = int(
            np.clip(frame_idx_slider.value - 1, a_min=0, a_max=len(recorded_data) - 1)
        )

    @reset_button.on_click
    def _(_) -> None:
        nonlocal frame_idx_slider
        frame_idx_slider.value = 0

    # ###########
    # Main loop
    # ###########
    while True:
        # Get data
        robot_root_state = recorded_data.robot_root_states_array[FRAME_IDX]
        object_root_state = recorded_data.object_root_states_array[FRAME_IDX]
        robot_joint_position = recorded_data.robot_joint_positions_array[FRAME_IDX]

        # ###########
        # Update viser objects
        # ###########
        # Robot
        kuka_allegro_frame.position = robot_root_state[:3]
        kuka_allegro_frame.wxyz = xyzw_to_wxyz(robot_root_state[3:7])
        kuka_allegro_joint_pos_viser_order = RecordedData.change_joint_order(
            robot_joint_position,
            from_order=recorded_data.robot_joint_names,
            to_order=kuka_allegro_viser_joint_names,
        )
        kuka_allegro_viser.update_cfg(kuka_allegro_joint_pos_viser_order)

        # Target robot
        if recorded_data.robot_joint_pos_targets_array is not None:
            robot_joint_pos_target = recorded_data.robot_joint_pos_targets_array[FRAME_IDX]
            target_kuka_allegro_frame.position = robot_root_state[:3]
            target_kuka_allegro_frame.wxyz = xyzw_to_wxyz(robot_root_state[3:7])
            target_kuka_allegro_joint_pos_viser_order = robot_joint_pos_target
            target_kuka_allegro_viser.update_cfg(target_kuka_allegro_joint_pos_viser_order)

        # Object
        object_frame.position = object_root_state[:3]
        object_frame.wxyz = xyzw_to_wxyz(object_root_state[3:7])

        # Table
        if recorded_data.table_root_states_array is not None:
            table_frame.position = recorded_data.table_root_states_array[FRAME_IDX, :3]
            table_frame.wxyz = xyzw_to_wxyz(
                recorded_data.table_root_states_array[FRAME_IDX, 3:7]
            )

        # Goal
        if recorded_data.goal_root_states_array is not None:
            goal_frame.position = recorded_data.goal_root_states_array[FRAME_IDX, :3]
            goal_frame.wxyz = xyzw_to_wxyz(
                recorded_data.goal_root_states_array[FRAME_IDX, 3:7]
            )

        # Palm
        palm_pose_R = kuka_allegro_viser._urdf.get_transform(
            frame_to="allegro_mount"
        ).copy()
        assert palm_pose_R.shape == (
            4,
            4,
        ), f"palm_pose_R.shape: {palm_pose_R.shape}"
        T_R_P = palm_pose_R
        T_W_R = RecordedData.pose_to_T(robot_root_state[:7])
        T_W_P = T_W_R @ T_R_P
        palm_xyz_xyzw = RecordedData.T_to_pose(T_W_P)
        palm_frame.position = palm_xyz_xyzw[:3]
        palm_frame.wxyz = xyzw_to_wxyz(palm_xyz_xyzw[3:7])

        # ###########
        # Sleep and update frame index
        # ###########
        # print(f"Sleeping for {recorded_data.dt} seconds")
        # print(f"recorded_data.time_array: {recorded_data.time_array}")
        # print(f"np.diff(recorded_data.time_array): {np.diff(recorded_data.time_array)}")
        # import matplotlib.pyplot as plt
        # plt.plot(recorded_data.time_array)
        # plt.title("recorded_data.time_array")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Time (s)")
        # plt.show()
        # plt.plot(np.diff(recorded_data.time_array))
        # plt.title("np.diff(recorded_data.time_array)")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Time Difference (s)")
        # plt.show()
        # object_positions = recorded_data.object_root_states_array[:, :3]
        # distances = np.linalg.norm(object_positions[1:] - object_positions[:-1], axis=-1)
        # plt.plot(distances)
        # plt.title("distances")
        # plt.xlabel("Frame Index")
        # plt.ylabel("Distance (m)")
        # plt.show()
        # breakpoint()

        # plt.show()
        # breakpoint()
        # time.sleep(recorded_data.dt)
        time.sleep(1/60)
        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(
                    frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1
                )
            )


if __name__ == "__main__":
    main()
