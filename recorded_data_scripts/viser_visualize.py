from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

from recorded_data_scripts.recorded_data import RecordedData

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
    # ###########
    # Load recorded data
    # ###########
    file_path = Path(
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-43-04.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-19_19-42-41.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-30-37.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-20_14-32-39.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-09.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_16-23-31.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_data/2025-10-27_17-18-32.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-42-11_sin_wave_arm_10-0s_1-0s_0-1rad.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-42-11_sin_wave_arm_10-0s_1-0s_0-1rad_isaac.npz"
        "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-42-11_sin_wave_arm_10-0s_1-0s_0-1rad_isaac_newgains.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-48-58_sin_wave_hand_10-0s_1-0s_0-2rad.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-48-58_sin_wave_hand_10-0s_1-0s_0-2rad_isaac.npz"
        # "/home/tylerlum/github_repos/sapg/recorded_robot_state/2025-11-02_18-48-58_sin_wave_hand_10-0s_1-0s_0-2rad_isaac_newgains.npz"
    )
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
        client.camera.position = (1, 1, 1)
        # client.camera.wxyz = (0, 0, 0, 1)
        client.camera.look_at = (0, 0, 0)

    # Load assets into viser
    KUKA_ALLEGRO_URDF_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
        "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/iiwa14_real.urdf"
    )
    assert KUKA_ALLEGRO_URDF_PATH.exists(), (
        f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
    )
    OBJECT_URDF_PATH = Path(
        # "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf"
        # "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/phone/model.urdf"
        # "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/040_large_marker/040_large_marker.urdf"
        "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/hammer_1/hammer_1.urdf"
    )
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
        "/robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )
    kuka_allegro_viser = ViserUrdf(
        SERVER, KUKA_ALLEGRO_URDF_PATH, root_node_name="/robot/state"
    )

    # Target robot
    if recorded_data.robot_joint_pos_targets_array is not None:
        target_kuka_allegro_frame = SERVER.scene.add_frame(
            "/target_robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
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

    # Palm
    palm_frame = SERVER.scene.add_frame(
        "/robot_palm", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )

    # Floating allegro hand
    allegro_frame = SERVER.scene.add_frame(
        "/allegro", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS
    )
    allegro_viser = ViserUrdf(SERVER, ALLEGRO_URDF_PATH, root_node_name="/allegro")

    # Object relative to floating allegro hand
    object_in_allegro_frame = SERVER.scene.add_frame(
        "/allegro/object",
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )
    _object_in_allegro_viser = ViserUrdf(
        SERVER, OBJECT_URDF_PATH, root_node_name="/allegro/object"
    )

    # Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
    kuka_allegro_viser_joint_names = kuka_allegro_viser._urdf.actuated_joint_names
    allegro_viser_joint_names = allegro_viser._urdf.actuated_joint_names

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
        start_loop_time = time.time()

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
        kuka_allegro_joint_pos_viser_order = robot_joint_position
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

        # Floating allegro hand
        allegro_joint_pos_viser_order = robot_joint_position[7:]
        allegro_viser.update_cfg(allegro_joint_pos_viser_order)

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

        # By default MOVE_FLOATING_ALLEGRO_HAND = False so we can see how the object is moving wrt a fixed allegro hand
        # Can set to True to debug and make sure that everything aligns
        MOVE_FLOATING_ALLEGRO_HAND = False
        if MOVE_FLOATING_ALLEGRO_HAND:
            allegro_frame.position = palm_xyz_xyzw[:3]
            allegro_frame.wxyz = xyzw_to_wxyz(palm_xyz_xyzw[3:7])
        else:
            # Keep floating allegro hand in a fixed position
            allegro_frame.position = recorded_data.robot_root_states_array[
                0, :3
            ] + np.array([0.5, -0.8, 0.7])
            allegro_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

        # Object relative to floating allegro hand
        T_W_O = RecordedData.pose_to_T(object_root_state[:7])
        T_P_W = np.linalg.inv(T_W_P)
        T_P_O = T_P_W @ T_W_O
        object_xyz_xyzw_P = RecordedData.T_to_pose(T_P_O)
        object_in_allegro_frame.position = object_xyz_xyzw_P[:3]
        object_in_allegro_frame.wxyz = xyzw_to_wxyz(object_xyz_xyzw_P[3:7])

        # ###########
        # Sleep and update frame index
        # ###########
        end_loop_time = time.time()
        loop_dt = end_loop_time - start_loop_time
        sleep_dt = recorded_data.dt - loop_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            print(f"Loop too slow! Desired FPS: {1.0 / recorded_data.dt:.1f}, Actual FPS: {1.0 / loop_dt:.1f}")
        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(
                    frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1
                )
            )


if __name__ == "__main__":
    main()
