from __future__ import annotations
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from recorded_data_scripts.recorded_data import RecordedData
import viser
from viser.extras import ViserUrdf

def main():
    # ###########
    # Load recorded data
    # ###########
    file_path = Path("/home/tylerlum/github_repos/sapg/recorded_data/2025-10-15_20-52-19.npz")
    assert file_path.exists(), f"File {file_path} does not exist"
    recorded_data = RecordedData.from_file(file_path)

    # ###########
    # Create server
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

    # ###########
    # Constants
    # ###########
    AXES_LENGTH = 0.2
    AXES_RADIUS = 0.01

    # ###########
    # Load assets into viser
    # ###########

    KUKA_ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf")
    assert KUKA_ALLEGRO_URDF_PATH.exists(), f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
    OBJECT_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf")
    assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
    ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf")
    assert ALLEGRO_URDF_PATH.exists(), f"ALLEGRO_URDF_PATH not found: {ALLEGRO_URDF_PATH}"

    kuka_allegro_frame = SERVER.scene.add_frame("/robot/state", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
    kuka_allegro_viser = ViserUrdf(
        SERVER, KUKA_ALLEGRO_URDF_PATH, root_node_name="/robot/state"
    )
    object_frame = SERVER.scene.add_frame("/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
    _object_viser = ViserUrdf(
        SERVER, OBJECT_URDF_PATH, root_node_name="/object"
    )

    palm_frame = SERVER.scene.add_frame("/robot_palm", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)

    # Keep floating allegro hand in place on the right
    allegro_frame = SERVER.scene.add_frame("/allegro", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
    allegro_viser = ViserUrdf(
        SERVER, ALLEGRO_URDF_PATH, root_node_name="/allegro"
    )

    # Place an object relative to the floating allegro hand
    object_in_allegro_frame = SERVER.scene.add_frame("/allegro/object", show_axes=True, axes_length=AXES_LENGTH, axes_radius=AXES_RADIUS)
    _object_in_allegro_viser = ViserUrdf(
        SERVER, OBJECT_URDF_PATH, root_node_name="/allegro/object"
    )

    # Initialize allegro frame position
    allegro_frame.position = recorded_data.robot_root_states_array[0, :3] + np.array([0.5, 0, 0])
    allegro_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

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
        FRAME_IDX = int(np.clip(frame_idx_slider.value, a_min=0, a_max=len(recorded_data) - 1))
        frame_idx_slider.label = get_frame_idx_slider_text(recorded_data=recorded_data, idx=FRAME_IDX)

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

        # Update state
        kuka_allegro_frame.position = robot_root_state[:3]
        kuka_allegro_frame.wxyz = robot_root_state[3:7][[3, 0, 1, 2]]
        object_frame.position = object_root_state[:3]
        object_frame.wxyz = object_root_state[3:7][[3, 0, 1, 2]]
        kuka_allegro_joint_pos_viser_order = RecordedData.change_joint_order(
            robot_joint_position,
            from_order=recorded_data.robot_joint_names,
            to_order=kuka_allegro_viser_joint_names,
        )
        kuka_allegro_viser.update_cfg(kuka_allegro_joint_pos_viser_order)

        allegro_joint_pos_viser_order = RecordedData.change_joint_order(
            robot_joint_position,
            from_order=recorded_data.robot_joint_names,
            to_order=allegro_viser_joint_names + list(set(recorded_data.robot_joint_names) - set(allegro_viser_joint_names)),
        )[:len(allegro_viser_joint_names)]
        allegro_viser.update_cfg(allegro_joint_pos_viser_order)

        # Visualize the palm of the robot
        palm_pose_R = kuka_allegro_viser._urdf.get_transform(frame_to="allegro_mount").copy()
        assert palm_pose_R.shape == (
            4,
            4,
        ), f"palm_pose_R.shape: {palm_pose_R.shape}"
        T_R_P = palm_pose_R
        T_W_R = RecordedData.pose_to_T(robot_root_state[:7])
        T_W_P = T_W_R @ T_R_P
        palm_xyz_xyzw_W = RecordedData.T_to_pose(T_W_P)
        palm_frame.position = palm_xyz_xyzw_W[:3]
        palm_frame.wxyz = palm_xyz_xyzw_W[3:7][[3, 0, 1, 2]]

        # By default MOVE_FLOATING_ALLEGRO_HAND = False so we can see how the object is moving wrt a fixed allegro hand
        # Can set to True to debug and make sure that everything aligns
        MOVE_FLOATING_ALLEGRO_HAND = False
        if MOVE_FLOATING_ALLEGRO_HAND:
            allegro_frame.position = palm_xyz_xyzw_W[:3]
            allegro_frame.wxyz = palm_xyz_xyzw_W[3:7][[3, 0, 1, 2]]

        # # Get object pose wrt palm pose
        T_W_O = RecordedData.pose_to_T(object_root_state[:7])
        T_P_W = np.linalg.inv(T_W_P)
        T_P_O = T_P_W @ T_W_O
        object_xyz_xyzw_P = RecordedData.T_to_pose(T_P_O)
        object_in_allegro_frame.position = object_xyz_xyzw_P[:3]
        object_in_allegro_frame.wxyz = object_xyz_xyzw_P[3:7][[3, 0, 1, 2]]

        # Sleep and update state
        time.sleep(recorded_data.dt)
        if not PAUSED:
            frame_idx_slider.value = int(
                np.clip(frame_idx_slider.value + 1, a_min=0, a_max=len(recorded_data) - 1)
            )



if __name__ == "__main__":
    main()