from __future__ import annotations
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from recorded_data_scripts.recorded_data import RecordedData
import rerun as rr
import yourdfpy
from datetime import timedelta

def build_joint_paths(urdf: yourdfpy.URDF, prefix: str) -> dict[str, str]:
    all_joints = urdf.joint_map  # include revolute + fixed + others
    rev = {n: j for n, j in all_joints.items() if j.type == "revolute"}  # final keys
    child_to_joint = {j.child: j for j in all_joints.values()}

    cache: dict[str, str] = {}

    def build(child: str) -> str:
        if child in cache:
            return cache[child]
        j = child_to_joint.get(child)
        if j is None:
            path = child  # root link
        else:
            path = f"{build(j.parent)}/{j.name}/{child}"
        cache[child] = path
        return path

    # Only return revolute joints, but paths include all intermediate links/joints
    return {name: f"{prefix}/{build(j.child)}" for name, j in rev.items()}

def update_joints(
    joint_name_to_pos: dict[str, float],
    joint_paths: dict[str, str],
    urdf: yourdfpy.URDF,
):
    for name, pos in joint_name_to_pos.items():
        if name not in joint_paths:
            continue
        path = joint_paths[name]
        j = urdf.joint_map[name]
        axis = j.axis.tolist()
        rr.log(
            path,
            rr.Transform3D(
                rotation_axis_angle=rr.RotationAxisAngle(
                    axis=axis,
                    angle=pos,
                ),
            ),
        )

def update_joints_array(
    joint_name_to_pos_array: dict[str, np.ndarray],
    joint_paths: dict[str, str],
    urdf: yourdfpy.URDF,
    time_array: np.ndarray,
):
    time_indexes = [
        rr.TimeColumn(
            "tick",
            duration=[timedelta(seconds=float(s - time_array[0])) for s in time_array],
        )
    ]
    for name, pos_array in joint_name_to_pos_array.items():
        if name not in joint_paths:
            continue
        path = joint_paths[name]
        j = urdf.joint_map[name]
        axis = j.axis.tolist()
        T = pos_array.shape[0]
        rr.send_columns(
            path,
            indexes=time_indexes,
            columns=[
                rr.Transform3D.columns(
                    rotation_axis_angle=[
                        rr.RotationAxisAngle(
                            axis=axis,
                            angle=pos_array[i],
                        )
                        for i in range(T)
                    ]
                ),
            ],
        )

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
    APPLICATION_ID = "rerun_visualize"
    rr.init(application_id=APPLICATION_ID, spawn=True)
    rr.set_time("tick", duration=timedelta(seconds=0))

    # Add world frame
    AXES_LENGTH = 0.2

    rr.log(
        "world",
        rr.Transform3D(clear=False, axis_length=AXES_LENGTH),
    )

    # ###########
    # Load assets into viser
    # ###########

    KUKA_ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf")
    assert KUKA_ALLEGRO_URDF_PATH.exists(), f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
    OBJECT_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf")
    assert OBJECT_URDF_PATH.exists(), f"OBJECT_URDF_PATH not found: {OBJECT_URDF_PATH}"
    ALLEGRO_URDF_PATH = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf")
    assert ALLEGRO_URDF_PATH.exists(), f"ALLEGRO_URDF_PATH not found: {ALLEGRO_URDF_PATH}"

    kuka_allegro_urdf = yourdfpy.URDF.load(KUKA_ALLEGRO_URDF_PATH)
    rr.log_file_from_path(
        KUKA_ALLEGRO_URDF_PATH,
        entity_path_prefix="/kuka_allegro",
    )

    rr.log_file_from_path(
        OBJECT_URDF_PATH,
        entity_path_prefix="/object",
    )

    rr.log("palm", rr.Transform3D(clear=False, axis_length=AXES_LENGTH))

    # Keep floating allegro hand in place on the right
    rr.log_file_from_path(
        ALLEGRO_URDF_PATH,
        entity_path_prefix="/allegro",
    )

    # Place an object relative to the floating allegro hand
    rr.log_file_from_path(
        OBJECT_URDF_PATH,
        entity_path_prefix="/allegro/object",
    )


    # BRITTLE: This is the most brittle part of the code, since it relies on the urdf structure
    # Need to check that the joint paths created here match what is created in the rerun viewer
    joint_paths = build_joint_paths(kuka_allegro_urdf, prefix="/kuka_allegro/kuka_allegro")

    # Columns is batched and fast
    # Log is easier to use but slow
    from typing import Literal
    MODE: Literal["columns", "log"] = "columns"
    if MODE == "columns":
        time_indexes = [
            rr.TimeColumn(
                "tick",
                duration=[timedelta(seconds=float(s - recorded_data.time_array[0])) for s in recorded_data.time_array],
            )
        ]
        rr.send_columns(
            "kuka_allegro",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    recorded_data.robot_root_states_array[t, :3]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=recorded_data.robot_root_states_array[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
            ),
        )
        rr.send_columns(
            "object",
            indexes=time_indexes,
            columns=rr.Transform3D.columns(
                translation=[
                    recorded_data.object_root_states_array[t, :3]
                    for t in range(len(recorded_data.time_array))
                ],
                quaternion=[
                    rr.Quaternion(
                        xyzw=recorded_data.object_root_states_array[t, 3:7],
                    )
                    for t in range(len(recorded_data.time_array))
                ],
            ),
        )
        joint_name_to_pos_array = {
            name: recorded_data.robot_joint_positions_array[:, i]
            for i, name in enumerate(recorded_data.robot_joint_names)
        }
        update_joints_array(
            joint_name_to_pos_array=joint_name_to_pos_array,
            joint_paths=joint_paths,
            urdf=kuka_allegro_urdf,
            time_array=recorded_data.time_array,
        )
    elif MODE == "log":
        pass
    else:
        raise ValueError(f"Invalid mode: {MODE}")

    # Initialize allegro frame position
    breakpoint()
    allegro_frame.position = recorded_data.robot_root_states_array[0, :3] + np.array([0.5, 0, 0])
    allegro_frame.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # Get joint names since the ordering of the urdf may not match the ordering of the robot_joint_names
    kuka_allegro_viser_joint_names = kuka_allegro_viser._urdf.actuated_joint_names
    allegro_viser_joint_names = allegro_viser._urdf.actuated_joint_names

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

    breakpoint()


if __name__ == "__main__":
    main()
