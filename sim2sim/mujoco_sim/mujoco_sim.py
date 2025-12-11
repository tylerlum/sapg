from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from isaacgymenvs.utils.utils import get_repo_root_dir

# ############################################################
# Constants
# ############################################################
N_IIWA_JOINTS = 7
IIWA_INIT_JOINT_POS = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])
IIWA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
IIWA_ACTUATOR_NAMES = [
    "actuator1",
    "actuator2",
    "actuator3",
    "actuator4",
    "actuator5",
    "actuator6",
    "actuator7",
]
assert (
    len(IIWA_INIT_JOINT_POS) == len(IIWA_JOINT_NAMES) == len(IIWA_ACTUATOR_NAMES) == 7
), (
    f"len(IIWA_INIT_JOINT_POS): {len(IIWA_INIT_JOINT_POS)}, len(IIWA_JOINT_NAMES): {len(IIWA_JOINT_NAMES)}, len(IIWA_ACTUATOR_NAMES): {len(IIWA_ACTUATOR_NAMES)}, expected: 7"
)

N_ALLEGRO_JOINTS = 16
ALLEGRO_INIT_JOINT_POS = np.zeros(16)
ALLEGRO_INIT_JOINT_POS[12] = 0.3
ALLEGRO_JOINT_NAMES = [
    "palmffj0",
    "palmffj1",
    "palmffj2",
    "palmffj3",
    "palmmfj0",
    "palmmfj1",
    "palmmfj2",
    "palmmfj3",
    "palmrfj0",
    "palmrfj1",
    "palmrfj2",
    "palmrfj3",
    "palmthj0",
    "palmthj1",
    "palmthj2",
    "palmthj3",
]
ALLEGRO_ACTUATOR_NAMES = [
    "palmffa0",
    "palmffa1",
    "palmffa2",
    "palmffa3",
    "palmmfa0",
    "palmmfa1",
    "palmmfa2",
    "palmmfa3",
    "palmrfa0",
    "palmrfa1",
    "palmrfa2",
    "palmrfa3",
    "palmtha0",
    "palmtha1",
    "palmtha2",
    "palmtha3",
]
assert (
    len(ALLEGRO_INIT_JOINT_POS)
    == len(ALLEGRO_JOINT_NAMES)
    == len(ALLEGRO_ACTUATOR_NAMES)
    == 16
), (
    f"len(ALLEGRO_INIT_JOINT_POS): {len(ALLEGRO_INIT_JOINT_POS)}, len(ALLEGRO_JOINT_NAMES): {len(ALLEGRO_JOINT_NAMES)}, len(ALLEGRO_ACTUATOR_NAMES): {len(ALLEGRO_ACTUATOR_NAMES)}, expected: 16"
)

N_JOINTS = N_IIWA_JOINTS + N_ALLEGRO_JOINTS
INIT_JOINT_POS = np.concatenate([IIWA_INIT_JOINT_POS, ALLEGRO_INIT_JOINT_POS])
JOINT_NAMES = IIWA_JOINT_NAMES + ALLEGRO_JOINT_NAMES
ACTUATOR_NAMES = IIWA_ACTUATOR_NAMES + ALLEGRO_ACTUATOR_NAMES
assert len(INIT_JOINT_POS) == len(JOINT_NAMES) == len(ACTUATOR_NAMES) == N_JOINTS, (
    f"len(INIT_JOINT_POS): {len(INIT_JOINT_POS)}, len(JOINT_NAMES): {len(JOINT_NAMES)}, len(ACTUATOR_NAMES): {len(ACTUATOR_NAMES)}, expected: {N_JOINTS}"
)

N_BODY_NAMES = 33
BODY_NAMES = [
    "world",
    "base",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "palmworld",
    "palmpalm",
    "palmff_base",
    "palmff_proximal",
    "palmff_medial",
    "palmff_distal",
    "palmff_tip",
    "palmmf_base",
    "palmmf_proximal",
    "palmmf_medial",
    "palmmf_distal",
    "palmmf_tip",
    "palmrf_base",
    "palmrf_proximal",
    "palmrf_medial",
    "palmrf_distal",
    "palmrf_tip",
    "palmth_base",
    "palmth_proximal",
    "palmth_medial",
    "palmth_distal",
    "palmth_tip",
    "table",
    "object",
]
assert len(BODY_NAMES) == N_BODY_NAMES, (
    f"len(BODY_NAMES): {len(BODY_NAMES)}, expected: {N_BODY_NAMES}"
)

# ############################################################
# Config
# ############################################################


@dataclass
class FrictionConfig:
    sliding_friction: float = 1.0
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001


@dataclass
class MujocoSimConfig:
    enable_viewer: bool
    sim_dt: float = 1.0 / 1000  # Need a high enough frequency to get stable physics
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    object_name: str = "044_flat_screwdriver"

    @property
    def sim_hz(self) -> float:
        return 1.0 / self.sim_dt

    @property
    def friction_array(self) -> np.ndarray:
        return np.array(
            [
                self.friction.sliding_friction,
                self.friction.torsional_friction,
                self.friction.rolling_friction,
            ]
        )


class MujocoSim:
    # ############################################################
    # Initialization
    # ############################################################
    def __init__(self, config: MujocoSimConfig):
        self.config = config
        self._init_scene()
        self.set_robot_joint_pos_targets(INIT_JOINT_POS)
        self.set_robot_joint_positions(INIT_JOINT_POS)

    def _init_scene(self) -> None:
        # Robot
        iiwa_xml_path = get_repo_root_dir() / "assets/mjcf/kuka_iiwa_14/scene.xml"
        assert iiwa_xml_path.exists(), f"Robot path does not exist: {iiwa_xml_path}"

        # Load mjspec from robot path
        spec = mujoco.MjSpec.from_file(str(iiwa_xml_path))
        spec.option.timestep = self.config.sim_dt

        allegro_xml_path = (
            get_repo_root_dir() / "assets/mjcf/wonik_allegro/right_hand_offset.xml"
        )
        assert allegro_xml_path.exists(), (
            f"Allegro XML path does not exist: {allegro_xml_path}"
        )
        allegro_spec = mujoco.MjSpec.from_file(str(allegro_xml_path))
        attachment_site = next(s for s in spec.sites if s.name == "attachment_site")
        attachment_site.attach_body(allegro_spec.worldbody, "palm", "")

        # Enable gravity compensation for robot bodies
        # https://mujoco.readthedocs.io/en/3.1.2/XMLreference.html#body-gravcomp
        for body in spec.bodies:
            body.gravcomp = 1.0

        # Move robot base to desired position
        robot_base_bodies = [body for body in spec.bodies if body.name == "base"]
        assert len(robot_base_bodies) == 1, (
            f"len(robot_base_bodies): {len(robot_base_bodies)}, expected: 1"
        )
        robot_base_body = robot_base_bodies[0]
        robot_base_body.pos = np.array([0.0, 0.8, 0.0])

        # Table
        WHITE_RGBA = np.array([1.0, 1.0, 1.0, 1.0])
        TABLE_LEN_X, TABLE_LEN_Y, TABLE_LEN_Z = 0.475, 0.4, 0.3
        TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z = 0.0, 0.0, 0.38
        table_body = spec.worldbody.add_body()
        table_body.name = "table"
        table_body.pos = np.array([TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z])

        table_geom = table_body.add_geom()
        table_geom.name = "table_geom"
        table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = np.array(
            [TABLE_LEN_X / 2, TABLE_LEN_Y / 2, TABLE_LEN_Z / 2]
        )  # Half extents
        table_geom.rgba = WHITE_RGBA
        table_geom.friction = self.config.friction_array.copy()

        # Object
        GREY_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
        OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z = 0.0, 0.0, 0.38 + 0.3
        object_body = spec.worldbody.add_body()
        object_body.name = "object"
        object_body.pos = np.array([OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z])

        object_free_joint = object_body.add_joint()
        object_free_joint.name = "object_free_joint"
        object_free_joint.type = mujoco.mjtJoint.mjJNT_FREE

        ADD_BOX_OBJECT = False
        if ADD_BOX_OBJECT:
            BOX_LEN_X, BOX_LEN_Y, BOX_LEN_Z = 0.30, 0.03, 0.02

            object_geom = object_body.add_geom()
            object_geom.name = "object_geom"
            object_geom.rgba = GREY_RGBA
            object_geom.friction = self.config.friction_array.copy()

            object_geom.type = mujoco.mjtGeom.mjGEOM_BOX
            object_geom.size = np.array(
                [BOX_LEN_X / 2, BOX_LEN_Y / 2, BOX_LEN_Z / 2]
            )  # Half extents
        else:
            # Use list of convex decomp meshes for object
            # Use run_coacd.py to generate convex decomp meshes

            from isaacgymenvs.utils.objects import NAME_TO_OBJECT

            object_name = self.config.object_name
            mesh_paths = NAME_TO_OBJECT[object_name].coacd_filepaths
            assert mesh_paths is not None, (
                f"mesh_paths is None for object_name: {object_name}"
            )
            assert len(mesh_paths) > 0, (
                f"len(mesh_paths) is 0 for object_name: {object_name}"
            )
            # mesh_paths = list((get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/044_flat_screwdriver").glob("decomp_*.obj"))

            for mesh_path in mesh_paths:
                assert mesh_path.exists(), f"Mesh file does not exist: {mesh_path}"
                mesh = spec.add_mesh()
                mesh.name = mesh_path.stem
                mesh.file = str(mesh_path)
                assert Path(mesh.file).exists(), (
                    f"Mesh file does not exist: {mesh.file}"
                )
                mesh.scale = np.array([1.0, 1.0, 1.0])

                object_geom = object_body.add_geom()
                object_geom.name = f"object_geom_{mesh_path.stem}"
                object_geom.rgba = GREY_RGBA
                object_geom.friction = self.config.friction_array.copy()
                object_geom.type = mujoco.mjtGeom.mjGEOM_MESH
                object_geom.meshname = mesh.name

        DISABLE_ROBOT_SELF_COLLISION = False
        if DISABLE_ROBOT_SELF_COLLISION:
            for geom in spec.geoms:
                if "iiwa" in geom.name or "palm" in geom.name or "finger" in geom.name:
                    geom.contype = 0
                    geom.conaffinity = 0

        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.config.sim_dt
        if self.config.enable_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self._print_model_info()
        self._validate()

    def _print_model_info(self) -> None:
        print()
        print("Model info:")
        print(f"  Number of joints (njnt): {self.mj_model.njnt}")
        print(f"  Number of DOFs (nv): {self.mj_model.nv}")
        print(f"  Number of actuators (nu): {self.mj_model.nu}")
        print(f"  data.qpos shape: {self.mj_data.qpos.shape}")
        print(f"  data.qvel shape: {self.mj_data.qvel.shape}")
        print(f"  data.ctrl shape: {self.mj_data.ctrl.shape}")
        print(f"  Joint names: {self.joint_names}")
        print(f"  Actuator names: {self.actuator_names}")
        print(f"  Body names: {self.body_names}")
        print()

    def _validate(self) -> None:
        assert JOINT_NAMES == self.joint_names[:N_JOINTS], (
            f"JOINT_NAMES: {JOINT_NAMES}, self.joint_names: {self.joint_names[:N_JOINTS]}"
        )
        assert ACTUATOR_NAMES == self.actuator_names[:N_JOINTS], (
            f"ACTUATOR_NAMES: {ACTUATOR_NAMES}, self.actuator_names: {self.actuator_names[:N_JOINTS]}"
        )
        assert BODY_NAMES == self.body_names, (
            f"BODY_NAMES: {BODY_NAMES}, self.body_names: {self.body_names}"
        )

    # ############################################################
    # Setting robot joint positions and targets
    # ############################################################
    def set_robot_joint_positions(self, q: np.ndarray) -> None:
        assert q.shape == (N_JOINTS,), f"q.shape: {q.shape}, expected: ({N_JOINTS},)"
        for i, joint_name in enumerate(JOINT_NAMES):
            joint_id = self.mj_model.joint(name=joint_name).id
            self.mj_data.qpos[joint_id] = q[i]

    def set_robot_joint_pos_targets(self, q_targets: np.ndarray) -> None:
        assert q_targets.shape == (N_JOINTS,), (
            f"q_targets.shape: {q_targets.shape}, expected: ({N_JOINTS},)"
        )
        self.robot_joint_pos_targets = q_targets.copy()

    # ############################################################
    # Setting object position and orientation
    # ############################################################
    def set_object_position(self, pos: np.ndarray) -> None:
        assert pos.shape == (3,), f"pos.shape: {pos.shape}, expected: (3,)"
        qpos_adr = self.mj_model.joint(name="object_free_joint").qposadr[0]
        self.mj_data.qpos[qpos_adr : qpos_adr + 3] = pos

    def set_object_quat_wxyz(self, quat_wxyz: np.ndarray) -> None:
        assert quat_wxyz.shape == (4,), (
            f"quat_wxyz.shape: {quat_wxyz.shape}, expected: (4,)"
        )
        qpos_adr = self.mj_model.joint(name="object_free_joint").qposadr[0]
        self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz

    # ############################################################
    # Getting body poses and simulation state
    # ############################################################
    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self.mj_model.body(name=body_name).id
        pos = self.mj_data.xpos[body_id]  # (3,) world position of body frame
        quat_wxyz = self.mj_data.xquat[
            body_id
        ]  # (4,) world orientation quaternion (w, x, y, z)
        return pos, quat_wxyz

    def get_sim_state(self) -> dict[str, np.ndarray]:
        # Usage:
        table_pos, table_quat_wxyz = self.get_body_pose("table")
        object_pos, object_quat_wxyz = self.get_body_pose("object")
        robot_base_pos, robot_base_quat_wxyz = self.get_body_pose(
            "base"
        )  # replace with actual base body name
        joint_ids = [self.mj_model.joint(name=name).id for name in JOINT_NAMES]
        joint_positions = self.mj_data.qpos[joint_ids]
        joint_velocities = self.mj_data.qvel[joint_ids]
        return {
            "table_pos": table_pos,
            "table_quat_wxyz": table_quat_wxyz,
            "object_pos": object_pos,
            "object_quat_wxyz": object_quat_wxyz,
            "robot_base_pos": robot_base_pos,
            "robot_base_quat_wxyz": robot_base_quat_wxyz,
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
        }

    # ############################################################
    # Stepping simulation
    # ############################################################
    def sim_step(self):
        actuator_ids = [self.mj_model.actuator(name=name).id for name in ACTUATOR_NAMES]
        for i, actuator_id in enumerate(actuator_ids):
            self.mj_data.ctrl[actuator_id] = self.robot_joint_pos_targets[i]
        mujoco.mj_step(self.mj_model, self.mj_data)

    # ############################################################
    # Getting body poses and simulation state
    # ############################################################
    def _continue_running(self) -> bool:
        if self.config.enable_viewer:
            return self.viewer.is_running()
        else:
            return True

    def run(self):
        loop_no_sleep_dts, loop_dts = [], []

        while self._continue_running():
            start_loop_no_sleep_time = time.time()

            # Step simulation
            self.sim_step()

            # Get simulation state
            sim_state_dict = self.get_sim_state()

            PRINT_SIM_STATE = False
            if PRINT_SIM_STATE:
                for key, value in sim_state_dict.items():
                    print(f"{key}: {value}")
                print()

            # Update viewer
            if self.config.enable_viewer:
                self.viewer.sync()

            # End of loop timekeeping
            end_loop_no_sleep_time = time.time()
            loop_no_sleep_dt = end_loop_no_sleep_time - start_loop_no_sleep_time
            loop_no_sleep_dts.append(loop_no_sleep_dt)

            sleep_dt = self.config.sim_dt - loop_no_sleep_dt
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dt = loop_no_sleep_dt + sleep_dt
            else:
                loop_dt = loop_no_sleep_dt
                print(
                    f"Simulation is running slower than real time, desired FPS = {1.0 / self.config.sim_dt:.1f}, actual FPS = {1.0 / loop_dt:.1f}"
                )
            loop_dts.append(loop_dt)

            # Get FPS
            PRINT_FPS_EVERY_N_SECONDS = 5.0
            PRINT_FPS_EVERY_N_STEPS = int(
                PRINT_FPS_EVERY_N_SECONDS / self.config.sim_dt
            )
            if len(loop_dts) == PRINT_FPS_EVERY_N_STEPS:
                loop_dt_array = np.array(loop_dts)
                loop_no_sleep_dt_array = np.array(loop_no_sleep_dts)
                fps_array = 1.0 / loop_dt_array
                fps_no_sleep_array = 1.0 / loop_no_sleep_dt_array
                print("FPS with sleep:")
                print(f"  Mean: {np.mean(fps_array):.1f}")
                print(f"  Median: {np.median(fps_array):.1f}")
                print(f"  Max: {np.max(fps_array):.1f}")
                print(f"  Min: {np.min(fps_array):.1f}")
                print(f"  Std: {np.std(fps_array):.1f}")
                print("FPS without sleep:")
                print(f"  Mean: {np.mean(fps_no_sleep_array):.1f}")
                print(f"  Median: {np.median(fps_no_sleep_array):.1f}")
                print(f"  Max: {np.max(fps_no_sleep_array):.1f}")
                print(f"  Min: {np.min(fps_no_sleep_array):.1f}")
                print(f"  Std: {np.std(fps_no_sleep_array):.1f}")
                print()
                loop_no_sleep_dts, loop_dts = [], []

    # ############################################################
    # Properties
    # ############################################################
    @property
    def joint_names(self) -> list[str]:
        return [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]

    @property
    def actuator_names(self) -> list[str]:
        return [self.mj_model.actuator(i).name for i in range(self.mj_model.nu)]

    @property
    def body_names(self) -> list[str]:
        return [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]

    @property
    def joint_ids(self) -> list[int]:
        return [self.mj_model.joint(name=name).id for name in self.joint_names]

    @property
    def actuator_ids(self) -> list[int]:
        return [self.mj_model.actuator(name=name).id for name in self.actuator_names]

    @property
    def body_ids(self) -> list[int]:
        return [self.mj_model.body(name=name).id for name in self.body_names]


def main():
    mujoco_sim_config = MujocoSimConfig(
        enable_viewer=True,
    )
    mujoco_sim = MujocoSim(mujoco_sim_config)
    mujoco_sim.run()


if __name__ == "__main__":
    main()
