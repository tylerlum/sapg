import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import viser
import viser.transforms as vtf
from viser_conversions import get_body_name, is_fixed_body, merge_geoms

import mujoco


class ViserMJModel:
    # ############################################################
    # Initialization
    # ############################################################
    def __init__(
        self,
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ):
        self.server = server
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.visual_handles = self._create_mesh_handles(
            mesh_type="visual", visible=True
        )
        self.collision_handles = self._create_mesh_handles(
            mesh_type="collision", visible=False
        )
        self._print_model_info()

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

    def _create_mesh_handles(
        self, mesh_type: Literal["visual", "collision"], visible: bool
    ) -> dict[int, viser.BatchedGlbHandle]:
        return self._create_mesh_handles_static(
            server=self.server,
            mj_model=self.mj_model,
            mesh_type=mesh_type,
            visible=visible,
        )

    @staticmethod
    def _create_mesh_handles_static(
        server: viser.ViserServer,
        mj_model: mujoco.MjModel,
        mesh_type: Literal["visual", "collision"],
        visible: bool,
        batch_size: int = 1,
    ) -> dict[int, viser.BatchedGlbHandle]:
        assert batch_size == 1, f"batch_size: {batch_size}, expected: 1 for now"

        # Group geoms by body
        body_geoms: defaultdict[int, list[int]] = defaultdict(list)

        for i in range(mj_model.ngeom):
            body_id = mj_model.geom_bodyid[i]
            is_collision = (
                mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0
            )

            # Add geom to body's list if it matches the type we're looking for
            if (mesh_type == "collision" and is_collision) or (
                mesh_type == "visual" and not is_collision
            ):
                body_geoms[body_id].append(i)

        handles = {}
        with server.atomic():
            for body_id, geom_indices in body_geoms.items():
                # Skip fixed world geometry
                if is_fixed_body(mj_model, body_id):
                    # We include fixed geometry to see things like the first link of robot arm
                    INCLUDE_FIXED_GEOMETRY = True
                    if not INCLUDE_FIXED_GEOMETRY:
                        continue

                # Get body name
                body_name = get_body_name(mj_model, body_id)

                # Merge geoms into a single mesh
                mesh = merge_geoms(mj_model, geom_indices)
                lod_ratio = 1000.0 / mesh.vertices.shape[0]

                # Create handle
                handle = server.scene.add_batched_meshes_trimesh(
                    f"/bodies/{body_name}/{mesh_type}",
                    mesh,
                    batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(
                        batch_size, axis=0
                    ),
                    batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(
                        batch_size, axis=0
                    ),
                    lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
                    visible=visible,
                )
                handles[body_id] = handle

        return handles

    # ############################################################
    # Updating viser
    # ############################################################
    def _update_viser(
        self,
    ) -> None:
        body_xpos = self.mj_data.xpos
        body_xmat = self.mj_data.xmat
        body_xmat = body_xmat.reshape(body_xmat.shape[0], 3, 3)
        self._update_viser_static(
            server=self.server,
            visual_handles=self.visual_handles,
            collision_handles=self.collision_handles,
            body_xpos=body_xpos[None],
            body_xmat=body_xmat[None],
        )

    @staticmethod
    def _update_viser_static(
        server: viser.ViserServer,
        visual_handles: dict[int, viser.BatchedGlbHandle],
        collision_handles: dict[int, viser.BatchedGlbHandle],
        body_xpos: np.ndarray,
        body_xmat: np.ndarray,
    ) -> None:
        batch_size, num_bodies = body_xpos.shape[:2]
        assert batch_size == 1, f"batch_size: {batch_size}, expected: 1 for now"
        assert body_xpos.shape == (batch_size, num_bodies, 3), (
            f"body_xpos.shape: {body_xpos.shape}, expected: ({batch_size}, {num_bodies}, 3)"
        )
        assert body_xmat.shape == (batch_size, num_bodies, 3, 3), (
            f"body_xmat.shape: {body_xmat.shape}, expected: ({batch_size}, {num_bodies}, 3, 3)"
        )

        with server.atomic():
            body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

            # Update both visual and collision handles symmetrically
            for handles_dict in [visual_handles, collision_handles]:
                for body_id, handle in handles_dict.items():
                    # Skip if handle is not visible
                    if not handle.visible:
                        continue

                    # Show all environments
                    handle.batched_positions = body_xpos[..., body_id, :]
                    handle.batched_wxyzs = body_xquat[..., body_id, :]

        server.flush()

    def update_cfg(
        self, q_dict: dict[str, float], actuator_dict: dict[str, float] | None = None
    ) -> None:
        assert set(q_dict.keys()).issubset(set(self.joint_names)), (
            f"q_dict.keys(): {q_dict.keys()}, expected subset of {self.joint_names} for joint positions"
        )

        # Set joint positions
        for name, value in q_dict.items():
            joint_id = self.mj_model.joint(name=name).id
            self.mj_data.qpos[joint_id] = value

        # Set actuator positions
        if actuator_dict is not None:
            assert set(actuator_dict.keys()).issubset(set(self.actuator_names)), (
                f"actuator_dict.keys(): {actuator_dict.keys()}, expected subset of {self.actuator_names} for actuator commands"
            )
            for name, value in actuator_dict.items():
                actuator_id = self.mj_model.actuator(name=name).id
                self.mj_data.ctrl[actuator_id] = value

        # Step simulation to update the state
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Update viser
        self._update_viser()

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

    def get_sim_state(
        self,
    ) -> tuple[
        dict[str, tuple[np.ndarray, np.ndarray]], dict[str, float], dict[str, float]
    ]:
        body_dict = {name: self.get_body_pose(name) for name in self.body_names}
        joint_dict = {
            name: self.mj_data.qpos[joint_id]
            for name, joint_id in zip(self.joint_names, self.joint_ids)
        }
        actuator_dict = {
            name: self.mj_data.ctrl[actuator_id]
            for name, actuator_id in zip(self.actuator_names, self.actuator_ids)
        }
        return {
            "body_dict": body_dict,
            "joint_dict": joint_dict,
            "actuator_dict": actuator_dict,
        }

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
    iiwa_xml_path = Path(
        "/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml"
    )
    assert iiwa_xml_path.exists(), f"IIWA XML path does not exist: {iiwa_xml_path}"
    mj_model = mujoco.MjModel.from_xml_path(str(iiwa_xml_path))
    mj_data = mujoco.MjData(mj_model)
    server = viser.ViserServer()
    viser_mj_model = ViserMJModel(
        server=server,
        mj_model=mj_model,
        mj_data=mj_data,
    )
    joint_names = viser_mj_model.joint_names
    viser_mj_model.update_cfg(q_dict={
        name: 0.0 for name in joint_names
    })

    # Sleep forever.
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
