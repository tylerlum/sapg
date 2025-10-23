import viser
import mujoco
import time
from viser_xml import load_mj_model
from typing import Literal
from viser_conversions import is_fixed_body, get_body_name, merge_geoms
import numpy as np
import viser.transforms as vtf

def create_mesh_handles(
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    mesh_type: Literal["visual", "collision"],
    visible: bool,
    batch_size: int = 1,
) -> dict[int, viser.BatchedGlbHandle]:
    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}

    for i in range(mj_model.ngeom):
        body_id = mj_model.geom_bodyid[i]
        is_collision = mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0

        # Add geom to body's list if it matches the type we're looking for
        if not any([
            (mesh_type == "collision" and is_collision),
            (mesh_type == "visual" and not is_collision),
        ]):
            continue
        if body_id not in body_geoms:
            body_geoms[body_id] = []
        body_geoms[body_id].append(i)

    handles = {}
    with server.atomic():
        for body_id, geom_indices in body_geoms.items():
            # Skip fixed world geometry
            if is_fixed_body(mj_model, body_id):
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
                batched_wxyzs=np.array([1.0, 0.0, 0.0, 0.0])[None].repeat(batch_size, axis=0),
                batched_positions=np.array([0.0, 0.0, 0.0])[None].repeat(batch_size, axis=0),
                lod=((2.0, lod_ratio),) if lod_ratio < 0.5 else "off",
                visible=visible,
            )
            handles[body_id] = handle

    return handles

def update_mujoco(
    server: viser.ViserServer,
    visual_handles: dict[int, viser.BatchedGlbHandle],
    collision_handles: dict[int, viser.BatchedGlbHandle],
    body_xpos: np.ndarray,
    body_xmat: np.ndarray,
) -> None:
    N, B = body_xpos.shape[:2]
    print(f"N: {N}, B: {B}")
    assert body_xpos.shape == (N, B, 3), f"body_xpos.shape: {body_xpos.shape}, expected: (N, B, 3)"
    assert body_xmat.shape == (N, B, 3, 3), f"body_xmat.shape: {body_xmat.shape}, expected: (N, B, 3, 3)"

    with server.atomic():
        body_xquat = vtf.SO3.from_matrix(body_xmat).wxyz

        # Update both visual and collision handles symmetrically
        for handles_dict in [visual_handles, collision_handles]:
            for body_id, handle in handles_dict.items():
                # Skip if handle is not visible
                if not handle.visible:
                    continue

                # Show all environments - apply scene_offset to all of them
                handle.batched_positions = body_xpos[..., body_id, :]
                handle.batched_wxyzs = body_xquat[..., body_id, :]

    server.flush()

IIWA_INIT_JOINT_POS = np.array([-1.571, 1.571, -0.000, 1.376, -0.000, 1.485, 2.358])
IIWA_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',]
IIWA_ACTUATOR_NAMES = ['actuator1', 'actuator2', 'actuator3', 'actuator4', 'actuator5', 'actuator6', 'actuator7',]
assert len(IIWA_INIT_JOINT_POS) == len(IIWA_JOINT_NAMES) == len(IIWA_ACTUATOR_NAMES) == 7, f"len(IIWA_INIT_JOINT_POS): {len(IIWA_INIT_JOINT_POS)}, len(IIWA_JOINT_NAMES): {len(IIWA_JOINT_NAMES)}, len(IIWA_ACTUATOR_NAMES): {len(IIWA_ACTUATOR_NAMES)}, expected: 7"

ALLEGRO_INIT_JOINT_POS = np.zeros(16)
ALLEGRO_INIT_JOINT_POS[12] = 0.3
ALLEGRO_JOINT_NAMES = ['palmffj0', 'palmffj1', 'palmffj2', 'palmffj3', 'palmmfj0', 'palmmfj1', 'palmmfj2', 'palmmfj3', 'palmrfj0', 'palmrfj1', 'palmrfj2', 'palmrfj3', 'palmthj0', 'palmthj1', 'palmthj2', 'palmthj3']
ALLEGRO_ACTUATOR_NAMES = ['palmffa0', 'palmffa1', 'palmffa2', 'palmffa3', 'palmmfa0', 'palmmfa1', 'palmmfa2', 'palmmfa3', 'palmrfa0', 'palmrfa1', 'palmrfa2', 'palmrfa3', 'palmtha0', 'palmtha1', 'palmtha2', 'palmtha3']
assert len(ALLEGRO_INIT_JOINT_POS) == len(ALLEGRO_JOINT_NAMES) == len(ALLEGRO_ACTUATOR_NAMES) == 16, f"len(ALLEGRO_INIT_JOINT_POS): {len(ALLEGRO_INIT_JOINT_POS)}, len(ALLEGRO_JOINT_NAMES): {len(ALLEGRO_JOINT_NAMES)}, len(ALLEGRO_ACTUATOR_NAMES): {len(ALLEGRO_ACTUATOR_NAMES)}, expected: 16"

INIT_JOINT_POS = np.concatenate([IIWA_INIT_JOINT_POS, ALLEGRO_INIT_JOINT_POS])
JOINT_NAMES = IIWA_JOINT_NAMES + ALLEGRO_JOINT_NAMES
ACTUATOR_NAMES = IIWA_ACTUATOR_NAMES + ALLEGRO_ACTUATOR_NAMES
assert len(INIT_JOINT_POS) == len(JOINT_NAMES) == len(ACTUATOR_NAMES) == 23, f"len(INIT_JOINT_POS): {len(INIT_JOINT_POS)}, len(JOINT_NAMES): {len(JOINT_NAMES)}, len(ACTUATOR_NAMES): {len(ACTUATOR_NAMES)}, expected: 23"


def sim_step(mj_model: mujoco.MjModel, mj_data: mujoco.MjData):
    actuator_ids = [mj_model.actuator(name=name).id for name in ACTUATOR_NAMES]
    for i, actuator_id in enumerate(actuator_ids):
        mj_data.ctrl[actuator_id] = INIT_JOINT_POS[i]
    mujoco.mj_step(mj_model, mj_data)


server = viser.ViserServer()
mj_model, mj_data = load_mj_model()
visual_handles = create_mesh_handles(
    server=server,
    mj_model=mj_model,
    mesh_type="visual",
    visible=False,
)
collision_handles = create_mesh_handles(
    server=server,
    mj_model=mj_model,
    mesh_type="collision",
    visible=True,
)

mujoco.mj_step(mj_model, mj_data)

while True:
    body_names = [mj_model.body(i).name for i in range(mj_model.nbody)]
    body_ids = [mj_model.body(name=name).id for name in body_names]
    body_xpositions = [mj_data.xpos[body_id] for body_id in body_ids]
    body_xquats = [mj_data.xquat[body_id] for body_id in body_ids]
    body_xmats = [mj_data.xmat[body_id].reshape(3, 3) for body_id in body_ids]
    update_mujoco(
        server=server,
        visual_handles=visual_handles,
        collision_handles=collision_handles,
        body_xpos=np.array(body_xpositions)[None],
        body_xmat=np.array(body_xmats)[None],
    )
    sim_step(mj_model, mj_data)
    dt = 1/60
    print(f"Sleeping for {dt} seconds")
    time.sleep(dt)