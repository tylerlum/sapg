import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

# Constants
AXES_LENGTH = 0.1
AXES_RADIUS = 0.01


def xyzw_to_wxyz(xyzw: np.ndarray) -> np.ndarray:
    assert xyzw.shape[-1] == 4, f"Expected xyzw to be (..., 4), got {xyzw.shape}"
    return xyzw[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(wxyz: np.ndarray) -> np.ndarray:
    assert wxyz.shape[-1] == 4, f"Expected wxyz to be (..., 4), got {wxyz.shape}"
    return wxyz[..., [1, 2, 3, 0]]


def main():
    # #########################################################
    # Create scene
    # #########################################################

    # Create server
    server = viser.ViserServer()

    @server.on_client_connect
    def _(client):
        DIST = 0.5
        client.camera.position = (DIST, DIST, DIST)
        client.camera.look_at = (0.0, 0.0, 0.0)

    # Create ground
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Create movable sphere frame
    sphere_frame = server.scene.add_transform_controls(
        name="/sphere_frame",
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
        scale=AXES_LENGTH * 2,
    )

    # Load sphere
    sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=2)
    sphere_vis = server.scene.add_mesh_simple(
        name="/sphere_frame/sphere",
        vertices=sphere.vertices,
        faces=sphere.faces,
    )

    # Store the output sphere vis handle so that we can remove it when saving a new sphere
    output_sphere_vis = None

    # #########################################################
    # Add controls
    # #########################################################
    with server.gui.add_folder("Scale Slider"):
        # Sphere scale state
        CURRENT_SPHERE_SCALE = 1.0
        LATEST_SPHERE_SCALE = CURRENT_SPHERE_SCALE
        scale_slider = server.gui.add_slider(
            label="Scale",
            min=0.01,
            max=10.0,
            step=0.01,
            initial_value=CURRENT_SPHERE_SCALE,
        )

    with server.gui.add_folder("Visibility"):
        show_sphere_frame_cb = server.gui.add_checkbox(
            label="Show sphere frame",
            initial_value=True,
        )

    with server.gui.add_folder("Buttons"):
        # Reset button resets the sphere frame to the origin
        reset_button = server.gui.add_button(
            label="Reset",
        )

        # Save button saves the sphere frame to the output sphere path
        save_button = server.gui.add_button(
            label="Save",
        )

    @sphere_frame.on_update
    def _(_):
        print(f"sphere_frame.position: {sphere_frame.position}, sphere_frame.wxyz: {sphere_frame.wxyz}")

    @show_sphere_frame_cb.on_update
    def _(_):
        sphere_frame.active_axes = (show_sphere_frame_cb.value, show_sphere_frame_cb.value, show_sphere_frame_cb.value)

    @reset_button.on_click
    def _(_):
        sphere_frame.position = (0, 0, 0)
        sphere_frame.wxyz = (1, 0, 0, 0)

    @save_button.on_click
    def _(_):
        # Save the mesh frame to the output mesh path
        position = sphere_frame.position
        wxyz = sphere_frame.wxyz
        xyzw = wxyz_to_xyzw(wxyz)
        T = np.eye(4)
        T[:3, 3] = position
        T[:3, :3] = R.from_quat(xyzw).as_matrix()

        # Must apply scale before applying transform so that the transform is not scaled
        # Must do deepcopy so that the original mesh is not modified
        output_sphere = deepcopy(sphere).apply_scale(LATEST_SPHERE_SCALE).apply_transform(T)

        from datetime import datetime
        output_sphere_path = Path(__file__).parent / f"output_sphere_{datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
        output_sphere_path.parent.mkdir(parents=True, exist_ok=True)
        output_sphere.export(output_sphere_path)
        print(f"Saved output sphere to {output_sphere_path}")

        GREEN_RGB = (0, 255, 0)
        nonlocal output_sphere_vis
        if output_sphere_vis is not None:
            print("Removing previous output sphere from scene")
            output_sphere_vis.remove()
            output_sphere_vis = None

        output_sphere_vis = server.scene.add_mesh_simple(
            name="/output_sphere",
            vertices=output_sphere.vertices,
            faces=output_sphere.faces,
            color=GREEN_RGB,
            opacity=0.5,
        )
        print("Added output sphere to scene")

    @scale_slider.on_update
    def _(_):
        nonlocal LATEST_SPHERE_SCALE
        LATEST_SPHERE_SCALE = scale_slider.value

        # Update scale value, but don't update the sphere mesh or else it may happen too often

    while True:
        if LATEST_SPHERE_SCALE != CURRENT_SPHERE_SCALE:
            print(f"Updating sphere scale from {CURRENT_SPHERE_SCALE} to {LATEST_SPHERE_SCALE}")
            CURRENT_SPHERE_SCALE = LATEST_SPHERE_SCALE
            sphere_vis._queue_update("vertices", sphere_vis.vertices * CURRENT_SPHERE_SCALE)

        print(
            f"sphere_frame.position: {sphere_frame.position}, sphere_frame.wxyz: {sphere_frame.wxyz}"
        )
        time.sleep(0.01)




if __name__ == "__main__":
    main()
