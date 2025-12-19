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


def viser_modify_mesh_interactive(input_mesh_path: Path, output_mesh_path: Path):
    # The goal of this function is to allow the user to modify a mesh interactively in viser.
    # More specifically, we will assume a fixed mesh origin frame and allow the user to translate and rotate the mesh wrt that fixed frame.
    # mesh_origin_frame = fixed frame
    # mesh_frame = moving frame that is rigidly attached to the mesh as the user modifies it

    # #########################################################
    # Create scene
    # #########################################################

    # Create server
    server = viser.ViserServer()

    # Create ground
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Create fixed mesh origin frame
    _mesh_origin_frame = server.scene.add_frame(
        name="/mesh_origin_frame",
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
        show_axes=True,
        axes_length=AXES_LENGTH,
        axes_radius=AXES_RADIUS,
    )

    # Create movable mesh frame
    mesh_frame = server.scene.add_transform_controls(
        name="/mesh_origin_frame/mesh_frame",
        position=(0, 0, 0),
        wxyz=(1, 0, 0, 0),
        scale=AXES_LENGTH * 2,
    )

    # Load mesh
    mesh = trimesh.load(input_mesh_path, force="mesh")
    mesh_vis = server.scene.add_mesh_simple(
        name="/mesh_origin_frame/mesh_frame/mesh",
        vertices=mesh.vertices,
        faces=mesh.faces,
    )

    # Store the output mesh vis handle so that we can remove it when saving a new mesh
    output_mesh_vis = None

    # #########################################################
    # Add controls
    # #########################################################
    with server.gui.add_folder("Pose Sliders"):
        # Moving slider moves the frame
        # Moving the frame moves the slider
        # To avoid infinite loop, need a stateful variable that tells us if the update is from user moving slider or user moving frame
        # If the user manually moves the frame, the sliders should update but not tell the frame to update.
        # If the user manually moves the sliders, the sliders should tell the frame to update.
        USER_MOVED_FRAME = False

        x_slider = server.gui.add_slider(
            label="X (m)",
            min=-1.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )
        y_slider = server.gui.add_slider(
            label="Y (m)",
            min=-1.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )
        z_slider = server.gui.add_slider(
            label="Z (m)",
            min=-1.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )
        R_slider = server.gui.add_slider(
            label="R (deg)",
            min=-180,
            max=180,
            step=0.1,
            initial_value=0.0,
        )
        P_slider = server.gui.add_slider(
            label="P (deg)",
            min=-180,
            max=180,
            step=0.1,
            initial_value=0.0,
        )
        Y_slider = server.gui.add_slider(
            label="Y (deg)",
            min=-180,
            max=180,
            step=0.1,
            initial_value=0.0,
        )

    with server.gui.add_folder("Scale Slider"):
        # Mesh scale state
        MESH_SCALE = 1.0
        scale_slider = server.gui.add_slider(
            label="Scale",
            min=0.01,
            max=10.0,
            step=0.01,
            initial_value=1.0,
        )

    with server.gui.add_folder("Visibility"):
        show_mesh_origin_frame_cb = server.gui.add_checkbox(
            label="Show mesh origin frame",
            initial_value=True,
        )
        show_mesh_frame_cb = server.gui.add_checkbox(
            label="Show mesh frame",
            initial_value=True,
        )

    with server.gui.add_folder("Buttons"):
        # Reset button resets the mesh frame to the origin
        reset_button = server.gui.add_button(
            label="Reset",
        )

        # Save button saves the mesh frame to the output mesh path
        save_button = server.gui.add_button(
            label="Save",
        )

    def update_frame_position_from_sliders_if_user_did_not_move_frame():
        if USER_MOVED_FRAME:
            # User moved the frame, so don't update the frame again
            return
        mesh_frame.position = (x_slider.value, y_slider.value, z_slider.value)

    def update_frame_wxyz_from_sliders_if_user_did_not_move_frame():
        if USER_MOVED_FRAME:
            # User moved the frame, so don't update the frame again
            return
        RPY = np.deg2rad(np.array([R_slider.value, P_slider.value, Y_slider.value]))
        xyzw = R.from_euler("xyz", RPY).as_quat()
        wxyz = xyzw_to_wxyz(xyzw)
        mesh_frame.wxyz = wxyz

    @x_slider.on_update
    def _(_):
        update_frame_position_from_sliders_if_user_did_not_move_frame()

    @y_slider.on_update
    def _(_):
        update_frame_position_from_sliders_if_user_did_not_move_frame()

    @z_slider.on_update
    def _(_):
        update_frame_position_from_sliders_if_user_did_not_move_frame()

    @R_slider.on_update
    def _(_):
        update_frame_wxyz_from_sliders_if_user_did_not_move_frame()

    @P_slider.on_update
    def _(_):
        update_frame_wxyz_from_sliders_if_user_did_not_move_frame()

    @Y_slider.on_update
    def _(_):
        update_frame_wxyz_from_sliders_if_user_did_not_move_frame()

    @mesh_frame.on_update
    def _(_):
        nonlocal USER_MOVED_FRAME

        # User moved the frame
        # Update the sliders, but make sure the sliders don't tell the frame to update again
        USER_MOVED_FRAME = True
        x_slider.value = mesh_frame.position[0]
        y_slider.value = mesh_frame.position[1]
        z_slider.value = mesh_frame.position[2]
        xyzw = wxyz_to_xyzw(mesh_frame.wxyz)
        RPY = np.round(np.rad2deg(R.from_quat(xyzw).as_euler("xyz")), 1)
        R_slider.value = RPY[0]
        P_slider.value = RPY[1]
        Y_slider.value = RPY[2]
        USER_MOVED_FRAME = False

    @show_mesh_origin_frame_cb.on_update
    def _(_):
        _mesh_origin_frame.show_axes = show_mesh_origin_frame_cb.value

    @show_mesh_frame_cb.on_update
    def _(_):
        mesh_frame.active_axes = (show_mesh_frame_cb.value, show_mesh_frame_cb.value, show_mesh_frame_cb.value)

    @reset_button.on_click
    def _(_):
        x_slider.value = 0.0
        y_slider.value = 0.0
        z_slider.value = 0.0
        R_slider.value = 0.0
        P_slider.value = 0.0
        Y_slider.value = 0.0

    @save_button.on_click
    def _(_):
        # Save the mesh frame to the output mesh path
        position = mesh_frame.position
        wxyz = mesh_frame.wxyz
        xyzw = wxyz_to_xyzw(wxyz)
        T = np.eye(4)
        T[:3, 3] = position
        T[:3, :3] = R.from_quat(xyzw).as_matrix()

        # Must apply scale before applying transform so that the transform is not scaled
        # Must do deepcopy so that the original mesh is not modified
        output_mesh = deepcopy(mesh).apply_scale(MESH_SCALE).apply_transform(T)

        output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
        output_mesh.export(output_mesh_path)
        print(f"Saved output mesh to {output_mesh_path}")

        GREEN_RGB = (0, 255, 0)
        nonlocal output_mesh_vis
        if output_mesh_vis is not None:
            print("Removing previous output mesh from scene")
            output_mesh_vis.remove()
            output_mesh_vis = None

        output_mesh_vis = server.scene.add_mesh_simple(
            name="/output_mesh",
            vertices=output_mesh.vertices,
            faces=output_mesh.faces,
            color=GREEN_RGB,
            opacity=0.5,
        )
        print("Added output mesh to scene")

    @scale_slider.on_update
    def _(_):
        nonlocal MESH_SCALE
        MESH_SCALE = scale_slider.value

        # After some experimentation, this is the only way I could update the scale
        # Modifying mesh_vis.vertices directly didn't work because of the equality check not working for vectorized np arrays
        # Also, running this does not modify mesh_vis.vertices, so this works (calls to _queue_update don't stack on each other)
        mesh_vis._queue_update("vertices", mesh_vis.vertices * MESH_SCALE)

    while True:
        print(
            f"mesh_frame.position: {mesh_frame.position}, mesh_frame.wxyz: {mesh_frame.wxyz}"
        )
        time.sleep(1.0)
        # breakpoint()


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_mesh_path", type=Path, required=True)
    parser.add_argument("--output_mesh_path", type=Path, required=True)
    args = parser.parse_args()

    assert args.input_mesh_path.exists(), (
        f"Input mesh path {args.input_mesh_path} does not exist"
    )

    viser_modify_mesh_interactive(
        input_mesh_path=args.input_mesh_path,
        output_mesh_path=args.output_mesh_path,
    )


if __name__ == "__main__":
    main()
