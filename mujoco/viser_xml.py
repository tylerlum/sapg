from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import yourdfpy
from trimesh.scene import Scene
from typing_extensions import assert_never

import viser

import time

import numpy as np
import tyro

import viser
# from viser.extras import ViserUrdf

import viser.transforms as tf


class ViserUrdf:
    """Helper for rendering URDFs in Viser. This is a self-contained example
    that uses only basic Viser features. It can be copied and modified if you
    need more fine-grained control.

    To move or control visibility of the entire robot, you can create a
    parent frame that the URDF will be attached to. This is because
    ViserUrdf creates the robot's geometry as children of the specified
    `root_node_name`, but doesn't create the root node itself.

    .. code-block:: python

        import time
        import numpy as np
        import viser
        from viser.extras import ViserUrdf
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        server = viser.ViserServer()

        # Create a parent frame for the robot.
        # ViserUrdf will attach the robot's geometry as children of this frame.
        robot_base = server.scene.add_frame("/robot", show_axes=False)

        # Load a URDF from robot_descriptions package.
        urdf = ViserUrdf(
            server,
            load_robot_description("panda_description"),
            root_node_name="/robot"
        )

        # Move the entire robot by updating the base frame.
        robot_base.position = (1.0, 0.0, 0.5)  # Move to (x=1, y=0, z=0.5).

        # Update joint configuration.
        urdf.update_cfg(np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))

        # Make the robot blink.
        while True:
            robot_base.visible = False
            time.sleep(0.2)
            robot_base.visible = True
            time.sleep(3.0)

    Args:
        target: ViserServer or ClientHandle object to add URDF to.
        urdf_or_path: Either a path to a URDF file or a yourdfpy URDF object.
        scale: Scale factor to apply to resize the URDF.
        root_node_name: Viser scene tree name for the root of the URDF geometry.
        mesh_color_override: Optional color to override the URDF's visual mesh colors. RGB or RGBA tuple.
        collision_mesh_color_override: Optional color to override the URDF's collision mesh colors. RGB or RGBA tuple.
        load_meshes: If true, shows the URDF's visual meshes. If a yourdfpy
            URDF object is used as input, this expects the URDF to have a visual
            scene graph configured.
        load_collision_meshes: If true, shows the URDF's collision meshes. If a
            yourdfpy URDF object is used as input, this expects the URDF to have a
            collision scene graph configured.
    """

    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        urdf_or_path: yourdfpy.URDF | Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        collision_mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
    ) -> None:
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        if isinstance(urdf_or_path, Path):
            urdf = yourdfpy.URDF.load(
                urdf_or_path,
                build_scene_graph=load_meshes,
                build_collision_scene_graph=load_collision_meshes,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes,
                filename_handler=partial(
                    yourdfpy.filename_handler_magic,
                    dir=urdf_or_path.parent,
                ),
            )
        else:
            urdf = urdf_or_path
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name
        self._load_meshes = load_meshes
        self._collision_root_frame: viser.FrameHandle | None = None
        self._visual_root_frame: viser.FrameHandle | None = None
        self._joint_frames: List[viser.SceneNodeHandle] = []
        self._meshes: List[viser.SceneNodeHandle] = []
        num_joints_to_repeat = 0
        if load_meshes:
            if urdf.scene is not None:
                num_joints_to_repeat += 1
                self._visual_root_frame = self._add_joint_frames_and_meshes(
                    urdf.scene,
                    root_node_name,
                    collision_geometry=False,
                    mesh_color_override=mesh_color_override,
                )
            else:
                warnings.warn(
                    "load_meshes is enabled but the URDF model does not have a visual scene configured. Not displaying."
                )
        if load_collision_meshes:
            if urdf.collision_scene is not None:
                num_joints_to_repeat += 1
                self._collision_root_frame = self._add_joint_frames_and_meshes(
                    urdf.collision_scene,
                    root_node_name,
                    collision_geometry=True,
                    mesh_color_override=collision_mesh_color_override,
                )
            else:
                warnings.warn(
                    "load_collision_meshes is enabled but the URDF model does not have a collision scene configured. Not displaying."
                )

        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat

    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return self._visual_root_frame is not None and self._visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        if self._visual_root_frame is not None:
            self._visual_root_frame.visible = visible
        else:
            warnings.warn(
                "Cannot set `.show_visual`, since no visual meshes were loaded."
            )

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return (
            self._collision_root_frame is not None
            and self._collision_root_frame.visible
        )

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        if self._collision_root_frame is not None:
            self._collision_root_frame.visible = visible
        else:
            warnings.warn(
                "Cannot set `.show_collision`, since no collision meshes were loaded."
            )

    def remove(self) -> None:
        """Remove URDF from scene."""
        # Some of this will be redundant, since children are removed when
        # parents are removed.
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()

    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(self._joint_map_values, self._joint_frames):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(
                joint.child, joint.parent, collision_geometry=not self._load_meshes
            )
            frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
            frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(
        self,
    ) -> dict[str, tuple[float | None, float | None]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: dict[str, tuple[float | None, float | None]] = {}
        for joint_name, joint in zip(
            self._urdf.actuated_joint_names, self._urdf.actuated_joints
        ):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            if joint.limit is None:
                out[joint_name] = (-np.pi, np.pi)
            else:
                out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)

    def _add_joint_frames_and_meshes(
        self,
        scene: Scene,
        root_node_name: str,
        collision_geometry: bool,
        mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None,
    ) -> viser.FrameHandle:
        """
        Helper function to add joint frames and meshes to the ViserUrdf object.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )

        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                link_name,
                scene.graph.transforms.parents[link_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)

            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            if mesh_color_override is None:
                self._meshes.append(self._target.scene.add_mesh_trimesh(name, mesh))
            elif len(mesh_color_override) == 3:
                self._meshes.append(
                    self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=mesh_color_override,
                    )
                )
            elif len(mesh_color_override) == 4:
                self._meshes.append(
                    self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=mesh_color_override[:3],
                        opacity=mesh_color_override[3],
                    )
                )
            else:
                assert_never(mesh_color_override)
        return root_frame


def _viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])



def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(
    urdf_path: Path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf"),
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF.
    #
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    tyro.cli(main)