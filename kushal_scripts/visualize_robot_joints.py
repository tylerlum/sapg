from __future__ import annotations

import time
import os
import numpy as np
import yourdfpy

import viser
from viser.extras import ViserUrdf


def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    
    # Default Kuka positions for first 7 joints
    default_kuka_pos = [0, 0.4, 0, -1, 0, 1.571, 0]
    
    joint_index = 0
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        
        # Use default Kuka position for first 7 joints, 0 for the rest
        if joint_index < len(default_kuka_pos):
            initial_pos = default_kuka_pos[joint_index]
        else:
            initial_pos = 0.0
        
        # Clamp initial position to joint limits
        initial_pos = max(lower, min(upper, initial_pos))
        
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
        joint_index += 1
        
    return slider_handles, initial_config


def main():
    # urdf_path = "/juno/u/kedia/get_a_grip/assets/wrist/robot.urdf"
    urdf_path = "/share/portal/kk837/ag2ag/ag2ag/assets/kuka_allegro/kuka_allegro.urdf"
    
    print(f"Loading URDF: {urdf_path}")
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return
    
    # Start viser server
    server = viser.ViserServer(port=8080)
    
    # Load URDF using yourdfpy first
    print("Loading URDF with yourdfpy...")
    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    
    # Create ViserUrdf with the loaded URDF object
    print("Creating ViserUrdf...")
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
    )
    
    print("URDF loaded successfully!")
    
    # Create sliders in GUI that help us move the robot joints
    with server.gui.add_folder("Joint Position Control"):
        slider_handles, initial_config = create_robot_control_sliders(
            server, viser_urdf
        )
    
    # Note: Visibility controls temporarily removed due to API uncertainty
    # The ViserUrdf object will display visual meshes by default
    
    # Set initial robot configuration
    if initial_config:
        viser_urdf.update_cfg(np.array(initial_config))
    
    # Create grid for reference
    trimesh_scene = urdf.scene or urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )
    
    # Create joint reset button
    reset_button = server.gui.add_button("Reset Joints")
    
    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q
    
    # Add info display
    joint_count = len(viser_urdf.get_actuated_joint_limits())
    info_text = server.gui.add_text(
        "Robot Info", 
        initial_value=f"Kuka Allegro Robot\nControllable Joints: {joint_count}"
    )
    
    print(f"\nRobot control interface ready!")
    print(f"View at: http://localhost:8081")
    print(f"Found {joint_count} controllable joints")
    print("Use the sliders to control joint positions")
    print("Press Ctrl+C to exit...")
    
    # Keep server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main() 