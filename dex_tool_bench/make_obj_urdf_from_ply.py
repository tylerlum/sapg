import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh
import viser




def load_and_process_mesh(mesh_path: Path, center: bool = True) -> trimesh.Trimesh:
    """Load mesh file (GLB, OBJ, PLY, STL, etc.)."""
    data = trimesh.load(str(mesh_path))
    
    # Handle different input types
    if isinstance(data, trimesh.Scene):
        mesh = data.dump(concatenate=True)
    elif isinstance(data, trimesh.Trimesh):
        mesh = data
    else:
        raise ValueError(f"Unsupported format or point cloud. Got: {type(data).__name__}")
    
    print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    if center:
        mesh.vertices -= mesh.centroid
    
    return mesh


def create_urdf(output_path: Path, mesh_name: str, obj_filename: str,
                scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                density: float = 400.0) -> Path:
    """Create URDF file referencing the OBJ mesh."""
    urdf = f"""<?xml version="1.0"?>
<robot name="{mesh_name}">

  <link name="{mesh_name}">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>
    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(urdf)
    return output_path


def save_mesh(output_path: Path, mesh: trimesh.Trimesh) -> Path:
    """Export mesh as OBJ (preserving original materials) and STL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export OBJ (trimesh preserves materials/textures if present)
    mesh.export(str(output_path), file_type="obj")
    
    # Also create STL for collision
    stl_path = output_path.with_suffix(".stl")
    mesh.export(str(stl_path), file_type="stl")
    
    return output_path


class MeshPreview:
    """Interactive viser preview for mesh before saving."""
    
    def __init__(self, mesh: trimesh.Trimesh, output_dir: Path, name: str, port: int = 8080):
        self.original_mesh = mesh
        self.mesh = mesh.copy()
        self.output_dir = output_dir
        self.name = name
        self.saved = False
        
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self._setup_scene()
        self._print_bounds()
        print(f"\nViser preview: http://localhost:{port}")
        print("Adjust parameters and click 'Save' when ready.\n")
    
    def _setup_scene(self):
        """Setup viser scene with mesh and controls."""
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.5, -0.5, 0.3)
            client.camera.look_at = (0.0, 0.0, 0.0)
        
        # Ground grid
        self.server.scene.add_grid("/ground", width=1, height=1, cell_size=0.05)
        
        # Coordinate axes at origin (world frame, fixed)
        self.server.scene.add_frame("/world_axes", axes_length=0.05, axes_radius=0.002)
        
        # Add mesh and bounding box (origin axes added after sliders are created)
        self._update_mesh_visual()
        self._update_bounding_box()
        
        # GUI Controls
        self.server.gui.add_markdown("## Mesh Conversion Preview")
        self.server.gui.add_markdown(f"**Name:** {self.name}")
        self.server.gui.add_markdown("---")
        
        # Scale controls (separate X, Y, Z)
        self.scale_x = self.server.gui.add_slider("Scale X", min=0.1, max=10.0, step=0.01, initial_value=1.0)
        self.scale_y = self.server.gui.add_slider("Scale Y", min=0.1, max=10.0, step=0.01, initial_value=1.0)
        self.scale_z = self.server.gui.add_slider("Scale Z", min=0.1, max=10.0, step=0.01, initial_value=1.0)
        self.scale_x.on_update(lambda _: self._on_transform_change())
        self.scale_y.on_update(lambda _: self._on_transform_change())
        self.scale_z.on_update(lambda _: self._on_transform_change())
        
        # Position offset
        self.offset_x = self.server.gui.add_slider("Offset X", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        self.offset_y = self.server.gui.add_slider("Offset Y", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        self.offset_z = self.server.gui.add_slider("Offset Z", min=-0.5, max=0.5, step=0.01, initial_value=0.0)
        self.offset_x.on_update(lambda _: self._on_offset_change())
        self.offset_y.on_update(lambda _: self._on_offset_change())
        self.offset_z.on_update(lambda _: self._on_offset_change())
        
        # Rotation
        self.rot_x = self.server.gui.add_slider("Rotate X (deg)", min=-180, max=180, step=5, initial_value=0)
        self.rot_y = self.server.gui.add_slider("Rotate Y (deg)", min=-180, max=180, step=5, initial_value=0)
        self.rot_z = self.server.gui.add_slider("Rotate Z (deg)", min=-180, max=180, step=5, initial_value=0)
        self.rot_x.on_update(lambda _: self._on_transform_change())
        self.rot_y.on_update(lambda _: self._on_transform_change())
        self.rot_z.on_update(lambda _: self._on_transform_change())
        
        # Now add origin axes (after sliders exist)
        self._update_origin_axes()
        
        # Density
        self.density_slider = self.server.gui.add_slider(
            "Density (kg/m³)", min=100, max=2000, step=50, initial_value=400
        )
        
        self.server.gui.add_markdown("---")
        
        # Info display
        self.info_text = self.server.gui.add_markdown("**Bounds:** --")
        self._update_info()
        
        self.server.gui.add_markdown("---")
        
        # Print bounds button
        self.print_bounds_btn = self.server.gui.add_button("Print Bounds to Console")
        self.print_bounds_btn.on_click(lambda _: self._print_bounds())
        
        # Save button
        self.save_button = self.server.gui.add_button("Save Files")
        self.save_button.on_click(lambda _: self._save())
        
        self.status_text = self.server.gui.add_markdown("*Click Save when ready*")
    
    def _update_mesh_visual(self):
        """Update the mesh visualization in viser."""
        vertices = self.mesh.vertices.astype(np.float32)
        faces = self.mesh.faces.astype(np.uint32)
        
        # Try to get vertex colors from mesh
        vertex_colors = None
        if hasattr(self.mesh.visual, 'vertex_colors') and self.mesh.visual.vertex_colors is not None:
            vc = self.mesh.visual.vertex_colors
            if vc.shape[0] == len(vertices):
                vertex_colors = vc[:, :3].astype(np.uint8)
        
        if vertex_colors is not None:
            self.server.scene.add_mesh_simple(
                "/mesh",
                vertices=vertices,
                faces=faces,
                vertex_colors=vertex_colors,
                flat_shading=False,
            )
        else:
            # Neutral gray if no colors
            self.server.scene.add_mesh_simple(
                "/mesh",
                vertices=vertices,
                faces=faces,
                color=(180, 180, 180),
                flat_shading=True,
            )
    
    def _update_origin_axes(self):
        """Update the mesh origin axes (shows where origin is after transforms)."""
        import viser.transforms as tf
        
        # Get current rotation and offset
        rx = np.radians(self.rot_x.value)
        ry = np.radians(self.rot_y.value)
        rz = np.radians(self.rot_z.value)
        offset = np.array([self.offset_x.value, self.offset_y.value, self.offset_z.value])
        
        # Create rotation quaternion from euler angles
        rot_matrix = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
        quat = trimesh.transformations.quaternion_from_matrix(rot_matrix)  # [w, x, y, z]
        wxyz = (quat[0], quat[1], quat[2], quat[3])
        
        self.server.scene.add_frame(
            "/mesh_origin",
            axes_length=0.1,
            axes_radius=0.004,
            wxyz=wxyz,
            position=tuple(offset),
        )
    
    def _on_offset_change(self):
        """Handle offset change."""
        self._on_transform_change()
    
    def _on_transform_change(self):
        """Apply all transforms and update visualization."""
        # Start fresh from original
        self.mesh = self.original_mesh.copy()
        
        # Apply rotation
        rx = np.radians(self.rot_x.value)
        ry = np.radians(self.rot_y.value)
        rz = np.radians(self.rot_z.value)
        rot_matrix = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
        self.mesh.apply_transform(rot_matrix)
        
        # Apply non-uniform scale
        scale = np.array([self.scale_x.value, self.scale_y.value, self.scale_z.value])
        self.mesh.vertices *= scale
        
        # Apply offset (origin)
        offset = np.array([self.offset_x.value, self.offset_y.value, self.offset_z.value])
        self.mesh.apply_translation(offset)
        
        self._update_mesh_visual()
        self._update_origin_axes()
        self._update_bounding_box()
        self._update_info()
    
    def _update_bounding_box(self):
        """Draw bounding box as dotted lines."""
        bounds = self.mesh.bounds
        min_pt, max_pt = bounds[0], bounds[1]
        
        # 8 corners of the bounding box
        corners = np.array([
            [min_pt[0], min_pt[1], min_pt[2]],  # 0
            [max_pt[0], min_pt[1], min_pt[2]],  # 1
            [max_pt[0], max_pt[1], min_pt[2]],  # 2
            [min_pt[0], max_pt[1], min_pt[2]],  # 3
            [min_pt[0], min_pt[1], max_pt[2]],  # 4
            [max_pt[0], min_pt[1], max_pt[2]],  # 5
            [max_pt[0], max_pt[1], max_pt[2]],  # 6
            [min_pt[0], max_pt[1], max_pt[2]],  # 7
        ])
        
        # 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]
        
        # Draw each edge as dashed line segments
        for i, (a, b) in enumerate(edges):
            # Create dashed effect with multiple short segments
            n_dashes = 5
            for j in range(n_dashes):
                t0 = j / n_dashes + 0.05
                t1 = (j + 0.5) / n_dashes
                p0 = corners[a] * (1 - t0) + corners[b] * t0
                p1 = corners[a] * (1 - t1) + corners[b] * t1
                points = np.stack([p0, p1]).astype(np.float32)
                self.server.scene.add_spline_catmull_rom(
                    f"/bbox/edge_{i}_{j}",
                    positions=points,
                    color=(255, 200, 0),
                    line_width=2.0,
                )
    
    def _print_bounds(self):
        """Print bounding box to console."""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        print(f"\n--- Bounding Box Extents ---")
        print(f"  X: [{bounds[0][0]:.4f}, {bounds[1][0]:.4f}]  size: {size[0]:.4f} m")
        print(f"  Y: [{bounds[0][1]:.4f}, {bounds[1][1]:.4f}]  size: {size[1]:.4f} m")
        print(f"  Z: [{bounds[0][2]:.4f}, {bounds[1][2]:.4f}]  size: {size[2]:.4f} m")
    
    def _update_info(self):
        """Update info display in GUI."""
        bounds = self.mesh.bounds
        size = bounds[1] - bounds[0]
        
        self.info_text.content = (
            f"**Bounds:** X=[{bounds[0][0]:.3f}, {bounds[1][0]:.3f}] "
            f"Y=[{bounds[0][1]:.3f}, {bounds[1][1]:.3f}] "
            f"Z=[{bounds[0][2]:.3f}, {bounds[1][2]:.3f}]\n\n"
            f"**Size:** {size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} m"
        )
    
    def _save(self):
        """Save all output files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        obj_path = self.output_dir / f"{self.name}.obj"
        urdf_path = self.output_dir / f"{self.name}.urdf"
        
        # Save mesh (OBJ + STL)
        save_mesh(obj_path, self.mesh)
        print(f"Saved: {obj_path}")
        print(f"Saved: {obj_path.with_suffix('.stl')}")
        
        # Save URDF
        create_urdf(urdf_path, self.name, f"{self.name}.obj", scale=(1.0, 1.0, 1.0), density=self.density_slider.value)
        print(f"Saved: {urdf_path}")
        
        self.saved = True
        self.status_text.content = f"**✓ Saved to:** `{self.output_dir}`"
    
    def run(self):
        """Run the preview server until files are saved."""
        try:
            while not self.saved:
                time.sleep(0.1)
            print("\nFiles saved. You can close the browser.")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\nCancelled.")


def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python make_obj_urdf_from_ply.py <input.glb>")
    #     return 1
    
    mesh_path = Path("/share/portal/kk837/sapg/dex_tool_bench/spatula.glb")
    
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}")
        return 1
    
    name = mesh_path.stem
    output_dir = mesh_path.parent / name
    
    print(f"Loading: {mesh_path}")
    mesh = load_and_process_mesh(mesh_path, center=True)
    
    preview = MeshPreview(mesh, output_dir, name)
    preview.run()
    
    return 0


if __name__ == "__main__":
    exit(main())

