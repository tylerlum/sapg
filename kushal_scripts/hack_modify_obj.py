"""Load an OBJ, paint colors on vertices, and save."""

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser


class MeshColorEditor:
    def __init__(self, obj_path: str, port: int = 8080):
        self.obj_path = Path(obj_path)
        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJ file not found: {obj_path}")

        self.port = port
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.mesh = trimesh.load(str(self.obj_path), force="mesh")

        # Extract original vertex colors from mesh (convert from textures/materials if needed)
        n_verts = len(self.mesh.vertices)
        try:
            # Convert textures/materials to vertex colors
            color_visual = self.mesh.visual.to_color()
            if hasattr(color_visual, 'vertex_colors') and color_visual.vertex_colors is not None:
                orig_colors = np.array(color_visual.vertex_colors)
                # Ensure RGBA format
                if orig_colors.shape[1] == 3:
                    orig_colors = np.hstack([orig_colors, np.full((n_verts, 1), 255, dtype=np.uint8)])
                self.vertex_colors = orig_colors.astype(np.uint8)
                print(f"Loaded {n_verts} vertices with colors from mesh")
            else:
                self.vertex_colors = np.full((n_verts, 4), [200, 200, 200, 255], dtype=np.uint8)
                print(f"No colors found, using default gray for {n_verts} vertices")
        except Exception as e:
            print(f"Could not load vertex colors: {e}, using default gray")
            self.vertex_colors = np.full((n_verts, 4), [200, 200, 200, 255], dtype=np.uint8)

        # Store original for reset
        self.orig_vertex_colors = self.vertex_colors.copy()

        self._setup_scene()
        self._setup_gui()

    def _setup_scene(self):
        @self.server.on_client_connect
        def _(client):
            client.camera.position = (0.3, 0.3, 0.3)
            client.camera.look_at = (0.0, 0.0, 0.0)

        self.server.scene.add_grid("/grid", width=1, height=1, cell_size=0.05)
        self._update_mesh_display()

    def _update_mesh_display(self):
        """Update the mesh in the scene with current vertex colors."""
        if hasattr(self, "mesh_handle") and self.mesh_handle is not None:
            self.mesh_handle.remove()

        # Create a trimesh with vertex colors applied
        display_mesh = trimesh.Trimesh(
            vertices=self.mesh.vertices,
            faces=self.mesh.faces,
            vertex_colors=self.vertex_colors,
        )

        self.mesh_handle = self.server.scene.add_mesh_trimesh(
            "/mesh",
            display_mesh,
        )

        # Set up click handler for painting
        @self.mesh_handle.on_click
        def _(event: viser.ScenePointerEvent):
            self._on_mesh_click(event)

    def _setup_gui(self):
        with self.server.gui.add_folder("Brush"):
            self.color_picker = self.server.gui.add_rgb(
                "Paint Color",
                initial_value=(255, 0, 0),
            )
            self.brush_radius = self.server.gui.add_slider(
                "Brush Radius",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=0.01,
            )

        with self.server.gui.add_folder("Actions"):
            self.btn_reset = self.server.gui.add_button("Reset Colors")
            self.btn_reset.on_click(lambda _: self._reset_colors())

            self.btn_save = self.server.gui.add_button("Save Modified OBJ")
            self.btn_save.on_click(lambda _: self._save())

        self.server.gui.add_markdown("**Click on mesh to paint vertices**")

    def _on_mesh_click(self, event: viser.ScenePointerEvent):
        """Paint vertices near the clicked point."""
        if event.ray_origin is None or event.ray_direction is None:
            return

        # Cast ray to find intersection point
        ray_origin = np.array(event.ray_origin)
        ray_direction = np.array(event.ray_direction)

        # Find closest vertex to ray
        # Project vertices onto ray and find closest
        verts = self.mesh.vertices
        to_verts = verts - ray_origin
        # Project onto ray direction
        t = np.dot(to_verts, ray_direction)
        t = np.maximum(t, 0)  # Only consider points in front
        closest_on_ray = ray_origin + t[:, None] * ray_direction
        distances = np.linalg.norm(verts - closest_on_ray, axis=1)

        # Find vertices within brush radius of the closest point
        click_point_idx = np.argmin(distances)
        click_point = verts[click_point_idx]

        # Color all vertices within brush radius of click point
        dists_from_click = np.linalg.norm(verts - click_point, axis=1)
        mask = dists_from_click < self.brush_radius.value

        color = self.color_picker.value
        self.vertex_colors[mask] = [color[0], color[1], color[2], 255]

        n_painted = np.sum(mask)
        print(f"Painted {n_painted} vertices at {click_point}")

        self._update_mesh_display()

    def _reset_colors(self):
        """Reset all vertex colors to original."""
        self.vertex_colors = self.orig_vertex_colors.copy()
        self._update_mesh_display()
        print("Colors reset.")

    def _save(self):
        """Save the mesh with modified vertex colors."""
        # Apply vertex colors to mesh
        self.mesh.visual.vertex_colors = self.vertex_colors

        # Save to dex_tool_bench/modified/real_flat_screwdriver_modified/
        output_dir = Path("/share/portal/kk837/sapg/assets/urdf/dex_tool_bench/modified/real_flat_screwdriver_modified")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "real_flat_screwdriver_modified.obj"
        self.mesh.export(str(output_path))
        print(f"Saved: {output_path}")

    def run(self):
        try:
            actual_port = self.server.get_port()
        except AttributeError:
            actual_port = self.port
        print(f"Viser: http://localhost:{actual_port}")
        print("Click on the mesh to paint vertices with the selected color.")

        while True:
            time.sleep(1.0)


def main():
    default_obj = "/share/portal/kk837/sapg/assets/urdf/dex_tool_bench/screwdriver/real_flat_screwdriver/real_flat_screwdriver.obj"
    obj_path = sys.argv[1] if len(sys.argv) > 1 else default_obj

    editor = MeshColorEditor(obj_path)
    editor.run()


if __name__ == "__main__":
    main()
