"""
Render object instances from mesh files and save as icon images.

This script loads .obj or .glb meshes from the dex_tool_bench assets folder,
renders them using trimesh, and saves the renders as PNG icons for use in plots.
"""

import os
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

ASSETS_DIR = Path("/share/portal/kk837/sapg/assets/urdf/dex_tool_bench")
OUTPUT_DIR = Path(__file__).parent / "object_instance_icons"

# Object instances to render (category -> list of object instance names)
OBJECT_INSTANCES = {
    "hammer": ["toy_hammer", "mallet"],
    "eraser": ["anvil_eraser", "expo_eraser"],
    "marker": ["sharpie_closed", "staples_open"],
    "screwdriver": ["red_screwdriver", "real_flat_screwdriver"],
    "brush": ["anvil_brush", "red_brush"],
    "spatula": ["black_spatula", "spoon_spatula"],
}

# Render settings
IMAGE_SIZE = (512, 512)


def find_mesh_file(category: str, obj_instance: str) -> Path:
    """Find the mesh file for a given object instance."""
    obj_dir = ASSETS_DIR / category / obj_instance
    
    # Try .obj first (has separate texture files), then .glb
    for ext in [".obj", ".glb"]:
        mesh_file = obj_dir / f"{obj_instance}{ext}"
        if mesh_file.exists():
            return mesh_file
    
    # Fallback: search for any mesh file
    for ext in [".obj", ".glb", ".stl"]:
        for f in obj_dir.glob(f"*{ext}"):
            return f
    
    raise FileNotFoundError(f"No mesh file found for {category}/{obj_instance}")


def load_mesh_with_texture(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh with textures properly resolved."""
    mesh_dir = mesh_path.parent
    
    # Create a resolver that can find textures in the same directory
    resolver = trimesh.visual.resolvers.FilePathResolver(mesh_dir)
    
    # Load the mesh with the resolver
    mesh = trimesh.load(
        mesh_path, 
        resolver=resolver,
        force='mesh'
    )
    
    # If it's a scene, combine into single mesh
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for name, geom in mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = list(mesh.geometry.values())[0]
    
    return mesh


def get_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract or compute vertex colors from mesh."""
    # Try to get vertex colors from visual
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors)
        if colors.shape[0] == len(mesh.vertices):
            return colors[:, :3] / 255.0
    
    # Try to sample from texture
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        try:
            if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                material = mesh.visual.material
                if hasattr(material, 'image') and material.image is not None:
                    # Sample colors from texture using UV coordinates
                    uv = mesh.visual.uv
                    img = np.array(material.image)
                    
                    # Normalize UV coordinates
                    uv = uv % 1.0  # Wrap UVs to [0, 1]
                    
                    # Convert UV to pixel coordinates
                    h, w = img.shape[:2]
                    px = (uv[:, 0] * (w - 1)).astype(int)
                    py = ((1 - uv[:, 1]) * (h - 1)).astype(int)  # Flip V
                    
                    # Clamp to valid range
                    px = np.clip(px, 0, w - 1)
                    py = np.clip(py, 0, h - 1)
                    
                    # Sample colors
                    colors = img[py, px, :3] / 255.0
                    return colors
        except Exception as e:
            print(f"    Warning: Could not sample texture: {e}")
    
    # Fallback: use face colors if available
    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors is not None:
        face_colors = mesh.visual.face_colors[:, :3] / 255.0
        # Convert face colors to vertex colors by averaging
        vertex_colors = np.zeros((len(mesh.vertices), 3))
        vertex_counts = np.zeros(len(mesh.vertices))
        for i, face in enumerate(mesh.faces):
            for v in face:
                vertex_colors[v] += face_colors[i]
                vertex_counts[v] += 1
        vertex_counts[vertex_counts == 0] = 1
        vertex_colors = vertex_colors / vertex_counts[:, np.newaxis]
        return vertex_colors
    
    # Default gray color
    return np.ones((len(mesh.vertices), 3)) * 0.7


def compute_optimal_view(mesh: trimesh.Trimesh):
    """
    Compute optimal camera view angles based on tool orientation convention.
    
    Convention: All tools have their long axis along X, with Z pointing up.
    We want to view from the front-side to show the tool's profile nicely.
    
    Returns (elevation, azimuth) in degrees.
    """
    # Get bounding box extents for logging
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    
    # View so that X axis (tool length) appears nearly vertical in the image
    # This makes tall/long tools fit well above bar charts
    azim = -30  # Angle to show some 3D depth
    elev = 80   # High elevation so X axis points upward
    
    print(f"    Extents: X={extents[0]:.3f}, Y={extents[1]:.3f}, Z={extents[2]:.3f}")
    print(f"    View: elev={elev}, azim={azim}")
    
    return elev, azim


def render_mesh_matplotlib(mesh: trimesh.Trimesh, output_path: Path) -> None:
    """Render mesh using matplotlib with proper colors and transparent background."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    fig = plt.figure(figsize=(5.12, 5.12), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh data
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    
    # Compute optimal viewing angle before centering/scaling
    elev, azim = compute_optimal_view(mesh)
    
    # Center the mesh using bounding box center (not vertex mean)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    vertices = vertices - bbox_center
    
    # Get the extents for proper aspect ratio
    extents = bbox_max - bbox_min
    max_extent = extents.max()
    
    # Normalize to [-1, 1] range while preserving aspect ratio
    if max_extent > 0:
        vertices = vertices / (max_extent / 2)
    
    # Get vertex colors
    vertex_colors = get_vertex_colors(mesh)
    
    # Compute face colors as average of vertex colors
    face_colors = np.zeros((len(faces), 3))
    for i, face in enumerate(faces):
        face_colors[i] = vertex_colors[face].mean(axis=0)
    
    # Create polygon collection
    poly3d = [[vertices[idx] for idx in face] for face in faces]
    
    # Add alpha channel (fully opaque faces)
    face_colors_rgba = np.column_stack([face_colors, np.ones(len(faces))])
    
    collection = Poly3DCollection(
        poly3d, 
        facecolors=face_colors_rgba,
        edgecolors='none',
        linewidths=0,
        alpha=1.0
    )
    ax.add_collection3d(collection)
    
    # Set equal aspect ratio with tighter limits for more zoom
    limit = 0.6
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Force equal aspect ratio (crucial for no distortion)
    ax.set_box_aspect([1, 1, 1])
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Remove axes
    ax.set_axis_off()
    
    # Set transparent background
    ax.set_facecolor('none')
    fig.patch.set_facecolor('none')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    # Remove margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save with transparent background
    plt.savefig(
        output_path, 
        dpi=100, 
        bbox_inches='tight',
        pad_inches=0.02,
        transparent=True
    )
    plt.close()


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Rendering object icons to: {OUTPUT_DIR}")
    print("=" * 60)
    
    for category, instances in OBJECT_INSTANCES.items():
        print(f"\n{category.upper()}:")
        
        for obj_instance in instances:
            output_file = OUTPUT_DIR / f"{obj_instance}.png"
            
            try:
                # Find mesh file
                mesh_path = find_mesh_file(category, obj_instance)
                print(f"  {obj_instance}: {mesh_path.name}")
                
                # Load mesh with textures
                mesh = load_mesh_with_texture(mesh_path)
                print(f"    Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
                # Check visual type
                visual_type = type(mesh.visual).__name__
                print(f"    Visual type: {visual_type}")
                
                # Render using matplotlib
                render_mesh_matplotlib(mesh, output_file)
                print(f"    Saved: {output_file.name}")
                
            except Exception as e:
                import traceback
                print(f"  {obj_instance}: ERROR - {e}")
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
