import trimesh
import numpy as np
from pathlib import Path

orig_thumb_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/left_sharpa_meshes/left_thumb_MC.STL")
assert orig_thumb_path.exists(), f"Object file {orig_thumb_path} does not exist"
output_thumb_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/left_sharpa_meshes/left_thumb_MC_modified.STL")
print(f"Loading object from {orig_thumb_path}")
orig_thumb = trimesh.load(orig_thumb_path)

bounds = orig_thumb.bounds
assert bounds.shape == (2, 3), f"Bounds shape is {bounds.shape}, expected (2, 3)"
x_min, x_max = bounds[:, 0]
y_min, y_max = bounds[:, 1]
z_min, z_max = bounds[:, 2]
print(f"x_min: {x_min}, x_max: {x_max}")
print(f"y_min: {y_min}, y_max: {y_max}")
print(f"z_min: {z_min}, z_max: {z_max}")
print("Number of vertices and faces in original mesh:", len(orig_thumb.vertices), len(orig_thumb.faces))

AMOUNT_CUT_OFF = 0.035
new_x_min = x_min + AMOUNT_CUT_OFF
new_x_max = x_max
new_y_min = y_min
new_y_max = y_max
new_z_min = z_min
new_z_max = z_max

# Boolean mask for vertices inside the new bounds
verts = orig_thumb.vertices
inside_mask = (
    (verts[:, 0] >= new_x_min) & (verts[:, 0] <= new_x_max) &
    (verts[:, 1] >= new_y_min) & (verts[:, 1] <= new_y_max) &
    (verts[:, 2] >= new_z_min) & (verts[:, 2] <= new_z_max)
)

# Keep only faces where *all three* vertices are inside
faces = orig_thumb.faces
face_mask = inside_mask[faces].all(axis=1)
new_faces = faces[face_mask]

# Create a new mesh with only the kept faces
new_mesh = orig_thumb.submesh([face_mask], append=True, repair=True)
# convex_mesh = new_mesh.convex_hull
new_mesh.fill_holes()
print("Watertight after fill:", new_mesh.is_watertight)

# Optionally, export or visualize
new_mesh.export(output_thumb_path)
print(f"Saved cropped mesh to {output_thumb_path} with {len(new_mesh.vertices)} vertices and {len(new_mesh.faces)} faces.")