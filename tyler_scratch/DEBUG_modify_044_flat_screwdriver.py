from pathlib import Path
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

orig_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured_orig.obj")
orig_obj_path_2 = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured_vhacd_orig.obj")
new_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured_vhacd.obj")
new_obj_path_2 = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured.obj")
assert orig_obj_path.exists(), f"Object file {orig_obj_path} does not exist"
assert orig_obj_path_2.exists(), f"Object file {orig_obj_path_2} does not exist"
print(f"Loading object from {orig_obj_path}")
print(f"Loading object from {orig_obj_path_2}")
orig_obj = trimesh.load(orig_obj_path)
orig_obj_2 = trimesh.load(orig_obj_path_2)

centroid = orig_obj.bounds.mean(axis=0)
print(f"bounds: {orig_obj.bounds}")
print(f"centroid: {centroid}")

T_translate = np.eye(4)
T_translate[:3, 3] = -centroid + np.array([-0.035, 0.035, 0.0])
translated_obj = orig_obj.apply_transform(T_translate)
translated_obj_2 = orig_obj_2.apply_transform(T_translate)

T = np.eye(4)
R = Rotation.from_euler('z', -135, degrees=True).as_matrix()
T[:3, :3] = R

rotated_obj = translated_obj.apply_transform(T)
rotated_obj_2 = translated_obj_2.apply_transform(T)

rescaled_obj = rotated_obj.apply_scale(1)
rescaled_obj_2 = rotated_obj_2.apply_scale(1)

# save the rescaled obj
rescaled_obj.export(new_obj_path)
rescaled_obj_2.export(new_obj_path_2)
print(f"Saved object to {new_obj_path}")
print(f"Saved object to {new_obj_path_2}")
