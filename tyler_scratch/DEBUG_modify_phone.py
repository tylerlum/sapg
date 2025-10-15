from pathlib import Path
import trimesh
import numpy as np

orig_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/phone/Iphone seceond version finished.obj")
new_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/phone/Iphone_seceondkversion_finished_modified.obj")
assert orig_obj_path.exists(), f"Object file {orig_obj_path} does not exist"
print(f"Loading object from {orig_obj_path}")
orig_obj = trimesh.load(orig_obj_path)

centroid = orig_obj.bounds.mean(axis=0)
print(f"bounds: {orig_obj.bounds}")
print(f"centroid: {centroid}")

T_translate = np.eye(4)
T_translate[:3, 3] = -centroid + np.array([0.0, 0.0, 0.0])
translated_obj = orig_obj.apply_transform(T_translate)

# new_x = orig_y
# new_y = orig_x
# new_z = -orig_z
T = np.eye(4)
R = np.array([
    [0, 1, 0],  # new_x = orig_y
    [1, 0, 0], # new_y = orig_x
    [0, 0, -1]  # new_z = -orig_z
])
T[:3, :3] = R

rotated_obj = translated_obj.apply_transform(T)

rescaled_obj = rotated_obj.apply_scale(0.001)

# save the rescaled obj
rescaled_obj.export(new_obj_path)
print(f"Saved object to {new_obj_path}")