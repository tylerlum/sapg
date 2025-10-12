from pathlib import Path
import trimesh
import numpy as np

orig_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/YcbHammer/collision_vhacd_orig.obj")
new_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/YcbHammer/collision_vhacd.obj")
assert orig_obj_path.exists(), f"Object file {orig_obj_path} does not exist"
print(f"Loading object from {orig_obj_path}")
orig_obj = trimesh.load(orig_obj_path)

# new_x = -orig_y
# new_y = orig_x
# new_z = orig_z
T = np.eye(4)
R = np.array([
    [0, -1, 0],  # new_x = -orig_y
    [1, 0, 0], # new_y = orig_x
    [0, 0, 1]  # new_z = orig_z
])
T[:3, :3] = R
T[:3, 3] = np.array([0.0, -0.02, 0.0])

rotated_obj = orig_obj.apply_transform(T)

# save the rotated obj
rotated_obj.export(new_obj_path)
print(f"Saved object to {new_obj_path}")