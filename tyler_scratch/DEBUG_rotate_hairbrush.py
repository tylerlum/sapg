from pathlib import Path
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation

orig_obj_path = Path("/juno/u/tylerlum/github_repos/InstantMesh/outputs/instant-mesh-large/meshes/Brush2_Masked.obj")
new_obj_path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/hairbrush/hairbrush.obj")
assert orig_obj_path.exists(), f"Object file {orig_obj_path} does not exist"
print(f"Loading object from {orig_obj_path}")
orig_obj = trimesh.load(orig_obj_path)

T = np.eye(4)
R = Rotation.from_euler('zx', [-115, 180], degrees=True).as_matrix()
T[:3, :3] = R

rotated_obj = orig_obj.apply_transform(T)

T2 = np.eye(4)
T2[:3, 3] = np.array([0.05, 0.0, 0.0075])
rotated_obj = rotated_obj.apply_transform(T2)

# save the rotated obj
new_obj_path.parent.mkdir(parents=True, exist_ok=True)
rotated_obj.export(new_obj_path)
print(f"Saved object to {new_obj_path}")