from scipy.spatial.transform import Rotation as R
import numpy as np

def print_euler_to_quat(euler_deg: np.ndarray):
    print(f"Euler: {euler_deg}")
    assert euler_deg.shape == (3,)
    euler = np.deg2rad(euler_deg)
    quat_xyzw = R.from_euler('xyz', euler).as_quat()
    print(f"Quat: {quat_xyzw}")
    print()

print_euler_to_quat(np.array([0, 0, 0]))
print_euler_to_quat(np.array([90, 0, 0]))
print_euler_to_quat(np.array([-90, 0, 0]))
print_euler_to_quat(np.array([0, 90, 0]))
print_euler_to_quat(np.array([0, -90, 0]))
print_euler_to_quat(np.array([0, 0, 90]))
print_euler_to_quat(np.array([0, 0, -90]))
print_euler_to_quat(np.array([90, 90, 0]))
print_euler_to_quat(np.array([90, 0, 90]))
print_euler_to_quat(np.array([90, -90, 0]))
print_euler_to_quat(np.array([0, 90, 90]))
print_euler_to_quat(np.array([90, 90, 90]))

def quat_to_euler(quat_xyzw: np.ndarray):
    assert quat_xyzw.shape == (4,)
    euler = R.from_quat(quat_xyzw).as_euler('xyz', degrees=True)
    print(f"Euler: {euler}")
    print()

quat_to_euler(np.array([-0.693, 0.138, 0.138, 0.693]))
quat_to_euler(np.array([-0.5, -0.5, -0.5, 0.5]))