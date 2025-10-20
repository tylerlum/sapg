import pytorch_kinematics as pk
from pathlib import Path
import torch

# URDF path
KUKA_ALLEGRO_URDF_PATH = Path(
    "/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/kuka_allegro_touch_sensor.urdf"
)
assert KUKA_ALLEGRO_URDF_PATH.exists(), (
    f"KUKA_ALLEGRO_URDF_PATH not found: {KUKA_ALLEGRO_URDF_PATH}"
)

# Batch of joint positions
N_BATCH = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
q = torch.zeros(N_BATCH, 23, dtype=torch.float32, device=DEVICE)

# Create chain that can do parallel forward kinematics across all links
chain = pk.build_chain_from_urdf(open(KUKA_ALLEGRO_URDF_PATH).read()).to(device=DEVICE, dtype=torch.float32)
link_names = chain.get_link_names()
print(f"link_names: {link_names}")
joint_names = chain.get_joint_parameter_names()
print(f"joint_names: {joint_names}")
assert len(joint_names) == 23, f"len(joint_names): {len(joint_names)}, expected: 23"

# Run forward kinematics
fk_dict = chain.forward_kinematics(q)
for link_name in link_names:
    assert link_name in fk_dict, f"link_name: {link_name} not in fk_dict ({fk_dict.keys()})"
    pose = fk_dict[link_name].get_matrix()
    assert pose.shape == (N_BATCH, 4, 4), f"pose.shape: {pose.shape}, expected: ({N_BATCH}, 4, 4)"
    print(f"link_name: {link_name}, pose.shape: {pose.shape}")

# Create serial chain that can compute Jacobian for a given end effector link
palm_chain = pk.SerialChain(chain, "iiwa7_link_7").to(device=DEVICE, dtype=torch.float32)
palm_link_names = palm_chain.get_link_names()
print(f"palm_link_names: {palm_link_names}")
palm_joint_names = palm_chain.get_joint_parameter_names()
print(f"palm_joint_names: {palm_joint_names}")
assert len(palm_joint_names) == 7, f"len(palm_joint_names): {len(palm_joint_names)}, expected: 7"

# Run Jacobian computation
J = palm_chain.jacobian(q)
assert J.shape == (N_BATCH, 6, 7), f"J.shape: {J.shape}, expected: ({N_BATCH}, 6, 7)"
print(f"J.shape: {J.shape}")

breakpoint()