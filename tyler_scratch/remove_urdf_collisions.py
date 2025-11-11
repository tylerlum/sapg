from pathlib import Path
from yourdfpy import URDF

original_urdf = Path("/juno/u/tylerlum/yufei_data/yufei_assets/kinova_gen3_urdf/kinova_scene.urdf")
# original_urdf = Path("/juno/u/tylerlum/github_repos/ros_kortex/kortex_description/arms/gen3/7dof/urdf/GEN3_URDF_V12.urdf")
# no_collisions_urdf = Path("/juno/u/tylerlum/yufei_data/yufei_assets/kinova_gen3_urdf/kinova_scene_no_collisions_v2.urdf")
no_collisions_urdf = Path("/juno/u/tylerlum/yufei_data/yufei_assets/kinova_gen3_urdf/kinova_scene_no_collisions_v3.urdf")
# no_collisions_urdf = Path("/juno/u/tylerlum/github_repos/ros_kortex/kortex_description/arms/gen3/7dof/urdf/GEN3_URDF_V12_no_collisions.urdf")

urdf = URDF.load(original_urdf)
link_names = [link.name for link in urdf.robot.links]
num_links = len(link_names)
print(f"num_links: {num_links}")
# link_names_to_remove = link_names[:int(num_links*0.75)]
# link_names_to_remove = [x for x in link_names if "hand" in x]
link_names_to_remove = [x for x in link_names]
print(f"link_names_to_remove: {link_names_to_remove}")
for link in urdf.robot.links:
    if link.name in link_names_to_remove:
        link.collisions.clear()

urdf.write_xml_file(no_collisions_urdf)
print(f"No collisions URDF saved to {no_collisions_urdf}")