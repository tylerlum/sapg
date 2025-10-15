from pathlib import Path

def generate_hammer_urdf(
    filepath: Path,
    handle_length=0.10,
    handle_width=0.05,
    handle_thickness=0.025,
    head_thickness=0.025,
    head_width=0.05,
    head_length=0.10,
    handle_mass=0.1,
    head_mass=0.2,
    robot_name="cuboidal_hammer",
):
    """
    Generate a URDF of a hammer consisting of a handle and head.

    handle_length: along +x
    handle_width:  along +y
    handle_thickness: along +z
    head_thickness: along +z
    head_width: along +x
    head_length: along +y
    """
    # Distance between centers (no collision)
    x_offset = handle_length / 2 + head_width / 2

    urdf = f"""<?xml version="1.0"?>
<robot name="{robot_name}">

  <!-- Handle -->
  <link name="handle">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{handle_length} {handle_width} {handle_thickness}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{handle_length} {handle_width} {handle_thickness}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{handle_mass}"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{head_width} {head_length} {head_thickness}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{head_width} {head_length} {head_thickness}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="{head_mass}"/>
      <inertia ixx="0.0002" iyy="0.0002" izz="0.0002" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- Joint connecting handle and head -->
  <joint name="handle_to_head" type="fixed">
    <parent link="handle"/>
    <child link="head"/>
    <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
  </joint>

</robot>
"""

    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")


if __name__ == "__main__":
    HANDLE_LENGTH = 0.3
    HANDLE_WIDTH = 0.03
    HANDLE_THICKNESS = 0.02
    HEAD_THICKNESS = 0.02
    HEAD_WIDTH = 0.03
    HEAD_LENGTH = 0.1
    HANDLE_MASS = 0.1
    HEAD_MASS = 0.2
    folder = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/cuboidal_hammer")
    assert folder.exists(), f"Folder {folder} does not exist"
    filename = f"cuboidal_hammer_{HANDLE_LENGTH}_{HANDLE_WIDTH}_{HANDLE_THICKNESS}_{HEAD_WIDTH}_{HEAD_LENGTH}_{HEAD_THICKNESS}_{HANDLE_MASS}_{HEAD_MASS}.urdf"
    filepath = folder / filename
    generate_hammer_urdf(
        filepath=filepath,
        handle_length=HANDLE_LENGTH,
        handle_width=HANDLE_WIDTH,
        handle_thickness=HANDLE_THICKNESS,
        head_thickness=HEAD_THICKNESS,
        head_width=HEAD_WIDTH,
        head_length=HEAD_LENGTH,
        handle_mass=HANDLE_MASS,
        head_mass=HEAD_MASS,
    )
