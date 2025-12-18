from pathlib import Path
from typing import Tuple, Union


def generate_cuboid_urdf_constant_density(
    filepath: Path, scale: Tuple[float, float, float], density: float = 400
) -> Path:
    urdf = f"""<?xml version="1.0"?>
<robot name="cuboid">

  <link name="cuboid">
    <!-- Handle -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{scale[0]} {scale[1]} {scale[2]}"/>
      </geometry>
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")
    return filepath


def generate_cylinder_urdf_constant_density(
    filepath: Path, height: float, diameter: float, density: float = 400
) -> Path:
    # In URDFs, cylinders are along z axis
    # But we rotate them to be along +x
    # Height is along +x
    # Radius is along +y and +z
    radius = diameter / 2
    urdf = f"""<?xml version="1.0"?>
<robot name="cylinder">

  <link name="cylinder">
    <visual>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder length="{height}" radius="{radius}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
      <geometry>
        <cylinder length="{height}" radius="{radius}"/>
      </geometry>
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")
    return filepath


def generate_handle_head_urdf_constant_density(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    density: float = 400,
):
    if len(handle_scale) == 3:
        handle_len_x, handle_len_y, handle_len_z = handle_scale
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{handle_len_x} {handle_len_y} {handle_len_z}"/>
        </geometry>
        """
    elif len(handle_scale) == 2:
        handle_height, handle_diameter = handle_scale
        handle_radius = handle_diameter / 2
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
        <geometry>
          <cylinder length="{handle_height}" radius="{handle_radius}"/>
        </geometry>
        """
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")

    if len(head_scale) == 3:
        head_len_x, head_len_y, head_len_z = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{head_scale[0]} {head_scale[1]} {head_scale[2]}"/>
        </geometry>
        """
    elif len(head_scale) == 2:
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="0 -1.5707963267948966 0"/>
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">

  <link name="handle_head">
    <!-- Handle -->
    <visual>
      {handle_text}
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      {handle_text}
    </collision>

    <!-- Head -->
    <visual>
      {head_text}
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      {head_text}
    </collision>

    <inertial>
      <density value="{density}"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")
    return filepath