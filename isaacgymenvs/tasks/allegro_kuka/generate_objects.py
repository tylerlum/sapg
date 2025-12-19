from pathlib import Path
import math
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
        # Default z is along cylinder axis
        # We rotate so it is along +x
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
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>
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


def compute_mass_and_inertia(scale: Union[Tuple[float, float, float], Tuple[float, float]], density: float):
    if len(scale) == 3:
        lx, ly, lz = scale
        v = lx * ly * lz
        m = v * density
        ixx = (1/12) * m * (ly**2 + lz**2)
        iyy = (1/12) * m * (lx**2 + lz**2)
        izz = (1/12) * m * (lx**2 + ly**2)
    elif len(scale) == 2:
        h, d = scale[0], scale[1]
        r = d / 2
        v = math.pi * (r**2) * h
        m = v * density
        izz = 0.5 * m * (r**2)
        iyy = (1/12) * m * (3*r**2 + h**2)
        ixx = iyy
    else:
        raise ValueError(f"Invalid scale: {scale}")
    return m, ixx, iyy, izz

def generate_handle_head_urdf_variable_density(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_density: float = 400,
    head_density: float = 800,
):
    if len(handle_scale) == 3:
        handle_len_x, handle_len_y, handle_len_z = handle_scale
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{handle_len_x} {handle_len_y} {handle_len_z}"/>
        </geometry>
        """
        handle_mass, handle_ixx, handle_iyy, handle_izz = compute_mass_and_inertia(scale=handle_scale, density=handle_density)

    elif len(handle_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +x
        handle_height, handle_diameter = handle_scale
        handle_radius = handle_diameter / 2
        handle_text = f"""\
        <origin xyz="0 0 0" rpy="0 -1.5707963267948966 0"/>
        <geometry>
          <cylinder length="{handle_height}" radius="{handle_radius}"/>
        </geometry>
        """
        # Note we flip ixx to the end because we rotate so it is along +x
        handle_mass, handle_izz, handle_iyy, handle_ixx = compute_mass_and_inertia(scale=handle_scale, density=handle_density)
    else:
        raise ValueError(f"Invalid handle scale: {handle_scale}")

    if len(head_scale) == 3:
        head_len_x, head_len_y, head_len_z = head_scale
        x_offset = handle_scale[0] / 2 + head_len_x / 2
        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
        head_mass, head_ixx, head_iyy, head_izz = compute_mass_and_inertia(scale=head_scale, density=head_density)
    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="{x_offset} 0 0" rpy="-1.5707963267948966 0 0"/>
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
        # Note we flip iyy to the end because we rotate so it is along +y
        head_mass, head_ixx, head_izz, head_iyy = compute_mass_and_inertia(scale=head_scale, density=head_density)
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    # Compute mass and inertia
    total_mass = handle_mass + head_mass

    # x_offset is the distance from handle center to head center
    com_x = (handle_mass * 0 + head_mass * x_offset) / total_mass

    d_handle = 0 - com_x
    d_head = x_offset - com_x

    ixx = handle_ixx + head_ixx
    iyy = (handle_iyy + handle_mass * d_handle**2) + (head_iyy + head_mass * d_head**2)
    izz = (handle_izz + handle_mass * d_handle**2) + (head_izz + head_mass * d_head**2)

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
      <origin xyz="{com_x} 0 0" rpy="0 0 0"/>
      <mass value="{handle_mass + head_mass}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")
    return filepath



def generate_handle_head_urdf_variable_density_2_links(
    filepath: Path,
    handle_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    head_scale: Union[Tuple[float, float, float], Tuple[float, float]],
    handle_density: float = 400,
    head_density: float = 800,
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
        # Default z is along cylinder axis
        # We rotate so it is along +x
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
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <box size="{head_len_x} {head_len_y} {head_len_z}"/>
        </geometry>
        """
    elif len(head_scale) == 2:
        # Default z is along cylinder axis
        # We rotate so it is along +y
        head_height, head_diameter = head_scale
        head_radius = head_diameter / 2
        x_offset = handle_scale[0] / 2 + head_radius

        head_text = f"""\
        <origin xyz="0 0 0" rpy="-1.5707963267948966 0 0"/>
        <geometry>
          <cylinder length="{head_height}" radius="{head_radius}"/>
        </geometry>
        """
    else:
        raise ValueError(f"Invalid head scale: {head_scale}")

    urdf = f"""<?xml version="1.0"?>
<robot name="handle_head">

  <link name="handle">
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

    <inertial>
      <density value="{handle_density}"/>
    </inertial>
  </link>

  <link name="head">
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
      <density value="{head_density}"/>
    </inertial>
  </link>

  <joint name="handle_head_joint" type="fixed">
    <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
    <parent link="handle"/>
    <child link="head"/>
  </joint>

</robot>
"""
    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ URDF written to {filepath}")
    return filepath
