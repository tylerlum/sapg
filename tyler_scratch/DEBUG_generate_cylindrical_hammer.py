from pathlib import Path
import math

def generate_cylindrical_hammer_urdf(
    filepath: Path,
    handle_length=0.3,
    handle_radius=0.015,
    head_radius=0.03,
    head_length=0.1,
    handle_mass=0.1,
    head_mass=0.2,
    robot_name="cylindrical_hammer",
):
    """
    Single-link cylindrical hammer URDF.
    Handle along +x, head along +y.
    """
    # Head center position
    x_offset = handle_length / 2 + head_radius
    y_offset = 0
    z_offset = 0

    total_mass = handle_mass + head_mass

    # Rotation: cylinders default along z-axis
    # Handle along x-axis: rotate -pi/2 around y-axis
    handle_rpy = f"0 {-math.pi/2} 0"
    # Head along y-axis: rotate pi/2 around x-axis
    head_rpy = f"{math.pi/2} 0 0"

    urdf = f"""<?xml version="1.0"?>
<robot name="{robot_name}">

  <link name="hammer">
    <!-- Handle -->
    <visual>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      <geometry>
        <cylinder radius="{handle_radius}" length="{handle_length}"/>
      </geometry>
      <material name="brown">
        <color rgba="0.55 0.27 0.07 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="{handle_rpy}"/>
      <geometry>
        <cylinder radius="{handle_radius}" length="{handle_length}"/>
      </geometry>
    </collision>

    <!-- Head -->
    <visual>
      <origin xyz="{x_offset} {y_offset} {z_offset}" rpy="{head_rpy}"/>
      <geometry>
        <cylinder radius="{head_radius}" length="{head_length}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{x_offset} {y_offset} {z_offset}" rpy="{head_rpy}"/>
      <geometry>
        <cylinder radius="{head_radius}" length="{head_length}"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="{total_mass}"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

</robot>
"""

    with open(filepath, "w") as f:
        f.write(urdf)
    print(f"✅ Cylindrical hammer URDF written to {filepath}")


if __name__ == "__main__":
    HANDLE_LENGTH = 0.3
    HANDLE_RADIUS = 0.015
    HEAD_RADIUS = 0.015
    HEAD_LENGTH = 0.1
    HANDLE_MASS = 0.1
    HEAD_MASS = 0.2

    folder = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/cylindrical_hammer")
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"cylindrical_hammer_{HANDLE_LENGTH}_{HANDLE_RADIUS}_{HEAD_RADIUS}_{HEAD_LENGTH}_{HANDLE_MASS}_{HEAD_MASS}".replace(".", "-") + ".urdf"
    filepath = folder / filename

    generate_cylindrical_hammer_urdf(
        filepath=filepath,
        handle_length=HANDLE_LENGTH,
        handle_radius=HANDLE_RADIUS,
        head_radius=HEAD_RADIUS,
        head_length=HEAD_LENGTH,
        handle_mass=HANDLE_MASS,
        head_mass=HEAD_MASS,
    )
