from pathlib import Path

def generate_single_link_hammer_urdf(
    filepath: Path,
    handle_length=0.10,
    handle_width=0.05,
    handle_thickness=0.025,
    head_thickness=0.025,
    head_width=0.05,
    head_length=0.10,
    robot_name="cuboidal_hammer",
    density=400,
):
    """
    Generate a URDF of a hammer as a single link with two visuals and collisions.

    handle_length: along +x
    handle_width:  along +y
    handle_thickness: along +z
    head_thickness: along +z
    head_width: along +x
    head_length: along +y
    """
    # Offset of head relative to handle center
    x_offset = handle_length / 2 + head_width / 2

    urdf = f"""<?xml version="1.0"?>
<robot name="{robot_name}">

  <link name="hammer">
    <!-- Handle -->
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

    <!-- Head -->
    <visual>
      <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{head_width} {head_length} {head_thickness}"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="{x_offset} 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="{head_width} {head_length} {head_thickness}"/>
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


if __name__ == "__main__":
    HANDLE_LENGTH = 0.24
    HANDLE_WIDTH = 0.03
    HANDLE_THICKNESS = 0.02
    HEAD_THICKNESS = 0.045
    HEAD_WIDTH = 0.05
    HEAD_LENGTH = 0.08

    folder = Path("/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/cuboidal_hammer")
    folder.mkdir(parents=True, exist_ok=True)

    filename = f"cuboidal_mallet_{HANDLE_LENGTH}_{HANDLE_WIDTH}_{HANDLE_THICKNESS}_{HEAD_WIDTH}_{HEAD_LENGTH}_{HEAD_THICKNESS}".replace(".", "-") + ".urdf"
    filepath = folder / filename

    generate_single_link_hammer_urdf(
        filepath=filepath,
        handle_length=HANDLE_LENGTH,
        handle_width=HANDLE_WIDTH,
        handle_thickness=HANDLE_THICKNESS,
        head_thickness=HEAD_THICKNESS,
        head_width=HEAD_WIDTH,
        head_length=HEAD_LENGTH,
    )
