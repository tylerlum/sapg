import json
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=Path, required=True)
parser.add_argument("--output_file", type=Path, required=True)
args = parser.parse_args()

input_file: Path = args.input_file
output_file: Path = args.output_file

assert input_file.exists(), f"Input file does not exist: {input_file}"
assert input_file.suffix == ".json", f"Input file is not a JSON file: {input_file}"
assert output_file.suffix == ".json", f"Output file is not a JSON file: {output_file}"

with open(input_file, "r") as f:
    goal_pose_robot_frame_data = json.load(f)

goal_pose_world_frame_data = [
    [x, y + 0.8, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_robot_frame_data
]
print(f"Found {len(goal_pose_world_frame_data)} goals")

MIN_Z = 0.6
# MIN_Z = 0.65
# MIN_Z = 0.0
for t in range(len(goal_pose_world_frame_data)):
    x, y, z, qx, qy, qz, qw = goal_pose_world_frame_data[t]
    if z >= MIN_Z:
        break
assert t < len(goal_pose_world_frame_data), f"No goal pose with z >= {MIN_Z} found"
first_t_with_high_z = t
print(f"First goal pose with z >= {MIN_Z} is at index {first_t_with_high_z}")

# Offset by 10cm in y direction
# goal_pose_world_frame_data = [[x, y + 0.1, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

# Offset by 5cm in -x direction
# goal_pose_world_frame_data = [[x - 0.05, y, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

# Offset by 2.5cm in -x direction
# goal_pose_world_frame_data = [[x - 0.025, y, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

# Offset by 2cm in -x direction
# goal_pose_world_frame_data = [[x - 0.02, y, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

# Offset by 3cm in z direction
# goal_pose_world_frame_data = [[x, y, z + 0.03, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

# Offset by 7cm in z direction
goal_pose_world_frame_data = [[x, y, z + 0.07, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_world_frame_data]

start_pose_world_frame = goal_pose_world_frame_data[0]
# start_pose_world_frame[0] += 0.02
# start_pose_world_frame[2] += 0.02

print(f"Now trimming the trajectory to only include goals with z >= {MIN_Z} (num goals: {len(goal_pose_world_frame_data[first_t_with_high_z:])})")
json_data = {
    "start_pose": start_pose_world_frame,
    "goals": goal_pose_world_frame_data[first_t_with_high_z:],
}

print(f"Saving to file: {output_file}")
with open(output_file, "w") as f:
    json.dump(json_data, f, indent=4)
print("Done")