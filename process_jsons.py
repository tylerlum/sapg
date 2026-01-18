from pathlib import Path
from tqdm import tqdm
import json
import subprocess

"""
/juno/u/kedia/FoundationPose/human_videos/Jan_17
├── brush
│   ├── anvil_brush
│   │   ├── sweep_forward
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── sweep_forward_right
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   └── red_brush
│       ├── sweep_forward
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       ├── sweep_forward_easy
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       └── sweep_forward_right
│           ├── cam_K.txt
│           ├── depth
│           ├── rgb
│           ├── rgb_video.mp4
│           └── subgoals.json
├── eraser
│   ├── amazon_eraser
│   │   ├── wipe_higher
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── wipe_lower
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   ├── anvil_eraser
│   │   ├── wipe_higher
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── wipe_lower
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   └── expo_eraser
│       ├── wipe_higher
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       └── wipe_lower
│           ├── cam_K.txt
│           ├── depth
│           ├── rgb
│           ├── rgb_video.mp4
│           └── subgoals.json
├── hammer
│   ├── hammer_2
│   │   ├── down_swing
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── side_swing
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   └── mallet
│       ├── down_swing
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       ├── down_swing_close
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       ├── down_swing_close_easy
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       └── side_swing
│           ├── cam_K.txt
│           ├── depth
│           ├── rgb
│           ├── rgb_video.mp4
│           └── subgoals.json
├── marker
│   ├── 040_large_marker
│   │   ├── write_c
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── write_smiley
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   ├── sharpie_closed
│   │   ├── write_c
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── write_smiley
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   └── staples_open
│       ├── write_c
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       └── write_smiley
│           ├── cam_K.txt
│           ├── depth
│           ├── rgb
│           ├── rgb_video.mp4
│           └── subgoals.json
├── screwdriver
│   ├── black_screwdriver
│   │   ├── side
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── top
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   ├── real_flat_screwdriver
│   │   ├── side
│   │   │   ├── cam_K.txt
│   │   │   ├── depth
│   │   │   ├── rgb
│   │   │   ├── rgb_video.mp4
│   │   │   └── subgoals.json
│   │   └── top
│   │       ├── cam_K.txt
│   │       ├── depth
│   │       ├── rgb
│   │       ├── rgb_video.mp4
│   │       └── subgoals.json
│   └── red_screwdriver
│       ├── side
│       │   ├── cam_K.txt
│       │   ├── depth
│       │   ├── rgb
│       │   ├── rgb_video.mp4
│       │   └── subgoals.json
│       └── top
│           ├── cam_K.txt
│           ├── depth
│           ├── rgb
│           ├── rgb_video.mp4
│           └── subgoals.json
└── spatula
    ├── black_spatula
    │   ├── flip_pancake
    │   │   ├── cam_K.txt
    │   │   ├── depth
    │   │   ├── rgb
    │   │   ├── rgb_video.mp4
    │   │   └── subgoals.json
    │   ├── flip_pancake_easy
    │   │   ├── cam_K.txt
    │   │   ├── depth
    │   │   ├── rgb
    │   │   ├── rgb_video.mp4
    │   │   └── subgoals.json
    │   └── serve_plate
    │       ├── cam_K.txt
    │       ├── depth
    │       ├── rgb
    │       ├── rgb_video.mp4
    │       └── subgoals.json
    └── spoon_spatula
        ├── flip_pancake
        │   ├── cam_K.txt
        │   ├── depth
        │   ├── rgb
        │   ├── rgb_video.mp4
        │   └── subgoals.json
        ├── flip_pancake_easy
        │   ├── cam_K.txt
        │   ├── depth
        │   ├── rgb
        │   ├── rgb_video.mp4
        │   └── subgoals.json
        └── serve_plate
            ├── cam_K.txt
            ├── depth
            ├── rgb
            ├── rgb_video.mp4
            └── subgoals.json
"""

# Get all subgoals JSONs
src_dir = Path("/juno/u/kedia/FoundationPose/human_videos/Jan_17")
assert src_dir.exists(), f"Source directory not found: {src_dir}"
subgoals_json_paths = list(src_dir.glob("**/subgoals.json"))
print(f"Found {len(subgoals_json_paths)} subgoals JSONs in {src_dir}")
breakpoint()

dst_dir = Path("dex_tool_bench/evaluation_trajectories")
assert dst_dir.exists(), f"Destination directory not found: {dst_dir}"

# Copy the subgoals JSONs to the evaluation_trajectories directory
raw_robot_frame_json_paths = []
for subgoals_json_path in tqdm(subgoals_json_paths, desc="Copying subgoals JSONs to evaluation_trajectories"):
    traj_name = subgoals_json_path.parent.name
    object_name = subgoals_json_path.parent.parent.name
    object_type = subgoals_json_path.parent.parent.parent.name
    new_dir = dst_dir / object_type / object_name
    assert new_dir.exists(), f"New directory not found: {new_dir}"
    new_file = new_dir / f"{traj_name}_raw_robot_frame.json"
    raw_robot_frame_json_paths.append(new_file)
    cp_cmd = f"cp {subgoals_json_path} {new_file}"
    print(f"Running command: {cp_cmd}")
    subprocess.run(cp_cmd, shell=True, check=True)
breakpoint()

# Convert the raw robot frame JSONs to world frame JSONs
raw_world_frame_json_paths = []
for raw_robot_frame_json_path in tqdm(raw_robot_frame_json_paths, desc="Converting raw robot frame JSONs to world frame JSONs"):
    with open(raw_robot_frame_json_path, "r") as f:
        goal_pose_robot_frame_data = json.load(f)
    goal_pose_world_frame_data = [[x, y + 0.8, z, qx, qy, qz, qw] for x, y, z, qx, qy, qz, qw in goal_pose_robot_frame_data]
    new_file = raw_robot_frame_json_path.parent / raw_robot_frame_json_path.name.replace("_raw_robot_frame.json", "_raw_world_frame.json")
    with open(new_file, "w") as f:
        json.dump(
            {
                "start_pose": goal_pose_world_frame_data[0],
                "goals": goal_pose_world_frame_data,
            },
            f,
            indent=4,
        )
    raw_world_frame_json_paths.append(new_file)

breakpoint()

# Process the world frame JSONs to:
# * Start after z >= MIN_Z
# * Downsample by DOWNSAMPLE_FACTOR
MIN_Z = 0.6
DOWNSAMPLE_FACTOR = 10
world_frame_min_z_json_paths = []
world_frame_min_z_downsampled_json_paths = []
for raw_world_frame_json_path in tqdm(raw_world_frame_json_paths, desc="Processing world frame JSONs to start after z >= MIN_Z"):
    with open(raw_world_frame_json_path, "r") as f:
        data = json.load(f)
    goal_pose_world_frame_data = data["goals"]
    for t in range(len(goal_pose_world_frame_data)):
        x, y, z, qx, qy, qz, qw = goal_pose_world_frame_data[t]
        if z >= MIN_Z:
            break
    assert t < len(goal_pose_world_frame_data), f"No goal pose with z >= {MIN_Z} found"
    first_t_with_high_z = t
    print(f"First goal pose with z >= {MIN_Z} is at index {first_t_with_high_z}")
    new_file = raw_world_frame_json_path.parent / raw_world_frame_json_path.name.replace("_raw_world_frame.json", f"_world_frame_min_z_{MIN_Z}.json")
    with open(new_file, "w") as f:
        json.dump(
            {
                "start_pose": data["start_pose"],
                "goals": goal_pose_world_frame_data[first_t_with_high_z:],
            },
            f,
            indent=4,
        )
    world_frame_min_z_json_paths.append(new_file)
    new_file_downsampled = new_file.parent / new_file.name.replace(f"_world_frame_min_z_{MIN_Z}.json", f"_world_frame_min_z_{MIN_Z}_downsampled_{DOWNSAMPLE_FACTOR}.json")
    with open(new_file_downsampled, "w") as f:
        json.dump(
            {
                "start_pose": data["start_pose"],
                "goals": goal_pose_world_frame_data[first_t_with_high_z:][::DOWNSAMPLE_FACTOR],
            },
            f,
            indent=4,
        )
    world_frame_min_z_downsampled_json_paths.append(new_file_downsampled)
breakpoint()