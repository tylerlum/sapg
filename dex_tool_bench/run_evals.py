import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

script_path = Path(__file__).parent / "eval_script.py"
assert script_path.exists(), f"Script not found: {script_path}"
DATE_STR = datetime.now().strftime("%Y-%m-%d")

object_type_to_real_object_names = {
    "hammer": ["hammer_2", "mallet"],
    "spatula": ["black_spatula", "wooden_spatula", "spoon_spatula"],
    "eraser": ["whiteboard_eraser", "anvil_eraser", "expo_eraser", "amazon_eraser"],
    "screwdriver": ["real_flat_screwdriver", "black_screwdriver", "red_screwdriver"],
    "marker": ["040_large_marker", "sharpie_closed", "staples_open"],
    "brush": ["red_brush", "anvil_brush", "lab_brush"],
}

object_type_to_primitive_object_names = {
    "hammer": ["primitive_cuboidal_hammer", "primitive_cylindrical_mallet"],
    "spatula": ["primitive_small_spatula", "primitive_large_spatula"],
    "eraser": ["primitive_small_eraser", "primitive_large_eraser"],
    "screwdriver": ["primitive_cuboidal_screwdriver", "primitive_cylindrical_screwdriver"],
    "marker": ["primitive_thin_marker", "primitive_thick_marker"],
    "brush": ["primitive_frontal_brush", "primitive_sideways_brush"],
}

object_type_to_object_names = {
    object_type: object_type_to_real_object_names[object_type] + object_type_to_primitive_object_names[object_type]
    for object_type in object_type_to_real_object_names.keys()
}

# HACK: Overwrite object names for debugging
# object_type_to_object_names["brush"] = ["red_brush"]

object_type_to_trajectory_names = {
    "hammer": ["horizontal_swing_higher"],
    "spatula": ["pick_and_place_human", "pick_and_place_human_hardinit"],
    "eraser": ["wipe_left_human", "wipe_left_human_2"],
    "screwdriver": ["top_down_screwing_human", "top_down_screwing_human_easyinit"],
    "marker": ["draw_circle_human", "draw_circle_human_hardinit"],
    "brush": ["complex"],
}

# HACK: Overwrite trajectory names for debugging
object_type_to_trajectory_names = {
    object_type: object_type_to_trajectory_names[object_type][:1]
    for object_type in object_type_to_trajectory_names.keys()
}

POLICY_NAME_TO_PATH = {
    "slowSpeed": Path("/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_slowSpeed"),
    "fastSpeed": Path("/juno/u/kedia/sapg/train_dir/latest_checkpoints/tools_fastSpeed"),
}
DOWNSAMPLE_FACTOR = 1
# NUM_EPISODES = 5
NUM_EPISODES = 1  # For debugging

# Validate everything
for policy_path in POLICY_NAME_TO_PATH.values():
    assert policy_path.exists(), f"Policy path not found: {policy_path}"

# Make in one big list
ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME = []
for object_type in object_type_to_real_object_names.keys():
    object_names = object_type_to_object_names[object_type]
    trajectory_names = object_type_to_trajectory_names[object_type]
    for object_name in object_names:
        for trajectory_name in trajectory_names:
            for policy_name, policy_path in POLICY_NAME_TO_PATH.items():
                ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME.append((object_type, object_name, trajectory_name, policy_name))

# Make sure all trajectories exist
evaluation_trajectories_dir = Path(__file__).parent / "evaluation_trajectories"
for object_type, object_name, trajectory_name, _ in ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME:
    trajectory_path = evaluation_trajectories_dir / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory path not found: {trajectory_path}"

print(f"Will evaluate {len(ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME)} combinations for {NUM_EPISODES} episodes each")

"""
Making output_directory structure like
<output_dir_datetime>
|--<object_type>
|   |--<object_name>
|   |   |--<trajectory_name>
|   |   |   |--<policy_name>
|   |   |   |   |--<eval.json>
"""

for object_type, object_name, trajectory_name, policy_name in tqdm(ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME, desc="Running evaluations"):
    print(f"Running evaluation for {object_type} {object_name} {trajectory_name} {policy_name}")
    output_dir = Path(f"evals/{DATE_STR}/{object_type}/{object_name}/{trajectory_name}/{policy_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"python \
        {script_path} \
        --object_type {object_type} \
        --object_name {object_name} \
        --trajectory_name {trajectory_name} \
        --policy_path {policy_path} \
        --output_dir {output_dir} \
        --num_episodes {NUM_EPISODES} \
        --downsample_factor {DOWNSAMPLE_FACTOR}"
    print(f"Running command: {cmd}")
    os.system(cmd)
    print()
