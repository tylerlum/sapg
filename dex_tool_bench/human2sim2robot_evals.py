from termcolor import colored
from subprocess import run
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def log_info(text):
    print(colored(text, "cyan"))

script_path = Path(__file__).parent / "eval_script.py"
assert script_path.exists(), f"Script not found: {script_path}"
DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

object_type_to_object_names = {
    "hammer": ["toy_hammer", "mallet"],
    "spatula": ["black_spatula", "spoon_spatula"],
    "eraser": ["anvil_eraser", "expo_eraser"],
    "screwdriver": ["real_flat_screwdriver", "red_screwdriver"],
    "marker": ["staples_open", "sharpie_closed"],
    "brush": ["red_brush", "anvil_brush"],
    # "brush": ["red_brush"],
    # "hammer": ["mallet"],
    # "spatula": ["black_spatula"],
    # "eraser": ["expo_eraser"],
    # "screwdriver": ["real_flat_screwdriver"],
    # "marker": ["staples_open"],
    # "brush": ["red_brush"],
    # "brush": ["anvil_brush"],

    # "Easy"
    # "spatula": ["black_spatula"],
    # "hammer": ["mallet"],
    # "brush": ["red_brush"],
}

object_type_to_trajectory_names = {
    "hammer": ["down_swing", "side_swing"],
    "spatula": ["serve_plate", "flip_pancake"],
    "eraser": ["wipe_higher", "wipe_lower"],
    "screwdriver": ["top", "side"],
    "marker": ["write_smiley", "write_c"],
    "brush": ["sweep_forward", "sweep_forward_right"]
    # "hammer": ["down_swing"],
    # "spatula": ["serve_plate"],
    # "eraser": ["wipe_higher"],
    # "screwdriver": ["top"],
    # "marker": ["write_smiley"],
    # "brush": ["sweep_forward"],

    # "Easy"
    # "spatula": ["flip_pancake_easy"],
    # "hammer": ["down_swing_close_easy", "down_swing_close"],
    # "brush": ["sweep_forward_easy"],
}
APPEND_TO_TRAJECTORY_NAMES = "_world_frame_min_z_0.6_downsampled_10"
# Append stuff to the trajectory names
for object_type in object_type_to_trajectory_names.keys():
    object_type_to_trajectory_names[object_type] = [
        f"{trajectory_name}{APPEND_TO_TRAJECTORY_NAMES}"
        for trajectory_name in object_type_to_trajectory_names[object_type]
    ]

# HACK: Overwrite trajectory names for debugging
# object_type_to_trajectory_names = {
#     object_type: object_type_to_trajectory_names[object_type][:1]
#     for object_type in object_type_to_trajectory_names.keys()
# }

# POLICY_NAME_TO_PATH = {
#     "2days_of_training": Path("/share/portal/kk837/sapg/train_dir/customPretraining/TYLER_HANDLE_HEAD/CONSTANT_DENSITY_NO_ACTION_DELAY_Speed_1.5_2025-12-27_23-07-47"),
#     "4days_of_training": Path("/share/portal/kk837/sapg/train_dir/customPretraining/TYLER_HANDLE_HEAD/FINETUNE_2x_SLOWER_SPEED_noActionDelay_2025-12-30_00-56-18/"),
#     "6days_of_training": Path("/share/portal/kk837/sapg/train_dir/customPretraining/FINETUNE_3x/FINETUNE_3x_SLOW_noActionDelay_2026-01-01_01-28-46"),
#     "8days_of_training": Path("/share/portal/kk837/sapg/train_dir/customPretraining/FINETUNE_4x/FINETUNE_4x_SLOWSPEED_ADD_ACTION_DELAY_2026-01-03_01-32-46"),
#     "10days_of_training": Path("/share/portal/kk837/sapg/train_dir/customPretraining/FINETUNE_5x/FINETUNE_5x_SLOW_SPEED_ADD_ACTION_DELAY_2026-01-05_02-10-22"),
#     "12days_of_training": Path("/share/portal/kk837/sapg/train_dir/LATEST/FINETUNING_1x/NEW_FT_FixedSize_True_Force_True_Scale_2_2026-01-14_23-35-22"),
# }
POLICY_NAME_TO_PATH = {
    "12days_of_training": Path("/share/portal/kk837/sapg/train_dir/LATEST/FINETUNING_1x/NEW_FT_FixedSize_True_Force_True_Scale_2_2026-01-14_23-35-22"),
    "human2sim2robot": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/2x_HAMMER_DOWN_SWING/NoChanges_2026-01-18_20-02-53"),
}

HUMAN2SIM2ROBOT_POLICY_NAME = "specialist"
OURS_POLICY_NAME = "simtoolreal"
OURS_POlICY_PATH = Path("/share/portal/kk837/sapg/train_dir/LATEST/FINETUNING_1x/NEW_FT_FixedSize_True_Force_True_Scale_2_2026-01-14_23-35-22")

SPECIALIST_POLICY_PATH_DICT = {
    "hammer": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/mallet_2026-01-29_02-45-20"),
    "spatula": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/black_spatula_2026-01-29_02-50-18"),
    "eraser": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/expo_eraser_2026-01-29_02-54-29"),
    "screwdriver": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/real_flat_screwdriver_2026-01-29_02-51-41"),
    "marker": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/staples_open_2026-01-29_02-55-48"),
    "brush": Path("/share/portal/kk837/sapg/train_dir/Human2Sim2Robot/1x/red_brush_2026-01-29_02-53-09"),
}


DOWNSAMPLE_FACTOR = 1
# NUM_EPISODES = 5
# NUM_EPISODES = 10
NUM_EPISODES = 10  # For debugging

# Validate everything
for policy_path in POLICY_NAME_TO_PATH.values():
    assert policy_path.exists(), f"Policy path not found: {policy_path}"

# Make in one big list
ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME = []
for object_type in object_type_to_object_names.keys():
    object_names = object_type_to_object_names[object_type]
    trajectory_names = object_type_to_trajectory_names[object_type]
    for object_name in object_names:
        for trajectory_name in trajectory_names:
            # for policy_name, policy_path in POLICY_NAME_TO_PATH.items():
            # evaluate first simtoolreal policy, then specialist policy
            simtoolreal_policy_name = "simtoolreal"
            simtoolreal_policy_path = OURS_POlICY_PATH
            specialist_policy_name = "specialist"
            specialist_policy_path = SPECIALIST_POLICY_PATH_DICT[object_type]
            ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME.append((object_type, object_name, trajectory_name, simtoolreal_policy_name, simtoolreal_policy_path))
            ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME.append((object_type, object_name, trajectory_name, specialist_policy_name, specialist_policy_path))

# Make sure all trajectories exist
evaluation_trajectories_dir = Path(__file__).parent / "evaluation_trajectories"
for object_type, object_name, trajectory_name, _, _ in ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME:
    trajectory_path = evaluation_trajectories_dir / object_type / object_name / f"{trajectory_name}.json"
    assert trajectory_path.exists(), f"Trajectory path not found: {trajectory_path}"

print(f"Will evaluate {len(ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME)} combinations for {NUM_EPISODES} episodes each")

"""
Making output_directory structure like
evals/<datetime>
|--<object_type>
|   |--<object_name>
|   |   |--<trajectory_name>
|   |   |   |--<policy_name>
|   |   |   |   |--<eval.json>
"""

total = len(ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME)
for i, (object_type, object_name, trajectory_name, policy_name, policy_path) in tqdm(enumerate(ALL_OBJECT_TYPE_OBJECT_NAME_TRAJECTORY_NAME_POLICY_NAME), desc="Running evaluations", total=total):
    import time
    start_time = time.time()
    log_info(f"{i}/{total} Running evaluation for {object_type} {object_name} {trajectory_name} {policy_name}")
    output_dir = Path(f"evals/{DATE_STR}/{object_type}/{object_name}/{trajectory_name}/{policy_name}")
    policy_stem = policy_path.name
    checkpoint_path = policy_path / "runs" / f"00_{policy_stem}" / "best" / "model.pth"
    config_path = policy_path / "runs" / f"00_{policy_stem}" / "config.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = f"python \
        {script_path} \
        --object_type {object_type} \
        --object_name {object_name} \
        --trajectory_name {trajectory_name} \
        --checkpoint_path {checkpoint_path} \
        --config_path {config_path} \
        --output_dir {output_dir} \
        --num_episodes {NUM_EPISODES} \
        --downsample_factor {DOWNSAMPLE_FACTOR} \
        --policy_name {policy_name}"
    log_info(f"Running command: {cmd}")
    # run(cmd, shell=True, check=True)
    import os
    os.system(cmd)
    log_info(f"{i}/{total} Done")
    end_time = time.time()
    log_info(f"Time taken for evaluation: {end_time - start_time:.2f} seconds")
