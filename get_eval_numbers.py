from pathlib import Path
import json

# evaluation_trajectories_dir = Path("evals/2026-01-17_17-40-36")
# evaluation_trajectories_dir = Path("evals/2026-01-17_18-26-36")
# evaluation_trajectories_dir = Path("evals/2026-01-18_18-46-20")
# evaluation_trajectories_dir = Path("evals/2026-01-18_18-49-05")
evaluation_trajectories_dir = Path("evals/2026-01-20_21-19-47")
assert evaluation_trajectories_dir.exists(), f"Evaluation trajectories directory not found: {evaluation_trajectories_dir}"

"""
└── spatula
    ├── black_spatula
    │   ├── flip_pancake_world_frame_min_z_0.6_downsampled_10
    │   │   └── slowSpeed
    │   │       ├── env_cfg.yaml
    │   │       ├── eval.json
    │   │       └── policy_config.yaml
    │   └── serve_plate_world_frame_min_z_0.6_downsampled_10
    │       └── slowSpeed
    │           ├── env_cfg.yaml
    │           ├── eval.json
    │           └── policy_config.yaml
    └── spoon_spatula
        ├── flip_pancake_world_frame_min_z_0.6_downsampled_10
        │   └── slowSpeed
        │       ├── env_cfg.yaml
        │       ├── eval.json
        │       └── policy_config.yaml
        └── serve_plate_world_frame_min_z_0.6_downsampled_10
            └── slowSpeed
                ├── env_cfg.yaml
                ├── eval.json
                └── policy_config.yaml
"""

eval_jsons = list(evaluation_trajectories_dir.rglob("eval.json"))
object_types, object_names, trajectory_names, policy_names = [], [], [], []
print(f"Found {len(eval_jsons)} eval JSONs")
avg_goal_pct_list = []
avg_time_sec_list = []
for eval_json in eval_jsons:
    policy_name = eval_json.parent.name
    traj_name = eval_json.parent.parent.name
    object_name = eval_json.parent.parent.parent.name
    object_type = eval_json.parent.parent.parent.parent.name
    object_types.append(object_type)
    object_names.append(object_name)
    trajectory_names.append(traj_name)
    policy_names.append(policy_name)
    with open(eval_json, "r") as f:
        data = json.load(f)
    avg_goal_pct = data["avg_goal_pct"]
    avg_time_sec = data["avg_time_sec"]
    avg_goal_pct_list.append(avg_goal_pct)
    avg_time_sec_list.append(avg_time_sec)

# Create a pretty table of results
from tabulate import tabulate
# Create a table with columns: object_type, object_name, trajectory_name, policy_name, avg_goal_pct
table_data = []
for obj_type, obj_name, traj_name, pol_name, avg_pct, avg_time_sec in zip(
    object_types, object_names, trajectory_names, policy_names, avg_goal_pct_list, avg_time_sec_list
):
    table_data.append([obj_type, obj_name, traj_name, pol_name, f"{avg_pct:.2f}%", f"{avg_time_sec:.2f}s"])

headers = ["Object Type", "Object Name", "Trajectory", "Policy", "Avg Goal %", "Avg Time (s)"]
print("\n" + "="*100)
print("EVALUATION RESULTS")
print("="*100)
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print("="*100)

# Print summary statistics
if avg_goal_pct_list:
    print(f"\nSummary Statistics:")
    print(f"  Total evaluations: {len(avg_goal_pct_list)}")
    print(f"  Mean success rate: {sum(avg_goal_pct_list)/len(avg_goal_pct_list):.2f}")
    print(f"  Max success rate: {max(avg_goal_pct_list):.2f}")
    print(f"  Min success rate: {min(avg_goal_pct_list):.2f}")
    print(f"  Mean time: {sum(avg_time_sec_list)/len(avg_time_sec_list):.2f}s")
    print(f"  Max time: {max(avg_time_sec_list):.2f}s")
    print(f"  Min time: {min(avg_time_sec_list):.2f}s")
