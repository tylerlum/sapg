from subprocess import run
from tqdm import tqdm
from pathlib import Path

"""
tree -L 3 /juno/u/kedia/FoundationPose/human_videos/Jan_15
/juno/u/kedia/FoundationPose/human_videos/Jan_15
├── brush
│   ├── anvil_brush
│   │   └── 20260115_231442
│   ├── lab_brush
│   │   └── 20260115_231622
│   └── red_brush
│       └── 20260115_231110
├── eraser
│   ├── amazon_eraser
│   │   └── 20260115_232955
│   ├── anvil_eraser
│   │   └── 20260115_233226
│   └── expo_eraser
│       └── 20260115_233123
├── hammer
│   ├── hammer_2
│   │   ├── clockwise
│   │   └── counter_clockwise
│   └── mallet
│       ├── clockwise
│       └── counter_clockwise
"""

DEMOS_DIR = Path("/juno/u/kedia/FoundationPose/human_videos/Jan_15")
assert DEMOS_DIR.exists(), f"DEMOS_DIR does not exist: {DEMOS_DIR}"

object_types = ["hammer", "mallet", "brush", "eraser", "screwdriver", "marker"]
object_type_dirs = [DEMOS_DIR / object_type for object_type in object_types if (DEMOS_DIR / object_type).exists()]
demo_dirs = []
for object_type_dir in object_type_dirs:
    object_name_dirs = sorted(list(object_type_dir.iterdir()))
    for object_name_dir in object_name_dirs:
        if not object_name_dir.is_dir():
            continue
        trajectory_dirs = sorted(list(object_name_dir.iterdir()))
        for trajectory_dir in trajectory_dirs:
            if not trajectory_dir.is_dir():
                continue
            demo_dirs.append(trajectory_dir)

print(f"Found {len(demo_dirs)} demo dirs")

for DEMO_DIR in tqdm(demo_dirs, desc="Processing demos"):
    if (DEMO_DIR / "hand_mask").exists() and (DEMO_DIR / "hand_pose_trajectory").exists():
        print(f"Skipping {DEMO_DIR} because it has already been processed")
        continue
    cmd = f"zsh process_demo.sh {DEMO_DIR}"
    print(f"Running command: {cmd}")
    run(cmd, shell=True, check=True)
