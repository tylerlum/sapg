import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

# Clear font cache and configure Times New Roman with fallbacks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

EVALS_DIR = "/share/portal/kk837/sapg/evals/2026-01-18_04-23-48"

# Days of training values
# Note: 2 days of training = 20B env steps
DAYS_OF_TRAINING = [2, 4, 6, 8, 10, 12]

# Convert days to billion env steps (2 days = 20B env steps)
BILLION_ENV_STEPS = [d * 10 for d in DAYS_OF_TRAINING]

# Training rewards at each checkpoint (for days 2, 4, 6, 8, 10, 12)
TRAINING_REWARD_DAYS = [2, 4, 6, 8, 10, 12]
TRAINING_REWARD_ENV_STEPS = [d * 10 for d in TRAINING_REWARD_DAYS]
TRAINING_REWARDS = [249, 3067, 5851, 8142, 11700, 12989]

# Tasks to plot
TASKS = ["hammer", "eraser", "marker", "screwdriver", "brush", "spatula"]


def load_eval_data(evals_dir: str) -> dict:
    """
    Load all eval.json files for real-world scanned objects, organized by task
    and days of training.
    
    Returns:
        dict: {task: {days: [list of avg_goal_pct values]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    
    evals_path = Path(evals_dir)
    
    for task in TASKS:
        task_dir = evals_path / task
        if not task_dir.exists():
            continue
            
        # Iterate through all tools in this task
        for tool_dir in task_dir.iterdir():
            if not tool_dir.is_dir():
                continue
                
            tool_name = tool_dir.name
            # Skip primitive tools
            if tool_name.startswith("primitive_"):
                continue
            
            # Find all eval.json files under this tool (across all trajectories)
            for eval_file in tool_dir.rglob("eval.json"):
                # Extract days of training from path
                # Path format: task/tool/trajectory/X_days_of_training/eval.json
                days_dir = eval_file.parent.name
                if "days_of_training" in days_dir:
                    days = int(days_dir.split("days")[0])
                    
                    # Load the eval.json
                    try:
                        with open(eval_file, "r") as f:
                            eval_data = json.load(f)
                            avg_goal_pct = eval_data.get("avg_goal_pct", 0)
                            data[task][days].append(avg_goal_pct)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not load {eval_file}: {e}")
    
    return data


def compute_averaged_statistics(data: dict) -> dict:
    """
    Compute mean and std averaged across all tasks for each days value.
    
    First averages within each category, then computes mean and std across
    the 6 category means (variance across categories, not across all samples).
    
    Returns:
        dict: {"mean": [...], "std": [...]}
    """
    means = []
    stds = []
    
    for days in DAYS_OF_TRAINING:
        # Get the mean for each category at this checkpoint
        category_means = []
        for task in TASKS:
            values = data[task][days]
            if values:
                category_means.append(np.mean(values))
        
        if category_means:
            means.append(np.mean(category_means))
            stds.append(np.std(category_means))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    return {"mean": means, "std": stds}


# Load and process the data
raw_data = load_eval_data(EVALS_DIR)
AVERAGED_DATA = compute_averaged_statistics(raw_data)

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

# Font sizes designed for single-column figure (~3.5" wide)
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "grid.alpha": 0.3,
        "mathtext.fontset": "stix",
    }
)


def plot_combined_figure():
    """Plot combined training reward and test performance.
    
    Wider layout with side-by-side subplots to show correlation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw={"wspace": 0.5})

    # ============ Left plot: Training Reward ============
    color_ours = "#20B2AA"  # Teal - consistent "Ours" color

    x1 = np.array(TRAINING_REWARD_ENV_STEPS)
    y1 = np.array(TRAINING_REWARDS)

    ax1.plot(
        x1,
        y1,
        color=color_ours,
        marker="s",  # Square marker
        markersize=4,
        markevery=1,
        zorder=10,
    )

    ax1.set_xlim(min(TRAINING_REWARD_ENV_STEPS) - 5, max(TRAINING_REWARD_ENV_STEPS) + 5)
    ax1.set_ylim(0, 15000)
    ax1.set_yticks([0, 3000, 6000, 9000, 12000, 15000])
    ax1.set_yticklabels(["0", "3k", "6k", "9k", "12k", "15k"])
    
    # Clean spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.8)
    ax1.spines["bottom"].set_linewidth(0.8)
    ax1.spines["left"].set_color("#333333")
    ax1.spines["bottom"].set_color("#333333")
    
    ax1.set_xticks([40, 80, 120])
    ax1.set_ylabel("Episode Reward", labelpad=0, fontsize=8)
    ax1.set_xlabel("Env Steps (B)", labelpad=2, fontsize=8)
    ax1.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333")

    # ============ Right plot: Tool Use Performance on DexToolBench ============
    y_mean = np.array(AVERAGED_DATA["mean"])
    y_std = np.array(AVERAGED_DATA["std"])
    x2 = np.array(BILLION_ENV_STEPS)

    ax2.errorbar(
        x2,
        y_mean,
        yerr=y_std,
        color=color_ours,
        marker="o",  # Circle marker
        markersize=4,
        capsize=2,
        capthick=0.8,
        zorder=10,
    )

    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_xlim(min(BILLION_ENV_STEPS) - 5, max(BILLION_ENV_STEPS) + 5)
    
    # Clean spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.8)
    ax2.spines["bottom"].set_linewidth(0.8)
    ax2.spines["left"].set_color("#333333")
    ax2.spines["bottom"].set_color("#333333")
    
    ax2.set_xticks([40, 80, 120])
    ax2.set_ylabel("Task Progress (%)", labelpad=0, fontsize=8)
    ax2.set_xlabel("Env Steps (B)", labelpad=2, fontsize=8)
    ax2.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333")

    # ============ Add inset images showing train vs test tools ============
    # Load the original combined image for the right half (real tools)
    tools_img_path = Path(__file__).parent / "plot_drafts" / "final_figures" / "Primitives_vs_DexToolBench (1).png"
    # Load the new primitives image
    primitives_img_path = Path(__file__).parent / "plot_drafts" / "final_figures" / "Primitive_Objects.png"
    
    if tools_img_path.exists() and primitives_img_path.exists():
        tools_img = mpimg.imread(tools_img_path)
        primitives_img = mpimg.imread(primitives_img_path)
        
        h, w = tools_img.shape[:2]
        
        # Get the right half (real tools) from original image
        right_half = tools_img[:, w//2:]
        
        # Calculate aspect ratio of original left half to match
        left_half_h, left_half_w = h, w // 2
        target_aspect = left_half_w / left_half_h
        
        # Crop new primitives image to match the aspect ratio of original left half
        prim_h, prim_w = primitives_img.shape[:2]
        current_aspect = prim_w / prim_h
        
        if current_aspect > target_aspect:
            # New image is wider, crop width
            new_w = int(prim_h * target_aspect)
            start_x = (prim_w - new_w) // 2
            left_half = primitives_img[:, start_x:start_x + new_w]
        else:
            # New image is taller, crop height
            new_h = int(prim_w / target_aspect)
            start_y = (prim_h - new_h) // 2
            left_half = primitives_img[start_y:start_y + new_h, :]
        
        # Add inset for training tools (primitives) - bottom right of left plot
        ax1_inset = inset_axes(ax1, width=0.5643, height=0.47025, 
                               bbox_to_anchor=(0.98, -0.03), bbox_transform=ax1.transAxes, loc="lower right")
        ax1_inset.imshow(left_half)
        ax1_inset.axis("off")
        ax1_inset.set_zorder(0)  # Draw below the line plot
        
        # Add inset for test tools (real) - bottom right of right plot
        ax2_inset = inset_axes(ax2, width=0.5643, height=0.47025, 
                               bbox_to_anchor=(0.98, -0.03), bbox_transform=ax2.transAxes, loc="lower right")
        ax2_inset.imshow(right_half)
        ax2_inset.axis("off")
        ax2_inset.set_zorder(0)  # Draw below the line plot

    # Save figure to file
    output_dir = Path(__file__).parent / "plot_drafts" / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sim_tp_and_reward_vs_env_steps.png"
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    plot_combined_figure()
