import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "grid.alpha": 0.3,
    }
)


def plot_combined_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plt.subplots_adjust(wspace=0.25)

    # ============ Left plot: Training Reward ============
    color1 = "#d62728"  # Red

    x1 = np.array(TRAINING_REWARD_ENV_STEPS)
    y1 = np.array(TRAINING_REWARDS)

    ax1.plot(
        x1,
        y1,
        color=color1,
        marker="o",
        markevery=1,
    )

    ax1.set_title("(Train) Episode Reward on Primitives", fontweight="bold")
    # ax1.set_yscale("log")
    ax1.set_xlim(min(TRAINING_REWARD_ENV_STEPS) - 5, max(TRAINING_REWARD_ENV_STEPS) + 5)
    ax1.set_ylim(0, 14000)
    ax1.set_yticks([1000, 3000, 5000, 7000, 9000, 11000, 13000])
    ax1.set_yticklabels(["1k", "3k", "5k", "7k", "9k", "11k", "13k"])
    ax1.grid(True, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_xticks(TRAINING_REWARD_ENV_STEPS)
    ax1.set_ylabel("Episode Reward")
    ax1.set_xlabel("Training Steps on Primitive Objects (Billions)")

    # ============ Right plot: Tool Use Performance on DexToolBench ============
    color2 = "#1f77b4"  # Deep Blue

    y_mean = np.array(AVERAGED_DATA["mean"])
    y_std = np.array(AVERAGED_DATA["std"])
    x2 = np.array(BILLION_ENV_STEPS)

    ax2.errorbar(
        x2,
        y_mean,
        yerr=y_std,
        color=color2,
        marker="o",
        capsize=4,
        capthick=1.5,
        label="Averaged across 6 object categories",
    )

    ax2.set_title("(Test) Performance on DexToolBench", fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(min(BILLION_ENV_STEPS) - 5, max(BILLION_ENV_STEPS) + 5)
    ax2.grid(True, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xticks(BILLION_ENV_STEPS)
    ax2.set_ylabel("Task Progress (%)")
    ax2.set_xlabel("Training Steps on Primitive Objects (Billions)")
    # ax2.legend(loc="lower right")  # Moved to figure caption

    # Save figure to file
    output_path = Path(__file__).parent / "plot_drafts" / "sim_tp_and_reward_vs_env_steps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    plot_combined_figure()
