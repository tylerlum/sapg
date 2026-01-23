import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

EVALS_DIR = "/share/portal/kk837/sapg/evals/2026-01-18_04-23-48"

# Days of training values (X-axis)
# Note: 2 days of training = 10B env steps
DAYS_OF_TRAINING = [2, 4, 6, 8, 10, 12]

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


def compute_statistics(data: dict) -> dict:
    """
    Compute mean and std for each task/days combination.
    
    Returns:
        dict: {Task: {"mean": [...], "std": [...]}}
    """
    result = {}
    
    for task in TASKS:
        task_title = task.capitalize()
        means = []
        stds = []
        
        for days in DAYS_OF_TRAINING:
            values = data[task][days]
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        result[task_title] = {
            "mean": means,
            "std": stds,
        }
    
    return result


# Load and process the data
raw_data = load_eval_data(EVALS_DIR)
RAW_DATA = compute_statistics(raw_data)

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
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 13,
        "lines.linewidth": 2.5,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
    }
)


def plot_research_strip():
    tasks = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]

    # 1 Row, 6 Columns wide figure
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))

    # Adjust layout
    plt.subplots_adjust(bottom=0.15, wspace=0.15, left=0.05, right=0.98)

    color = "#1f77b4"  # Deep Blue

    # Iterate through the 6 tasks
    for i, task in enumerate(tasks):
        ax = axes[i]

        if task in RAW_DATA:
            task_data = RAW_DATA[task]
            # Prepend 0 for origin point (0 days = 0 progress)
            y_mean = np.array([0] + task_data["mean"])
            y_std = np.array([0] + task_data["std"])
            x = np.array([0] + DAYS_OF_TRAINING)

            # Plot Line
            ax.plot(
                x,
                y_mean,
                color=color,
                marker="o",
                markevery=1,
            )

            # Plot Shadow (Variance)
            ax.fill_between(
                x,
                y_mean - y_std,
                y_mean + y_std,
                color=color,
                alpha=0.15,
                edgecolor="none",
            )

        # Formatting
        ax.set_title(task, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle="--")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0] + DAYS_OF_TRAINING)

        if i == 0:
            ax.set_ylabel("Task Progress (%)")
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Days of Training")

    # Save figure to file
    output_path = Path(__file__).parent / "sim_tp_vs_days_of_training.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    plot_research_strip()
