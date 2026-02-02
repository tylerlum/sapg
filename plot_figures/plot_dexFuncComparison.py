from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

TASKS = ["Brush (easy)", "Brush (hard)"]

METHODS = ["Kinematic\nRetargeting", "Fixed\nGrasp", "SimToolReal\n(ours)"]

# Values are percentages (0-100) - dummy values, replace with real data
# RAW_DATA = {
#     "Brush (easy)": {
#         "Kinematic Retargeting": 0,
#         "DexFunc": 65.7,
#         "Ours": 100,
#     },
#     "Brush (hard)": {
#         "Kinematic Retargeting": 0,
#         "DexFunc": 7.9,
#         "Ours": 100,
#     },
# }

RAW_DATA = {
    "Brush (easy)": {
        "Kinematic\nRetargeting": 8.11,
        "Fixed\nGrasp": 61.02,
        "SimToolReal\n(ours)": 98,
    },
    "Brush (hard)": {
        "Kinematic\nRetargeting": 0,
        "Fixed\nGrasp": 10.81,
        "SimToolReal\n(ours)": 82,
    },
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
)


def plot_bar_comparison_grid():
    # Modern, professional color palette with better contrast
    colors = {
        "Kinematic\nRetargeting": "#7B68EE",  # Medium slate blue
        "Fixed\nGrasp": "#FFA07A",  # Light salmon
        "SimToolReal\n(ours)": "#20B2AA",  # Light sea green
    }

    n_methods = len(METHODS)
    width = 0.6
    x_positions = np.arange(n_methods)
    
    # Create separate figures for each task
    for idx, task in enumerate(TASKS):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        # Extract data for this task
        values = [RAW_DATA[task][method] for method in METHODS]
        
        # Create bars for each method
        bars = ax.bar(
            x_positions,
            values,
            width,
            color=[colors[method] for method in METHODS],
            alpha=0.9,
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )
        
        # Formatting
        ax.set_ylim(0, 105)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(METHODS, fontsize=14)
        ax.set_ylabel("Task Progress (%)", labelpad=2, fontsize=14)
        
        # Aesthetics
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        
        # Improved grid
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.8, color="#CCCCCC", zorder=0)
        ax.set_axisbelow(True)
        
        # Adjust tick parameters
        ax.tick_params(axis="both", which="major", length=5, width=1.2, colors="#333333")
        
        plt.tight_layout()
        
        # Save with task-specific filename
        filename = "dexfunc_comparison_easy.png" if "easy" in task else "dexfunc_comparison_hard.png"
        plt.savefig(
            Path(__file__).parent / "plot_drafts" / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        plt.close()
        print(f"Saved figure to {Path(__file__).parent / 'plot_drafts' / filename}")


if __name__ == "__main__":
    plot_bar_comparison_grid()
