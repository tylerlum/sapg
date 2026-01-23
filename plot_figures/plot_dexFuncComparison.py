from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

TASKS = ["Brush (easy)", "Brush (hard)"]

METHODS = ["Kinematic Retargeting", "DexFunc", "Ours"]

# Values are percentages (0-100) - dummy values, replace with real data
RAW_DATA = {
    "Brush (easy)": {
        "Kinematic Retargeting": 0,
        "DexFunc": 65.7,
        "Ours": 100,
    },
    "Brush (hard)": {
        "Kinematic Retargeting": 0,
        "DexFunc": 7.9,
        "Ours": 100,
    },
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
)


def plot_bar_comparison_grid():
    # Modern, professional color palette with better contrast
    colors = {
        "Kinematic Retargeting": "#7B68EE",  # Medium slate blue
        "DexFunc": "#FFA07A",  # Light salmon
        "Ours": "#20B2AA",  # Light sea green
    }

    # Create figure with 2 rows, 1 column (Brush easy on top, Brush hard on bottom)
    fig, axes = plt.subplots(2, 1, figsize=(6, 7))
    
    n_methods = len(METHODS)
    width = 0.6
    x_positions = np.arange(n_methods)
    
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        
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
        ax.set_xticklabels(METHODS, fontsize=11)
        ax.set_ylabel("Task Progress (%)", fontweight="bold", labelpad=8, fontsize=13)
        
        # Add description to the left of the plot
        label = "Without\nTool Rotation" if "easy" in task else "With\nTool Rotation"
        ax.text(
            -0.26, 0.5, label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
        )
        
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
    
    # Add legend at the top, centered over task progress label + bars
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colors[method], edgecolor="white", label=method) for method in METHODS]
    fig.legend(
        legend_handles,
        METHODS,
        loc="upper center",
        bbox_to_anchor=(0.56, 1.02),  # Centered over task progress label + bars
        ncol=3,
        frameon=True,
        framealpha=0.95,
        edgecolor="#333333",
        fontsize=11,
        columnspacing=1.5,
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, left=0.22, hspace=0.35)
    
    plt.savefig(
        Path(__file__).parent / "plot_drafts" / "dexfunc_comparison_grid.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(
        f"Saved figure to {Path(__file__).parent / 'plot_drafts' / 'dexfunc_comparison_grid.png'}"
    )


if __name__ == "__main__":
    plot_bar_comparison_grid()
