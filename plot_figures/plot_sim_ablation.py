import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Clear font cache and configure Times New Roman with fallbacks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

ENV_STEPS = [0, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5]

# Reordered list: "Ours" is FIRST so it appears at the top of the legend
METHODS = ["Ours", "No SAPG", "No Asymmetric Critic"]

# PASTE YOUR REAL DATA HERE
RAW_DATA = {
    "No Asymmetric Critic": {
        "mean": [5, 12, 18, 24, 28, 32, 35, 38, 40, 41],
        "std": [2, 3, 3, 4, 4, 3, 3, 2, 2, 2],
    },
    "No SAPG": {
        "mean": [5, 18, 30, 42, 52, 60, 65, 68, 70, 72],
        "std": [2, 3, 5, 5, 4, 4, 3, 3, 2, 2],
    },
    "Ours": {
        "mean": [5, 22, 45, 65, 80, 88, 92, 95, 97, 98],
        "std": [2, 4, 5, 5, 4, 3, 2, 1, 1, 1],
    },
}

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


def plot_single_ablation():
    """Plot ablation study.
    
    Designed for single-column width in two-column paper (~3.5 inches).
    """
    # Styles configuration - Ours = Teal (consistent across all figures)
    styles = {
        "Ours": {"color": "#20B2AA", "marker": "o", "zorder": 10},
        "No SAPG": {"color": "#555555", "marker": "s", "zorder": 5},
        "No Asymmetric Critic": {"color": "#A9A9A9", "marker": "^", "zorder": 4},
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)

    # Loop through the METHODS list (which puts "Ours" first in the legend)
    for method in METHODS:
        if method in RAW_DATA:
            data = RAW_DATA[method]
            x = np.array(ENV_STEPS)
            y = np.array(data["mean"])
            std = np.array(data["std"])

            style = styles[method]

            # Plot Line
            ax.plot(
                x,
                y,
                label=method,
                color=style["color"],
                marker=style["marker"],
                markevery=1,
                zorder=style["zorder"],
            )

            # Plot Shade
            ax.fill_between(
                x,
                y - std,
                y + std,
                color=style["color"],
                alpha=0.12,
                edgecolor="none",
                zorder=style["zorder"] - 1,
            )

    # Formatting
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    
    # Subtle horizontal grid
    ax.grid(axis="y", linestyle="--", alpha=0.25, linewidth=0.5, color="#888888", zorder=0)
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # X-axis: convert to millions for cleaner look (consistent with Steps (B) style)
    ax.set_xticks([0, 3e5, 6e5, 9e5])
    ax.set_xticklabels(["0", "0.3", "0.6", "0.9"])

    ax.set_xlabel("Env Steps (M)", labelpad=6)
    ax.set_ylabel("Reward", labelpad=6)

    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333")

    # Legend - no box, positioned below x-axis labels
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0,
    )

    output_dir = Path(__file__).parent / "plot_drafts" / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sim_ablation.png"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    plot_single_ablation()
