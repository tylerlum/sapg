import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# Path to your folder containing hammer.png, eraser.png, etc.
PATH_TO_ICONS = Path(__file__).parent / "tool_icons"
# Ensure the directory exists or change this to your specific path
# assert PATH_TO_ICONS.exists(), f"Tool icons directory does not exist: {PATH_TO_ICONS}"

OBJECTS = ["Hammer", "Brush"]
METHODS = ["VideoGen", "VLM", "Human Demo"]

# PASTE YOUR REAL DATA HERE
RAW_DATA = {
    "Hammer": {
        "VideoGen": {"mean": 55, "std": 6},
        "VLM": {"mean": 58, "std": 5},
        "Human Demo": {"mean": 92, "std": 3},
    },
    "Brush": {
        "VideoGen": {"mean": 48, "std": 5},
        "VLM": {"mean": 52, "std": 6},
        "Human Demo": {"mean": 95, "std": 2},
    },
}

# ==========================================
# 2. HELPER FUNCTION FOR ICONS
# ==========================================


def add_icon_labels(ax, x_coords, labels, zoom=0.08):
    """
    Replaces x-axis text labels with images.
    """
    # Hide the existing text labels
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)

    for x, label in zip(x_coords, labels):
        filename = f"{label.lower()}.png"
        path = os.path.join(PATH_TO_ICONS, filename)

        try:
            img = mpimg.imread(path)
            imagebox = OffsetImage(img, zoom=zoom)

            ab = AnnotationBbox(
                imagebox,
                (x, 0),
                xybox=(0, -25),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
            )
            ax.add_artist(ab)

        except FileNotFoundError:
            print(f"Warning: Could not find {path}. Creating a text label instead.")
            ax.text(x, -5, label, ha="center", va="top", fontsize=10)


# ==========================================
# 3. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
)


def plot_task_spec_comparison_with_icons():
    colors = {
        "VideoGen": "#A9A9A9",  # Medium Grey
        "VLM": "#696969",  # Darker Grey
        "Human Demo": "#1f77b4",  # Professional Deep Blue
    }

    # Changed constrained_layout to False to manually handle margins for icons
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=False)
    plt.subplots_adjust(bottom=0.2)  # Reserve bottom 20% for icons

    x = np.arange(len(OBJECTS))
    width = 0.25
    multiplier = 0

    for method in METHODS:
        means = [RAW_DATA[obj][method]["mean"] for obj in OBJECTS]
        stds = [RAW_DATA[obj][method]["std"] for obj in OBJECTS]

        offset = width * multiplier

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=method,
            color=colors[method],
            capsize=4,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

        multiplier += 1

    ax.set_ylabel("Task Progress (%)")
    ax.set_ylim(0, 105)

    # Calculate Center
    # Bars are at [0], [width], [2*width]. Center is at [width].
    center_offset = width
    x_centers = x + center_offset

    # Set ticks (labels hidden inside helper function)
    ax.set_xticks(x_centers)

    # --- ADD ICONS HERE ---
    # zoom=0.08 matches your previous successful plot
    add_icon_labels(ax, x_centers, OBJECTS, zoom=0.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", frameon=False, ncol=1)

    plt.show()


if __name__ == "__main__":
    plot_task_spec_comparison_with_icons()
