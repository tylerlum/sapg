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
# assert PATH_TO_ICONS.exists(), f"Tool icons directory does not exist: {PATH_TO_ICONS}"

OBJECTS = ["Hammer", "Brush"]
METHODS = ["Retargeting", "Grasp-Only", "Ours"]

# Values are percentages (0-100)
RAW_DATA = {
    "Hammer": {
        "Retargeting": {"mean": 45, "std": 5},
        "Grasp-Only": {"mean": 62, "std": 4},
        "Ours": {"mean": 88, "std": 3},
    },
    "Brush": {
        "Retargeting": {"mean": 30, "std": 6},
        "Grasp-Only": {"mean": 55, "std": 5},
        "Ours": {"mean": 92, "std": 2},
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
    ax.tick_params(axis="x", length=0)  # Remove tick lines

    for x, label in zip(x_coords, labels):
        # Construct filename: "Hammer" -> "./icons/hammer.png"
        filename = f"{label.lower()}.png"
        path = os.path.join(PATH_TO_ICONS, filename)

        try:
            # Load Image
            img = mpimg.imread(path)

            # Create an "OffsetImage"
            imagebox = OffsetImage(img, zoom=zoom)

            # AnnotationBbox places the imagebox at a specific (x, y) coordinate
            ab = AnnotationBbox(
                imagebox,
                (x, 0),
                xybox=(0, -25),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
            )  # Remove the box border around the image
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


def plot_bar_comparison_with_icons():
    colors = {
        "Retargeting": "#A9A9A9",  # Dark Grey
        "Grasp-Only": "#D3D3D3",   # Light Grey
        "Ours": "#1f77b4",         # Professional Deep Blue
    }

    # Changed constrained_layout to False to manually handle margins for icons
    # Increased height slightly to accommodate icons
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
            zorder=3
        )

        multiplier += 1

    # Formatting
    ax.set_ylabel("Task Progress (%)")
    ax.set_ylim(0, 105)

    # Center the labels
    # We have 3 bars. The center is the middle bar (index 1).
    # [0], [width], [2*width] -> Center is [width]
    center_offset = width
    x_centers = x + center_offset
    
    # Set ticks (labels are hidden inside the helper function)
    ax.set_xticks(x_centers)

    # --- ADD ICONS HERE ---
    # zoom=0.08 matches your previous successful plot
    add_icon_labels(ax, x_centers, OBJECTS, zoom=0.08)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="upper left", frameon=False, ncol=1)

    plt.show()


if __name__ == "__main__":
    plot_bar_comparison_with_icons()