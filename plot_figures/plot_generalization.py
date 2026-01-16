import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

OBJECTS = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]
METHODS = ["In-Distribution (Primitives)", "Real-World Scanned Objects"]

RAW_DATA = {
    "Hammer": {
        "In-Distribution (Primitives)": {"mean": 95, "std": 2},
        "Real-World Scanned Objects": {"mean": 82, "std": 4},
    },
    "Eraser": {
        "In-Distribution (Primitives)": {"mean": 92, "std": 3},
        "Real-World Scanned Objects": {"mean": 78, "std": 5},
    },
    "Marker": {
        "In-Distribution (Primitives)": {"mean": 88, "std": 4},
        "Real-World Scanned Objects": {"mean": 70, "std": 6},
    },
    "Screwdriver": {
        "In-Distribution (Primitives)": {"mean": 96, "std": 2},
        "Real-World Scanned Objects": {"mean": 85, "std": 3},
    },
    "Brush": {
        "In-Distribution (Primitives)": {"mean": 90, "std": 3},
        "Real-World Scanned Objects": {"mean": 75, "std": 5},
    },
    "Spatula": {
        "In-Distribution (Primitives)": {"mean": 98, "std": 1},
        "Real-World Scanned Objects": {"mean": 88, "std": 4},
    },
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
)


def plot_full_category_comparison():
    # --- COLOR UPDATE HERE ---
    # Using Shades of Blue to signify "This is the same method"
    colors = {
        "In-Distribution (Primitives)": "#87CEEB",  # Light Blue
        "Real-World Scanned Objects": "#1f77b4",  # Deep Blue
    }

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    x = np.arange(len(OBJECTS))
    width = 0.35
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

    center_offset = width / 2
    ax.set_xticks(x + center_offset)
    ax.set_xticklabels(OBJECTS)

    ax.tick_params(axis="x", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc="upper right", frameon=False, ncol=1)

    plt.show()


if __name__ == "__main__":
    plot_full_category_comparison()
