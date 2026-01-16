import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# The Ablation Categories (X-Axis)
METHODS = ["No Random\nDisturbance", "No Delay", "No SysID", "Ours"]

# PASTE YOUR REAL DATA HERE
# Logic:
# - "No Disturb/Delay": Sim is very high (easier), Real is low (gap).
# - "Ours": Sim and Real are close and high.
RAW_DATA = {
    "No Random\nDisturbance": {
        "sim_mean": 98,
        "sim_std": 2,  # Works perfectly in easy sim
        "real_mean": 30,
        "real_std": 8,  # Fails in real world
    },
    "No Delay": {
        "sim_mean": 95,
        "sim_std": 3,  # Works great in sim
        "real_mean": 45,
        "real_std": 6,  # Latency kills it in real world
    },
    "No SysID": {
        "sim_mean": 90,
        "sim_std": 4,  # Decent sim
        "real_mean": 60,
        "real_std": 5,  # Bad parameters hurt real
    },
    "Ours": {
        "sim_mean": 92,
        "sim_std": 3,
        "real_mean": 89,
        "real_std": 4,  # The gap is tiny!
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
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
        "grid.alpha": 0.3,
    }
)


def plot_sim_real_gap():
    # Colors:
    # Sim = Light, Real = Dark/Solid (The important one)
    colors = {
        "Simulation": "#B0C4DE",  # Light Steel Blue
        "Real World": "#1f77b4",  # The "Ours" Blue
    }

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    x = np.arange(len(METHODS))
    width = 0.35  # Width of individual bar

    # Extract data lists
    sim_means = [RAW_DATA[m]["sim_mean"] for m in METHODS]
    sim_stds = [RAW_DATA[m]["sim_std"] for m in METHODS]

    real_means = [RAW_DATA[m]["real_mean"] for m in METHODS]
    real_stds = [RAW_DATA[m]["real_std"] for m in METHODS]

    # Plot Simulation Bars (Left)
    ax.bar(
        x - width / 2,
        sim_means,
        width,
        yerr=sim_stds,
        label="Simulation",
        color=colors["Simulation"],
        capsize=4,
        edgecolor="black",
        linewidth=0.7,
    )

    # Plot Real World Bars (Right)
    ax.bar(
        x + width / 2,
        real_means,
        width,
        yerr=real_stds,
        label="Real World",
        color=colors["Real World"],
        capsize=4,
        edgecolor="black",
        linewidth=0.7,
    )

    # Formatting
    ax.set_ylabel("Task Progress (%)")
    ax.set_ylim(0, 105)

    # Center X-labels
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)

    # Remove tick marks for clean look
    ax.tick_params(axis="x", length=0)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    # Placed in the gap usually found in these plots (Sim is high, Real is low on the left)
    # So 'upper left' or 'center left' often works well.
    ax.legend(loc="upper right", frameon=False, ncol=1)

    plt.show()


if __name__ == "__main__":
    plot_sim_real_gap()
