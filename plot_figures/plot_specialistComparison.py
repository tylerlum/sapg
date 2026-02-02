import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Clear font cache and configure Times New Roman with fallbacks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

EVAL_BASE_DIR = Path("/share/portal/kk837/sapg/evals/2026-01-22_13-34-24/hammer")

# Mapping: Tool A = hammer_2, Tool B = mallet
# Traj A = down_swing, Traj B = side_swing
# Specialist trained on Tool A + Traj A
# Ours = zero-shot policy (12days_of_training)

EVAL_PATHS = {
    "Obj A / Traj A": {
        "Specialist": EVAL_BASE_DIR / "hammer_2/down_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "hammer_2/down_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Obj A / Traj B": {
        "Specialist": EVAL_BASE_DIR / "hammer_2/side_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "hammer_2/side_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Obj B / Traj A": {
        "Specialist": EVAL_BASE_DIR / "mallet/down_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "mallet/down_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Obj B / Traj B": {
        "Specialist": EVAL_BASE_DIR / "mallet/side_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "mallet/side_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
}

# 3-setting version
SETTINGS_3 = [
    "Obj A / Traj A",
    "Obj B / Traj A",
    "Obj A / Traj B",
]

# 4-setting version (all combinations)
# Ordered: training condition first, then vary tool, then vary traj, then both
SETTINGS_4 = [
    "Obj A / Traj A",
    "Obj A / Traj B",
    "Obj B / Traj A",
    "Obj B / Traj B",
]

# Order: Ours first (highlight your method), then baseline
METHODS = ["Ours", "Specialist"]
METHOD_LABELS = {
    "Ours": "Ours\n(Zero-Shot)",
    "Specialist": "Specialist\n(Trained on Obj A, Traj A)",
}


def load_eval_data(settings):
    """Load evaluation data from JSON files and compute mean/stderr."""
    raw_data = {}
    for setting in settings:
        raw_data[setting] = {}
        for method in METHODS:
            eval_path = EVAL_PATHS[setting][method]
            with open(eval_path, "r") as f:
                data = json.load(f)
            
            episode_pcts = data["episode_goal_pcts"]
            n = len(episode_pcts)
            mean = np.mean(episode_pcts)
            std = np.std(episode_pcts)
            stderr = std / np.sqrt(n)  # Standard error = std / sqrt(n)
            raw_data[setting][method] = {"mean": mean, "std": stderr}
    
    return raw_data

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


def plot_specialist_comparison(settings, output_name, figsize=(3.5, 2.8)):
    """Plot specialist comparison for given settings.
    
    Designed for single-column width in two-column paper (~3.5 inches).
    """
    raw_data = load_eval_data(settings)
    
    colors = {
        "Ours": "#20B2AA",  # Light sea green (teal)
        "Specialist": "#FFA07A",  # Light salmon
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(settings))
    width = 0.34  # Slightly narrower for more whitespace
    n_methods = len(METHODS)

    for i, method in enumerate(METHODS):
        means = [raw_data[setting][method]["mean"] for setting in settings]
        stds = [raw_data[setting][method]["std"] for setting in settings]

        offset = (i - (n_methods - 1) / 2) * width

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=METHOD_LABELS[method],
            color=colors[method],
            alpha=0.9,
            capsize=3,
            error_kw={
                "elinewidth": 1.0,
                "capthick": 1.0,
                "alpha": 0.8,
            },
            edgecolor="none",
            zorder=3,
        )

    # Y-axis - slight headroom for error bars
    ax.set_ylabel("Task Progress (%)", labelpad=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlim(-0.5, len(settings) - 0.5)

    # Add vertical dotted line after first bar group (Tool A / Traj A)
    # This separates the specialist's training setting from out-of-distribution settings
    ax.axvline(x=0.5, color="#888888", linestyle=":", linewidth=1.0, zorder=1)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333")
    ax.tick_params(axis="x", length=0)

    # Legend - no box, centered relative to figure, positioned below x-axis labels
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.00),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        borderaxespad=0,
    )
    # Center-align multi-line legend text
    for text in legend.get_texts():
        text.set_multialignment("center")

    output_dir = Path(__file__).parent / "plot_drafts" / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    # 3-setting version - single column width
    plot_specialist_comparison(SETTINGS_3, "specialist_comparison_3settings.png", figsize=(3.5, 1.75))
