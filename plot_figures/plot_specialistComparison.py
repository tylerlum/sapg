import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

EVAL_BASE_DIR = Path("/share/portal/kk837/sapg/evals/2026-01-22_13-34-24/hammer")

# Mapping: Human Obj = hammer_2, Novel Obj = mallet
# Human Task = down_swing, Novel Task = side_swing
# Specialist = human2sim2robot, Ours = 12days_of_training

EVAL_PATHS = {
    "Human Obj\nHuman Task": {
        "Specialist": EVAL_BASE_DIR / "hammer_2/down_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "hammer_2/down_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Human Obj\nNovel Task": {
        "Specialist": EVAL_BASE_DIR / "hammer_2/side_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "hammer_2/side_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Novel Obj\nHuman Task": {
        "Specialist": EVAL_BASE_DIR / "mallet/down_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "mallet/down_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
    "Novel Obj\nNovel Task": {
        "Specialist": EVAL_BASE_DIR / "mallet/side_swing_world_frame_min_z_0.6_downsampled_10/human2sim2robot/eval.json",
        "Ours": EVAL_BASE_DIR / "mallet/side_swing_world_frame_min_z_0.6_downsampled_10/12days_of_training/eval.json",
    },
}

# 3-setting version (original)
SETTINGS_3 = [
    "Human Obj\nHuman Task",
    "Novel Obj\nHuman Task",
    "Novel Obj\nNovel Task",
]

# 4-setting version (includes Human Obj + Novel Task)
SETTINGS_4 = [
    "Human Obj\nHuman Task",
    "Novel Obj\nHuman Task",
    "Human Obj\nNovel Task",
    "Novel Obj\nNovel Task",
]

METHODS = ["Specialist", "Ours"]


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


def plot_specialist_comparison(settings, output_name, figsize=(7, 4)):
    """Plot specialist comparison for given settings."""
    raw_data = load_eval_data(settings)
    
    # Modern color palette
    colors = {
        "Specialist": "#FFA07A",  # Light salmon
        "Ours": "#20B2AA",  # Light sea green
    }

    # Wide aspect ratio (longer width, shorter height)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    x = np.arange(len(settings))
    width = 0.35
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
            label=method,
            color=colors[method],
            alpha=0.9,
            capsize=4,
            error_kw={
                "elinewidth": 1.5,
                "capthick": 1.5,
                "alpha": 0.7,
            },
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

    # Formatting
    ax.set_ylabel("Task Progress (%)", fontweight="bold", labelpad=8)
    ax.set_ylim(0, 105)

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=12)
    ax.set_xlim(-0.5, len(settings) - 0.5)

    # Aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.8, color="#CCCCCC", zorder=0)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="major", length=5, width=1.2, colors="#333333")
    ax.tick_params(axis="x", length=0)

    # Legend - horizontal at top
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=len(METHODS),
        frameon=True,
        framealpha=0.95,
        edgecolor="#333333",
        fontsize=12,
    )

    output_path = Path(__file__).parent / "plot_drafts" / output_name
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
    # 3-setting version (original)
    plot_specialist_comparison(SETTINGS_3, "specialist_comparison_3settings.png", figsize=(7, 4))
    
    # 4-setting version (includes Human Obj + Novel Task)
    plot_specialist_comparison(SETTINGS_4, "specialist_comparison_4settings.png", figsize=(9, 4))
