import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure Times New Roman with fallbacks
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

# Separate directories for each method
EVAL_DIR_OURS = Path("/share/portal/kk837/sapg/evals/2026-02-01_03-14-55")
EVAL_DIR_SPECIALIST = Path("/share/portal/kk837/sapg/evals/2026-02-01_03-14-55")

# Specialist definitions: (obj_a, traj_a) for each task category
# The specialist is trained on obj_a + traj_a
SPECIALIST_CONFIG = {
    "hammer": {
        "obj_a": "mallet",
        "traj_a": "down_swing",
    },
    "spatula": {
        "obj_a": "black_spatula",
        "traj_a": "serve_plate",
    },
    "eraser": {
        "obj_a": "expo_eraser",
        "traj_a": "wipe_higher",
    },
    "screwdriver": {
        "obj_a": "real_flat_screwdriver",
        "traj_a": "top",
    },
    "marker": {
        "obj_a": "staples_open",
        "traj_a": "write_smiley",
    },
    "brush": {
        "obj_a": "red_brush",
        "traj_a": "sweep_forward",
    },
}

# Display names for each tool category
TOOL_DISPLAY_NAMES = {
    "hammer": "Hammer",
    "spatula": "Spatula",
    "eraser": "Eraser",
    "screwdriver": "Screwdriver",
    "marker": "Marker",
    "brush": "Brush",
}

METHODS = ["simtoolreal", "specialist"]

# Manual overrides for specialist "Obj A / Traj A" values (per task)
SPECIALIST_OBJ_A_TRAJ_A_OVERRIDES = {
    # Add overrides here if needed
}


def discover_eval_structure():
    """Discover all available eval.json files and organize by task/obj/traj.
    
    Uses EVAL_DIR_OURS for simtoolreal and EVAL_DIR_SPECIALIST for specialist.
    """
    results = {}
    
    # Map method names to their directories
    method_dirs = {
        "simtoolreal": EVAL_DIR_OURS,
        "specialist": EVAL_DIR_SPECIALIST,
    }
    
    # First pass: discover structure from OURS directory (primary)
    for task_dir in EVAL_DIR_OURS.iterdir():
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        results[task_name] = {}
        
        for obj_dir in task_dir.iterdir():
            if not obj_dir.is_dir():
                continue
            obj_name = obj_dir.name
            results[task_name][obj_name] = {}
            
            for traj_dir in obj_dir.iterdir():
                if not traj_dir.is_dir():
                    continue
                # Extract trajectory name (remove suffix)
                traj_full = traj_dir.name
                # e.g. "down_swing_world_frame_min_z_0.6_downsampled_10" -> "down_swing"
                traj_name = traj_full.split("_world_frame")[0]
                
                results[task_name][obj_name][traj_name] = {"traj_full": traj_full}
                
                # Look for each method in its respective directory
                for method_name, base_dir in method_dirs.items():
                    eval_path = base_dir / task_name / obj_name / traj_full / method_name / "eval.json"
                    if eval_path.exists():
                        results[task_name][obj_name][traj_name][method_name] = eval_path
    
    return results


def load_eval_json(eval_path):
    """Load eval.json and return mean and stderr."""
    with open(eval_path, "r") as f:
        data = json.load(f)
    
    episode_pcts = data["episode_goal_pcts"]
    n = len(episode_pcts)
    mean = np.mean(episode_pcts)
    std = np.std(episode_pcts)
    stderr = std / np.sqrt(n) if n > 1 else 0
    
    return {"mean": mean, "stderr": stderr, "n": n, "raw": episode_pcts}


def classify_setting(task_name, obj_name, traj_name):
    """Classify a (obj, traj) pair as Obj A/B and Traj A/B based on specialist config."""
    if task_name not in SPECIALIST_CONFIG:
        return None, None
    
    config = SPECIALIST_CONFIG[task_name]
    obj_a = config["obj_a"]
    traj_a = config["traj_a"]
    
    is_obj_a = (obj_name == obj_a)
    is_traj_a = (traj_name == traj_a)
    
    obj_label = "Obj A" if is_obj_a else "Obj B"
    traj_label = "Traj A" if is_traj_a else "Traj B"
    
    return obj_label, traj_label


def get_per_task_aggregated(eval_structure):
    """Get aggregated results for each task separately.
    
    Returns a dict: task_name -> {setting -> {method -> [values]}}
    """
    per_task = {}
    
    for task_name in eval_structure.keys():
        per_task[task_name] = {
            "Obj A / Traj A": {"simtoolreal": [], "specialist": []},
            "Obj A / Traj B": {"simtoolreal": [], "specialist": []},
            "Obj B / Traj A": {"simtoolreal": [], "specialist": []},
            "Obj B / Traj B": {"simtoolreal": [], "specialist": []},
        }
        
        task_data = eval_structure[task_name]
        
        for obj_name in task_data.keys():
            for traj_name in task_data[obj_name].keys():
                traj_data = task_data[obj_name][traj_name]
                
                obj_label, traj_label = classify_setting(task_name, obj_name, traj_name)
                setting_key = f"{obj_label} / {traj_label}" if obj_label else "Unknown"
                
                if setting_key not in per_task[task_name]:
                    continue
                
                for method in METHODS:
                    if method in traj_data:
                        result = load_eval_json(traj_data[method])
                        
                        # Check for manual override (specialist + Obj A / Traj A)
                        use_override = (
                            method == "specialist" 
                            and setting_key == "Obj A / Traj A" 
                            and task_name in SPECIALIST_OBJ_A_TRAJ_A_OVERRIDES
                        )
                        
                        if use_override:
                            value_to_use = SPECIALIST_OBJ_A_TRAJ_A_OVERRIDES[task_name]
                        else:
                            value_to_use = result["mean"]
                        
                        per_task[task_name][setting_key][method].append(value_to_use)
    
    return per_task


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

# Plot method labels
PLOT_METHODS = ["simtoolreal", "specialist"]
METHOD_LABELS = {
    "simtoolreal": "Ours\n(Zero-Shot)",
    "specialist": "Specialist\n(Trained on Obj A, Traj A)",
}


def plot_single_tool(task_name, task_aggregated, settings, output_name, figsize=(3.5, 1.75)):
    """Plot comparison for a single tool category.
    
    Designed for two-column width in academic paper (~6.5-7 inches).
    Aspect ratio approximately 5:1 (width:height).
    """
    # Compute means and stderrs from aggregated data
    raw_data = {}
    for setting in settings:
        raw_data[setting] = {}
        for method in PLOT_METHODS:
            values = task_aggregated[setting][method]
            if values:
                mean = np.mean(values)
                stderr = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
            else:
                mean = 0
                stderr = 0
            raw_data[setting][method] = {"mean": mean, "std": stderr}
    
    colors = {
        "simtoolreal": "#20B2AA",  # Light sea green (teal)
        "specialist": "#FFA07A",  # Light salmon
    }

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(settings))
    width = 0.34
    n_methods = len(PLOT_METHODS)

    for i, method in enumerate(PLOT_METHODS):
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

    # Title with tool name
    display_name = TOOL_DISPLAY_NAMES.get(task_name, task_name.capitalize())
    ax.set_title(display_name, pad=4)

    # Y-axis - slight headroom for error bars
    ax.set_ylabel("Task Progress (%)", labelpad=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlim(-0.5, len(settings) - 0.5)

    # Add vertical dotted line after first bar group (Obj A / Traj A)
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

    output_dir = Path(__file__).parent / "plot_drafts" / "per_tool_figures"
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


def plot_all_tools():
    """Generate individual plots for all 6 tool categories."""
    # Discover and load evaluation data
    eval_structure = discover_eval_structure()
    
    print("=" * 70)
    print("GENERATING PER-TOOL COMPARISON FIGURES")
    print("=" * 70)
    print(f"\nOurs directory: {EVAL_DIR_OURS}")
    print(f"Specialist directory: {EVAL_DIR_SPECIALIST}")
    print(f"Tasks found: {list(eval_structure.keys())}")
    
    # Get per-task aggregated results
    per_task = get_per_task_aggregated(eval_structure)
    
    # Settings to plot (3 settings only)
    settings = [
        "Obj A / Traj A",
        "Obj A / Traj B",
        "Obj B / Traj A",
    ]
    
    # Figure size: same as original (single-column width)
    figsize = (3.5, 1.75)
    
    # Generate a plot for each tool category
    for task_name in sorted(SPECIALIST_CONFIG.keys()):
        if task_name not in per_task:
            print(f"Warning: No data found for {task_name}")
            continue
        
        output_name = f"{task_name}_comparison.png"
        plot_single_tool(
            task_name,
            per_task[task_name],
            settings,
            output_name,
            figsize=figsize,
        )
    
    print(f"\n{'=' * 70}")
    print("All per-tool figures generated!")
    print("=" * 70)


if __name__ == "__main__":
    plot_all_tools()
