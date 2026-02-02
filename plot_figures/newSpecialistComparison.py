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

# EVAL_BASE_DIR = Path("/share/portal/kk837/sapg/evals/2026-01-30_01-15-26")
# EVAL_BASE_DIR = Path("/share/portal/kk837/sapg/evals/2026-01-30_01-44-43")

# Separate directories for each method
EVAL_DIR_OURS = Path("/share/portal/kk837/sapg/evals/2026-01-30_12-18-21")
EVAL_DIR_SPECIALIST = Path("/share/portal/kk837/sapg/evals/2026-01-30_17-58-38")

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

METHODS = ["simtoolreal", "specialist"]

# Manual overrides for specialist "Obj A / Traj A" values (per task)
# These will be used instead of discovered eval.json values
# Format: task_name -> value (percentage)
SPECIALIST_OBJ_A_TRAJ_A_OVERRIDES = {
    # "hammer": 100.0,
    # "spatula": 100.0,
    # "eraser": 100.0,
    # "screwdriver": 43/46.0*100.0,
    # "marker": 46/51.0*100.0,
    # "brush": 35/39.0*100.0,
    # "brush": 100.0,
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


def analyze_results():
    """Main analysis function - compute averages across all tasks."""
    eval_structure = discover_eval_structure()
    
    print("=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nOurs directory: {EVAL_DIR_OURS}")
    print(f"Specialist directory: {EVAL_DIR_SPECIALIST}")
    print(f"Tasks found: {list(eval_structure.keys())}")
    
    # Store results by setting for aggregation
    aggregated = {
        "Obj A / Traj A": {"simtoolreal": [], "specialist": []},
        "Obj A / Traj B": {"simtoolreal": [], "specialist": []},
        "Obj B / Traj A": {"simtoolreal": [], "specialist": []},
        "Obj B / Traj B": {"simtoolreal": [], "specialist": []},
    }
    
    # Per-task results
    for task_name in sorted(eval_structure.keys()):
        print(f"\n{'=' * 70}")
        print(f"TASK: {task_name.upper()}")
        if task_name in SPECIALIST_CONFIG:
            cfg = SPECIALIST_CONFIG[task_name]
            print(f"  Specialist trained on: {cfg['obj_a']} + {cfg['traj_a']}")
        print("=" * 70)
        
        task_data = eval_structure[task_name]
        
        for obj_name in sorted(task_data.keys()):
            for traj_name in sorted(task_data[obj_name].keys()):
                traj_data = task_data[obj_name][traj_name]
                
                obj_label, traj_label = classify_setting(task_name, obj_name, traj_name)
                setting_key = f"{obj_label} / {traj_label}" if obj_label else "Unknown"
                
                print(f"\n  {obj_name} / {traj_name} [{setting_key}]:")
                
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
                            override_val = SPECIALIST_OBJ_A_TRAJ_A_OVERRIDES[task_name]
                            print(f"    {method:15s}: {override_val:6.2f}% (OVERRIDE, was {result['mean']:.2f}%)")
                            value_to_use = override_val
                        else:
                            print(f"    {method:15s}: {result['mean']:6.2f}% ± {result['stderr']:5.2f}% (n={result['n']})")
                            value_to_use = result["mean"]
                        
                        # Aggregate
                        if setting_key in aggregated:
                            aggregated[setting_key][method].append(value_to_use)
                    else:
                        print(f"    {method:15s}: NOT FOUND")
    
    # Print aggregated results
    print(f"\n{'=' * 70}")
    print("AGGREGATED RESULTS ACROSS ALL TASKS")
    print("=" * 70)
    
    for setting in ["Obj A / Traj A", "Obj A / Traj B", "Obj B / Traj A", "Obj B / Traj B"]:
        print(f"\n{setting}:")
        for method in METHODS:
            values = aggregated[setting][method]
            if values:
                mean = np.mean(values)
                stderr = np.std(values) / np.sqrt(len(values)) if len(values) > 1 else 0
                print(f"  {method:15s}: {mean:6.2f}% ± {stderr:5.2f}% (n_tasks={len(values)})")
            else:
                print(f"  {method:15s}: No data")
    
    # Overall average
    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGE (all settings)")
    print("=" * 70)
    
    for method in METHODS:
        all_values = []
        for setting in aggregated:
            all_values.extend(aggregated[setting][method])
        if all_values:
            mean = np.mean(all_values)
            stderr = np.std(all_values) / np.sqrt(len(all_values)) if len(all_values) > 1 else 0
            print(f"  {method:15s}: {mean:6.2f}% ± {stderr:5.2f}% (n_total={len(all_values)})")
    
    return eval_structure, aggregated


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


def plot_specialist_comparison(aggregated, settings, output_name, figsize=(3.5, 2.8)):
    """Plot specialist comparison for given settings.
    
    Designed for single-column width in two-column paper (~3.5 inches).
    """
    # Compute means and stderrs from aggregated data
    raw_data = {}
    for setting in settings:
        raw_data[setting] = {}
        for method in PLOT_METHODS:
            values = aggregated[setting][method]
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
    width = 0.34  # Slightly narrower for more whitespace
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

    # Y-axis - slight headroom for error bars
    ax.set_ylabel("Task Progress (%)", labelpad=0)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_xlim(-0.5, len(settings) - 0.5)

    # Add vertical dotted line after first bar group (Obj A / Traj A)
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
    # Analyze results and get aggregated data
    eval_structure, aggregated = analyze_results()
    
    # 3-setting version - single column width
    settings_3 = [
        "Obj A / Traj A",
        "Obj A / Traj B",
        "Obj B / Traj A",
    ]
    plot_specialist_comparison(aggregated, settings_3, "new_specialist_comparison_3settings_revised.png", figsize=(3.5, 1.75))
    
    # 4-setting version
    settings_4 = [
        "Obj A / Traj A",
        "Obj A / Traj B",
        "Obj B / Traj A",
        "Obj B / Traj B",
    ]
    plot_specialist_comparison(aggregated, settings_4, "new_specialist_comparison_4settings_revised.png", figsize=(3.5, 1.75))
