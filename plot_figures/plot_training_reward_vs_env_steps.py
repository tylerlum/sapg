import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional

# Configure Times New Roman font
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Nimbus Roman', 'Liberation Serif']

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================

BASE_DATA_DIR = Path(__file__).parent / "data" / "seeded_runs"
NUM_SEEDS = 5

# Method name patterns to match in column names
METHOD_PATTERNS = {
    "ours": "00_OURS_",
    "ppo": "00_PPO_",
    "symmetric": "00_SYMMETRIC_",
}

# Metric configurations
METRIC_CONFIG = {
    "reward": {
        "folder": "reward",
        "column_pattern": "rewards/time",
        "ylabel": "Episode Reward",
        "title": "Training Reward Comparison (Ablation Study)",
        "output_file": "training_reward_vs_env_steps.png",
    },
    "success": {
        "folder": "success",
        "column_pattern": "successes/time",
        "ylabel": "Success Rate",
        "title": "Training Success Rate Comparison (Ablation Study)",
        "output_file": "training_success_vs_env_steps.png",
    },
}

# Multiply logged steps by this factor to get true env steps
ENV_STEPS_MULTIPLIER = 16 * 24576  # = 393216


def find_metric_column(df: pd.DataFrame, method_pattern: str, metric_pattern: str) -> Optional[str]:
    """Find the metric column matching a method pattern (excluding MIN/MAX)."""
    for col in df.columns:
        if method_pattern in col and metric_pattern in col and "__MIN" not in col and "__MAX" not in col:
            return col
    return None


def load_seed_data(csv_path: Path, metric_pattern: str) -> dict:
    """
    Load a single seed CSV and extract training data for each method.
    
    Returns:
        dict: {method_name: {"steps": [...], "values": [...]}}
    """
    df = pd.read_csv(csv_path)
    
    data = {}
    for method_name, pattern in METHOD_PATTERNS.items():
        col_name = find_metric_column(df, pattern, metric_pattern)
        if col_name is not None:
            # Get non-null values
            mask = df[col_name].notna() & (df[col_name] != "")
            steps = df.loc[mask, "Step"].astype(float).values
            values = df.loc[mask, col_name].astype(float).values
            
            # Sort by steps
            sort_idx = np.argsort(steps)
            data[method_name] = {
                "steps": steps[sort_idx],
                "values": values[sort_idx],
            }
    
    return data


def load_all_seeds(metric: str) -> dict:
    """
    Load data from all seed files for a given metric.
    
    Returns:
        dict: {method_name: [{"steps": [...], "values": [...]}, ...]} for each seed
    """
    config = METRIC_CONFIG[metric]
    seed_data_dir = BASE_DATA_DIR / config["folder"]
    
    all_data = {method: [] for method in METHOD_PATTERNS.keys()}
    
    for seed_idx in range(NUM_SEEDS):
        csv_path = seed_data_dir / f"seed{seed_idx}.csv"
        seed_data = load_seed_data(csv_path, config["column_pattern"])
        
        for method_name in METHOD_PATTERNS.keys():
            if method_name in seed_data:
                all_data[method_name].append(seed_data[method_name])
    
    return all_data


def interpolate_to_common_steps(all_data: dict, num_points: int = 2000) -> dict:
    """
    Interpolate all seeds to common step values for averaging.
    
    Returns:
        dict: {method_name: {"steps": [...], "values_matrix": np.array (num_seeds x num_points)}}
    """
    interpolated = {}
    
    for method_name, seed_list in all_data.items():
        if not seed_list:
            continue
        
        # Find common step range across all seeds
        min_step = max(data["steps"].min() for data in seed_list)
        max_step = min(data["steps"].max() for data in seed_list)
        
        # Create common step grid
        common_steps = np.linspace(min_step, max_step, num_points)
        
        # Interpolate each seed to common steps
        values_matrix = np.zeros((len(seed_list), num_points))
        for i, data in enumerate(seed_list):
            values_matrix[i] = np.interp(common_steps, data["steps"], data["values"])
        
        interpolated[method_name] = {
            "steps": common_steps,
            "values_matrix": values_matrix,
        }
    
    return interpolated


def smooth_data(values: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Apply rolling average smoothing to reduce noise.
    """
    if len(values) < window_size:
        return values
    
    # Use uniform filter for smoothing
    smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='same')
    # Handle edge effects by using original values at boundaries
    half_window = window_size // 2
    smoothed[:half_window] = values[:half_window]
    smoothed[-half_window:] = values[-half_window:]
    
    return smoothed


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

# Method display names and colors - Ours = Teal (consistent across all figures)
METHOD_DISPLAY_CONFIG = {
    "ours": {"label": "SAPG + Asymmetric Critic (Ours)", "color": "#20B2AA", "marker": "o", "zorder": 10},
    "ppo": {"label": "PPO + Asymmetric Critic", "color": "#555555", "marker": "s", "zorder": 5},
    "symmetric": {"label": "SAPG + Symmetric Critic", "color": "#D2691E", "marker": "^", "zorder": 4},  # Chocolate brown - more visible
}


def plot_training_comparison(metric: str):
    """Plot training curves for a given metric (reward or success)."""
    config = METRIC_CONFIG[metric]
    
    # Load and process data
    all_seed_data = load_all_seeds(metric)
    interpolated_data = interpolate_to_common_steps(all_seed_data)
    
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    
    # Plot in order: ours first for legend ordering
    method_order = ["ours", "ppo", "symmetric"]
    
    for method_name in method_order:
        display_config = METHOD_DISPLAY_CONFIG[method_name]
        if method_name not in interpolated_data:
            continue
        
        steps = interpolated_data[method_name]["steps"]
        values_matrix = interpolated_data[method_name]["values_matrix"]
        
        # Smooth each seed's values
        smoothed_matrix = np.array([
            smooth_data(values_matrix[i], window_size=100)
            for i in range(values_matrix.shape[0])
        ])
        
        # Compute mean and std across seeds
        mean_values = smoothed_matrix.mean(axis=0)
        std_values = smoothed_matrix.std(axis=0)
        
        # Convert to true env steps
        true_env_steps = steps * ENV_STEPS_MULTIPLIER
        
        # Plot mean line
        ax.plot(
            true_env_steps,
            mean_values,
            color=display_config["color"],
            label=display_config["label"],
            linewidth=1.5,
            zorder=display_config["zorder"],
        )
        
        # Plot shaded region for ±1 std
        ax.fill_between(
            true_env_steps,
            mean_values - std_values,
            mean_values + std_values,
            color=display_config["color"],
            alpha=0.2,
            edgecolor="none",
            zorder=display_config["zorder"] - 1,
        )
    
    ax.set_xlabel("Env Steps (B)", labelpad=2)
    ax.set_ylabel(config["ylabel"], labelpad=4)
    ax.set_xlim(0, 9e9)  # End at 9B steps
    
    # Y-axis formatting
    if metric == "reward":
        ax.set_ylim(0, 8500)
        ax.set_yticks([0, 2000, 4000, 6000, 8000])
        ax.set_yticklabels(["0", "2k", "4k", "6k", "8k"])
    else:
        ax.set_ylim(0, None)
    
    # Clean spines (no grid, matching other plots)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    
    ax.tick_params(axis="both", which="major", length=3, width=0.8, colors="#333333")
    
    # X-axis ticks in billions (0, 1B, 2B, ..., 9B)
    ax.set_xticks([i * 1e9 for i in range(10)])
    ax.set_xticklabels(["0"] + [f"{i}" for i in range(1, 10)])
    
    # Legend - inside plot, top left
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
    )
    
    # Save figure to file
    output_dir = Path(__file__).parent / "plot_drafts" / "final_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config["output_file"]
    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
        edgecolor="none",
    )
    plt.close()
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    # Generate both plots
    plot_training_comparison("reward")
    plot_training_comparison("success")
