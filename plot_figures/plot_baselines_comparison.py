import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

OBJECTS = ["Hammer", "Brush"]
METHODS = ["Retargeting", "Grasp-Only", "Ours"]

# Values are percentages (0-100)
RAW_DATA = {
    "Hammer": {
        "Retargeting": {"mean": 45, "std": 5},
        "Grasp-Only":  {"mean": 62, "std": 4},
        "Ours":        {"mean": 88, "std": 3}
    },
    "Brush": {
        "Retargeting": {"mean": 30, "std": 6},
        "Grasp-Only":  {"mean": 55, "std": 5},
        "Ours":        {"mean": 92, "std": 2}
    }
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

def plot_bar_comparison():
    colors = {
        "Retargeting": "#A9A9A9",  # Dark Grey
        "Grasp-Only":  "#D3D3D3",  # Light Grey
        "Ours":        "#1f77b4"   # Professional Deep Blue
    }

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    x = np.arange(len(OBJECTS)) 
    width = 0.25 
    multiplier = 0

    for method in METHODS:
        means = [RAW_DATA[obj][method]["mean"] for obj in OBJECTS]
        stds  = [RAW_DATA[obj][method]["std"] for obj in OBJECTS]
        
        offset = width * multiplier
        
        ax.bar(x + offset, means, width, yerr=stds, label=method, 
               color=colors[method], capsize=4, edgecolor='black', linewidth=0.7)
        
        multiplier += 1

    # Formatting
    ax.set_ylabel('Task Progress (%)')
    ax.set_ylim(0, 105)
    
    # Center the labels
    center_offset = width  
    ax.set_xticks(x + center_offset) 
    ax.set_xticklabels(OBJECTS)
    
    # --- CHANGE HERE: Remove the tick marks (the little lines) ---
    ax.tick_params(axis='x', length=0) 

    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0) 
    ax.set_axisbelow(True)

    ax.legend(loc='upper left', frameon=False, ncol=1)

    plt.show()

if __name__ == "__main__":
    plot_bar_comparison()