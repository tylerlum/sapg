import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# The categories (X-axis groups)
OBJECTS = ["Hammer", "Brush"]

# The order of bars (Left -> Right)
# "Ours" is last to be on the rightmost side
METHODS = ["Retargeting", "Grasp-Only", "Ours"]

# PASTE YOUR REAL DATA HERE
# Values are percentages (0-100)
RAW_DATA = {
    "Hammer": {
        "Retargeting": {"mean": 45, "std": 5},
        "Grasp-Only":  {"mean": 62, "std": 4},
        "Ours":        {"mean": 88, "std": 3}  # Highest
    },
    "Brush": {
        "Retargeting": {"mean": 30, "std": 6},
        "Grasp-Only":  {"mean": 55, "std": 5},
        "Ours":        {"mean": 92, "std": 2}  # Highest
    }
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

# Academic Style Settings
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
    # Colors: Muted Greys for baselines, Strong Blue for Ours
    # This guides the reviewer's eye immediately to "Ours"
    colors = {
        "Retargeting": "#A9A9A9",  # Dark Grey
        "Grasp-Only":  "#D3D3D3",  # Light Grey
        "Ours":        "#1f77b4"   # Professional Deep Blue
    }

    # Setup Figure for Single Column (approx 4-5 inches wide)
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # Bar Configuration
    x = np.arange(len(OBJECTS))  # label locations (0, 1)
    width = 0.25  # width of the bars
    multiplier = 0

    # Loop through methods to plot bars
    for method in METHODS:
        means = [RAW_DATA[obj][method]["mean"] for obj in OBJECTS]
        stds  = [RAW_DATA[obj][method]["std"] for obj in OBJECTS]
        
        # Calculate offset for grouped bars
        offset = width * multiplier
        
        # Plot Bar with Error Bars (capsize adds the little T on top)
        rects = ax.bar(x + offset, means, width, yerr=stds, label=method, 
                       color=colors[method], capsize=4, edgecolor='black', linewidth=0.7)
        
        multiplier += 1

    # Formatting
    ax.set_ylabel('Task Progress (%)')
    ax.set_ylim(0, 105)
    
    # Set X-Ticks to be in the center of the group
    # (0 + width) centers it because we have 3 bars starting at 0, 0.25, 0.50
    center_offset = width  
    ax.set_xticks(x + center_offset) 
    ax.set_xticklabels(OBJECTS)
    
    # Aesthetic touches
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0) # Grid behind bars
    ax.set_axisbelow(True) # Ensure grid stays behind bars

    # Legend
    # Placed nicely in the top left or outside if preferred. 
    # For single column, 'upper left' inside the plot usually saves space.
    ax.legend(loc='upper left', frameon=False, ncol=1)

    plt.show()

if __name__ == "__main__":
    plot_bar_comparison()