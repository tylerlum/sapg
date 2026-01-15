import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

OBJECTS = ["Hammer", "Brush"]
METHODS = ["VideoGen", "VLM", "Human Demo"]

# PASTE YOUR REAL DATA HERE
# Logic: VideoGen and VLM are similar (and lower), Human Demo is best.
RAW_DATA = {
    "Hammer": {
        "VideoGen":   {"mean": 55, "std": 6},
        "VLM":        {"mean": 58, "std": 5}, # Similar to VideoGen
        "Human Demo": {"mean": 92, "std": 3}  # Significantly better
    },
    "Brush": {
        "VideoGen":   {"mean": 48, "std": 5},
        "VLM":        {"mean": 52, "std": 6}, # Similar to VideoGen
        "Human Demo": {"mean": 95, "std": 2}  # Significantly better
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

def plot_task_spec_comparison():
    # Colors:
    # Baselines = Greys (to fade back)
    # Human Demo = Blue (to stand out as the winner)
    colors = {
        "VideoGen":   "#A9A9A9",  # Medium Grey
        "VLM":        "#696969",  # Darker Grey (distinguishes it from VideoGen)
        "Human Demo": "#1f77b4"   # Professional Deep Blue
    }

    # Figure Size: Fits nicely in a single column
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    x = np.arange(len(OBJECTS)) 
    width = 0.25 # Width of bars
    multiplier = 0

    for method in METHODS:
        means = [RAW_DATA[obj][method]["mean"] for obj in OBJECTS]
        stds  = [RAW_DATA[obj][method]["std"] for obj in OBJECTS]
        
        offset = width * multiplier
        
        # Plot Bar
        ax.bar(x + offset, means, width, yerr=stds, label=method, 
               color=colors[method], capsize=4, edgecolor='black', linewidth=0.7,
               zorder=3)
        
        multiplier += 1

    # Formatting
    ax.set_ylabel('Task Progress (%)')
    ax.set_ylim(0, 105)
    
    # Center the labels
    # We have 3 bars. Center is at (width * 3) / 2 = 1.5 * width
    # But multiplier starts at 0, so the middle bar is at index 1.
    # The bars are at: [0], [width], [2*width]. 
    # Center is at [width] (the middle bar).
    center_offset = width  
    ax.set_xticks(x + center_offset) 
    ax.set_xticklabels(OBJECTS)
    
    # Remove tick marks
    ax.tick_params(axis='x', length=0) 

    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0) 
    ax.set_axisbelow(True)

    # Legend location
    # 'upper left' is safe here since the bars on the left (VideoGen) are low.
    ax.legend(loc='upper left', frameon=False, ncol=1)

    plt.show()

if __name__ == "__main__":
    plot_task_spec_comparison()