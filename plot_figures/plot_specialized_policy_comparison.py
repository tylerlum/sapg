import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# The 4 Generalization Settings
SETTINGS = [
    "Demo Obj\nDemo Traj", 
    "Demo Obj\nNovel Traj", 
    "Novel Obj\nDemo Traj", 
    "Novel Obj\nNovel Traj"
]

METHODS = ["Specialized Policy", "Ours"]

# Values are percentages (0-100)
# Note: "Specialized" starts high (90) then crashes (45, 40, 15)
# "Ours" stays consistently high.
RAW_DATA = {
    "Demo Obj\nDemo Traj": {
        "Specialized Policy": {"mean": 92, "std": 3},
        "Ours":               {"mean": 94, "std": 2} # Similar performance
    },
    "Demo Obj\nNovel Traj": {
        "Specialized Policy": {"mean": 45, "std": 8}, # Big drop
        "Ours":               {"mean": 88, "std": 4}
    },
    "Novel Obj\nDemo Traj": {
        "Specialized Policy": {"mean": 40, "std": 7}, # Big drop
        "Ours":               {"mean": 85, "std": 5}
    },
    "Novel Obj\nNovel Traj": {
        "Specialized Policy": {"mean": 15, "std": 5}, # Catastrophic failure
        "Ours":               {"mean": 82, "std": 6}
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
    'xtick.labelsize': 10, # Slightly smaller for 4 labels
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

def plot_generalization_gap():
    colors = {
        "Specialized Policy": "#d62728", # Red (signals overfitting/failure later)
        "Ours":               "#1f77b4"  # The same Deep Blue
    }

    # Figure size: slightly wider than the previous one to fit 4 groups
    # but still fits easily in a column or 2/3rds of a column.
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    x = np.arange(len(SETTINGS)) 
    width = 0.35 # Slightly wider bars since we only have 2 per group
    multiplier = 0

    # Center alignment calculation
    # We have 2 bars. The center of the group is at x + width/2
    
    for method in METHODS:
        means = [RAW_DATA[setting][method]["mean"] for setting in SETTINGS]
        stds  = [RAW_DATA[setting][method]["std"] for setting in SETTINGS]
        
        offset = width * multiplier
        
        # Plot Bar
        ax.bar(x + offset, means, width, yerr=stds, label=method, 
               color=colors[method], capsize=4, edgecolor='black', linewidth=0.7,
               zorder=3) # zorder=3 ensures bars are on top of grid
        
        multiplier += 1

    # Formatting
    ax.set_ylabel('Task Progress (%)')
    ax.set_ylim(0, 105)
    
    # Center the labels: The bars start at x and x+width. 
    # Center is exactly between them: x + width/2
    ax.set_xticks(x + width / 2) 
    ax.set_xticklabels(SETTINGS)
    
    # Remove tick marks
    ax.tick_params(axis='x', length=0) 

    # Aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0) 
    ax.set_axisbelow(True)

    # Legend location
    # 'upper right' is good here because the bars on the right might be lower 
    # for the baseline, but Ours is high. 'lower left' is usually safe too.
    # Let's try upper right but give it a semi-transparent background just in case.
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='white')

    plt.show()

if __name__ == "__main__":
    plot_generalization_gap()