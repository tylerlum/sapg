import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# The 4 Generalization Settings (Updated labels to match your request)
SETTINGS = [
    "Demo Obj\nDemo Traj", 
    "Demo Obj\nNovel Traj", 
    "Novel Obj\nDemo Traj", 
    "Novel Obj\nNovel Traj"
]

METHODS = ["Specialized Policy", "Ours"]

# Values are percentages (0-100)
RAW_DATA = {
    "Demo Obj\nDemo Traj": {
        "Specialized Policy": {"mean": 92, "std": 3},
        "Ours":               {"mean": 94, "std": 2} 
    },
    "Demo Obj\nNovel Traj": {
        "Specialized Policy": {"mean": 45, "std": 8}, 
        "Ours":               {"mean": 88, "std": 4}
    },
    "Novel Obj\nDemo Traj": {
        "Specialized Policy": {"mean": 40, "std": 7}, 
        "Ours":               {"mean": 85, "std": 5}
    },
    "Novel Obj\nNovel Traj": {
        "Specialized Policy": {"mean": 15, "std": 5}, 
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
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

def plot_generalization_gap():
    # Colors:
    # Specialized Policy = Dark Charcoal Grey (#555555)
    # Ours = Professional Deep Blue (#1f77b4)
    colors = {
        "Specialized Policy": "#555555", 
        "Ours":               "#1f77b4"
    }

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)

    x = np.arange(len(SETTINGS)) 
    width = 0.35 
    multiplier = 0

    for method in METHODS:
        means = [RAW_DATA[setting][method]["mean"] for setting in SETTINGS]
        stds  = [RAW_DATA[setting][method]["std"] for setting in SETTINGS]
        
        offset = width * multiplier
        
        # Plot Bar
        # Added zorder=3 so bars sit on top of the grid lines
        ax.bar(x + offset, means, width, yerr=stds, label=method, 
               color=colors[method], capsize=4, edgecolor='black', linewidth=0.7,
               zorder=3)
        
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
    # Since the bars on the right are low for the grey baseline, 
    # 'upper right' is a safe open space.
    ax.legend(loc='upper right', frameon=False, ncol=1)

    plt.show()

if __name__ == "__main__":
    plot_generalization_gap()