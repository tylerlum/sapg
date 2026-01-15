import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

ENV_STEPS = [0, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5] 

# Reordered list: "Ours" is FIRST so it appears at the top of the legend
METHODS = ["Ours", "No SAPG", "No Asymmetric Critic"]

# PASTE YOUR REAL DATA HERE
RAW_DATA = {
    "No Asymmetric Critic": {
        "mean": [5, 12, 18, 24, 28, 32, 35, 38, 40, 41],
        "std":  [2,  3,  3,  4,  4,  3,  3,  2,  2,  2]
    },
    "No SAPG": {
        "mean": [5, 18, 30, 42, 52, 60, 65, 68, 70, 72],
        "std":  [2,  3,  5,  5,  4,  4,  3,  3,  2,  2]
    },
    "Ours": {
        "mean": [5, 22, 45, 65, 80, 88, 92, 95, 97, 98],
        "std":  [2,  4,  5,  5,  4,  3,  2,  1,  1,  1]
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
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'grid.alpha': 0.3,
})

def plot_single_ablation():
    # Styles configuration
    # Ours = Blue
    # No SAPG = Dark Grey (Charcoal)
    # No Asymmetric Critic = Lighter Grey
    styles = {
        "Ours":                 {"color": "#1f77b4", "marker": "o", "zorder": 10}, 
        "No SAPG":              {"color": "#555555", "marker": "s", "zorder": 5}, 
        "No Asymmetric Critic": {"color": "#A9A9A9", "marker": "^", "zorder": 4}  
    }

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    
    # Loop through the METHODS list (which puts "Ours" first in the legend)
    for method in METHODS:
        if method in RAW_DATA:
            data = RAW_DATA[method]
            x = np.array(ENV_STEPS)
            y = np.array(data["mean"])
            std = np.array(data["std"])
            
            style = styles[method]
            
            # Plot Line
            ax.plot(x, y, label=method, 
                    color=style["color"], marker=style["marker"], 
                    markevery=1, zorder=style["zorder"])
            
            # Plot Shade
            ax.fill_between(x, y - std, y + std, 
                            color=style["color"], alpha=0.15, edgecolor='none', 
                            zorder=style["zorder"]-1) 

    # Formatting
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Scientific notation
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    ax.set_xlabel("Env Iters")
    ax.set_ylabel("Reward") 

    # Legend at Upper Left
    ax.legend(loc='upper left', frameon=False)

    plt.show()

if __name__ == "__main__":
    plot_single_ablation()