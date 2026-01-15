import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# The X-Axis values (e.g., Checkpoints at 0, 100k, 200k... 1M steps)
# Ensure your data lists below match the length of this list (10 items).
ENV_STEPS = [0, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5] 

# PASTE YOUR REAL DATA HERE
# Structure: "Category" -> "Method" -> "mean" (Y-value) and "std" (Error bar size)
# If you don't have standard deviation, you can set 'std' to a list of zeros.
RAW_DATA = {
    "Hammer": {
        "In-Distribution (Primitives)": {
            "mean": [5, 12, 25, 35, 48, 60, 72, 85, 92, 95],
            "std":  [2, 3,  4,  5,  5,  6,  5,  4,  3,  2]
        },
        "Real-World Scanned Objects": {
            "mean": [2, 5, 15, 22, 30, 45, 55, 65, 70, 75],
            "std":  [1, 2, 3,  4,  5,  6,  7,  6,  5,  4]
        }
    },
    "Eraser": {
        "In-Distribution (Primitives)": {
            "mean": [10, 20, 35, 50, 65, 75, 85, 90, 95, 98],
            "std":  [2,  3,  5,  6,  5,  4,  3,  2,  2,  1]
        },
        "Real-World Scanned Objects": {
            "mean": [5, 10, 20, 35, 45, 55, 65, 70, 72, 74],
            "std":  [2, 3,  4,  5,  6,  6,  5,  5,  4,  4]
        }
    },
    "Marker": {
        "In-Distribution (Primitives)": {
            "mean": [8, 18, 30, 45, 60, 72, 80, 88, 93, 96],
            "std":  [2, 4,  5,  6,  5,  4,  3,  3,  2,  1]
        },
        "Real-World Scanned Objects": {
            "mean": [4, 8, 18, 28, 40, 50, 60, 65, 68, 70],
            "std":  [1, 2, 4,  5,  6,  5,  5,  4,  4,  3]
        }
    },
    "Screwdriver": {
        "In-Distribution (Primitives)": {
            "mean": [12, 25, 40, 55, 70, 80, 88, 92, 96, 99],
            "std":  [3,  5,  6,  6,  5,  4,  3,  2,  2,  1]
        },
        "Real-World Scanned Objects": {
            "mean": [8, 15, 25, 38, 50, 60, 70, 75, 80, 82],
            "std":  [2, 3,  5,  6,  7,  6,  5,  4,  3,  3]
        }
    },
    "Brush": {
        "In-Distribution (Primitives)": {
            "mean": [6, 15, 28, 42, 55, 68, 78, 85, 90, 94],
            "std":  [2, 3,  5,  6,  6,  5,  4,  3,  2,  2]
        },
        "Real-World Scanned Objects": {
            "mean": [3, 8, 18, 28, 38, 48, 58, 65, 70, 72],
            "std":  [1, 2, 4,  5,  6,  7,  6,  5,  4,  3]
        }
    },
    "Spatula": {
        "In-Distribution (Primitives)": {
            "mean": [15, 30, 45, 60, 75, 85, 90, 95, 98, 100],
            "std":  [3,  5,  6,  5,  4,  3,  2,  2,  1,  0]
        },
        "Real-World Scanned Objects": {
            "mean": [10, 20, 35, 45, 55, 65, 75, 80, 85, 88],
            "std":  [2,  4,  5,  6,  7,  6,  5,  4,  3,  2]
        }
    }
}

# ==========================================
# 2. PLOTTING SCRIPT
# ==========================================

# Academic Style Settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 13,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
})

def plot_research_strip():
    categories = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]
    
    # 1 Row, 6 Columns wide figure
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    
    # Adjust layout to make room for bottom legend
    plt.subplots_adjust(bottom=0.25, wspace=0.15, left=0.05, right=0.98)

    # Styling for the two specific lines
    styles = {
        "In-Distribution (Primitives)": {"color": "#1f77b4", "marker": "o"}, # Blue Circle
        "Real-World Scanned Objects":   {"color": "#d62728", "marker": "s"}  # Red Square
    }

    # Iterate through the 6 categories
    for i, category in enumerate(categories):
        ax = axes[i]
        
        if category in RAW_DATA:
            cat_data = RAW_DATA[category]
            
            for method_name, style in styles.items():
                if method_name in cat_data:
                    # Retrieve data
                    y_mean = np.array(cat_data[method_name]["mean"])
                    y_std = np.array(cat_data[method_name]["std"])
                    x = np.array(ENV_STEPS)
                    
                    # Plot Line
                    ax.plot(x, y_mean, label=method_name, 
                            color=style["color"], marker=style["marker"], markevery=1)
                    
                    # Plot Shadow (Variance)
                    ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                                    color=style["color"], alpha=0.15, edgecolor='none')

        # Formatting
        ax.set_title(category, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--')
        
        # Clean look (remove top/right borders)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Scientific notation for X axis
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # Y-label only on the first plot
        if i == 0:
            ax.set_ylabel("Task Progress (%)")
        else:
            ax.set_yticklabels([]) # Hide numbers
        
        ax.set_xlabel("Env Steps")

    # Legend (grab handles from the last plotted axis)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), 
               ncol=2, frameon=False)

    plt.show()

if __name__ == "__main__":
    plot_research_strip()