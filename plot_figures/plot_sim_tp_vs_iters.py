import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for Publication Quality ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,  # Slightly smaller to fit 6 in a row
    'ytick.labelsize': 10,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5,
    'lines.markersize': 6,  # Size of the checkpoint dots
    'grid.alpha': 0.3,
})

def generate_dummy_data(steps=10):
    x = np.linspace(0, 1e6, steps)
    
    start_y = np.random.uniform(5, 15)
    end_y = np.random.uniform(75, 95)
    noise = np.random.normal(0, 2, steps)
    
    y_mean = np.linspace(start_y, end_y, steps) + noise
    y_mean = np.clip(y_mean, 0, 100)
    
    # Standard deviation for shading
    y_std = np.linspace(2, 6, steps) + np.random.uniform(0, 1, steps)
    
    return x, y_mean, y_std

def plot_research_strip():
    categories = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]
    
    # 1 Row, 6 Columns. Wide figure size is crucial here.
    fig, axes = plt.subplots(1, 6, figsize=(24, 4), constrained_layout=True)
    
    methods = [("Our Method", "#1f77b4", "o"), ("Baseline", "#d62728", "s")] # Added marker types

    for i, (ax, category) in enumerate(zip(axes, categories)):
        
        for method_name, color, marker in methods:
            x, y, std = generate_dummy_data(10)
            
            # Line plot with Markers to show checkpoints
            ax.plot(x, y, label=method_name, color=color, marker=marker, markevery=1)
            
            # Fill between for variance
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, edgecolor='none')

        # Formatting
        ax.set_title(category, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--')
        
        # Clean aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Scientific notation for X axis (1e6 steps)
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # Only set Y-label for the first plot
        if i == 0:
            ax.set_ylabel("Task Progress (%)")
        else:
            # Hide Y tick labels for inner plots to save space (optional, but cleaner)
            ax.set_yticklabels([])
        
        # X-label for all (since it's a single row)
        ax.set_xlabel("Env Steps")

    # Legend at the bottom
    # bbox_to_anchor centers it relative to the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0.5, -0.1),  # Push it down below the axis labels
               ncol=2, frameon=False)

    plt.show()

if __name__ == "__main__":
    plot_research_strip()