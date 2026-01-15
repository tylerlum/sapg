import matplotlib.pyplot as plt
import numpy as np

# --- Configuration for Publication Quality ---
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

def generate_dummy_data(steps=10, difficulty='hard'):
    x = np.linspace(0, 1e6, steps)
    
    # Adjust starting/ending points based on difficulty
    if difficulty == 'easy': # In-distribution (Primitives)
        start_y = np.random.uniform(10, 25)
        end_y = np.random.uniform(85, 98)
        noise_scale = 1.5
    else: # Hard (Real-world scans)
        start_y = np.random.uniform(5, 15)
        end_y = np.random.uniform(70, 90) # Slightly lower performance
        noise_scale = 2.5
    
    noise = np.random.normal(0, noise_scale, steps)
    
    y_mean = np.linspace(start_y, end_y, steps) + noise
    
    # Add a slight "curve" (logarithmic learning) rather than perfectly straight
    # This looks more realistic for ML training
    curve_factor = np.log1p(np.linspace(0, 5, steps)) / np.log1p(5)
    y_mean = start_y + (end_y - start_y) * curve_factor + noise
    
    y_mean = np.clip(y_mean, 0, 100)
    
    # Standard deviation
    y_std = np.linspace(2, 6, steps) + np.random.uniform(0, 1, steps)
    
    return x, y_mean, y_std

def plot_research_strip():
    categories = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]
    
    # 1 Row, 6 Columns. 
    # Adjusted figsize to make sure there is vertical room for the legend
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5), constrained_layout=False)
    
    # Manually adjust spacing to leave room at bottom for legend
    plt.subplots_adjust(bottom=0.25, wspace=0.15, left=0.05, right=0.98)

    # Updated Legend Names
    methods = [
        ("In-Distribution (Primitives)", "#1f77b4", "o", 'easy'),  # Blue, Circle
        ("Real-World Scanned Objects", "#d62728", "s", 'hard')     # Red, Square
    ]

    for i, (ax, category) in enumerate(zip(axes, categories)):
        
        for method_name, color, marker, difficulty in methods:
            x, y, std = generate_dummy_data(10, difficulty)
            
            # Line plot with Markers
            ax.plot(x, y, label=method_name, color=color, marker=marker, markevery=1)
            
            # Fill between
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, edgecolor='none')

        # Formatting
        ax.set_title(category, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Scientific notation for X axis
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        # Y-label only on far left
        if i == 0:
            ax.set_ylabel("Task Progress (%)")
        else:
            ax.set_yticklabels([])
        
        ax.set_xlabel("Env Steps")

    # Legend Handling
    # We grab handles/labels from the first plot
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Legend is placed relative to the FIGURE (fig.legend), not an axis.
    # loc='lower center' combined with bbox_to_anchor=(0.5, 0.02) ensures it sits 
    # right at the bottom margin we created with subplots_adjust.
    fig.legend(handles, labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.05), 
               ncol=2, frameon=False)

    plt.show()

if __name__ == "__main__":
    plot_research_strip()