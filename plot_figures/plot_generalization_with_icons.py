import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import os
from pathlib import Path

# ==========================================
# 1. CONFIGURATION & DATA ENTRY
# ==========================================

# Path to your folder containing hammer.png, eraser.png, etc.
PATH_TO_ICONS = Path(__file__).parent / "tool_icons" 
assert PATH_TO_ICONS.exists(), f"Tool icons directory does not exist: {PATH_TO_ICONS}"

OBJECTS = ["Hammer", "Eraser", "Marker", "Screwdriver", "Brush", "Spatula"]
METHODS = ["In-Distribution (Primitives)", "Real-World Scanned Objects"]

RAW_DATA = {
    "Hammer": { "In-Distribution (Primitives)": {"mean": 95, "std": 2}, "Real-World Scanned Objects":   {"mean": 82, "std": 4} },
    "Eraser": { "In-Distribution (Primitives)": {"mean": 92, "std": 3}, "Real-World Scanned Objects":   {"mean": 78, "std": 5} },
    "Marker": { "In-Distribution (Primitives)": {"mean": 88, "std": 4}, "Real-World Scanned Objects":   {"mean": 70, "std": 6} },
    "Screwdriver": { "In-Distribution (Primitives)": {"mean": 96, "std": 2}, "Real-World Scanned Objects":   {"mean": 85, "std": 3} },
    "Brush": { "In-Distribution (Primitives)": {"mean": 90, "std": 3}, "Real-World Scanned Objects":   {"mean": 75, "std": 5} },
    "Spatula": { "In-Distribution (Primitives)": {"mean": 98, "std": 1}, "Real-World Scanned Objects":   {"mean": 88, "std": 4} }
}

# ==========================================
# 2. HELPER FUNCTION FOR ICONS
# ==========================================

def add_icon_labels(ax, x_coords, labels, zoom=0.08):
    """
    Replaces x-axis text labels with images.
    x_coords: The center position of each group (where the tick used to be)
    labels: The list of category names (Hammer, Brush, etc.)
    zoom: Adjust this to make your icons bigger/smaller
    """
    # Hide the existing text labels
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0) # Remove tick lines

    for x, label in zip(x_coords, labels):
        # Construct filename: "Hammer" -> "./icons/hammer.png"
        filename = f"{label.lower()}.png"
        path = os.path.join(PATH_TO_ICONS, filename)
        
        try:
            # Load Image
            img = mpimg.imread(path)
            
            # Create an "OffsetImage" (Matplotlib's container for images)
            # zoom=0.15 is a good starting point for 512x512 icons. 
            # If your icons are small (64x64), try zoom=0.8 or 1.0
            imagebox = OffsetImage(img, zoom=zoom)
            
            # AnnotationBbox places the imagebox at a specific (x, y) coordinate
            # xy=(x, 0) is the anchor point on the axis
            # xybox=(0, -25) shifts it 25 "points" down (padding)
            ab = AnnotationBbox(imagebox, (x, 0),
                                xybox=(0, -25), 
                                xycoords='data',
                                boxcoords="offset points",
                                frameon=False) # Remove the box border around the image
            ax.add_artist(ab)
            
        except FileNotFoundError:
            print(f"Warning: Could not find {path}. Creating a text label instead.")
            ax.text(x, -5, label, ha='center', va='top', fontsize=10)


# ==========================================
# 3. PLOTTING SCRIPT
# ==========================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3,
})

def plot_with_icons():
    # Colors: Sky Blue (Light) vs Dark Navy (Dark)
    colors = {
        "In-Distribution (Primitives)": "#87CEEB", 
        "Real-World Scanned Objects":   "#1f77b4"  
    }

    # Increased bottom margin to make room for icons
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    plt.subplots_adjust(bottom=0.2) # Reserve bottom 20% for icons

    x = np.arange(len(OBJECTS)) 
    width = 0.35 
    multiplier = 0

    for method in METHODS:
        means = [RAW_DATA[obj][method]["mean"] for obj in OBJECTS]
        stds  = [RAW_DATA[obj][method]["std"] for obj in OBJECTS]
        
        offset = width * multiplier
        
        ax.bar(x + offset, means, width, yerr=stds, label=method, 
               color=colors[method], capsize=4, edgecolor='black', linewidth=0.7,
               zorder=3)
        
        multiplier += 1

    ax.set_ylabel('Task Progress (%)')
    ax.set_ylim(0, 105)
    
    # Calculate center of the groups
    center_offset = width / 2 
    x_centers = x + center_offset
    
    # Set ticks at the center (but we will hide the labels inside the function)
    ax.set_xticks(x_centers)
    
    # --- ADD ICONS HERE ---
    # Adjust 'zoom' depending on your actual image resolution!
    add_icon_labels(ax, x_centers, OBJECTS)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0) 
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', frameon=False, ncol=1)

    plt.show()

if __name__ == "__main__":
    plot_with_icons()