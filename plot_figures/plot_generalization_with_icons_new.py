import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pathlib import Path

# ==========================================
# 1. CONFIGURATION & DATA
# ==========================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

PATH_TO_INSTANCE_ICONS = BASE_DIR / "object_instance_icons"

# Categories ordered by average performance (descending) for visual narrative
CATEGORIES = ["eraser", "marker", "brush", "spatula", "screwdriver", "hammer"]

OBJECT_INSTANCES = {
    "hammer": ["hammer_2", "mallet"],
    "eraser": ["anvil_eraser", "expo_eraser"],
    "marker": ["sharpie_closed", "staples_open"],
    "screwdriver": ["red_screwdriver", "real_flat_screwdriver"],
    "brush": ["anvil_brush", "red_brush"],
    "spatula": ["black_spatula", "spoon_spatula"],
}

TASK_SUCCESS_RATES = {
    "eraser": {
        "anvil_eraser": {"task1": 100.0, "task2": 100.0},
        "expo_eraser": {"task1": 100.0, "task2": 100.0},
    },
    "marker": {
        "sharpie_closed": {"task1": 80.0, "task2": 78.0},
        "staples_open": {"task1": 91.0, "task2": 66.1},
    },
    "brush": {
        "red_brush": {"task1": 82.0, "task2": 61.6},
        "anvil_brush": {"task1": 51.1, "task2": 100.0},
    },
    "spatula": {
        "black_spatula": {"task1": 75.0, "task2": 48.0},
        "spoon_spatula": {"task1": 85.0, "task2": 95.5},
    },
    "screwdriver": {
        "red_screwdriver": {"task1": 76.0, "task2": 2.6},
        "real_flat_screwdriver": {"task1": 61.68, "task2": 55.81},
    },
    "hammer": {
        "hammer_2": {"task1": 0.0, "task2": 0.0},
        "mallet": {"task1": 0.0, "task2": 0.0},
    },
}

TASK_KEYS = ["task1", "task2"]
TASK_LABELS = {"task1": "Task A", "task2": "Task B"}

# Refined color palette: richer, more saturated, with clear light/dark distinction
# Using a cohesive palette inspired by modern data visualization
CATEGORY_COLORS = {
    "eraser":      {"base": "#F59E0B", "light": "#FCD34D", "dark": "#D97706"},  # Amber
    "marker":      {"base": "#8B5CF6", "light": "#C4B5FD", "dark": "#6D28D9"},  # Violet
    "brush":       {"base": "#EC4899", "light": "#F9A8D4", "dark": "#BE185D"},  # Pink
    "spatula":     {"base": "#10B981", "light": "#6EE7B7", "dark": "#047857"},  # Emerald
    "screwdriver": {"base": "#3B82F6", "light": "#93C5FD", "dark": "#1D4ED8"},  # Blue
    "hammer":      {"base": "#6B7280", "light": "#D1D5DB", "dark": "#374151"},  # Gray (failure)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def add_icon_below_axis(ax, x_pos, icon_path, y_offset=-0.12, zoom=0.045):
    """Add an icon below the x-axis at the specified position."""
    try:
        img = mpimg.imread(icon_path)
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (x_pos, 0),
            xybox=(0, y_offset * 100),  # Convert to points
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            frameon=False,
            pad=0,
            clip_on=False,
        )
        ax.add_artist(ab)
        return True
    except Exception as e:
        print(f"Warning: Could not load icon {icon_path}: {e}")
        return False


def draw_category_bracket(ax, x_start, x_end, category, color):
    """Draw a subtle bracket and label under each category group."""
    y_base = -0.28  # In axes fraction
    
    # Category label
    ax.annotate(
        category.upper(),
        xy=((x_start + x_end) / 2, 0),
        xytext=(0, y_base * 100 - 8),
        xycoords=("data", "axes fraction"),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="600",
        color=color,
        annotation_clip=False,
        fontfamily="sans-serif",
    )


# ==========================================
# 3. MAIN PLOTTING FUNCTION
# ==========================================

def plot_verbs_with_icons():
    """
    Create a refined bar chart showing averaged task success rates across object categories.
    Each category has 2 instances, each with one bar showing the average of both tasks.
    """
    
    # Modern, clean style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.facecolor": "#FAFBFC",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E5E7EB",
        "axes.linewidth": 0.8,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.5,
    })
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#FAFBFC")
    
    # Layout parameters - thicker bars, simpler layout
    bar_width = 0.28
    instance_gap = 0.12       # Gap between instances within category
    category_gap = 0.35       # Gap between categories
    
    icon_positions = []
    category_spans = []
    
    x_cursor = 0.0
    
    for cat_idx, category in enumerate(CATEGORIES):
        colors = CATEGORY_COLORS[category]
        cat_start = x_cursor
        
        # Sort instances by average task progress (descending)
        def get_avg(inst):
            tasks = TASK_SUCCESS_RATES.get(category, {}).get(inst, {})
            return (tasks.get(TASK_KEYS[0], 0.0) + tasks.get(TASK_KEYS[1], 0.0)) / 2.0
        
        sorted_instances = sorted(OBJECT_INSTANCES[category], key=get_avg, reverse=True)
        
        for inst_idx, obj_instance in enumerate(sorted_instances):
            obj_tasks = TASK_SUCCESS_RATES.get(category, {}).get(obj_instance, {})
            v1 = float(obj_tasks.get(TASK_KEYS[0], 0.0))
            v2 = float(obj_tasks.get(TASK_KEYS[1], 0.0))
            
            # Average across both tasks
            avg_val = (v1 + v2) / 2.0
            
            # Bar position
            x_pos = x_cursor
            
            # Draw bar with base color
            ax.bar(
                x_pos, avg_val, bar_width,
                color=colors["base"],
                edgecolor=colors["dark"],
                linewidth=0.8,
                zorder=3,
            )
            
            # Icon position (centered on bar)
            icon_positions.append({"x": x_pos, "instance": obj_instance})
            
            # Move cursor
            x_cursor += bar_width + instance_gap
        
        cat_end = x_cursor - instance_gap
        category_spans.append((cat_start, cat_end, category, colors["dark"]))
        
        x_cursor += category_gap - instance_gap  # Adjust for category gap
    
    # Y-axis styling
    ax.set_ylabel("Task Progress (%)", fontweight="500", color="#374151")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    
    # Remove x-axis ticks (we use icons instead)
    ax.set_xticks([])
    ax.set_xlim(-0.2, x_cursor - category_gap + instance_gap + 0.2)
    
    # Grid styling
    ax.yaxis.grid(True, linestyle="-", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    # Spine styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["left"].set_color("#9CA3AF")
    
    # Add icons below axis
    for icon_info in icon_positions:
        icon_path = PATH_TO_INSTANCE_ICONS / f"{icon_info['instance']}.png"
        add_icon_below_axis(ax, icon_info["x"], icon_path, y_offset=-0.16, zoom=0.06)
    
    # Add category labels
    for cat_start, cat_end, category, color in category_spans:
        draw_category_bracket(ax, cat_start, cat_end, category, color)
    
    # Adjust layout to make room for icons and labels
    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.08, right=0.97)
    
    # Save
    output_path = BASE_DIR / "plot_drafts" / "no_verbs_generalization_with_icons.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_verbs_with_icons()
