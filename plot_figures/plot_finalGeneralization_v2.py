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
CATEGORIES = ["hammer", "marker", "eraser", "brush", "spatula", "screwdriver"]

OBJECT_INSTANCES = {
    "hammer": ["toy_hammer", "mallet"],
    "eraser": ["anvil_eraser", "expo_eraser"],
    "marker": ["sharpie_closed", "staples_open"],
    "screwdriver": ["red_screwdriver", "real_flat_screwdriver"],
    "brush": ["anvil_brush", "red_brush"],
    "spatula": ["black_spatula", "spoon_spatula"],
}

# Short display names for object instances (edit these as needed)
INSTANCE_DISPLAY_NAMES = {
    "toy_hammer": "Claw",
    "mallet": "Mallet",
    "sharpie_closed": "Sharpie",
    "staples_open": "Staples",
    "anvil_eraser": "Handle",
    "expo_eraser": "Flat",
    "red_brush": "Red",
    "anvil_brush": "Blue",
    "black_spatula": "Flat",
    "spoon_spatula": "Spoon",
    "red_screwdriver": "Short",
    "real_flat_screwdriver": "Long",
}

TASK_SUCCESS_RATES = {
    "eraser": {
        "anvil_eraser": {"wipe_smile": 100.0, "wipe_c": 100.0},
        "expo_eraser": {"wipe_smile": 100.0, "wipe_c": 100.0},
    },
    "marker": {
        "sharpie_closed": {"draw_smile": 80.0, "write_c": 77.6},
        "staples_open": {"draw_smile": 90.6, "write_c": 66.2},
    },
    "brush": {
        "red_brush": {"sweep_fwd": 82.7, "sweep_right": 61.4},
        "anvil_brush": {"sweep_fwd": 51.1, "sweep_right": 100.0},
    },
    "spatula": {
        "black_spatula": {"serve_plate": 77.5, "flip_over": 46.1},
        "spoon_spatula": {"serve_plate": 85.0, "flip_over": 95.5},
    },
    "screwdriver": {
        "red_screwdriver": {"spin_vert": 37.9, "spin_horiz": 75.6},
        "real_flat_screwdriver": {"spin_vert": 55.8, "spin_horiz": 61.7},
    },
    "hammer": {
        "toy_hammer": {"swing_down": 100.0, "swing_side": 95.5},
        "mallet": {"swing_down": 84.4, "swing_side": 77.5},
    },
}

# Refined color palette: warm-to-cool gradient following category order
CATEGORY_COLORS = {
    "hammer":      {"base": "#F59E0B", "light": "#FCD34D", "dark": "#D97706"},
    "marker":      {"base": "#F97316", "light": "#FDBA74", "dark": "#EA580C"},
    "eraser":      {"base": "#EC4899", "light": "#F9A8D4", "dark": "#BE185D"},
    "brush":       {"base": "#8B5CF6", "light": "#C4B5FD", "dark": "#6D28D9"},
    "spatula":     {"base": "#3B82F6", "light": "#93C5FD", "dark": "#1D4ED8"},
    "screwdriver": {"base": "#10B981", "light": "#6EE7B7", "dark": "#047857"},
}

# Subtler light/dark variants for instance coloring
CATEGORY_COLORS_SUBTLE = {
    "hammer":      {"light": "#FBBF24", "dark": "#F59E0B", "edge": "#D97706"},
    "marker":      {"light": "#FDBA74", "dark": "#F97316", "edge": "#EA580C"},
    "eraser":      {"light": "#F472B6", "dark": "#DB2777", "edge": "#9D174D"},
    "brush":       {"light": "#A78BFA", "dark": "#7C3AED", "edge": "#5B21B6"},
    "spatula":     {"light": "#60A5FA", "dark": "#2563EB", "edge": "#1D4ED8"},
    "screwdriver": {"light": "#34D399", "dark": "#059669", "edge": "#047857"},
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def add_icon_below_axis(ax, x_pos, icon_path, y_offset=-0.12, zoom=0.045, flip=False):
    """Add an icon below the x-axis at the specified position."""
    try:
        img = mpimg.imread(icon_path)
        if flip:
            img = img[::-1, ::-1]
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(
            imagebox,
            (x_pos, 0),
            xybox=(0, y_offset * 100),
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
    ax.text(
        (x_start + x_end) / 2 - 0.08, -0.29,
        category.upper(),
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=color,
        clip_on=False,
        fontfamily="sans-serif",
    )


def plot_instance_colors_v2(categories, output_name, figsize=(6, 4)):
    """
    Version 2: Task names inside bars starting from x-axis, numbers on top.
    Each object instance has its own color (light vs dark).
    """
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bar_width = 0.22
    task_gap = 0.03
    instance_gap = 0.18
    category_gap = 0.40
    instance_width = 2 * bar_width + task_gap
    
    icon_positions = []
    category_spans = []
    x_cursor = 0.3
    
    for cat_idx, category in enumerate(categories):
        cat_colors = CATEGORY_COLORS[category]
        cat_start = x_cursor
        
        def get_avg(inst):
            tasks = TASK_SUCCESS_RATES.get(category, {}).get(inst, {})
            task_values = list(tasks.values())
            if len(task_values) >= 2:
                return (task_values[0] + task_values[1]) / 2.0
            return 0.0
        
        sorted_instances = sorted(OBJECT_INSTANCES[category], key=get_avg, reverse=True)
        
        for inst_idx, obj_instance in enumerate(sorted_instances):
            obj_tasks = TASK_SUCCESS_RATES.get(category, {}).get(obj_instance, {})
            task_names = list(obj_tasks.keys())
            task_values = list(obj_tasks.values())
            
            v1 = float(task_values[0]) if len(task_values) > 0 else 0.0
            v2 = float(task_values[1]) if len(task_values) > 1 else 0.0
            
            x1 = x_cursor
            x2 = x_cursor + bar_width + task_gap
            
            task1_name = task_names[0] if len(task_names) > 0 else ""
            task2_name = task_names[1] if len(task_names) > 1 else ""
            
            subtle_colors = CATEGORY_COLORS_SUBTLE[category]
            if inst_idx == 0:
                bar_color = subtle_colors["light"]
                edge_color = subtle_colors["edge"]
            else:
                bar_color = subtle_colors["dark"]
                edge_color = subtle_colors["edge"]
            
            # Draw bar 1
            ax.bar(x1, v1, bar_width, color=bar_color, edgecolor=edge_color,
                   linewidth=0.7, zorder=3)
            
            # Task name INSIDE bar with semi-transparent background
            ax.text(x1, 3, task1_name.replace("_", " "), ha="center", va="bottom",
                    fontsize=8, fontweight="normal", color="black", zorder=4,
                    rotation=90, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=1.0, edgecolor="none"))
            
            # Number on TOP of bar
            ax.text(x1, v1 + 1.5, f"{v1:.0f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#374151", zorder=4)
            
            # Draw bar 2
            ax.bar(x2, v2, bar_width, color=bar_color, edgecolor=edge_color,
                   linewidth=0.7, zorder=3)
            
            # Task name INSIDE bar with semi-transparent background
            ax.text(x2, 3, task2_name.replace("_", " "), ha="center", va="bottom",
                    fontsize=8, fontweight="normal", color="black", zorder=4,
                    rotation=90, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=1.0, edgecolor="none"))
            
            # Number on TOP of bar
            ax.text(x2, v2 + 1.5, f"{v2:.0f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#374151", zorder=4)
            
            icon_x = (x1 + x2) / 2
            icon_positions.append({"x": icon_x, "instance": obj_instance})
            x_cursor += instance_width + instance_gap
        
        cat_end = x_cursor - instance_gap
        category_spans.append((cat_start, cat_end, category, cat_colors["dark"]))
        x_cursor += category_gap - instance_gap
    
    ax.set_ylabel("Task Progress (%)", fontweight="500", color="#374151")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([])
    ax.set_xlim(0.0, x_cursor - category_gap + instance_gap - 0.1)
    
    ax.grid(False)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    for icon_info in icon_positions:
        icon_path = PATH_TO_INSTANCE_ICONS / f"{icon_info['instance']}.png"
        is_hammer = icon_info['instance'] in ['toy_hammer', 'mallet']
        icon_y_off = -0.175 if is_hammer else -0.16
        add_icon_below_axis(ax, icon_info["x"], icon_path, y_offset=icon_y_off, zoom=0.065, flip=is_hammer)
        
        instance_name = INSTANCE_DISPLAY_NAMES.get(icon_info['instance'], icon_info['instance'])
        ax.text(
            icon_info["x"], -0.21,
            instance_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#374151",
            clip_on=False,
        )
    
    for cat_start, cat_end, category, color in category_spans:
        draw_category_bracket(ax, cat_start, cat_end, category, color)
    
    plt.subplots_adjust(bottom=0.30, top=0.92, left=0.10, right=0.97)
    
    output_path = BASE_DIR / "plot_drafts" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Figure saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_instance_colors_v2(CATEGORIES, "final_figures/verbs_generalization_instance_colors_v2.png", figsize=(10, 4))
