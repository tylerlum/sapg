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
        "anvil_eraser": {"erase_high": 100.0, "erase_low": 100.0},
        "expo_eraser": {"erase_high": 100.0, "erase_low": 100.0},
    },
    "marker": {
        "sharpie_closed": {"write_smiley": 80.0, "write_c": 78.0},
        "staples_open": {"write_smile": 91.0, "write_c": 66.1},
    },
    "brush": {
        "red_brush": {"sweep_forward": 82.0, "sweep_right": 61.6},
        "anvil_brush": {"sweep_forward": 51.1, "sweep_right": 100.0},
    },
    "spatula": {
        "black_spatula": {"serve_plate": 75.0, "flip_over": 48.0},
        "spoon_spatula": {"serve_plate": 85.0, "flip_over": 95.5},
    },
    "screwdriver": {
        "red_screwdriver": {"spin_vertical": 76.0, "spin_horizontal": 2.6},
        "real_flat_screwdriver": {"spin_vertical": 61.68, "spin_horizontal": 55.81},
    },
    "hammer": {
        "hammer_2": {"swing_down": 0.0, "swing_side": 0.0},
        "mallet": {"swing_down": 0.0, "swing_side": 0.0},
    },
}

# Task names vary by category - we'll extract them dynamically from the data

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

# Subtler light/dark variants for instance coloring (less contrast)
CATEGORY_COLORS_SUBTLE = {
    "eraser":      {"light": "#FBBF24", "dark": "#F59E0B", "edge": "#D97706"},  # Amber: light-amber vs amber
    "marker":      {"light": "#A78BFA", "dark": "#7C3AED", "edge": "#5B21B6"},  # Violet: medium vs deeper
    "brush":       {"light": "#F472B6", "dark": "#DB2777", "edge": "#9D174D"},  # Pink: medium vs deeper
    "spatula":     {"light": "#34D399", "dark": "#059669", "edge": "#047857"},  # Emerald: medium vs deeper
    "screwdriver": {"light": "#60A5FA", "dark": "#2563EB", "edge": "#1D4ED8"},  # Blue: medium vs deeper
    "hammer":      {"light": "#9CA3AF", "dark": "#6B7280", "edge": "#4B5563"},  # Gray: medium vs deeper
}

# Unique colors for each object instance (12 distinct colors)
INSTANCE_COLORS = {
    # Eraser instances
    "anvil_eraser":         {"base": "#F59E0B", "light": "#FCD34D", "dark": "#D97706"},  # Amber
    "expo_eraser":          {"base": "#EA580C", "light": "#FDBA74", "dark": "#C2410C"},  # Orange
    # Marker instances
    "sharpie_closed":       {"base": "#8B5CF6", "light": "#C4B5FD", "dark": "#6D28D9"},  # Violet
    "staples_open":         {"base": "#A855F7", "light": "#E9D5FF", "dark": "#7C3AED"},  # Purple
    # Brush instances
    "red_brush":            {"base": "#EC4899", "light": "#F9A8D4", "dark": "#BE185D"},  # Pink
    "anvil_brush":          {"base": "#F43F5E", "light": "#FDA4AF", "dark": "#E11D48"},  # Rose
    # Spatula instances
    "black_spatula":        {"base": "#10B981", "light": "#6EE7B7", "dark": "#047857"},  # Emerald
    "spoon_spatula":        {"base": "#14B8A6", "light": "#5EEAD4", "dark": "#0F766E"},  # Teal
    # Screwdriver instances
    "red_screwdriver":      {"base": "#3B82F6", "light": "#93C5FD", "dark": "#1D4ED8"},  # Blue
    "real_flat_screwdriver": {"base": "#0EA5E9", "light": "#7DD3FC", "dark": "#0284C7"},  # Sky
    # Hammer instances
    "hammer_2":             {"base": "#6B7280", "light": "#D1D5DB", "dark": "#374151"},  # Gray
    "mallet":               {"base": "#78716C", "light": "#D6D3D1", "dark": "#57534E"},  # Stone
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
# 3. MAIN PLOTTING FUNCTIONS
# ==========================================

def plot_uniform_colors(categories, output_name, figsize=(6, 4)):
    """
    Version 1: Same color for all bars within each category.
    No light/dark distinction between tasks - uniform category color.
    """
    
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
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    
    bar_width = 0.22
    task_gap = 0.03
    instance_gap = 0.18
    category_gap = 0.40
    instance_width = 2 * bar_width + task_gap
    
    icon_positions = []
    category_spans = []
    x_cursor = 0.0
    
    for cat_idx, category in enumerate(categories):
        colors = CATEGORY_COLORS[category]
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
            
            # UNIFORM COLOR: Use base color for all bars
            ax.bar(x1, v1, bar_width, color=colors["base"], edgecolor=colors["dark"],
                   linewidth=0.7, zorder=3)
            ax.text(x1, v1 + 2, task1_name.replace("_", "\n"), ha="center", va="bottom",
                    fontsize=6, fontweight="500", color="#374151", zorder=4)
            
            ax.bar(x2, v2, bar_width, color=colors["base"], edgecolor=colors["dark"],
                   linewidth=0.7, zorder=3)
            ax.text(x2, v2 + 2, task2_name.replace("_", "\n"), ha="center", va="bottom",
                    fontsize=6, fontweight="500", color="#374151", zorder=4)
            
            icon_x = (x1 + x2) / 2
            icon_positions.append({"x": icon_x, "instance": obj_instance})
            x_cursor += instance_width + instance_gap
        
        cat_end = x_cursor - instance_gap
        category_spans.append((cat_start, cat_end, category, colors["dark"]))
        x_cursor += category_gap - instance_gap
    
    ax.set_ylabel("Task Progress (%)", fontweight="500", color="#374151")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([])
    ax.set_xlim(-0.15, x_cursor - category_gap + instance_gap + 0.15)
    
    ax.yaxis.grid(True, linestyle="-", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["left"].set_color("#9CA3AF")
    
    for icon_info in icon_positions:
        icon_path = PATH_TO_INSTANCE_ICONS / f"{icon_info['instance']}.png"
        add_icon_below_axis(ax, icon_info["x"], icon_path, y_offset=-0.16, zoom=0.065)
    
    for cat_start, cat_end, category, color in category_spans:
        draw_category_bracket(ax, cat_start, cat_end, category, color)
    
    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.10, right=0.97)
    
    output_path = BASE_DIR / "plot_drafts" / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def plot_instance_colors(categories, output_name, figsize=(6, 4)):
    """
    Version 2: Each object instance has its own color (light vs dark).
    Both tasks for the same instance share that instance's color.
    Instance 1 in category = light color, Instance 2 = dark color.
    """
    
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
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    
    bar_width = 0.22
    task_gap = 0.03
    instance_gap = 0.18
    category_gap = 0.40
    instance_width = 2 * bar_width + task_gap
    
    icon_positions = []
    category_spans = []
    x_cursor = 0.0
    
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
            
            # Instance 1 (first/higher performing) gets light color
            # Instance 2 (second/lower performing) gets dark color
            # Using subtle palette for less contrast
            subtle_colors = CATEGORY_COLORS_SUBTLE[category]
            if inst_idx == 0:
                bar_color = subtle_colors["light"]
                edge_color = subtle_colors["edge"]
            else:
                bar_color = subtle_colors["dark"]
                edge_color = subtle_colors["edge"]
            
            # Both tasks for this instance use the SAME color
            ax.bar(x1, v1, bar_width, color=bar_color, edgecolor=edge_color,
                   linewidth=0.7, zorder=3)
            ax.text(x1, v1 + 2, task1_name.replace("_", "\n"), ha="center", va="bottom",
                    fontsize=6, fontweight="500", color="#374151", zorder=4)
            
            ax.bar(x2, v2, bar_width, color=bar_color, edgecolor=edge_color,
                   linewidth=0.7, zorder=3)
            ax.text(x2, v2 + 2, task2_name.replace("_", "\n"), ha="center", va="bottom",
                    fontsize=6, fontweight="500", color="#374151", zorder=4)
            
            icon_x = (x1 + x2) / 2
            icon_positions.append({"x": icon_x, "instance": obj_instance})
            x_cursor += instance_width + instance_gap
        
        cat_end = x_cursor - instance_gap
        category_spans.append((cat_start, cat_end, category, cat_colors["dark"]))
        x_cursor += category_gap - instance_gap
    
    ax.set_ylabel("Task Progress (%)", fontweight="500", color="#374151")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([])
    ax.set_xlim(-0.15, x_cursor - category_gap + instance_gap + 0.15)
    
    ax.yaxis.grid(True, linestyle="-", alpha=0.6, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.spines["left"].set_color("#9CA3AF")
    
    for icon_info in icon_positions:
        icon_path = PATH_TO_INSTANCE_ICONS / f"{icon_info['instance']}.png"
        add_icon_below_axis(ax, icon_info["x"], icon_path, y_offset=-0.16, zoom=0.065)
    
    for cat_start, cat_end, category, color in category_spans:
        draw_category_bracket(ax, cat_start, cat_end, category, color)
    
    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.10, right=0.97)
    
    output_path = BASE_DIR / "plot_drafts" / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def plot_category_subset(categories, output_name, figsize=(6, 4)):
    """
    Create a refined bar chart showing both task success rates for a subset of categories.
    Each category has 2 instances, each with two bars (light/dark) for the two tasks.
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
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#FAFBFC")
    
    # Layout parameters - two bars per instance
    bar_width = 0.22
    task_gap = 0.03           # Gap between task bars within an instance
    instance_gap = 0.18       # Gap between instances within category
    category_gap = 0.40       # Gap between categories
    
    # Width of one instance (2 bars + gap between them)
    instance_width = 2 * bar_width + task_gap
    
    icon_positions = []
    category_spans = []
    
    x_cursor = 0.0
    
    for cat_idx, category in enumerate(categories):
        colors = CATEGORY_COLORS[category]
        cat_start = x_cursor
        
        # Sort instances by average task progress (descending)
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
            
            # Bar positions for this instance
            x1 = x_cursor
            x2 = x_cursor + bar_width + task_gap
            
            # Get task names
            task1_name = task_names[0] if len(task_names) > 0 else ""
            task2_name = task_names[1] if len(task_names) > 1 else ""
            
            # Draw task 1 bar (light color)
            ax.bar(
                x1, v1, bar_width,
                color=colors["light"],
                edgecolor=colors["dark"],
                linewidth=0.7,
                zorder=3,
            )
            # Task 1 name on top
            ax.text(
                x1, v1 + 2, task1_name.replace("_", "\n"),
                ha="center", va="bottom",
                fontsize=6, fontweight="500",
                color="#374151", rotation=0,
                zorder=4,
            )
            
            # Draw task 2 bar (dark color)
            ax.bar(
                x2, v2, bar_width,
                color=colors["dark"],
                edgecolor="#1F2937",
                linewidth=0.7,
                zorder=3,
            )
            # Task 2 name on top
            ax.text(
                x2, v2 + 2, task2_name.replace("_", "\n"),
                ha="center", va="bottom",
                fontsize=6, fontweight="500",
                color="#374151", rotation=0,
                zorder=4,
            )
            
            # Icon position (centered between the two bars)
            icon_x = (x1 + x2) / 2
            icon_positions.append({"x": icon_x, "instance": obj_instance})
            
            # Move cursor
            x_cursor += instance_width + instance_gap
        
        cat_end = x_cursor - instance_gap
        category_spans.append((cat_start, cat_end, category, colors["dark"]))
        
        x_cursor += category_gap - instance_gap  # Adjust for category gap
    
    # Y-axis styling
    ax.set_ylabel("Task Progress (%)", fontweight="500", color="#374151")
    ax.set_ylim(0, 125)
    ax.set_yticks([0, 25, 50, 75, 100])
    
    # Remove x-axis ticks (we use icons instead)
    ax.set_xticks([])
    ax.set_xlim(-0.15, x_cursor - category_gap + instance_gap + 0.15)
    
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
        add_icon_below_axis(ax, icon_info["x"], icon_path, y_offset=-0.16, zoom=0.065)
    
    # Add category labels
    for cat_start, cat_end, category, color in category_spans:
        draw_category_bracket(ax, cat_start, cat_end, category, color)
    
    # Adjust layout to make room for icons and labels
    plt.subplots_adjust(bottom=0.22, top=0.92, left=0.10, right=0.97)
    
    # Save
    output_path = BASE_DIR / "plot_drafts" / output_name
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✓ Figure saved to: {output_path}")
    plt.close()


def plot_verbs_with_icons():
    """Generate three plots: two with 3 categories each, and one combined."""
    # Split categories into two groups
    categories_1 = ["eraser", "marker", "brush"]
    categories_2 = ["spatula", "screwdriver", "hammer"]
    all_categories = ["eraser", "marker", "brush", "spatula", "screwdriver", "hammer"]
    
    # Two separate plots (3 categories each)
    plot_category_subset(categories_1, "verbs_generalization_part1.png", figsize=(6, 4))
    plot_category_subset(categories_2, "verbs_generalization_part2.png", figsize=(6, 4))
    
    # Combined plot (all 6 categories)
    plot_category_subset(all_categories, "verbs_generalization_all.png", figsize=(10, 4))


def plot_all_versions():
    """Generate all plot versions including uniform and instance-colored variants."""
    all_categories = ["eraser", "marker", "brush", "spatula", "screwdriver", "hammer"]
    
    # Original version (light/dark by task within category)
    plot_category_subset(all_categories, "verbs_generalization_original.png", figsize=(10, 4))
    
    # Version 1: Uniform category colors (same color for all bars in category)
    plot_uniform_colors(all_categories, "verbs_generalization_uniform_colors.png", figsize=(10, 4))
    
    # Version 2: Unique colors per object instance
    plot_instance_colors(all_categories, "verbs_generalization_instance_colors.png", figsize=(10, 4))


if __name__ == "__main__":
    plot_all_versions()
