"""Generate multiple primitive tools for the dex_tool_bench.

This script defines various tool configurations and generates both URDF and OBJ files
for each tool using the functions from create_obj_urdf.py.
"""

from pathlib import Path
from typing import List

from create_obj_urdf import (
    Cuboid,
    Cylinder,
    ToolConfig,
    create_tool,
)
import shutil


# Base output directory for all generated tools
# BASE_OUTPUT_DIR = Path("/share/portal/kk837/sapg/assets/urdf/dex_tool_bench")
BASE_OUTPUT_DIR = Path("/home/tylerlum/github_repos/sapg/assets/urdf/dex_tool_bench")


# =============================================================================
# Hammer Configurations
# =============================================================================

HAMMER_CONFIGS: List[ToolConfig] = [
    # Cuboidal hammer: cuboid handle + cuboid head
    ToolConfig(
        name="primitive_cuboidal_hammer",
        handle=Cuboid(length=0.25, width=0.03, height=0.02),
        head=Cuboid(length=0.02, width=0.11, height=0.02),
        tool_type="hammer",
    ),
    # Cylindrical mallet: cylinder handle + cylinder head (wider/thicker head)
    ToolConfig(
        name="primitive_cylindrical_mallet",
        handle=Cylinder(length=0.2, radius=0.01),
        head=Cylinder(length=0.10, radius=0.02),
        tool_type="hammer",
    ),
]


# =============================================================================
# Screwdriver Configurations
# =============================================================================

SCREWDRIVER_CONFIGS: List[ToolConfig] = [
    # Cuboidal screwdriver: cuboid handle (grip) + cuboid head (flat shaft)
    ToolConfig(
        name="primitive_cuboidal_screwdriver",
        handle=Cuboid(length=0.12, width=0.03, height=0.02),  # Compact grip
        head=Cuboid(length=0.15, width=0.01, height=0.01),    # Thin flat shaft
        tool_type="screwdriver",
    ),
    # Cylindrical screwdriver: cylinder handle (grip) + cuboid head (flat shaft)
    ToolConfig(
        name="primitive_cylindrical_screwdriver",
        handle=Cylinder(length=0.10, radius=0.015),  # Round grip
        head=Cuboid(length=0.12, width=0.01, height=0.01),  # Thin short flat shaft
        tool_type="screwdriver",
    ),
]


# =============================================================================
# Eraser Configurations
# =============================================================================

ERASER_CONFIGS: List[ToolConfig] = [
    # Small whiteboard eraser: thickness along Y (width)
    ToolConfig(
        name="primitive_small_eraser",
        handle=Cuboid(length=0.10, width=0.025, height=0.05),  # 10cm long, 2.5cm thick, 5cm tall
        head=Cuboid(length=0.001, width=0.001, height=0.001),  # Near-zero head
        tool_type="eraser",
    ),
    # Large whiteboard eraser: thickness along Y (width)
    ToolConfig(
        name="primitive_large_eraser",
        handle=Cuboid(length=0.14, width=0.04, height=0.06),   # 14cm long, 4cm thick, 6cm tall
        head=Cuboid(length=0.001, width=0.001, height=0.001),  # Near-zero head
        tool_type="eraser",
    ),
]


# =============================================================================
# Spatula Configurations
# =============================================================================

SPATULA_CONFIGS: List[ToolConfig] = [
    # Small spatula: thin handle + narrow flat blade
    ToolConfig(
        name="primitive_small_spatula",
        handle=Cuboid(length=0.15, width=0.02, height=0.015),  # Thin grip handle
        head=Cuboid(length=0.08, width=0.04, height=0.01),    # Flat narrow blade
        tool_type="spatula",
    ),
    # Large spatula: cylinder handle + wide flat blade
    ToolConfig(
        name="primitive_large_spatula",
        handle=Cylinder(length=0.20, radius=0.012),  # Round grip handle
        head=Cuboid(length=0.10, width=0.07, height=0.015),    # Flat wide blade
        tool_type="spatula",
    ),
]


# =============================================================================
# Marker Configurations
# =============================================================================

MARKER_CONFIGS: List[ToolConfig] = [
    # Thin marker: slim cylinder body + small tip
    ToolConfig(
        name="primitive_thin_marker",
        handle=Cylinder(length=0.10, radius=0.01),  # Slim marker body
        head=Cuboid(length=0.015, width=0.005, height=0.005),  # Small tip
        tool_type="marker",
    ),
    # Thick marker: chunky cylinder body + larger tip
    ToolConfig(
        name="primitive_thick_marker",
        handle=Cylinder(length=0.14, radius=0.015),  # Chunky marker body
        head=Cuboid(length=0.03, width=0.01, height=0.01),  # Larger tip
        tool_type="marker",
    ),
]


# =============================================================================
# Brush Configurations
# =============================================================================

BRUSH_CONFIGS: List[ToolConfig] = [
    # frontal brush: cuboid handle + wide blade
    ToolConfig(
        name="primitive_frontal_brush",
        handle=Cuboid(length=0.15, width=0.035, height=0.02),  # Grip handle
        head=Cuboid(length=0.09, width=0.12, height=0.03),    # Thin flat blade
        tool_type="brush",
    ),
    # sideways brush: cylinder handle + less width more height blade
    ToolConfig(
        name="primitive_sideways_brush",
        handle=Cylinder(length=0.15, radius=0.015),  # Round grip handle
        head=Cuboid(length=0.10, width=0.06, height=0.07),   # Shorter thin blade
        tool_type="brush",
    ),
]


# =============================================================================
# All Tool Configurations
# =============================================================================

TOOL_CONFIGS: List[ToolConfig] = [
    *HAMMER_CONFIGS,
    *SCREWDRIVER_CONFIGS,
    *ERASER_CONFIGS,
    *SPATULA_CONFIGS,
    *MARKER_CONFIGS,
    *BRUSH_CONFIGS,
]


# =============================================================================
# Generation
# =============================================================================

def generate_all_tools() -> None:
    """Generate all tool configurations."""
    print("=" * 60)
    print("Generating tools")
    print("=" * 60)
    
    for config in TOOL_CONFIGS:
        print(f"\n--- {config.name} ---")
        output_dir = BASE_OUTPUT_DIR / config.tool_type / config.name
        # if output_dir already exists, erase it
        if output_dir.exists():
            shutil.rmtree(output_dir)
        create_tool(output_dir, config)
    
    print("\n" + "=" * 60)
    print(f"Generated {len(TOOL_CONFIGS)} tools")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_tools()
