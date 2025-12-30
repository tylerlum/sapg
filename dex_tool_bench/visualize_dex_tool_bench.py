"""Visualize all tools in the dex_tool_bench folder using viser.

This script finds all URDF and OBJ files in the dex_tool_bench assets folder
and displays them organized by tool type with section titles.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import trimesh
import viser
from viser.extras import ViserUrdf


# Base directory for all dex_tool_bench assets
ASSETS_DIR = Path("/share/portal/kk837/sapg/assets/urdf/dex_tool_bench")


def find_all_tools_by_type(base_dir: Path) -> Dict[str, List[Tuple[Path, Optional[Path]]]]:
    """Find all tools grouped by tool type (parent folder).
    
    Returns:
        Dict mapping tool_type -> list of (urdf_path, obj_path) tuples.
    """
    tools_by_type = defaultdict(list)
    urdf_files = sorted(base_dir.rglob("*.urdf"))
    
    for urdf_path in urdf_files:
        # Get tool type from folder structure: base_dir / tool_type / tool_name / file.urdf
        relative_path = urdf_path.relative_to(base_dir)
        if len(relative_path.parts) >= 2:
            tool_type = relative_path.parts[0]
        else:
            tool_type = "other"
        
        # Look for matching OBJ file
        obj_path = urdf_path.with_suffix(".obj")
        if not obj_path.exists():
            obj_files = list(urdf_path.parent.glob("*.obj"))
            obj_path = obj_files[0] if obj_files else None
        
        tools_by_type[tool_type].append((urdf_path, obj_path))
    
    return dict(tools_by_type)


def main() -> None:
    """Visualize all tools in the dex_tool_bench folder."""
    
    # Find all tools grouped by type
    tools_by_type = find_all_tools_by_type(ASSETS_DIR)
    
    if not tools_by_type:
        print(f"No tools found in {ASSETS_DIR}")
        return
    
    total_tools = sum(len(tools) for tools in tools_by_type.values())
    print(f"Found {total_tools} tools in {len(tools_by_type)} categories:")
    for tool_type, tools in tools_by_type.items():
        print(f"\n  [{tool_type.upper()}]")
        for urdf, obj in tools:
            obj_status = "✓" if obj else "✗"
            print(f"    - {urdf.stem} (OBJ: {obj_status})")
    
    # Start viser server
    server = viser.ViserServer(port=8080)
    print(f"\nViser server running at http://localhost:8080")
    
    # Layout parameters
    spacing_y = 0.35  # Space between tools along Y axis
    urdf_obj_offset = 0.12  # Y offset between URDF and OBJ for same tool
    section_spacing_x = 0.4  # Space between sections along X axis
    title_height = 0.15  # Height of title above tools
    x_offset = -0.5  # Shift all objects back in X direction
    
    current_x = x_offset
    
    # Load tools by section (sections along X, tools along Y)
    for tool_type, tools in sorted(tools_by_type.items()):
        print(f"\nLoading section: {tool_type}")
        
        # Add section title at the front of the section
        server.scene.add_label(
            f"/section_{tool_type}/title",
            text=tool_type.upper().replace("_", " "),
            position=(current_x, -0.15, title_height),
        )
        
        # Load each tool in this section (along Y axis)
        for i, (urdf_path, obj_path) in enumerate(tools):
            y_pos = i * spacing_y
            
            tool_name = urdf_path.stem
            print(f"  Loading: {tool_name}")
            
            # Load URDF
            server.scene.add_frame(
                f"/section_{tool_type}/{tool_name}_urdf",
                position=(current_x, y_pos, 0.0),
                axes_length=0.05,
                axes_radius=0.002,
            )
            ViserUrdf(
                server,
                urdf_or_path=urdf_path,
                root_node_name=f"/section_{tool_type}/{tool_name}_urdf",
                load_meshes=True,
                load_collision_meshes=False,
            )
            
            # Load OBJ if it exists (offset in Y from URDF)
            if obj_path and obj_path.exists():
                # Load mesh with material
                mesh = trimesh.load(str(obj_path), process=False)
                
                server.scene.add_frame(
                    f"/section_{tool_type}/{tool_name}_obj",
                    position=(current_x, y_pos + urdf_obj_offset, 0.0),
                    axes_length=0.05,
                    axes_radius=0.002,
                )
                
                server.scene.add_mesh_trimesh(
                    name=f"/section_{tool_type}/{tool_name}_obj/mesh",
                    mesh=mesh,
                )
        
        # Move to next section along X axis
        current_x += section_spacing_x
    
    # Add grid for reference
    server.scene.add_grid(
        "/grid",
        width=4,
        height=4,
        position=(0.0, 0.0, -0.01),
    )
    
    print(f"\nLoaded {total_tools} tools in {len(tools_by_type)} sections")
    print("Press Ctrl+C to exit.")
    
    # Keep the server running
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
