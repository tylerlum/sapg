"""Visualize all trajectory data from FoundationPose human videos using viser.

Select an object category (e.g., spatula, brush) from the dropdown.
Each task within that category is shown in a separate section with all
objects performing that task displayed side by side.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import viser

# Data directory
# DATA_DIR = Path("/juno/u/kedia/FoundationPose/human_videos/Jan_17")
DATA_DIR = Path("dex_tool_bench/evaluation_trajectories")

# Assets directory for tool meshes
ASSETS_DIR = Path("/juno/u/kedia/sapg/assets/urdf/dex_tool_bench")

# Mesh scale
MESH_SCALE = 1.0

# Visualization settings
TRAJECTORY_LINE_WIDTH = 3.0
TRAJECTORY_POINT_SIZE = 0.008
FRAME_AXES_LENGTH = 0.03
FRAME_AXES_RADIUS = 0.002
ANIMATION_FPS = 30
OBJECT_SPACING_X = 0.5   # Space between objects within a task
TASK_SPACING_Y = 0.45    # Space between task sections

# Colors for different tool types (RGB 0-255)
TOOL_COLORS = {
    "brush": (50, 200, 50),
    "eraser": (200, 50, 50),
    "hammer": (200, 150, 50),
    "marker": (50, 50, 200),
    "screwdriver": (200, 50, 200),
    "spatula": (50, 200, 200),
}
DEFAULT_COLOR = (150, 150, 150)


@dataclass
class TrajectoryInfo:
    """Information about a single trajectory."""
    tool_type: str
    object_name: str
    task_name: str
    json_path: Path


@dataclass
class TrajectoryAnimation:
    """Pre-computed trajectory data for animation."""
    frame_handle: viser.SceneNodeHandle
    positions: np.ndarray
    wxyz_quats: np.ndarray
    num_frames: int


@dataclass
class CategoryView:
    """All scene elements for a category view."""
    category: str
    scene_handles: List[viser.SceneNodeHandle] = field(default_factory=list)
    animations: List[TrajectoryAnimation] = field(default_factory=list)


# Cache for mesh paths
_mesh_path_cache: Dict[Tuple[str, str], Optional[Path]] = {}


def load_trajectory_as_array(json_path: Path) -> np.ndarray:
    """Load trajectory as (N, 7) numpy array."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if "goals" in data:
        data = data["goals"]
    return np.array(data, dtype=np.float32)


def find_obj_mesh(tool_type: str, object_name: str) -> Optional[Path]:
    """Find the OBJ mesh file for a tool."""
    obj_path = ASSETS_DIR / tool_type / object_name / f"{object_name}.obj"
    if obj_path.exists():
        return obj_path
    
    folder_path = ASSETS_DIR / tool_type / object_name
    if folder_path.exists():
        obj_files = list(folder_path.glob("*.obj"))
        if obj_files:
            return obj_files[0]
    return None


def get_mesh_path(tool_type: str, object_name: str) -> Optional[Path]:
    """Get cached mesh path."""
    cache_key = (tool_type, object_name)
    if cache_key not in _mesh_path_cache:
        _mesh_path_cache[cache_key] = find_obj_mesh(tool_type, object_name)
    return _mesh_path_cache[cache_key]


def load_mesh_fresh(obj_path: Path) -> Optional[trimesh.Trimesh]:
    """Load a fresh mesh from disk."""
    try:
        mesh = trimesh.load(str(obj_path), process=False, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
            else:
                return None
        if MESH_SCALE != 1.0:
            mesh.apply_scale(MESH_SCALE)
        return mesh
    except Exception as e:
        print(f"    Warning: Failed to load mesh {obj_path}: {e}")
        return None


def find_all_trajectories(base_dir: Path) -> List[TrajectoryInfo]:
    """Find all trajectories and return as flat list."""
    trajectories = []
    
    # for json_path in sorted(base_dir.rglob("subgoals.json")):
    # for json_path in sorted(base_dir.rglob("*world_frame_min_z_0.6.json")):
    for json_path in sorted(base_dir.rglob("*world_frame_min_z_0.6_downsampled_10.json")):
        print(f"Found trajectory: {json_path}")
        relative_path = json_path.relative_to(base_dir)
        parts = relative_path.parts
        
        if len(parts) >= 4:
            tool_type = parts[0]
            object_name = parts[1]
            task_name = parts[2]
            trajectories.append(TrajectoryInfo(
                tool_type=tool_type,
                object_name=object_name,
                task_name=task_name,
                json_path=json_path,
            ))
        elif len(parts) == 3:
            tool_type = parts[0]
            object_name = parts[1]
            task_name = Path(parts[2]).stem
            trajectories.append(TrajectoryInfo(
                tool_type=tool_type,
                object_name=object_name,
                task_name=task_name,
                json_path=json_path,
            ))
    
    return trajectories


def get_trajectories_by_category(
    trajectories: List[TrajectoryInfo]
) -> Dict[str, Dict[str, List[TrajectoryInfo]]]:
    """Group trajectories by category (tool_type) -> task_name -> list of trajectories."""
    by_category: Dict[str, Dict[str, List[TrajectoryInfo]]] = defaultdict(lambda: defaultdict(list))
    
    for traj in trajectories:
        by_category[traj.tool_type][traj.task_name].append(traj)
    
    # Convert to regular dicts
    return {cat: dict(tasks) for cat, tasks in by_category.items()}


def normalize_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Normalize trajectory to start at origin."""
    if len(trajectory) == 0:
        return trajectory
    normalized = trajectory.copy()
    start_pos = trajectory[0, :3].copy()
    normalized[:, :3] = trajectory[:, :3] - start_pos
    return normalized


def convert_xyzw_to_wxyz(quaternions: np.ndarray) -> np.ndarray:
    """Convert quaternions from xyzw to wxyz format."""
    return quaternions[:, [3, 0, 1, 2]]


def create_category_view(
    server: viser.ViserServer,
    category: str,
    tasks: Dict[str, List[TrajectoryInfo]],
) -> CategoryView:
    """Create visualization for a category with sections for each task."""
    
    view = CategoryView(category=category)
    color = TOOL_COLORS.get(category, DEFAULT_COLOR)
    
    # Add category title at top
    title_handle = server.scene.add_label(
        "/view/title",
        text=f"{category.upper()}",
        position=(0, -0.4, 0.2),
    )
    view.scene_handles.append(title_handle)
    
    # Calculate layout
    task_names = sorted(tasks.keys())
    current_y = 0.0
    max_width = 0.0
    
    for task_idx, task_name in enumerate(task_names):
        task_trajectories = tasks[task_name]
        num_objects = len(task_trajectories)
        
        # Calculate x positions for objects in this task
        task_width = (num_objects - 1) * OBJECT_SPACING_X
        max_width = max(max_width, task_width)
        start_x = -task_width / 2
        
        # Add task section label
        task_label = server.scene.add_label(
            f"/view/task_{task_idx}/label",
            text=task_name.replace("_", " "),
            position=(-task_width / 2 - 0.2, current_y, 0.1),
        )
        view.scene_handles.append(task_label)
        
        # Add each object in this task
        for obj_idx, traj_info in enumerate(sorted(task_trajectories, key=lambda t: t.object_name)):
            x_pos = start_x + obj_idx * OBJECT_SPACING_X
            base_path = f"/view/task_{task_idx}/obj_{obj_idx}"
            
            # Load trajectory
            trajectory = load_trajectory_as_array(traj_info.json_path)
            trajectory = normalize_trajectory(trajectory)
            
            if len(trajectory) < 2:
                continue
            
            # Apply offset
            offset = np.array([x_pos, current_y, 0.0], dtype=np.float32)
            positions = trajectory[:, :3] + offset
            wxyz_quats = convert_xyzw_to_wxyz(trajectory[:, 3:7])
            
            # Add trajectory line
            line_handle = server.scene.add_spline_catmull_rom(
                f"{base_path}/line",
                positions=positions,
                color=color,
                line_width=TRAJECTORY_LINE_WIDTH,
            )
            view.scene_handles.append(line_handle)
            
            # Add start/end markers
            start_handle = server.scene.add_icosphere(
                f"{base_path}/start",
                radius=TRAJECTORY_POINT_SIZE * 1.5,
                color=(50, 255, 50),
                position=tuple(positions[0]),
            )
            view.scene_handles.append(start_handle)
            
            end_handle = server.scene.add_icosphere(
                f"{base_path}/end",
                radius=TRAJECTORY_POINT_SIZE * 1.5,
                color=(255, 50, 50),
                position=tuple(positions[-1]),
            )
            view.scene_handles.append(end_handle)
            
            # Add animated frame
            frame_handle = server.scene.add_frame(
                f"{base_path}/pose",
                position=tuple(positions[0]),
                wxyz=tuple(wxyz_quats[0]),
                axes_length=FRAME_AXES_LENGTH,
                axes_radius=FRAME_AXES_RADIUS,
            )
            view.scene_handles.append(frame_handle)
            
            # Add tool mesh
            mesh_path = get_mesh_path(traj_info.tool_type, traj_info.object_name)
            if mesh_path is not None:
                tool_mesh = load_mesh_fresh(mesh_path)
                if tool_mesh is not None:
                    mesh_handle = server.scene.add_mesh_trimesh(
                        f"{base_path}/pose/mesh",
                        mesh=tool_mesh,
                    )
                    view.scene_handles.append(mesh_handle)
            else:
                fallback = trimesh.creation.box(extents=(0.02, 0.01, 0.01))
                fallback.visual.face_colors = [*color, 255]
                mesh_handle = server.scene.add_mesh_trimesh(
                    f"{base_path}/pose/mesh",
                    mesh=fallback,
                )
                view.scene_handles.append(mesh_handle)
            
            # Store animation
            view.animations.append(TrajectoryAnimation(
                frame_handle=frame_handle,
                positions=positions,
                wxyz_quats=wxyz_quats,
                num_frames=len(positions),
            ))
            
            # Add object label below
            obj_label = server.scene.add_label(
                f"{base_path}/label",
                text=traj_info.object_name.replace("_", " "),
                position=(x_pos, current_y - 0.15, -0.05),
            )
            view.scene_handles.append(obj_label)
        
        # Move to next task section
        current_y += TASK_SPACING_Y
    
    # Add grid
    total_height = current_y + 0.5
    grid_handle = server.scene.add_grid(
        "/view/grid",
        width=max(max_width + 1.0, 2.0),
        height=total_height,
        position=(0, total_height / 2 - 0.3, -0.01),
        cell_size=0.1,
    )
    view.scene_handles.append(grid_handle)
    
    return view


def clear_view(view: CategoryView) -> None:
    """Remove all scene elements."""
    for handle in view.scene_handles:
        handle.remove()
    view.scene_handles.clear()
    view.animations.clear()


def main() -> None:
    """Main visualization with category dropdown."""
    
    # Find all trajectories
    all_trajectories = find_all_trajectories(DATA_DIR)
    
    if not all_trajectories:
        print(f"No trajectories found in {DATA_DIR}")
        return
    
    # Group by category then task
    by_category = get_trajectories_by_category(all_trajectories)
    categories = sorted(by_category.keys())
    
    print(f"Found {len(all_trajectories)} trajectories across {len(categories)} categories:")
    for cat in categories:
        tasks = by_category[cat]
        total_trajs = sum(len(t) for t in tasks.values())
        print(f"  {cat.upper()}: {len(tasks)} tasks, {total_trajs} trajectories")
        for task_name, trajs in sorted(tasks.items()):
            objects = [t.object_name for t in trajs]
            print(f"    - {task_name}: {', '.join(objects)}")
    
    # Pre-cache mesh paths
    print("\nLocating meshes...")
    for traj in all_trajectories:
        get_mesh_path(traj.tool_type, traj.object_name)
    
    # Start viser server
    server = viser.ViserServer(port=8080)
    print(f"\nViser server running at http://localhost:8080")
    
    # Current view state
    current_view: Optional[CategoryView] = None
    current_category: Optional[str] = None
    
    # Create GUI
    with server.gui.add_folder("Controls"):
        category_dropdown = server.gui.add_dropdown(
            "Category",
            options=categories,
            initial_value=categories[0] if categories else "",
        )
        
        speed_slider = server.gui.add_slider(
            "Speed",
            min=0.1,
            max=3.0,
            step=0.1,
            initial_value=1.0,
        )
    
    def switch_category(category: str) -> None:
        """Switch to a different category."""
        nonlocal current_view, current_category
        
        if category == current_category:
            return
        
        # Clear old view
        if current_view is not None:
            clear_view(current_view)
        
        # Create new view
        print(f"\nLoading category: {category}")
        current_view = create_category_view(server, category, by_category[category])
        current_category = category
        print(f"  Loaded {len(current_view.animations)} trajectories")
    
    @category_dropdown.on_update
    def on_category_change(event: viser.GuiEvent) -> None:
        switch_category(category_dropdown.value)
    
    # Load initial category
    if categories:
        switch_category(categories[0])
    
    print("\nUse the dropdown to select different object categories.")
    print("Each task within the category is shown in a separate row.")
    print("Press Ctrl+C to exit.")
    
    # Animation loop
    frame_idx = 0
    base_dt = 1.0 / ANIMATION_FPS
    
    while True:
        start_time = time.time()
        
        if current_view is not None:
            for anim in current_view.animations:
                pose_idx = frame_idx % anim.num_frames
                anim.frame_handle.position = anim.positions[pose_idx]
                anim.frame_handle.wxyz = anim.wxyz_quats[pose_idx]
        
        frame_idx += 1
        
        dt = base_dt / speed_slider.value
        elapsed = time.time() - start_time
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
