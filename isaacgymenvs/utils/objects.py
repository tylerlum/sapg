from dataclasses import dataclass
from pathlib import Path
import trimesh
import numpy as np
from typing import List, Optional, Tuple, Dict

from isaacgymenvs.utils.utils import get_repo_root_dir


@dataclass
class Object:
    filepath: Path
    """Path to the object URDF file."""

    scale: Tuple[float, float, float]
    """Scale of the object in x, y, z directions. TODO: Document this more since it is not metric scale but scale given to policy."""

    need_vhacd: bool
    """Whether the object needs a V-HACD decomposition (its convex hull is very different from the original mesh)"""

    coacd_filepaths: Optional[List[Path]] = None
    """List of paths to the COACD decomposition files. If None, the object does not have a COACD decomposition."""

    def __post_init__(self):
        assert self.filepath.exists(), f"Filepath {self.filepath} does not exist"

        # if self.coacd_filepaths is not None:
        #     assert len(self.coacd_filepaths) > 0, (
        #         f"coacd_filepaths is empty: {self.coacd_filepaths}"
        #     )
        #     for coacd_filepath in self.coacd_filepaths:
        #         assert coacd_filepath.exists(), (
        #             f"COACD file {coacd_filepath} does not exist"
        #         )

    def get_object_mesh_path_and_scale(self) -> Tuple[Path, np.ndarray]:
        from yourdfpy import URDF
        object_urdf_path = self.filepath

        assert object_urdf_path.exists(), object_urdf_path
        urdf = URDF.load(str(object_urdf_path))

        mesh_path_and_scale_list = []
        for link in urdf.robot.links:
            if len(link.collisions) == 0:
                continue

            for i, collision_link in enumerate(link.collisions):
                mesh_path = object_urdf_path.parent / collision_link.geometry.mesh.filename
                assert mesh_path.exists(), mesh_path

                mesh_scale = (
                    np.array([1, 1, 1])
                    if collision_link.geometry.mesh.scale is None
                    else np.array(collision_link.geometry.mesh.scale)
                )
                mesh_path_and_scale_list.append((mesh_path, mesh_scale))

        # Assume urdf has only 1 link with only 1 collision mesh
        assert len(mesh_path_and_scale_list) == 1, (
            f"{mesh_path_and_scale_list} has len {len(mesh_path_and_scale_list)}"
        )

        mesh_path, mesh_scale = mesh_path_and_scale_list[0]
        return mesh_path, mesh_scale

    def get_object_mesh(self) -> trimesh.Trimesh:
        mesh_path, mesh_scale = self.get_object_mesh_path_and_scale()
        mesh = trimesh.load_mesh(str(mesh_path))
        mesh.apply_scale(mesh_scale)
        return mesh


def rescale_by_factor(scale: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
    return (scale[0] * factor, scale[1] * factor, scale[2] * factor)

NAME_TO_OBJECT: Dict[str, Object] = {}  # Ultra hack to remove all other objects not in DEXTOOL BENCH
# DEXTOOL BENCH OBJECTS

#HAMMERS
HAMMER_NAME_TO_OBJECT = {
    "hammer_2": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/hammer/hammer_2/hammer_2.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/scanned_hammer_2"
            ).glob("decomp_*.obj")
        ),
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=True,
    ),
    "mallet": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/hammer/mallet/mallet.urdf"
        ),
        coacd_filepaths=None,
        # coacd_filepaths=list(
        #     (
        #         get_repo_root_dir()
        #         / "assets/urdf/tyler_objects_convex_decomp/mallet"
        #     ).glob("decomp_*.obj")
        # ),
        scale=rescale_by_factor((0.24, 0.03, 0.02), factor=25),
        # need_vhacd=True,
        need_vhacd=False,
    ),
    "toy_hammer": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/hammer/toy_hammer/toy_hammer.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.10, 0.0225, 0.015), factor=25),
        need_vhacd=False,
    ),
    "new_hammer_2": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/hammer/new_hammer_2/new_hammer_2.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
}

#overwrite NAME_TO_OBJECT with HAMMER_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(HAMMER_NAME_TO_OBJECT)

##SCREWDRIVERS
SCREWDRIVER_NAME_TO_OBJECT = {
    "real_flat_screwdriver": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/screwdriver/real_flat_screwdriver/real_flat_screwdriver.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/real_flat_screwdriver"
            ).glob("decomp_*.obj")
        ),
        scale=rescale_by_factor((0.1, 0.03, 0.03), factor=25),
        # scale=rescale_by_factor((0.1, 0.035, 0.025), factor=25),
        # need_vhacd=True,
        need_vhacd=False,
    ),
    "red_screwdriver": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/screwdriver/red_screwdriver/red_screwdriver.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.07, 0.035, 0.035), factor=25),
        need_vhacd=True,
    ),
    "black_screwdriver": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/screwdriver/black_screwdriver/black_screwdriver.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.08, 0.03, 0.03), factor=25),
        need_vhacd=True,
    ),
}
#overwrite NAME_TO_OBJECT with SCREWDRIVER_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(SCREWDRIVER_NAME_TO_OBJECT)

# ERASERS
ERASER_NAME_TO_OBJECT = {
    "whiteboard_eraser": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/eraser/whiteboard_eraser/whiteboard_eraser.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.12965531, 0.0337145 , 0.06038587), factor=25),
        need_vhacd=False,
    ),
    "anvil_eraser": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/eraser/anvil_eraser/anvil_eraser.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.09, 0.032, 0.01), factor=25),
        need_vhacd=True,
    ),
    "expo_eraser": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/eraser/expo_eraser/expo_eraser.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.10, 0.028, 0.05), factor=25),
        need_vhacd=True,
    ),
    "amazon_eraser": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/eraser/amazon_eraser/amazon_eraser.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.10, 0.028, 0.05), factor=25),
        need_vhacd=True,
    ),
}

#overwrite NAME_TO_OBJECT with ERASER_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(ERASER_NAME_TO_OBJECT)

#SPATULAS
SPATULA_NAME_TO_OBJECT = {
    "black_spatula": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/spatula/black_spatula/spatula.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.2, 0.015, 0.0075), factor=25),
        # need_vhacd=True,
        need_vhacd=False,
    ),
    "wooden_spatula": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/spatula/wooden_spatula/wooden_spatula.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.17, 0.02, 0.0065), factor=25),
        need_vhacd=True,
    ),
    "spoon_spatula": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/spatula/spoon_spatula/spoon_spatula.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.12, 0.02, 0.02), factor=25),
        need_vhacd=True,
    ),
}
#overwrite NAME_TO_OBJECT with SPATULA_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(SPATULA_NAME_TO_OBJECT)

#MARKERS
MARKER_NAME_TO_OBJECT = {
    "040_large_marker": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/marker/040_large_marker/040_large_marker.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/040_large_marker"
            ).glob("decomp_*.obj")
        ),
        # scale=rescale_by_factor((0.121277, 0.019341, 0.021183), factor=25),
        scale=rescale_by_factor((0.09, 0.018, 0.018), factor=25),  # Re-measured in real
        need_vhacd=True,
    ),  
    "sharpie_closed": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/marker/sharpie_closed/sharpie_closed.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.085, 0.022, 0.022), factor=25),
        need_vhacd=True,
    ),
    "staples_open": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/marker/staples_open/staples_open.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.12, 0.018, 0.018), factor=25),
        need_vhacd=True,
    ),
}
#overwrite NAME_TO_OBJECT with MARKER_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(MARKER_NAME_TO_OBJECT)

BRUSH_NAME_TO_OBJECT = {
    "red_brush": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/brush/red_brush/red_brush.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.1, 0.02, 0.015), factor=25),
        need_vhacd=False,
    ),
    "green_brush": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/brush/green_brush/green_brush.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.15, 0.02, 0.015), factor=25),
        need_vhacd=True,
    ),
    "anvil_brush": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/brush/anvil_brush/anvil_brush.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.12, 0.035, 0.02), factor=25),
        need_vhacd=True,
    ),
    "lab_brush": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench/brush/lab_brush/lab_brush.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.06, 0.022, 0.022), factor=25),
        need_vhacd=True,
    ),
}
#overwrite NAME_TO_OBJECT with BRUSH_NAME_TO_OBJECT even if they share keys
NAME_TO_OBJECT.update(BRUSH_NAME_TO_OBJECT)

from dex_tool_bench.generate_tools import TOOL_CONFIGS
for tool_config in TOOL_CONFIGS:
    NAME_TO_OBJECT[tool_config.name] = Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/dex_tool_bench"
            / tool_config.tool_type
            / tool_config.name
            / f"{tool_config.name}.urdf"
        ),
        coacd_filepaths=None,
        scale=rescale_by_factor(tool_config.handle.get_scale(), factor=25),
        need_vhacd=False,  # Primitive tools are not convex decomp'd
    )