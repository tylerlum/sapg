from dataclasses import dataclass
from pathlib import Path
import trimesh
import numpy as np
from typing import List, Optional, Tuple

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


RESCALE_FACTOR = 1.25

def rescale_by_factor(scale: Tuple[float, float, float], factor: float) -> Tuple[float, float, float]:
    return (scale[0] * factor, scale[1] * factor, scale[2] * factor)

def rescale(scale: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return rescale_by_factor(scale, RESCALE_FACTOR)


NAME_TO_OBJECT = {
    "blue_cuboid": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid/blue_cuboid.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/blue_cuboid"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((4.0, 0.75, 1.0)),
        need_vhacd=False,
    ),
    "blue_cuboid_real_iphone": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_iphone/blue_cuboid_real_iphone.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_iphone/cuboid.obj"]
        ),
        scale=rescale((3.0, 1.4, 0.2)),
        need_vhacd=False,
    ),
    "blue_cuboid_fake_iphone": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_fake_iphone/blue_cuboid_fake_iphone.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_fake_iphone/cuboid.obj"]
        ),
        scale=rescale((2.0, 1.25, 0.5)),
        need_vhacd=False,
    ),
    "blue_cuboid_real_hammer": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_hammer/blue_cuboid_real_hammer.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_hammer/cuboid.obj"]
        ),
        scale=rescale((2.0, 0.55, 0.35)),
        need_vhacd=False,
    ),
    "blue_cuboid_fake_hammer": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_fake_hammer/blue_cuboid_fake_hammer.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_fake_hammer/cuboid.obj"]
        ),
        scale=rescale((2.5, 0.75, 0.65)),
        need_vhacd=False,
    ),
    "blue_cuboid_real_screwdriver": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_screwdriver/blue_cuboid_real_screwdriver.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_real_screwdriver/cuboid.obj"]
        ),
        scale=rescale((1.3, 0.7, 0.5)),
        need_vhacd=False,
    ),
    "blue_cuboid_thick": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_thick/blue_cuboid_thick.urdf"
        ),
        coacd_filepaths=(
            [get_repo_root_dir() / "assets/urdf/tyler_objects/blue_cuboid_thick/cuboid.obj"]
        ),
        scale=rescale((3.0, 2.0, 1.25)),
        need_vhacd=False,
    ),
    "scanned_hammer_1": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/hammer_1/hammer_1.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_1"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "scanned_hammer_2": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/hammer_2/hammer_2.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_2"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.25, 0.2)),
        need_vhacd=True,
    ),
    "scanned_hammer_2_coacd": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_2/hammer_2.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_2"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.25, 0.2)),
        need_vhacd=False,
    ),
    "scanned_hammer_2_coacd2": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_2_2pieces/hammer_2.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/hammer_2_2pieces"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.25, 0.2)),
        need_vhacd=False,
    ),
    "YcbHammer": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/YcbHammer/model.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/YcbHammer"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "cuboidal_mallet": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_mallet_0-24_0-03_0-02_0-05_0-08_0-045.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale_by_factor((0.24, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
    "cuboidal_hammer": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-25_0-03_0-02_0-02_0-11_0-02.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale_by_factor((0.25, 0.03, 0.02), factor=25),
        need_vhacd=False,
    ),
    "cylindrical_hammer": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-3_0-015_0-015_0-1_0-1_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cuboidal_hammer_1-25x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-375_0-0375_0-025_0-0375_0-125_0-025_0-1_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cuboidal_hammer_1-5x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-44999999999999996_0-045_0-03_0-045_0-15000000000000002_0-03_0-1_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cuboidal_hammer_1-75x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-525_0-0525_0-035_0-0525_0-17500000000000002_0-035_0-1_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cuboidal_hammer_2x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cuboidal_hammer/cuboidal_hammer_0-6_0-06_0-04_0-06_0-2_0-04_0-1_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cylindrical_hammer_1-25x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-375_0-01875_0-01875_0-125_0-125_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cylindrical_hammer_1-5x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-44999999999999996_0-0225_0-0225_0-15000000000000002_0-15000000000000002_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cylindrical_hammer_1-75x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-525_0-02625_0-02625_0-17500000000000002_0-17500000000000002_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "cylindrical_hammer_2x": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/cylindrical_hammer/cylindrical_hammer_0-6_0-03_0-03_0-2_0-2_0-2.urdf"
        ),
        coacd_filepaths=None,  # Don't currently have COACD for object made of primitives
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=False,
    ),
    "040_large_marker": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/040_large_marker/040_large_marker.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/040_large_marker"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "whiteboard_eraser": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/whiteboard_eraser/source/model.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/whiteboard_eraser"
            ).glob("decomp_*.obj")
        ),
        scale=rescale_by_factor((0.12965531, 0.0337145 , 0.06038587), factor=25),
        need_vhacd=True,
    ),
    "phone": Object(
        filepath=(get_repo_root_dir() / "assets/urdf/tyler_objects/phone/model.urdf"),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/phone"
            ).glob("decomp_*.obj")
        ),
        scale=rescale_by_factor((0.22601312, 0.1112792 , 0.01462), factor=25),
        need_vhacd=True,
    ),
    "iphone15pro": Object(
        filepath=(get_repo_root_dir() / "assets/urdf/tyler_objects/iphone15pro/model.urdf"),
        coacd_filepaths=None,
        scale=rescale_by_factor((0.15954332, 0.0777093 , 0.01231273), factor=25),
        need_vhacd=True,
    ),
    "044_flat_screwdriver": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/044_flat_screwdriver"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "hairbrush": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/hairbrush/hairbrush.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/hairbrush"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "hairbrush_modified": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/hairbrush_modified/hairbrush_modified.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/hairbrush_modified"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
    "real_flat_screwdriver": Object(
        filepath=(
            get_repo_root_dir()
            / "assets/urdf/tyler_objects/real_flat_screwdriver/real_flat_screwdriver.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir()
                / "assets/urdf/tyler_objects_convex_decomp/real_flat_screwdriver"
            ).glob("decomp_*.obj")
        ),
        scale=rescale_by_factor((0.1, 0.03, 0.03), factor=25),
        need_vhacd=True,
    ),
    "mallet": Object(
        filepath=(
            get_repo_root_dir() / "assets/urdf/tyler_objects/mallet/mallet.urdf"
        ),
        coacd_filepaths=list(
            (
                get_repo_root_dir() / "assets/urdf/tyler_objects_convex_decomp/mallet"
            ).glob("decomp_*.obj")
        ),
        scale=rescale((3.0, 0.5, 0.5)),
        need_vhacd=True,
    ),
}
