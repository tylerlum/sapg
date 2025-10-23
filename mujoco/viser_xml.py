from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import yourdfpy
from trimesh.scene import Scene
from typing_extensions import assert_never

import viser

import time

import tyro

# from viser.extras import ViserUrdf

import viser.transforms as tf


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

import mujoco
import trimesh.visual
import viser.transforms as vtf
from typing_extensions import override
from mujoco import mj_id2name, mjtGeom, mjtObj


if TYPE_CHECKING:
  import mujoco

import trimesh.visual.material
from PIL import Image


def mujoco_mesh_to_trimesh(
  mj_model: mujoco.MjModel, geom_idx: int, verbose: bool = False
) -> trimesh.Trimesh:
  """Convert a MuJoCo mesh geometry to a trimesh with textures if available.

  Args:
      mj_model: MuJoCo model object
      geom_idx: Index of the geometry in the model
      verbose: If True, print debug information during conversion

  Returns:
      A trimesh object with texture/material applied if available
  """

  # Get the mesh ID for this geometry.
  mesh_id = mj_model.geom_dataid[geom_idx]

  # Get mesh data ranges from MuJoCo.
  vert_start = int(mj_model.mesh_vertadr[mesh_id])
  vert_count = int(mj_model.mesh_vertnum[mesh_id])
  face_start = int(mj_model.mesh_faceadr[mesh_id])
  face_count = int(mj_model.mesh_facenum[mesh_id])

  # Extract vertices and faces.
  # mesh_vert shape: (total_verts_in_model, 3)
  # We extract our mesh's vertices.
  vertices = mj_model.mesh_vert[
    vert_start : vert_start + vert_count
  ]  # Shape: (vert_count, 3)
  assert vertices.shape == (
    vert_count,
    3,
  ), f"Expected vertices shape ({vert_count}, 3), got {vertices.shape}"

  # mesh_face shape: (total_faces_in_model, 3)
  # Each face has 3 vertex indices.
  faces = mj_model.mesh_face[
    face_start : face_start + face_count
  ]  # Shape: (face_count, 3)
  assert faces.shape == (
    face_count,
    3,
  ), f"Expected faces shape ({face_count}, 3), got {faces.shape}"

  # Check if this mesh has texture coordinates.
  texcoord_adr = mj_model.mesh_texcoordadr[mesh_id]
  texcoord_num = mj_model.mesh_texcoordnum[mesh_id]

  if texcoord_num > 0:
    # This mesh has UV coordinates.
    if verbose:
      print(f"Mesh has {texcoord_num} texture coordinates")

    # Extract texture coordinates.
    # mesh_texcoord is a 2D array with shape (nmeshtexcoord, 2).
    texcoords = mj_model.mesh_texcoord[texcoord_adr : texcoord_adr + texcoord_num]
    assert texcoords.shape == (
      texcoord_num,
      2,
    ), f"Expected texcoords shape ({texcoord_num}, 2), got {texcoords.shape}"

    # Get per-face texture coordinate indices.
    # For each face vertex, this tells us which texcoord to use.
    # mesh_facetexcoord is a 2D array with shape (nmeshface, 3).
    face_texcoord_idx = mj_model.mesh_facetexcoord[face_start : face_start + face_count]
    assert face_texcoord_idx.shape == (face_count, 3), (
      f"Expected face_texcoord_idx shape ({face_count}, 3), got {face_texcoord_idx.shape}"
    )

    # Since the same vertex can have different UVs in different faces,
    # we need to duplicate vertices. Each face will get its own 3 vertices.

    # Duplicate vertices for each face reference.
    # faces.flatten() gives us vertex indices in order: [v0_f0, v1_f0, v2_f0, v0_f1, v1_f1, v2_f1, ...]
    new_vertices = vertices[faces.flatten()]  # Shape: (face_count * 3, 3)
    assert new_vertices.shape == (
      face_count * 3,
      3,
    ), f"Expected new_vertices shape ({face_count * 3}, 3), got {new_vertices.shape}"

    # Get UV coordinates for each duplicated vertex.
    # face_texcoord_idx.flatten() gives us texcoord indices in the same order.
    new_uvs = texcoords[face_texcoord_idx.flatten()]  # Shape: (face_count * 3, 2)
    assert new_uvs.shape == (
      face_count * 3,
      2,
    ), f"Expected new_uvs shape ({face_count * 3}, 2), got {new_uvs.shape}"

    # Create new faces - now just sequential since vertices are duplicated.
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], ...]
    new_faces = np.arange(face_count * 3).reshape(-1, 3)  # Shape: (face_count, 3)
    assert new_faces.shape == (
      face_count,
      3,
    ), f"Expected new_faces shape ({face_count}, 3), got {new_faces.shape}"

    # Create the mesh (process=False to preserve all vertices).
    mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Now handle material and texture.
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      # This geometry has a material.
      rgba = mj_model.mat_rgba[matid]  # Shape: (4,)
      # mat_texid is 2D (nmat x mjNTEXROLE), get the RGB/RGBA texture.
      # Try RGB first (index 1), then RGBA (index 8).
      texid = int(mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
      if texid < 0:
        texid = int(
          mj_model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)]
        )

      if texid >= 0 and texid < mj_model.ntex:
        # This material has a texture.
        if verbose:
          print(f"Material has texture ID {texid}")

        # Extract texture data.
        tex_width = mj_model.tex_width[texid]
        tex_height = mj_model.tex_height[texid]
        tex_nchannel = mj_model.tex_nchannel[texid]
        tex_adr = mj_model.tex_adr[texid]

        # Calculate texture data size.
        tex_size = tex_width * tex_height * tex_nchannel

        # Extract raw texture data.
        tex_data = mj_model.tex_data[tex_adr : tex_adr + tex_size]
        assert tex_data.shape == (tex_size,), (
          f"Expected tex_data shape ({tex_size},), got {tex_data.shape}"
        )

        # Reshape texture data based on number of channels.
        # Note: MuJoCo uses OpenGL convention (origin at bottom-left)
        # but GLTF/GLB expects top-left origin, so we flip vertically.
        if tex_nchannel == 1:
          # Grayscale.
          tex_array = tex_data.reshape(tex_height, tex_width)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="L")
        elif tex_nchannel == 3:
          # RGB.
          tex_array = tex_data.reshape(tex_height, tex_width, 3)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGB")
        elif tex_nchannel == 4:
          # RGBA.
          tex_array = tex_data.reshape(tex_height, tex_width, 4)
          # Flip vertically for GLTF convention.
          tex_array = np.flipud(tex_array)
          image = Image.fromarray(tex_array.astype(np.uint8), mode="RGBA")
        else:
          if verbose:
            print(f"Unsupported number of texture channels: {tex_nchannel}")
          image = None

        if image is not None:
          # Create material with texture.
          # Set PBR properties for proper rendering:
          # - metallicFactor=0.0: non-metallic (dielectric) material
          # - roughnessFactor=1.0: fully rough (diffuse) surface
          material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
          )

          # Apply texture visual with UV coordinates.
          mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
          if verbose:
            print(f"Applied texture: {tex_width}x{tex_height}, {tex_nchannel} channels")
        else:
          # Just use material color - convert from [0,1] to [0,255].
          rgba_255 = (rgba * 255).astype(np.uint8)
          mesh.visual = trimesh.visual.ColorVisuals(
            vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
          )
      else:
        # Material but no texture - use material color.
        if verbose:
          print(f"Material has no texture, using color: {rgba}")
        rgba_255 = (rgba * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(
          vertex_colors=np.tile(rgba_255, (len(new_vertices), 1))
        )
    else:
      # No material - use default color based on collision/visual.
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision.
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual.

      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(new_vertices), 1))
      )
      if verbose:
        print(
          f"No material, using default {'collision' if is_collision else 'visual'} color"
        )

  else:
    # No texture coordinates - simpler case.
    if verbose:
      print("Mesh has no texture coordinates")

    # Create mesh with original vertices and faces (process=False to avoid vertex removal).
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Apply material color if available.
    matid = mj_model.geom_matid[geom_idx]

    if matid >= 0 and matid < mj_model.nmat:
      rgba = mj_model.mat_rgba[matid]
      rgba_255 = (rgba * 255).astype(np.uint8)
      # Use actual vertex count after mesh creation.
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Applied material color: {rgba}")
    else:
      # Default color.
      is_collision = (
        mj_model.geom_contype[geom_idx] != 0 or mj_model.geom_conaffinity[geom_idx] != 0
      )
      if is_collision:
        color = np.array([204, 102, 102, 128], dtype=np.uint8)  # Red-ish for collision.
      else:
        color = np.array([31, 128, 230, 255], dtype=np.uint8)  # Blue-ish for visual.

      # Use actual vertex count after mesh creation.
      mesh.visual = trimesh.visual.ColorVisuals(
        vertex_colors=np.tile(color, (len(mesh.vertices), 1))
      )
      if verbose:
        print(f"Using default {'collision' if is_collision else 'visual'} color")

  # Final sanity checks.
  assert mesh.vertices.shape[1] == 3, (
    f"Vertices should be Nx3, got {mesh.vertices.shape}"
  )
  assert mesh.faces.shape[1] == 3, f"Faces should be Nx3, got {mesh.faces.shape}"
  assert len(mesh.vertices) > 0, "Mesh has no vertices"
  assert len(mesh.faces) > 0, "Mesh has no faces"

  if verbose:
    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

  return mesh


def create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh:
  """Create a mesh for primitive geom types (sphere, box, capsule, cylinder, plane).

  Args:
    mj_model: MuJoCo model containing geom definition
    geom_id: Index of the geom to create mesh for

  Returns:
    Trimesh representation of the primitive geom
  """
  size = mj_model.geom_size[geom_id]
  geom_type = mj_model.geom_type[geom_id]
  rgba = mj_model.geom_rgba[geom_id].copy()

  material = trimesh.visual.material.PBRMaterial(  # type: ignore
    baseColorFactor=rgba,
    metallicFactor=0.0,
    roughnessFactor=1.0,
  )

  if geom_type == mjtGeom.mjGEOM_SPHERE:
    mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
  elif geom_type == mjtGeom.mjGEOM_BOX:
    mesh = trimesh.creation.box(extents=2.0 * size)
  elif geom_type == mjtGeom.mjGEOM_CAPSULE:
    mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_CYLINDER:
    mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
  elif geom_type == mjtGeom.mjGEOM_PLANE:
    mesh = trimesh.creation.box((20, 20, 0.01))
  else:
    raise ValueError(f"Unsupported primitive geom type: {geom_type}")

  mesh.visual = trimesh.visual.TextureVisuals(material=material)  # type: ignore
  return mesh


def merge_geoms(mj_model: mujoco.MjModel, geom_ids: list[int]) -> trimesh.Trimesh:
  """Merge multiple geoms into a single trimesh.

  Args:
    mj_model: MuJoCo model containing geom definitions
    geom_ids: List of geom indices to merge

  Returns:
    Single merged trimesh with all geoms transformed to their local poses
  """
  meshes = []
  for geom_id in geom_ids:
    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      mesh = mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      mesh = create_primitive_mesh(mj_model, geom_id)

    pos = mj_model.geom_pos[geom_id]
    quat = mj_model.geom_quat[geom_id]
    transform = np.eye(4)
    transform[:3, :3] = vtf.SO3(quat).as_matrix()
    transform[:3, 3] = pos
    mesh.apply_transform(transform)
    meshes.append(mesh)

  if len(meshes) == 1:
    return meshes[0]
  return trimesh.util.concatenate(meshes)


def rotation_quat_from_vectors(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
  """Compute quaternion (wxyz format) that rotates from_vec to to_vec.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    Quaternion in wxyz format that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.array([1.0, 0.0, 0.0, 0.0])

  if np.allclose(from_vec, -to_vec):
    # 180 degree rotation - pick arbitrary perpendicular axis.
    perp = np.array([1.0, 0.0, 0.0])
    if abs(from_vec[0]) > 0.9:
      perp = np.array([0.0, 1.0, 0.0])
    axis = np.cross(from_vec, perp)
    axis = axis / np.linalg.norm(axis)
    return np.array([0.0, axis[0], axis[1], axis[2]])  # wxyz for 180 deg.

  # Standard quaternion from two vectors.
  cross = np.cross(from_vec, to_vec)
  dot = np.dot(from_vec, to_vec)
  w = 1.0 + dot
  quat = np.array([w, cross[0], cross[1], cross[2]])
  quat = quat / np.linalg.norm(quat)
  return quat


def rotation_matrix_from_vectors(
  from_vec: np.ndarray, to_vec: np.ndarray
) -> np.ndarray:
  """Create rotation matrix that rotates from_vec to to_vec using Rodrigues formula.

  Args:
    from_vec: Source vector (3D)
    to_vec: Target vector (3D)

  Returns:
    3x3 rotation matrix that rotates from_vec to to_vec.
  """
  from_vec = from_vec / np.linalg.norm(from_vec)
  to_vec = to_vec / np.linalg.norm(to_vec)

  if np.allclose(from_vec, to_vec):
    return np.eye(3)

  if np.allclose(from_vec, -to_vec):
    return np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

  # Rodrigues rotation formula.
  v = np.cross(from_vec, to_vec)
  s = np.linalg.norm(v)
  c = np.dot(from_vec, to_vec)
  vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
  return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def is_fixed_body(mj_model: mujoco.MjModel, body_id: int) -> bool:
  """Check if a body is fixed (welded to world).

  A body is considered fixed if it has:
  - No degrees of freedom (body_dofnum == 0)
  - World as parent (body_parentid == 0)

  Args:
    mj_model: MuJoCo model
    body_id: Body index

  Returns:
    True if body is fixed to world, False if movable.
  """
  return mj_model.body_dofnum[body_id] == 0 and mj_model.body_parentid[body_id] == 0


def get_body_name(mj_model: mujoco.MjModel, body_id: int) -> str:
  """Get body name with fallback to ID-based name.

  Args:
    mj_model: MuJoCo model
    body_id: Body index

  Returns:
    Body name or "body_{body_id}" if name not found.
  """
  body_name = mj_id2name(mj_model, mjtObj.mjOBJ_BODY, body_id)
  if not body_name:
    body_name = f"body_{body_id}"
  return body_name













class DebugVisualizer(ABC):
  """Abstract base class for viewer-agnostic debug visualization.

  This allows manager terms to draw debug visualizations without knowing the underlying
  viewer implementation.
  """

  env_idx: int
  """Index of the environment being visualized."""

  @abstractmethod
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Add an arrow from start to end position.

    Args:
      start: Start position (3D vector)
      end: End position (3D vector)
      color: RGBA color (values 0-1)
      width: Arrow shaft width
      label: Optional label for this arrow
    """
    ...

  @abstractmethod
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost/transparent rendering of a robot at a target pose.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override (0=transparent, 1=opaque) - may not be used by all implementations
      label: Optional label for this ghost
    """
    ...

  @abstractmethod
  def clear(self) -> None:
    """Clear all debug visualizations."""
    ...


class NullDebugVisualizer:
  """No-op visualizer when visualization is disabled."""

  def __init__(self, env_idx: int = 0):
    self.env_idx = env_idx

  def add_arrow(self, start, end, color, width=0.015, label=None) -> None:
    pass

  def add_ghost_mesh(self, qpos, model, alpha=0.5, label=None) -> None:
    pass

  def clear(self) -> None:
    pass


########################################



class ViserDebugVisualizer(DebugVisualizer):
  """Debug visualizer for Viser viewer.

  This implementation uses Viser's scene graph to add visualization primitives
  like arrows and batched meshes.
  """

  def __init__(
    self,
    server: viser.ViserServer,
    mj_model: mujoco.MjModel,
    env_idx: int,
    env_origin: np.ndarray | None = None,
  ):
    """Initialize the Viser debug visualizer.

    Args:
      server: Viser server instance
      mj_model: MuJoCo model (not used for ghost rendering, kept for compatibility)
      env_idx: Index of the environment being visualized
      env_origin: World origin offset for this environment
    """
    self.server = server
    self.mj_model = mj_model
    self.env_idx = env_idx
    self.env_origin = env_origin if env_origin is not None else np.zeros(3)

    # Queued arrows for batched rendering
    self._queued_arrows: list[
      tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]
    ] = []

    # Batched arrow mesh handles
    self._arrow_shaft_handle: viser.BatchedMeshHandle | None = None
    self._arrow_head_handle: viser.BatchedMeshHandle | None = None

    # Ghost mesh handles
    self._ghost_handles: dict[int, viser.SceneNodeHandle] = {}

    # Cache ghost meshes by model hash to handle deepcopy'd models
    self._ghost_meshes: dict[int, dict[int, trimesh.Trimesh]] = {}

    # Cache arrow mesh components for batched rendering
    self._arrow_shaft_mesh: trimesh.Trimesh | None = None
    self._arrow_head_mesh: trimesh.Trimesh | None = None

    # Reusable MjData for ghost rendering
    self._viz_data = mujoco.MjData(mj_model)

  @override
  def add_arrow(
    self,
    start: np.ndarray | torch.Tensor,
    end: np.ndarray | torch.Tensor,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    """Queue an arrow for batched rendering.

    Arrows are not rendered immediately but queued and rendered together
    in the next _synchronize() call for efficiency.
    """
    if isinstance(start, torch.Tensor):
      start = start.cpu().numpy()
    if isinstance(end, torch.Tensor):
      end = end.cpu().numpy()

    start = start + self.env_origin
    end = end + self.env_origin

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
      return

    # Queue the arrow for batched rendering
    self._queued_arrows.append((start, end, color, width))

  @override
  def add_ghost_mesh(
    self,
    qpos: np.ndarray | torch.Tensor,
    model: mujoco.MjModel,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    """Add a ghost mesh by rendering the robot at a different pose.

    For Viser, we create meshes once and update their poses for efficiency.

    Args:
      qpos: Joint positions for the ghost pose
      model: MuJoCo model with pre-configured appearance (geom_rgba for colors)
      alpha: Transparency override
      label: Optional label for this ghost
    """
    if isinstance(qpos, torch.Tensor):
      qpos = qpos.cpu().numpy()

    # Use model hash to support models with same structure but different colors
    model_hash = hash((model.ngeom, model.nbody, model.nq))

    self._viz_data.qpos[:] = qpos
    mujoco.mj_forward(model, self._viz_data)

    # Group geoms by body
    body_geoms: dict[int, list[int]] = {}
    for i in range(model.ngeom):
      body_id = model.geom_bodyid[i]
      is_collision = model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0
      if is_collision:
        continue

      if model.body_dofnum[body_id] == 0 and model.body_parentid[body_id] == 0:
        continue

      if body_id not in body_geoms:
        body_geoms[body_id] = []
      body_geoms[body_id].append(i)

    # Update or create mesh for each body
    for body_id, geom_indices in body_geoms.items():
      body_pos = self._viz_data.xpos[body_id] + self.env_origin
      body_quat = self._mat_to_quat(self._viz_data.xmat[body_id].reshape(3, 3))

      # Check if we already have a handle for this body
      if body_id in self._ghost_handles:
        handle = self._ghost_handles[body_id]
        handle.wxyz = body_quat
        handle.position = body_pos
      else:
        # Create mesh if not cached
        if model_hash not in self._ghost_meshes:
          self._ghost_meshes[model_hash] = {}

        if body_id not in self._ghost_meshes[model_hash]:
          meshes = []
          for geom_id in geom_indices:
            mesh = self._create_geom_mesh_from_model(model, geom_id)
            if mesh is not None:
              geom_pos = model.geom_pos[geom_id]
              geom_quat = model.geom_quat[geom_id]
              transform = np.eye(4)
              transform[:3, :3] = vtf.SO3(geom_quat).as_matrix()
              transform[:3, 3] = geom_pos
              mesh.apply_transform(transform)
              meshes.append(mesh)

          if not meshes:
            continue

          combined_mesh = (
            meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
          )

          self._ghost_meshes[model_hash][body_id] = combined_mesh
        else:
          combined_mesh = self._ghost_meshes[model_hash][body_id]

        body_name = get_body_name(model, body_id)
        handle_name = f"/debug/env_{self.env_idx}/ghost/body_{body_name}"

        # Extract color from geom (convert RGBA 0-1 to RGB 0-255)
        rgba = model.geom_rgba[geom_indices[0]].copy()
        color_uint8 = (rgba[:3] * 255).astype(np.uint8)

        handle = self.server.scene.add_mesh_simple(
          handle_name,
          combined_mesh.vertices,
          combined_mesh.faces,
          color=tuple(color_uint8),
          opacity=alpha,
          wxyz=body_quat,
          position=body_pos,
          cast_shadow=False,
          receive_shadow=False,
        )
        self._ghost_handles[body_id] = handle

  def _create_geom_mesh_from_model(
    self, mj_model: mujoco.MjModel, geom_id: int
  ) -> trimesh.Trimesh | None:
    """Create a trimesh from a MuJoCo geom using the specified model.

    Args:
      mj_model: MuJoCo model containing geom definition
      geom_id: Index of the geom to create mesh for

    Returns:
      Trimesh representation of the geom, or None if unsupported type
    """
    from mujoco import mjtGeom

    from mjlab.viewer.viser_conversions import (
      create_primitive_mesh,
      mujoco_mesh_to_trimesh,
    )

    geom_type = mj_model.geom_type[geom_id]

    if geom_type == mjtGeom.mjGEOM_MESH:
      return mujoco_mesh_to_trimesh(mj_model, geom_id, verbose=False)
    else:
      return create_primitive_mesh(mj_model, geom_id)

  def _sync_arrows(self) -> None:
    """Render all queued arrows using batched meshes.

    This should be called by the main visualizer after all debug visualizations
    have been queued for the current frame.
    """
    if not self._queued_arrows:
      # Remove arrow meshes if no arrows to render
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
        self._arrow_shaft_handle = None
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()
        self._arrow_head_handle = None
      return

    # Create arrow mesh components if needed (unit-sized base meshes)
    if self._arrow_shaft_mesh is None:
      # Unit cylinder: radius=1.0, height=1.0
      self._arrow_shaft_mesh = trimesh.creation.cylinder(radius=1.0, height=1.0)
      self._arrow_shaft_mesh.apply_translation(np.array([0, 0, 0.5]))  # Center at z=0.5

    if self._arrow_head_mesh is None:
      # Unit cone: radius=3.0, height=1.0 (base at z=0, tip at z=1.0 by default)
      head_width = 3.0
      self._arrow_head_mesh = trimesh.creation.cone(radius=head_width, height=1.0)
      # No translation needed - cone already has base at z=0

    # Prepare batched data
    num_arrows = len(self._queued_arrows)
    shaft_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    shaft_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    shaft_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    head_positions = np.zeros((num_arrows, 3), dtype=np.float32)
    head_wxyzs = np.zeros((num_arrows, 4), dtype=np.float32)
    head_scales = np.zeros((num_arrows, 3), dtype=np.float32)
    head_colors = np.zeros((num_arrows, 3), dtype=np.uint8)

    z_axis = np.array([0, 0, 1])
    shaft_length_ratio = 0.8
    head_length_ratio = 0.2

    for i, (start, end, color, width) in enumerate(self._queued_arrows):
      direction = end - start
      length = np.linalg.norm(direction)
      direction = direction / length

      rotation_quat = rotation_quat_from_vectors(z_axis, direction)

      # Shaft: scale width in XY, length in Z
      shaft_length = shaft_length_ratio * length
      shaft_positions[i] = start
      shaft_wxyzs[i] = rotation_quat
      shaft_scales[i] = [width, width, shaft_length]  # Per-axis scale
      shaft_colors[i] = (np.array(color[:3]) * 255).astype(
        np.uint8
      )  # Convert 0-1 to 0-255

      # Head: position at end of shaft
      # The cone has its base at z=0, so after scaling by head_length,
      # the base is still at z=0 in local coords
      # We want the base at the end of the shaft (at shaft_length)
      head_length = head_length_ratio * length
      head_position = start + direction * shaft_length
      head_positions[i] = head_position
      head_wxyzs[i] = rotation_quat
      head_scales[i] = [width, width, head_length]  # Per-axis scale
      head_colors[i] = (np.array(color[:3]) * 255).astype(
        np.uint8
      )  # Convert 0-1 to 0-255

    # Check if we need to recreate handles (number of arrows changed)
    needs_recreation = (
      self._arrow_shaft_handle is None
      or self._arrow_head_handle is None
      or len(shaft_positions) != len(self._arrow_shaft_handle.batched_positions)
    )

    if needs_recreation:
      # Remove old handles
      if self._arrow_shaft_handle is not None:
        self._arrow_shaft_handle.remove()
      if self._arrow_head_handle is not None:
        self._arrow_head_handle.remove()

      # Create new batched meshes
      self._arrow_shaft_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_shafts",
        self._arrow_shaft_mesh.vertices,
        self._arrow_shaft_mesh.faces,
        batched_wxyzs=shaft_wxyzs,
        batched_positions=shaft_positions,
        batched_scales=shaft_scales,
        batched_colors=shaft_colors,
        opacity=0.5,
        cast_shadow=False,
        receive_shadow=False,
      )

      self._arrow_head_handle = self.server.scene.add_batched_meshes_simple(
        f"/debug/env_{self.env_idx}/arrow_heads",
        self._arrow_head_mesh.vertices,
        self._arrow_head_mesh.faces,
        batched_wxyzs=head_wxyzs,
        batched_positions=head_positions,
        batched_scales=head_scales,
        batched_colors=head_colors,
        opacity=0.5,
        cast_shadow=False,
        receive_shadow=False,
      )
    else:
      # Update existing handles (guaranteed to exist by needs_recreation check)
      assert self._arrow_shaft_handle is not None
      assert self._arrow_head_handle is not None

      self._arrow_shaft_handle.batched_positions = shaft_positions
      self._arrow_shaft_handle.batched_wxyzs = shaft_wxyzs
      self._arrow_shaft_handle.batched_scales = shaft_scales
      self._arrow_shaft_handle.batched_colors = shaft_colors

      self._arrow_head_handle.batched_positions = head_positions
      self._arrow_head_handle.batched_wxyzs = head_wxyzs
      self._arrow_head_handle.batched_scales = head_scales
      self._arrow_head_handle.batched_colors = head_colors

  @override
  def clear(self) -> None:
    """Clear all debug visualizations.

    Clears the arrow queue. Ghost meshes are kept and pose-updated for efficiency
    within the same environment, but removed when switching environments.
    """
    self._queued_arrows.clear()

  def clear_all(self) -> None:
    """Clear all debug visualizations including ghosts.

    Called when switching to a different environment.
    """
    self.clear()

    # Remove arrow meshes
    if self._arrow_shaft_handle is not None:
      self._arrow_shaft_handle.remove()
      self._arrow_shaft_handle = None
    if self._arrow_head_handle is not None:
      self._arrow_head_handle.remove()
      self._arrow_head_handle = None

    # Remove ghost meshes
    for handle in self._ghost_handles.values():
      handle.remove()
    self._ghost_handles.clear()

  @staticmethod
  def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    return vtf.SO3.from_matrix(mat).wxyz


########################################


class ViserUrdf:
    def __init__(
        self,
        target: viser.ViserServer | viser.ClientHandle,
        urdf_or_path: yourdfpy.URDF | Path,
        scale: float = 1.0,
        root_node_name: str = "/",
        mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        collision_mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None = None,
        load_meshes: bool = True,
        load_collision_meshes: bool = False,
    ) -> None:
        assert root_node_name.startswith("/")
        assert len(root_node_name) == 1 or not root_node_name.endswith("/")

        if isinstance(urdf_or_path, Path):
            urdf = yourdfpy.URDF.load(
                urdf_or_path,
                build_scene_graph=load_meshes,
                build_collision_scene_graph=load_collision_meshes,
                load_meshes=load_meshes,
                load_collision_meshes=load_collision_meshes,
                filename_handler=partial(
                    yourdfpy.filename_handler_magic,
                    dir=urdf_or_path.parent,
                ),
            )
        else:
            urdf = urdf_or_path
        assert isinstance(urdf, yourdfpy.URDF)

        self._target = target
        self._urdf = urdf
        self._scale = scale
        self._root_node_name = root_node_name
        self._load_meshes = load_meshes
        self._collision_root_frame: viser.FrameHandle | None = None
        self._visual_root_frame: viser.FrameHandle | None = None
        self._joint_frames: List[viser.SceneNodeHandle] = []
        self._meshes: List[viser.SceneNodeHandle] = []
        num_joints_to_repeat = 0
        if load_meshes:
            if urdf.scene is not None:
                num_joints_to_repeat += 1
                self._visual_root_frame = self._add_joint_frames_and_meshes(
                    urdf.scene,
                    root_node_name,
                    collision_geometry=False,
                    mesh_color_override=mesh_color_override,
                )
            else:
                warnings.warn(
                    "load_meshes is enabled but the URDF model does not have a visual scene configured. Not displaying."
                )
        if load_collision_meshes:
            if urdf.collision_scene is not None:
                num_joints_to_repeat += 1
                self._collision_root_frame = self._add_joint_frames_and_meshes(
                    urdf.collision_scene,
                    root_node_name,
                    collision_geometry=True,
                    mesh_color_override=collision_mesh_color_override,
                )
            else:
                warnings.warn(
                    "load_collision_meshes is enabled but the URDF model does not have a collision scene configured. Not displaying."
                )

        self._joint_map_values = [*self._urdf.joint_map.values()] * num_joints_to_repeat

    @property
    def show_visual(self) -> bool:
        """Returns whether the visual meshes are currently visible."""
        return self._visual_root_frame is not None and self._visual_root_frame.visible

    @show_visual.setter
    def show_visual(self, visible: bool) -> None:
        """Set whether the visual meshes are currently visible."""
        if self._visual_root_frame is not None:
            self._visual_root_frame.visible = visible
        else:
            warnings.warn(
                "Cannot set `.show_visual`, since no visual meshes were loaded."
            )

    @property
    def show_collision(self) -> bool:
        """Returns whether the collision meshes are currently visible."""
        return (
            self._collision_root_frame is not None
            and self._collision_root_frame.visible
        )

    @show_collision.setter
    def show_collision(self, visible: bool) -> None:
        """Set whether the collision meshes are currently visible."""
        if self._collision_root_frame is not None:
            self._collision_root_frame.visible = visible
        else:
            warnings.warn(
                "Cannot set `.show_collision`, since no collision meshes were loaded."
            )

    def remove(self) -> None:
        """Remove URDF from scene."""
        # Some of this will be redundant, since children are removed when
        # parents are removed.
        for frame in self._joint_frames:
            frame.remove()
        for mesh in self._meshes:
            mesh.remove()

    def update_cfg(self, configuration: np.ndarray) -> None:
        """Update the joint angles of the visualized URDF."""
        self._urdf.update_cfg(configuration)
        for joint, frame_handle in zip(self._joint_map_values, self._joint_frames):
            assert isinstance(joint, yourdfpy.Joint)
            T_parent_child = self._urdf.get_transform(
                joint.child, joint.parent, collision_geometry=not self._load_meshes
            )
            frame_handle.wxyz = tf.SO3.from_matrix(T_parent_child[:3, :3]).wxyz
            frame_handle.position = T_parent_child[:3, 3] * self._scale

    def get_actuated_joint_limits(
        self,
    ) -> dict[str, tuple[float | None, float | None]]:
        """Returns an ordered mapping from actuated joint names to position limits."""
        out: dict[str, tuple[float | None, float | None]] = {}
        for joint_name, joint in zip(
            self._urdf.actuated_joint_names, self._urdf.actuated_joints
        ):
            assert isinstance(joint_name, str)
            assert isinstance(joint, yourdfpy.Joint)
            if joint.limit is None:
                out[joint_name] = (-np.pi, np.pi)
            else:
                out[joint_name] = (joint.limit.lower, joint.limit.upper)
        return out

    def get_actuated_joint_names(self) -> Tuple[str, ...]:
        """Returns a tuple of actuated joint names, in order."""
        return tuple(self._urdf.actuated_joint_names)

    def _add_joint_frames_and_meshes(
        self,
        scene: Scene,
        root_node_name: str,
        collision_geometry: bool,
        mesh_color_override: tuple[float, float, float]
        | tuple[float, float, float, float]
        | None,
    ) -> viser.FrameHandle:
        """
        Helper function to add joint frames and meshes to the ViserUrdf object.
        """
        prefix = "collision" if collision_geometry else "visual"
        prefixed_root_node_name = (f"{root_node_name}/{prefix}").replace("//", "/")
        root_frame = self._target.scene.add_frame(
            prefixed_root_node_name, show_axes=False
        )

        # Add coordinate frame for each joint.
        for joint in self._urdf.joint_map.values():
            assert isinstance(joint, yourdfpy.Joint)
            self._joint_frames.append(
                self._target.scene.add_frame(
                    _viser_name_from_frame(
                        scene,
                        joint.child,
                        prefixed_root_node_name,
                    ),
                    show_axes=False,
                )
            )

        # Add the URDF's meshes/geometry to viser.
        for link_name, mesh in scene.geometry.items():
            assert isinstance(mesh, trimesh.Trimesh)
            T_parent_child = self._urdf.get_transform(
                link_name,
                scene.graph.transforms.parents[link_name],
                collision_geometry=collision_geometry,
            )
            name = _viser_name_from_frame(scene, link_name, prefixed_root_node_name)

            # Scale + transform the mesh. (these will mutate it!)
            #
            # It's important that we use apply_transform() instead of unpacking
            # the rotation/translation terms, since the scene graph transform
            # can also contain scale and reflection terms.
            mesh = mesh.copy()
            mesh.apply_scale(self._scale)
            mesh.apply_transform(T_parent_child)

            if mesh_color_override is None:
                self._meshes.append(self._target.scene.add_mesh_trimesh(name, mesh))
            elif len(mesh_color_override) == 3:
                self._meshes.append(
                    self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=mesh_color_override,
                    )
                )
            elif len(mesh_color_override) == 4:
                self._meshes.append(
                    self._target.scene.add_mesh_simple(
                        name,
                        mesh.vertices,
                        mesh.faces,
                        color=mesh_color_override[:3],
                        opacity=mesh_color_override[3],
                    )
                )
            else:
                assert_never(mesh_color_override)
        return root_frame


def _viser_name_from_frame(
    scene: Scene,
    frame_name: str,
    root_node_name: str = "/",
) -> str:
    """Given the (unique) name of a frame in our URDF's kinematic tree, return a
    scene node name for viser.

    For a robot manipulator with four frames, that looks like:


            ((shoulder)) == ((elbow))
               / /             |X|
              / /           ((wrist))
         ____/ /____           |X|
        [           ]       [=======]
        [ base_link ]        []   []
        [___________]


    this would map a name like "elbow" to "base_link/shoulder/elbow".
    """
    assert root_node_name.startswith("/")
    assert len(root_node_name) == 1 or not root_node_name.endswith("/")

    frames = []
    while frame_name != scene.graph.base_frame:
        frames.append(frame_name)
        frame_name = scene.graph.transforms.parents[frame_name]
    if root_node_name != "/":
        frames.append(root_node_name)
    return "/".join(frames[::-1])



def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf
) -> tuple[list[viser.GuiInputHandle[float]], list[float]]:
    """Create slider for each joint of the robot. We also update robot model
    when slider moves."""
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0
        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,
            max=upper,
            step=1e-3,
            initial_value=initial_pos,
        )
        slider.on_update(  # When sliders move, we update the URDF configuration.
            lambda _: viser_urdf.update_cfg(
                np.array([slider.value for slider in slider_handles])
            )
        )
        slider_handles.append(slider)
        initial_config.append(initial_pos)
    return slider_handles, initial_config


def main(
    urdf_path: Path = Path("/home/tylerlum/github_repos/sapg/assets/urdf/kuka_allegro_description/allegro_touch_sensor.urdf"),
    load_meshes: bool = True,
    load_collision_meshes: bool = False,
) -> None:
    # Start viser server.
    server = viser.ViserServer()

    # Load URDF.
    #
    # This takes either a yourdfpy.URDF object or a path to a .urdf file.
    assert urdf_path.exists(), f"URDF path {urdf_path} does not exist"
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=load_meshes,
        load_collision_meshes=load_collision_meshes,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
    )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf
        )

    # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = load_meshes
    show_collision_meshes_cb.visible = load_collision_meshes

    # Set initial robot configuration.
    viser_urdf.update_cfg(np.array(initial_config))

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        for s, init_q in zip(slider_handles, initial_config):
            s.value = init_q

    # Sleep forever.
    while True:
        time.sleep(10.0)


def load_mj_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION = 1.0, 0.005, 0.0001
    iiwa_xml_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/kuka_iiwa_14/scene.xml")
    assert iiwa_xml_path.exists(), f"Robot path does not exist: {iiwa_xml_path}"

    # Load mjspec from robot path
    spec = mujoco.MjSpec.from_file(str(iiwa_xml_path))
    spec.option.timestep = 1.0 / 1000.0

    allegro_xml_path = Path("/home/tylerlum/github_repos/mujoco_menagerie/wonik_allegro/right_hand_offset.xml")
    assert allegro_xml_path.exists(), f"Allegro XML path does not exist: {allegro_xml_path}"
    allegro_spec = mujoco.MjSpec.from_file(str(allegro_xml_path))
    attachment_site = next(s for s in spec.sites if s.name == "attachment_site")
    attachment_site.attach_body(allegro_spec.worldbody, "palm", "")

# <origin rpy="0 -1.5708 0.785398" xyz="0.008219 -0.02063 0.08086"/>

    # Table
    WHITE_RGBA = np.array([1.0, 1.0, 1.0, 1.0])
    TABLE_LEN_X, TABLE_LEN_Y, TABLE_LEN_Z = 0.475, 0.4, 0.3
    TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z = 0.0, -0.8, 0.38
    table_body = spec.worldbody.add_body()
    table_body.name = "table"
    table_body.pos = np.array([TABLE_POS_X, TABLE_POS_Y, TABLE_POS_Z])
    table_geom = table_body.add_geom()
    table_geom.name = "table_geom"
    table_geom.type = mujoco.mjtGeom.mjGEOM_BOX
    table_geom.size = np.array([TABLE_LEN_X / 2, TABLE_LEN_Y / 2, TABLE_LEN_Z / 2])  # Half extents
    table_geom.pos = np.array([0.0, 0.0, 0.0])
    table_geom.rgba = WHITE_RGBA
    table_geom.friction = np.array([SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION])

    # Object
    BLACK_RGBA = np.array([0.0, 0.0, 0.0, 1.0])
    OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z = 0.0, -0.8, 0.38 + 0.3
    mesh = spec.add_mesh()
    mesh.name = "object_mesh"
    # mesh.file = "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/044_flat_screwdriver/044_flat_screwdriver/google_16k/textured_vhacd.obj"
    mesh.file = "/home/tylerlum/github_repos/sapg/assets/urdf/tyler_objects/040_large_marker/040_large_marker/google_16k/textured_vhacd.obj"
    assert Path(mesh.file).exists(), f"Mesh file does not exist: {mesh.file}"
    mesh.scale = np.array([1.0, 1.0, 1.0])
    object_body = spec.worldbody.add_body()
    object_body.name = "object"
    object_body.pos = np.array([OBJECT_POS_X, OBJECT_POS_Y, OBJECT_POS_Z])
    object_geom = object_body.add_geom()
    object_geom.name = "object_geom"
    object_geom.type = mujoco.mjtGeom.mjGEOM_MESH
    object_geom.meshname = mesh.name
    # object_geom.rgba = BLACK_RGBA
    object_geom.friction = np.array([SLIDING_FRICTION, TORSIONAL_FRICTION, ROLLING_FRICTION])
    object_free_joint = object_body.add_joint()
    object_free_joint.name = "object_free_joint"
    object_free_joint.type = mujoco.mjtJoint.mjJNT_FREE

    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 1.0 / 1000.0
    return mj_model, mj_data



def main2():
    server = viser.ViserServer()
    mj_model, _ = load_mj_model()

    visualizer = ViserDebugVisualizer(
        server=server,
        mj_model=mj_model,
        env_idx=0,
    )
    while True:
        breakpoint()
        time.sleep(10.0)

if __name__ == "__main__":
    # tyro.cli(main)
    tyro.cli(main2)