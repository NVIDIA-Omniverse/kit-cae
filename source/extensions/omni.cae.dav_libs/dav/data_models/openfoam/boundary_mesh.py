# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
r"""
OpenFOAM Boundary Mesh Data Model
==================================

This module provides a data model implementation for OpenFOAM boundary meshes.

Boundary meshes represent 2.5D surface geometry extracted from a 3D OpenFOAM polyMesh:
- A subset of faces from the parent mesh (defined by startFace and nFaces)
- Each face becomes a polygonal cell (2D surface element)
- Explicit point coordinates and face connectivity

OpenFOAM Boundary Structure
----------------------------
In OpenFOAM, boundaries are defined in the `boundary` file with:
- **startFace**: Starting face index in the parent mesh's face list
- **nFaces**: Number of faces in this boundary
- The face data (points, connectivity) comes from the parent mesh's `faces` and `points` files

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous, refers to boundary faces)
- **Face IDs**: wp.int32 (references to parent mesh faces)
- **Indices**: wp.int32 (same as IDs)

2.5D Element Treatment
-----------------------
- Boundary faces are treated as 2.5D surface elements (polygonal cells)
- get_num_faces() returns 1 (the face itself is treated as 1 face for consistency with 2.5D models)
- This allows operators like cell_faces to extract boundary faces as surface meshes
- No interpolation or point location support (2.5D elements)

Key Features
------------
- Represents OpenFOAM boundary patches as 2.5D surface meshes
- Maintains startFace and nFaces for tracking parent mesh relationship
- Supports arbitrary polygonal faces (triangles, quads, general polygons)
- BVH-based locators for efficient spatial queries
- Explicit cell links for point-to-cell queries

Limitations
-----------
- Interpolation is not supported (2.5D surface elements)
- find_cell_containing_point returns empty (not implemented for 2.5D)
- Does not support boundary conditions or field data (pure geometry)
"""

from typing import Any

import warp as wp

import dav
from dav import locators


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    """Vertex positions (full array from parent mesh)."""
    faces: wp.array(dtype=wp.int32)
    """Flattened array of vertex indices for all faces (full array from parent mesh)."""
    face_offsets: wp.array(dtype=wp.int32)
    """Offsets into faces array for each face (full array from parent mesh, size = total_faces + 1)."""
    start_face: wp.int32
    """Starting face index in the parent mesh (this boundary starts here)."""
    n_faces: wp.int32
    """Number of faces in this boundary."""
    cell_bvh_id: wp.uint64
    """BVH ID for spatial acceleration."""
    cell_links: locators.CellLinks
    """Point-to-cell connectivity structure."""


def create_handle(points: wp.array, faces: wp.array, face_offsets: wp.array, start_face: int = 0, n_faces: int = None) -> DatasetHandle:
    """Create an OpenFOAM boundary mesh dataset handle.

    This creates a VIEW into the parent mesh data. The handle stores the FULL arrays
    from the parent mesh but only exposes a subset of faces as cells.

    Args:
        points: Full array of 3D point coordinates from parent mesh (wp.vec3f)
        faces: Full flattened array of vertex indices for all faces from parent mesh (wp.int32)
        face_offsets: Full offsets array into faces from parent mesh (wp.int32, length = total_faces + 1)
        start_face: Starting face index in the parent mesh for this boundary (default: 0)
        n_faces: Number of faces in this boundary (if None, uses all faces from start_face to end)

    Returns:
        DatasetHandle: A new boundary mesh dataset handle that views a subset of the parent mesh

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or indices are out of bounds

    Note:
        - This creates a VIEW into the parent mesh data, not a copy
        - The handle stores the FULL arrays but only exposes faces [start_face, start_face+n_faces)
        - face_offsets must have length = total_faces + 1 where total_faces >= start_face + n_faces
        - cell_bvh_id and cell_links will be initialized to default values
        - Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links()
          to build spatial acceleration structures after creating the dataset

    Example:
        >>> import warp as wp
        >>> from dav.data_models.openfoam.boundary_mesh import create_handle
        >>> # Parent mesh with 10 faces
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=wp.vec3f)
        >>> faces = wp.array([0, 1, 2, 1, 3, 2, ...], dtype=wp.int32)  # 10 faces worth
        >>> face_offsets = wp.array([0, 3, 6, 9, ...], dtype=wp.int32)  # 11 elements
        >>> # Create boundary viewing faces 5-7 (3 faces)
        >>> handle = create_handle(points, faces, face_offsets, start_face=5, n_faces=3)
    """
    # Validate points
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    # Validate faces
    if faces is None or faces.ndim != 1:
        raise ValueError("faces must be a 1D warp array")
    if faces.dtype != wp.int32:
        raise ValueError(f"faces must have dtype wp.int32, got {faces.dtype}")

    # Validate face_offsets
    if face_offsets is None or face_offsets.ndim != 1:
        raise ValueError("face_offsets must be a 1D warp array")
    if face_offsets.dtype != wp.int32:
        raise ValueError(f"face_offsets must have dtype wp.int32, got {face_offsets.dtype}")
    if face_offsets.shape[0] < 2:
        raise ValueError("face_offsets must have at least 2 elements (for at least 1 face)")

    # Total number of faces in parent mesh
    # Note: We don't validate face_offsets values to avoid downloading data from device
    # Users are expected to provide valid offset arrays (start with 0, non-decreasing, last == faces.shape[0])
    total_faces = face_offsets.shape[0] - 1

    # Derive n_faces if not provided
    if n_faces is None:
        n_faces = total_faces - start_face

    # Validate start_face and n_faces
    if start_face < 0:
        raise ValueError(f"start_face must be non-negative, got {start_face}")
    if start_face >= total_faces:
        raise ValueError(f"start_face ({start_face}) must be < total_faces ({total_faces})")
    if n_faces < 0:
        raise ValueError(f"n_faces must be non-negative, got {n_faces}")
    if start_face + n_faces > total_faces:
        raise ValueError(f"start_face ({start_face}) + n_faces ({n_faces}) = {start_face + n_faces} exceeds total_faces ({total_faces})")

    handle = DatasetHandle()
    handle.points = points
    handle.faces = faces
    handle.face_offsets = face_offsets
    handle.start_face = wp.int32(start_face)
    handle.n_faces = wp.int32(n_faces)
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()  # Empty cell links
    return handle


def create_dataset(points: wp.array, faces: wp.array, face_offsets: wp.array, start_face: int = 0, n_faces: int = None) -> dav.Dataset:
    """
    Create a boundary mesh dataset handle (alias for create_handle).
    """
    handle = create_handle(points, faces, face_offsets, start_face, n_faces)
    return dav.Dataset(DataModel, handle, points.device)


@wp.struct
class CellHandle:
    """Cell handle for a boundary mesh face.

    In boundary meshes, cells ARE faces (2.5D surface elements).
    """

    cell_id: wp.int32


class CellAPI:
    @staticmethod
    @dav.func
    def is_valid(cell: CellHandle) -> wp.bool:
        """Check if a cell handle is valid."""
        return cell.cell_id >= 0

    @staticmethod
    @dav.func
    def empty() -> CellHandle:
        """Create an empty (invalid) cell handle."""
        cell = CellHandle()
        cell.cell_id = -1
        return cell

    @staticmethod
    @dav.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        """Get the cell ID from a cell handle."""
        return cell.cell_id

    @staticmethod
    @dav.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of vertices in a boundary face (cell)."""
        face_idx = ds.start_face + cell.cell_id  # Map to parent mesh face index
        start = ds.face_offsets[face_idx]
        end = ds.face_offsets[face_idx + 1]
        return end - start

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a vertex ID from a boundary face (cell) by local index."""
        face_idx = ds.start_face + cell.cell_id  # Map to parent mesh face index
        start = ds.face_offsets[face_idx]
        return ds.faces[start + local_idx]

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell.

        For boundary meshes (2.5D surface elements), this returns 1.
        Each cell (boundary face) is treated as having 1 face (itself) for
        consistency with other 2.5D data models (polydata, surface_mesh).
        This allows operators like cell_faces to work correctly.
        """
        return 1

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face.

        For boundary meshes, face 0 is the cell itself (all its vertices).
        """
        assert face_idx == 0, "Boundary mesh cells only have one face (face_idx must be 0)"
        parent_face_idx = ds.start_face + cell.cell_id  # Map to parent mesh face index
        start = ds.face_offsets[parent_face_idx]
        end = ds.face_offsets[parent_face_idx + 1]
        return end - start

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face.

        For boundary meshes, face 0 returns the cell's vertices directly.
        """
        assert face_idx == 0, "Boundary mesh cells only have one face (face_idx must be 0)"
        parent_face_idx = ds.start_face + cell.cell_id  # Map to parent mesh face index
        start = ds.face_offsets[parent_face_idx]
        return ds.faces[start + local_idx]


class DatasetAPI:
    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        """Get cell ID from cell index (same for boundary meshes)."""
        return cell_idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        """Get cell index from cell ID (same for boundary meshes)."""
        return cell_id

    @staticmethod
    @dav.func
    def get_cell(ds: DatasetHandle, cell_id: wp.int32) -> CellHandle:
        """Get a cell handle from a cell ID."""
        cell = CellHandle()
        cell.cell_id = cell_id
        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        """Get the number of faces (cells) in the boundary mesh."""
        return ds.n_faces

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        """Get the number of vertices in the boundary mesh."""
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        """Get point ID from point index (same for boundary meshes)."""
        return point_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        """Get point index from point ID (same for boundary meshes)."""
        return point_id

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        """Get a vertex position by point ID."""
        return ds.points[point_id]

    @staticmethod
    @dav.func
    def get_field_id_from_idx(dataset: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        """Get field ID from field index (same for boundary meshes)."""
        return local_idx

    @staticmethod
    @dav.func
    def get_field_idx_from_id(dataset: DatasetHandle, id: wp.int32) -> wp.int32:
        """Get field index from field ID (same for boundary meshes)."""
        return id

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'boundary_mesh')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (success, locator) - Success flag and locator instance
        """
        return (True, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build the cell links for the dataset.

        Args:
            data_model: The data model module
            ds: The dataset
            device: Device to build the links on

        Returns:
            tuple: (success, links) - Success flag and CellLinks instance
        """
        cell_links = locators.build_cell_links(data_model, ds, device)
        if cell_links is not None:
            ds.cell_links = cell_links
            return (True, cell_links)
        else:
            ds.cell_links = None
            return (False, None)


class CellLocatorAPI:
    @staticmethod
    @dav.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        """Evaluate position within a cell to get interpolation weights.

        Note: Interpolation is not supported for boundary meshes (2.5D surface elements).
        This always returns an empty interpolated cell handle.
        """
        return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)(0.0)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell (boundary face) containing a point.

        Note: Point location is not supported for boundary meshes (2.5D surface elements).
        This always returns an empty cell handle.
        """
        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        return False


CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(-1), emptyCellId=wp.int32(-1), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (OpenFOAM Boundary Mesh)"


# DataModel protocol implementation
class DataModel(metaclass=DataModelMeta):
    """OpenFOAM Boundary Mesh data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle
    PointIdHandle = wp.int32
    CellIdHandle = wp.int32

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    CellLinksAPI = CellLinksAPI
    CellLocatorAPI = CellLocatorAPI


def get_data_model():
    """Factory function to get the data model for OpenFOAM boundary meshes."""
    return DataModel
