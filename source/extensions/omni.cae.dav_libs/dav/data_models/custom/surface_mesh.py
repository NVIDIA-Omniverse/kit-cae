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
Surface Mesh Data Model (UsdGeomMesh-based)
============================================

This module provides a data model implementation for 2.5D surface meshes
following the UsdGeomMesh specification.

Surface meshes represent 2.5D geometry with:
- Explicit point coordinates (vertices)
- Face connectivity via faceVertexCounts and faceVertexIndices
- Support for arbitrary polygonal faces (triangles, quads, n-gons)

UsdGeomMesh Specification
-------------------------
The data model follows USD's UsdGeomMesh schema:
- **points**: Array of 3D vertex positions
- **faceVertexCounts**: Number of vertices per face
- **faceVertexIndices**: Flattened array of vertex indices forming faces

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous, refers to faces)
- **Indices**: wp.int32 (same as IDs)

Key Features
------------
- Follows UsdGeomMesh specification for compatibility
- Supports arbitrary polygonal faces (triangles, quads, n-gons)
- Explicit topology stored in faceVertexCounts and faceVertexIndices
- BVH-based locators for efficient spatial queries
- Explicit cell links for point-to-cell queries
- Maximum 8 vertices per face (configurable via MAX_FACE_VERTICES)

Current Limitations
-------------------
- Interpolation is not yet supported
- find_cell_containing_point returns empty (not yet implemented)
- No support for subdivision surfaces (treated as polygonal mesh only)
- No support for normals, UVs, or vertex colors in this version

Future Enhancements
-------------------
- Barycentric interpolation for triangles
- Bilinear/generalized barycentric for quads and n-gons
- Point-in-polygon tests for find_cell_containing_point
- Support for face normals and vertex attributes
"""

from typing import Any

import warp as wp

import dav
from dav import locators

MAX_FACE_VERTICES = 8  # maximum number of vertices per face we will support


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    """Vertex positions (UsdGeomMesh 'points')."""
    face_vertex_counts: wp.array(dtype=wp.int32)
    """Number of vertices per face (UsdGeomMesh 'faceVertexCounts')."""
    face_vertex_indices: wp.array(dtype=wp.int32)
    """Flattened vertex indices for all faces (UsdGeomMesh 'faceVertexIndices')."""
    face_offsets: wp.array(dtype=wp.int32)
    """Precomputed offsets into face_vertex_indices (derived from faceVertexCounts)."""
    cell_bvh_id: wp.uint64
    """BVH ID for spatial acceleration."""
    cell_links: locators.CellLinks
    """Point-to-cell connectivity structure."""


def create_handle(points: wp.array, face_vertex_indices: wp.array, face_vertex_counts: wp.array = None, face_vertex_offsets: wp.array = None) -> DatasetHandle:
    """Create a surface mesh dataset handle following UsdGeomMesh specification.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete dav.Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D vertex positions (wp.vec3f) - UsdGeomMesh 'points'
        face_vertex_indices: Flattened vertex indices for all faces (wp.int32) - UsdGeomMesh 'faceVertexIndices'
        face_vertex_counts: Number of vertices per face (wp.int32) - UsdGeomMesh 'faceVertexCounts' (optional if face_vertex_offsets provided)
        face_vertex_offsets: Offsets into face_vertex_indices (wp.int32, size=num_faces+1) (optional if face_vertex_counts provided)

    Returns:
        DatasetHandle: A new surface mesh dataset handle

    Raises:
        ValueError: If array dimensions or dtypes are invalid, if array sizes are inconsistent,
                   or if neither face_vertex_counts nor face_vertex_offsets is provided

    Note:
        - Exactly one of face_vertex_counts or face_vertex_offsets must be provided.
        - If face_vertex_counts is provided, face_vertex_offsets will be computed (exclusive scan).
        - If face_vertex_offsets is provided, face_vertex_counts will be computed (differences).
        - The cell_bvh_id and cell_links will be initialized to default values.
        - Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links() to build
          the spatial acceleration structures after creating the dataset.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.surface_mesh import create_handle
        >>> # Triangle mesh example with counts
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=wp.vec3f)
        >>> face_vertex_counts = wp.array([3, 3], dtype=wp.int32)  # Two triangles
        >>> face_vertex_indices = wp.array([0, 1, 2, 1, 3, 2], dtype=wp.int32)
        >>> handle = create_handle(points, face_vertex_indices, face_vertex_counts=face_vertex_counts)
        >>> # Or with offsets
        >>> face_vertex_offsets = wp.array([0, 3, 6], dtype=wp.int32)  # Offsets for two triangles
        >>> handle = create_handle(points, face_vertex_indices, face_vertex_offsets=face_vertex_offsets)
    """
    # Validate points
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    # Validate face_vertex_indices
    if face_vertex_indices is None or face_vertex_indices.ndim != 1:
        raise ValueError("face_vertex_indices must be a 1D warp array")
    if face_vertex_indices.dtype != wp.int32:
        raise ValueError(f"face_vertex_indices must have dtype wp.int32, got {face_vertex_indices.dtype}")

    # Validate that at least one of counts or offsets is provided
    if face_vertex_counts is None and face_vertex_offsets is None:
        raise ValueError("Either face_vertex_counts or face_vertex_offsets must be provided")
    if face_vertex_counts is not None and face_vertex_offsets is not None:
        raise ValueError("Provide either face_vertex_counts or face_vertex_offsets, not both")

    import numpy as np

    # Compute the missing array (counts or offsets)
    if face_vertex_counts is not None:
        # We have counts, compute offsets
        if face_vertex_counts.ndim != 1:
            raise ValueError("face_vertex_counts must be a 1D warp array")
        if face_vertex_counts.dtype != wp.int32:
            raise ValueError(f"face_vertex_counts must have dtype wp.int32, got {face_vertex_counts.dtype}")
        if face_vertex_counts.shape[0] == 0:
            raise ValueError("face_vertex_counts array cannot be empty")

        num_faces = face_vertex_counts.shape[0]
        face_counts_numpy = face_vertex_counts.numpy()

        # Compute prefix sum for offsets (exclusive scan)
        offsets_numpy = np.zeros(num_faces + 1, dtype=np.int32)
        offsets_numpy[1:] = np.cumsum(face_counts_numpy)

        # Validate that total matches face_vertex_indices length
        total_indices = offsets_numpy[-1]
        if total_indices != face_vertex_indices.shape[0]:
            raise ValueError(f"Sum of face_vertex_counts ({total_indices}) must equal length of face_vertex_indices ({face_vertex_indices.shape[0]})")

        face_offsets = wp.array(offsets_numpy, dtype=wp.int32, device=points.device)

    else:
        # We have offsets, compute counts
        if face_vertex_offsets.ndim != 1:
            raise ValueError("face_vertex_offsets must be a 1D warp array")
        if face_vertex_offsets.dtype != wp.int32:
            raise ValueError(f"face_vertex_offsets must have dtype wp.int32, got {face_vertex_offsets.dtype}")
        if face_vertex_offsets.shape[0] < 2:
            raise ValueError("face_vertex_offsets must have at least 2 elements (for at least 1 face)")

        num_faces = face_vertex_offsets.shape[0] - 1
        offsets_numpy = face_vertex_offsets.numpy()

        # Validate first offset is 0
        if offsets_numpy[0] != 0:
            raise ValueError(f"face_vertex_offsets must start with 0, got {offsets_numpy[0]}")

        # Validate that offsets are non-decreasing
        if not np.all(offsets_numpy[1:] >= offsets_numpy[:-1]):
            raise ValueError("face_vertex_offsets must be non-decreasing")

        # Validate that last offset matches face_vertex_indices length
        total_indices = offsets_numpy[-1]
        if total_indices != face_vertex_indices.shape[0]:
            raise ValueError(f"Last offset in face_vertex_offsets ({total_indices}) must equal length of face_vertex_indices ({face_vertex_indices.shape[0]})")

        # Compute counts from offsets (differences)
        counts_numpy = np.diff(offsets_numpy)
        face_vertex_counts = wp.array(counts_numpy, dtype=wp.int32, device=points.device)
        face_offsets = face_vertex_offsets

    handle = DatasetHandle()
    handle.points = points
    handle.face_vertex_counts = face_vertex_counts
    handle.face_vertex_indices = face_vertex_indices
    handle.face_offsets = face_offsets
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()  # Empty cell links
    return handle


def create_dataset(points: wp.array, face_vertex_indices: wp.array, face_vertex_counts: wp.array = None, face_vertex_offsets: wp.array = None) -> dav.Dataset:
    """Create a surface mesh dataset following UsdGeomMesh specification.

    This is the recommended function for creating surface mesh datasets. It creates both the
    dataset handle and wraps it in a dav.Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D vertex positions (wp.vec3f) - UsdGeomMesh 'points'
        face_vertex_indices: Flattened vertex indices for all faces (wp.int32) - UsdGeomMesh 'faceVertexIndices'
        face_vertex_counts: Number of vertices per face (wp.int32) - UsdGeomMesh 'faceVertexCounts' (optional if face_vertex_offsets provided)
        face_vertex_offsets: Offsets into face_vertex_indices (wp.int32, size=num_faces+1) (optional if face_vertex_counts provided)

    Returns:
        dav.Dataset: A new dav.Dataset instance with surface mesh data model

    Raises:
        ValueError: If array dimensions or dtypes are invalid, if array sizes are inconsistent,
                   or if neither face_vertex_counts nor face_vertex_offsets is provided

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom.surface_mesh import create_dataset
        >>> # Triangle mesh example with counts
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=wp.vec3f)
        >>> face_vertex_counts = wp.array([3, 3], dtype=wp.int32)  # Two triangles
        >>> face_vertex_indices = wp.array([0, 1, 2, 1, 3, 2], dtype=wp.int32)
        >>> dataset = create_dataset(points, face_vertex_indices, face_vertex_counts=face_vertex_counts)
        >>> print(dataset.get_num_cells())
        2
    """
    handle = create_handle(points, face_vertex_indices, face_vertex_counts, face_vertex_offsets)
    return dav.Dataset(DataModel, handle, points.device)


@wp.struct
class CellHandle:
    """Cell handle for a surface mesh face.

    In surface meshes, cells are faces (polygons).
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
        """Get the number of vertices in a face."""
        return ds.face_vertex_counts[cell.cell_id]

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a vertex ID from a face by local index."""
        start_offset = ds.face_offsets[cell.cell_id]
        return ds.face_vertex_indices[start_offset + local_idx]

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell.

        For surface meshes (2D cells), this returns 0 as the concept of
        3D cell faces doesn't apply. The cells themselves ARE the faces.
        """
        return 0

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face.

        For surface meshes, this returns 0 as faces don't apply.
        """
        return 0

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face.

        For surface meshes, this returns -1 as faces don't apply.
        """
        return -1


class DatasetAPI:
    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        """Get cell ID from cell index (same for surface meshes)."""
        return cell_idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        """Get cell index from cell ID (same for surface meshes)."""
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
        """Get the number of faces in the mesh."""
        return ds.face_vertex_counts.shape[0]

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        """Get the number of vertices in the mesh."""
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        """Get point ID from point index (same for surface meshes)."""
        return point_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        """Get point index from point ID (same for surface meshes)."""
        return point_id

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        """Get a vertex position by point ID."""
        return ds.points[point_id]

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'surface_mesh')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (success, locator) - Success flag and locator instance
        """
        locator = locators.build_cell_locator(data_model, ds, device)
        if locator is not None:
            ds.cell_bvh_id = locator.get_bvh_id()
            return (True, locator)
        else:
            ds.cell_bvh_id = 0
            return (False, None)

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

        Note: Interpolation is not yet implemented for surface meshes.
        This always returns an empty interpolated cell handle.
        """
        # TODO: Implement interpolation for triangles, quads, and n-gons
        return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell (face) containing a point.

        Note: Cell location is not yet implemented for surface meshes.
        This always returns an empty cell handle.
        """
        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        return False


# use generic cell links model
CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(-1), emptyCellId=wp.int32(-1), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (Surface Mesh)"


# DataModel protocol implementation
class DataModel(metaclass=DataModelMeta):
    """Surface Mesh data model implementation following UsdGeomMesh specification."""

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
    """Get the Surface Mesh data model."""
    return DataModel
