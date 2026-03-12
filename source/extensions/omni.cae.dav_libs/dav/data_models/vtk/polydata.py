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
VTK PolyData Data Model
=======================

This module provides a data model implementation for VTK PolyData (vtkPolyData).

PolyData represents surface meshes and point clouds with:
- Explicit point coordinates
- Separate arrays for different cell types: verts (0D), lines (1D), polys (2D)
- Explicit cell-to-point connectivity stored in arrays
- Triangle strips are NOT supported (will raise errors)

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous across all cell types)
- **Indices**: wp.int32 (same as IDs for polydata)

VTK PolyData Structure
----------------------
VTK PolyData has three separate cell arrays:
1. **Verts**: 0D cells (individual vertices/points)
2. **Lines**: 1D cells (polylines)
3. **Polys**: 2D cells (triangles, quads, general polygons) - treated as 2.5D elements

Cell ID Ordering:
- Cell IDs are assigned sequentially: verts first, then lines, then polys
- Example: If 5 verts and 3 lines, polys start at cell_id 8

2.5D Element Treatment:
- Polys are treated as 2.5D surface elements (like TRI_3, QUAD_4, NGON_n in SIDS)
- get_num_faces() returns 1 for polys (the polygon itself is treated as 1 face)
- This allows cell_faces operator to extract polys as surface meshes
- No interpolation or point location support for 2.5D elements

Key Features
------------
- Supports vertices, lines, and polygons in separate arrays
- Explicit topology stored in connectivity arrays
- BVH-based locators for efficient cell location
- Explicit cell links for point-to-cell queries
- Maximum 8 points per cell (configurable via MAX_CELL_POINTS)

Limitations
-----------
- Triangle strips (VTK_TRIANGLE_STRIP) are not supported and will raise errors
"""

from typing import Any

import warp as wp

import dav
from dav import locators

MAX_CELL_POINTS = 8  # maximum number of points per cell we will support

# Cell type constants for distinguishing between verts, lines, and polys
CT_VERT = wp.constant(0)
CT_LINE = wp.constant(1)
CT_POLY = wp.constant(2)


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)

    # Verts arrays (0D cells)
    verts_offsets: wp.array(dtype=wp.int32)  # length = num_verts + 1
    verts_connectivity: wp.array(dtype=wp.int32)

    # Lines arrays (1D cells)
    lines_offsets: wp.array(dtype=wp.int32)  # length = num_lines + 1
    lines_connectivity: wp.array(dtype=wp.int32)

    # Polys arrays (2D cells - triangles, quads, polygons)
    polys_offsets: wp.array(dtype=wp.int32)  # length = num_polys + 1
    polys_connectivity: wp.array(dtype=wp.int32)

    cell_links: locators.CellLinks


def create_handle(
    points: wp.array,
    verts_offsets: wp.array = None,
    verts_connectivity: wp.array = None,
    lines_offsets: wp.array = None,
    lines_connectivity: wp.array = None,
    polys_offsets: wp.array = None,
    polys_connectivity: wp.array = None,
) -> DatasetHandle:
    """Create a polydata dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        verts_offsets: Optional offsets into verts_connectivity (wp.int32, length = num_verts + 1)
        verts_connectivity: Optional vertex connectivity (wp.int32)
        lines_offsets: Optional offsets into lines_connectivity (wp.int32, length = num_lines + 1)
        lines_connectivity: Optional line connectivity (wp.int32)
        polys_offsets: Optional offsets into polys_connectivity (wp.int32, length = num_polys + 1)
        polys_connectivity: Optional polygon connectivity (wp.int32)

    Returns:
        DatasetHandle: A new polydata dataset handle

    Raises:
        ValueError: If array dimensions or dtypes are invalid
        ValueError: If at least one cell type is not provided

    Note:
        - At least one of verts, lines, or polys must be provided
        - Empty arrays are created for missing cell types
        - Cell IDs are assigned sequentially: verts first, then lines, then polys
        - cell_bvh_id and cell_links will be initialized to default values
        - Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links()
          to build spatial acceleration structures after creating the dataset

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.polydata import create_handle
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3f)
        >>> polys_offsets = wp.array([0, 3], dtype=wp.int32)
        >>> polys_connectivity = wp.array([0, 1, 2], dtype=wp.int32)
        >>> handle = create_handle(points, polys_offsets=polys_offsets, polys_connectivity=polys_connectivity)
    """
    # Validate points
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    device = points.device

    # Helper to create empty offset/connectivity arrays
    def make_empty_arrays():
        return (wp.array([0], dtype=wp.int32, device=device), wp.zeros(0, dtype=wp.int32, device=device))

    # Process verts
    if verts_offsets is not None or verts_connectivity is not None:
        if verts_offsets is None or verts_connectivity is None:
            raise ValueError("Both verts_offsets and verts_connectivity must be provided together")
        if verts_offsets.dtype != wp.int32:
            raise ValueError(f"verts_offsets must have dtype wp.int32, got {verts_offsets.dtype}")
        if verts_connectivity.dtype != wp.int32:
            raise ValueError(f"verts_connectivity must have dtype wp.int32, got {verts_connectivity.dtype}")
    else:
        verts_offsets, verts_connectivity = make_empty_arrays()

    # Process lines
    if lines_offsets is not None or lines_connectivity is not None:
        if lines_offsets is None or lines_connectivity is None:
            raise ValueError("Both lines_offsets and lines_connectivity must be provided together")
        if lines_offsets.dtype != wp.int32:
            raise ValueError(f"lines_offsets must have dtype wp.int32, got {lines_offsets.dtype}")
        if lines_connectivity.dtype != wp.int32:
            raise ValueError(f"lines_connectivity must have dtype wp.int32, got {lines_connectivity.dtype}")
    else:
        lines_offsets, lines_connectivity = make_empty_arrays()

    # Process polys
    if polys_offsets is not None or polys_connectivity is not None:
        if polys_offsets is None or polys_connectivity is None:
            raise ValueError("Both polys_offsets and polys_connectivity must be provided together")
        if polys_offsets.dtype != wp.int32:
            raise ValueError(f"polys_offsets must have dtype wp.int32, got {polys_offsets.dtype}")
        if polys_connectivity.dtype != wp.int32:
            raise ValueError(f"polys_connectivity must have dtype wp.int32, got {polys_connectivity.dtype}")
    else:
        polys_offsets, polys_connectivity = make_empty_arrays()

    # Ensure at least one cell type is provided
    num_verts = verts_offsets.shape[0] - 1
    num_lines = lines_offsets.shape[0] - 1
    num_polys = polys_offsets.shape[0] - 1

    if num_verts == 0 and num_lines == 0 and num_polys == 0:
        raise ValueError("At least one cell type (verts, lines, or polys) must be provided")

    handle = DatasetHandle()
    handle.points = points
    handle.verts_offsets = verts_offsets
    handle.verts_connectivity = verts_connectivity
    handle.lines_offsets = lines_offsets
    handle.lines_connectivity = lines_connectivity
    handle.polys_offsets = polys_offsets
    handle.polys_connectivity = polys_connectivity
    handle.cell_links = locators.CellLinks()  # Empty cell links
    return handle


def create_dataset(
    points: wp.array,
    verts_offsets: wp.array = None,
    verts_connectivity: wp.array = None,
    lines_offsets: wp.array = None,
    lines_connectivity: wp.array = None,
    polys_offsets: wp.array = None,
    polys_connectivity: wp.array = None,
) -> dav.Dataset:
    """Create a polydata dataset.

    This is the recommended function for creating polydata datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        verts_offsets: Optional offsets into verts_connectivity (wp.int32, length = num_verts + 1)
        verts_connectivity: Optional vertex connectivity (wp.int32)
        lines_offsets: Optional offsets into lines_connectivity (wp.int32, length = num_lines + 1)
        lines_connectivity: Optional line connectivity (wp.int32)
        polys_offsets: Optional offsets into polys_connectivity (wp.int32, length = num_polys + 1)
        polys_connectivity: Optional polygon connectivity (wp.int32)

    Returns:
        dav.Dataset: A new Dataset instance with polydata data model

    Raises:
        ValueError: If array dimensions or dtypes are invalid
        ValueError: If at least one cell type is not provided

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.polydata import create_dataset
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3f)
        >>> polys_offsets = wp.array([0, 3], dtype=wp.int32)
        >>> polys_connectivity = wp.array([0, 1, 2], dtype=wp.int32)
        >>> dataset = create_dataset(points, polys_offsets=polys_offsets, polys_connectivity=polys_connectivity)
        >>> print(dataset.get_num_cells())
        1
    """
    handle = create_handle(points, verts_offsets, verts_connectivity, lines_offsets, lines_connectivity, polys_offsets, polys_connectivity)
    return dav.Dataset(DataModel, handle, points.device)


@wp.struct
class CellHandle:
    cell_id: wp.int32
    cell_type: wp.int32  # CT_VERT, CT_LINE, or CT_POLY
    local_id: wp.int32  # Index within the specific cell type array


class CellAPI:
    @staticmethod
    @dav.func
    def is_valid(cell: CellHandle) -> wp.bool:
        return cell.cell_id >= 0

    @staticmethod
    @dav.func
    def empty() -> CellHandle:
        cell = CellHandle()
        cell.cell_id = -1
        cell.cell_type = -1
        cell.local_id = -1
        return cell

    @staticmethod
    @dav.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        return cell.cell_id

    @staticmethod
    @dav.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        if cell.cell_type == CT_VERT:
            start = ds.verts_offsets[cell.local_id]
            end = ds.verts_offsets[cell.local_id + 1]
            return end - start
        elif cell.cell_type == CT_LINE:
            start = ds.lines_offsets[cell.local_id]
            end = ds.lines_offsets[cell.local_id + 1]
            return end - start
        elif cell.cell_type == CT_POLY:
            start = ds.polys_offsets[cell.local_id]
            end = ds.polys_offsets[cell.local_id + 1]
            return end - start
        return 0

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        if cell.cell_type == CT_VERT:
            start = ds.verts_offsets[cell.local_id]
            return ds.verts_connectivity[start + local_idx]
        elif cell.cell_type == CT_LINE:
            start = ds.lines_offsets[cell.local_id]
            return ds.lines_connectivity[start + local_idx]
        elif cell.cell_type == CT_POLY:
            start = ds.polys_offsets[cell.local_id]
            return ds.polys_connectivity[start + local_idx]
        return -1

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell.

        For polydata:
        - Polys (2.5D surface elements) return 1 (the polygon itself is treated as 1 face)
        - Verts and lines return 0 (no faces)
        """
        if cell.cell_type == CT_POLY:
            return 1  # 2.5D element - treat as having 1 face (itself)
        return 0

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face.

        For polys, face 0 is the polygon itself (all its vertices).
        """
        if cell.cell_type == CT_POLY and face_idx == 0:
            start = ds.polys_offsets[cell.local_id]
            end = ds.polys_offsets[cell.local_id + 1]
            return end - start
        return 0

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face.

        For polys, face 0 returns the polygon's vertices directly.
        """
        if cell.cell_type == CT_POLY and face_idx == 0:
            start = ds.polys_offsets[cell.local_id]
            return ds.polys_connectivity[start + local_idx]
        return -1


class CellLocatorAPI:
    """Static API for cell location operations."""

    @staticmethod
    @dav.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        """Evaluate position within a cell.

        Note: Interpolation is not supported for PolyData (2.5D/1D/0D elements).
        Always returns empty.
        """
        return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find cell containing a point.

        Note: Point location is not supported for PolyData (2.5D/1D/0D elements).
        Always returns empty.
        """
        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        return False


class DatasetAPI:
    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        return cell_idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        return cell_id

    @staticmethod
    @dav.func
    def get_cell(ds: DatasetHandle, cell_id: wp.int32) -> CellHandle:
        """Get a cell by its global cell ID.

        Cell IDs are assigned sequentially: verts first, then lines, then polys.
        """
        cell = CellHandle()
        cell.cell_id = cell_id

        num_verts = ds.verts_offsets.shape[0] - 1
        num_lines = ds.lines_offsets.shape[0] - 1

        if cell_id < num_verts:
            cell.cell_type = CT_VERT
            cell.local_id = cell_id
        elif cell_id < num_verts + num_lines:
            cell.cell_type = CT_LINE
            cell.local_id = cell_id - num_verts
        else:
            cell.cell_type = CT_POLY
            cell.local_id = cell_id - num_verts - num_lines

        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        num_verts = ds.verts_offsets.shape[0] - 1
        num_lines = ds.lines_offsets.shape[0] - 1
        num_polys = ds.polys_offsets.shape[0] - 1
        return num_verts + num_lines + num_polys

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        return point_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        return point_id

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        return ds.points[point_id]

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


# use generic cell links API
CellLinksAPI = locators.get_cell_links_api(emptyPointId=-1, emptyCellId=-1, DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (VTK PolyData)"


# DataModel protocol implementation
class DataModel(metaclass=DataModelMeta):
    """VTK PolyData data model implementation."""

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
    """Factory function to get the data model for VTK PolyData."""
    return DataModel
