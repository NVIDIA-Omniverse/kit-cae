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
SIDS Unstructured Grid Data Model
===================================

This module provides a data model implementation for CGNS/SIDS Unstructured Grid.

CGNS (CFD General Notation System) SIDS (Standard Interface Data Structures) represents
computational meshes with:
- Explicit point coordinates
- Element sections with connectivity arrays
- Support for multiple element types (TETRA, HEXA, PYRA, PENTA, etc.)

Type System
-----------
- **Point IDs**: wp.int32 (0-based internally, CGNS convention)
- **Cell IDs**: wp.int32 (0-based internally)
- **Indices**: wp.int32 (0-based)

IMPORTANT - Node Ordering Conventions:
======================================
CGNS/SIDS and VTK use different node ordering conventions for some element types.
Our shape functions follow VTK conventions (as documented in each shape function module).
This module handles the necessary node reordering from CGNS to VTK convention.

Element Ordering Differences:
-----------------------------
0. TRI_3, QUAD_4, and NGON_n (2.5D Surface Elements):
   These are 2.5D surface elements embedded in 3D space.
   - get_num_faces() returns 1 (the element itself is treated as 1 face)
   - This allows cell_faces operator to extract them as surface meshes
   - No interpolation or point location support for 2.5D elements
   - CGNS and VTK use the same ordering for these elements

1. HEXA_8 (Hexahedron):
   CGNS: 0-1-2-3 (bottom face), 4-5-6-7 (top face)
   VTK:  0-1-2-3 (bottom face), 4-5-6-7 (top face)
   → SAME ordering, no reordering needed

2. PENTA_6 (Wedge/Prism): **DIFFERENT ORDERING**
   CGNS: Bottom triangle nodes 0-2-1, top triangle nodes 3-5-4
   VTK:  Bottom triangle nodes 0-1-2, top triangle nodes 3-4-5
   → Reordering needed: CGNS[0,1,2,3,4,5] → VTK[0,2,1,3,5,4]

3. PYRA_5 (Pyramid):
   CGNS: 0-1-2-3 (quad base), 4 (apex)
   VTK:  0-1-2-3 (quad base), 4 (apex)
   → SAME ordering, no reordering needed

4. TETRA_4 (Tetrahedron):
   CGNS: 0-1-2-3
   VTK:  0-1-2-3
   → SAME ordering, no reordering needed

Key Features
------------
- Supports multiple element types in same mesh
- Explicit topology stored in element connectivity arrays
- BVH-based locators for efficient cell location
- Explicit cell links for point-to-cell queries
- Maximum 8 points per cell (configurable via MAX_CELL_POINTS)
- Automatic node reordering from CGNS to VTK conventions
"""

from typing import Any

import warp as wp

import dav
from dav import locators
from dav.shape_functions import dispatcher as shape_functions_dispatcher

from . import sids_shapes


@wp.struct
class DatasetHandle:
    grid_coords: wp.array(dtype=wp.vec3f)
    element_type: wp.int32
    element_range: wp.vec2i  # NOTE: this is inclusive range [start, end]
    element_connectivity: wp.array(dtype=wp.int32)
    element_start_offset: wp.array(dtype=wp.int32)
    cell_bvh_id: wp.uint64
    cell_links: locators.CellLinks


def create_handle(grid_coords: wp.array, element_type: int, element_range: wp.vec2i, element_connectivity: wp.array, element_start_offset: wp.array = None) -> DatasetHandle:
    """Create a CGNS SIDS unstructured dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        grid_coords: Array of 3D grid coordinates (wp.vec3f)
        element_type: CGNS element type ID (e.g., HEXA_8, TETRA_4, MIXED, etc.)
        element_range: Inclusive range [start, end] of element IDs (wp.vec2i)
        element_connectivity: Array of point indices forming element connectivity (wp.int32).
                             For MIXED sections, includes element type IDs inline
        element_start_offset: Optional array of offsets into element_connectivity (wp.int32).
                             When present, length must be num_elements + 1 (includes past-the-end offset).
                             Can be empty array if not needed (Default: None)

    Returns:
        DatasetHandle: A new CGNS SIDS unstructured dataset handle

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or if element_range is invalid

    Note:
        The cell_bvh_id and cell_links will be initialized to default values.
        Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links() to build
        the spatial acceleration structures after creating the dataset.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.sids.unstructured import create_handle, HEXA_8
        >>> grid_coords = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> element_range = wp.vec2i(1, 10)  # Elements 1-10 (CGNS 1-indexed)
        >>> element_connectivity = wp.array([1, 2, 3, 4, 5, 6, 7, 8, ...], dtype=wp.int32)
        >>> element_start_offset = wp.array([0, 8, 16, ...], dtype=wp.int32)
        >>> handle = create_handle(grid_coords, HEXA_8, element_range,
        ...                        element_connectivity, element_start_offset)
    """
    from . import utils as sids_utils  # avoid circular import by importing here

    # Validate grid_coords
    if grid_coords is None or grid_coords.ndim != 1:
        raise ValueError("grid_coords must be a 1D warp array")
    if grid_coords.dtype != wp.vec3f:
        raise ValueError(f"grid_coords must have dtype wp.vec3f, got {grid_coords.dtype}")
    if grid_coords.shape[0] == 0:
        raise ValueError("grid_coords array cannot be empty")

    # Validate element_range (inclusive range [start, end])
    if element_range.y < element_range.x:
        raise ValueError(f"element_range end ({element_range.y}) must be >= start ({element_range.x})")

    # Validate element_connectivity
    if element_connectivity is None or element_connectivity.ndim != 1:
        raise ValueError("element_connectivity must be a 1D warp array")
    if element_connectivity.dtype != wp.int32:
        raise ValueError(f"element_connectivity must have dtype wp.int32, got {element_connectivity.dtype}")
    if element_connectivity.shape[0] == 0:
        raise ValueError("element_connectivity array cannot be empty")

    # Validate element_start_offset
    if element_start_offset is None or element_start_offset.ndim != 1:
        if element_type in (sids_shapes.ET_NGON_n, sids_shapes.ET_NFACE_n):
            raise ValueError(f"element_start_offset is required for element type {element_type}")
        elif element_type == sids_shapes.ET_MIXED:
            element_start_offset = sids_utils.compute_mixed_element_start_offset(element_range, element_connectivity)
    elif element_start_offset.dtype != wp.int32:
        raise ValueError(f"element_start_offset must have dtype wp.int32, got {element_start_offset.dtype}")

    if element_start_offset is not None:
        # element_start_offset is optional: either empty or has length = num_elements + 1
        # CGNS stores num_elements + 1 (includes past-the-end offset for computing last element length)
        num_elements = element_range.y - element_range.x + 1
        if element_start_offset.shape[0] > 0 and element_start_offset.shape[0] != num_elements + 1:
            raise ValueError(f"element_start_offset length ({element_start_offset.shape[0]}) must be empty or {num_elements + 1}, got {element_start_offset.shape[0]}")

    handle = DatasetHandle()
    handle.grid_coords = grid_coords
    handle.element_type = wp.int32(element_type)
    handle.element_range = element_range
    handle.element_connectivity = element_connectivity
    handle.element_start_offset = element_start_offset
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()  # Empty cell links
    return handle


def create_dataset(grid_coords: wp.array, element_type: int, element_range: wp.vec2i, element_connectivity: wp.array, element_start_offset: wp.array = None) -> dav.Dataset:
    """Create a CGNS SIDS unstructured dataset.

    This is the recommended function for creating CGNS SIDS unstructured datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        grid_coords: Array of 3D grid coordinates (wp.vec3f)
        element_type: CGNS element type ID (e.g., HEXA_8, TETRA_4, MIXED, etc.)
        element_range: Inclusive range [start, end] of element IDs (wp.vec2i)
        element_connectivity: Array of point indices forming element connectivity (wp.int32).
                             For MIXED sections, includes element type IDs inline
        element_start_offset: Optional array of offsets into element_connectivity (wp.int32).
                             When present, length must be num_elements + 1 (includes past-the-end offset).
                             Can be empty array if not needed

    Returns:
        dav.Dataset: A new Dataset instance with CGNS SIDS unstructured data model

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or if element_range is invalid

    Example:
        >>> import warp as wp
        >>> from dav.data_models.sids.unstructured import create_dataset, ET_HEXA_8
        >>> grid_coords = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> element_range = wp.vec2i(1, 10)  # Elements 1-10 (CGNS 1-indexed)
        >>> element_connectivity = wp.array([1, 2, 3, 4, 5, 6, 7, 8, ...], dtype=wp.int32)
        >>> element_start_offset = wp.array([0, 8, 16, ...], dtype=wp.int32)
        >>> dataset = create_dataset(grid_coords, ET_HEXA_8, element_range,
        ...                          element_connectivity, element_start_offset)
        >>> print(dataset.get_num_cells())
        10
    """
    from . import utils as sids_utils  # avoid circular import by importing here

    handle = create_handle(grid_coords, element_type, element_range, element_connectivity, element_start_offset)
    if element_type == sids_shapes.ET_MIXED:
        assert handle.element_start_offset is not None, "element_start_offset must be provided/computed for MIXED element type"
        assert handle.element_start_offset.shape[0] == (element_range.y - element_range.x + 1) + 1, "invalid element_start_offset length for MIXED element type"
        # need to determine element types used by the MIXED section to build the appropriate shape functions API
        element_types_in_mixed = sids_utils.get_element_types_in_mixed_section(handle.element_range, handle.element_connectivity, handle.element_start_offset)
        element_types = element_types_in_mixed + [sids_shapes.ET_MIXED]  # Ensure MIXED type itself is included for API construction
    else:
        element_types = [element_type]

    # sort element types
    element_types = sorted(set(element_types))
    data_model = get_data_model([sids_shapes.get_element_type_as_string(et) for et in element_types])
    return dav.Dataset(data_model, handle, grid_coords.device)


@wp.struct
class CellHandle:
    cell_id: wp.int32


class DatasetAPI:
    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
        return ds.element_range.x + idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        return id - ds.element_range.x

    @staticmethod
    @dav.func
    def get_cell(dataset: DatasetHandle, id: wp.int32) -> CellHandle:
        cell = CellHandle()
        cell.cell_id = id
        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.element_range.y - ds.element_range.x + 1

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.grid_coords.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        return point_idx + 1  # SIDS uses 1-based indexing

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        return point_id - 1  # Convert from 1-based to 0-based

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        return ds.grid_coords[DatasetAPI.get_point_idx_from_id(ds, point_id)]

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None):
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'unstructured')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (status, locator) - Status code and CellLocator instance
        """
        locator = locators.build_cell_locator(data_model, ds, device)
        if locator is not None:
            ds.cell_bvh_id = locator.get_bvh_id()
            return (True, locator)
        else:
            ds.cell_bvh_id = 0
            return (False, None)

    @staticmethod
    def build_cell_links(data_model, ds: DatasetHandle, device=None):
        """Build the cell links for the dataset.

        Args:
            data_model: The data model module
            ds: The dataset
            device: Device to build the links on

        Returns:
            tuple: (status, links) - Status code and CellLinks instance
        """
        cell_links = locators.build_cell_links(data_model, ds, device)
        if cell_links is not None:
            ds.cell_links = cell_links
            return (True, cell_links)
        else:
            ds.cell_links = None
            return (False, None)


# use generic cell links api
CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(0), emptyCellId=wp.int32(0), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


@dav.cached
def get_data_model(dm_element_types: list[int | str]) -> Any:
    """Get the data model module for a given element type.

    For this unstructured data model, we return the same module regardless of element type,
    since it supports multiple element types in the same mesh. This function is provided
    for API consistency and future extensibility.

    Args:
        dm_element_types: List of CGNS element type IDs or string names (e.g., HEXA_8, TETRA_4, MIXED, etc.
        or "hexa_8", "tetra_4", "mixed", etc.)

    Returns:
        The data model module (in this case, the current module)
    """

    # convert dm_element_types to ints if they are provided as strings
    dm_element_types = [sids_shapes.get_element_type_from_string(et) if isinstance(et, str) else et for et in dm_element_types]

    ShapesLibrary = sids_shapes.get_shapes_library(dm_element_types)

    DATA_MODEL_ELEMENT_TYPES = wp.static(dm_element_types)

    class DataModelMeta(type):
        def __repr__(cls):
            element_type_names = [sids_shapes.get_element_type_as_string(et) for et in DATA_MODEL_ELEMENT_TYPES]
            return f"DataModel (SIDS Unstructured, element types: {element_type_names})"

    class DataModel(metaclass=DataModelMeta):
        pass

    DataModel.DatasetHandle = DatasetHandle
    DataModel.CellHandle = CellHandle
    DataModel.PointIdHandle = wp.int32  # point id is just an integer index
    DataModel.CellIdHandle = wp.int32  # cell id is just an integer index
    DataModel.DatasetAPI = DatasetAPI
    DataModel.CellLinksAPI = CellLinksAPI

    # Standalone helpers defined before SIDSCellAPI so that methods can call them
    # without referencing SIDSCellAPI mid-class-body (which would be an empty closure cell).

    @dav.func
    def _sids_cell_is_valid(cell: CellHandle) -> wp.bool:
        # SIDS cell ids are 1-based; invalid cell has id <= 0
        return cell.cell_id > 0

    @dav.func
    def _sids_cell_get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        assert _sids_cell_is_valid(cell), "Cell is not valid"
        if wp.static(sids_shapes.ET_NFACE_n in DATA_MODEL_ELEMENT_TYPES or sids_shapes.ET_NGON_n in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_NGON_n or ds.element_type == sids_shapes.ET_NFACE_n
            # For NGON_n, the number of points is determined from the offset range
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            end_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x + 1]  # Next offset
            return end_offset - start_offset
        elif wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # For MIXED, we need to read the element type from the connectivity array
            # The first integer in the connectivity for this cell is the element type
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            cell_type = ds.element_connectivity[start_offset]
            return ShapesLibrary.get_num_corner_nodes(cell_type)
        else:
            return ShapesLibrary.get_num_corner_nodes(ds.element_type)

    @dav.func
    def _sids_cell_get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _sids_cell_is_valid(cell), "Cell is not valid"
        assert local_idx >= 0, "Local point index must be non-negative"
        assert local_idx < _sids_cell_get_num_points(cell, ds), "Local point index out of range"

        if wp.static(sids_shapes.ET_NFACE_n in DATA_MODEL_ELEMENT_TYPES or sids_shapes.ET_NGON_n in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_NGON_n or ds.element_type == sids_shapes.ET_NFACE_n
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            return ds.element_connectivity[start_offset + local_idx]
        elif wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # no need to defer to Shape function for this one.
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            # +1 to skip the cell type at the start of the connectivity for this cell
            return ds.element_connectivity[start_offset + local_idx + 1]
        else:
            # uniform cell type.
            start_offset = (cell.cell_id - ds.element_range.x) * ShapesLibrary.get_num_all_nodes(ds.element_type)
            return ds.element_connectivity[start_offset + local_idx]

    @dav.func
    def _sids_cell_get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        assert _sids_cell_is_valid(cell), "Cell is not valid"
        assert ds.element_type != sids_shapes.ET_NFACE_n, "NFACE_n elements should not call get_num_faces"

        if wp.static(sids_shapes.ET_NGON_n in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_NGON_n, "N-gon element type expected in dataset"
            return 1
        elif wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # For MIXED, we need to read the element type from the connectivity array
            # The first integer in the connectivity for this cell is the element type
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            cell_type = ds.element_connectivity[start_offset]
            return ShapesLibrary.get_num_faces(cell_type)
        else:
            return ShapesLibrary.get_num_faces(ds.element_type)

    @dav.func
    def _sids_cell_get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _sids_cell_is_valid(cell), "Cell is not valid"
        assert ds.element_type != sids_shapes.ET_NFACE_n, "NFACE_n elements should not call get_face_num_points"
        assert face_idx >= 0, "Face index must be non-negative"
        assert face_idx < _sids_cell_get_num_faces(cell, ds), "Face index out of range"

        if wp.static(sids_shapes.ET_NGON_n in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_NGON_n, "N-gon element type expected in dataset"
            return _sids_cell_get_num_points(cell, ds)  # The whole element is one face with all its points
        elif wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # For MIXED, we need to read the element type from the connectivity array
            # The first integer in the connectivity for this cell is the element type
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            cell_type = ds.element_connectivity[start_offset]
            return ShapesLibrary.get_num_face_corner_nodes(cell_type, face_idx)
        else:
            return ShapesLibrary.get_num_face_corner_nodes(ds.element_type, face_idx)

    @dav.func
    def _sids_cell_get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _sids_cell_is_valid(cell), "Cell is not valid"
        assert ds.element_type != sids_shapes.ET_NFACE_n, "NFACE_n elements should not call get_face_point_id"
        assert face_idx >= 0, "Face index must be non-negative"
        assert face_idx < _sids_cell_get_num_faces(cell, ds), "Face index out of range"
        assert local_idx >= 0, "Local point index must be non-negative"
        assert local_idx < _sids_cell_get_face_num_points(cell, face_idx, ds), "Local point index out of range"

        if wp.static(sids_shapes.ET_NGON_n in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_NGON_n, "N-gon element type expected in dataset"
            return _sids_cell_get_point_id(cell, local_idx, ds)  # The whole element is one face, so local_idx maps directly to the cell's point ids
        elif wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # For MIXED, we need to read the element type from the connectivity array
            # The first integer in the connectivity for this cell is the element type
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            cell_type = ds.element_connectivity[start_offset]
            cell_local_idx = ShapesLibrary.get_face_corner_node_index(cell_type, face_idx, local_idx)
            return _sids_cell_get_point_id(cell, cell_local_idx, ds)
        else:
            cell_local_idx = ShapesLibrary.get_face_corner_node_index(ds.element_type, face_idx, local_idx)
            return _sids_cell_get_point_id(cell, cell_local_idx, ds)

    class SIDSCellAPI:
        @staticmethod
        @dav.func
        def is_valid(cell: CellHandle) -> wp.bool:
            return _sids_cell_is_valid(cell)

        @staticmethod
        @dav.func
        def empty() -> CellHandle:
            cell = CellHandle()
            cell.cell_id = 0  # Invalid cell ID (SIDS is 1-based, so 0 is invalid)
            return cell

        @staticmethod
        @dav.func
        def get_cell_id(cell: CellHandle) -> wp.int32:
            return cell.cell_id

        @staticmethod
        @dav.func
        def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            return _sids_cell_get_num_points(cell, ds)

        @staticmethod
        @dav.func
        def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            return _sids_cell_get_point_id(cell, local_idx, ds)

        @staticmethod
        @dav.func
        def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            """Get the number of faces in a cell."""
            return _sids_cell_get_num_faces(cell, ds)

        @staticmethod
        @dav.func
        def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get the number of points in a face."""
            return _sids_cell_get_face_num_points(cell, face_idx, ds)

        @staticmethod
        @dav.func
        def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get a point ID from a face."""
            return _sids_cell_get_face_point_id(cell, face_idx, local_idx, ds)

    DataModel.CellAPI = SIDSCellAPI

    ShapeDispatch = shape_functions_dispatcher.get_shape_dispatcher(DataModel, ShapesLibrary, DATA_MODEL_ELEMENT_TYPES)

    # Standalone helper defined before SIDSCellLocatorAPI so that evaluate_position and
    # find_cell_containing_point can call it without referencing SIDSCellLocatorAPI
    # mid-class-body (which would be an empty closure cell).
    @dav.func
    def _sids_get_cell_element_type(ds: DatasetHandle, cell: CellHandle) -> wp.int32:
        if wp.static(sids_shapes.ET_MIXED in DATA_MODEL_ELEMENT_TYPES):
            assert ds.element_type == sids_shapes.ET_MIXED, "MIXED element type expected in dataset"
            # For MIXED, we need to read the element type from the connectivity array
            # The first integer in the connectivity for this cell is the element type
            start_offset = ds.element_start_offset[cell.cell_id - ds.element_range.x]  # Convert to 0-based index
            return ds.element_connectivity[start_offset]
        else:
            return ds.element_type

    class SIDSCellLocatorAPI:
        @staticmethod
        @dav.func
        def _get_cell_element_type(ds: DatasetHandle, cell: CellHandle) -> wp.int32:
            return _sids_get_cell_element_type(ds, cell)

        @staticmethod
        @dav.func
        def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
            assert SIDSCellAPI.is_valid(cell), "Cell is not valid"
            cell_element_type = _sids_get_cell_element_type(ds, cell)
            return ShapeDispatch.get_weights(position, cell, ds, cell_element_type)

        @staticmethod
        @dav.func
        def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
            assert ds.cell_bvh_id != 0, "Cell locator not built for dataset"

            cell_element_type = wp.int32(0)

            if SIDSCellAPI.is_valid(hint):
                cell_element_type = _sids_get_cell_element_type(ds, hint)
                if ShapeDispatch.is_point_in_cell(position, hint, ds, cell_element_type):
                    return hint

            radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
            query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
            cell_idx = wp.int32(-1)
            while wp.bvh_query_next(query, cell_idx):
                cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
                cell = DatasetAPI.get_cell(ds, cell_id)
                assert SIDSCellAPI.is_valid(cell), "Queried cell from BVH is not valid"
                cell_element_type = _sids_get_cell_element_type(ds, cell)
                if ShapeDispatch.is_point_in_cell(position, cell, ds, cell_element_type):
                    return cell

            return SIDSCellAPI.empty()

        @staticmethod
        @dav.func
        def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
            assert SIDSCellAPI.is_valid(cell), "Cell is not valid"
            cell_element_type = _sids_get_cell_element_type(ds, cell)
            return ShapeDispatch.is_point_in_cell(point, cell, ds, cell_element_type)

    DataModel.CellLocatorAPI = SIDSCellLocatorAPI
    return DataModel
