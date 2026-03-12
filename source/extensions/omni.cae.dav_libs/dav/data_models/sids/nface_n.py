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
SIDS NFACE_n Polyhedral Data Model
====================================

This module provides a data model implementation for CGNS/SIDS polyhedral meshes
using NFACE_n and NGON_n element sections.

CGNS Polyhedral Representation
-------------------------------
CGNS represents arbitrary polyhedral cells using a two-level indirection:

- **NGON_n**: Defines arbitrary polygonal faces (face → vertex connectivity)
- **NFACE_n**: Defines polyhedral cells (cell → face connectivity)

Each NFACE_n element stores a list of NGON_n IDs (face indices). The sign of each
NGON_n ID indicates face orientation:
- **Positive**: Face normal points outward from the cell
- **Negative**: Face normal points inward (reversed winding order)

This allows representing arbitrary convex polyhedra: hexahedra, prisms, pyramids,
and more complex cells with any number of faces.

Compositional Architecture
--------------------------
This data model uses a **compositional approach** that reuses the existing
unstructured grid infrastructure:

1. **NFACE_n cells** are stored in an `UnstructuredDataModel.DatasetHandle`
   - Each cell's "connectivity" is a list of NGON_n IDs (face references)

2. **NGON_n faces** are stored in multiple `UnstructuredDataModel.DatasetHandle`s
   - Each face's "connectivity" is a list of vertex IDs
   - Binary search on `ngon_n_element_range_starts` locates the correct block

3. **Face traversal** happens dynamically:
   - Cell → NFACE connectivity → NGON IDs → NGON connectivity → Vertices
   - Handles orientation by reversing winding order for negative NGON IDs

Interpolation via Inverse Distance Weighting
--------------------------------------------
Since arbitrary polyhedra don't have standard parametric coordinates, we use
**inverse distance weighting (IDW)** for point location and field interpolation:

1. Iterate over the unique vertices of the polyhedron
2. Compute inverse-distance weights to the query point
3. Normalize weights to sum to 1

**Characteristics:**
- C⁰ continuity
- Simple and robust for convex polyhedra
- Works with arbitrary face counts and vertex counts
- Suitable for visualization and probing (not FEA-grade accuracy)

**Limitations:**
- Weights depend on vertex distribution and are not FEM-grade
- Does not provide smooth (C¹) gradients

Vertex Counting (Unique)
------------------------
The `get_num_points()` and `get_point_id()` methods return counts/IDs for
**unique vertices per cell**. Vertices shared between faces are returned once.

For example, a hexahedron has:
- 8 unique vertices

This supports IDW interpolation over unique vertices and avoids valence-based
weighting from duplicated face vertices.

Face Orientation Handling
--------------------------
When a face is referenced with a negative NGON_n ID, the winding order is
reversed to ensure consistent outward-facing normals:

- Original: [v0, v1, v2, v3]
- Reversed: [v0, v3, v2, v1]  (keep first, reverse rest)

This maintains CCW winding when viewed from outside the cell.

Example Usage
-------------
.. code-block:: python

    from dav.data_models.sids import nface_n

    # Assume ds is a nface_n.DatasetHandle loaded from CGNS

    # Get a polyhedral cell
    cell_id = nface_n.DatasetAPI.get_cell_id_from_idx(ds, 0)
    cell = nface_n.DatasetAPI.get_cell(ds, cell_id)

    # Query cell topology
    num_faces = nface_n.CellAPI.get_num_faces(cell, ds)  # e.g., 6 for hex
    for face_idx in range(num_faces):
        num_pts = nface_n.CellAPI.get_face_num_points(cell, face_idx, ds)
        print(f"Face {face_idx} has {num_pts} vertices")

    # Locate and interpolate at a point
    position = wp.vec3f(1.0, 2.0, 3.0)
    hint = nface_n.CellAPI.empty()

    found_cell = nface_n.DatasetAPI.find_cell_containing_point(ds, position, hint)
    weights = nface_n.DatasetAPI.evaluate_position(ds, position, found_cell)

Future Enhancements
-------------------
- **Mean Value Coordinates**: Higher quality interpolation (smooth, C¹)
- **Centroid Caching**: Precompute and store centroids during dataset creation
- **Unique Vertex Iteration**: Helper methods for unique vertex access
- **Non-convex Faces**: Ear-clipping triangulation for complex faces
- **Cell Links**: Point-to-cell connectivity (currently not supported)

References
----------
- CGNS SIDS: https://cgns.github.io/CGNS_docs_current/sids/
- Tetrahedral Decomposition: Standard technique in computational geometry
- Mean Value Coordinates: Ju et al. 2005, "Mean value coordinates for closed triangular meshes"
"""

import warp as wp

import dav
from dav import locators
from dav.shape_functions import star_convex_polyhedron
from dav.shape_functions import utils as shape_functions_utils

from . import sids_shapes
from .unstructured import CellHandle
from .unstructured import DatasetHandle as UGDatasetHandle

EPSILON = 1e-6


@wp.struct
class DatasetHandle:
    nface_n_block: UGDatasetHandle
    """UnstructuredDataModel.DatasetHandle for the NFACE_n element section (cell → face connectivity)"""

    ngon_n_element_range_starts: wp.array(dtype=wp.int32)
    """Starting element ID for each NGON_n element block"""

    ngon_n_blocks: wp.array(dtype=UGDatasetHandle)
    """Array of UnstructuredDataModel.DatasetHandle for NGON_n element sections (face → vertex connectivity)"""

    cell_bvh_id: wp.uint64
    """BVH ID for cell locator acceleration structure"""

    cell_links: locators.CellLinks
    """"CellLinks structure for point-to-cell connectivity (optional, may be None if not built)"""

    # -- acceleration structures --
    nface_n_connectivity: wp.array(dtype=wp.int32)
    """Flattened array of vertex IDs for all cells, with duplicates removed (for unique vertex iteration)"""

    nface_n_connectivity_offsets: wp.array(dtype=wp.int32)
    """Offsets into nface_n_connectivity for each cell's vertex list"""

    nface_n_cell_centers: wp.array(dtype=wp.vec3f)
    """Centers of each cell"""

    nface_n_cell_face_centers: wp.array(dtype=wp.vec3f)
    """Centers of each face within each cell"""


def create_handle(nface_n_block: UGDatasetHandle, ngon_n_blocks: list[UGDatasetHandle]) -> DatasetHandle:
    """Create a CGNS/SIDS NFACE_n polyhedral dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    This data model uses a compositional approach where NFACE_n cells reference
    NGON_n faces, and NGON_n faces reference vertices.

    Args:
        nface_n_block: UnstructuredDataModel.DatasetHandle containing the NFACE_n element section.
                      Each cell's connectivity is a list of NGON_n IDs (face references)
        ngon_n_blocks: List of UnstructuredDataModel.DatasetHandles, one for each NGON_n element section.
                      Each contains face-to-vertex connectivity

    Returns:
        DatasetHandle: A new NFACE_n polyhedral dataset handle

    Raises:
        ValueError: If arrays have invalid dimensions or dtypes, or if sizes are inconsistent

    Note:
        The cell_bvh_id will be initialized to 0. Use DatasetAPI.build_cell_locator() to build
        the spatial acceleration structure after creating the dataset.
    """
    from . import utils as sids_utils  # avoid circular import

    if not nface_n_block:
        raise ValueError("nface_n_block must be a valid UnstructuredDataModel.DatasetHandle")

    device = nface_n_block.grid_coords.device

    # Validate ngon_n_blocks
    if ngon_n_blocks is None or len(ngon_n_blocks) == 0:
        raise ValueError("ngon_n_blocks must be a non-empty list of UnstructuredDataModel.DatasetHandle")
    if not all(block.grid_coords.device == device for block in ngon_n_blocks):
        raise ValueError("All blocks in ngon_n_blocks must be on the same device as nface_n_block")

    # Build acceleration structures

    # sort ngon_n_blocks by their starting element ID (assumes element IDs are contiguous and start from 1)
    ngon_n_blocks = sorted(ngon_n_blocks, key=lambda block: block.element_range.x)
    ngon_n_element_range_starts = wp.array([block.element_range.x for block in ngon_n_blocks], dtype=wp.int32, device=device)

    handle = DatasetHandle()
    handle.nface_n_block = nface_n_block
    handle.ngon_n_element_range_starts = ngon_n_element_range_starts
    handle.ngon_n_blocks = wp.array(ngon_n_blocks, dtype=UGDatasetHandle, device=device)
    handle.cell_bvh_id = wp.uint64(0)

    # Now, populate the nfaced_connectivity and offsets arrays for direct access to face vertices without repeated lookups
    sids_utils.populate_nface_n_connectivity(handle)

    # allocate arrays that will be populated later
    nb_cells = nface_n_block.element_range.y - nface_n_block.element_range.x + 1
    handle.nface_n_cell_centers = wp.zeros(nb_cells, dtype=wp.vec3f, device=device)

    nb_faces = nface_n_block.element_connectivity.shape[0]  # total number of faces across all cells
    handle.nface_n_cell_face_centers = wp.zeros(nb_faces, dtype=wp.vec3f, device=device)

    return handle


def create_dataset(nface_n_block: UGDatasetHandle, ngon_n_blocks: list[UGDatasetHandle]) -> dav.Dataset:
    """Create a CGNS/SIDS NFACE_n polyhedral dataset.

    This is the recommended function for creating NFACE_n polyhedral datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    This data model uses a compositional approach where NFACE_n cells reference
    NGON_n faces, and NGON_n faces reference vertices.

    Args:
        nface_n_block: UnstructuredDataModel.DatasetHandle containing the NFACE_n element section.
                      Each cell's connectivity is a list of NGON_n IDs (face references)
        ngon_n_blocks: List of UnstructuredDataModel.DatasetHandles, one for each NGON_n element section.
                      Each contains face-to-vertex connectivity

    Returns:
        Dataset: A new Dataset instance with NFACE_n polyhedral data model

    Raises:
        ValueError: If arrays have invalid dimensions or dtypes, or if sizes are inconsistent

    Example:
        >>> import warp as wp
        >>> from dav.data_models.sids import nface_n, unstructured
        >>>
        >>> # Create NFACE_n block (cells -> faces)
        >>> nface_points = wp.array(...)  # Grid coordinates
        >>> nface_connectivity = wp.array([1, 2, 3, 4, 5, 6], dtype=wp.int32)  # NGON IDs
        >>> nface_block = unstructured.create_handle(...)
        >>>
        >>> # Create NGON_n blocks (faces -> vertices)
        >>> ngon_block1 = unstructured.create_handle(...)
        >>> ngon_block2 = unstructured.create_handle(...)
        >>> ngon_blocks = [ngon_block1, ngon_block2]
        >>>
        >>> dataset = create_dataset(nface_block, ngon_blocks)
        >>> print(dataset.get_num_cells())
    """
    handle = create_handle(nface_n_block, ngon_n_blocks)
    data_model = DataModel

    cell_centers_kernel = shape_functions_utils.get_compute_cell_centers_kernel(data_model)
    cell_face_centers_kernel = shape_functions_utils.get_compute_face_centers_kernel(data_model)

    nb_cells = DataModel.DatasetAPI.get_num_cells(handle)
    device = nface_n_block.grid_coords.device
    with dav.scoped_timer("sids.nface_n.preprocess_polyhedra"):
        wp.launch(cell_centers_kernel, dim=nb_cells, inputs=[handle, handle.nface_n_cell_centers], device=device)
        wp.launch(cell_face_centers_kernel, dim=nb_cells, inputs=[handle, handle.nface_n_block.element_start_offset, handle.nface_n_cell_face_centers], device=device)

    return dav.Dataset(DataModel, handle, device=nface_n_block.grid_coords.device, ngon_n_blocks=ngon_n_blocks)


class _ElementBlockAPI:
    """Helper API for element blocks (NFACE_n and NGON_n) to access connectivity information."""

    @staticmethod
    @dav.func
    def get_num_nodes(elem_id: wp.int32, block: UGDatasetHandle) -> wp.int32:
        assert elem_id >= block.element_range.x and elem_id <= block.element_range.y, "Element ID is out of bounds for block"

        elem_idx = elem_id - block.element_range.x
        offset_start = block.element_start_offset[elem_idx]
        offset_end = block.element_start_offset[elem_idx + 1]
        return offset_end - offset_start

    @staticmethod
    @dav.func
    def get_node_id(elem_id: wp.int32, local_idx: wp.int32, block: UGDatasetHandle) -> wp.int32:
        assert elem_id >= block.element_range.x and elem_id <= block.element_range.y, "Element ID is out of bounds for block"
        assert local_idx >= 0, "Local index must be non-negative"
        assert local_idx < _ElementBlockAPI.get_num_nodes(elem_id, block), "Local index is out of bounds for element"

        elem_idx = elem_id - block.element_range.x
        offset_start = block.element_start_offset[elem_idx]
        return block.element_connectivity[offset_start + local_idx]


class CellAPI:
    @staticmethod
    @dav.func
    def is_valid(cell: CellHandle) -> wp.bool:
        # SIDS cell ids are 1-based.
        return cell.cell_id > 0

    @staticmethod
    @dav.func
    def empty() -> CellHandle:
        return CellHandle(cell_id=0)

    @staticmethod
    @dav.func
    def get_cell_id(cell: CellHandle) -> wp.int32:
        return cell.cell_id

    @staticmethod
    @dav.func
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of points for a polyhedral cell."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"

        cell_idx = cell.cell_id - ds.nface_n_block.element_range.x
        return ds.nface_n_connectivity_offsets[cell_idx + 1] - ds.nface_n_connectivity_offsets[cell_idx]

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID for a polyhedral cell by local index."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"
        assert local_idx >= 0 and local_idx < CellAPI.get_num_points(cell, ds), "Local index is out of bounds for cell"

        cell_idx = cell.cell_id - ds.nface_n_block.element_range.x
        connectivity_start = ds.nface_n_connectivity_offsets[cell_idx]
        return ds.nface_n_connectivity[connectivity_start + local_idx]

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces for a polyhedral cell."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"

        # num_faces aka num nodes in the nface_n block.
        return _ElementBlockAPI.get_num_nodes(cell.cell_id, ds.nface_n_block)

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"
        assert face_idx >= 0 and face_idx < CellAPI.get_num_faces(cell, ds), "Face index is out of bounds for cell"

        ngon_cell_id = _ElementBlockAPI.get_node_id(cell.cell_id, face_idx, ds.nface_n_block)
        ngon_cell_id_abs = wp.abs(ngon_cell_id)
        ngon_block_idx = DatasetAPI._get_ngon_block_idx(ds, ngon_cell_id_abs)
        assert ngon_block_idx >= 0, "NGON block index is out of bounds"
        assert ngon_block_idx < ds.ngon_n_blocks.shape[0], "NGON block index is out of bounds"
        return _ElementBlockAPI.get_num_nodes(ngon_cell_id_abs, ds.ngon_n_blocks[ngon_block_idx])

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face of a polyhedral cell.

        Handles face orientation: if ngon_id is negative (inward-facing normal),
        the winding order is reversed to always return outward-facing normals.
        Reversal keeps the first vertex and reverses the rest: [0,1,2,3] -> [0,3,2,1]
        """
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"
        assert face_idx >= 0 and face_idx < CellAPI.get_num_faces(cell, ds), "Face index is out of bounds for cell"
        assert local_idx >= 0 and local_idx < CellAPI.get_face_num_points(cell, face_idx, ds), "Local index is out of bounds for face"

        ngon_cell_id = _ElementBlockAPI.get_node_id(cell.cell_id, face_idx, ds.nface_n_block)
        ngon_cell_id_abs = wp.abs(ngon_cell_id)
        ngon_block_idx = DatasetAPI._get_ngon_block_idx(ds, ngon_cell_id_abs)
        assert ngon_block_idx >= 0, "NGON block index is out of bounds"
        assert ngon_block_idx < ds.ngon_n_blocks.shape[0], "NGON block index is out of bounds"

        ngon_block = ds.ngon_n_blocks[ngon_block_idx]

        # If ngon_cell_id is negative, reverse winding order to flip the normal direction
        # Keep first point, reverse the rest: [0,1,2,3] -> [0,3,2,1]
        if wp.sign(ngon_cell_id) < 0 and local_idx > 0:
            nb_face_points = _ElementBlockAPI.get_num_nodes(ngon_cell_id_abs, ngon_block)
            return _ElementBlockAPI.get_node_id(ngon_cell_id_abs, nb_face_points - local_idx, ngon_block)
        else:
            return _ElementBlockAPI.get_node_id(ngon_cell_id_abs, local_idx, ngon_block)


class DatasetAPI:
    @staticmethod
    @dav.func
    def _get_ngon_block_idx(ds: DatasetHandle, ngon_id: wp.int32) -> wp.int32:
        assert ngon_id > 0, "ngon_id must be positive to get block index"
        idx = wp.lower_bound(ds.ngon_n_element_range_starts, ngon_id)
        if idx < 0 or (idx == 0 and ngon_id < ds.ngon_n_element_range_starts[0]):
            return -1
        elif ngon_id < ds.ngon_n_element_range_starts[idx]:
            return idx - 1
        else:
            return idx

    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
        assert idx >= 0 and idx < DatasetAPI.get_num_cells(ds), "Cell index is out of bounds"
        return ds.nface_n_block.element_range.x + idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        assert id >= ds.nface_n_block.element_range.x and id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"
        return id - ds.nface_n_block.element_range.x

    @staticmethod
    @dav.func
    def get_cell(ds: DatasetHandle, id: wp.int32) -> CellHandle:
        cell = CellHandle()
        if id >= ds.nface_n_block.element_range.x and id <= ds.nface_n_block.element_range.y:
            cell.cell_id = id
        else:
            cell.cell_id = 0  # invalid cell
        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.nface_n_block.element_range.y - ds.nface_n_block.element_range.x + 1

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.nface_n_block.grid_coords.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        assert point_idx >= 0 and point_idx < DatasetAPI.get_num_points(ds), "Point index is out of bounds"
        return point_idx + 1  # SIDS point IDs are 1-based

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        assert point_id > 0 and point_id <= DatasetAPI.get_num_points(ds), "Point ID is out of bounds"
        return point_id - 1

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, id: wp.int32) -> wp.vec3f:
        assert id > 0 and id <= DatasetAPI.get_num_points(ds), "Point ID is out of bounds"
        return ds.nface_n_block.grid_coords[id - 1]  # SIDS point IDs are 1-based

    @staticmethod
    def build_cell_locator(data_model, ds: DatasetHandle, device=None):
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'nface_n')
            ds: The dataset
            device: Device to build the locator on

        Returns:
            tuple: (status, locator) - Status code and CellLocator instance
        """
        # locators.build_cell_locator will use operators.cell_bounds to compute cell bounds per cell and build
        # a BVH using those bounds which works great for NFACE_n too!
        locator = locators.build_cell_locator(data_model, ds, device)
        assert locator is not None
        ds.cell_bvh_id = locator.get_bvh_id()
        return (True, locator)

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


class PolyhedralCellAPI:
    """API for polyhedral cell operations, such as point location and interpolation."""

    @staticmethod
    @dav.func
    def get_cell_center(cell: CellHandle, ds: DatasetHandle) -> wp.vec3f:
        """Get the precomputed cell center for a given cell."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"

        cell_idx = cell.cell_id - ds.nface_n_block.element_range.x
        return ds.nface_n_cell_centers[cell_idx]

    @staticmethod
    @dav.func
    def get_face_center(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.vec3f:
        """Get the precomputed face center for a given face of a cell."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id >= ds.nface_n_block.element_range.x and cell.cell_id <= ds.nface_n_block.element_range.y, "Cell ID is out of bounds for NFACE_n block"
        assert face_idx >= 0 and face_idx < CellAPI.get_num_faces(cell, ds), "Face index is out of bounds for cell"

        cell_idx = cell.cell_id - ds.nface_n_block.element_range.x
        oid_offset = ds.nface_n_block.element_start_offset[cell_idx]
        return ds.nface_n_cell_face_centers[oid_offset + face_idx]


# use generic cell links api
CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(0), emptyCellId=wp.int32(0), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


class PartialDataModel:
    """CGNS/SIDS NFACE_n data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = CellHandle

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    PolyhedralCellAPI = PolyhedralCellAPI


StarConvexPolyhedron = star_convex_polyhedron.get_shape(PartialDataModel)


class CellLocatorAPI:
    @staticmethod
    @dav.func
    def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return StarConvexPolyhedron.get_weights(position, cell, ds, sids_shapes.ET_NFACE_n)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        assert ds.cell_bvh_id != 0, "Cell locator BVH has not been built for the dataset. Call DatasetAPI.build_cell_locator() first."

        if CellAPI.is_valid(hint):
            if StarConvexPolyhedron.is_point_in_cell(position, hint, ds, sids_shapes.ET_NFACE_n):
                return hint

        radius = wp.vec3f(EPSILON)
        query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
        cell_idx = wp.int32(-1)
        while wp.bvh_query_next(query, cell_idx):
            cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
            cell = DatasetAPI.get_cell(ds, cell_id)
            assert CellAPI.is_valid(cell), "BVH query returned an invalid cell handle"
            if StarConvexPolyhedron.is_point_in_cell(position, cell, ds, sids_shapes.ET_NFACE_n):
                return cell

        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return StarConvexPolyhedron.is_point_in_cell(point, cell, ds, sids_shapes.ET_NFACE_n)


class DataModelMeta(type):
    """Metaclass for the NFACE_n data model to allow for future extensibility and dynamic features if needed."""

    def __repr__(self):
        return "DataModel (NFACE_n)"


class DataModel(metaclass=DataModelMeta):
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
    """Get the NFACE_n data model."""
    return DataModel
