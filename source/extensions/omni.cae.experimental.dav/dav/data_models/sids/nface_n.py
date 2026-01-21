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

Interpolation via Tetrahedral Decomposition
--------------------------------------------
Since arbitrary polyhedra don't have standard parametric coordinates, we use
**tetrahedral decomposition** for point location and field interpolation:

1. Compute cell **centroid** (average of face vertices, with duplicates)
2. **Triangulate** each face using fan triangulation
3. Form **tetrahedra**: centroid + each face triangle
4. Test which tet contains the query point
5. Use standard **tet shape functions** for interpolation

**Characteristics:**
- C⁰ continuity (discontinuous gradients at tet boundaries)
- Simple and robust for convex polyhedra
- Works with arbitrary face counts and vertex counts
- Suitable for visualization and probing (not FEA-grade accuracy)

**Limitations:**
- Interpolation weights are for tet vertices (centroid + 3 face points)
- Does not directly interpolate from polyhedron vertices
- Field values at centroid must be computed separately if needed

Vertex Counting with Duplicates
--------------------------------
The `get_num_points()` and `get_point_id()` methods return counts/IDs that
**include duplicates** (vertices shared between faces are counted multiple times).

For example, a hexahedron has:
- 8 unique vertices
- But `get_num_points()` returns 24 (6 faces × 4 vertices per face)

This is documented and intentional - the API iterates over face vertices in
face-major order. For geometric operations (like centroid), this produces a
weighted average that is correct for regular polyhedra.

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
    i_cell = nface_n.DatasetAPI.evaluate_position(ds, position, found_cell)

    if nface_n.InterpolatedCellAPI.is_inside(i_cell):
        # Use weights for interpolation (weights for centroid + 3 triangle vertices)
        for i in range(4):
            weight = nface_n.InterpolatedCellAPI.get_weight(i_cell, i)
            # ... interpolate field data

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
from dav import locators

from .unstructured import DataModel as UnstructuredDataModel

MAX_CELL_POINTS = 32
vec32 = wp.vec(length=MAX_CELL_POINTS, dtype=wp.float32)  #  wp.types.vector(length=MAX_CELL_POINTS, dtype=wp.float32)


@wp.struct
class DatasetHandle:
    nface_n_block: UnstructuredDataModel.DatasetHandle
    ngon_n_element_range_starts: wp.array(dtype=wp.int32)  # inclusive range start for each NGON_n element block
    ngon_n_blocks: wp.array(dtype=UnstructuredDataModel.DatasetHandle)
    cell_bvh_id: wp.uint64


@wp.struct
class InterpolatedCellHandle:
    cell_id: wp.int32
    inside: wp.bool
    weights: vec32


class CellAPI:
    @staticmethod
    @wp.func
    def is_valid(cell: UnstructuredDataModel.CellHandle) -> wp.bool:
        return UnstructuredDataModel.CellAPI.is_valid(cell)

    @staticmethod
    @wp.func
    def empty() -> UnstructuredDataModel.CellHandle:
        return UnstructuredDataModel.CellAPI.empty()

    @staticmethod
    @wp.func
    def get_cell_id(cell: UnstructuredDataModel.CellHandle) -> wp.int32:
        return UnstructuredDataModel.CellAPI.get_cell_id(cell)

    @staticmethod
    @wp.func
    def get_num_points(cell: UnstructuredDataModel.CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of points for a polyhedral cell.

        NOTE: This returns the total count of points from all faces, which WILL
        include duplicate points (vertices shared between faces are counted multiple
        times). This is an iteration count, not a unique vertex count. For example,
        a hexahedron has 8 unique vertices but this will return 6 faces × 4 points = 24.
        """
        n_faces = UnstructuredDataModel.CellAPI.get_num_points(cell, ds.nface_n_block)
        n_points = wp.int32(0)
        for i in range(n_faces):
            ngon_id = UnstructuredDataModel.CellAPI.get_point_id(cell, i, ds.nface_n_block)
            ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
            if UnstructuredDataModel.CellAPI.is_valid(ngon_cell):
                n_points += UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ds.ngon_n_blocks[ngon_block_idx])
        return n_points

    @staticmethod
    @wp.func
    def get_point_id(cell: UnstructuredDataModel.CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID for a polyhedral cell by local index.

        NOTE: This iterates through points in face-major order and WILL return
        duplicate point IDs since vertices shared between faces are counted multiple
        times. The local_idx range is [0, get_num_points()-1] which includes duplicates.
        """
        n_faces = UnstructuredDataModel.CellAPI.get_num_points(cell, ds.nface_n_block)
        idx = wp.int32(0)
        for i in range(n_faces):
            ngon_id = UnstructuredDataModel.CellAPI.get_point_id(cell, i, ds.nface_n_block)

            ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
            ngon_block = ds.ngon_n_blocks[ngon_block_idx]
            nb_ngon_points = UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ngon_block)
            if idx + nb_ngon_points > local_idx:
                return UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, local_idx - idx, ngon_block)
            idx += nb_ngon_points

        wp.printf(
            "WARNING: get_point_id: local_idx is out of bounds for cell {}\n",
            UnstructuredDataModel.CellAPI.get_cell_id(cell),
        )
        return -1

    @staticmethod
    @wp.func
    def get_num_faces(cell: UnstructuredDataModel.CellHandle, ds: DatasetHandle) -> wp.int32:
        # i.e. number of ngons
        return UnstructuredDataModel.CellAPI.get_num_points(cell, ds.nface_n_block)

    @staticmethod
    @wp.func
    def get_face_num_points(cell: UnstructuredDataModel.CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        ngon_id = UnstructuredDataModel.CellAPI.get_point_id(cell, face_idx, ds.nface_n_block)
        ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
        return UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ds.ngon_n_blocks[ngon_block_idx])

    @staticmethod
    @wp.func
    def get_face_point_id(
        cell: UnstructuredDataModel.CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle
    ) -> wp.int32:
        """Get a point ID from a face of a polyhedral cell.

        Handles face orientation: if ngon_id is negative (inward-facing normal),
        the winding order is reversed to always return outward-facing normals.
        Reversal keeps the first vertex and reverses the rest: [0,1,2,3] -> [0,3,2,1]
        """
        ngon_id = UnstructuredDataModel.CellAPI.get_point_id(cell, face_idx, ds.nface_n_block)
        sign_ngon_id = wp.sign(ngon_id)

        ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
        ngon_block = ds.ngon_n_blocks[ngon_block_idx]
        ngon_nb_points = UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ngon_block)

        if sign_ngon_id < 0:
            # Reverse winding order to flip the normal direction
            # Keep first point, reverse the rest: [0,1,2,3] -> [0,3,2,1]
            if local_idx == 0:
                return UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, 0, ngon_block)
            else:
                # Reverse: point at index i goes to index (n - i)
                reversed_idx = ngon_nb_points - local_idx
                return UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, reversed_idx, ngon_block)
        else:
            return UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, local_idx, ngon_block)


class DatasetAPI:
    @staticmethod
    @wp.func
    def _get_ngon_block_idx(ds: DatasetHandle, ngon_id: wp.int32) -> wp.int32:
        idx = wp.lower_bound(ds.ngon_n_element_range_starts, ngon_id)
        if idx < 0 or (idx == 0 and ngon_id < ds.ngon_n_element_range_starts[0]):
            return -1
        elif ngon_id < ds.ngon_n_element_range_starts[idx]:
            return idx - 1
        else:
            return idx

    @staticmethod
    @wp.func
    def _get_ngon_cell(ds: DatasetHandle, ngon_id: wp.int32) -> tuple[UnstructuredDataModel.CellHandle, wp.int32]:
        """helper to locate the ngon cell to avoid repeated loolups"""
        # ngon_id can be negative to indicate the orientation of the ngon (inward [-ve] or outward [+ve] facing normal)
        abs_ngon_id = wp.abs(ngon_id)
        assert abs_ngon_id > 0, "abs_ngon_id is out of bounds"
        ngon_block_idx = DatasetAPI._get_ngon_block_idx(ds, abs_ngon_id)

        assert ngon_block_idx >= 0, "ngon_block_idx is out of bounds"
        assert ngon_block_idx < ds.ngon_n_blocks.shape[0], "ngon_block_idx is out of bounds"

        return UnstructuredDataModel.DatasetAPI.get_cell(ds.ngon_n_blocks[ngon_block_idx], abs_ngon_id), ngon_block_idx

    @staticmethod
    @wp.func
    def _get_ngon_normal(
        ngon_block: UnstructuredDataModel.DatasetHandle, ngon_cell: UnstructuredDataModel.CellHandle
    ) -> wp.vec3f:
        nb_ngon_pts = UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ngon_block)
        if nb_ngon_pts < 3:
            return wp.vec3f(0.0, 0.0, 0.0)

        p0_id = UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, 0, ngon_block)
        p1_id = UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, 1, ngon_block)
        p2_id = UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, 2, ngon_block)

        p0 = UnstructuredDataModel.DatasetAPI.get_point(ngon_block, p0_id)
        p1 = UnstructuredDataModel.DatasetAPI.get_point(ngon_block, p1_id)
        p2 = UnstructuredDataModel.DatasetAPI.get_point(ngon_block, p2_id)

        v1 = p1 - p0
        v2 = p2 - p0
        return wp.cross(v1, v2)

    @staticmethod
    @wp.func
    def _is_outside_ngon(
        ngon_block: UnstructuredDataModel.DatasetHandle,
        ngon_cell: UnstructuredDataModel.CellHandle,
        position: wp.vec3f,
        sign_ngon_id: wp.int32,
    ) -> wp.bool:
        normal = DatasetAPI._get_ngon_normal(ngon_block, ngon_cell) * wp.float32(sign_ngon_id)
        p0_id = UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, 0, ngon_block)
        p0 = UnstructuredDataModel.DatasetAPI.get_point(ngon_block, p0_id)
        to_position = position - p0
        dot_pos = wp.dot(normal, to_position)
        return dot_pos > -1e-6

    @staticmethod
    @wp.func
    def _is_outsize_nface(
        ds: DatasetHandle, nface_cell: UnstructuredDataModel.CellHandle, position: wp.vec3f
    ) -> wp.bool:
        nb_ngons = UnstructuredDataModel.CellAPI.get_num_points(nface_cell, ds.nface_n_block)
        for ngon_idx in range(nb_ngons):
            ngon_id = UnstructuredDataModel.CellAPI.get_point_id(nface_cell, ngon_idx, ds.nface_n_block)
            ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
            ngon_block = ds.ngon_n_blocks[ngon_block_idx]
            if DatasetAPI._is_outside_ngon(ngon_block, ngon_cell, position, wp.sign(ngon_id)):
                return True
        return False

    @staticmethod
    @wp.func
    def get_cell_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_cell_id_from_idx(ds.nface_n_block, idx)

    @staticmethod
    @wp.func
    def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_cell_idx_from_id(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def get_cell(ds: DatasetHandle, id: wp.int32) -> UnstructuredDataModel.CellHandle:
        return UnstructuredDataModel.DatasetAPI.get_cell(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_num_cells(ds.nface_n_block)

    @staticmethod
    @wp.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_num_points(ds.nface_n_block)

    @staticmethod
    @wp.func
    def get_point_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_point_id_from_idx(ds.nface_n_block, idx)

    @staticmethod
    @wp.func
    def get_point_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_point_idx_from_id(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def get_point(ds: DatasetHandle, id: wp.int32) -> wp.vec3f:
        return UnstructuredDataModel.DatasetAPI.get_point(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def get_cell_link(ds: DatasetHandle, id: wp.int32) -> UnstructuredDataModel.CellLinkHandle:
        return UnstructuredDataModel.DatasetAPI.get_cell_link(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def get_field_id_from_idx(ds: DatasetHandle, local_idx: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_field_id_from_idx(ds.nface_n_block, local_idx)

    @staticmethod
    @wp.func
    def get_field_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
        return UnstructuredDataModel.DatasetAPI.get_field_idx_from_id(ds.nface_n_block, id)

    @staticmethod
    @wp.func
    def evaluate_position(
        ds: DatasetHandle, position: wp.vec3f, cell: UnstructuredDataModel.CellHandle
    ) -> InterpolatedCellHandle:
        """Evaluate position for a polyhedral cell using inverse distance weighting.

        Uses all face vertices (including duplicates) to compute shape function weights
        via inverse distance weighting. Also checks insideness by testing against face planes.

        Method:
        1. For each face, compute normal and test if position is on correct side
        2. For each vertex, compute inverse distance weight
        3. Normalize weights

        Note: Vertices shared by multiple faces appear multiple times in the iteration,
        which naturally gives them higher weight proportional to their valence.
        """
        i_cell = InterpolatedCellHandle(cell_id=cell.cell_id, inside=False, weights=vec32(0.0))

        if not UnstructuredDataModel.CellAPI.is_valid(cell):
            i_cell.cell_id = 0
            return i_cell

        # Single pass: compute weights and test insideness
        nb_ngons = UnstructuredDataModel.CellAPI.get_num_points(cell, ds.nface_n_block)
        total_weight = wp.float32(0.0)
        weights = vec32(0.0)
        weights_offset = wp.int32(0)
        for ngon_idx in range(nb_ngons):
            ngon_id = UnstructuredDataModel.CellAPI.get_point_id(cell, ngon_idx, ds.nface_n_block)
            ngon_cell, ngon_block_idx = DatasetAPI._get_ngon_cell(ds, ngon_id)
            ngon_block = ds.ngon_n_blocks[ngon_block_idx]
            nb_ngon_pts = UnstructuredDataModel.CellAPI.get_num_points(ngon_cell, ngon_block)

            assert nb_ngon_pts >= 3, "nb_ngon_pts is < 3 for ngon!"
            assert weights_offset + nb_ngon_pts <= MAX_CELL_POINTS, "weights_offset + nb_ngon_pts is out of bounds"

            # compute weights for the points on the face
            for pt_idx in range(nb_ngon_pts):
                pt_id = UnstructuredDataModel.CellAPI.get_point_id(ngon_cell, pt_idx, ngon_block)
                pt = UnstructuredDataModel.DatasetAPI.get_point(ngon_block, pt_id)
                dist = wp.length(position - pt)
                if dist < 1e-7:  # Position coincides with a vertex
                    i_cell.weights[weights_offset + pt_idx] = 1.0
                    i_cell.inside = True
                    return i_cell
                else:
                    weight = 1.0 / dist
                    weights[weights_offset + pt_idx] = weight
                    total_weight += weight
            weights_offset += nb_ngon_pts

            # detect if position is outside the cell based on the normal of the face
            if DatasetAPI._is_outside_ngon(ngon_block, ngon_cell, position, wp.sign(ngon_id)):
                total_weight = 0.0
                break

        # Normalize weights
        if total_weight > 0.0:
            i_cell.weights = weights / total_weight
            i_cell.inside = True
        return i_cell

    @staticmethod
    @wp.func
    def find_cell_containing_point(
        ds: DatasetHandle, position: wp.vec3f, hint: UnstructuredDataModel.CellHandle
    ) -> UnstructuredDataModel.CellHandle:
        if ds.cell_bvh_id == 0:
            wp.printf("ERROR: Cell locator not built for dataset\n")
            return UnstructuredDataModel.CellAPI.empty()

        if UnstructuredDataModel.CellAPI.is_valid(hint):
            if not DatasetAPI._is_outsize_nface(ds, hint, position):
                return hint

        radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
        query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
        cell_idx = wp.int32(-1)
        while wp.bvh_query_next(query, cell_idx):
            cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
            cell = DatasetAPI.get_cell(ds, cell_id)
            if not DatasetAPI._is_outsize_nface(ds, cell, position):
                return cell
        return UnstructuredDataModel.CellAPI.empty()

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
        raise NotImplementedError("Cell links are not supported for NFACE_n data model")


class InterpolatedCellAPI:
    """Operations on interpolated cells for polyhedral elements."""

    @staticmethod
    @wp.func
    def empty() -> InterpolatedCellHandle:
        return InterpolatedCellHandle(cell_id=0, inside=False, weights=vec32(0.0))  # type: ignore

    @staticmethod
    @wp.func
    def is_valid(i_cell: InterpolatedCellHandle) -> wp.bool:
        return i_cell.cell_id > 0

    @staticmethod
    @wp.func
    def get_cell_id(i_cell: InterpolatedCellHandle) -> wp.int32:
        return i_cell.cell_id

    @staticmethod
    @wp.func
    def is_inside(i_cell: InterpolatedCellHandle) -> wp.bool:
        return i_cell.inside

    @staticmethod
    @wp.func
    def get_weight(i_cell: InterpolatedCellHandle, local_idx: wp.int32) -> wp.float32:
        assert local_idx >= 0 and local_idx < MAX_CELL_POINTS, "local_idx is out of bounds"
        return i_cell.weights[local_idx]


class CellLinksAPI:
    """Operations on cell links for polyhedral elements.

    Delegates to UnstructuredDataModel.CellLinksAPI since we use
    the same CellLinkHandle type.
    """

    @staticmethod
    @wp.func
    def empty() -> UnstructuredDataModel.CellLinkHandle:
        return UnstructuredDataModel.CellLinksAPI.empty()

    @staticmethod
    @wp.func
    def is_valid(cell_link: UnstructuredDataModel.CellLinkHandle) -> wp.bool:
        return UnstructuredDataModel.CellLinksAPI.is_valid(cell_link)

    @staticmethod
    @wp.func
    def get_point_id(cell_link: UnstructuredDataModel.CellLinkHandle) -> wp.int32:
        return UnstructuredDataModel.CellLinksAPI.get_point_id(cell_link)

    @staticmethod
    @wp.func
    def get_num_cells(cell_link: UnstructuredDataModel.CellLinkHandle, ds: DatasetHandle) -> wp.int32:
        return UnstructuredDataModel.CellLinksAPI.get_num_cells(cell_link, ds.nface_n_block)

    @staticmethod
    @wp.func
    def get_cell_id(cell_link: UnstructuredDataModel.CellLinkHandle, cell_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        return UnstructuredDataModel.CellLinksAPI.get_cell_id(cell_link, cell_idx, ds.nface_n_block)


class DataModel:
    """CGNS/SIDS NFACE_n data model implementation."""

    # Handle types
    DatasetHandle = DatasetHandle
    CellHandle = UnstructuredDataModel.CellHandle
    InterpolatedCellHandle = InterpolatedCellHandle
    CellLinkHandle = UnstructuredDataModel.CellLinkHandle
    PointIdHandle = UnstructuredDataModel.PointIdHandle
    CellIdHandle = UnstructuredDataModel.CellIdHandle

    # API types
    DatasetAPI = DatasetAPI
    CellAPI = CellAPI
    InterpolatedCellAPI = InterpolatedCellAPI
    CellLinksAPI = CellLinksAPI
