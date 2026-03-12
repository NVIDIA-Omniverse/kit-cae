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
OpenFOAM PolyMesh Data Model
============================

This module provides a data model implementation for OpenFOAM polyMesh format.

OpenFOAM polyMesh is a general-purpose polyhedral mesh format that supports:
- Arbitrary polyhedral cells (tetrahedra, hexahedra, prisms, pyramids, polyhedra)
- Face-based connectivity (owner/neighbor cell relationships)
- Boundary patches and zones
- Efficient storage for both structured-like and unstructured meshes

PolyMesh Structure
------------------
The data model represents the core OpenFOAM polyMesh components:
- **points**: Array of 3D vertex positions
- **faces**: List of faces, each defined by vertex indices
- **owner**: Cell ID that owns each face (internal and boundary)
- **neighbour**: Cell ID on the other side of internal faces
- **boundary**: Patch definitions for boundary faces

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous)
- **Face IDs**: wp.int32 (0-based, contiguous)
- **Indices**: wp.int32 (same as IDs)

Important Implementation Notes
------------------------------
**Polyhedral Cell Vertex Iteration:**

For polyhedral cells, the ``CellAPI.get_num_points()`` and ``CellAPI.get_point_id()``
methods iterate through vertices in **face-major order with duplicates**. This means:

1. **Vertex Count Includes Duplicates**: ``get_num_points()`` returns the sum of
   vertices across all faces, where shared vertices are counted multiple times.
   For example, a hexahedron has 8 unique vertices but ``get_num_points()``
   returns 24 (6 faces × 4 vertices per face).

2. **Iteration Order**: ``get_point_id(cell, local_idx, ds)`` iterates through
   vertices in face-major order:
   - First all vertices of face 0
   - Then all vertices of face 1
   - And so on...
"""

from typing import Any

import warp as wp

import dav
from dav import locators
from dav.shape_functions import star_convex_polyhedron
from dav.shape_functions import utils as shape_functions_utils

from . import utils

EPSILON = 1e-6


@wp.struct
class DatasetHandle:
    """Dataset handle for OpenFOAM polyMesh.

    This structure will contain all necessary arrays for representing
    a polyMesh dataset.
    """

    points: wp.array(dtype=wp.vec3f)
    """Vertex positions."""

    faces: wp.array(dtype=wp.int32)
    """Flattened array of vertex indices for all faces."""

    face_offsets: wp.array(dtype=wp.int32)
    """Offsets into faces array for each face (size = num_faces + 1)."""

    owner: wp.array(dtype=wp.int32)
    """Cell ID that owns each face (size = num_faces)."""

    neighbour: wp.array(dtype=wp.int32)
    """Cell ID on other side of internal faces.

    OpenFOAM supports two formats:
    1. Short format: length = num_internal_faces (< num_faces), only internal face neighbors
    2. Full format: length = num_faces, with -1 or owner_id for boundary faces

    For boundary faces (where neighbour is -1 or equals owner), there is no neighbor cell.
    """

    # -- acceleration structures for cell queries --
    cell_faces: wp.array(dtype=wp.int32)
    """Flattened array of face indices for all cells."""

    cell_face_offsets: wp.array(dtype=wp.int32)
    """Offsets into cell_faces array for each cell (size = num_cells + 1)."""

    cell_points: wp.array(dtype=wp.int32)
    """Flattened array of unique point indices for all cells."""

    cell_points_offsets: wp.array(dtype=wp.int32)
    """Offsets into cell_points array for each cell (size = num_cells + 1)."""

    cell_centers: wp.array(dtype=wp.vec3f)
    """Precomputed cell centers for spatial queries."""

    face_centers: wp.array(dtype=wp.vec3f)
    """Precomputed face centers for spatial queries."""

    cell_bvh_id: wp.uint64
    """BVH ID for spatial acceleration."""

    cell_links: locators.CellLinks
    """Point-to-cell connectivity structure."""


def create_handle(points: wp.array, faces: wp.array, owner: wp.array, neighbour: wp.array, face_offsets: wp.array) -> DatasetHandle:
    """Create an OpenFOAM polyMesh dataset handle.

    Args:
        points: Array of 3D vertex positions (wp.vec3f)
        faces: Flattened array of vertex indices for all faces (wp.int32)
        owner: Cell ID that owns each face (wp.int32, size = num_faces)
        neighbour: Neighbor cell ID for internal faces (wp.int32).
                   Two formats supported:
                   1. Short format: size = num_internal_faces (< num_faces), only internal neighbors
                   2. Full format: size = num_faces, with -1 or owner_id for boundary faces
        face_offsets: Offsets into faces array (wp.int32, size = num_faces + 1)

    Returns:
        DatasetHandle: A new polyMesh dataset handle

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or if array sizes are inconsistent

    Note:
        - If cell_faces and cell_face_offsets are not provided, they will be computed from owner/neighbour
        - neighbour.shape[0] <= owner.shape[0]
        - All arrays will be placed on the same device as points
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
    if faces.shape[0] == 0:
        raise ValueError("faces array cannot be empty")

    # Validate owner
    if owner is None or owner.ndim != 1:
        raise ValueError("owner must be a 1D warp array")
    if owner.dtype != wp.int32:
        raise ValueError(f"owner must have dtype wp.int32, got {owner.dtype}")
    if owner.shape[0] == 0:
        raise ValueError("owner array cannot be empty")

    num_faces = owner.shape[0]
    assert num_faces == face_offsets.shape[0] - 1, "face_offsets length must be num_faces + 1"

    # Validate neighbour
    if neighbour is None or neighbour.ndim != 1:
        raise ValueError("neighbour must be a 1D warp array")
    if neighbour.dtype != wp.int32:
        raise ValueError(f"neighbour must have dtype wp.int32, got {neighbour.dtype}")
    if neighbour.shape[0] > num_faces:
        raise ValueError(f"neighbour length ({neighbour.shape[0]}) cannot exceed owner length ({num_faces}). In OpenFOAM, neighbour contains only internal faces.")

    # Validate or compute face_offsets
    if face_offsets is None:
        raise ValueError("face_offsets must be provided (cannot infer from faces array alone)")
    if face_offsets.ndim != 1:
        raise ValueError("face_offsets must be a 1D warp array")
    if face_offsets.dtype != wp.int32:
        raise ValueError(f"face_offsets must have dtype wp.int32, got {face_offsets.dtype}")
    if face_offsets.shape[0] != num_faces + 1:
        raise ValueError(f"face_offsets length ({face_offsets.shape[0]}) must be num_faces + 1 ({num_faces + 1})")

    # Compute cell-face connectivity from owner/neighbour
    cell_faces, cell_face_offsets = utils.compute_cell_face_connectivity(owner, neighbour)
    num_cells = cell_face_offsets.shape[0] - 1

    # Create handle
    handle = DatasetHandle()
    handle.points = points
    handle.faces = faces
    handle.face_offsets = face_offsets
    handle.owner = owner
    handle.neighbour = neighbour
    handle.cell_faces = cell_faces
    handle.cell_face_offsets = cell_face_offsets
    handle.cell_centers = wp.zeros(num_cells, dtype=wp.vec3f, device=points.device)  # Placeholder, should be computed
    handle.face_centers = wp.zeros(num_faces, dtype=wp.vec3f, device=points.device)  # Placeholder, should be computed
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()  # Empty cell links

    return handle


def create_dataset(points: wp.array, faces: wp.array, owner: wp.array, neighbour: wp.array, face_offsets: wp.array) -> dav.Dataset:
    """Create a polyMesh dataset handle with minimal required arrays.

    This is the recommended funciton for creating a polyMesh dataset when you only have the core arrays. It will compute the
    necessary connectivity structures for you.

    Args:
        points: Array of 3D vertex positions (wp.vec3f)
        faces: Flattened array of vertex indices for all faces (wp.int32)
        owner: Cell ID that owns each face (wp.int32, size = num_faces)
        neighbour: Neighbor cell ID for internal faces (wp.int32).

    Returns:
        dav.Dataset: A new dataset instance with the provided arrays and computed connectivity structures.

    """
    handle = create_handle(points, faces, owner, neighbour, face_offsets)
    utils.populate_cell_point_connectivity(handle)
    utils.populate_face_centers(handle)

    data_model = DataModel

    nb_cells = handle.cell_centers.shape[0]
    cell_centers_kernels = shape_functions_utils.get_compute_cell_centers_kernel(data_model)
    wp.launch(cell_centers_kernels, dim=nb_cells, inputs=[handle], outputs=[handle.cell_centers], device=handle.points.device)

    return dav.Dataset(DataModel, handle, points.device)


@wp.struct
class CellHandle:
    """Cell handle for a polyMesh cell.

    In polyMesh, cells are arbitrary polyhedra defined by their faces.
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
        """Get the number of unique points for a polyhedral cell.

        Uses the precomputed cell_points / cell_points_offsets arrays which
        contain deduplicated vertex IDs per cell.
        """
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id < ds.cell_points_offsets.shape[0] - 1, "Cell ID out of bounds"

        return ds.cell_points_offsets[cell.cell_id + 1] - ds.cell_points_offsets[cell.cell_id]

    @staticmethod
    @dav.func
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a unique vertex ID from a cell by local index.

        Uses the precomputed cell_points / cell_points_offsets arrays which
        contain deduplicated vertex IDs per cell.
        """
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id < ds.cell_points_offsets.shape[0] - 1, "Cell ID out of bounds"
        assert local_idx >= 0, "Local index must be non-negative"
        assert local_idx < CellAPI.get_num_points(cell, ds), "Local index out of bounds"

        connectivity_start = ds.cell_points_offsets[cell.cell_id]
        return ds.cell_points[connectivity_start + local_idx]

    @staticmethod
    @dav.func
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert cell.cell_id < ds.cell_face_offsets.shape[0] - 1, "Cell ID out of bounds"

        # Use cell_face_offsets to get the number of faces
        start = ds.cell_face_offsets[cell.cell_id]
        end = ds.cell_face_offsets[cell.cell_id + 1]
        return end - start

    @staticmethod
    @dav.func
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert face_idx >= 0, "Face index must be non-negative"
        assert face_idx < CellAPI.get_num_faces(cell, ds), "Face index out of bounds"

        # Get the global face ID for this cell's local face index
        cell_face_start = ds.cell_face_offsets[cell.cell_id]

        # Get the global face ID
        global_face_id = ds.cell_faces[cell_face_start + face_idx]

        # Get the number of vertices in this face using face_offsets
        face_start = ds.face_offsets[global_face_id]
        face_end = ds.face_offsets[global_face_id + 1]
        return face_end - face_start

    @staticmethod
    @dav.func
    def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get a point ID from a face.

        In OpenFOAM, faces are oriented from owner→neighbor. When the neighbor cell
        accesses the face, we flip the vertex order so the normal points outward.
        """
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        assert face_idx >= 0, "Face index must be non-negative"
        assert local_idx >= 0, "Local index must be non-negative"
        assert face_idx < CellAPI.get_num_faces(cell, ds), "Face index out of bounds"
        assert local_idx < CellAPI.get_face_num_points(cell, face_idx, ds), "Local index out of bounds"

        # Get the global face ID for this cell's local face index
        cell_face_start = ds.cell_face_offsets[cell.cell_id]

        # Get the global face ID
        global_face_id = ds.cell_faces[cell_face_start + face_idx]

        # Get the vertex from the face using face_offsets
        face_start = ds.face_offsets[global_face_id]
        face_end = ds.face_offsets[global_face_id + 1]
        face_num_points = face_end - face_start

        # Check point index bounds
        if local_idx < 0 or local_idx >= face_num_points:
            return -1

        # Check if this cell is the owner or neighbor of the face
        # Faces are oriented owner→neighbor, so neighbors see reversed vertex order
        is_owner = ds.owner[global_face_id] == cell.cell_id

        # If neighbor, reverse the vertex order (flip normal to point outward)
        if is_owner or local_idx == 0:
            # Owner: use face vertices in original order
            return ds.faces[face_start + local_idx]
        else:
            # Neighbor: reverse vertex order (last vertex first, first vertex last)
            return ds.faces[face_start + (face_num_points - local_idx)]


class DatasetAPI:
    @staticmethod
    @dav.func
    def get_cell_id_from_idx(ds: DatasetHandle, cell_idx: wp.int32) -> wp.int32:
        """Get cell ID from cell index (same for polyMesh)."""
        return cell_idx

    @staticmethod
    @dav.func
    def get_cell_idx_from_id(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        """Get cell index from cell ID (same for polyMesh)."""
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
        """Get the number of cells in the mesh.

        The number of cells is derived from the cell_face_offsets array size.
        """
        return ds.cell_face_offsets.shape[0] - 1

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        """Get the number of vertices in the mesh."""
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        """Get point ID from point index (same for polyMesh)."""
        return point_idx

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        """Get point index from point ID (same for polyMesh)."""
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
            data_model: The data model module (should be 'polymesh')
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
        # TODO: Implement cell links building from polyMesh connectivity
        cell_links = locators.build_cell_links(data_model, ds, device)
        if cell_links is not None:
            ds.cell_links = cell_links
            return (True, cell_links)
        else:
            ds.cell_links = None
            return (False, None)


class PolyhedralCellAPI:
    """API for polyhedral cell-specific operations."""

    @staticmethod
    @dav.func
    def get_cell_center(cell: CellHandle, ds: DatasetHandle) -> wp.vec3f:
        """Compute the cell center as the average of its vertices."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return ds.cell_centers[cell.cell_id]

    @staticmethod
    @dav.func
    def get_face_center(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.vec3f:
        """Compute the face center as the average of its vertices."""
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        global_face_id = ds.cell_faces[ds.cell_face_offsets[cell.cell_id] + face_idx]
        return ds.face_centers[global_face_id]


CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(-1), emptyCellId=wp.int32(-1), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


class PartialDataModel:
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
        return StarConvexPolyhedron.get_weights(position, cell, ds, 0)

    @staticmethod
    @dav.func
    def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
        assert ds.cell_bvh_id != 0, "Cell locator BVH has not been built for the dataset. Call DatasetAPI.build_cell_locator() first."

        if CellAPI.is_valid(hint):
            if StarConvexPolyhedron.is_point_in_cell(position, hint, ds, 0):
                return hint

        radius = wp.vec3f(EPSILON)
        query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
        cell_idx = wp.int32(-1)
        while wp.bvh_query_next(query, cell_idx):
            cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
            cell = DatasetAPI.get_cell(ds, cell_id)
            assert CellAPI.is_valid(cell), "BVH query returned an invalid cell handle"
            if StarConvexPolyhedron.is_point_in_cell(position, cell, ds, 0):
                return cell

        return CellAPI.empty()

    @staticmethod
    @dav.func
    def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        assert CellAPI.is_valid(cell), "Invalid cell handle"
        return StarConvexPolyhedron.is_point_in_cell(point, cell, ds, 0)


class DataModelMeta(type):
    def __repr__(cls):
        return "DataModel (OpenFOAM PolyMesh)"


# DataModel protocol implementation
class DataModel(metaclass=DataModelMeta):
    """OpenFOAM PolyMesh data model implementation."""

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
    """Factory function to get the data model for OpenFOAM polyMesh."""
    return DataModel
