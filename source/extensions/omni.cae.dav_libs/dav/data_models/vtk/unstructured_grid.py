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
VTK Unstructured Grid Data Model
==================================

This module provides a data model implementation for VTK Unstructured Grid (vtkUnstructuredGrid).

Unstructured grids represent the most general mesh type with:
- Explicit point coordinates
- Arbitrary cell types (tetrahedra, hexahedra, pyramids, wedges, etc.)
- Explicit cell-to-point connectivity stored in arrays

Type System
-----------
- **Point IDs**: wp.int32 (0-based, contiguous)
- **Cell IDs**: wp.int32 (0-based, contiguous)
- **Indices**: wp.int32 (same as IDs for unstructured grids)

Key Features
------------
- Supports multiple cell types in same mesh
- Explicit topology stored in connectivity arrays
- BVH-based locators for efficient cell location
- Explicit cell links for point-to-cell queries
"""

from typing import Any

import warp as wp

import dav
from dav import locators
from dav.shape_functions import dispatcher as shape_functions_dispatcher
from dav.shape_functions import utils as shape_functions_utils

from .. import utils as data_model_utils
from . import vtk_shapes


@wp.struct
class DatasetHandle:
    points: wp.array(dtype=wp.vec3f)
    cell_types: wp.array(dtype=wp.int32)
    cell_offsets: wp.array(dtype=wp.int32)
    cell_connectivity: wp.array(dtype=wp.int32)
    cell_bvh_id: wp.uint64
    cell_links: locators.CellLinks

    # for polyhedral cells
    faces_offsets: wp.array(dtype=wp.int32)  # Optional
    faces_connectivity: wp.array(dtype=wp.int32)  # Optional
    face_locations_offsets: wp.array(dtype=wp.int32)  # Optional
    face_locations_connectivity: wp.array(dtype=wp.int32)  # Optional

    # -- polyhedra acceleration structures
    cell_centers: wp.array(dtype=wp.vec3f)  # Optional
    face_centers: wp.array(dtype=wp.vec3f)  # Optional
    face_orientations: wp.array(dtype=wp.int8)  # Optional, +1 or -1 for outward/inward facing faces


def create_handle(
    points: wp.array,
    cell_types: wp.array,
    cell_offsets: wp.array,
    cell_connectivity: wp.array,
    faces_offsets: wp.array = None,
    faces_connectivity: wp.array = None,
    face_locations_offsets: wp.array = None,
    face_locations_connectivity: wp.array = None,
) -> DatasetHandle:
    """Create an unstructured grid dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        cell_types: Array of VTK cell type IDs for each cell (wp.int32)
        cell_offsets: Array of offsets into cell_connectivity for each cell (wp.int32).
                      Length is num_cells + 1, with last element = len(cell_connectivity)
        cell_connectivity: Array of point IDs forming cell connectivity (wp.int32)

        faces_offsets: (Optional) Array of offsets into faces_connectivity for each cell's faces (wp.int32).
                        Length is num_cells + 1, with last element = len(faces_connectivity)
        faces_connectivity: (Optional) Array of point IDs forming face connectivity (wp.int32)
        face_locations_offsets: (Optional) Array of offsets into face_locations_connectivity for each cell's faces (wp.int32)
        face_locations_connectivity: (Optional) Array of face locations (wp.int32)

    Returns:
        DatasetHandle: A new unstructured grid dataset handle

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or if array sizes are inconsistent

    Note:
        The cell_bvh_id and cell_links will be initialized to default values.
        Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links() to build
        the spatial acceleration structures after creating the dataset.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.unstructured_grid import create_handle
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> cell_types = wp.array([12, 11, ...], dtype=wp.int32)  # VTK_HEXAHEDRON, VTK_VOXEL, etc.
        >>> cell_offsets = wp.array([0, 8, 16, ...], dtype=wp.int32)
        >>> cell_connectivity = wp.array([0, 1, 2, 3, 4, 5, 6, 7, ...], dtype=wp.int32)
        >>> handle = create_handle(points, cell_types, cell_offsets, cell_connectivity)
    """
    # Validate points
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError(f"points must have dtype wp.vec3f, got {points.dtype}")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    # Validate cell_types
    if cell_types is None or cell_types.ndim != 1:
        raise ValueError("cell_types must be a 1D warp array")
    if cell_types.dtype != wp.int32:
        raise ValueError(f"cell_types must have dtype wp.int32, got {cell_types.dtype}")
    if cell_types.shape[0] == 0:
        raise ValueError("cell_types array cannot be empty")

    # Validate cell_offsets
    if cell_offsets is None or cell_offsets.ndim != 1:
        raise ValueError("cell_offsets must be a 1D warp array")
    if cell_offsets.dtype != wp.int32:
        raise ValueError(f"cell_offsets must have dtype wp.int32, got {cell_offsets.dtype}")

    num_cells = cell_types.shape[0]
    if cell_offsets.shape[0] != num_cells + 1:
        raise ValueError(f"cell_offsets length ({cell_offsets.shape[0]}) must be num_cells + 1 ({num_cells + 1})")

    # Validate cell_connectivity
    if cell_connectivity is None or cell_connectivity.ndim != 1:
        raise ValueError("cell_connectivity must be a 1D warp array")
    if cell_connectivity.dtype != wp.int32:
        raise ValueError(f"cell_connectivity must have dtype wp.int32, got {cell_connectivity.dtype}")

    # Validate offsets consistency (last offset should equal connectivity length)
    # Note: can't access array values on host without .numpy(), so skip this check for now
    # as it would require device synchronization

    handle = DatasetHandle()
    handle.points = points
    handle.cell_types = cell_types
    handle.cell_offsets = cell_offsets
    handle.cell_connectivity = cell_connectivity
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()  # Empty cell links
    handle.faces_offsets = faces_offsets
    handle.faces_connectivity = faces_connectivity
    handle.face_locations_offsets = face_locations_offsets
    handle.face_locations_connectivity = face_locations_connectivity

    if faces_offsets:
        # these will be populate later.
        device = points.device
        handle.cell_centers = wp.zeros(num_cells, dtype=wp.vec3f, device=device)
        num_faces = handle.face_locations_connectivity.shape[0]
        handle.face_centers = wp.zeros(num_faces, dtype=wp.vec3f, device=device)
        handle.face_orientations = wp.zeros(num_faces, dtype=wp.int8, device=device)

    return handle


def create_dataset(
    points: wp.array,
    cell_types: wp.array,
    cell_offsets: wp.array,
    cell_connectivity: wp.array,
    faces_offsets: wp.array = None,
    faces_connectivity: wp.array = None,
    face_locations_offsets=None,
    face_locations_connectivity: wp.array = None,
    unique_cell_types: list[int] = None,
) -> dav.Dataset:
    """Create an unstructured grid dataset.

    This is the recommended function for creating unstructured grid datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        cell_types: Array of VTK cell type IDs for each cell (wp.int32)
        cell_offsets: Array of offsets into cell_connectivity for each cell (wp.int32).
                      Length is num_cells + 1, with last element = len(cell_connectivity)
        cell_connectivity: Array of point IDs forming cell connectivity (wp.int32)
        faces_offsets: (Optional) Array of offsets into faces_connectivity for each cell's faces (wp.int32).
                        Length is num_cells + 1, with last element = len(faces_connectivity)
        faces_connectivity: (Optional) Array of point IDs forming face connectivity (wp.int32)
        face_locations_offsets: (Optional) Array of offsets into face_locations_connectivity for each cell's faces (wp.int32)
        face_locations_connectivity: (Optional) Array of face locations (wp.int32)
        unique_cell_types: Optional list of unique cell types present in the dataset. If not provided,
                      it will be computed from cell_types array. The data model may be optimized
                      to only support the specified cell types.

    Returns:
        Dataset: A new Dataset instance with unstructured grid data model

    Raises:
        ValueError: If array dimensions or dtypes are invalid, or if array sizes are inconsistent

    Example:
        >>> import warp as wp
        >>> from dav.data_models.vtk.unstructured_grid import create_dataset
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>> cell_types = wp.array([12, 11, ...], dtype=wp.int32)  # VTK_HEXAHEDRON, VTK_VOXEL, etc.
        >>> cell_offsets = wp.array([0, 8, 16, ...], dtype=wp.int32)
        >>> cell_connectivity = wp.array([0, 1, 2, 3, 4, 5, 6, 7, ...], dtype=wp.int32)
        >>> dataset = create_dataset(points, cell_types, cell_offsets, cell_connectivity)
        >>> print(dataset.get_num_cells())
    """
    if not unique_cell_types:
        unique_cell_types = data_model_utils.get_unique_values(cell_types, max_value=255).numpy().tolist()

    unsupported_types = list(set(unique_cell_types) - set(vtk_shapes.get_supported_cell_types()))
    if unsupported_types:
        raise ValueError(f"Unsupported VTK cell types found in dataset: {unsupported_types}. Supported types are: {vtk_shapes.get_supported_cell_types()}")

    device = points.device
    handle = create_handle(points, cell_types, cell_offsets, cell_connectivity, faces_offsets, faces_connectivity, face_locations_offsets, face_locations_connectivity)

    unique_cell_types = sorted(set(unique_cell_types))
    data_model = get_data_model([vtk_shapes.get_cell_type_as_string(ct) for ct in unique_cell_types])

    if vtk_shapes.VTKCellTypes.VTK_POLYHEDRON in unique_cell_types:
        with dav.scoped_timer("vtk.unstructured_grid.preprocess_polyhedra"):
            # TODO: this can be computed lazily.
            # we need to update the handle with polyhedra specific acceleration structures.
            nb_cells = handle.cell_types.shape[0]

            # compute cell centers for polyhedral cells
            cell_centers_kernel = shape_functions_utils.get_compute_cell_centers_kernel(data_model)
            wp.launch(cell_centers_kernel, dim=nb_cells, inputs=[handle], outputs=[handle.cell_centers], device=device)

            # compute face centers for polyhedral cells
            face_centers_kernel = shape_functions_utils.get_compute_face_centers_kernel(data_model)
            wp.launch(face_centers_kernel, dim=nb_cells, inputs=[handle, handle.face_locations_offsets], outputs=[handle.face_centers], device=device)

            # determine face orientations for polyhedral cells
            face_orientations_kernel = shape_functions_utils.get_compute_face_orientations_kernel(data_model)
            wp.launch(
                face_orientations_kernel,
                dim=nb_cells,
                inputs=[handle, handle.cell_centers, handle.face_locations_offsets, handle.face_centers],
                outputs=[handle.face_orientations],
                device=device,
            )

    return dav.Dataset(data_model, handle, device)


@wp.struct
class CellHandle:
    cell_id: wp.int32
    cell_type: wp.int32


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
        cell = CellHandle()
        cell.cell_id = cell_id
        cell.cell_type = ds.cell_types[cell_id]
        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.cell_types.shape[0]

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
    def build_cell_locator(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
        """Build a spatial acceleration structure for cell location queries.

        Args:
            data_model: The data model module (should be 'unstructured_grid')
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


# use generic cell links model
CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(-1), emptyCellId=wp.int32(-1), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


@dav.cached
def get_data_model(unique_cell_types: list[int | str]):
    """
    Factory function to get the data model class for VTK Unstructured Grid.
    The unique_cell_types argument can be used to optimize the shape function APIs for only the cell
    types present in the dataset.

    Args:
        unique_cell_types: List of unique VTK cell types present in the dataset. Each element
                           can be either an integer (VTK cell type) or a string (VTK cell type name) e.g.
                            [VTK_HEXAHEDRON, VTK_TETRA, ...] or ["hexahedron", "tetra", ...].

    Returns:
        DataModel: The data model class for VTK Unstructured Grid
    """

    # convert string cell types to integers if needed
    unique_cell_types = [vtk_shapes.get_cell_type_from_string(ct) if isinstance(ct, str) else ct for ct in unique_cell_types]

    ShapesLibrary = vtk_shapes.get_shapes_library(unique_cell_types)

    class DataModelMeta(type):
        def __repr__(cls):
            return f"DataModel (VTK Unstructured Grid, Cell Types: {[vtk_shapes.get_cell_type_as_string(ct) for ct in unique_cell_types]})"

    class DataModel(metaclass=DataModelMeta):
        pass

    DataModel.PointIdHandle = wp.int32
    DataModel.CellIdHandle = wp.int32
    DataModel.DatasetHandle = DatasetHandle
    DataModel.CellHandle = CellHandle
    DataModel.DatasetAPI = DatasetAPI
    DataModel.CellLinksAPI = CellLinksAPI

    # Standalone helpers defined before UGCellAPI so that methods can call each other
    # without referencing UGCellAPI mid-class-body (empty closure cell).
    @dav.func
    def _ug_cell_is_valid(cell: CellHandle) -> wp.bool:
        return cell.cell_id >= 0

    @dav.func
    def _ug_cell_get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        assert _ug_cell_is_valid(cell), "Cell is not valid"
        start_offset = ds.cell_offsets[cell.cell_id]
        end_offset = ds.cell_offsets[cell.cell_id + 1]
        return end_offset - start_offset

    @dav.func
    def _ug_cell_get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _ug_cell_is_valid(cell), "Cell is not valid"
        start_offset = ds.cell_offsets[cell.cell_id]
        return ds.cell_connectivity[start_offset + local_idx]

    @dav.func
    def _ug_cell_get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        assert _ug_cell_is_valid(cell), "Cell is not valid"
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYHEDRON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYHEDRON:
                return ds.face_locations_offsets[cell.cell_id + 1] - ds.face_locations_offsets[cell.cell_id]
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYGON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYGON:
                return 1
        return ShapesLibrary.get_num_faces(cell.cell_type)

    @dav.func
    def _ug_cell_get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _ug_cell_is_valid(cell), "Cell is not valid"
        assert face_idx >= 0 and face_idx < _ug_cell_get_num_faces(cell, ds), "Face index out of bounds"
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYGON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYGON:
                assert face_idx == 0, "Polygonal cell should only have one face with index 0"
                return _ug_cell_get_num_points(cell, ds)
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYHEDRON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYHEDRON:
                face_location = ds.face_locations_offsets[cell.cell_id] + face_idx
                face_id = ds.face_locations_connectivity[face_location]
                return ds.faces_offsets[face_id + 1] - ds.faces_offsets[face_id]
        return ShapesLibrary.get_num_face_corner_nodes(cell.cell_type, face_idx)

    @dav.func
    def _ug_cell_get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        assert _ug_cell_is_valid(cell), "Cell is not valid"
        assert face_idx >= 0 and face_idx < _ug_cell_get_num_faces(cell, ds), "Face index out of bounds"
        assert local_idx >= 0 and local_idx < _ug_cell_get_face_num_points(cell, face_idx, ds), "Local index out of bounds"
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYGON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYGON:
                assert face_idx == 0, "Polygonal cell should only have one face with index 0"
                start_offset = ds.cell_offsets[cell.cell_id]
                return ds.cell_connectivity[start_offset + local_idx]
        if wp.static(vtk_shapes.VTKCellTypes.VTK_POLYHEDRON in unique_cell_types):
            if cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYHEDRON:
                face_location = ds.face_locations_offsets[cell.cell_id] + face_idx
                face_id = ds.face_locations_connectivity[face_location]
                face_offset_start = ds.faces_offsets[face_id]
                face_offset_end = ds.faces_offsets[face_id + 1]
                if local_idx > 0 and ds.face_orientations[face_id] < -1:
                    local_idx = face_offset_end - face_offset_start - local_idx
                return ds.faces_connectivity[face_offset_start + local_idx]
        cell_local_idx = ShapesLibrary.get_face_corner_node_index(cell.cell_type, face_idx, local_idx)
        return _ug_cell_get_point_id(cell, cell_local_idx, ds)

    class UGCellAPI:
        @staticmethod
        @dav.func
        def is_valid(cell: CellHandle) -> wp.bool:
            return _ug_cell_is_valid(cell)

        @staticmethod
        @dav.func
        def empty() -> CellHandle:
            cell = CellHandle()
            cell.cell_id = -1
            cell.cell_type = vtk_shapes.VTKCellTypes.VTK_EMPTY_CELL
            return cell

        @staticmethod
        @dav.func
        def get_cell_id(cell: CellHandle) -> wp.int32:
            return cell.cell_id

        @staticmethod
        @dav.func
        def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            """Get the number of points in a cell."""
            return _ug_cell_get_num_points(cell, ds)

        @staticmethod
        @dav.func
        def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            return _ug_cell_get_point_id(cell, local_idx, ds)

        @staticmethod
        @dav.func
        def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            """Get the number of faces in a cell."""
            return _ug_cell_get_num_faces(cell, ds)

        @staticmethod
        @dav.func
        def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get the number of points in a face."""
            return _ug_cell_get_face_num_points(cell, face_idx, ds)

        @staticmethod
        @dav.func
        def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get a point ID from a face."""
            return _ug_cell_get_face_point_id(cell, face_idx, local_idx, ds)

    DataModel.CellAPI = UGCellAPI

    class UGPolyhedralCellAPI:
        @staticmethod
        @dav.func
        def get_cell_center(cell: CellHandle, ds: DatasetHandle) -> wp.vec3f:
            """Get the center of a polyhedral cell."""
            assert UGCellAPI.is_valid(cell), "Cell is not valid"
            assert cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYHEDRON, "Cell is not a polyhedron"
            return ds.cell_centers[cell.cell_id]

        @staticmethod
        @dav.func
        def get_face_center(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.vec3f:
            """Get the center of a face in a polyhedral cell."""
            assert UGCellAPI.is_valid(cell), "Cell is not valid"
            assert cell.cell_type == vtk_shapes.VTKCellTypes.VTK_POLYHEDRON, "Cell is not a polyhedron"
            assert face_idx >= 0 and face_idx < UGCellAPI.get_num_faces(cell, ds), "Face index out of bounds"
            face_location = ds.face_locations_offsets[cell.cell_id] + face_idx
            return ds.face_centers[ds.face_locations_connectivity[face_location]]

    DataModel.PolyhedralCellAPI = UGPolyhedralCellAPI

    # now data model is populated enough to build shape function dispatcher.
    ShapeDispatch = shape_functions_dispatcher.get_shape_dispatcher(DataModel, ShapesLibrary, unique_cell_types)

    class UGCellLocatorAPI:
        """Static API for cell location operations."""

        @staticmethod
        @dav.func
        def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
            assert UGCellAPI.is_valid(cell), "Invalid cell handle"
            return ShapeDispatch.get_weights(position, cell, ds, cell.cell_type)

        @staticmethod
        @dav.func
        def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
            assert ds.cell_bvh_id != 0, "Cell locator not built for dataset"

            if UGCellAPI.is_valid(hint):
                if ShapeDispatch.is_point_in_cell(position, hint, ds, hint.cell_type):
                    return hint

            radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
            query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
            cell_idx = wp.int32(-1)
            while wp.bvh_query_next(query, cell_idx):
                cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
                cell = DatasetAPI.get_cell(ds, cell_id)
                if ShapeDispatch.is_point_in_cell(position, cell, ds, cell.cell_type):
                    return cell
            return UGCellAPI.empty()

        @staticmethod
        @dav.func
        def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
            assert UGCellAPI.is_valid(cell), "Invalid cell handle"
            return ShapeDispatch.is_point_in_cell(point, cell, ds, cell.cell_type)

    DataModel.CellLocatorAPI = UGCellLocatorAPI
    return DataModel
