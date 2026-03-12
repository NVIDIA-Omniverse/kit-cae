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
NOTE:
For EnSight nfaced_n cells, the face normal can be inward or outward facing
depending on the order of the face connectivity. This means that we need
to handle that explicitly to ensure that the shape functions are computed correctly
"""

from typing import Any

import numpy as np
import warp as wp

import dav
from dav import locators
from dav.shape_functions import dispatcher as shape_functions_dispatcher
from dav.shape_functions import utils as shape_functions_utils

from . import ensight_shapes
from . import utils as ensight_utils
from .common import DatasetHandle, Piece

EPSILON = 1e-6


def create_piece_handle(
    element_type: wp.int32,
    connectivity: wp.array(dtype=wp.int32),
    element_node_counts: wp.array(dtype=wp.int32) = None,
    element_face_counts: wp.array(dtype=wp.int32) = None,
    face_node_counts: wp.array(dtype=wp.int32) = None,
) -> Piece:
    """Create an EnSight Gold unstructured part piece.

    Args:
        element_type: Element type for the piece.
        connectivity: Connectivity for the piece.
        element_node_counts: Counts of nodes for each element (for nsided elements).
        element_face_counts: Counts of faces for each element (for nfaced elements).
        face_node_counts: Counts of nodes for each face (for nfaced elements).
    """
    if connectivity is None or connectivity.ndim != 1:
        raise ValueError("connectivity must be a 1D warp array")
    if connectivity.dtype != wp.int32:
        raise ValueError("connectivity must have dtype wp.int32")

    device = connectivity.device

    num_elements = 0  # we will deduce below

    if element_type == ensight_shapes.EN_nsided:
        if element_node_counts is None:
            raise ValueError("element_node_counts must be provided for nsided elements")
        if element_face_counts is not None:
            raise ValueError("element_face_counts must not be provided for nsided elements")
        if face_node_counts is not None:
            raise ValueError("face_node_counts must not be provided for nsided elements")
        if element_node_counts.device != device:
            raise ValueError("element_node_counts must be on the same device as connectivity")

        # for nsided elements, the number of elements is just the length of the element_node_counts array
        num_elements = element_node_counts.shape[0]

    elif element_type == ensight_shapes.EN_nfaced:
        if element_node_counts is not None:
            raise ValueError("element_node_counts must not be provided for nfaced elements")
        if element_face_counts is None:
            raise ValueError("element_face_counts must be provided for nfaced elements")
        if element_face_counts.device != device:
            raise ValueError("element_face_counts must be on the same device as connectivity")
        if face_node_counts.device != device:
            raise ValueError("face_node_counts must be on the same device as connectivity")

        # for nfaced elements, the number of elements is just the length of the element_face_counts array
        num_elements = element_face_counts.shape[0]
    else:
        if element_node_counts is not None:
            raise ValueError("element_node_counts must not be provided for uniform elements")
        if element_face_counts is not None:
            raise ValueError("element_face_counts must not be provided for uniform elements")
        if face_node_counts is not None:
            raise ValueError("face_node_counts must not be provided for uniform elements")

        ShapesLibrary = ensight_shapes.get_shapes_library([element_type])
        nb_nodes_per_element = ShapesLibrary.get_num_all_nodes(element_type)
        assert connectivity.shape[0] % nb_nodes_per_element == 0, "Connectivity length must be a multiple of the number of nodes per element for uniform elements"
        num_elements = connectivity.shape[0] // nb_nodes_per_element

    if num_elements < 0:
        # TODO: this is really not an error, but we can handle this later.
        raise ValueError("num_elements must be >= 0")

    piece = Piece()
    piece.element_type = element_type
    piece.num_elements = num_elements
    piece.connectivity = connectivity

    # each counts array will be converted to offsets since counts are provided by EnSight
    # but offsets are more convenient for us.
    if element_node_counts is not None:
        piece.element_node_offsets = wp.zeros(shape=element_node_counts.shape[0] + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(element_node_counts, piece.element_node_offsets, inclusive=False, add_trailing_sum=True)
    if element_face_counts is not None:
        piece.element_face_offsets = wp.zeros(shape=element_face_counts.shape[0] + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(element_face_counts, piece.element_face_offsets, inclusive=False, add_trailing_sum=True)
    if face_node_counts is not None:
        piece.face_node_offsets = wp.zeros(shape=face_node_counts.shape[0] + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(face_node_counts, piece.face_node_offsets, inclusive=False, add_trailing_sum=True)

    if element_type == ensight_shapes.EN_nfaced:
        # compute nfaced_connectivity and element_node_offsets for unique node lookup
        # for polyhedra.
        ensight_utils.populate_nfaced_connectivity(piece)

        # allocate arrays for other acceleration structures.
        piece.element_centers = wp.zeros(shape=(num_elements,), dtype=wp.vec3f, device=device)
        num_faces = piece.element_face_offsets[-1:].numpy().item()  # total number of faces
        piece.element_face_centers = wp.zeros(shape=(num_faces,), dtype=wp.vec3f, device=device)
        piece.element_face_signs = wp.zeros(shape=(num_faces,), dtype=wp.int8, device=device)

    return piece


def create_handle(points: wp.array, pieces: list[Piece]) -> DatasetHandle:
    """Create an EnSight Gold unstructured part dataset handle.

    .. note::
        This function is for advanced use. Most users should use :func:`create_dataset` instead,
        which creates a complete Dataset object ready for use with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        pieces: List of Piece objects containing element data

    Returns:
        DatasetHandle: A new EnSight Gold unstructured part dataset handle

    Raises:
        ValueError: If arrays have invalid dimensions or dtypes, or if pieces are on different devices

    Note:
        The cell_bvh_id and cell_links will be initialized to default values.
        Use DatasetAPI.build_cell_locator() and DatasetAPI.build_cell_links() to build
        the spatial acceleration structures after creating the dataset.
    """

    # Validate points
    if points is None or points.ndim != 1:
        raise ValueError("points must be a 1D warp array")
    if points.dtype != wp.vec3f:
        raise ValueError("points must have dtype wp.vec3f")
    if points.shape[0] == 0:
        raise ValueError("points array cannot be empty")

    # Validate pieces
    if pieces is None or len(pieces) == 0:
        raise ValueError("pieces must be a non-empty list of Piece objects")

    device = points.device

    # Validate that all pieces are on the same device as points
    for i, piece in enumerate(pieces):
        if piece.connectivity.device != device:
            raise ValueError(f"Piece {i} connectivity must be on the same device as points")

    # Build piece_offsets array from element counts
    np_counts = np.array([piece.num_elements for piece in pieces], dtype=np.int32)
    np_offsets = np.concatenate([[0], np.cumsum(np_counts)])
    assert np_offsets.shape[0] == len(pieces) + 1, "piece_offsets must have length num_pieces + 1"

    # Create the dataset handle
    handle = DatasetHandle()
    handle.points = points
    handle.pieces = wp.array(pieces, dtype=Piece, device=device)
    handle.piece_offsets = wp.array(np_offsets, dtype=wp.int32, device=device)
    handle.num_elements = int(np_offsets[-1])
    handle.cell_bvh_id = wp.uint64(0)
    handle.cell_links = locators.CellLinks()
    return handle


def create_dataset(points: wp.array, pieces: list[Piece]):
    """Create an EnSight Gold unstructured part dataset.

    This is the recommended function for creating EnSight Gold unstructured datasets. It creates both the
    dataset handle and wraps it in a Dataset object that can be used with operators and fields.

    Args:
        points: Array of 3D point coordinates (wp.vec3f)
        pieces: List of Piece objects containing element data

    Returns:
        Dataset: A new Dataset instance with EnSight Gold unstructured part data model

    Raises:
        ValueError: If arrays have invalid dimensions or dtypes, or if pieces are on different devices

    Example:
        >>> import warp as wp
        >>> from dav.data_models.ensight_gold.unstructured_part import create_piece_handle, create_dataset
        >>> from dav.data_models.ensight_gold import ensight_shapes
        >>>
        >>> # Create points
        >>> points = wp.array([[0, 0, 0], [1, 0, 0], ...], dtype=wp.vec3f)
        >>>
        >>> # Create a piece with hexa8 elements
        >>> connectivity = wp.array([0, 1, 2, 3, 4, 5, 6, 7, ...], dtype=wp.int32)
        >>> piece1 = create_piece_handle(ensight_shapes.EN_hexa8, 10, connectivity)
        >>>
        >>> # Create another piece with tetra4 elements
        >>> connectivity2 = wp.array([0, 1, 2, 3, ...], dtype=wp.int32)
        >>> piece2 = create_piece_handle(ensight_shapes.EN_tetra4, 5, connectivity2)
        >>>
        >>> # Create the dataset
        >>> dataset = create_dataset(points, [piece1, piece2])
        >>> print(dataset.get_num_cells())
        15
    """

    device = points.device
    handle = create_handle(points, pieces)
    element_types_in_pieces = list({int(piece.element_type) for piece in pieces})

    if ensight_shapes.EN_nfaced in element_types_in_pieces:
        # create a temporary data model for just nfaced elems.
        nfaced_only_data_model = get_data_model([ensight_shapes.get_element_type_as_string(ensight_shapes.EN_nfaced)])

        cell_centers_kernel = shape_functions_utils.get_compute_cell_centers_kernel(nfaced_only_data_model)
        face_centers_kernel = shape_functions_utils.get_compute_face_centers_kernel(nfaced_only_data_model)
        face_orientations_kernel = shape_functions_utils.get_compute_face_orientations_kernel(nfaced_only_data_model)

        # build acceleration structures to support CellLocatorAPI
        for piece in filter(lambda p: p.element_type == ensight_shapes.EN_nfaced, pieces):
            # create a temporary dataset handle for this piece to compute the acceleration structures
            piece_ds_handle = create_handle(points, [piece])

            nb_cells = piece.num_elements
            wp.launch(cell_centers_kernel, dim=nb_cells, inputs=[piece_ds_handle], outputs=[piece.element_centers], device=device)

            wp.launch(face_centers_kernel, dim=nb_cells, inputs=[piece_ds_handle, piece.element_face_offsets], outputs=[piece.element_face_centers], device=device)

            wp.launch(
                face_orientations_kernel,
                dim=nb_cells,
                inputs=[piece_ds_handle, piece.element_centers, piece.element_face_offsets, piece.element_face_centers],
                outputs=[piece.element_face_signs],
                device=device,
            )

    element_types_in_pieces = sorted(set(element_types_in_pieces))
    element_types_in_pieces = [ensight_shapes.get_element_type_as_string(et) for et in element_types_in_pieces]
    data_model = get_data_model(element_types_in_pieces)  # get the data model for the element types in the pieces
    return dav.Dataset(data_model, handle, device=points.device, pieces=pieces)


PointIdHandle = wp.int32  # point id is just an integer index
CellIdHandle = wp.int32  # cell id is just an integer index


@wp.struct
class CellHandle:
    cell_id: wp.int32

    # additional info to avoid repeated lookups;
    # all info is redundant.
    piece_idx: wp.int32
    piece_element_offset: wp.int32
    element_type: wp.int32


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
        if cell_id >= 0 and cell_id < ds.num_elements:
            piece_idx = EnSightAPI.get_piece_idx(ds, cell_id)
            assert piece_idx >= 0, "Cell ID out of range"
            assert piece_idx < ds.pieces.shape[0], "Cell ID out of range"
            cell.cell_id = cell_id
            cell.piece_idx = piece_idx
            cell.piece_element_offset = cell_id - ds.piece_offsets[piece_idx]
            cell.element_type = ds.pieces[piece_idx].element_type
        else:
            cell.cell_id = -1  # invalid cell id
        return cell

    @staticmethod
    @dav.func
    def get_num_cells(ds: DatasetHandle) -> wp.int32:
        return ds.num_elements

    @staticmethod
    @dav.func
    def get_num_points(ds: DatasetHandle) -> wp.int32:
        return ds.points.shape[0]

    @staticmethod
    @dav.func
    def get_point_id_from_idx(ds: DatasetHandle, point_idx: wp.int32) -> wp.int32:
        return point_idx + 1

    @staticmethod
    @dav.func
    def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
        return point_id - 1

    @staticmethod
    @dav.func
    def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
        return ds.points[DatasetAPI.get_point_idx_from_id(ds, point_id)]

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


class EnSightAPI:
    """Internal API"""

    @staticmethod
    @dav.func
    def get_piece_idx(ds: DatasetHandle, cell_id: wp.int32) -> wp.int32:
        piece_idx = wp.lower_bound(ds.piece_offsets, cell_id)
        if piece_idx >= 0:
            if cell_id < ds.piece_offsets[piece_idx]:
                piece_idx -= 1
        return piece_idx


# use generic cell links model
CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(-1), emptyCellId=wp.int32(-1), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)


@dav.cached
def get_data_model(dm_element_types: list[int | str]):
    """Factory function to get the data model for a given set of element types.

    Args:
        dm_element_types: List of element types that will be used in the dataset. This is used to determine
                         which acceleration structures to build. Each element can be either an integer (element type ID)
                         or a string (element type name), for example [EN_hexa8, EN_tetra4] or ["hexa8", "tetra4"].

    """

    # convert string element types to integers if needed
    dm_element_types = [ensight_shapes.get_element_type_from_string(et) if isinstance(et, str) else et for et in dm_element_types]

    ShapesLibrary = ensight_shapes.get_shapes_library(dm_element_types)

    DATA_MODEL_ELEMENT_TYPES = wp.static(dm_element_types)

    class DataModelMeta(type):
        def __repr__(cls):
            element_type_names = [ensight_shapes.get_element_type_as_string(et) for et in DATA_MODEL_ELEMENT_TYPES]
            return f"DataModel (EnSight Gold Unstructured Part, element types: {element_type_names})"

    class DataModel(metaclass=DataModelMeta):
        pass

    DataModel.CellIdHandle = CellIdHandle
    DataModel.PointIdHandle = PointIdHandle
    DataModel.DatasetHandle = DatasetHandle
    DataModel.CellHandle = CellHandle
    DataModel.DatasetAPI = DatasetAPI
    DataModel.CellLinksAPI = CellLinksAPI

    class EnSightCellAPI:
        @staticmethod
        @dav.func
        def is_valid(cell: CellHandle) -> wp.bool:
            return cell.cell_id >= 0

        @staticmethod
        @dav.func
        def empty() -> CellHandle:
            cell = CellHandle()
            cell.cell_id = -1
            return cell

        @staticmethod
        @dav.func
        def get_cell_id(cell: CellHandle) -> wp.int32:
            return cell.cell_id

        @staticmethod
        @dav.func
        def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            assert cell.cell_id >= 0, "Invalid cell handle"

            piece_idx = cell.piece_idx
            assert piece_idx >= 0, "Cell ID out of range"
            assert piece_idx < ds.pieces.shape[0], "Cell ID out of range"

            element_type = cell.element_type

            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES or ensight_shapes.EN_nfaced in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided or element_type == ensight_shapes.EN_nfaced:
                    # nsided cell i.e. polygons
                    # nfaced cell i.e. polyhedra (we're looking up the acceleration structure we computed for unique node connectivity)
                    piece = ds.pieces[piece_idx]
                    element_offset = cell.piece_element_offset
                    offset_start = piece.element_node_offsets[element_offset]
                    offset_end = piece.element_node_offsets[element_offset + 1]
                    assert offset_start >= 0 and offset_end >= offset_start, "Invalid element node offsets"
                    assert offset_end - offset_start <= dav.config.max_points_per_cell, "Element has too many points"
                    return offset_end - offset_start

            return ShapesLibrary.get_num_corner_nodes(element_type)

        @staticmethod
        @dav.func
        def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get the point IDs for a cell.

            Returns:
                wp.int32: The number of points returned
            """
            assert cell.cell_id >= 0, "Invalid cell handle"

            piece = ds.pieces[cell.piece_idx]
            element_offset = cell.piece_element_offset
            element_type = cell.element_type

            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided:
                    # nsided cell i.e. polygons
                    # nfaced cell i.e. polyhedra (we're looking up the acceleration structure we computed for unique node connectivity)
                    offset_start = piece.element_node_offsets[element_offset]
                    offset_end = piece.element_node_offsets[element_offset + 1]
                    assert offset_start >= 0 and offset_end >= offset_start, "Invalid element node offsets"
                    assert offset_end - offset_start <= dav.config.max_points_per_cell, "Element has too many points"
                    assert local_idx >= 0 and local_idx < offset_end - offset_start, "Local index out of range"
                    return piece.connectivity[offset_start + local_idx]

            if wp.static(ensight_shapes.EN_nfaced in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nfaced:
                    # nfaced cell i.e. polyhedra (we're looking up the acceleration structure we computed for unique node connectivity)
                    offset_start = piece.element_node_offsets[element_offset]
                    offset_end = piece.element_node_offsets[element_offset + 1]
                    assert offset_start >= 0 and offset_end >= offset_start, "Invalid element node offsets"
                    assert offset_end - offset_start <= dav.config.max_points_per_cell, "Element has too many points"
                    assert local_idx >= 0 and local_idx < offset_end - offset_start, "Local index out of range"
                    return piece.nfaced_connectivity[offset_start + local_idx]

            # uniform cell (including non-linear cells)
            nb_pts_per_element = ShapesLibrary.get_num_all_nodes(piece.element_type)
            assert local_idx >= 0 and local_idx < nb_pts_per_element, "Local index out of range"
            return piece.connectivity[element_offset * nb_pts_per_element + local_idx]

        @staticmethod
        @dav.func
        def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            assert cell.cell_id >= 0, "Invalid cell handle"

            piece = ds.pieces[cell.piece_idx]
            element_offset = cell.piece_element_offset
            element_type = cell.element_type

            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided:
                    # nsided cell i.e. polygons
                    return 1  # nsided cells have exactly one face

            if wp.static(ensight_shapes.EN_nfaced in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nfaced:
                    # nfaced cell i.e. polyhedra
                    face_offset_start = piece.element_face_offsets[element_offset]
                    face_offset_end = piece.element_face_offsets[element_offset + 1]
                    assert face_offset_start >= 0 and face_offset_end >= face_offset_start, "Invalid element face offsets"
                    return face_offset_end - face_offset_start

            # uniform cell (including non-linear cells)
            return ShapesLibrary.get_num_faces(element_type)

        @staticmethod
        @dav.func
        def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            assert cell.cell_id >= 0, "Invalid cell handle"
            assert face_idx >= 0, "Face index out of range"

            piece = ds.pieces[cell.piece_idx]
            element_offset = cell.piece_element_offset
            element_type = cell.element_type

            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided:
                    # nsided cell i.e. polygons; all points are on the single face
                    return piece.element_node_offsets[element_offset + 1] - piece.element_node_offsets[element_offset]

            if wp.static(ensight_shapes.EN_nfaced in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nfaced:
                    # nfaced cell i.e. polyhedra
                    face_offset_start = piece.element_face_offsets[element_offset]
                    face_offset_end = piece.element_face_offsets[element_offset + 1]
                    assert face_offset_start >= 0 and face_offset_end >= face_offset_start, "Invalid element face offsets"
                    assert face_idx < face_offset_end - face_offset_start, "Face index out of range for this element"

                    face_offset = face_offset_start + face_idx
                    face_node_offset_start = piece.face_node_offsets[face_offset]
                    face_node_offset_end = piece.face_node_offsets[face_offset + 1]
                    assert face_node_offset_start >= 0 and face_node_offset_end >= face_node_offset_start, "Invalid face node offsets"
                    return face_node_offset_end - face_node_offset_start

            # uniform cell (including non-linear cells)
            return ShapesLibrary.get_num_face_corner_nodes(element_type, face_idx)

        @staticmethod
        @dav.func
        def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Get point IDs from a face.

            Returns:
                wp.int32: The number of points returned
            """
            assert cell.cell_id >= 0, "Invalid cell handle"
            assert face_idx >= 0, "Face index out of range"
            assert local_idx >= 0, "Local index out of range"

            piece = ds.pieces[cell.piece_idx]
            element_offset = cell.piece_element_offset
            element_type = cell.element_type

            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided:
                    # nsided cell i.e. polygons
                    return piece.connectivity[piece.element_node_offsets[element_offset] + local_idx]

            if wp.static(ensight_shapes.EN_nfaced in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nfaced:
                    # nfaced cell i.e. polyhedra
                    # lookup ds.element_face_signs to determine the face point ordering (i.e. whether the face normal is inward or outward facing)
                    # and adjust local_idx accordingly
                    face_offset_start = piece.element_face_offsets[element_offset]
                    face_offset = face_offset_start + face_idx
                    face_node_offset_start = piece.face_node_offsets[face_offset]
                    face_node_offset_end = piece.face_node_offsets[face_offset + 1]

                    face_sign = piece.element_face_signs[face_offset]
                    if face_sign < 0 and local_idx > 0:
                        # inward facing, reverse the local index
                        # Keep first point, reverse the rest: [0,1,2,3] -> [0,3,2,1]
                        nb_faces_points = face_node_offset_end - face_node_offset_start
                        local_idx = nb_faces_points - local_idx
                    return piece.connectivity[face_node_offset_start + local_idx]

            # uniform cell (including non-linear cells)
            cell_local_idx = ShapesLibrary.get_face_corner_node_index(element_type, face_idx, local_idx)
            nb_pts_per_element = ShapesLibrary.get_num_all_nodes(element_type)
            return piece.connectivity[element_offset * nb_pts_per_element + cell_local_idx]

    DataModel.CellAPI = EnSightCellAPI

    class EnSightPolyhedralCellAPI:
        @staticmethod
        @dav.func
        def get_cell_center(cell: CellHandle, ds: DatasetHandle) -> wp.vec3f:
            assert cell.cell_id >= 0, "Invalid cell handle"
            piece = ds.pieces[cell.piece_idx]
            assert piece.element_type == ensight_shapes.EN_nfaced, "Cell is not a polyhedron"

            element_offset = cell.piece_element_offset
            return piece.element_centers[element_offset]

        @staticmethod
        @dav.func
        def get_face_center(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.vec3f:
            assert cell.cell_id >= 0, "Invalid cell handle"
            piece = ds.pieces[cell.piece_idx]
            assert piece.element_type == ensight_shapes.EN_nfaced, "Cell is not a polyhedron"
            element_offset = cell.piece_element_offset
            face_offset_start = piece.element_face_offsets[element_offset]
            face_offset = face_offset_start + face_idx
            return piece.element_face_centers[face_offset]

    DataModel.PolyhedralCellAPI = EnSightPolyhedralCellAPI

    ShapeDispatch = shape_functions_dispatcher.get_shape_dispatcher(DataModel, ShapesLibrary, DATA_MODEL_ELEMENT_TYPES)

    # Standalone helper defined before EnSightCellLocatorAPI so that find_cell_containing_point
    # can call it without referencing EnSightCellLocatorAPI mid-class-body (empty closure cell).
    @dav.func
    def _ensight_cell_is_inside(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.bool:
        assert EnSightCellAPI.is_valid(cell), "Invalid cell handle"

        element_type = cell.element_type
        if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
            if element_type == ensight_shapes.EN_nsided:
                # nsided cell i.e. polygons;
                return False

        return ShapeDispatch.is_point_in_cell(position, cell, ds, element_type)

    class EnSightCellLocatorAPI:
        @staticmethod
        @dav.func
        def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
            assert cell.cell_id >= 0, "Invalid cell handle"

            empty = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)
            element_type = cell.element_type
            if wp.static(ensight_shapes.EN_nsided in DATA_MODEL_ELEMENT_TYPES):
                if element_type == ensight_shapes.EN_nsided:
                    # nsided cell i.e. polygons; no interpolation
                    return empty

            return ShapeDispatch.get_weights(position, cell, ds, element_type)

        @staticmethod
        @dav.func
        def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
            assert ds.cell_bvh_id != 0, "Cell locator BVH has not been built for the dataset. Call DatasetAPI.build_cell_locator() first."

            if EnSightCellAPI.is_valid(hint):
                if _ensight_cell_is_inside(ds, position, hint):
                    return hint

            radius = wp.vec3f(EPSILON)
            query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
            cell_idx = wp.int32(-1)
            while wp.bvh_query_next(query, cell_idx):
                cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
                cell = DatasetAPI.get_cell(ds, cell_id)
                if _ensight_cell_is_inside(ds, position, cell):
                    return cell

            return EnSightCellAPI.empty()

        @staticmethod
        @dav.func
        def _is_inside(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> wp.bool:
            return _ensight_cell_is_inside(ds, position, cell)

        @staticmethod
        @dav.func
        def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
            assert EnSightCellAPI.is_valid(cell), "Invalid cell handle"
            return _ensight_cell_is_inside(ds, point, cell)

    DataModel.CellLocatorAPI = EnSightCellLocatorAPI

    return DataModel
