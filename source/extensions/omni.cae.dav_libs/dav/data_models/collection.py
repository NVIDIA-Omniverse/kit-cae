# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["get_collection_data_model"]

from typing import Any

import warp as wp

import dav


@dav.cached
def get_collection_data_model(data_model: dav.DataModel) -> dav.DataModel:
    """Create a collection data model from a base data model.

    The collection data model is a data model for a collection of datasets. Currently,
    we only support collections of datasets with the same data model.

    Args:
        data_model: The data model for all datasets in the collection

    Returns:
        dav.DataModel: The data model for the collection of datasets

    Example:
        >>> from dav.data_models import collection
        >>> collection_data_model = collection.get_collection_data_model(data_model)
    """

    @wp.struct
    class CollectionDatasetHandle:
        pieces: wp.array(dtype=data_model.DatasetHandle)
        piece_bvh_id: wp.uint64

    @wp.struct
    class CollectionCellHandle:
        cell_id: wp.vec2i
        piece_cell_id: data_model.CellIdHandle

    # Standalone helpers defined before CollectionCellAPI so that methods can call each other
    # without referencing CollectionCellAPI mid-class-body (empty closure cell).
    @dav.func
    def _collection_cell_is_valid(cell: CollectionCellHandle) -> wp.bool:
        return cell.cell_id.x >= 0 and cell.cell_id.y >= 0

    @dav.func
    def _get_piece_cell(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> data_model.CellHandle:
        """Get the piece cell from the cell handle."""
        assert _collection_cell_is_valid(cell), "Invalid cell handle for _get_piece_cell"
        piece = ds.pieces[cell.cell_id.x]
        piece_cell = data_model.DatasetAPI.get_cell(piece, cell.piece_cell_id)
        assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for _get_piece_cell"
        return piece_cell

    class CollectionCellAPI:
        """Static API for operations on cells."""

        @staticmethod
        @dav.func
        def is_valid(cell: CollectionCellHandle) -> wp.bool:
            return _collection_cell_is_valid(cell)

        @staticmethod
        @dav.func
        def empty() -> CollectionCellHandle:
            return CollectionCellHandle(cell_id=wp.vec2i(-1, -1), piece_cell_id=data_model.CellIdHandle(0))

        @staticmethod
        @dav.func
        def get_cell_id(cell: CollectionCellHandle) -> wp.vec2i:
            return cell.cell_id

        @staticmethod
        @dav.func
        def get_num_points(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> wp.int32:
            return data_model.CellAPI.get_num_points(_get_piece_cell(cell, ds), ds.pieces[cell.cell_id.x])

        @staticmethod
        @dav.func
        def get_point_id(cell: CollectionCellHandle, local_idx: wp.int32, ds: CollectionDatasetHandle) -> wp.vec2i:
            assert _collection_cell_is_valid(cell), "Invalid cell handle for get_point_id"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = _get_piece_cell(cell, ds)
            piece_pt_id = data_model.CellAPI.get_point_id(piece_cell, local_idx, piece)
            piece_pt_idx = data_model.DatasetAPI.get_point_idx_from_id(piece, piece_pt_id)
            return wp.vec2i(cell.cell_id.x, piece_pt_idx)

        @staticmethod
        @dav.func
        def get_num_faces(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> wp.int32:
            assert _collection_cell_is_valid(cell), "Invalid cell handle for get_num_faces"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = _get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_num_faces"
            return data_model.CellAPI.get_num_faces(piece_cell, piece)

        @staticmethod
        @dav.func
        def get_face_num_points(cell: CollectionCellHandle, face_idx: wp.int32, ds: CollectionDatasetHandle) -> wp.int32:
            assert _collection_cell_is_valid(cell), "Invalid cell handle for get_face_num_points"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = _get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_face_num_points"
            return data_model.CellAPI.get_face_num_points(piece_cell, face_idx, piece)

        @staticmethod
        @dav.func
        def get_face_point_id(cell: CollectionCellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: CollectionDatasetHandle) -> wp.vec2i:
            assert _collection_cell_is_valid(cell), "Invalid cell handle for get_face_point_id"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = _get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_face_point_id"
            piece_pt_id = data_model.CellAPI.get_face_point_id(piece_cell, face_idx, local_idx, piece)
            piece_pt_idx = data_model.DatasetAPI.get_point_idx_from_id(piece, piece_pt_id)
            return wp.vec2i(cell.cell_id.x, piece_pt_idx)

    class CollectionCellLinksAPI:
        """Static API for cell links operations.

        Cell links delegate to the underlying piece's cell links.
        Since points don't cross block boundaries, each point's cells
        are entirely contained within a single piece.
        """

        @staticmethod
        @dav.func
        def is_valid(point_id: wp.vec2i) -> wp.bool:
            """Check if point id is valid."""
            return point_id.x >= 0 and point_id.y >= 0

        @staticmethod
        @dav.func
        def get_num_cells(point_id: wp.vec2i, ds: CollectionDatasetHandle) -> wp.int32:
            """Get the number of cells in the cell link."""
            piece_pt_id = data_model.DatasetAPI.get_point_id_from_idx(ds.pieces[point_id.x], point_id.y)
            return data_model.CellLinksAPI.get_num_cells(piece_pt_id, ds.pieces[point_id.x])

        @staticmethod
        @dav.func
        def get_cell_id(point_id: wp.vec2i, local_idx: wp.int32, ds: CollectionDatasetHandle) -> wp.vec2i:
            """Get the cell ID for a given point at the specified local index.

            Returns cell IDs in collection format (block_id, local_cell_idx).
            """

            piece = ds.pieces[point_id.x]
            piece_pt_id = data_model.DatasetAPI.get_point_id_from_idx(piece, point_id.y)

            # Get the piece cell ID from the wrapped cell link
            piece_cell_id = data_model.CellLinksAPI.get_cell_id(piece_pt_id, local_idx, piece)
            piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(piece, piece_cell_id)

            return wp.vec2i(point_id.x, piece_cell_idx)

    class CollectionDatasetAPI:
        @staticmethod
        @dav.func
        def get_cell_id_from_idx(dataset: CollectionDatasetHandle, local_idx: wp.int32) -> wp.vec2i:
            """
            Convert a local index to a cell id. The cell id is a tuple of the piece index and the local index within the piece.
            """
            cell_count = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                piece = dataset.pieces[piece_idx]
                piece_cell_count = data_model.DatasetAPI.get_num_cells(piece)
                if cell_count + piece_cell_count > local_idx:
                    return wp.vec2i(piece_idx, local_idx - cell_count)
                cell_count += piece_cell_count

            wp.printf("Invalid cell index %d for collection", local_idx)
            return wp.vec2i(-1, -1)

        @staticmethod
        @dav.func
        def get_cell_idx_from_id(dataset: CollectionDatasetHandle, id: wp.vec2i) -> wp.int32:
            cell_count = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                piece = dataset.pieces[piece_idx]
                if id.x == piece_idx:
                    return cell_count + id.y
                cell_count += data_model.DatasetAPI.get_num_cells(piece)
            wp.printf("Invalid cell id %d, %d for collection", id.x, id.y)
            return wp.int32(-1)

        @staticmethod
        @dav.func
        def get_cell(dataset: CollectionDatasetHandle, id: wp.vec2i) -> CollectionCellHandle:
            assert id.x >= 0 and id.x < dataset.pieces.shape[0], "Invalid piece index for get_cell"
            assert id.y >= 0, "Invalid cell index for get_cell"
            piece = dataset.pieces[id.x]
            piece_cell_id = data_model.DatasetAPI.get_cell_id_from_idx(piece, id.y)
            return CollectionCellHandle(cell_id=id, piece_cell_id=piece_cell_id)

        @staticmethod
        @dav.func
        def get_num_cells(dataset: CollectionDatasetHandle) -> wp.int32:
            num_cells = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                num_cells += data_model.DatasetAPI.get_num_cells(dataset.pieces[piece_idx])
            return num_cells

        @staticmethod
        @dav.func
        def get_num_points(dataset: CollectionDatasetHandle) -> wp.int32:
            num_points = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                num_points += data_model.DatasetAPI.get_num_points(dataset.pieces[piece_idx])
            return num_points

        @staticmethod
        @dav.func
        def get_point_id_from_idx(dataset: CollectionDatasetHandle, local_idx: wp.int32) -> wp.vec2i:
            """
            Convert a local index to a point id. The point id is a tuple of the piece index and the local index within the piece.
            """
            point_count = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                piece = dataset.pieces[piece_idx]
                piece_point_count = data_model.DatasetAPI.get_num_points(piece)
                if point_count + piece_point_count > local_idx:
                    return wp.vec2i(piece_idx, local_idx - point_count)
                point_count += piece_point_count
            wp.printf("Invalid point index %d for collection", local_idx)
            return wp.vec2i(-1, -1)

        @staticmethod
        @dav.func
        def get_point_idx_from_id(dataset: CollectionDatasetHandle, id: wp.vec2i) -> wp.int32:
            point_count = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                piece = dataset.pieces[piece_idx]
                if id.x == piece_idx:
                    return point_count + id.y
                point_count += data_model.DatasetAPI.get_num_points(piece)
            wp.printf("Invalid point id %d, %d for collection", id.x, id.y)
            return wp.int32(-1)

        @staticmethod
        @dav.func
        def get_point(dataset: CollectionDatasetHandle, id: wp.vec2i) -> wp.vec3f:
            piece = dataset.pieces[id.x]
            return data_model.DatasetAPI.get_point(piece, data_model.DatasetAPI.get_point_id_from_idx(piece, id.y))

        @staticmethod
        def build_cell_locator(data_model: dav.DataModel, dataset: CollectionDatasetHandle, device: Any) -> tuple[bool, Any]:
            """Build a spatial acceleration structure for cell location queries.

            Note: This should not be called directly. Instead, call Dataset.build_cell_locator()
            which builds a BVH across all pieces in the collection and stores the piece_bvh_id.
            """
            raise NotImplementedError(
                "build_cell_locator should not be called on collection data model directly. Use Dataset.build_cell_locator() which handles collection piece locators."
            )

        @staticmethod
        def build_cell_links(data_model: dav.DataModel, dataset: CollectionDatasetHandle, device: Any) -> tuple[bool, Any]:
            """Build the cell links for the collection.

            Note: This should not be called directly. Instead, call Dataset.build_cell_links()
            which builds cell links for each piece in the collection individually.
            """
            raise NotImplementedError(
                "build_cell_links should not be called on collection data model directly. Use Dataset.build_cell_links() which handles building links for each piece."
            )

    class CollectionCellLocatorAPI:
        @staticmethod
        @dav.func
        def evaluate_position(dataset: CollectionDatasetHandle, position: wp.vec3f, cell: CollectionCellHandle) -> wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32):
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for evaluate_position"
            piece = dataset.pieces[cell.cell_id.x]
            piece_cell = data_model.DatasetAPI.get_cell(piece, cell.piece_cell_id)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for evaluate_position"
            return data_model.CellLocatorAPI.evaluate_position(piece, position, piece_cell)

        @staticmethod
        @dav.func
        def find_cell_containing_point(dataset: CollectionDatasetHandle, position: wp.vec3f, hint: CollectionCellHandle) -> CollectionCellHandle:
            """
            Find the cell containing a given point.

            Args:
                dataset: The dataset to query
                position: The point to locate in world coordinates
                hint: A hint cell to start the search

            Returns:
                CollectionCellHandle: The cell containing the point, or empty cell if outside
            """
            if dataset.piece_bvh_id == 0:
                wp.printf("ERROR: Piece locator not built for collection dataset\n")
                return CollectionCellAPI.empty()

            if CollectionCellAPI.is_valid(hint):
                # If hint is valid, get the piece specified by the hint and check if the point can be found in that piece.
                piece = dataset.pieces[hint.cell_id.x]
                piece_hint = data_model.DatasetAPI.get_cell(piece, hint.piece_cell_id)
                piece_cell = data_model.CellLocatorAPI.find_cell_containing_point(piece, position, piece_hint)
                if data_model.CellAPI.is_valid(piece_cell):
                    piece_cell_id = data_model.CellAPI.get_cell_id(piece_cell)
                    piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(piece, piece_cell_id)
                    return CollectionCellHandle(cell_id=wp.vec2i(hint.cell_id.x, piece_cell_idx), piece_cell_id=piece_cell_id)

            # # If hint is not valid, query the piece BVH to find the piece containing the point.
            # radius = wp.vec3f(1.0e-2, 1.0e-2, 1.0e-2)
            # query = wp.bvh_query_aabb(dataset.piece_bvh_id, position - radius, position + radius)
            # piece_idx = wp.int32(-1)
            # while wp.bvh_query_next(query, piece_idx):
            # BUG: not using BVH here since the nested bvh query seems to
            # brek this outer look when using CUDA. Need to debug what's going on.
            # Until then, we're using a simple loop over all pieces.
            empty_piece_hint = data_model.CellAPI.empty()
            for piece_idx in range(dataset.pieces.shape[0]):
                # For each piece, find the cell containing the point.
                piece = dataset.pieces[piece_idx]
                piece_cell = data_model.CellLocatorAPI.find_cell_containing_point(piece, position, empty_piece_hint)
                if data_model.CellAPI.is_valid(piece_cell):
                    piece_cell_id = data_model.CellAPI.get_cell_id(piece_cell)
                    piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(piece, piece_cell_id)
                    return CollectionCellHandle(cell_id=wp.vec2i(piece_idx, piece_cell_idx), piece_cell_id=piece_cell_id)

            # wp.printf("Nothing found for point (%f, %f, %f) bvh id: %d\n", position.x, position.y, position.z, dataset.piece_bvh_id)
            return CollectionCellAPI.empty()

        @staticmethod
        @dav.func
        def point_in_cell(dataset: CollectionDatasetHandle, point: wp.vec3f, cell: CollectionCellHandle) -> wp.bool:
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for point_in_cell"
            piece = dataset.pieces[cell.cell_id.x]
            piece_cell = data_model.DatasetAPI.get_cell(piece, cell.piece_cell_id)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for point_in_cell"
            return data_model.CellLocatorAPI.point_in_cell(piece, point, piece_cell)

    class CollectionDataModel:
        DatasetHandle = CollectionDatasetHandle
        CellHandle = CollectionCellHandle

        CellIdHandle = wp.vec2i
        PointIdHandle = wp.vec2i
        DatasetAPI = CollectionDatasetAPI
        CellAPI = CollectionCellAPI
        CellLinksAPI = CollectionCellLinksAPI
        CellLocatorAPI = CollectionCellLocatorAPI

    return CollectionDataModel
