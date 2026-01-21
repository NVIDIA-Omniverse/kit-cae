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

from .typing import DataModel


def get_collection_data_model(data_model: DataModel) -> DataModel:
    """Create a collection data model from a base data model.

    The collection data model is a data model for a collection of datasets. Currently,
    we only support collections of datasets with the same data model.

    Args:
        data_model: The data model for all datasets in the collection

    Returns:
        DataModel: The data model for the collection of datasets

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

    @wp.struct
    class CollectionInterpolatedCellHandle:
        cell_id: wp.vec2i
        inside: wp.bool  # redundant, but convenient
        piece_interpolated_cell: data_model.InterpolatedCellHandle

    @wp.struct
    class CollectionCellLinkHandle:
        """Handle for cell links in collection.

        Wraps a piece's cell link and tracks which block it belongs to.
        """

        point_id: wp.vec2i
        piece_point_id: data_model.PointIdHandle

    class CollectionCellAPI:
        """Static API for operations on cells."""

        @staticmethod
        @wp.func
        def _get_piece_cell(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> data_model.CellHandle:
            """Get the piece cell from the cell handle. We use asserts to check validity in debug mode
            without incurring a performance penalty in release mode."""
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for _get_piece_cell"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = data_model.DatasetAPI.get_cell(piece, cell.piece_cell_id)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for _get_piece_cell"
            return piece_cell

        @staticmethod
        @wp.func
        def is_valid(cell: CollectionCellHandle) -> wp.bool:
            return cell.cell_id.x >= 0 and cell.cell_id.y >= 0

        @staticmethod
        @wp.func
        def empty() -> CollectionCellHandle:
            return CollectionCellHandle(cell_id=wp.vec2i(-1, -1), piece_cell_id=data_model.CellIdHandle(0))

        @staticmethod
        @wp.func
        def get_cell_id(cell: CollectionCellHandle) -> wp.vec2i:
            return cell.cell_id

        @staticmethod
        @wp.func
        def get_num_points(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> wp.int32:
            return data_model.CellAPI.get_num_points(
                CollectionCellAPI._get_piece_cell(cell, ds), ds.pieces[cell.cell_id.x]
            )

        @staticmethod
        @wp.func
        def get_point_id(cell: CollectionCellHandle, local_idx: wp.int32, ds: CollectionDatasetHandle) -> wp.vec2i:
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for get_point_id"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = CollectionCellAPI._get_piece_cell(cell, ds)
            piece_pt_id = data_model.CellAPI.get_point_id(piece_cell, local_idx, piece)
            piece_pt_idx = data_model.DatasetAPI.get_point_idx_from_id(piece, piece_pt_id)
            return wp.vec2i(cell.cell_id.x, piece_pt_idx)

        @staticmethod
        @wp.func
        def get_num_faces(cell: CollectionCellHandle, ds: CollectionDatasetHandle) -> wp.int32:
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for get_num_faces"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = CollectionCellAPI._get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_num_faces"
            return data_model.CellAPI.get_num_faces(piece_cell, piece)

        @staticmethod
        @wp.func
        def get_face_num_points(
            cell: CollectionCellHandle, face_idx: wp.int32, ds: CollectionDatasetHandle
        ) -> wp.int32:
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for get_face_num_points"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = CollectionCellAPI._get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_face_num_points"
            return data_model.CellAPI.get_face_num_points(piece_cell, face_idx, piece)

        @staticmethod
        @wp.func
        def get_face_point_id(
            cell: CollectionCellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: CollectionDatasetHandle
        ) -> wp.vec2i:
            assert CollectionCellAPI.is_valid(cell), "Invalid cell handle for get_face_point_id"
            piece = ds.pieces[cell.cell_id.x]
            piece_cell = CollectionCellAPI._get_piece_cell(cell, ds)
            assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for get_face_point_id"
            piece_pt_id = data_model.CellAPI.get_face_point_id(piece_cell, face_idx, local_idx, piece)
            piece_pt_idx = data_model.DatasetAPI.get_point_idx_from_id(piece, piece_pt_id)
            return wp.vec2i(cell.cell_id.x, piece_pt_idx)

    class CollectionInterpolatedCellAPI:
        """Static API for operations on interpolated cells."""

        @staticmethod
        @wp.func
        def empty() -> CollectionInterpolatedCellHandle:
            return CollectionInterpolatedCellHandle(
                cell_id=wp.vec2i(-1, -1), inside=False, piece_interpolated_cell=data_model.InterpolatedCellAPI.empty()
            )

        @staticmethod
        @wp.func
        def is_valid(i_cell: CollectionInterpolatedCellHandle) -> wp.bool:
            return (
                i_cell.cell_id.x >= 0
                and i_cell.cell_id.y >= 0
                and data_model.InterpolatedCellAPI.is_valid(i_cell.piece_interpolated_cell)
            )

        @staticmethod
        @wp.func
        def get_cell_id(i_cell: CollectionInterpolatedCellHandle) -> wp.vec2i:
            return i_cell.cell_id

        @staticmethod
        @wp.func
        def is_inside(i_cell: CollectionInterpolatedCellHandle) -> wp.bool:
            return i_cell.inside

        @staticmethod
        @wp.func
        def get_weight(i_cell: CollectionInterpolatedCellHandle, local_idx: wp.int32) -> wp.float32:
            assert CollectionInterpolatedCellAPI.is_valid(i_cell), "Invalid interpolated cell handle for get_weight"
            assert data_model.InterpolatedCellAPI.is_valid(
                i_cell.piece_interpolated_cell
            ), "Invalid piece interpolated cell for get_weight"
            return data_model.InterpolatedCellAPI.get_weight(i_cell.piece_interpolated_cell, local_idx)

    class CollectionCellLinksAPI:
        """Static API for cell links operations.

        Cell links delegate to the underlying piece's cell links.
        Since points don't cross block boundaries, each point's cells
        are entirely contained within a single piece.
        """

        @staticmethod
        @wp.func
        def empty() -> CollectionCellLinkHandle:
            """Create an empty cell link."""
            cell_link = CollectionCellLinkHandle()
            cell_link.point_id = wp.vec2i(-1, -1)
            cell_link.piece_cell_link = data_model.CellLinksAPI.empty()
            return cell_link

        @staticmethod
        @wp.func
        def is_valid(cell_link: CollectionCellLinkHandle) -> wp.bool:
            """Check if cell link is valid."""
            return cell_link.point_id.x >= 0 and cell_link.point_id.y >= 0

        @staticmethod
        @wp.func
        def get_point_id(cell_link: CollectionCellLinkHandle) -> wp.vec2i:
            """Get the point ID for a given cell link."""
            return cell_link.point_id

        @staticmethod
        @wp.func
        def get_num_cells(cell_link: CollectionCellLinkHandle, ds: CollectionDatasetHandle) -> wp.int32:
            """Get the number of cells in the cell link."""
            return data_model.CellLinksAPI.get_num_cells(cell_link.piece_cell_link, ds.pieces[cell_link.point_id.x])

        @staticmethod
        @wp.func
        def get_cell_id(
            cell_link: CollectionCellLinkHandle, cell_idx: wp.int32, ds: CollectionDatasetHandle
        ) -> wp.vec2i:
            """Get the cell ID for a given cell index in the cell link.

            Returns cell IDs in collection format (block_id, local_cell_idx).
            """
            # Get the piece cell ID from the wrapped cell link
            piece_cell_id = data_model.CellLinksAPI.get_cell_id(
                cell_link.piece_cell_link, cell_idx, ds.pieces[cell_link.point_id.x]
            )

            piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(ds.pieces[cell_link.point_id.x], piece_cell_id)

            return wp.vec2i(cell_link.point_id.x, piece_cell_idx)

    class CollectionDatasetAPI:
        @staticmethod
        @wp.func
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
        @wp.func
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
        @wp.func
        def get_cell(dataset: CollectionDatasetHandle, id: wp.vec2i) -> CollectionCellHandle:
            assert id.x >= 0 and id.x < dataset.pieces.shape[0], "Invalid piece index for get_cell"
            assert id.y >= 0, "Invalid cell index for get_cell"
            piece = dataset.pieces[id.x]
            piece_cell_id = data_model.DatasetAPI.get_cell_id_from_idx(piece, id.y)
            return CollectionCellHandle(cell_id=id, piece_cell_id=piece_cell_id)

        @staticmethod
        @wp.func
        def get_num_cells(dataset: CollectionDatasetHandle) -> wp.int32:
            num_cells = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                num_cells += data_model.DatasetAPI.get_num_cells(dataset.pieces[piece_idx])
            return num_cells

        @staticmethod
        @wp.func
        def get_num_points(dataset: CollectionDatasetHandle) -> wp.int32:
            num_points = wp.int32(0)
            for piece_idx in range(dataset.pieces.shape[0]):
                num_points += data_model.DatasetAPI.get_num_points(dataset.pieces[piece_idx])
            return num_points

        @staticmethod
        @wp.func
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
        @wp.func
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
        @wp.func
        def get_point(dataset: CollectionDatasetHandle, id: wp.vec2i) -> wp.vec3f:
            piece = dataset.pieces[id.x]
            return data_model.DatasetAPI.get_point(piece, data_model.DatasetAPI.get_point_id_from_idx(piece, id.y))

        @staticmethod
        @wp.func
        def get_cell_link(dataset: CollectionDatasetHandle, point_id: wp.vec2i) -> CollectionCellLinkHandle:
            """Get the cell link for a given point id.

            Args:
                dataset: The collection dataset to query
                point_id: The point id in collection format (block_id, local_point_idx)

            Returns:
                CollectionCellLinkHandle: The cell link for the given point
            """
            block_id = point_id.x
            local_point_idx = point_id.y

            # Get the piece dataset
            piece = dataset.pieces[block_id]

            # Convert local point index to piece's point ID format
            piece_point_id = data_model.DatasetAPI.get_point_id_from_idx(piece, local_point_idx)

            # Wrap in collection cell link handle
            return CollectionCellLinkHandle(point_id=point_id, piece_point_id=piece_point_id)

        @staticmethod
        @wp.func
        def evaluate_position(
            dataset: CollectionDatasetHandle, position: wp.vec3f, cell: CollectionCellHandle
        ) -> CollectionInterpolatedCellHandle:
            if not CollectionCellAPI.is_valid(cell):
                wp.printf("Invalid cell handle for evaluate_position\n")
                return CollectionInterpolatedCellAPI.empty()
            else:
                piece = dataset.pieces[cell.cell_id.x]
                piece_cell = data_model.DatasetAPI.get_cell(piece, cell.piece_cell_id)
                assert data_model.CellAPI.is_valid(piece_cell), "Invalid piece cell for evaluate_position"
                i_piece_cell = data_model.DatasetAPI.evaluate_position(piece, position, piece_cell)
                assert data_model.InterpolatedCellAPI.is_valid(
                    i_piece_cell
                ), "Invalid interpolated piece cell for evaluate_position"
                if data_model.InterpolatedCellAPI.is_valid(i_piece_cell):
                    return CollectionInterpolatedCellHandle(
                        cell_id=cell.cell_id,
                        piece_interpolated_cell=i_piece_cell,
                        inside=data_model.InterpolatedCellAPI.is_inside(i_piece_cell),
                    )
                else:
                    wp.printf("Invalid interpolated piece cell for evaluate_position\n")
                    return CollectionInterpolatedCellAPI.empty()

        @staticmethod
        @wp.func
        def find_cell_containing_point(
            dataset: CollectionDatasetHandle, position: wp.vec3f, hint: CollectionCellHandle
        ) -> CollectionCellHandle:
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
                piece_cell = data_model.DatasetAPI.find_cell_containing_point(piece, position, piece_hint)
                if data_model.CellAPI.is_valid(piece_cell):
                    piece_cell_id = data_model.CellAPI.get_cell_id(piece_cell)
                    piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(piece, piece_cell_id)
                    return CollectionCellHandle(
                        cell_id=wp.vec2i(hint.cell_id.x, piece_cell_idx), piece_cell_id=piece_cell_id
                    )

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
                piece_cell = data_model.DatasetAPI.find_cell_containing_point(piece, position, empty_piece_hint)
                if data_model.CellAPI.is_valid(piece_cell):
                    piece_cell_id = data_model.CellAPI.get_cell_id(piece_cell)
                    piece_cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(piece, piece_cell_id)
                    return CollectionCellHandle(
                        cell_id=wp.vec2i(piece_idx, piece_cell_idx), piece_cell_id=piece_cell_id
                    )

            # wp.printf("Nothing found for point (%f, %f, %f) bvh id: %d\n", position.x, position.y, position.z, dataset.piece_bvh_id)
            return CollectionCellAPI.empty()

        @staticmethod
        def build_cell_locator(
            data_model: DataModel, dataset: CollectionDatasetHandle, device: Any
        ) -> tuple[bool, Any]:
            """Build a spatial acceleration structure for cell location queries.

            Note: This should not be called directly. Instead, call Dataset.build_cell_locator()
            which builds a BVH across all pieces in the collection and stores the piece_bvh_id.
            """
            raise NotImplementedError(
                "build_cell_locator should not be called on collection data model directly. "
                "Use Dataset.build_cell_locator() which handles collection piece locators."
            )

        @staticmethod
        def build_cell_links(data_model: DataModel, dataset: CollectionDatasetHandle, device: Any) -> tuple[bool, Any]:
            """Build the cell links for the collection.

            Note: This should not be called directly. Instead, call Dataset.build_cell_links()
            which builds cell links for each piece in the collection individually.
            """
            raise NotImplementedError(
                "build_cell_links should not be called on collection data model directly. "
                "Use Dataset.build_cell_links() which handles building links for each piece."
            )

    class CollectionDataModel:
        DatasetHandle = CollectionDatasetHandle
        CellHandle = CollectionCellHandle
        InterpolatedCellHandle = CollectionInterpolatedCellHandle
        CellLinkHandle = CollectionCellLinkHandle

        CellIdHandle = wp.vec2i
        PointIdHandle = wp.vec2i
        DatasetAPI = CollectionDatasetAPI
        CellAPI = CollectionCellAPI
        InterpolatedCellAPI = CollectionInterpolatedCellAPI
        CellLinksAPI = CollectionCellLinksAPI

    return CollectionDataModel
