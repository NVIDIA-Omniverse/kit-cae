# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any, Protocol

import warp as wp


class CellHandle(Protocol):
    """Opaque handle for a cell (wp.struct passed to kernels)."""

    ...


class InterpolatedCellHandle(Protocol):
    """Opaque handle for an interpolated cell (wp.struct passed to kernels)."""

    ...


class DatasetHandle(Protocol):
    """Opaque handle for a dataset (wp.struct passed to kernels)."""

    ...


class CellLinkHandle(Protocol):
    """Opaque handle for cell links, i.e., cells containing a point (wp.struct passed to kernels)."""

    ...


class PointIdHandle(Protocol):
    """Opaque handle for point IDs. Data models assign this to a Warp dtype (e.g., wp.int32)."""

    ...


class CellIdHandle(Protocol):
    """Opaque handle for cell IDs. Data models assign this to a Warp dtype (e.g., wp.int32)."""

    ...


class CellAPI(Protocol):
    """Static API for operations on CellHandle."""

    @staticmethod
    def empty() -> CellHandle:
        """Create an empty cell.

        Returns:
            CellHandle: An empty cell
        """
        ...

    @staticmethod
    def is_valid(cell: CellHandle) -> wp.bool:
        """Check if a cell is valid.

        Args:
            cell: The cell to check (CellHandle)

        Returns:
            wp.bool: True if the cell is valid, False otherwise
        """
        ...

    @staticmethod
    def get_cell_id(cell: CellHandle) -> CellIdHandle:
        """Get the id of a cell.

        Args:
            cell: The cell to query (CellHandle)

        Returns:
            CellIdHandle: The id of the cell
        """
        ...

    @staticmethod
    def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a cell.

        Args:
            cell: The cell to query (CellHandle)
            ds: The dataset containing the cell (DatasetHandle)

        Returns:
            wp.int32: The number of points in the cell
        """
        ...

    @staticmethod
    def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> PointIdHandle:
        """Get the point id from a cell's local index.

        Args:
            cell: The cell to query (CellHandle)
            local_idx: The local index within the cell (wp.int32, 0-based)
            ds: The dataset containing the cell (DatasetHandle)

        Returns:
            PointIdHandle: The global point id
        """
        ...

    @staticmethod
    def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of faces in a cell.

        Args:
            cell: The cell to query (CellHandle)
            ds: The dataset containing the cell (DatasetHandle)

        Returns:
            wp.int32: The number of faces in the cell
        """
        ...

    @staticmethod
    def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
        """Get the number of points in a face.

        Args:
            cell: The cell to query (CellHandle)
            face_idx: Local face index (wp.int32, 0-based)
            ds: The dataset containing the cell (DatasetHandle)

        Returns:
            wp.int32: The number of points in the face
        """
        ...

    @staticmethod
    def get_face_point_id(
        cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle
    ) -> PointIdHandle:
        """Get a point ID from a face.

        Args:
            cell: The cell to query (CellHandle)
            face_idx: Local face index (wp.int32, 0-based)
            local_idx: Local index within the face (wp.int32, 0-based)
            ds: The dataset containing the cell (DatasetHandle)

        Returns:
            PointIdHandle: The global point id at the given position in the face
        """
        ...


class InterpolatedCellAPI(Protocol):
    """Static API for operations on InterpolatedCellHandle."""

    ...

    @staticmethod
    def empty() -> InterpolatedCellHandle:
        """Create an empty interpolated cell.

        Returns:
            InterpolatedCellHandle: An empty interpolated cell
        """
        ...

    @staticmethod
    def is_valid(i_cell: InterpolatedCellHandle) -> wp.bool:
        """Check if an interpolated cell is valid.

        Args:
            i_cell: The interpolated cell to check (InterpolatedCellHandle)
        Returns:
            wp.bool: True if the interpolated cell is valid, False otherwise
        """
        ...

    @staticmethod
    def get_cell_id(i_cell: InterpolatedCellHandle) -> CellIdHandle:
        """Get the cell id from an interpolated cell.

        Args:
            i_cell: The interpolated cell to query (InterpolatedCellHandle)
        Returns:
            CellIdHandle: The cell id
        """
        ...

    @staticmethod
    def is_inside(i_cell: InterpolatedCellHandle) -> wp.bool:
        """Check if the interpolated cell is inside the cell.

        Args:
            i_cell: The interpolated cell to check (InterpolatedCellHandle)
        Returns:
            wp.bool: True if inside, False otherwise
        """
        ...

    @staticmethod
    def get_weight(i_cell: InterpolatedCellHandle, local_idx: wp.int32) -> wp.float32:
        """Get the interpolation weight for a given local index.

        Args:
            i_cell: The interpolated cell to query (InterpolatedCellHandle)
            local_idx: The local index within the cell (wp.int32, 0-based)
        Returns:
            wp.float32: The interpolation weight at the given local index
        """
        ...


class CellLinksAPI(Protocol):
    """Static API for operations on CellLinkHandle."""

    ...

    @staticmethod
    def empty() -> CellLinkHandle:
        """Create an empty cell links.

        Returns:
            CellLinkHandle: An empty cell links
        """
        ...

    @staticmethod
    def is_valid(cell_link: CellLinkHandle) -> wp.bool:
        """Check if cell link is valid.

        Args:
            cell_link: The cell link to check (CellLinkHandle)
        Returns:
            wp.bool: True if the cell link is valid, False otherwise
        """
        ...

    @staticmethod
    def get_point_id(cell_link: CellLinkHandle) -> PointIdHandle:
        """Get the point id for a given cell link.

        Args:
            cell_link: The cell link to query (CellLinkHandle)
        Returns:
            PointIdHandle: The point id
        """
        ...

    @staticmethod
    def get_num_cells(cell_link: CellLinkHandle, ds: DatasetHandle) -> wp.int32:
        """Get the number of cells in the cell link.

        Args:
            cell_link: The cell link to query (CellLinkHandle)
            ds: The dataset containing the cell link (DatasetHandle)
        Returns:
            wp.int32: The number of cells in the cell link
        """
        ...

    @staticmethod
    def get_cell_id(cell_link: CellLinkHandle, cell_idx: wp.int32, ds: DatasetHandle) -> CellIdHandle:
        """Get the cell id for a given cell index in the cell link.

        Args:
            cell_link: The cell link to query (CellLinkHandle)
            cell_idx: The cell index to query (wp.int32, 0-based)
            ds: The dataset containing the cell link (DatasetHandle)
        Returns:
            CellIdHandle: The cell id at the given cell index
        """
        ...


class DatasetAPI(Protocol):
    """Static API for operations on DatasetHandle."""

    @staticmethod
    def get_cell_id_from_idx(dataset: DatasetHandle, local_idx: wp.int32) -> CellIdHandle:
        """Get a cell id from a dataset by local index.

        Args:
            dataset: The dataset to query (DatasetHandle)
            local_idx: The local index within the dataset (wp.int32, 0-based)
        Returns:
            CellIdHandle: The cell id at the given local index
        """
        ...

    @staticmethod
    def get_cell_idx_from_id(dataset: DatasetHandle, id: CellIdHandle) -> wp.int32:
        """Get a cell index from a dataset by cell id.

        Args:
            dataset: The dataset to query (DatasetHandle)
            id: The cell id
        Returns:
            wp.int32: The cell index (0-based)
        """
        ...

    @staticmethod
    def get_cell(dataset: DatasetHandle, id: CellIdHandle) -> CellHandle:
        """Get a cell from the dataset by id.

        Args:
            dataset: The dataset to query (DatasetHandle)
            id: The cell id

        Returns:
            CellHandle: The cell at the given id
        """
        ...

    @staticmethod
    def get_num_cells(dataset: DatasetHandle) -> wp.int32:
        """Get the number of cells in the dataset.

        Args:
            dataset: The dataset to query (DatasetHandle)

        Returns:
            wp.int32: The number of cells in the dataset
        """
        ...

    @staticmethod
    def get_num_points(dataset: DatasetHandle) -> wp.int32:
        """Get the number of points in the dataset.

        Args:
            dataset: The dataset to query (DatasetHandle)

        Returns:
            wp.int32: The number of points in the dataset
        """
        ...

    @staticmethod
    def get_point_id_from_idx(dataset: DatasetHandle, local_idx: wp.int32) -> PointIdHandle:
        """Get a point id from a dataset by local index.

        Args:
            dataset: The dataset to query (DatasetHandle)
            local_idx: The local index within the dataset (wp.int32, 0-based)
        Returns:
            PointIdHandle: The point id at the given local index
        """
        ...

    @staticmethod
    def get_point_idx_from_id(dataset: DatasetHandle, id: PointIdHandle) -> wp.int32:
        """Get a point index from a dataset by point id.

        Args:
            dataset: The dataset to query (DatasetHandle)
            id: The point id
        Returns:
            wp.int32: The point index (0-based)
        """
        ...

    @staticmethod
    def get_point(dataset: DatasetHandle, id: PointIdHandle) -> wp.vec3f:
        """Get a point from the dataset by id.

        Args:
            dataset: The dataset to query (DatasetHandle)
            id: The point id

        Returns:
            wp.vec3f: The point at the given id
        """
        ...

    @staticmethod
    def get_cell_link(dataset: DatasetHandle, point_id: PointIdHandle) -> CellLinkHandle:
        """Get the cell link for a given point id.

        Args:
            dataset: The dataset to query (DatasetHandle)
            point_id: The point id to query
        Returns:
            CellLinkHandle: The cell link for the given point id
        """
        ...

    @staticmethod
    def evaluate_position(dataset: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> InterpolatedCellHandle:
        """Evaluate a position within a cell to get interpolation weights.

        Args:
            dataset: The dataset to query (DatasetHandle)
            position: The position to evaluate (wp.vec3f)
            hint: The cell containing the position (CellHandle)

        Returns:
            InterpolatedCellHandle: The interpolated cell information
        """
        ...

    @staticmethod
    def find_cell_containing_point(dataset: DatasetHandle, point: wp.vec3f, hint: CellHandle) -> CellHandle:
        """Find the cell containing a given point.

        Args:
            dataset: The dataset to query (DatasetHandle)
            point: The point to locate (wp.vec3f)
            hint: A hint cell to start the search (CellHandle)

        Returns:
            CellHandle: The cell containing the point
        """
        ...

    @staticmethod
    def build_cell_locator(data_model, dataset: DatasetHandle, device: str) -> tuple[bool, Any]:
        """Build a cell locator for the dataset. This is not a wp.func
           and hence cannot be called from within a kernel.

        Args:
            data_model: The data model module
            dataset: The dataset to build the locator for (DatasetHandle)
            device: The device to use for building the locator

        Returns:
            tuple[bool, Any]: A tuple containing a success flag and the cell locator handle
        """
        ...

    @staticmethod
    def build_cell_links(data_model, dataset: DatasetHandle, device: str) -> tuple[bool, Any]:
        """Build the cell links for the dataset. This is not a wp.func
        and hence cannot be called from within a kernel.

        Args:
            data_model: The data model module
            dataset: The dataset to build the links for (DatasetHandle)
            device: The device to use for building the links

        Returns:
            tuple[bool, Any]: A tuple containing a success flag and the cell links handle
        """
        ...


class DataModel(Protocol):
    """Protocol defining the interface for data models used in DAV operations.

    This protocol defines the complete API contract that data model implementations
    must satisfy. It combines:
    - Handle types: Opaque struct types passed to Warp kernels
    - API types: Static method interfaces for operating on handles

    Example usage in operators:
        def my_operator(data_model: DataModel, dataset_handle: data_model.DatasetHandle):
            # Access dataset API
            num_cells = data_model.DatasetAPI.get_num_cells(dataset_handle)

            # Get a cell by index
            cell_id = data_model.DatasetAPI.get_cell_id_from_idx(dataset_handle, wp.int32(0))
            cell = data_model.DatasetAPI.get_cell(dataset_handle, cell_id)

            # Check if cell is valid
            if data_model.CellAPI.is_valid(cell):
                # Do something with the cell
                pass
    """

    # Handle types - opaque structs passed to kernels
    DatasetHandle: type[DatasetHandle]
    """The dataset handle type for this data model."""

    CellHandle: type[CellHandle]
    """The cell handle type for this data model."""

    InterpolatedCellHandle: type[InterpolatedCellHandle]
    """The interpolated cell handle type for this data model."""

    CellLinkHandle: type[CellLinkHandle]
    """The cell link handle type for this data model."""

    PointIdHandle: type
    """The point ID handle type for this data model (e.g., wp.int32, wp.int64)."""

    CellIdHandle: type
    """The cell ID handle type for this data model (e.g., wp.int32, wp.int64)."""

    # API types - static method interfaces for operations
    DatasetAPI: type[DatasetAPI]
    """Static API for dataset operations."""

    CellAPI: type[CellAPI]
    """Static API for cell operations."""

    InterpolatedCellAPI: type[InterpolatedCellAPI]
    """Static API for interpolated cell operations."""

    CellLinksAPI: type[CellLinksAPI]
    """Static API for cell links operations."""
