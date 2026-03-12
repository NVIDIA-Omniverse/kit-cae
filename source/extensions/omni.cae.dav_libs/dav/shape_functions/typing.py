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

import dav

from .. import config


class ShapeAPI(Protocol):
    """
    Protocol for shape API.
    """

    @staticmethod
    @dav.func
    def is_point_in_cell(point: wp.vec3f, cell: Any, dataset: Any, cell_type: wp.int32) -> wp.bool:
        """
        Check if a position is inside a cell.

        Args:
            point: The position to check (wp.vec3f)
            cell: The cell to check against (type depends on the data model)
            dataset: The dataset containing the cell (type depends on the data model)
            cell_type: The type of the cell (data model specific)

        Returns:
            bool: True if the position is inside the cell, False otherwise.
        """
        ...

    @staticmethod
    @dav.func
    def get_weights(point: wp.vec3f, cell: Any, dataset: Any, cell_type: wp.int32) -> wp.vec(length=config.max_points_per_cell, dtype=wp.float32):
        """
        Get interpolation weights for a position inside a cell.

        Args:
            point: The position to get weights for (wp.vec3f)
            cell: The cell to get weights for (type depends on the data model)
            dataset: The dataset containing the cell (type depends on the data model)
            cell_type: The type of the cell (data model specific)

        Returns:
            wp.vec(length=config.max_points_per_cell, dtype=wp.float32):
            The interpolation weights for the position inside the cell.
        """
        ...


class ShapeDispatchAPI(Protocol):
    """
    Protocol for shape dispatch API.

    This is a variation of the ShapeAPI except it adds the data model specific cell_type
    as an argument to the functions, which can be used to dispatch to different
    shape function implementations.
    """

    @staticmethod
    @dav.func
    def is_point_in_cell(point: wp.vec3f, cell: Any, dataset: Any, cell_type: wp.int32) -> wp.bool:
        """
        Check if a position is inside a cell of a specific type.

        Args:
            point: The position to check (wp.vec3f)
            cell: The cell to check against (type depends on the data model)
            dataset: The dataset containing the cell (type depends on the data model)
            cell_type: The type of the cell (data model specific)

        Returns:
            bool: True if the position is inside the cell, False otherwise.
        """
        ...

    @staticmethod
    @dav.func
    def get_weights(point: wp.vec3f, cell: Any, dataset: Any, cell_type: wp.int32) -> wp.vec(length=config.max_points_per_cell, dtype=wp.float32):
        """
        Get interpolation weights for a position inside a cell of a specific type.

        Args:
            point: The position to get weights for (wp.vec3f)
            cell: The cell to get weights for (type depends on the data model)
            dataset: The dataset containing the cell (type depends on the data model)
            cell_type: The type of the cell (data model specific)

        Returns:
            wp.vec(length=config.max_points_per_cell, dtype=wp.float32):
            The interpolation weights for the position inside the cell.
        """
        ...


class UniformShapesLibraryAPI(Protocol):
    """
    Protocol for uniform shapes library API.

    This is used to define the interface for uniform shapes library classes, which provide static methods for querying shape information
    based on cell types. The API is generated from a list of shape definitions, which specify the topology and shape function mappings for each cell type.

    The `cell_type` refers to data model specific cell type, while `shape_type` refers to
    the shapes defined in this package.
    """

    @staticmethod
    @dav.func
    def get_shape_function_type(cell_type: wp.int32) -> wp.int32:
        """
        Get the shape function type for a given cell type.

        Args:
            cell_type: The cell type to query (data model specific)

        Returns:
            int: The shape function type corresponding to the cell type, or ELEMENT_TYPE_NONE if the cell type is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_num_all_nodes(cell_type: wp.int32) -> wp.int32:
        """
        Get the total number of nodes for a given cell type.

        Args:
            cell_type: The cell type to query (data model specific)
        Returns:
            int: The total number of nodes for the cell type, or 0 if the cell type is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_num_corner_nodes(cell_type: wp.int32) -> wp.int32:
        """
        Get the number of corner nodes for a given cell type.

        Args:
            cell_type: The cell type to query (data model specific)
        Returns:
            int: The number of corner nodes for the cell type, or 0 if the cell type is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_corner_node_index_in_vtk_order(cell_type: wp.int32, node_idx: wp.int32) -> wp.int32:
        """
        Get the VTK corner node index for a given cell type and node index.

        This is used to map the node ordering of the data model to the expected node ordering of the shape functions, which is based on VTK's convention.

        Args:
            cell_type: The cell type to query (data model specific)
            node_idx: The node index in the data model's ordering (0-based)

        Returns:
            int: The VTK corner node index corresponding to the given cell type and node index, or -1 if the cell type or node index is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_num_faces(cell_type: wp.int32) -> wp.int32:
        """
        Get the number of faces for a given cell type.

        Args:
            cell_type: The cell type to query (data model specific)
        Returns:
            int: The number of faces for the cell type, or 0 if the cell type is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_num_face_corner_nodes(cell_type: wp.int32, face_idx: wp.int32) -> wp.int32:
        """
        Get the number of corner nodes for a given face of a cell type.

        Args:
            cell_type: The cell type to query (data model specific)
            face_idx: The face index to query (0-based)
        Returns:
            int: The number of corner nodes for the specified face of the cell type, or 0 if the cell type or face index is not supported.
        """
        ...

    @staticmethod
    @dav.func
    def get_face_corner_node_index(cell_type: wp.int32, face_idx: wp.int32, node_idx: wp.int32) -> wp.int32:
        """
        Get the corner node index for a given face and node of a cell type.
        Winding order of the face corner nodes is expected to be such that face normal is outward facing for a
        right-handed coordinate system.

        Args:
            cell_type: The cell type to query (data model specific)
            face_idx: The face index to query (0-based)
            node_idx: The node index in the face's ordering (0-based)
        Returns:
            int: The corner node index corresponding to the specified face and node of the cell type, or -1 if the cell type, face index, or node index is not supported.
        """
        ...
