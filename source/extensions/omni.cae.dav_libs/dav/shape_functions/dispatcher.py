# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import warp as wp

import dav

from . import types
from .typing import ShapeDispatchAPI, UniformShapesLibraryAPI


@dav.cached
def get_shape_dispatcher(data_model: dav.DataModel, shapes_library: UniformShapesLibraryAPI, unique_cell_types: list[int]) -> ShapeDispatchAPI:
    """
    Factory function to create a shape functions dispatcher API class.

    This works on a partially populated DataModel. Only DataSetAPI, CellAPI, DatasetHandle, CellHandle
    need to be provided.

    Args:
        data_model: The data model to create the dispatcher for (partially populated)
        shapes_library: The uniform shapes library API to use for shape function implementations
        unique_cell_types: A list of unique cell types present in the dataset, used to determine which shape functions to retrieve

    Returns:
        A class that implements the ShapeDispatchAPI protocol, with dispatching logic based on cell type.
    """
    SHAPE_TYPES = {shapes_library.get_shape_function_type(ct) for ct in unique_cell_types}

    class VoidShape:
        """A fallback shape implementation that returns default values for unsupported cell types."""

        @staticmethod
        @dav.func
        def is_point_in_cell(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.bool:
            return False

        @staticmethod
        @dav.func
        def get_weights(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.vec(
            length=dav.config.max_points_per_cell, dtype=wp.float32
        ):
            return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)

    VOID_SHAPE = VoidShape
    TETRA_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_TETRA).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_TETRA in SHAPE_TYPES else VOID_SHAPE
    PYRA_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_PYRA).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_PYRA in SHAPE_TYPES else VOID_SHAPE
    PENTA_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_PENTA).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_PENTA in SHAPE_TYPES else VOID_SHAPE
    HEXA_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_HEXA).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_HEXA in SHAPE_TYPES else VOID_SHAPE
    VOXEL_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_VOXEL).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_VOXEL in SHAPE_TYPES else VOID_SHAPE
    POLYHEDRON_SHAPE = types.get_shape_module(types.ELEMENT_TYPE_POLYHEDRON).get_shape(data_model, shapes_library) if types.ELEMENT_TYPE_POLYHEDRON in SHAPE_TYPES else VOID_SHAPE

    # Dynamically create a new class that implements the ShapeFunctionsDispatcherAPI protocol
    class ShapeDispatcher:
        """Implements the ShapeDispatchAPI protocol by dispatching to the appropriate shape function implementations based on cell type."""

        @staticmethod
        @dav.func
        def is_point_in_cell(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.bool:
            # we deliberately avoid static since that was causing the kernel hash to change which forced unnecessary recompilation.
            shape_type = shapes_library.get_shape_function_type(cell_type)
            if shape_type == types.ELEMENT_TYPE_TETRA:
                return TETRA_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_PYRA:
                return PYRA_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_PENTA:
                return PENTA_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_HEXA:
                return HEXA_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_VOXEL:
                return VOXEL_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_POLYHEDRON:
                return POLYHEDRON_SHAPE.is_point_in_cell(point, cell, dataset, cell_type)
            return False

        @staticmethod
        @dav.func
        def get_weights(point: wp.vec3f, cell: data_model.CellHandle, dataset: data_model.DatasetHandle, cell_type: wp.int32) -> wp.vec(
            length=dav.config.max_points_per_cell, dtype=wp.float32
        ):
            # we deliberately avoid static since that was causing the kernel hash to change which forced unnecessary recompilation.
            shape_type = shapes_library.get_shape_function_type(cell_type)
            if shape_type == types.ELEMENT_TYPE_TETRA:
                return TETRA_SHAPE.get_weights(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_PYRA:
                return PYRA_SHAPE.get_weights(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_PENTA:
                return PENTA_SHAPE.get_weights(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_HEXA:
                return HEXA_SHAPE.get_weights(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_VOXEL:
                return VOXEL_SHAPE.get_weights(point, cell, dataset, cell_type)
            elif shape_type == types.ELEMENT_TYPE_POLYHEDRON:
                return POLYHEDRON_SHAPE.get_weights(point, cell, dataset, cell_type)
            return wp.vec(length=dav.config.max_points_per_cell, dtype=wp.float32)

    return ShapeDispatcher
