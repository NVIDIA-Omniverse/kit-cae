# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Collection field model for handling multiple field pieces.

This module provides a field model that wraps multiple field pieces,
similar to how the collection data model wraps multiple datasets.

Example usage:
    ```python
    import warp as wp
    from dav.fields import array, collection

    # Get base field model for float32
    base_model = array.get_field_model(wp.float32)

    # Get collection field model
    collection_model = collection.get_field_model(base_model)

    # Create collection field handle with multiple pieces
    field = collection_model.FieldHandle()
    field.association = AssociationType.VERTEX.value
    field.pieces = wp.array([piece1, piece2, piece3], dtype=base_model.FieldHandle)

    # The collection automatically queries each piece for its size
    total_count = collection_model.FieldAPI.get_count(field)
    ```
"""

__all__ = ["get_field_model"]

from typing import Any

import warp as wp

from .._helpers import cached
from .typing import FieldModel


@cached
def get_field_model(field_model: FieldModel) -> FieldModel:
    """Create a collection field model from a base field model.

    Args:
        field_model: The base field model for individual pieces

    Returns:
        FieldModel: A collection field model that handles multiple field pieces

    Example:
        >>> import warp as wp
        >>> from dav.fields import array, collection
        >>> base_model = array.get_field_model(wp.float32)
        >>> coll_model = collection.get_field_model(base_model)
    """

    @wp.struct
    class CollectionFieldHandle:
        """Field handle for a collection of field pieces.

        The collection maps global indices to (piece_idx, local_idx) pairs.
        Each piece is queried for its size using the base field model's get_count() API.
        """

        association: wp.int32
        pieces: wp.array(dtype=field_model.FieldHandle)

    class CollectionFieldAPI:
        """Static API for operations on collection field handles."""

        @staticmethod
        @wp.func
        def get_association(field: CollectionFieldHandle) -> wp.int32:
            """Get the association type of the field."""
            return field.association

        @staticmethod
        @wp.func
        def get_count(field: CollectionFieldHandle) -> wp.int32:
            """Get the total number of elements across all pieces.

            Args:
                field: Collection field handle

            Returns:
                Total number of elements
            """
            total = wp.int32(0)
            for piece_idx in range(field.pieces.shape[0]):
                total += field_model.FieldAPI.get_count(field.pieces[piece_idx])
            return total

        @staticmethod
        @wp.func
        def _global_idx_to_piece_idx(field: CollectionFieldHandle, global_idx: wp.int32) -> wp.vec2i:
            """Convert global index to (piece_idx, local_idx) pair.

            Args:
                field: Collection field handle
                global_idx: Global index across all pieces

            Returns:
                wp.vec2i where x=piece_idx, y=local_idx, or (-1, -1) if invalid
            """
            offset = wp.int32(0)
            for piece_idx in range(field.pieces.shape[0]):
                piece_size = field_model.FieldAPI.get_count(field.pieces[piece_idx])
                if global_idx < offset + piece_size:
                    return wp.vec2i(piece_idx, global_idx - offset)
                offset += piece_size

            wp.printf("Invalid global index %d for collection field\n", global_idx)
            return wp.vec2i(-1, -1)

        @staticmethod
        @wp.func
        def get(field: CollectionFieldHandle, idx: wp.int32):
            """Get a value from the field by global index.

            Args:
                field: Collection field handle
                idx: Global index

            Returns:
                The value at the given index
            """
            piece_info = CollectionFieldAPI._global_idx_to_piece_idx(field, idx)
            if piece_info.x < 0:
                return field_model.FieldAPI.zero()

            piece_idx = piece_info.x
            local_idx = piece_info.y
            return field_model.FieldAPI.get(field.pieces[piece_idx], local_idx)

        @staticmethod
        @wp.func
        def set(field: CollectionFieldHandle, idx: wp.int32, value: Any):
            """Set a value in the field by global index.

            Args:
                field: Collection field handle
                idx: Global index
                value: Value to set
            """
            piece_info = CollectionFieldAPI._global_idx_to_piece_idx(field, idx)
            if piece_info.x < 0:
                return

            piece_idx = piece_info.x
            local_idx = piece_info.y
            field_model.FieldAPI.set(field.pieces[piece_idx], local_idx, value)

        @staticmethod
        @wp.func
        def zero():
            """Get a zero value of the appropriate type."""
            return field_model.FieldAPI.zero()

        @staticmethod
        @wp.func
        def zero_s():
            """Get a zero scalar value of the appropriate type."""
            return field_model.FieldAPI.zero_s()

        @staticmethod
        @wp.func
        def zero_vec3():
            """Get a zero vec3 value of the appropriate type."""
            return field_model.FieldAPI.zero_vec3()

    class CollectionFieldModel:
        """Field model for collection of field pieces."""

        FieldHandle = CollectionFieldHandle
        FieldAPI = CollectionFieldAPI

    return CollectionFieldModel
