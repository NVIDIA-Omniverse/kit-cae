# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from enum import Enum
from typing import Any, Protocol

import warp as wp

from ..data_models.typing import CellHandle, DatasetHandle


class AssociationType(Enum):
    """Enumeration for field association types."""

    # TODO: not sure this is best location for this.

    NOT_SPECIFIED = -1
    VERTEX = 0
    CELL = 1


# =============================================================================
# FieldHandle Protocol
# =============================================================================


class FieldHandle(Protocol):
    """Opaque handle for a field (wp.struct passed to kernels)."""

    ...


# =============================================================================
# FieldAPI Protocol
# =============================================================================


class FieldAPI(Protocol):
    """Static API for operations on FieldHandle."""

    @staticmethod
    def get_association(field: FieldHandle) -> AssociationType:
        """Get the association type of the field.

        Returns:
            AssociationType: The association type of the field
        """
        ...

    @staticmethod
    def get_count(field: FieldHandle) -> int:
        """Get the number of elements in the field.

        Args:
            field: FieldHandle struct containing the field data
        Returns:
            int: The number of elements in the field
        """
        ...

    @staticmethod
    def get(field: FieldHandle, idx: wp.int32) -> Any:
        """Get a value from the field by index.

        Args:
            field: FieldHandle struct containing the field data
            idx: Index of the value to get
        Returns:
            Any: The value at the given index
        """
        ...

    @staticmethod
    def set(field: FieldHandle, idx: wp.int32, value: Any):
        """Set a value in the field by index.

        Args:
            field: FieldHandle struct containing the field data
            idx: Index of the value to set
            value: Value to set at the given index
        """
        ...

    @staticmethod
    def zero() -> Any:
        """Get a zero value of the appropriate type.

        Returns:
            Any: A zero value of the appropriate type
        """
        ...

    @staticmethod
    def zero_s() -> Any:
        """Get a zero scalar value of the appropriate type.

        Returns:
            Any: A zero scalar value of the appropriate type
        """
        ...

    @staticmethod
    def zero_vec3() -> Any:
        """Get a zero vec3 value of the appropriate type.

        Returns:
            Any: A zero vec3 value of the appropriate type
        """
        ...


# =============================================================================
# InterpolatedFieldAPI Protocol
# =============================================================================


class InterpolatedFieldAPI(Protocol):
    """Static API for interpolated field operations."""

    @staticmethod
    def get(ds: DatasetHandle, field: FieldHandle, cell: CellHandle, position: wp.vec3f) -> Any:
        """Interpolate or retrieve field value at a position within a cell.

        Args:
            ds: Dataset struct from the data model
            field: FieldHandle struct containing the field data
            cell: Cell struct from the data model
            position: Position in 3D space (for vertex-associated fields)
        Returns:
            Interpolated or retrieved field value (scalar or vector)
        """
        ...


# =============================================================================
# FieldModel Protocol
# =============================================================================


class FieldModel(Protocol):
    """Protocol defining the interface for field modles in DAV operations.
    This protocol defines the complete API contract that field model implementations
    must satisfy. It combines:
    - Handle types: Opaque struct types passed to Warp kernels
    - API types: Static method interfaces for operating on handles

    Example usage in operators:
        def my_operator(field_model: FieldModel, field_handle: field_model.FieldHandle):
            # Access field API
            association = field_model.FieldAPI.get_association(field_handle)
            value = field_model.FieldAPI.get(field_handle, 0)
            field_model.FieldAPI.set(field_handle, 0, value)
            zero_value = field_model.FieldAPI.zero()
            zero_scalar_value = field_model.FieldAPI.zero_s()
            zero_vec3_value = field_model.FieldAPI.zero_vec3()
    """

    FieldHandle: type[FieldHandle]
    """The field handle type for this field model."""

    FieldAPI: type[FieldAPI]
    """The field API type for this field model."""
