# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any

import warp as wp

from .._helpers import cached
from .typing import FieldModel

# =============================================================================
# FieldHandle Types
# =============================================================================


@wp.struct
class FieldHandleF:
    """Field containing float32 data."""

    association: wp.int32
    data: wp.array(dtype=wp.float32)
    data_1: wp.array(dtype=wp.float32)
    data_2: wp.array(dtype=wp.float32)


@wp.struct
class FieldHandleD:
    """Field containing float64 data."""

    association: wp.int32
    data: wp.array(dtype=wp.float64)
    data_1: wp.array(dtype=wp.float64)
    data_2: wp.array(dtype=wp.float64)


@wp.struct
class FieldHandleI:
    """Field containing int32 data."""

    association: wp.int32
    data: wp.array(dtype=wp.int32)
    data_1: wp.array(dtype=wp.int32)
    data_2: wp.array(dtype=wp.int32)


@wp.struct
class FieldHandleVec3f:
    """Field containing vec3f data."""

    association: wp.int32
    data: wp.array(dtype=wp.vec3f)


@wp.struct
class FieldHandleVec3d:
    """Field containing vec3d data."""

    association: wp.int32
    data: wp.array(dtype=wp.vec3d)


@wp.struct
class FieldHandleVec3i:
    """Field containing vec3i data."""

    association: wp.int32
    data: wp.array(dtype=wp.vec3i)


_field_handle_types = {
    wp.float32: FieldHandleF,
    wp.float64: FieldHandleD,
    wp.int32: FieldHandleI,
    wp.vec3f: FieldHandleVec3f,
    wp.vec3d: FieldHandleVec3d,
    wp.vec3i: FieldHandleVec3i,
}

# =============================================================================
# FieldAPI Types
# =============================================================================


class FieldAPIBase:
    """Base class for field APIs."""

    @staticmethod
    @wp.func
    def get_association(field: Any) -> wp.int32:
        """Get the association type of the field."""
        return field.association

    @staticmethod
    @wp.func
    def get_count(field: Any) -> wp.int32:
        """Get the number of elements in the field."""
        return field.data.shape[0]


class ScalarFieldAPI_f(FieldAPIBase):
    """Field API for float32 scalar data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.float32:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.float32):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.float32:
        """Get a zero value of the appropriate type."""
        return wp.float32(0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float32:
        """Get a zero scalar value of the appropriate type."""
        return wp.float32(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3f:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3f(0.0, 0.0, 0.0)


class ScalarFieldAPI_d(FieldAPIBase):
    """Field API for float64 scalar data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.float64:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.float64):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.float64:
        """Get a zero value of the appropriate type."""
        return wp.float64(0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float64:
        """Get a zero scalar value of the appropriate type."""
        return wp.float64(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3d:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3d(0.0, 0.0, 0.0)


class ScalarFieldAPI_i(FieldAPIBase):
    """Field API for int32 scalar data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.int32:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.int32):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.int32:
        """Get a zero value of the appropriate type."""
        return wp.int32(0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.int32:
        """Get a zero scalar value of the appropriate type."""
        return wp.int32(0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3i:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3i(0, 0, 0)


class VectorFieldAPI_f(FieldAPIBase):
    """Field API for vec3f data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3f:
        """Get a value from the field by index."""
        return wp.vec3f(field.data[idx], field.data_1[idx], field.data_2[idx])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3f):
        """Set a value in the field by index."""
        field.data[idx] = value.x
        field.data_1[idx] = value.y
        field.data_2[idx] = value.z

    @staticmethod
    @wp.func
    def zero() -> wp.vec3f:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3f(0.0, 0.0, 0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float32:
        """Get a zero scalar value of the appropriate type."""
        return wp.float32(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3f:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3f(0.0, 0.0, 0.0)


class VectorFieldAPI_d(FieldAPIBase):
    """Field API for vec3d data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3d:
        """Get a value from the field by index."""
        return wp.vec3d(field.data[idx], field.data_1[idx], field.data_2[idx])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3d):
        """Set a value in the field by index."""
        field.data[idx] = value.x
        field.data_1[idx] = value.y
        field.data_2[idx] = value.z

    @staticmethod
    @wp.func
    def zero() -> wp.vec3d:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3d(0.0, 0.0, 0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float64:
        """Get a zero scalar value of the appropriate type."""
        return wp.float64(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3d:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3d(0.0, 0.0, 0.0)


class VectorFieldAPI_i(FieldAPIBase):
    """Field API for vec3i data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3i:
        """Get a value from the field by index."""
        return wp.vec3i(field.data[idx], field.data_1[idx], field.data_2[idx])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3i):
        """Set a value in the field by index."""
        field.data[idx] = value[0]
        field.data_1[idx] = value[1]
        field.data_2[idx] = value[2]

    @staticmethod
    @wp.func
    def zero() -> wp.vec3i:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3i(0, 0, 0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.int32:
        """Get a zero scalar value of the appropriate type."""
        return wp.int32(0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3i:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3i(0, 0, 0)


class VectorFieldAPI_vec3f(FieldAPIBase):
    """Field API for vec3f data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3f:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3f):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.vec3f:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3f(0.0, 0.0, 0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float32:
        """Get a zero scalar value of the appropriate type."""
        return wp.float32(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3f:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3f(0.0, 0.0, 0.0)


class VectorFieldAPI_vec3d(FieldAPIBase):
    """Field API for vec3d data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3d:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3d):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.vec3d:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3d(0.0, 0.0, 0.0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.float64:
        """Get a zero scalar value of the appropriate type."""
        return wp.float64(0.0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3d:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3d(0.0, 0.0, 0.0)


class VectorFieldAPI_vec3i(FieldAPIBase):
    """Field API for vec3i data."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3i:
        """Get a value from the field by index."""
        return field.data[idx]

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3i):
        """Set a value in the field by index."""
        field.data[idx] = value

    @staticmethod
    @wp.func
    def zero() -> wp.vec3i:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3i(0, 0, 0)

    @staticmethod
    @wp.func
    def zero_s() -> wp.int32:
        """Get a zero scalar value of the appropriate type."""
        return wp.int32(0)

    @staticmethod
    @wp.func
    def zero_vec3() -> wp.vec3i:
        """Get a zero vec3 value of the appropriate type."""
        return wp.vec3i(0, 0, 0)


# dictionary of field api types for each value dtype
# the key is the value dtype, the value is a dictionary of storage dtypes and their corresponding field api types
# the storage dtype is the key, the field api type is the value
# Scalar value types only support scalar storage dtypes
# Vector value types support both scalar and vector storage dtypes
# We don't support type conversions so float32 cannot be fixed with vec3f storage, etc.
_field_api_types = {
    wp.float32: {wp.float32: ScalarFieldAPI_f},
    wp.float64: {wp.float64: ScalarFieldAPI_d},
    wp.int32: {wp.int32: ScalarFieldAPI_i},
    wp.vec3f: {wp.float32: VectorFieldAPI_f, wp.vec3f: VectorFieldAPI_vec3f},
    wp.vec3d: {wp.float64: VectorFieldAPI_d, wp.vec3d: VectorFieldAPI_vec3d},
    wp.vec3i: {wp.int32: VectorFieldAPI_i, wp.vec3i: VectorFieldAPI_vec3i},
}

# =============================================================================
# FieldModel Types and helper functions
# =============================================================================


def _get_scalar_dtype(dtype):
    """Get the scalar dtype of a vector dtype."""
    if dtype == wp.vec3f:
        return wp.float32
    elif dtype == wp.vec3d:
        return wp.float64
    elif dtype == wp.vec3i:
        return wp.int32
    elif dtype == wp.float32:
        return wp.float32
    elif dtype == wp.float64:
        return wp.float64
    elif dtype == wp.int32:
        return wp.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype} for scalar dtype")


@cached
def get_field_model(value_dtype, storage_dtype=None) -> FieldModel:
    if storage_dtype is None:
        # if no storage dtype is specified, use the value dtype for storage
        storage_dtype = value_dtype

    elif _get_scalar_dtype(value_dtype) != _get_scalar_dtype(storage_dtype):
        raise ValueError(f"Incompatible value and storage dtypes: {value_dtype} and {storage_dtype}")

    # use storage dtype to get field handle type
    field_handle_type = _field_handle_types[storage_dtype]

    # use value dtype to get field api type
    field_api_type = _field_api_types[value_dtype][storage_dtype]

    class FieldModel:
        FieldHandle = field_handle_type
        FieldAPI = field_api_type

    return FieldModel
