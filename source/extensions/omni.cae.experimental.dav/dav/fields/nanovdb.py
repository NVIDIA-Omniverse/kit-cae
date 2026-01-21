# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""NanoVDB-based field implementation for volume data.

This module provides field models for NanoVDB volume data using wp.Volume.
It supports float32 and vec3f data types with both 'ij' (Fortran) and 'xy' (Cartesian) indexing.

Example usage:
    ```python
    import warp as wp
    from dav.fields.nanovdb import get_field_model

    # Get a field model for float32 with xy indexing
    model = get_field_model(wp.float32, "xy")

    # Use in kernels
    @wp.kernel
    def process_volume(field: model.FieldHandle):
        idx = wp.tid()
        value = model.FieldAPI.get(field, idx)
        # Process value...
    ```
"""

__all__ = ["get_field_model"]

from typing import Any

import warp as wp

from .._helpers import cached
from .typing import FieldModel

# =============================================================================
# FieldHandle Type (shared across all NanoVDB fields)
# =============================================================================


@wp.struct
class FieldHandleNanoVDB:
    """Field containing NanoVDB volume data.

    This handle type is shared across all NanoVDB field types since
    volume_id is storage-type agnostic.
    """

    association: wp.int32
    volume_id: wp.uint64
    dims: wp.vec3i  # Volume dimensions for index to ijk conversion
    origin: wp.vec3i  # Volume origin (Ni, Nj, Nk)


# =============================================================================
# Internal utility functions for index conversion
# =============================================================================


@wp.func
def _idx_to_ijk_ij(idx: wp.int32, dims: wp.vec3i, origin: wp.vec3i) -> wp.vec3i:
    """Convert linear index to (i, j, k) coordinates for 'ij' (Fortran/column-major) indexing.

    In Fortran order, the FIRST dimension varies fastest:
    idx = i + j * Ni + k * Ni * Nj

    Args:
        idx: Linear index
        dims: Volume dimensions (Ni, Nj, Nk)
        origin: Volume origin (Ni, Nj, Nk)

    Returns:
        (i, j, k) coordinates
    """
    k = idx // (dims[0] * dims[1])
    remainder = idx % (dims[0] * dims[1])
    j = remainder // dims[0]
    i = remainder % dims[0]
    return wp.vec3i(i, j, k) + origin


@wp.func
def _idx_to_ijk_xy(idx: wp.int32, dims: wp.vec3i, origin: wp.vec3i) -> wp.vec3i:
    """Convert linear index to (i, j, k) coordinates for 'xy' (C/row-major) indexing.

    In C order, the LAST dimension varies fastest:
    idx = i * Nj * Nk + j * Nk + k
    (where i=x, j=y, k=z)

    Args:
        idx: Linear index
        dims: Volume dimensions (Nx, Ny, Nz)
        origin: Volume origin (Ni, Nj, Nk)

    Returns:
        (x, y, z) coordinates (which map to i, j, k)
    """
    i = idx // (dims[1] * dims[2])
    remainder = idx % (dims[1] * dims[2])
    j = remainder // dims[2]
    k = remainder % dims[2]
    return wp.vec3i(i, j, k) + origin


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
        return field.dims[0] * field.dims[1] * field.dims[2]


# -----------------------------------------------------------------------------
# Float32 Field APIs
# -----------------------------------------------------------------------------


class NanoVDBFieldAPI_f_ij(FieldAPIBase):
    """Field API for float32 NanoVDB volumes with 'ij' (Fortran) indexing."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.float32:
        """Get value at linear index."""
        ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)
        return wp.volume_lookup_f(field.volume_id, ijk[0], ijk[1], ijk[2])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.float32):
        """Set value at linear index."""
        ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)
        wp.volume_store_f(field.volume_id, ijk[0], ijk[1], ijk[2], value)

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


class NanoVDBFieldAPI_f_xy(FieldAPIBase):
    """Field API for float32 NanoVDB volumes with 'xy' (Cartesian) indexing."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.float32:
        """Get value at linear index."""
        ijk = _idx_to_ijk_xy(idx, field.dims, field.origin)
        return wp.volume_lookup_f(field.volume_id, ijk[0], ijk[1], ijk[2])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.float32):
        """Set value at linear index."""
        ijk = _idx_to_ijk_xy(idx, field.dims, field.origin)
        wp.volume_store_f(field.volume_id, ijk[0], ijk[1], ijk[2], value)

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


# -----------------------------------------------------------------------------
# Vec3f Field APIs
# -----------------------------------------------------------------------------


class NanoVDBFieldAPI_vec3f_ij(FieldAPIBase):
    """Field API for vec3f NanoVDB volumes with 'ij' (Fortran) indexing."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3f:
        """Get value at linear index."""
        ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)
        return wp.volume_lookup_v(field.volume_id, ijk[0], ijk[1], ijk[2])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3f):
        """Set value at linear index."""
        ijk = _idx_to_ijk_ij(idx, field.dims, field.origin)
        wp.volume_store_v(field.volume_id, ijk[0], ijk[1], ijk[2], value)

    @staticmethod
    @wp.func
    def zero() -> wp.vec3f:
        """Get a zero value of the appropriate type."""
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


class NanoVDBFieldAPI_vec3f_xy(FieldAPIBase):
    """Field API for vec3f NanoVDB volumes with 'xy' (Cartesian) indexing."""

    @staticmethod
    @wp.func
    def get(field: Any, idx: wp.int32) -> wp.vec3f:
        """Get value at linear index."""
        ijk = _idx_to_ijk_xy(idx, field.dims, field.origin)
        return wp.volume_lookup_v(field.volume_id, ijk[0], ijk[1], ijk[2])

    @staticmethod
    @wp.func
    def set(field: Any, idx: wp.int32, value: wp.vec3f):
        """Set value at linear index."""
        ijk = _idx_to_ijk_xy(idx, field.dims, field.origin)
        wp.volume_store_v(field.volume_id, ijk[0], ijk[1], ijk[2], value)

    @staticmethod
    @wp.func
    def zero() -> wp.vec3f:
        """Get a zero value of the appropriate type."""
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


# =============================================================================
# Field API type mapping
# =============================================================================

_field_api_types = {
    wp.float32: {"ij": NanoVDBFieldAPI_f_ij, "xy": NanoVDBFieldAPI_f_xy},
    wp.vec3f: {"ij": NanoVDBFieldAPI_vec3f_ij, "xy": NanoVDBFieldAPI_vec3f_xy},
}

# =============================================================================
# FieldModel factory function
# =============================================================================


@cached
def get_field_model(dtype, indexing: str = "xy") -> FieldModel:
    """Get a FieldModel for NanoVDB volumes.

    Args:
        dtype: Data type (wp.float32 or wp.vec3f)
        indexing: Index ordering - "ij" (Fortran) or "xy" (Cartesian). Default: "xy"
            - "ij": i varies fastest, then j, then k (idx = i + j*Ni + k*Ni*Nj)
            - "xy": x varies fastest, then y, then z (idx = x + y*Nx + z*Nx*Ny)

    Returns:
        FieldModel with FieldHandle and FieldAPI types

    Raises:
        ValueError: If dtype is not supported or indexing is invalid

    Example:
        >>> model = get_field_model(wp.float32, "xy")
        >>> # Use model.FieldHandle and model.FieldAPI in kernels
    """
    if dtype not in _field_api_types:
        raise ValueError(f"Unsupported dtype: {dtype}. NanoVDB only supports wp.float32 and wp.vec3f")

    if indexing not in ["ij", "xy"]:
        raise ValueError(f"Invalid indexing: {indexing}. Must be 'ij' or 'xy'")

    field_api_type = _field_api_types[dtype][indexing]

    class FieldModel:
        FieldHandle = FieldHandleNanoVDB
        FieldAPI = field_api_type

    return FieldModel
