# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""DAV Fields - Field implementations for different data structures.

This package provides field model factory functions for different underlying data structures:
- `array`: Field models backed by wp.array (dense storage)
- `nanovdb`: Field models backed by wp.Volume/NanoVDB (sparse storage)
- `collection`: Field models for collections of field pieces
- `typing`: Common types and protocols used across field implementations

Example usage:
    ```python
    import warp as wp
    from dav.fields import array, nanovdb, collection

    # Get a field model for float32 array
    model = array.get_field_model(wp.float32)

    # Get a field model for float32 NanoVDB with xy indexing
    vdb_model = nanovdb.get_field_model(wp.float32, "xy")

    # Get a collection field model
    coll_model = collection.get_field_model(model)
    ```
"""

# Import factory functions from implementations
from . import array, collection, nanovdb

# Import types and protocols
from .typing import AssociationType, FieldAPI, FieldHandle, FieldModel, InterpolatedFieldAPI

__all__ = [
    # Submodules with factory functions
    "array",
    "collection",
    "nanovdb",
    # Types and protocols
    "AssociationType",
    "FieldAPI",
    "FieldHandle",
    "FieldModel",
    "InterpolatedFieldAPI",
]
