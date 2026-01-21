# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Data models for DAV.

This package contains data model implementations and utilities for working with
different dataset formats in DAV.

Available modules:
    - typing: Protocol definitions for data models
    - validation: Utilities for validating data model implementations
    - collection: Factory for creating collection data models
    - vtk: VTK data model implementations (image_data, structured_grid, unstructured_grid)
    - sids: SIDS data model implementations (unstructured)
"""

from . import validation

# Import types and protocols
from .typing import (
    CellAPI,
    CellHandle,
    CellIdHandle,
    CellLinkHandle,
    CellLinksAPI,
    DataModel,
    DatasetAPI,
    DatasetHandle,
    InterpolatedCellAPI,
    InterpolatedCellHandle,
    PointIdHandle,
)

__all__ = [
    # Submodules
    "validation",
    # -  Data model types and protocols -----------------------------------------
    # Types and protocols
    "DataModel",
    # Handle types
    "DatasetHandle",
    "CellHandle",
    "InterpolatedCellHandle",
    "CellLinkHandle",
    "PointIdHandle",
    "CellIdHandle",
    # API types
    "DatasetAPI",
    "CellAPI",
    "InterpolatedCellAPI",
    "CellLinksAPI",
]
