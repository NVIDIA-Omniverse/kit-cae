# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
DAV - Data Analysis and Visualization Framework

A flexible framework for scientific data analysis and visualization that works
seamlessly across different data models (VTK, SIDS, etc.) through protocol-based
abstraction.
"""

__version__ = "0.1.0"

from . import config, data_models, fields, io, operators
from ._helpers import cached
from .data_models import (
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
from .dataset import Dataset, DatasetCollection
from .field import Field, FieldCollection
from .fields import AssociationType, FieldAPI, FieldHandle, FieldModel, InterpolatedFieldAPI
from .timer import scoped_timer
from .typing import DatasetLike, FieldLike

__all__ = [
    # Config
    "config",
    # Main classes
    "scoped_timer",
    # Types and protocols
    "DatasetLike",
    "FieldLike",
    # Main classes
    "Dataset",
    "DatasetCollection",
    "Field",
    "FieldCollection",
    # Field types and protocols
    "AssociationType",
    "FieldAPI",
    "FieldHandle",
    "FieldModel",
    "InterpolatedFieldAPI",
    # Data model types and protocols
    "DataModel",
    "DatasetHandle",
    "CellHandle",
    "InterpolatedCellHandle",
    "CellLinkHandle",
    "PointIdHandle",
    "CellIdHandle",
    "DatasetAPI",
    "CellAPI",
    "InterpolatedCellAPI",
    "CellLinksAPI",
    # Submodules
    "data_models",
    "fields",
    "io",
    "operators",
]
