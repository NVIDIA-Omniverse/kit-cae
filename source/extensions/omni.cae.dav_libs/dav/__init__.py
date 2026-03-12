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

from warp import config as wp_config

# Setup warp configuration before importing any other modules, to ensure that the configuration is applied globally.
# Enable vector component overwrites for Warp
# This is necessary for the DAV code to work correctly.
wp_config.enable_vector_component_overwrites = True
# keeping this disabled for now since it causes some issues with
# testing and Kit-CAE
# wp_config.max_unroll = 1  # to improve compilation times
wp_config.enable_backward = False  # this is simply not needed in our case
# wp_config.verbose = True


from .core import cached, config, recorder, register_pre_compile_hook, scoped_timer, utils
from .data_models import CellAPI, CellHandle, CellIdHandle, CellLinksAPI, DataModel, DatasetAPI, DatasetHandle, PointIdHandle
from .dataset import Dataset, DatasetCollection
from .field import Field
from .fields import AssociationType, FieldAPI, FieldHandle, FieldModel, InterpolatedFieldAPI
from .typing import DatasetLike, FieldLike

__all__ = [
    # Core exports
    "cached",
    "func",
    "kernel",
    "config",
    "recorder",
    "register_pre_compile_hook",
    "scoped_timer",
    "utils",
    # Types and protocols
    "DatasetLike",
    "FieldLike",
    # Main classes
    "Dataset",
    "DatasetCollection",
    "Field",
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
    "PointIdHandle",
    "CellIdHandle",
    "DatasetAPI",
    "CellAPI",
    "CellLinksAPI",
    # Submodules
    # (import dav.data_models, dav.fields, dav.io, dav.operators directly as needed)
]


func = utils.func
kernel = utils.kernel
