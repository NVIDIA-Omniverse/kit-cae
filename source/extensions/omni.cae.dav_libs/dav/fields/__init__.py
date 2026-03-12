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
DAV Fields - Field implementations for different data structures.
"""

# Only types are imported.
# Import types and protocols
from .typing import AssociationType, FieldAPI, FieldHandle, FieldModel, InterpolatedFieldAPI

__all__ = [
    # Submodules with factory functions
    # (import fields.array, fields.nanovdb, etc. to access the factory functions explicitly)
    # Types and protocols
    "AssociationType",
    "FieldAPI",
    "FieldHandle",
    "FieldModel",
    "InterpolatedFieldAPI",
]
