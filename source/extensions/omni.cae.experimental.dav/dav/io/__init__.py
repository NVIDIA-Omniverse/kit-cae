# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Convenience I/O functions for loading datasets from files.

These functions provide quick ways to load data for testing and prototyping.
For production use, consider loading data directly with your preferred library
and constructing DAV datasets explicitly to maintain full control over the
data model and loading process.

Note:
    All I/O functions are optional convenience wrappers. For VTK data, it's
    recommended to use VTK or PyVista directly along with
    ``dav.data_models.vtk.utils.vtk_to_dataset`` for more control.
"""

from . import cgns

__all__ = ["cgns"]
