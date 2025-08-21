# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

"""
A module for getting paths to CAE materials.
"""

__all__ = ["get_cae_materials"]

from .extension import Extension


def get_cae_materials() -> str:
    """
    Return the path to the CAE materials file.

    Returns
    -------
    str
        The path to the CAE materials file.
    """
    return "cae_materials.mdl"
