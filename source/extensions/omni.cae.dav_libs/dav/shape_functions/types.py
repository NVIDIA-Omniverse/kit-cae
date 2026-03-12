# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from . import hexa_8, penta_6, pyra_5, star_convex_polyhedron, tetra_4, voxel_8

# Element type constants
ELEMENT_TYPE_NONE = -1
ELEMENT_TYPE_TETRA = 0
ELEMENT_TYPE_PYRA = 1
ELEMENT_TYPE_PENTA = 2
ELEMENT_TYPE_HEXA = 3
ELEMENT_TYPE_VOXEL = 4
ELEMENT_TYPE_POLYHEDRON = 127


def get_shape_module(shape_function_type: int):
    if shape_function_type == ELEMENT_TYPE_HEXA:
        return hexa_8
    elif shape_function_type == ELEMENT_TYPE_PYRA:
        return pyra_5
    elif shape_function_type == ELEMENT_TYPE_PENTA:
        return penta_6
    elif shape_function_type == ELEMENT_TYPE_TETRA:
        return tetra_4
    elif shape_function_type == ELEMENT_TYPE_VOXEL:
        return voxel_8
    elif shape_function_type == ELEMENT_TYPE_POLYHEDRON:
        return star_convex_polyhedron
    else:
        raise ValueError(f"Unsupported shape function type: {shape_function_type}")


__all__ = [
    "ELEMENT_TYPE_NONE",
    "ELEMENT_TYPE_TETRA",
    "ELEMENT_TYPE_PYRA",
    "ELEMENT_TYPE_PENTA",
    "ELEMENT_TYPE_HEXA",
    "ELEMENT_TYPE_VOXEL",
    "ELEMENT_TYPE_POLYHEDRON",
    "get_shape_module",
]
