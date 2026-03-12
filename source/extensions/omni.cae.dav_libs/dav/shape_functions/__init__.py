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
Shape Functions Package

This package provides shape functions for various FEM element types.
Each module implements:
  - compute_shape_functions: Evaluate shape functions at parametric coordinates
  - compute_shape_derivatives: Evaluate shape function derivatives
  - compute_parametric_coordinates: Find parametric coordinates for a world position
  - is_inside: Test if parametric coordinates are inside the element

All implementations follow VTK's canonical node ordering.

Module naming convention follows CGNS style with node count suffix:
  - hexa_8: 8-node hexahedron
  - tetra_4: 4-node tetrahedron
  - voxel_8: 8-node axis-aligned voxel (optimization of hexa_8)
  - pyra_5: 5-node pyramid
  - penta_6: 6-node pentahedron/prism
  - star_convex_polyhedron: Arbitrary polyhedron with star-convex shape functions
"""

# Element type constants
from .types import ELEMENT_TYPE_HEXA, ELEMENT_TYPE_NONE, ELEMENT_TYPE_PENTA, ELEMENT_TYPE_POLYHEDRON, ELEMENT_TYPE_PYRA, ELEMENT_TYPE_TETRA, ELEMENT_TYPE_VOXEL

__all__ = ["ELEMENT_TYPE_NONE", "ELEMENT_TYPE_TETRA", "ELEMENT_TYPE_PYRA", "ELEMENT_TYPE_PENTA", "ELEMENT_TYPE_HEXA", "ELEMENT_TYPE_VOXEL", "ELEMENT_TYPE_POLYHEDRON"]
