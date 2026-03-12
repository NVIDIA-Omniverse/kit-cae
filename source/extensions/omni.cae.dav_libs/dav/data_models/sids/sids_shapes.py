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
SIDS Uniform Shape Definitions

This module defines topology and shape function metadata for uniform SIDS/CGNS element types.
It provides:
  - Element type constants
  - Shape data structures storing connectivity patterns and node orderings
  - API for querying element topology (nodes, faces, etc.)
  - Factory function to build the shape metadata

SIDS uses CGNS node ordering natively. For most elements, CGNS and VTK use the same ordering.
However, PENTA_6 requires reordering from CGNS to VTK convention for shape functions.
"""

from dav import shape_functions

# SIDS/CGNS element type constants (matching cgnslib.h)
ET_ElementTypeNull = 0
ET_ElementTypeUserDefined = 1
ET_NODE = 2
ET_BAR_2 = 3
ET_BAR_3 = 4
ET_TRI_3 = 5
ET_TRI_6 = 6
ET_QUAD_4 = 7
ET_QUAD_8 = 8
ET_QUAD_9 = 9
ET_TETRA_4 = 10
ET_TETRA_10 = 11
ET_PYRA_5 = 12
ET_PYRA_14 = 13
ET_PENTA_6 = 14
ET_PENTA_15 = 15
ET_PENTA_18 = 16
ET_HEXA_8 = 17
ET_HEXA_20 = 18
ET_HEXA_27 = 19
ET_MIXED = 20
ET_PYRA_13 = 21
ET_NGON_n = 22
ET_NFACE_n = 23
ET_BAR_4 = 24
ET_TRI_9 = 25
ET_TRI_10 = 26
ET_QUAD_12 = 27
ET_QUAD_16 = 28
ET_TETRA_16 = 29
ET_TETRA_20 = 30
ET_PYRA_21 = 31
ET_PYRA_29 = 32
ET_PYRA_30 = 33
ET_PENTA_24 = 34
ET_PENTA_38 = 35
ET_PENTA_40 = 36
ET_HEXA_32 = 37
ET_HEXA_56 = 38
ET_HEXA_64 = 39
ET_BAR_5 = 40
ET_TRI_12 = 41
ET_TRI_15 = 42
ET_QUAD_P4_16 = 43
ET_QUAD_25 = 44
ET_TETRA_22 = 45
ET_TETRA_34 = 46
ET_TETRA_35 = 47
ET_PYRA_P4_29 = 48
ET_PYRA_50 = 49
ET_PYRA_55 = 50
ET_PENTA_33 = 51
ET_PENTA_66 = 52
ET_PENTA_75 = 53
ET_HEXA_44 = 54
ET_HEXA_98 = 55
ET_HEXA_125 = 56

ET_NofValidElementTypes = 57


# NOTE: All faces MUST be specified with outward facing normals (right-hand rule).
# SIDS uses CGNS node ordering natively. For PENTA_6, we store both CGNS and VTK orderings.
# Face definitions use VTK ordering (after CGNS-to-VTK conversion for PENTA_6).
# All node IDs are specified using CGNS 1-based convention, then converted to 0-based before creating arrays.
# Shapes are ordered as they appear in CGNS header file (cgnslib.h)
SIDS_SHAPES = [
    # NULL (0)
    {
        "element_type": ET_ElementTypeNull,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [],
        "element_node_ids_vtk": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # USER_DEFINED (1)
    {
        "element_type": ET_ElementTypeUserDefined,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [],
        "element_node_ids_vtk": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # NODE (2)
    {
        "element_type": ET_NODE,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1],
        "element_node_ids_vtk": [1],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # BAR_2 (3)
    {
        "element_type": ET_BAR_2,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2],
        "element_node_ids_vtk": [1, 2],
        "element_mid_ids": [],
        "element_face_node_ids": [],  # 1D elements have no faces
    },
    # BAR_3 (4)
    {
        "element_type": ET_BAR_3,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2],
        "element_node_ids_vtk": [1, 2],
        "element_mid_ids": list(range(3, 4)),  # CGNS 1-based: 3
        "element_face_node_ids": [],
    },
    # TRI_3 (5)
    {
        "element_type": ET_TRI_3,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 3]],  # 2.5D element - 1 face (itself)
    },
    # TRI_6 (6)
    {
        "element_type": ET_TRI_6,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": list(range(4, 7)),  # CGNS 1-based: 4-6
        "element_face_node_ids": [[1, 2, 3]],
    },
    # QUAD_4 (7)
    {
        "element_type": ET_QUAD_4,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 3, 4]],  # 2.5D element - 1 face (itself)
    },
    # QUAD_8 (8)
    {
        "element_type": ET_QUAD_8,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 9)),  # CGNS 1-based: 5-8
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # QUAD_9 (9)
    {
        "element_type": ET_QUAD_9,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 10)),  # CGNS 1-based: 5-9
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # TETRA_4 (10)
    {
        "element_type": ET_TETRA_4,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # TETRA_10 (11)
    {
        "element_type": ET_TETRA_10,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 11)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # PYRA_5 (12)
    {
        "element_type": ET_PYRA_5,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PYRA_14 (13)
    {
        "element_type": ET_PYRA_14,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 15)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PENTA_6 (14)
    {
        "element_type": ET_PENTA_6,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_15 (15)
    {
        "element_type": ET_PENTA_15,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 16)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_18 (16)
    {
        "element_type": ET_PENTA_18,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 19)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # HEXA_8 (17)
    {
        "element_type": ET_HEXA_8,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_20 (18)
    {
        "element_type": ET_HEXA_20,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 21)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_27 (19)
    {
        "element_type": ET_HEXA_27,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 28)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # MIXED (20)
    {
        "element_type": ET_MIXED,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [],
        "element_node_ids_vtk": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # PYRA_13 (21)
    {
        "element_type": ET_PYRA_13,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 14)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # NGON_n (22)
    {
        "element_type": ET_NGON_n,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [],
        "element_node_ids_vtk": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # NFACE_n (23)
    {
        "element_type": ET_NFACE_n,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [],
        "element_node_ids_vtk": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    # BAR_4 (24)
    {
        "element_type": ET_BAR_4,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2],
        "element_node_ids_vtk": [1, 2],
        "element_mid_ids": list(range(3, 5)),  # CGNS 1-based: 3-4
        "element_face_node_ids": [],
    },
    # TRI_9 (25)
    {
        "element_type": ET_TRI_9,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": list(range(4, 10)),  # CGNS 1-based: 4-9
        "element_face_node_ids": [[1, 2, 3]],
    },
    # TRI_10 (26)
    {
        "element_type": ET_TRI_10,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": list(range(4, 11)),  # CGNS 1-based: 4-10
        "element_face_node_ids": [[1, 2, 3]],
    },
    # QUAD_12 (27)
    {
        "element_type": ET_QUAD_12,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 13)),  # CGNS 1-based: 5-12
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # QUAD_16 (28)
    {
        "element_type": ET_QUAD_16,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 17)),  # CGNS 1-based: 5-16
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # TETRA_16 (29)
    {
        "element_type": ET_TETRA_16,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 17)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # TETRA_20 (30)
    {
        "element_type": ET_TETRA_20,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 21)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # PYRA_21 (31)
    {
        "element_type": ET_PYRA_21,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 22)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PYRA_29 (32)
    {
        "element_type": ET_PYRA_29,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 30)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PYRA_30 (33)
    {
        "element_type": ET_PYRA_30,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 31)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PENTA_24 (34)
    {
        "element_type": ET_PENTA_24,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 25)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_38 (35)
    {
        "element_type": ET_PENTA_38,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 39)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_40 (36)
    {
        "element_type": ET_PENTA_40,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 41)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # HEXA_32 (37)
    {
        "element_type": ET_HEXA_32,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 33)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_56 (38)
    {
        "element_type": ET_HEXA_56,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 57)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_64 (39)
    {
        "element_type": ET_HEXA_64,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 65)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # BAR_5 (40)
    {
        "element_type": ET_BAR_5,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2],
        "element_node_ids_vtk": [1, 2],
        "element_mid_ids": list(range(3, 6)),  # CGNS 1-based: 3-5
        "element_face_node_ids": [],
    },
    # TRI_12 (41)
    {
        "element_type": ET_TRI_12,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": list(range(4, 13)),  # CGNS 1-based: 4-12
        "element_face_node_ids": [[1, 2, 3]],
    },
    # TRI_15 (42)
    {
        "element_type": ET_TRI_15,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3],
        "element_node_ids_vtk": [1, 2, 3],
        "element_mid_ids": list(range(4, 16)),  # CGNS 1-based: 4-15
        "element_face_node_ids": [[1, 2, 3]],
    },
    # QUAD_P4_16 (43)
    {
        "element_type": ET_QUAD_P4_16,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 17)),  # CGNS 1-based: 5-16
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # QUAD_25 (44)
    {
        "element_type": ET_QUAD_25,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 26)),  # CGNS 1-based: 5-25
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    # TETRA_22 (45)
    {
        "element_type": ET_TETRA_22,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 23)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # TETRA_34 (46)
    {
        "element_type": ET_TETRA_34,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 35)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # TETRA_35 (47)
    {
        "element_type": ET_TETRA_35,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids_cgns": [1, 2, 3, 4],
        "element_node_ids_vtk": [1, 2, 3, 4],
        "element_mid_ids": list(range(5, 36)),
        "element_face_node_ids": [[1, 2, 4], [2, 3, 4], [3, 1, 4], [1, 3, 2]],
    },
    # PYRA_P4_29 (48)
    {
        "element_type": ET_PYRA_P4_29,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 30)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PYRA_50 (49)
    {
        "element_type": ET_PYRA_50,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 51)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PYRA_55 (50)
    {
        "element_type": ET_PYRA_55,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5],
        "element_node_ids_vtk": [1, 2, 3, 4, 5],
        "element_mid_ids": list(range(6, 56)),
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    # PENTA_33 (51)
    {
        "element_type": ET_PENTA_33,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 34)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_66 (52)
    {
        "element_type": ET_PENTA_66,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 67)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # PENTA_75 (53)
    {
        "element_type": ET_PENTA_75,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": list(range(7, 76)),
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    # HEXA_44 (54)
    {
        "element_type": ET_HEXA_44,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 45)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_98 (55)
    {
        "element_type": ET_HEXA_98,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 99)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    # HEXA_125 (56)
    {
        "element_type": ET_HEXA_125,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids_cgns": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_node_ids_vtk": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": list(range(9, 126)),
        "element_face_node_ids": [[1, 5, 8, 4], [2, 3, 7, 6], [1, 2, 6, 5], [4, 8, 7, 3], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
]

_ELEMENT_TYPE_TO_STRING = {v: k[3:].lower() for k, v in globals().items() if k.startswith("ET_")}
_STRING_TO_ELEMENT_TYPE = {v: k for k, v in _ELEMENT_TYPE_TO_STRING.items()}


def get_element_type_as_string(element_type: int) -> str:
    """Return the string name of the given element type. For example ET_TETRA_4 -> 'tetra_4'."""
    return _ELEMENT_TYPE_TO_STRING.get(element_type, f"unknown_{element_type}")


def get_element_type_from_string(name: str) -> int:
    """Return the element type integer for the given string name. For example 'tetra_4' -> ET_TETRA_4.

    Raises:
        KeyError: If the string does not match any known element type.
    """
    try:
        return _STRING_TO_ELEMENT_TYPE[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown element type string: '{name}'. Valid names: {sorted(_STRING_TO_ELEMENT_TYPE)}") from None


def is_supported_sids_element_type(element_type: int) -> bool:
    """Check if the given element type is supported by SIDS."""
    return any(shape["element_type"] == element_type for shape in SIDS_SHAPES)


def get_supported_sids_element_types() -> list[int]:
    """Return a list of all supported SIDS element types."""
    return [shape["element_type"] for shape in SIDS_SHAPES]


def get_shapes_library(chosen_element_types: list[int] = None, default_element_type: int = ET_ElementTypeNull):
    """
    Build a shape functions API for the specified element types.

    Args:
        chosen_element_types: List of element types to include in the API (default: all supported types).
        default_element_type: The default element type to use as fallback (default: ET_ElementTypeNull).

    Returns:
       A class with static @wp.func methods for quering shape function properties by element type.
    """
    from dav.shape_functions import utils as shape_functions_utils

    if chosen_element_types is None:
        chosen_element_types = get_supported_sids_element_types()

    if len(chosen_element_types) == 0:
        raise ValueError("At least one element type must be specified for the shape functions API.")

    chosen_element_types = set(chosen_element_types)  # Remove duplicates
    if default_element_type in chosen_element_types:
        chosen_element_types.remove(default_element_type)

    # ensure default element type is included in the API at index 0
    chosen_element_types = [default_element_type] + list(chosen_element_types)

    # filter SIDS_SHAPES to include only the chosen element types
    shapes = [shape for shape in SIDS_SHAPES if shape["element_type"] in chosen_element_types]

    # need to convert all ids to 0-based for use in the API
    new_shapes = []
    for old_shape in shapes:
        new_shape = {
            "cell_type": old_shape["element_type"],
            "shape_function_type": old_shape["shape_function_type"],
            "element_node_ids": [id - 1 for id in old_shape["element_node_ids_cgns"]],
            "element_node_ids_vtk": [id - 1 for id in old_shape["element_node_ids_vtk"]],
            "element_mid_ids": [id - 1 for id in old_shape["element_mid_ids"]],
            "element_face_node_ids": [[id - 1 for id in face] for face in old_shape["element_face_node_ids"]],
        }
        new_shapes.append(new_shape)

    return shape_functions_utils.build_shape_functions_library(new_shapes)
