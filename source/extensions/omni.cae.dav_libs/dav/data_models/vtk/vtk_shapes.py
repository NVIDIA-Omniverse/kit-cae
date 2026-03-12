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
VTK Uniform Shape Definitions

This module defines topology and shape function metadata for uniform VTK cell types.
It provides:
  - Element type constants
  - Shape data structures storing connectivity patterns and node orderings
  - API for querying element topology (nodes, faces, etc.)
  - Factory function to build the shape metadata

VTK uses its own canonical node ordering, which matches the ordering expected by
our shape function modules.
"""

__all__ = ["VTKCellTypes", "is_supported_vtk_cell_type", "get_supported_cell_types", "get_shapes_library", "get_cell_type_as_string", "get_cell_type_from_string"]


from dav import shape_functions


class VTKCellTypes:
    # Linear cells
    VTK_EMPTY_CELL = 0
    VTK_VERTEX = 1
    VTK_POLY_VERTEX = 2  # TODO: needs special handling
    VTK_LINE = 3
    VTK_POLY_LINE = 4  # TODO: needs special handling
    VTK_TRIANGLE = 5
    VTK_TRIANGLE_STRIP = 6  # TODO: needs special handling
    VTK_POLYGON = 7  # needs special handling
    VTK_PIXEL = 8
    VTK_QUAD = 9
    VTK_TETRA = 10
    VTK_VOXEL = 11
    VTK_HEXAHEDRON = 12
    VTK_WEDGE = 13
    VTK_PYRAMID = 14
    VTK_PENTAGONAL_PRISM = 15
    VTK_HEXAGONAL_PRISM = 16

    # Quadratic, isoparametric cells
    VTK_QUADRATIC_EDGE = 21
    VTK_QUADRATIC_TRIANGLE = 22
    VTK_QUADRATIC_QUAD = 23
    VTK_QUADRATIC_POLYGON = 36
    VTK_QUADRATIC_TETRA = 24
    VTK_QUADRATIC_HEXAHEDRON = 25
    VTK_QUADRATIC_WEDGE = 26
    VTK_QUADRATIC_PYRAMID = 27
    VTK_BIQUADRATIC_QUAD = 28
    VTK_TRIQUADRATIC_HEXAHEDRON = 29
    VTK_TRIQUADRATIC_PYRAMID = 37
    VTK_QUADRATIC_LINEAR_QUAD = 30
    VTK_QUADRATIC_LINEAR_WEDGE = 31
    VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32
    VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33
    VTK_BIQUADRATIC_TRIANGLE = 34

    # Cubic, isoparametric cell
    VTK_CUBIC_LINE = 35

    # Special class of cells formed by convex group of points
    VTK_CONVEX_POINT_SET = 41

    # Polyhedron cell (consisting of polygonal faces)
    VTK_POLYHEDRON = 42

    # Higher order cells in parametric form
    VTK_PARAMETRIC_CURVE = 51
    VTK_PARAMETRIC_SURFACE = 52
    VTK_PARAMETRIC_TRI_SURFACE = 53
    VTK_PARAMETRIC_QUAD_SURFACE = 54
    VTK_PARAMETRIC_TETRA_REGION = 55
    VTK_PARAMETRIC_HEX_REGION = 56

    # Higher order cells
    VTK_HIGHER_ORDER_EDGE = 60
    VTK_HIGHER_ORDER_TRIANGLE = 61
    VTK_HIGHER_ORDER_QUAD = 62
    VTK_HIGHER_ORDER_POLYGON = 63
    VTK_HIGHER_ORDER_TETRAHEDRON = 64
    VTK_HIGHER_ORDER_WEDGE = 65
    VTK_HIGHER_ORDER_PYRAMID = 66
    VTK_HIGHER_ORDER_HEXAHEDRON = 67

    # Arbitrary order Lagrange elements
    VTK_LAGRANGE_CURVE = 68
    VTK_LAGRANGE_TRIANGLE = 69
    VTK_LAGRANGE_QUADRILATERAL = 70
    VTK_LAGRANGE_TETRAHEDRON = 71
    VTK_LAGRANGE_HEXAHEDRON = 72
    VTK_LAGRANGE_WEDGE = 73
    VTK_LAGRANGE_PYRAMID = 74

    # Arbitrary order Bezier elements
    VTK_BEZIER_CURVE = 75
    VTK_BEZIER_TRIANGLE = 76
    VTK_BEZIER_QUADRILATERAL = 77
    VTK_BEZIER_TETRAHEDRON = 78
    VTK_BEZIER_HEXAHEDRON = 79
    VTK_BEZIER_WEDGE = 80
    VTK_BEZIER_PYRAMID = 81

    VTK_NUMBER_OF_CELL_TYPES = 82


_VTK_CELL_SHAPES = [
    {
        "cell_type": VTKCellTypes.VTK_EMPTY_CELL,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [],
        "element_mid_ids": [],
        "element_face_node_ids": [],
    },
    {"cell_type": VTKCellTypes.VTK_VERTEX, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [0], "element_mid_ids": [], "element_face_node_ids": []},
    {"cell_type": VTKCellTypes.VTK_LINE, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [0, 1], "element_mid_ids": [], "element_face_node_ids": []},
    {
        "cell_type": VTKCellTypes.VTK_TRIANGLE,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [0, 1, 2],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 1, 2]],
    },
    {"cell_type": VTKCellTypes.VTK_POLYGON, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [], "element_mid_ids": [], "element_face_node_ids": []},
    {
        "cell_type": VTKCellTypes.VTK_QUAD,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [0, 1, 2, 3],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 1, 2, 3]],
    },
    {
        "cell_type": VTKCellTypes.VTK_TETRA,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids": [0, 1, 2, 3],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]],
    },
    {
        "cell_type": VTKCellTypes.VTK_VOXEL,
        "shape_function_type": shape_functions.ELEMENT_TYPE_VOXEL,
        "element_node_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 2, 6, 4], [3, 1, 5, 7], [1, 0, 4, 5], [2, 3, 7, 6], [0, 1, 3, 2], [5, 4, 6, 7]],
    },
    {
        "cell_type": VTKCellTypes.VTK_HEXAHEDRON,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 4, 7, 3], [1, 2, 6, 5], [0, 1, 5, 4], [3, 7, 6, 2], [0, 3, 2, 1], [4, 5, 6, 7]],
    },
    {
        "cell_type": VTKCellTypes.VTK_WEDGE,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids": [0, 1, 2, 3, 4, 5],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0]],
    },
    {
        "cell_type": VTKCellTypes.VTK_PYRAMID,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids": [0, 1, 2, 3, 4],
        "element_mid_ids": [],
        "element_face_node_ids": [[0, 3, 2, 1], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
    },
    {
        "cell_type": VTKCellTypes.VTK_POLYHEDRON,
        "shape_function_type": shape_functions.ELEMENT_TYPE_POLYHEDRON,
        "element_node_ids": [],  # VTK polyhedron cell connectivity is defined by faces
        "element_mid_ids": [],
        "element_face_node_ids": [],  # VTK polyhedron cell connectivity is defined by faces
    },
]


_CELL_TYPE_TO_STRING = {v: k[4:].lower() for k, v in vars(VTKCellTypes).items() if k.startswith("VTK_")}
_STRING_TO_CELL_TYPE = {v: k for k, v in _CELL_TYPE_TO_STRING.items()}


def get_cell_type_as_string(cell_type: int) -> str:
    """Return the string name of the given VTK cell type. For example VTKCellTypes.VTK_TETRA -> 'tetra'."""
    return _CELL_TYPE_TO_STRING.get(cell_type, f"unknown_{cell_type}")


def get_cell_type_from_string(name: str) -> int:
    """Return the VTK cell type integer for the given string name. For example 'tetra' -> VTKCellTypes.VTK_TETRA.

    Raises:
        KeyError: If the string does not match any known cell type.
    """
    try:
        return _STRING_TO_CELL_TYPE[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown cell type string: '{name}'. Valid names: {sorted(_STRING_TO_CELL_TYPE)}") from None


def is_supported_vtk_cell_type(cell_type: int) -> bool:
    """
    Check if the given VTK cell type is supported by our shape functions.

    Args:
        cell_type: VTK cell type constant
    Returns:
        True if supported, False otherwise
    """
    return any(shape["cell_type"] == cell_type for shape in _VTK_CELL_SHAPES)


def get_supported_cell_types() -> list[int]:
    """
    Get a list of supported VTK cell types.

    Returns:
        List of VTK cell type constants that are supported
    """
    return [shape["cell_type"] for shape in _VTK_CELL_SHAPES]


def get_shapes_library(chosen_vtk_cell_types: list[int] = None, default_cell_type: int = VTKCellTypes.VTK_EMPTY_CELL):
    """
    Build a shape functions API for the specified VTK cell types.

    Args:
        chosen_vtk_cell_types: List of VTK cell type constants to include
        default_cell_type: Cell type to use as default/fallback (default: VTK_EMPTY_CELL).

    Returns:
        A class with static @wp.func methods for querying shape information.
    """
    from dav.shape_functions import utils as shape_functions_utils

    if chosen_vtk_cell_types is None:
        chosen_vtk_cell_types = get_supported_cell_types()

    # Filter shapes based on chosen cell types
    unique_cell_types = set(chosen_vtk_cell_types)
    if default_cell_type in unique_cell_types:
        unique_cell_types.remove(default_cell_type)

    # Ensure default_cell_type MUST be first
    chosen_cell_types = [default_cell_type] + list(unique_cell_types)
    shapes = [shape for shape in _VTK_CELL_SHAPES if shape["cell_type"] in chosen_cell_types]

    return shape_functions_utils.build_shape_functions_library(shapes)
