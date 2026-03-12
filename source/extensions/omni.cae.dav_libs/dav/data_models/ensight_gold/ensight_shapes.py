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
EnSight Uniform Shape Definitions

This module defines topology and shape function metadata for uniform EnSight element types
(EN_point through EN_hexa20). It provides:
  - Element type constants
  - Shape data structures storing connectivity patterns and node orderings
  - API for querying element topology (nodes, faces, etc.)
  - Factory function to build the shape metadata
"""

from dav import shape_functions

# EnSight Gold element type codes
EN_invalid = -1
EN_point = 0
EN_bar2 = 1
EN_bar3 = 2
EN_tria3 = 3
EN_tria6 = 4
EN_quad4 = 5
EN_quad8 = 6
EN_tetra4 = 7
EN_tetra10 = 8
EN_pyramid5 = 9
EN_pyramid13 = 10
EN_penta6 = 11
EN_penta15 = 12
EN_hexa8 = 13
EN_hexa20 = 14
EN_nsided = 15
EN_nfaced = 16

# NOTE: All faces MUST be specified with outward facing normals (right-hand rule).
_ENSIGHT_CELL_SHAPES = [
    {"element_type": EN_invalid, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [], "element_mid_ids": [], "element_face_node_ids": []},
    {"element_type": EN_point, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [1], "element_mid_ids": [], "element_face_node_ids": []},
    {"element_type": EN_bar2, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [1, 2], "element_mid_ids": [], "element_face_node_ids": []},
    {"element_type": EN_bar3, "shape_function_type": shape_functions.ELEMENT_TYPE_NONE, "element_node_ids": [1, 3], "element_mid_ids": [2], "element_face_node_ids": []},
    {
        "element_type": EN_tria3,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [1, 2, 3],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 3]],
    },
    {
        "element_type": EN_tria6,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [1, 2, 3],
        "element_mid_ids": [4, 5, 6],
        "element_face_node_ids": [[1, 2, 3]],
    },
    {
        "element_type": EN_quad4,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [1, 2, 3, 4],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    {
        "element_type": EN_quad8,
        "shape_function_type": shape_functions.ELEMENT_TYPE_NONE,
        "element_node_ids": [1, 2, 3, 4],
        "element_mid_ids": [5, 6, 7, 8],
        "element_face_node_ids": [[1, 2, 3, 4]],
    },
    {
        "element_type": EN_tetra4,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids": [1, 2, 3, 4],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 3, 2], [1, 2, 4], [2, 3, 4], [3, 1, 4]],
    },
    {
        "element_type": EN_tetra10,
        "shape_function_type": shape_functions.ELEMENT_TYPE_TETRA,
        "element_node_ids": [1, 2, 3, 4],
        "element_mid_ids": [5, 6, 7, 8, 9, 10],
        "element_face_node_ids": [[1, 3, 2], [1, 2, 4], [2, 3, 4], [3, 1, 4]],
    },
    {
        "element_type": EN_pyramid5,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids": [1, 2, 3, 4, 5],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    {
        "element_type": EN_pyramid13,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PYRA,
        "element_node_ids": [1, 2, 3, 4, 5],
        "element_mid_ids": [6, 7, 8, 9, 10, 11, 12, 13, 14],
        "element_face_node_ids": [[1, 4, 3, 2], [1, 2, 5], [2, 3, 5], [3, 4, 5], [4, 1, 5]],
    },
    {
        "element_type": EN_penta6,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    {
        "element_type": EN_penta15,
        "shape_function_type": shape_functions.ELEMENT_TYPE_PENTA,
        "element_node_ids": [1, 2, 3, 4, 5, 6],
        "element_node_ids_vtk": [1, 3, 2, 4, 6, 5],
        "element_mid_ids": [7, 8, 9, 10, 11, 12, 13, 14, 15],
        "element_face_node_ids": [[1, 2, 5, 4], [2, 3, 6, 5], [3, 1, 4, 6], [1, 3, 2], [4, 5, 6]],
    },
    {
        "element_type": EN_hexa8,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": [],
        "element_face_node_ids": [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    {
        "element_type": EN_hexa20,
        "shape_function_type": shape_functions.ELEMENT_TYPE_HEXA,
        "element_node_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        "element_mid_ids": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "element_face_node_ids": [[1, 2, 6, 5], [2, 3, 7, 6], [3, 4, 8, 7], [4, 1, 5, 8], [1, 4, 3, 2], [5, 6, 7, 8]],
    },
    {"element_type": EN_nfaced, "shape_function_type": shape_functions.ELEMENT_TYPE_POLYHEDRON, "element_node_ids": [], "element_mid_ids": [], "element_face_node_ids": []},
]


_ELEMENT_TYPE_TO_STRING = {v: k[3:].lower() for k, v in globals().items() if k.startswith("EN_")}
_STRING_TO_ELEMENT_TYPE = {v: k for k, v in _ELEMENT_TYPE_TO_STRING.items()}


def get_element_type_as_string(element_type: int) -> str:
    """Return the string name of the given element type. For example EN_tetra4 -> 'tetra4'."""
    return _ELEMENT_TYPE_TO_STRING.get(element_type, f"unknown_{element_type}")


def get_element_type_from_string(name: str) -> int:
    """Return the element type integer for the given string name. For example 'tetra4' -> EN_tetra4.

    Raises:
        KeyError: If the string does not match any known element type.
    """
    try:
        return _STRING_TO_ELEMENT_TYPE[name.lower()]
    except KeyError:
        raise KeyError(f"Unknown element type string: '{name}'. Valid names: {sorted(_STRING_TO_ELEMENT_TYPE)}") from None


def supported_element_types():
    """Return a list of supported EnSight element types."""
    return [shape["element_type"] for shape in _ENSIGHT_CELL_SHAPES]


def is_supported_element_type(element_type: int) -> bool:
    """Check if the given element type is supported."""
    return element_type in supported_element_types()


def get_shapes_library(chosen_element_types: list[int] = None, default_element_type: int = EN_invalid):
    """
    Get a ShapeFunctionsAPI for the specified EnSight element types.

    Args:
        chosen_element_types: List of EnSight element type constants to include in the API (default: all supported types).
        default_element_type: Default element type to use for any unsupported types (default: EN_invalid)

    Returns:
        ShapeFunctionsAPI instance containing shape function metadata for the chosen element types.
    """
    from dav.shape_functions import utils as shape_functions_utils

    if chosen_element_types is None:
        chosen_element_types = supported_element_types()

    if chosen_element_types is None or len(chosen_element_types) == 0:
        raise ValueError("At least one element type must be specified for the shape functions API.")

    chosen_element_types = set(chosen_element_types)  # Remove duplicates
    if default_element_type in chosen_element_types:
        chosen_element_types.remove(default_element_type)

    # ensure default element type is included in the API at index 0
    chosen_element_types = [default_element_type] + list(chosen_element_types)

    # filter SIDS_SHAPES to include only the chosen element types
    shapes = [shape for shape in _ENSIGHT_CELL_SHAPES if shape["element_type"] in chosen_element_types]

    # need to convert all ids to 0-based for use in the API
    new_shapes = []
    for old_shape in shapes:
        new_shape = {
            "cell_type": old_shape["element_type"],
            "shape_function_type": old_shape["shape_function_type"],
            "element_node_ids": [id - 1 for id in old_shape["element_node_ids"]],
            "element_mid_ids": [id - 1 for id in old_shape["element_mid_ids"]],
            "element_face_node_ids": [[id - 1 for id in face] for face in old_shape["element_face_node_ids"]],
        }
        if "element_node_ids_vtk" in old_shape:
            new_shape["element_node_ids_vtk"] = [id - 1 for id in old_shape["element_node_ids_vtk"]]
        new_shapes.append(new_shape)

    return shape_functions_utils.build_shape_functions_library(new_shapes)
