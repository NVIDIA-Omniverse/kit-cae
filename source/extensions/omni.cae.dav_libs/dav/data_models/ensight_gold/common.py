# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import warp as wp

from dav import locators


@wp.struct
class Piece:
    """
    Data structure representing an EnSight Gold piece.
    EnSight pieces store "counts" for items, however, we
    convert these to offsets for easier indexing.
    """

    element_type: wp.int32
    """Element type for the piece."""

    num_elements: wp.int32
    """Number of elements in the piece."""

    connectivity: wp.array(dtype=wp.int32)
    """Connectivity for the piece."""

    # for nsided (and as acceleration for nfaced):
    element_node_offsets: wp.array(dtype=wp.int32)
    """Offsets into connectivity for each element (for nsided elements).
       For nsided elements, this is provided by EnSight.

       Offsets into nfaced_connectivity for each element (for nfaced elements).
       For nfaced elements, we compute it to make it easier to identify all nodes for each element.
       In which case, these are offsets for the "nfaced_connectivity" array instead of the "connectivity" array.
    """

    # for nfaced:
    element_face_offsets: wp.array(dtype=wp.int32)
    """Offsets into face_node_offsets for each face (for nfaced elements)."""

    face_node_offsets: wp.array(dtype=wp.int32)
    """Offsets into connectivity for each face (for nfaced elements)."""

    # acceleration structure: to look up nfaced element nodes
    nfaced_connectivity: wp.array(dtype=wp.int32)
    """Connectivity for the piece (for nfaced elements)."""

    # -- acceleration structure for nfaced elements --
    element_centers: wp.array(dtype=wp.vec3f)
    """Centers of each element (for nfaced elements)."""

    element_face_centers: wp.array(dtype=wp.vec3f)
    """Centers of each face (for nfaced elements)."""

    element_face_signs: wp.array(dtype=wp.int8)
    """Sign of each face (for nfaced elements). Used to determine if face normal is inward (-1) or outward facing (+1)."""


@wp.struct
class DatasetHandle:  # AKA, Part in EnSight terminology
    """
    Data structure representing an EnSight Gold part.
    """

    points: wp.array(dtype=wp.vec3f)
    """Array of 3D point coordinates."""

    pieces: wp.array(dtype=Piece)
    """Array of pieces."""

    # -- acceleration structures --
    piece_offsets: wp.array(dtype=wp.int32)
    """
    Offsets into pieces array for each piece. Length is num_pieces + 1, with last element = len(pieces).
    Use piece_offsets[i+1] - piece_offsets[i] to get the number of elements in piece i.
    """

    num_elements: wp.int32
    """Total number of elements in the part."""

    # -- locators --
    cell_bvh_id: wp.uint64
    """BVH id for the cell locator. Building using cell bounding boxes. Used to find cells containing a point."""

    cell_links: locators.CellLinks
    """Cell links for the part. Used to look up cells by point id."""
