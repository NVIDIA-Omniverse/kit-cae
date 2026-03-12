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
EnSight Gold Utility Functions

This module provides utility functions for processing EnSight Gold data structures,
particularly for handling nfaced (polyhedral) elements.
"""

from logging import getLogger

import warp as wp

import dav

from .common import Piece

logger = getLogger(__name__)


# ============================================================================#
# Nfaced Connectivity Population
# ============================================================================#
def populate_nfaced_connectivity(piece: Piece):
    """Generate nfaced connectivity for a given piece and update it.

    This function extracts unique node connectivity for each nfaced element by merging
    the connectivity of all its constituent faces while removing duplicate nodes
    that are shared between faces.

    Algorithm:
    1. For each nfaced element, count the total number of unique nodes across all its faces
       (deduplicating nodes within each element).
    2. Convert element node counts to offsets using exclusive scan with trailing sum,
       establishing where each element's connectivity will be stored.
    3. For each nfaced element, populate the connectivity array with unique node IDs
       (deduplicating nodes as encountered across faces).

    Note: this function operates on a **partially filled** piece handle.

    Args:
        piece: Piece handle (partially filled with connectivity, element_face_offsets, face_node_offsets).
               The function will populate nfaced_connectivity and element_node_offsets.

    Returns:
        None: The piece is modified in-place.
    """
    nb_elements = piece.num_elements
    if nb_elements <= 0:
        # No elements, nothing to do - set empty arrays
        device = piece.connectivity.device
        piece.element_node_offsets = wp.zeros(shape=1, dtype=wp.int32, device=device)
        piece.nfaced_connectivity = wp.zeros(shape=0, dtype=wp.int32, device=device)
        return

    device = piece.connectivity.device

    # Validate that required arrays are present
    if piece.element_face_offsets is None or piece.element_face_offsets.shape[0] != nb_elements + 1:
        raise ValueError(f"element_face_offsets must have shape [{nb_elements + 1}]")
    if piece.face_node_offsets is None:
        raise ValueError("face_node_offsets must be provided for nfaced elements")

    # Step 1: For each nfaced element, count the unique nodes across all its faces.
    element_node_counts = wp.zeros(shape=nb_elements, dtype=wp.int32, device=device)
    wp.launch(_compute_unique_element_node_counts_kernel, dim=nb_elements, inputs=[piece], outputs=[element_node_counts], device=device)

    # Step 2: Convert counts to offsets using exclusive scan with trailing sum.
    element_node_offsets = wp.zeros(shape=nb_elements + 1, dtype=wp.int32, device=device)
    dav.utils.array_scan(element_node_counts, element_node_offsets, inclusive=False, add_trailing_sum=True)

    # Step 3: Build connectivity array with unique node IDs for each element.
    total_unique_nodes = element_node_offsets.numpy()[-1]
    nfaced_connectivity = wp.zeros(shape=int(total_unique_nodes), dtype=wp.int32, device=device)
    wp.launch(_compute_unique_element_node_connectivity_kernel, dim=nb_elements, inputs=[piece, element_node_offsets], outputs=[nfaced_connectivity], device=device)

    # Update the piece with the computed arrays
    piece.nfaced_connectivity = nfaced_connectivity
    piece.element_node_offsets = element_node_offsets


@dav.kernel
def _compute_unique_element_node_counts_kernel(piece: Piece, element_node_counts: wp.array(dtype=wp.int32)):
    """Compute the number of unique nodes for each element.

    This uses an insertion sort approach which is O(n^2) in the worst case, but since
    the number of nodes per element is typically small (< max_points_per_cell), this
    should be efficient enough.

    Args:
        piece: Piece handle containing element and face data.
        element_node_counts: Output array to store the number of unique nodes for each element.
    """
    element_idx = wp.tid()

    # Get the range of faces for this element
    face_start = piece.element_face_offsets[element_idx]
    face_end = piece.element_face_offsets[element_idx + 1]

    count = wp.int32(0)

    # Temporary array to store seen point IDs for this element
    seen_point_ids = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.int32)

    # Iterate over all faces of this element
    for face_offset in range(face_start, face_end):
        # Get the range of nodes for this face
        node_start = piece.face_node_offsets[face_offset]
        node_end = piece.face_node_offsets[face_offset + 1]

        # Iterate over all nodes in this face
        for node_offset in range(node_start, node_end):
            point_id = piece.connectivity[node_offset]

            # Check if we've already seen this point ID for this element
            already_seen = wp.bool(False)
            for seen_idx in range(count):
                if seen_point_ids[seen_idx] == point_id:
                    already_seen = True
                    break

            # If we haven't seen this point ID before, add it to the list
            if not already_seen:
                if count < dav.config.max_points_per_cell:
                    seen_point_ids[count] = point_id
                    count += 1
                else:
                    # Exceeded maximum points per cell - this is an error
                    wp.printf("ERROR: Element %d exceeded max_points_per_cell (%d)\n", element_idx, dav.config.max_points_per_cell)
                    break

    element_node_counts[element_idx] = count


@dav.kernel
def _compute_unique_element_node_connectivity_kernel(piece: Piece, element_node_offsets: wp.array(dtype=wp.int32), nfaced_connectivity: wp.array(dtype=wp.int32)):
    """Compute the unique element node connectivity for each element.

    This uses an insertion sort approach which is O(n^2) in the worst case, but since
    the number of nodes per element is typically small (< max_points_per_cell), this
    should be efficient enough.

    Args:
        piece: Piece handle containing element and face data.
        element_node_offsets: Array containing the start offset for each element's connectivity.
        nfaced_connectivity: Output array to store the unique node connectivity for all elements.
    """
    element_idx = wp.tid()

    # Get the output offset for this element
    output_offset = element_node_offsets[element_idx]
    output_end = element_node_offsets[element_idx + 1]

    # Get the range of faces for this element
    face_start = piece.element_face_offsets[element_idx]
    face_end = piece.element_face_offsets[element_idx + 1]

    count = wp.int32(0)

    # Iterate over all faces of this element
    for face_offset in range(face_start, face_end):
        # Get the range of nodes for this face
        node_start = piece.face_node_offsets[face_offset]
        node_end = piece.face_node_offsets[face_offset + 1]

        # Iterate over all nodes in this face
        for node_offset in range(node_start, node_end):
            point_id = piece.connectivity[node_offset]

            # Check if we've already seen this point ID for this element
            already_seen = wp.bool(False)
            for seen_idx in range(count):
                if nfaced_connectivity[output_offset + seen_idx] == point_id:
                    already_seen = True
                    break

            # If we haven't seen this point ID before, add it to the output
            if not already_seen:
                if output_offset + count < output_end:
                    nfaced_connectivity[output_offset + count] = point_id
                    count += 1
                else:
                    # This shouldn't happen if counts were computed correctly
                    wp.printf("ERROR: Output buffer overflow for element %d\n", element_idx)
                    break


# ============================================================================#
# Nfaced Element Face Sign Population
# ============================================================================#


def populate_nfaced_element_face_signs(piece: Piece, points: wp.array(dtype=wp.vec3f)):
    """Generate element face signs for nfaced elements to determine if face normals are inward or outward facing.

    This function computes a sign (+1 or -1) for each face of nfaced elements based on the order of the face connectivity.
    The sign indicates whether the face normal is outward facing (+1) or inward facing (-1).

    Note: this function operates on a **partially filled** piece handle and requires that nfaced_connectivity has already been populated.

    Args:
        piece: Piece handle (partially filled with connectivity, element_face_offsets, face_node_offsets, nfaced_connectivity).
               The function will populate element_face_signs.
        points: Array of point positions, used to compute face normals and centroids.
    Returns:
        None: The piece is modified in-place.
    """

    nb_elements = piece.num_elements
    if nb_elements <= 0:
        # No elements, nothing to do - set empty array
        device = piece.connectivity.device
        piece.element_face_signs = wp.zeros(shape=0, dtype=wp.int32, device=device)
        return

    device = piece.connectivity.device

    # Validate that required arrays are present
    if piece.element_face_offsets is None or piece.element_face_offsets.shape[0] != nb_elements + 1:
        raise ValueError(f"element_face_offsets must have shape [{nb_elements + 1}]")
    if piece.face_node_offsets is None:
        raise ValueError("face_node_offsets must be provided for nfaced elements")
    if piece.nfaced_connectivity is None:
        raise ValueError("nfaced_connectivity must be populated before computing element face signs")
    if piece.element_node_offsets is None:
        raise ValueError("element_node_offsets must be populated before computing element face signs")

    # Create output array for element face signs
    total_faces = piece.element_face_offsets[-1:].numpy().item()  # Total number of faces across all elements
    element_face_signs = wp.zeros(shape=total_faces, dtype=wp.int32, device=device)

    # Launch kernel to compute face signs for each face of nfaced elements
    wp.launch(_compute_nfaced_element_face_signs_kernel, dim=nb_elements, inputs=[piece, points], outputs=[element_face_signs], device=device)

    # Update the piece with the computed element face signs
    piece.element_face_signs = element_face_signs


@dav.kernel
def _compute_nfaced_element_face_signs_kernel(piece: Piece, points: wp.array(dtype=wp.vec3f), element_face_signs: wp.array(dtype=wp.int32)):
    """Compute the face sign for each face of nfaced elements.

    The algorithm is as follows:
    1. For each element, determine it's centroid by iterating over its unique nodes (from nfaced_connectivity) and averaging their positions.
    2. For each face of the element, compute the face normal using the right-hand rule based on the order of the face connectivity.
    3. Compute the vector from a point on the face to the element centroid.
    4. Compute the dot product between the face normal and the vector from a point on the face to the element centroid.
       - If the dot product is positive, the face normal is inward facing (-1).
       - If the dot product is negative, the face normal is outward facing (+1).
    """
    element_idx = wp.tid()

    # Get the range of faces for this element
    face_start = piece.element_face_offsets[element_idx]
    face_end = piece.element_face_offsets[element_idx + 1]

    # Step 1: Compute element centroid
    num_nodes = piece.element_node_offsets[element_idx + 1] - piece.element_node_offsets[element_idx]
    assert num_nodes > 0, "Element must have at least one node to compute centroid"
    centroid = wp.vec3(0.0)
    for node_idx in range(num_nodes):
        point_id = piece.nfaced_connectivity[piece.element_node_offsets[element_idx] + node_idx]
        assert point_id > 0, "Point IDs should be 1-based in EnSight Gold format"
        assert point_id - 1 < points.shape[0], "Point ID exceeds number of points in the dataset"

        point_pos = points[point_id - 1]
        centroid += point_pos
    centroid /= wp.float32(num_nodes)

    # Step 2-4: Compute face signs
    for face_offset in range(face_start, face_end):
        # Get the range of nodes for this face
        node_start = piece.face_node_offsets[face_offset]
        node_end = piece.face_node_offsets[face_offset + 1]

        # Compute face normal using right-hand rule (assuming counter-clockwise ordering of nodes)
        if node_end - node_start < 3:
            # Not enough nodes to define a plane, default to outward facing
            element_face_signs[face_offset] = 1
            continue

        p0 = points[piece.connectivity[node_start] - 1]
        p1 = points[piece.connectivity[node_start + 1] - 1]
        p2 = points[piece.connectivity[node_start + 2] - 1]

        edge1 = p1 - p0
        edge2 = p2 - p0
        face_normal = wp.cross(edge1, edge2)

        # Compute vector from face point to element centroid
        to_element_vector = centroid - p0

        # Compute dot product to determine sign
        dot_product = wp.dot(face_normal, to_element_vector)
        if dot_product >= 0:
            element_face_signs[face_offset] = -1  # Inward facing normal
        else:
            element_face_signs[face_offset] = 1  # Outward facing normal


if dav.config.compile_kernels_aot:
    from dav.core import aot

    logger.info("Compiling EnSight Gold utility kernels ...")
    wp.compile_aot_module(__name__, device=aot.get_devices())
