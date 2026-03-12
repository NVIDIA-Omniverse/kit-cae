# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
import warp as wp

import dav

from . import nface_n, sids_shapes

logger = getLogger(__name__)

AllShapesLibrary = sids_shapes.get_shapes_library()


# ================================================================================================================================
# Utility function to compute the element_start_offset array for a MIXED section based on the connectivity array.
# ================================================================================================================================
def compute_mixed_element_start_offset(element_range: wp.vec2i, element_connectivity: wp.array(dtype=wp.int32)) -> wp.array:
    """
    Compute the element_start_offset array for a MIXED section.

    Args:
        element_range (wp.vec2i): The range of element IDs (start, end).
        element_connectivity (wp.array): The connectivity array for the MIXED section, which includes element type IDs inline.

    Returns:
        wp.array: The computed element_start_offset array of shape (num_elements + 1,) with dtype wp.int32.
    """
    # SIDS element range is inclusive, so we add 1 to get the correct number of elements
    num_elements = element_range.y - element_range.x + 1
    start_offset = wp.zeros((num_elements + 1,), dtype=wp.int32, device=element_connectivity.device)

    wp.launch(_compute_mixed_element_start_offset_kernel, dim=1, inputs=[element_connectivity, num_elements], outputs=[start_offset], device=element_connectivity.device)

    assert start_offset[-1:].numpy()[0] == element_connectivity.shape[0], "Last value of start_offset must equal the length of element_connectivity"
    return start_offset


@dav.kernel
def _compute_mixed_element_start_offset_kernel(element_connectivity: wp.array(dtype=wp.int32), num_elements: wp.int32, start_offset: wp.array(dtype=wp.int32)):
    """
    Warp kernel to compute the element_start_offset array for a MIXED section.

    Args:
        element_connectivity (wp.array): The connectivity array for the MIXED section, which includes element type IDs inline.
        start_offset (wp.array): Output array to store the computed start offsets. Must be pre-allocated with length = num_elements + 1.
    """
    assert start_offset.shape[0] == num_elements + 1, "start_offset array must have length num_elements + 1"
    assert wp.tid() == 0, "This kernel should be launched with a single thread since it is not parallelized"

    offset = wp.int32(0)
    for elem_idx in range(num_elements):
        start_offset[elem_idx] = offset
        elem_type_id = element_connectivity[offset]
        num_nodes = AllShapesLibrary.get_num_all_nodes(elem_type_id)
        offset += 1 + num_nodes  # Move past the element type ID and its node connectivity
    start_offset[num_elements] = offset  # Final value is the total length of the connectivity array


# ================================================================================================================================
# Utility function to determine the unique element types present in a MIXED section based on the connectivity and start offsets.
# ================================================================================================================================
def get_element_types_in_mixed_section(element_range: wp.vec2i, element_connectivity: wp.array(dtype=wp.int32), element_start_offset: wp.array(dtype=wp.int32)) -> list:
    """
    Determine the unique element types present in a MIXED section based on the connectivity and start offsets.

    Args:
        element_range (wp.vec2i): The range of element IDs (start, end).
        element_connectivity (wp.array): The connectivity array for the MIXED section, which includes element type IDs inline.
        element_start_offset (wp.array): The start offset array for the MIXED section.

    Returns:
        list: A list of unique element type IDs present in the MIXED section.
    """
    # SIDS element range is inclusive, so we add 1 to get the correct number of elements
    num_elements = element_range.y - element_range.x + 1

    element_type_counts = wp.zeros(sids_shapes.ET_NofValidElementTypes, dtype=wp.int32, device=element_connectivity.device)
    wp.launch(
        _count_element_types_in_mixed_section_kernel, dim=num_elements, inputs=[element_connectivity, element_start_offset, element_type_counts], device=element_connectivity.device
    )

    unique_element_types = np.flatnonzero(element_type_counts.numpy()).tolist()  # Get the indices of non-zero counts, which correspond to element types present
    return unique_element_types


@dav.kernel
def _count_element_types_in_mixed_section_kernel(
    element_connectivity: wp.array(dtype=wp.int32), element_start_offset: wp.array(dtype=wp.int32), element_type_counts: wp.array(dtype=wp.int32)
):
    """
    Warp kernel to count the occurrences of each element type in a MIXED section.

    Args:
        element_connectivity (wp.array): The connectivity array for the MIXED section, which includes element type IDs inline.
        element_start_offset (wp.array): The start offset array for the MIXED section.
        element_type_counts (wp.array): Output array to store the counts of each element type. Must be pre-allocated with length equal to the number of valid element types.
    """
    elem_idx = wp.tid()
    element_type = element_connectivity[element_start_offset[elem_idx]]
    assert element_type >= 0 and element_type < element_type_counts.shape[0], "Invalid element type ID in connectivity array"
    if element_type >= 0 and element_type < element_type_counts.shape[0]:
        wp.atomic_add(element_type_counts, element_type, 1)  # Increment the count for this element type


# ================================================================================================================================
# Utility function to generate nface_n connectivity from a given dataset handle.
# ================================================================================================================================
def populate_nface_n_connectivity(ds: nface_n.DatasetHandle):
    """Generate nface_n connectivity for a given dataset handle and update it.

    This function extracts unique node connectivity for each nface element by merging
    the connectivity of all its constituent ngon faces while removing duplicate nodes
    that are shared between faces.

    Algorithm:
    1. For each nface element, count the total number of unique nodes across all its faces
       (deduplicating nodes within each element).
    2. Convert element node counts to offsets using exclusive scan with trailing sum,
       establishing where each element's connectivity will be stored.
    3. For each nface element, populate the connectivity array with unique node IDs
       (deduplicating nodes as encountered across faces).

    Note: this function operates on a **partially filled** dataset handle.

    Args:
        ds: Dataset handle (partially filled with nface_n_block and ngon_n_blocks).

    Returns:
        Dataset handle with nfaced_connectivity and nfaced_connectivity_offsets populated.
    """

    # Algorithm:
    # 1. For each nface element, count unique nodes across all its faces (deduplicating within each element).
    # 2. Convert element_node_counts to offsets (element_node_offsets) with trailing sum.
    # 3. Generate flattened element_node_connectivity array with unique nodes per element (deduplicating within each element).

    nb_elements = ds.nface_n_block.element_range.y - ds.nface_n_block.element_range.x + 1
    assert nb_elements > 0, "nb_elements must be > 0"

    device = ds.nface_n_block.grid_coords.device

    # Step 1: For each nface element, count the unique nodes across all its faces.
    element_node_counts = wp.zeros(shape=nb_elements, dtype=wp.int32, device=device)
    wp.launch(_compute_unique_element_node_counts_kernel, dim=nb_elements, inputs=[ds], outputs=[element_node_counts], device=device)

    # Step 2: Convert counts to offsets using exclusive scan with trailing sum.
    element_node_offsets = wp.zeros(shape=nb_elements + 1, dtype=wp.int32, device=device)
    dav.utils.array_scan(element_node_counts, element_node_offsets, inclusive=False, add_trailing_sum=True)

    # Step 3: Build connectivity array with unique node IDs for each element.
    element_node_connectivity = wp.zeros(shape=element_node_offsets[-1:].numpy()[0].item(), dtype=wp.int32, device=device)
    wp.launch(_compute_unique_element_node_connectivity_kernel, dim=nb_elements, inputs=[ds, element_node_offsets], outputs=[element_node_connectivity], device=device)

    ds.nface_n_connectivity = element_node_connectivity
    ds.nface_n_connectivity_offsets = element_node_offsets
    return True


@dav.kernel
def _compute_unique_element_node_counts_kernel(ds: nface_n.DatasetHandle, element_node_counts: wp.array(dtype=wp.int32)):
    """Compute the number of unique nodes for each element by counting the number of nodes in each face of the element.

    This uses a insertion sort which is O(n^2) in the worst case, but since the number of faces per element is typically small,
    this should be efficient enough for our use case.

    Args:
        ds: Dataset handle.
        element_node_counts: Array to store the number of nodes for each element.
    """
    nface_idx = wp.tid()
    nface_cell_id = nface_n.DatasetAPI.get_cell_id_from_idx(ds, nface_idx)
    nface_cell = nface_n.DatasetAPI.get_cell(ds, nface_cell_id)

    count = wp.int32(0)

    # Temporary array to store seen point IDs for this cell
    seen_point_ids = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.int32)

    num_faces = nface_n.CellAPI.get_num_faces(nface_cell, ds)
    for face_idx in range(num_faces):
        num_points = nface_n.CellAPI.get_face_num_points(nface_cell, face_idx, ds)
        for point_idx in range(num_points):
            point_id = nface_n.CellAPI.get_face_point_id(nface_cell, face_idx, point_idx, ds)

            # Check if we've already seen this point ID for this cell
            already_seen = wp.bool(False)
            for seen_idx in range(count):
                if seen_point_ids[seen_idx] == point_id:
                    already_seen = True
                    break

            # If we haven't seen this point ID before, add it to the list and increment the count
            if not already_seen:
                assert count < dav.config.max_points_per_cell, "Exceeded maximum points per cell"
                if count < dav.config.max_points_per_cell:  # Ensure we don't exceed the max points per cell
                    seen_point_ids[count] = point_id
                    count += 1
                else:
                    break

    element_node_counts[nface_idx] = count


@dav.kernel
def _compute_unique_element_node_connectivity_kernel(
    ds: nface_n.DatasetHandle, element_node_offsets: wp.array(dtype=wp.int32), element_node_connectivity: wp.array(dtype=wp.int32)
):
    """Compute the unique element node connectivity for each element by iterating over the faces of the element and storing the unique point IDs.

    This uses a insertion sort which is O(n^2) in the worst case, but since the number of faces per element is typically small,
    this should be efficient enough for our use case.

    Args:
        ds: Dataset handle.
        element_node_offsets: Array containing the start offset for each element's connectivity in the output array.
        element_node_connectivity: Output array to store the unique node connectivity for all elements. Must be pre-allocated with length equal to the total number of unique nodes across all elements.
    """
    nface_idx = wp.tid()
    nface_cell_id = nface_n.DatasetAPI.get_cell_id_from_idx(ds, nface_idx)
    nface_cell = nface_n.DatasetAPI.get_cell(ds, nface_cell_id)

    offset = element_node_offsets[nface_idx]

    count = wp.int32(0)

    num_faces = nface_n.CellAPI.get_num_faces(nface_cell, ds)
    for face_idx in range(num_faces):
        num_points = nface_n.CellAPI.get_face_num_points(nface_cell, face_idx, ds)
        for point_idx in range(num_points):
            point_id = nface_n.CellAPI.get_face_point_id(nface_cell, face_idx, point_idx, ds)

            # Check if we've already seen this point ID for this cell
            already_seen = wp.bool(False)
            for seen_idx in range(count):
                # TODO: check if it'd be faster to use a wp.vec for seen_point_ids here.
                if element_node_connectivity[offset + seen_idx] == point_id:
                    already_seen = True
                    break

            # If we haven't seen this point ID before, add it to the list and increment the count
            if not already_seen:
                assert count < dav.config.max_points_per_cell, "Exceeded maximum points per cell"
                assert offset + count < element_node_offsets[nface_idx + 1], "Output connectivity array overflow."
                element_node_connectivity[offset + count] = point_id
                count += 1


if dav.config.compile_kernels_aot:
    from dav.core import aot

    logger.info("Compiling SIDS utility kernels ...")
    wp.compile_aot_module(__name__, device=aot.get_devices())
