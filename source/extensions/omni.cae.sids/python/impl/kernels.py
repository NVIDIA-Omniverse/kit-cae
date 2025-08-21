# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from typing import Any

import warp as wp


@wp.struct
class Elements_t:
    """
    This struct is similar to SIDS Elements_t for unstructured dataset.
    However, since Warp imposes a limit on lengths of arrays, we can only support arrays
    with length <= (2**31 - 1). While that is a reasonable limit for number of elements,
    the `elementConnectivity` array for a polyhedral section can easily overrun that limit.
    In that case, the physical SIDS section is split into multiple with `elementStartOffsetShit`
    used to convert `elementStartOffset` value for an element to true offset.
    """

    elementType: wp.int8
    elementRange: wp.vec2i
    elementConnectivity: wp.array(dtype=wp.int32)
    elementStartOffset: wp.array(dtype=wp.int64)
    elementStartOffsetShift: wp.int64 = wp.int64(0)


ET_ElementTypeNull = wp.constant(0)
ET_ElementTypeUserDefined = wp.constant(1)
ET_NODE = wp.constant(2)
ET_BAR_2 = wp.constant(3)
ET_TRI_3 = wp.constant(5)
ET_QUAD_4 = wp.constant(7)
ET_TETRA_4 = wp.constant(10)
ET_PYRA_5 = wp.constant(12)
ET_PENTA_6 = wp.constant(14)
ET_HEXA_8 = wp.constant(17)
ET_MIXED = wp.constant(20)
ET_NGON_n = wp.constant(22)
ET_NFACE_n = wp.constant(23)


@wp.func
def is_uniform(elementType: wp.int8) -> bool:
    if elementType == ET_MIXED or elementType == ET_NGON_n or elementType == ET_NFACE_n:
        return False
    return True


@wp.func
def get_vertex_count(elementType: wp.int8) -> wp.int32:
    if elementType == ET_BAR_2:
        return 2
    if elementType == ET_TRI_3:
        return 3
    if elementType == ET_QUAD_4:
        return 4
    if elementType == ET_TETRA_4:
        return 4
    if elementType == ET_PYRA_5:
        return 5
    if elementType == ET_PENTA_6:
        return 6
    if elementType == ET_HEXA_8:
        return 8
    return 0


@wp.kernel
def compute_centroid_accumulate(
    section: Elements_t, coords: wp.array(dtype=Any), centers: wp.array(dtype=Any), counts: wp.array(dtype=wp.int32)
):

    tid = wp.tid()

    if is_uniform(section.elementType):
        vtx_count = get_vertex_count(section.elementType)
        e_start = tid * vtx_count
        e_end = e_start + vtx_count
    else:
        # this downcast is safe since we are limited to 2**31 - 1 ids
        # per section
        e_start = wp.int32(section.elementStartOffset[tid] - section.elementStartOffsetShift)
        e_end = wp.int32(section.elementStartOffset[tid + 1] - section.elementStartOffsetShift)

        if section.elementType == ET_MIXED:
            # skip interleaved element type info in connectivity array
            e_start = e_start + 1

    o_center = centers.dtype()
    o_count = wp.int32(0)
    for i in range(e_start, e_end):
        i_center = coords[section.elementConnectivity[i] - 1]
        o_center += i_center
        o_count += 1
    centers[tid] = o_center
    counts[tid] = o_count


@wp.overload
def compute_centroid_accumulate(
    section: Elements_t,
    coords: wp.array(dtype=wp.vec3f),
    centers: wp.array(dtype=wp.vec3f),
    counts: wp.array(dtype=wp.int32),
): ...


@wp.overload
def compute_centroid_accumulate(
    section: Elements_t,
    coords: wp.array(dtype=wp.vec3d),
    centers: wp.array(dtype=wp.vec3d),
    counts: wp.array(dtype=wp.int32),
): ...


# @wp.overload
# def compute_centroid_accumulate(section: Elements_t,
#                                 coords: wp.array(dtype=wp.vec3d),
#                                 centers: wp.array(dtype=wp.vec3f),
#                                 counts: wp.array(dtype=wp.int32)):
#     ...


@wp.kernel
def compute_centroid_accumulate_nface(
    section: Elements_t,
    ngon_section: Elements_t,
    coords: wp.array(dtype=Any),
    centers: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
):

    tid = wp.tid()

    # this downcast is safe since we are limited to 2**31 - 1 ids
    # per section
    e_start = wp.int32(section.elementStartOffset[tid] - section.elementStartOffsetShift)
    e_end = wp.int32(section.elementStartOffset[tid + 1] - section.elementStartOffsetShift)

    o_center = centers[tid]
    o_count = counts[tid]
    for face_idx in range(e_start, e_end):
        face_id = wp.abs(section.elementConnectivity[face_idx])  # we don't care about face orientation, so strip sign
        if face_id >= ngon_section.elementRange.x and face_id <= ngon_section.elementRange.y:
            f_start = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x]
                - ngon_section.elementStartOffsetShift
            )
            f_end = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x + 1]
                - ngon_section.elementStartOffsetShift
            )
            for node_idx in range(f_start, f_end):
                node_id = ngon_section.elementConnectivity[node_idx]
                o_center += coords[node_id - 1]
                o_count += wp.int32(1)
    centers[tid] = o_center
    counts[tid] = o_count


@wp.overload
def compute_centroid_accumulate_nface(
    section: Elements_t,
    ngon_section: Elements_t,
    coords: wp.array(dtype=wp.vec3f),
    centers: wp.array(dtype=wp.vec3f),
    counts: wp.array(dtype=wp.int32),
): ...


@wp.overload
def compute_centroid_accumulate_nface(
    section: Elements_t,
    ngon_section: Elements_t,
    coords: wp.array(dtype=wp.vec3d),
    centers: wp.array(dtype=wp.vec3d),
    counts: wp.array(dtype=wp.int32),
): ...


@wp.kernel
def compute_centroids_reduce(centers: wp.array(dtype=Any), counts: wp.array(dtype=wp.int32)):
    id = wp.tid()
    count = centers[id].x.dtype(counts[id])
    if count > 0.0:
        center = centers[id]
        centers[id] = centers.dtype(center.x / count, center.y / count, center.z / count)


@wp.overload
def compute_centroids_reduce(centers: wp.array(dtype=wp.vec3f), counts: wp.array(dtype=wp.int32)): ...


@wp.overload
def compute_centroids_reduce(centers: wp.array(dtype=wp.vec3d), counts: wp.array(dtype=wp.int32)): ...


@wp.kernel
def cell_2_point_data(
    section: Elements_t,
    counts: wp.array(dtype=wp.int32),
    nb_arrays: wp.int32,
    cell_array0: wp.array(dtype=Any),
    cell_array1: wp.array(dtype=Any),
    cell_array2: wp.array(dtype=Any),
    point_array0: wp.array(dtype=Any),
    point_array1: wp.array(dtype=Any),
    point_array2: wp.array(dtype=Any),
):

    cellId = wp.tid()
    if is_uniform(section.elementType):
        vtx_count = get_vertex_count(section.elementType)
        e_start = cellId * vtx_count
        e_end = e_start + vtx_count
    else:
        # this downcast is safe since we are limited to 2**31 - 1 ids
        # per section
        e_start = wp.int32(section.elementStartOffset[cellId] - section.elementStartOffsetShift)
        e_end = wp.int32(section.elementStartOffset[cellId + 1] - section.elementStartOffsetShift)

        if section.elementType == ET_MIXED:
            # skip interleaved element type info in connectivity array
            e_start = e_start + 1

    data0 = cell_array0[cellId + (section.elementRange.x - 1)]
    if nb_arrays >= 2:
        data1 = cell_array1[cellId + (section.elementRange.x - 1)]
    if nb_arrays >= 3:
        data2 = cell_array2[cellId + (section.elementRange.x - 1)]

    for i in range(e_start, e_end):
        ptId = section.elementConnectivity[i]
        if ptId < 1 or ptId > point_array0.shape[0]:
            wp.printf(
                "ERROR: invalid ptId: %d (< 1 or > %d). Is connectivity array valid?\n", ptId, point_array0.shape[0]
            )
        else:
            wp.atomic_add(point_array0, ptId - 1, data0)
            if nb_arrays >= 2:
                wp.atomic_add(point_array1, ptId - 1, data1)
            if nb_arrays >= 3:
                wp.atomic_add(point_array2, ptId - 1, data2)
            wp.atomic_add(counts, ptId - 1, wp.int32(1))


@wp.kernel
def cell_2_point_data_nface(
    section: Elements_t,
    ngon_section: Elements_t,
    counts: wp.array(dtype=wp.int32),
    nb_arrays: wp.int32,
    cell_array0: wp.array(dtype=Any),
    cell_array1: wp.array(dtype=Any),
    cell_array2: wp.array(dtype=Any),
    point_array0: wp.array(dtype=Any),
    point_array1: wp.array(dtype=Any),
    point_array2: wp.array(dtype=Any),
):

    tid = wp.tid()

    # this downcast is safe since we are limited to 2**31 - 1 ids
    # per section
    e_start = wp.int32(section.elementStartOffset[tid] - section.elementStartOffsetShift)
    e_end = wp.int32(section.elementStartOffset[tid + 1] - section.elementStartOffsetShift)

    cell_id = tid + section.elementRange.x
    # if cell_id < 1 or cell_id > cell_array0.shape[0]:
    #     wp.printf("ERROR: invalid cell_id: %d (< 1 or > %d). Is connectivity array valid?\n", cell_id, cell_array0.shape[0])
    #     return

    data0 = cell_array0[cell_id - 1]
    if nb_arrays >= 2:
        data1 = cell_array1[cell_id - 1]
    if nb_arrays >= 3:
        data2 = cell_array2[cell_id - 1]
    for face_idx in range(e_start, e_end):
        face_id = wp.abs(section.elementConnectivity[face_idx])  # we don't care about face orientation, so strip sign
        if face_id >= ngon_section.elementRange.x and face_id <= ngon_section.elementRange.y:
            f_start = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x]
                - ngon_section.elementStartOffsetShift
            )
            f_end = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x + 1]
                - ngon_section.elementStartOffsetShift
            )
            for node_idx in range(f_start, f_end):
                node_id = ngon_section.elementConnectivity[node_idx]
                # if node_id < 1 or node_id > point_array0.shape[0]:
                #     wp.printf("ERROR: invalid node_id: %d (< 1 or > %d). Is connectivity array valid?\n", node_id, point_array0.shape[0])
                #     return

                wp.atomic_add(point_array0, node_id - 1, data0)
                if nb_arrays >= 2:
                    wp.atomic_add(point_array1, node_id - 1, data1)
                if nb_arrays >= 3:
                    wp.atomic_add(point_array2, node_id - 1, data2)
                wp.atomic_add(counts, node_id - 1, wp.int32(1))


@wp.kernel
def cell_2_point_data_reduce_1d(point_array: wp.array(dtype=Any), counts: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    count = point_array.dtype(counts[tid])
    if count > 0.0:
        point_array[tid] = point_array[tid] / count


@wp.kernel
def cell_2_point_data_reduce_vec3f(point_array: wp.array(dtype=wp.vec3f), counts: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    count = wp.float32(counts[tid])
    if count > 0.0:
        point_array[tid] = point_array[tid] / count


@wp.kernel
def compute_pt_mask(section: Elements_t, mask: wp.array(dtype=wp.bool)):
    id = wp.tid()
    if is_uniform(section.elementType):
        vtx_count = get_vertex_count(section.elementType)
        e_start = id * vtx_count
        e_end = e_start + vtx_count
    else:
        # this downcast is safe since we are limited to 2**31 - 1 ids
        # per section
        e_start = wp.int32(section.elementStartOffset[id] - section.elementStartOffsetShift)
        e_end = wp.int32(section.elementStartOffset[id + 1] - section.elementStartOffsetShift)

        if section.elementType == ET_MIXED:
            # skip interleaved element type info in connectivity array
            e_start = e_start + 1

    for i in range(e_start, e_end):
        ptId = section.elementConnectivity[i]
        # wp.atomic_max(mask, ptId - 1, True)
        mask[ptId - 1] = True  # is this atomic enough? atomic_* is not supported for bool


@wp.kernel
def compute_pt_mask_nface(section: Elements_t, ngon_section: Elements_t, mask: wp.array(dtype=wp.bool)):
    id = wp.tid()
    # this downcast is safe since we are limited to 2**31 - 1 ids
    # per section
    e_start = wp.int32(section.elementStartOffset[id] - section.elementStartOffsetShift)
    e_end = wp.int32(section.elementStartOffset[id + 1] - section.elementStartOffsetShift)

    for face_idx in range(e_start, e_end):
        face_id = wp.abs(section.elementConnectivity[face_idx])  # we don't care about face orientation, so strip sign
        if face_id >= ngon_section.elementRange.x and face_id <= ngon_section.elementRange.y:
            f_start = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x]
                - ngon_section.elementStartOffsetShift
            )
            f_end = wp.int32(
                ngon_section.elementStartOffset[face_id - ngon_section.elementRange.x + 1]
                - ngon_section.elementStartOffsetShift
            )
            for node_idx in range(f_start, f_end):
                ptId = ngon_section.elementConnectivity[node_idx]
                # wp.atomic_max(mask, ptId - 1, True)
                mask[ptId - 1] = True  # is this atomic enough? atomic_* is not supported for bool
