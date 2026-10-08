# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Private compute implementation for public operator utilities."""

from typing import Any

import warp as wp
import warp_simdata as simdata


@wp.kernel(enable_backward=False)
def _copy_offset_indices(
    source: wp.array(dtype=wp.int32),
    destination: wp.array(dtype=wp.int32),
    destination_offset: wp.int32,
    point_offset: wp.int32,
):
    index = wp.tid()
    destination[destination_offset + index] = source[index] + point_offset


@wp.kernel(enable_backward=False)
def _transform_points(
    source: wp.array(dtype=wp.vec3f),
    matrix: wp.mat44,
    destination: wp.array(dtype=wp.vec3f),
):
    index = wp.tid()
    point = source[index]
    transformed = matrix * wp.vec4f(point[0], point[1], point[2], 1.0)
    destination[index] = wp.vec3f(transformed[0], transformed[1], transformed[2])


@simdata.cached
def _get_extract_elements_kernel(field_model: simdata.FieldModel, indices_model: simdata.FieldModel, dtype: Any):
    @wp.kernel(enable_backward=False, module="unique")
    def extract_elements_kernel(
        field: field_model.FieldHandle,
        indices: indices_model.FieldHandle,
        result: wp.array(dtype=dtype),
    ):
        tid = wp.tid()
        index = indices_model.FieldAPI.get(indices, tid)
        result[tid] = field_model.FieldAPI.get(field, index)

    return extract_elements_kernel


def gather_field(field: simdata.Field, indices: simdata.Field) -> wp.array:
    """Gather a SimData field without forcing either field through host memory."""

    result = wp.empty(indices.size, dtype=field.dtype, device=field.device)
    kernel = _get_extract_elements_kernel(field.field_model, indices.field_model, field.dtype)
    wp.launch(
        kernel,
        dim=indices.size,
        inputs=[field.handle, indices.handle, result],
        device=field.device,
    )
    return result


def concatenate_arrays(arrays):
    """Concatenate same-typed one-dimensional Warp arrays on their device."""

    if not arrays:
        raise ValueError("at least one array is required")
    result = wp.empty(sum(array.shape[0] for array in arrays), dtype=arrays[0].dtype, device=arrays[0].device)
    offset = 0
    for array in arrays:
        wp.copy(result, array, dest_offset=offset, count=array.shape[0])
        offset += array.shape[0]
    return result


def merge_surface_geometry(surfaces):
    """Merge surface geometry without routing arrays through host memory."""

    points = concatenate_arrays([surface.handle.points for surface in surfaces])
    counts = concatenate_arrays([surface.handle.face_vertex_counts for surface in surfaces])
    indices = wp.empty(
        sum(surface.handle.face_vertex_indices.shape[0] for surface in surfaces),
        dtype=wp.int32,
        device=points.device,
    )
    point_offset = 0
    index_offset = 0
    for surface in surfaces:
        source = surface.handle.face_vertex_indices
        wp.launch(
            _copy_offset_indices,
            dim=source.shape[0],
            inputs=[source, indices, index_offset, point_offset],
            device=points.device,
        )
        point_offset += surface.handle.points.shape[0]
        index_offset += source.shape[0]
    return points, counts, indices


def transform_points(points, matrix):
    """Transform a vec3f point array without routing it through host memory."""

    import numpy as np

    matrix = np.asarray(matrix, dtype=np.float32).reshape(4, 4)
    result = wp.empty(points.shape[0], dtype=wp.vec3f, device=points.device)
    wp.launch(
        _transform_points,
        dim=points.shape[0],
        inputs=[points, wp.mat44(*matrix.reshape(-1).tolist()), result],
        device=points.device,
    )
    return result


def local_transform_matrix(prim):
    """Read OVStage's composed local matrix in our column-vector convention.

    Population and direct producers author omni:xform in USD's row-vector
    convention. Transpose once at this boundary; absent transforms are identity.
    """
    import numpy as np

    value = prim.get_attribute("omni:xform").get()
    return np.eye(4, dtype=np.float64) if value is None else np.asarray(value, dtype=np.float64).reshape(4, 4).T
