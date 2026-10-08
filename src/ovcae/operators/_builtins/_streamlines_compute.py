# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Warp-specific implementation helpers for built-in Streamlines."""

import warp as wp
import warp_simdata as simdata


@simdata.cached
def _get_transform_points_kernel(data_model: simdata.DataModel):
    @wp.kernel(enable_backward=False, module="unique")
    def transform_points(
        dataset: data_model.DatasetHandle,
        matrix: wp.mat44,
        result: wp.array(dtype=wp.vec3f),
    ):
        tid = wp.tid()
        node_id = data_model.DatasetAPI.get_node_id_from_idx(dataset, tid)
        point = data_model.DatasetAPI.get_node_coords(dataset, node_id)
        transformed = matrix * wp.vec4f(point[0], point[1], point[2], 1.0)
        result[tid] = wp.vec3f(transformed[0], transformed[1], transformed[2])

    return transform_points


def transform_seed_points(seed_prim, dataset: simdata.Dataset):
    """Apply one captured seed prim's local transform to its point dataset."""

    import numpy as np
    from warp_simdata.data_models.custom import point_cloud

    from .._compute import local_transform_matrix

    matrix = local_transform_matrix(seed_prim)
    if np.array_equal(matrix, np.eye(4)):
        return dataset

    points = wp.empty(dataset.get_num_nodes(), dtype=wp.vec3f, device=dataset.device)
    matrix_value = wp.mat44(*matrix.astype(np.float32).reshape(-1).tolist())
    wp.launch(
        _get_transform_points_kernel(dataset.data_model),
        dim=dataset.get_num_nodes(),
        inputs=[dataset.handle, matrix_value, points],
        device=dataset.device,
    )
    return point_cloud.create_dataset(points)


@wp.kernel(enable_backward=False)
def _curve_phases(phases: wp.array(dtype=wp.float32)):
    index = wp.tid()
    state = wp.rand_init(1729, index)
    phases[index] = wp.randf(state)


def curve_phases(count: int, device):
    """Generate repeatable animation phases on-device, one per output curve.

    Phases depend on curve order, not evaluation time: repeated evaluations with
    the same curve ordering keep their offsets. Changed topology can reassign
    offsets to trajectories when their output indices change.
    """
    phases = wp.empty(count, dtype=wp.float32, device=device)
    if count:
        wp.launch(_curve_phases, dim=count, inputs=[phases], device=device)
    return phases
