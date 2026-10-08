# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Warp-specific implementation helpers for built-in Points."""

from __future__ import annotations

import numpy as np
import warp as wp
import warp_simdata as simdata

from .._compute import gather_field


def point_dataset(source: simdata.Dataset, *, max_count: int, use_cell_points: bool) -> simdata.Dataset:
    """Convert source nodes and fields to a possibly decimated point cloud."""

    from warp_simdata.data_models.custom import point_cloud
    from warp_simdata.operators import node_element_counts, node_field, node_splats

    result = node_splats.compute(source, radius=0.0, sharpness=1.0)
    for name in source.get_field_names():
        converted = node_field.compute(source, name, output_field_name=name)
        result.add_field(name, converted.get_field(name), warn_if_exists=False)

    count = source.get_num_nodes()
    selected = None
    if use_cell_points:
        if source.get_num_elems() == 0:
            raise ValueError("useCellPoints requires a source dataset with cells")
        counts = node_element_counts.compute(source, field_name="_ovcae_element_count").get_field(
            "_ovcae_element_count"
        )
        selected = np.flatnonzero(counts.to_array().numpy() > 0).astype(np.int32, copy=False)
        count = len(selected)

    if max_count > 0 and count > max_count:
        stride = int(np.ceil(count / max_count))
        selected = np.arange(0, count, stride, dtype=np.int32) if selected is None else selected[::stride]

    if selected is None or len(selected) == source.get_num_nodes():
        return result

    indices = wp.array(selected, dtype=wp.int32, device=source.device)
    index_field = simdata.Field.from_array(indices, simdata.AssociationType.NODE)
    points = gather_field(
        simdata.Field.from_array(result.handle.points, simdata.AssociationType.NODE),
        index_field,
    )
    subset = point_cloud.create_dataset(points)
    for name in result.get_field_names():
        values = gather_field(result.get_field(name), index_field)
        subset.add_field(
            name,
            simdata.Field.from_array(values, simdata.AssociationType.NODE),
            warn_if_exists=False,
        )
    return subset
