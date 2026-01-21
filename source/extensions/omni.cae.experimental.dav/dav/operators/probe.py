# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
r"""
This module provides the probe operator for sampling field values at arbitrary positions.

The probe operator interpolates field values at specified probe point locations. The probe points
are provided as a dataset, and positions are extracted from that dataset's point locations.
"""

from logging import getLogger

import dav
import warp as wp

logger = getLogger(__name__)


@dav.cached
def get_kernel(
    data_model: dav.DataModel, field_model_in, interpolator, field_model_out, positions_data_model: dav.DataModel
):
    @wp.kernel(enable_backward=False)
    def probe(
        ds: data_model.DatasetHandle,
        field_in: field_model_in.FieldHandle,
        positions_ds: positions_data_model.DatasetHandle,
        field_out: field_model_out.FieldHandle,
    ):
        sample_idx = wp.tid()
        # Get position from positions dataset using its data model
        pt_id = positions_data_model.DatasetAPI.get_point_id_from_idx(positions_ds, sample_idx)
        pos = positions_data_model.DatasetAPI.get_point(positions_ds, pt_id)
        # wp.printf("\n\n***** Probing position %d: (%f, %f, %f)\n", sample_idx, pos.x, pos.y, pos.z)
        cell = data_model.DatasetAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        value = field_model_out.FieldAPI.zero()
        if data_model.CellAPI.is_valid(cell):
            # wp.printf("  Found cell id: %d\n", data_model.CellAPI.get_cell_id(cell))
            value = interpolator.get(ds, field_in, cell, pos)
        field_model_out.FieldAPI.set(field_out, sample_idx, value)

    return probe


def compute(
    dataset: dav.DatasetLike, field_name: str, positions: dav.DatasetLike, output_field_name: str = "probed_values"
) -> dav.DatasetLike:
    """
    Probe a field at given positions within the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name of the field to probe.
        positions (dav.DatasetLike): Dataset containing probe point positions.
        output_field_name (str): Name for the probed field (default: "probed_values").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the probed field values.
                 The field has NOT_SPECIFIED association as it corresponds to probe points.

    Raises:
        KeyError: If the specified field is not found in the dataset.
    """
    # Get field from dataset
    try:
        field_in = dataset.get_field(field_name)
    except KeyError:
        raise KeyError(
            f"Field '{field_name}' not found in dataset. Available fields: {list(dataset.fields.keys())}"
        ) from None

    # Get probe positions from the positions dataset
    nb_samples = positions.get_num_points()
    device = dataset.device

    # Build cell locator if needed
    with dav.scoped_timer("probe.build_cell_locator"):
        dataset.build_cell_locator()

    # Create output field (always AoS)
    with dav.scoped_timer("probe.allocate_results"):
        out_data = wp.zeros(nb_samples, dtype=field_in.dtype, device=device)
        field_out = dav.Field.from_array(out_data, dav.AssociationType.VERTEX)

    with dav.scoped_timer("probe.get_kernel"):
        interpolator = field_in.get_interpolated_field_api(dataset.data_model)
        kernel = get_kernel(
            dataset.data_model, field_in.field_model, interpolator, field_out.field_model, positions.data_model
        )

    with dav.scoped_timer("probe.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(
            kernel,
            dim=nb_samples,
            inputs=[dataset.handle, field_in.handle, positions.handle, field_out.handle],
            device=device,
        )

    result = positions.shallow_copy()
    result.add_field(output_field_name, field_out)
    return result
