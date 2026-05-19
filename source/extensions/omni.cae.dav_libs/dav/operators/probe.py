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

import warp as wp

import dav

logger = getLogger(__name__)


@dav.cached(aot="operators.probe", aot_roles={"data_model": "dataset", "positions_data_model": "positions"})
def get_kernel(data_model: dav.DataModel, field_model_in, interpolator, field_model_out, positions_data_model: dav.DataModel):
    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_probe_kernel")
    def probe(ds: data_model.DatasetHandle, field_in: field_model_in.FieldHandle, positions_ds: positions_data_model.DatasetHandle, field_out: field_model_out.FieldHandle):
        sample_idx = wp.tid()
        # Get position from positions dataset using its data model
        pt_id = positions_data_model.DatasetAPI.get_point_id_from_idx(positions_ds, sample_idx)
        pos = positions_data_model.DatasetAPI.get_point(positions_ds, pt_id)
        # wp.printf("\n\n***** Probing position %d: (%f, %f, %f)\n", sample_idx, pos.x, pos.y, pos.z)
        cell = data_model.CellLocatorAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        value = field_model_out.FieldAPI.zero()
        if data_model.CellAPI.is_valid(cell):
            # wp.printf("  Found cell id: %d\n", data_model.CellAPI.get_cell_id(cell))
            value = interpolator.get(ds, field_in, cell, pos)
        field_model_out.FieldAPI.set(field_out, sample_idx, value)

    return probe


@dav.cached(aot="operators.probe_masked", aot_roles={"data_model": "dataset", "positions_data_model": "positions"})
def get_kernel_with_mask(data_model: dav.DataModel, field_model_in, interpolator, field_model_out, positions_data_model: dav.DataModel):
    from dav.fields import array as fields_array

    mask_field_model = fields_array.get_field_model_AoS(wp.uint32, 1)

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_probe_masked_kernel")
    def probe_masked(
        ds: data_model.DatasetHandle,
        field_in: field_model_in.FieldHandle,
        positions_ds: positions_data_model.DatasetHandle,
        field_out: field_model_out.FieldHandle,
        mask_out: mask_field_model.FieldHandle,
    ):
        sample_idx = wp.tid()
        pt_id = positions_data_model.DatasetAPI.get_point_id_from_idx(positions_ds, sample_idx)
        pos = positions_data_model.DatasetAPI.get_point(positions_ds, pt_id)
        cell = data_model.CellLocatorAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        value = field_model_out.FieldAPI.zero()
        mask = wp.uint32(0)
        if data_model.CellAPI.is_valid(cell):
            value = interpolator.get(ds, field_in, cell, pos)
            mask = wp.uint32(1)
        field_model_out.FieldAPI.set(field_out, sample_idx, value)
        mask_field_model.FieldAPI.set(mask_out, sample_idx, mask)

    return probe_masked


def compute(
    dataset: dav.DatasetLike, field_name: str, positions: dav.DatasetLike, output_field_name: str = "probed_values", output_mask_field_name: str | None = None
) -> dav.DatasetLike:
    """
    Probe a field at given positions within the dataset.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name of the field to probe.
        positions (dav.DatasetLike): Dataset containing probe point positions.
        output_field_name (str): Name for the probed field (default: "probed_values").
        output_mask_field_name (str | None): If provided, a uint32 mask field is generated
            with this name. Values are 0x01 for valid sample locations and 0x00 for invalid ones.

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
        raise KeyError(f"Field '{field_name}' not found in dataset. Available fields: {dataset.get_field_names()}") from None

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
        if output_mask_field_name is not None:
            mask_data = wp.zeros(nb_samples, dtype=wp.uint32, device=device)
            mask_field = dav.Field.from_array(mask_data, dav.AssociationType.VERTEX)

    with dav.scoped_timer("probe.get_kernel"):
        interpolator = field_in.get_interpolated_field_api(dataset.data_model)
        if output_mask_field_name is not None:
            kernel = get_kernel_with_mask(dataset.data_model, field_in.field_model, interpolator, field_out.field_model, positions.data_model)
        else:
            kernel = get_kernel(dataset.data_model, field_in.field_model, interpolator, field_out.field_model, positions.data_model)

    with dav.scoped_timer("probe.launch", cuda_filter=wp.TIMING_ALL):
        if output_mask_field_name is not None:
            wp.launch(kernel, dim=nb_samples, inputs=[dataset.handle, field_in.handle, positions.handle, field_out.handle, mask_field.handle], device=device)
        else:
            wp.launch(kernel, dim=nb_samples, inputs=[dataset.handle, field_in.handle, positions.handle, field_out.handle], device=device)

    result = positions.shallow_copy()
    result.add_field(output_field_name, field_out)
    if output_mask_field_name is not None:
        result.add_field(output_mask_field_name, mask_field)
    return result


if dav.config.compile_kernels_aot:
    from dav.core import aot
    from dav.fields import array as fields_array
    from dav.fields import utils as field_utils

    for spec, kernel_fn in [("operators.probe", get_kernel), ("operators.probe_masked", get_kernel_with_mask)]:
        for data_model in aot.get_data_models(specialization=f"{spec}.dataset"):
            for field_model in aot.get_field_models(specialization=spec):
                interpolator = field_utils.create_interpolated_field_api(data_model, field_model)

                # determine output field model for the selected input field.
                in_dtype = type(field_model.FieldAPI.zero())
                in_scalar_dtype = dav.utils.get_scalar_dtype(in_dtype)
                in_vector_length = dav.utils.get_vector_length(in_dtype)
                field_model_out = fields_array.get_field_model_AoS(in_scalar_dtype, in_vector_length)

                for positions_data_model in aot.get_data_models(specialization=f"{spec}.positions"):
                    logger.info(f"Compiling {spec} kernels for data model: {data_model}, field model: {field_model}, positions data model: {positions_data_model}")
                    kernel = kernel_fn(data_model, field_model, interpolator, field_model_out, positions_data_model)
                    wp.compile_aot_module(kernel.module, device=aot.get_devices())
