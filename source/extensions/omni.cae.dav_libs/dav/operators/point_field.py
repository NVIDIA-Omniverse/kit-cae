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
This module provides operations for converting a field to a point-associated field.
"""

from collections import namedtuple
from logging import getLogger

import warp as wp

import dav
from dav.fields import array as fields_array

logger = getLogger(__name__)

PointFieldKernels = namedtuple("PointFieldKernels", ["accumulate", "average"])


@dav.cached(aot="operators.point_field")
def get_kernels(data_model: dav.DataModel, field_model_in: dav.FieldModel) -> PointFieldKernels:
    value_type = type(field_model_in.FieldAPI.zero())

    # Output is always AoS with the same scalar type and vector length as the input.
    field_model_out = fields_array.get_field_model_AoS(dav.utils.get_scalar_dtype(value_type), dav.utils.get_vector_length(value_type))
    div_i32 = dav.utils.get_div_by_i32_function(value_type)

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_point_field_accumulate_kernel")
    def cell_2_point_accumulate(ds: data_model.DatasetHandle, field_in: field_model_in.FieldHandle, field_sum: wp.array(dtype=value_type), field_count: wp.array(dtype=wp.int32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            val = field_model_in.FieldAPI.get(field_in, cell_idx)

            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)

                # Accumulate value atomically
                field_sum[pt_idx] += val  # this is atomic (see warp docs)

                # Increment count
                field_count[pt_idx] += 1  # this is atomic (see warp docs)

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_point_field_average_kernel")
    def cell_2_point_average(field_sum: wp.array(dtype=value_type), field_count: wp.array(dtype=wp.int32), field_out: field_model_out.FieldHandle):
        pt_idx = wp.tid()
        count = field_count[pt_idx]

        if count > 0:
            avg_val = div_i32(field_sum[pt_idx], count)
            field_model_out.FieldAPI.set(field_out, pt_idx, avg_val)
        else:
            field_model_out.FieldAPI.set(field_out, pt_idx, field_model_out.FieldAPI.zero())

    return PointFieldKernels(cell_2_point_accumulate, cell_2_point_average)


def compute(dataset: dav.DatasetLike, field_name: str, output_field_name: str = "point_field") -> dav.DatasetLike:
    """
    Convert a field to a point-associated field by averaging cell values.

    This operator converts cell-associated fields to point-associated fields by
    averaging the values of all cells that share each point. It uses atomic operations
    to accumulate values and counts, then performs division in a separate pass.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name of the input field (cell or point associated).
        output_field_name (str): Name for the output field (default: "point_field").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the converted field.
                 If input is already point-associated, returns it unchanged.

    Raises:
        KeyError: If the specified field is not found in the dataset.
    """
    # Get field from dataset
    field_in = dataset.get_field(field_name)
    if field_in is None:
        raise KeyError(f"Field '{field_name}' not found in dataset. Available fields: {dataset.get_field_names()}")

    # Create shallow copy to hold result
    result = dataset.shallow_copy()

    match field_in.association:
        case dav.AssociationType.CELL:
            # Convert cell field to point field
            device = dataset.device
            nb_cells = dataset.get_num_cells()
            nb_points = dataset.get_num_points()

            # Create output field (always AoS) with same value dtype as input
            with dav.scoped_timer("point_field.allocate_results"):
                out_data = wp.zeros(nb_points, dtype=field_in.dtype, device=device)
                field_out = dav.Field.from_array(out_data, dav.AssociationType.VERTEX)

                # Allocate temporary arrays for accumulation
                field_sum = wp.zeros(nb_points, dtype=field_in.dtype, device=device)
                field_count = wp.zeros(nb_points, dtype=wp.int32, device=device)

            with dav.scoped_timer("point_field.get_kernel"):
                kernels = get_kernels(dataset.data_model, field_in.field_model)

            with dav.scoped_timer("point_field.accumulate", cuda_filter=wp.TIMING_ALL):
                # First pass: accumulate values and counts using atomics
                wp.launch(kernels.accumulate, dim=nb_cells, inputs=[dataset.handle, field_in.handle, field_sum, field_count], device=device)

            with dav.scoped_timer("point_field.average", cuda_filter=wp.TIMING_ALL):
                # Second pass: compute averages
                wp.launch(kernels.average, dim=nb_points, inputs=[field_sum, field_count, field_out.handle], device=device)

            result.add_field(output_field_name, field_out, warn_if_exists=False)

        case dav.AssociationType.VERTEX:
            # Already point-associated, just add it
            result.add_field(output_field_name, field_in, warn_if_exists=False)

        case dav.AssociationType.NOT_SPECIFIED:
            raise ValueError("Input field association type is NOT_SPECIFIED is not supported for conversion to point-associated field.")
        case _:
            raise ValueError(f"Input field association type {field_in.association} is not supported for conversion to point-associated field.")

    return result


if dav.config.compile_kernels_aot:
    from dav.core import aot

    dataset_data_models = aot.get_data_models(specialization="operators.point_field.dataset")
    field_models = aot.get_field_models(specialization="operators.point_field")

    for data_model in dataset_data_models:
        for field_model in field_models:
            logger.info(f"Compiling kernels for data model: {data_model}, field model: {field_model}")
            for kernel in get_kernels(data_model, field_model):
                wp.compile_aot_module(kernel.module, device=aot.get_devices())
