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
This module provides operations for converting a field to a cell-associated field.
"""

from logging import getLogger

import dav
import warp as wp

logger = getLogger(__name__)


@dav.cached
def get_kernel(data_model: dav.DataModel, field_model_in, field_model_out):
    @wp.kernel(enable_backward=False)
    def point_2_cell(
        ds: data_model.DatasetHandle, field_in: field_model_in.FieldHandle, field_out: field_model_out.FieldHandle
    ):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            acc = field_model_in.FieldAPI.zero()
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
                val = field_model_in.FieldAPI.get(field_in, pt_idx)
                acc += val
            avg_val = acc / wp.float32(num_points)
            field_model_out.FieldAPI.set(field_out, cell_idx, avg_val)
        else:
            field_model_out.FieldAPI.set(field_out, cell_idx, field_model_in.FieldAPI.zero())

    return point_2_cell


def compute(dataset: dav.DatasetLike, field_name: str, output_field_name: str = "cell_field") -> dav.DatasetLike:
    """
    Convert a field to a cell-associated field by averaging point values.

    Args:
        dataset (dav.DatasetLike): The dataset containing cells and points.
        field_name (str): Name of the input field (point or cell associated).
        output_field_name (str): Name for the output field (default: "cell_field").

    Returns:
        dav.DatasetLike: A new dataset (shallow copy) containing the converted field.
                 If input is already cell-associated, returns it unchanged.

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

    # Create shallow copy to hold result
    result = dataset.shallow_copy()

    match field_in.association:
        case dav.AssociationType.VERTEX:
            # Convert point field to cell field
            device = dataset.device
            nb_cells = dataset.get_num_cells()

            # Create output field (always AoS) with same value dtype as input
            with dav.scoped_timer("cell_field.allocate_results"):
                out_data = wp.zeros(nb_cells, dtype=field_in.dtype, device=device)
                field_out = dav.Field.from_array(out_data, dav.AssociationType.CELL)

            with dav.scoped_timer("cell_field.get_kernel"):
                # Get kernel with potentially different field models for input and output
                kernel = get_kernel(dataset.data_model, field_in.field_model, field_out.field_model)

            with dav.scoped_timer("cell_field.launch", cuda_filter=wp.TIMING_ALL):
                wp.launch(
                    kernel, dim=nb_cells, inputs=[dataset.handle, field_in.handle, field_out.handle], device=device
                )

            result.add_field(output_field_name, field_out)

        case dav.AssociationType.CELL:
            # Already cell-associated, just add it
            result.add_field(output_field_name, field_in)

        case dav.AssociationType.NOT_SPECIFIED:
            raise ValueError(
                "Input field association type is NOT_SPECIFIED is not supported for conversion to cell-associated field."
            )
        case _:
            raise ValueError(
                f"Input field association type {field_in.association} is not supported for conversion to cell-associated field."
            )

    return result
