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
from typing import Any

import warp as wp

import dav
from dav.core import aot

logger = getLogger(__name__)


@dav.cached
def create_interpolated_field_api(data_model: dav.DataModel, field_model: dav.FieldModel) -> type[dav.InterpolatedFieldAPI]:
    """Create an interpolated field API for a given data model and field model combination.

    This is an internal function used by Field and FieldCollection classes.

    This function generates a specialized InterpolatedFieldAPI class that works with
    the provided data model and field model. It handles both cell-associated and
    vertex-associated fields with proper interpolation.

    Args:
        data_model: The data model defining dataset operations
        field_model: The field model for field operations (may be a collection model)

    Returns:
        InterpolatedFieldAPI class with get() method for field interpolation
    """
    if hasattr(data_model.DatasetAPI, "create_interpolated_field_api"):
        # Defer to data model to create the interpolated field API, if the data model implements it.
        return data_model.DatasetAPI.create_interpolated_field_api(field_model)

    mul_f32 = dav.utils.get_mul_by_f32_function(type(field_model.FieldAPI.zero()))

    @dav.func
    def get_cell_value(ds: data_model.DatasetHandle, field: field_model.FieldHandle, cell: data_model.CellHandle):
        cell_id = data_model.CellAPI.get_cell_id(cell)
        cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(ds, cell_id)
        return field_model.FieldAPI.get(field, cell_idx)

    @dav.func
    def get_point_value(ds: data_model.DatasetHandle, field: field_model.FieldHandle, cell: data_model.CellHandle, position: wp.vec3f):
        nb_pts = data_model.CellAPI.get_num_points(cell, ds)
        value = field_model.FieldAPI.zero()
        weights = data_model.CellLocatorAPI.evaluate_position(ds, position, cell)
        assert nb_pts <= dav.config.max_points_per_cell, f"Number of points in cell ({nb_pts}) exceeds max_points_per_cell ({dav.config.max_points_per_cell})"

        for i in range(nb_pts):
            pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
            pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
            pt_value = field_model.FieldAPI.get(field, pt_idx)
            value += mul_f32(pt_value, weights[i])

        return value

    class GenericInterpolatedFieldMeta(type):
        def __repr__(self):
            return f"GenericInterpolatedFieldAPI[{data_model}, {field_model}]"

    class GenericInterpolatedFieldAPI(metaclass=GenericInterpolatedFieldMeta):
        @staticmethod
        @dav.func
        def get(ds: data_model.DatasetHandle, field: field_model.FieldHandle, cell: data_model.CellHandle, position: wp.vec3f):
            if not data_model.CellAPI.is_valid(cell):
                wp.printf("Cell is not valid\n")
                return field_model.FieldAPI.zero()
            elif field_model.FieldAPI.get_association(field) == wp.static(dav.AssociationType.VERTEX.value):
                return get_point_value(ds, field, cell, position)
            elif field_model.FieldAPI.get_association(field) == wp.static(dav.AssociationType.CELL.value):
                return get_cell_value(ds, field, cell)
            else:
                wp.printf("Unsupported association type: %d\n", field_model.FieldAPI.get_association(field))
                return field_model.FieldAPI.zero()

    return GenericInterpolatedFieldAPI


@dav.cached
def get_copy_field_kernel(field_model: dav.FieldModel, output_dtype: Any):
    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_copy_field_kernel")
    def copy_field_kernel(field: field_model.FieldHandle, result: wp.array(dtype=output_dtype)):
        idx = wp.tid()
        result[idx] = field_model.FieldAPI.get(field, idx)

    return copy_field_kernel


@dav.cached
def get_compute_field_range_kernel(field_model: dav.FieldModel):
    value_type = type(field_model.FieldAPI.zero())
    scalar_type = dav.utils.get_scalar_dtype(value_type)

    if dav.utils.is_vector_dtype(value_type):
        length = dav.utils.get_vector_length(value_type)

        @dav.func
        def get_magnitude(value: value_type) -> wp.float64:
            # For floating point vectors, convert to float64 and compute length
            if wp.static(scalar_type == wp.float64):
                # Already float64, just compute length
                return wp.length(value)
            elif wp.static(scalar_type == wp.float32):
                # Convert float32 components to float64
                return wp.float64(wp.length(value))
            else:
                f_value = wp.vec(dtype=wp.float64, length=length)
                for i in range(length):
                    f_value[i] = wp.float64(value[i])
                return wp.length(f_value)

        @dav.kernel(module="unique")
        @dav.utils.set_qualname("dav_compute_field_range_kernel")
        def compute_field_range_kernel(field: field_model.FieldHandle, component_ranges: wp.array(ndim=2, dtype=wp.float64), magnitude_range: wp.array(dtype=wp.float64)):
            assert component_ranges.shape[0] == length
            assert magnitude_range.shape[0] == 2

            idx = wp.tid()
            value = field_model.FieldAPI.get(field, idx)
            for i in range(length):
                # Convert component to float64 for storage
                comp_f64 = wp.float64(value[i])
                wp.atomic_min(component_ranges, i, 0, comp_f64)
                wp.atomic_max(component_ranges, i, 1, comp_f64)

            mag = get_magnitude(value)
            wp.atomic_min(magnitude_range, 0, mag)
            wp.atomic_max(magnitude_range, 1, mag)
    else:

        @dav.kernel(module="unique")
        @dav.utils.set_qualname("dav_compute_field_range_kernel")
        def compute_field_range_kernel(field: field_model.FieldHandle, component_ranges: wp.array(ndim=2, dtype=wp.float64), magnitude_range: wp.array(dtype=wp.float64)):
            idx = wp.tid()
            value = field_model.FieldAPI.get(field, idx)

            # Component range (actual value range) as float64
            value_f64 = wp.float64(value)
            wp.atomic_min(component_ranges, 0, 0, value_f64)
            wp.atomic_max(component_ranges, 0, 1, value_f64)

            # Not really used, but let's just store it if we
            # feel like we need it for some reason.
            # Magnitude range (absolute value range) as float64
            mag = wp.abs(wp.float64(value))
            wp.atomic_min(magnitude_range, 0, mag)
            wp.atomic_max(magnitude_range, 1, mag)

    return compute_field_range_kernel


if dav.config.compile_kernels_aot:
    for field_model in aot.get_field_models():
        kernel = get_compute_field_range_kernel(field_model)
        logger.info(f"Compiling field utility kernels for {field_model}")
        wp.compile_aot_module(kernel.module, device=aot.get_devices())

    # skipping `get_copy_field_kernel` for now since I am not
    # sure if we need to compile it
