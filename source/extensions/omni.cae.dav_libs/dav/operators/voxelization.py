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
This module provides field voxelization to regular grids.

The voxelization operator samples a field onto a regular grid (image data) at specified
resolution and origin. It supports both regular array output and NanoVDB sparse volume output.
"""

from logging import getLogger
from typing import Any

import numpy as np
import warp as wp

import dav
from dav.data_models.vtk import image_data

logger = getLogger(__name__)


@dav.cached(aot="operators.voxelization")
def get_kernel(data_model: dav.DataModel, field_model_in: dav.FieldModel, interpolator: dav.InterpolatedFieldAPI, field_model_out: dav.FieldModel):
    input_dtype = type(field_model_in.FieldAPI.zero())
    output_dtype = type(field_model_out.FieldAPI.zero())
    static_cast = dav.utils.get_cast_function(input_dtype, output_dtype)

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_voxelize_kernel")
    def voxelize(
        ds: data_model.DatasetHandle, field_in: field_model_in.FieldHandle, image_ds: image_data.DatasetHandle, field_out: field_model_out.FieldHandle, bg_value: output_dtype
    ):
        out_point_idx = wp.tid()
        pos = image_data.DatasetAPI.get_point(image_ds, image_data.DatasetAPI.get_point_id_from_idx(image_ds, out_point_idx))
        cell = data_model.CellLocatorAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        if data_model.CellAPI.is_valid(cell):
            value = interpolator.get(ds, field_in, cell, pos)
            field_model_out.FieldAPI.set(field_out, out_point_idx, static_cast(value))
        else:
            field_model_out.FieldAPI.set(field_out, out_point_idx, bg_value)

    return voxelize


@dav.cached(aot="operators.voxelization_masked")
def get_kernel_with_mask(
    data_model: dav.DataModel, field_model_in: dav.FieldModel, interpolator: dav.InterpolatedFieldAPI, field_model_out: dav.FieldModel, mask_field_model: dav.FieldModel
):
    input_dtype = type(field_model_in.FieldAPI.zero())
    output_dtype = type(field_model_out.FieldAPI.zero())
    mask_dtype = type(mask_field_model.FieldAPI.zero())
    static_cast = dav.utils.get_cast_function(input_dtype, output_dtype)

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_voxelize_masked_kernel")
    def voxelize_masked(
        ds: data_model.DatasetHandle,
        field_in: field_model_in.FieldHandle,
        image_ds: image_data.DatasetHandle,
        field_out: field_model_out.FieldHandle,
        bg_value: output_dtype,
        mask_out: mask_field_model.FieldHandle,
    ):
        out_point_idx = wp.tid()
        pos = image_data.DatasetAPI.get_point(image_ds, image_data.DatasetAPI.get_point_id_from_idx(image_ds, out_point_idx))
        cell = data_model.CellLocatorAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        mask = mask_dtype(0)
        if data_model.CellAPI.is_valid(cell):
            value = interpolator.get(ds, field_in, cell, pos)
            field_model_out.FieldAPI.set(field_out, out_point_idx, static_cast(value))
            mask = mask_dtype(1)
        else:
            field_model_out.FieldAPI.set(field_out, out_point_idx, bg_value)
        mask_field_model.FieldAPI.set(mask_out, out_point_idx, mask)

    return voxelize_masked


def get_output_dtype(in_dtype, use_nanovdb: bool):
    """Determine the output dtype for voxelization based on the input field and whether NanoVDB is used."""
    if use_nanovdb:
        # NanoVDB only supports float32 and vec3f, so we need to convert if the input field is not already in a supported format.
        if dav.utils.is_vector_dtype(in_dtype):
            if dav.utils.get_vector_length(in_dtype) != 3:
                raise ValueError(f"Unsupported vector length {dav.utils.get_vector_length(in_dtype)} for NanoVDB voxelization. Only vec3 types are supported.")
            return wp.vec3f
        else:
            return wp.float32
    else:
        # If not using NanoVDB, we can keep the same dtype as the input field.
        return in_dtype


def _normalize_output_association(output_association: dav.AssociationType | str) -> dav.AssociationType:
    if isinstance(output_association, dav.AssociationType):
        if output_association in (dav.AssociationType.VERTEX, dav.AssociationType.CELL):
            return output_association
        raise ValueError(f"Unsupported voxelization output association: {output_association}")

    association_name = str(output_association)
    if association_name in ("point", "vertex"):
        return dav.AssociationType.VERTEX
    if association_name == "cell":
        return dav.AssociationType.CELL
    raise ValueError(f"Unsupported voxelization output association: {output_association}")


def compute(
    dataset: dav.DatasetLike,
    field_name: str,
    min_ijk: wp.vec3i,
    max_ijk: wp.vec3i,
    voxel_size: wp.vec3f,
    use_nanovdb: bool = True,
    bg_value: Any = None,
    output_field_name: str = "Volume",
    output_mask_field_name: str | None = None,
    output_association: dav.AssociationType | str = dav.AssociationType.CELL,
) -> dav.DatasetLike:
    """
    Voxelize a field into a dataset. The mapping of ijk extents to world coordinates is done using the same
    convention as OpenVDB/NanoVDB for cell-centered output. That is,

       pos_world = voxel_size * (ijk + 0.5)

    For point-centered output, samples are taken directly at grid points:

       pos_world = voxel_size * ijk

    Args:
        dataset: The dataset to voxelize.
        field_name: Name of the field to voxelize.
        min_ijk: The minimum ijk coordinates of the voxelization (inclusive).
        max_ijk: The maximum ijk coordinates of the voxelization (inclusive).
        voxel_size: The size of the voxels.
        use_nanovdb: Whether to use NanoVDB for the output field (True) or regular array (False).
        bg_value: The background value to use for the output field. If None, a default value is used based on the field dtype.
        output_field_name: Name for the voxelized output field (default: "Volume").
        output_mask_field_name (str | None): If provided, a mask field is generated with this name.
            Values are 1.0/0x01 for valid sample locations and 0.0/0x00 for invalid ones.
            When use_nanovdb=True, the mask is a float32 NanoVDB volume; otherwise it is a uint32 array.
        output_association: Use "cell" for cell-centered output or "point"/"vertex" for point-centered output.

    Returns:
        dav.DatasetLike: The voxelized dataset. Presently, we return a dataset that follows the VTK image data model.

    Note:
        NanoVDB fields always use 'ij' (Fortran/column-major) indexing where the first dimension varies fastest.

    Raises:
        KeyError: If the specified field is not found in the dataset.
    """
    output_association = _normalize_output_association(output_association)

    # Get field from dataset
    try:
        field_in = dataset.get_field(field_name)
    except KeyError:
        raise KeyError(f"Field '{field_name}' not found in dataset. Available fields: {dataset.get_field_names()}") from None

    target_dtype = get_output_dtype(field_in.dtype, use_nanovdb)
    if dav.utils.is_integral_dtype(field_in.dtype) and dav.utils.is_floating_point_dtype(target_dtype):
        # this happens with NanoVDB, so warn the user so they are aware.
        logger.warning(f"Converting non-floating point field dtype {field_in.dtype} to floating point dtype {target_dtype}.")

    device = dataset.device

    # Build cell locator if needed
    dataset.build_cell_locator()

    dims = max_ijk - min_ijk + 1
    assert dims.x > 0 and dims.y > 0 and dims.z > 0, "Dimensions must be positive"
    nb_samples = int(np.prod(dims).tolist())
    sample_offset = wp.vec3f(0.0) if output_association == dav.AssociationType.VERTEX else wp.vec3f(0.5)
    sample_origin = wp.cw_mul(sample_offset, voxel_size)

    # NanoVDB values are rendered/sampled as voxels centered on their index locations.
    # Cell-centered output already matches that convention in the current render path;
    # point-centered output needs the voxel body shifted back half a cell.
    volume_offset = wp.vec3f(-0.5) if output_association == dav.AssociationType.VERTEX else wp.vec3f(0.0)
    volume_translation = wp.cw_mul(volume_offset, voxel_size)

    field_in.get_range()  # pre-compute the range of the input field
    if bg_value is None:
        if dav.utils.is_vector_dtype(target_dtype):
            bg_value = target_dtype(0 if dav.utils.is_integral_dtype(field_in.dtype) else 0.0)
        else:
            # Create bg_value with the correct warp type to match target_dtype
            bg_value = target_dtype(int(field_in.get_range()[0])) if dav.utils.is_integral_dtype(field_in.dtype) else target_dtype(float(field_in.get_range()[0]))

    with dav.scoped_timer("voxelization.allocate_results"):
        # Create output field based on use_nanovdb flag
        if use_nanovdb:
            from dav.fields import nanovdb as field_nanovdb
            from dav.fields.nanovdb import allocate_nanovdb_volume

            volume_out = allocate_nanovdb_volume(min_ijk, max_ijk, voxel_size, bg_value, device, translation=volume_translation)
            field_out = dav.Field.from_volume(volume_out, dims=dims, association=dav.AssociationType.VERTEX, origin=min_ijk)
            result_field_out = dav.Field.from_volume(volume_out, dims=dims, association=output_association, origin=min_ijk)

            # Since NanoVDB has a bg-value, the range computation can often get messed up. So, we pre-compute the range
            # of the input field and copy it to the result field.
            result_field_out._range_cache = field_in._range_cache  # copy the range cache from the input field to the result field

            if output_mask_field_name is not None:
                mask_volume = allocate_nanovdb_volume(min_ijk, max_ijk, voxel_size, wp.float32(0.0), device, translation=volume_translation)
                mask_field = dav.Field.from_volume(mask_volume, dims=dims, association=dav.AssociationType.VERTEX, origin=min_ijk)
                result_mask_field = dav.Field.from_volume(mask_volume, dims=dims, association=output_association, origin=min_ijk)
                mask_field_model = field_nanovdb.get_field_model(wp.float32)
        else:
            from dav.fields import array as field_array

            # Create regular array field
            data_array = wp.zeros(nb_samples, dtype=field_in.dtype, device=device)
            field_out = dav.Field.from_array(data_array, dav.AssociationType.VERTEX)
            result_field_out = dav.Field.from_array(data_array, output_association)

            if output_mask_field_name is not None:
                mask_array = wp.zeros(nb_samples, dtype=wp.uint32, device=device)
                mask_field = dav.Field.from_array(mask_array, dav.AssociationType.VERTEX)
                result_mask_field = dav.Field.from_array(mask_array, output_association)
                mask_field_model = field_array.get_field_model_AoS(wp.uint32, 1)

        # now, we need to create a dataset where points give us the sample locations.
        # ideally, we add a custom data model for this, for now, we'll use VTK's.
        image_dataset_temp = image_data.create_dataset(origin=sample_origin, spacing=voxel_size, extent_min=min_ijk, extent_max=max_ijk, device=device)
        image_dataset_handle = image_dataset_temp.handle
        assert image_data.DatasetAPI.get_num_points(image_dataset_handle) == nb_samples

    with dav.scoped_timer("voxelization.get_kernel"):
        if output_mask_field_name is not None:
            kernel = get_kernel_with_mask(
                dataset.data_model, field_in.field_model, field_in.get_interpolated_field_api(dataset.data_model), field_out.field_model, mask_field_model
            )
        else:
            kernel = get_kernel(dataset.data_model, field_in.field_model, field_in.get_interpolated_field_api(dataset.data_model), field_out.field_model)
    with dav.scoped_timer("voxelization.launch", cuda_filter=wp.TIMING_ALL):
        if output_mask_field_name is not None:
            wp.launch(kernel, dim=nb_samples, inputs=[dataset.handle, field_in.handle, image_dataset_handle, field_out.handle, bg_value, mask_field.handle], device=device)
        else:
            wp.launch(kernel, dim=nb_samples, inputs=[dataset.handle, field_in.handle, image_dataset_handle, field_out.handle, bg_value], device=device)

    # For returned dataset, keep the existing cell-centered grid shape by default.
    # Point-centered output uses the IJK range as point extents instead.
    result_extent_max = max_ijk + wp.vec3i(1, 1, 1) if output_association == dav.AssociationType.CELL else max_ijk
    result_dataset = image_data.create_dataset(origin=wp.vec3f(0.0), spacing=voxel_size, extent_min=min_ijk, extent_max=result_extent_max, device=device)
    if output_association == dav.AssociationType.CELL:
        assert image_data.DatasetAPI.get_num_cells(result_dataset.handle) == nb_samples
    else:
        assert image_data.DatasetAPI.get_num_points(result_dataset.handle) == nb_samples
    result_dataset.add_field(output_field_name, result_field_out)
    if output_mask_field_name is not None:
        result_dataset.add_field(output_mask_field_name, result_mask_field)

    return result_dataset


if dav.config.compile_kernels_aot:
    from dav.core import aot
    from dav.fields import array as field_array
    from dav.fields import nanovdb as field_nanovdb
    from dav.fields import utils as field_utils

    for spec, kernel_fn, with_mask in [("operators.voxelization", get_kernel, False), ("operators.voxelization_masked", get_kernel_with_mask, True)]:
        for data_model in aot.get_data_models(specialization=f"{spec}.dataset"):
            for in_field_model in aot.get_field_models(specialization=spec):
                in_dtype = type(in_field_model.FieldAPI.zero())
                interpolator = field_utils.create_interpolated_field_api(data_model, in_field_model)
                for use_nanovdb in [True, False]:
                    try:
                        out_dtype = get_output_dtype(in_dtype, use_nanovdb)
                    except ValueError:
                        # skip unsupported configurations (e.g. vector types with unsupported lengths for NanoVDB)
                        continue

                    if use_nanovdb:
                        out_field_model = field_nanovdb.get_field_model(out_dtype)
                        mask_field_model_aot = field_nanovdb.get_field_model(wp.float32)
                    else:
                        out_scalar_dtype = dav.utils.get_scalar_dtype(out_dtype)
                        out_length = dav.utils.get_vector_length(out_dtype)
                        out_field_model = field_array.get_field_model_AoS(out_scalar_dtype, out_length)
                        mask_field_model_aot = field_array.get_field_model_AoS(wp.uint32, 1)

                    logger.info(f"Compiling {spec} kernels for data model: {data_model}, field model: {in_field_model}, use_nanovdb: {use_nanovdb}")
                    extra_args = (mask_field_model_aot,) if with_mask else ()
                    kernel = kernel_fn(data_model, in_field_model, interpolator, out_field_model, *extra_args)
                    wp.compile_aot_module(kernel.module, device=aot.get_devices())
