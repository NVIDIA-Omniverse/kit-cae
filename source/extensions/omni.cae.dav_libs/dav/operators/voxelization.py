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


def _allocate_nanovdb_volume(min_ijk: wp.vec3i, max_ijk: wp.vec3i, voxel_size: wp.vec3f, bg_value: Any, device) -> wp.Volume:
    """`warp.Volume.allocate` is slow for large volumes. Hence we use allocate_by_tiles to allocate a volume."""
    tile_min = min_ijk / 8  # wp.vec3i, so this is integer division
    tile_max = max_ijk / 8  # wp.vec3i, so this is integer division
    tiles_shape = tile_max - tile_min + 1
    tiles = wp.array((np.indices(tiles_shape).reshape(3, -1).T + tile_min) * 8, dtype=wp.vec3i, device=device)
    return wp.Volume.allocate_by_tiles(tiles, voxel_size=voxel_size, bg_value=bg_value, device=device)


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


def compute(
    dataset: dav.DatasetLike,
    field_name: str,
    min_ijk: wp.vec3i,
    max_ijk: wp.vec3i,
    voxel_size: wp.vec3f,
    use_nanovdb: bool = True,
    bg_value: Any = None,
    output_field_name: str = "Volume",
) -> dav.DatasetLike:
    """
    Voxelize a field into a dataset. The mapping of ijk extents to world coordinates is done using the same
    convention as OpenVDB/NanoVDB. That is,

       pos_world = voxel_size * (ijk + 0.5)

    Args:
        dataset: The dataset to voxelize.
        field_name: Name of the field to voxelize.
        min_ijk: The minimum ijk coordinates of the voxelization (inclusive).
        max_ijk: The maximum ijk coordinates of the voxelization (inclusive).
        voxel_size: The size of the voxels.
        use_nanovdb: Whether to use NanoVDB for the output field (True) or regular array (False).
        bg_value: The background value to use for the output field. If None, a default value is used based on the field dtype.
        output_field_name: Name for the voxelized output field (default: "Volume").

    Returns:
        dav.DatasetLike: The voxelized dataset. Presently, we return a dataset that follows the VTK image data model.

    Note:
        NanoVDB fields always use 'ij' (Fortran/column-major) indexing where the first dimension varies fastest.

    Raises:
        KeyError: If the specified field is not found in the dataset.
    """
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
            volume_out = _allocate_nanovdb_volume(min_ijk, max_ijk, voxel_size, bg_value, device)
            field_out = dav.Field.from_volume(volume_out, dims=dims, association=dav.AssociationType.VERTEX, origin=min_ijk)
            result_field_out = dav.Field.from_volume(volume_out, dims=dims, association=dav.AssociationType.CELL, origin=min_ijk)

            # Since NanoVDB has a bg-value, the range compoutation can often get messed up. So, we pre-compute the range
            # of the input field and copy it to the result field.
            result_field_out._range_cache = field_in._range_cache  # copy the range cache from the input field to the result field
        else:
            # Create regular array field
            num_points = int(np.prod(dims))
            data_array = wp.zeros(num_points, dtype=field_in.dtype, device=device)
            field_out = dav.Field.from_array(data_array, dav.AssociationType.VERTEX)
            result_field_out = dav.Field.from_array(data_array, dav.AssociationType.CELL)

        # now, we need to create a dataset where the points in the dataset are centered at the voxel centers.
        # ideally, we add a custom data model for this, for now, we'll use VTK's.
        image_dataset_temp = image_data.create_dataset(
            origin=wp.cw_mul(wp.vec3f(min_ijk) + wp.vec3f(0.5), wp.vec3f(voxel_size)), spacing=voxel_size, extent_min=wp.vec3i(0, 0, 0), extent_max=max_ijk - min_ijk, device=device
        )
        image_dataset_handle = image_dataset_temp.handle
        assert image_data.DatasetAPI.get_num_points(image_dataset_handle) == np.prod(dims).tolist()

    with dav.scoped_timer("voxelization.get_kernel"):
        kernel = get_kernel(dataset.data_model, field_in.field_model, field_in.get_interpolated_field_api(dataset.data_model), field_out.field_model)
    with dav.scoped_timer("voxelization.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(kernel, dim=np.prod(dims).tolist(), inputs=[dataset.handle, field_in.handle, image_dataset_handle, field_out.handle, bg_value], device=device)

    # For returned dataset, we want to setup the grid so that the field is associated with cell-centers.
    result_dataset = image_data.create_dataset(origin=wp.vec3f(0.0), spacing=wp.vec3f(*voxel_size), extent_min=min_ijk, extent_max=max_ijk + wp.vec3i(1, 1, 1), device=device)
    assert image_data.DatasetAPI.get_num_cells(result_dataset.handle) == np.prod(dims).tolist()
    result_dataset.add_field(output_field_name, result_field_out)

    return result_dataset


if dav.config.compile_kernels_aot:
    from dav.core import aot
    from dav.fields import array as field_array
    from dav.fields import nanovdb as field_nanovdb
    from dav.fields import utils as field_utils

    dataset_data_models = aot.get_data_models(specialization="operators.voxelization.dataset")
    field_models = aot.get_field_models(specialization="operators.voxelization")

    for data_model in dataset_data_models:
        for in_field_model in field_models:
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
                else:
                    out_scalar_dtype = dav.utils.get_scalar_dtype(out_dtype)
                    out_length = dav.utils.get_vector_length(out_dtype)
                    out_field_model = field_array.get_field_model_AoS(out_scalar_dtype, out_length)

                # now compile the kernel.
                logger.info(f"Compiling kernels for data model: {data_model}, field model: {in_field_model}, use_nanovdb: {use_nanovdb}")
                kernel = get_kernel(data_model, in_field_model, interpolator, out_field_model)
                wp.compile_aot_module(kernel.module, device=aot.get_devices())
