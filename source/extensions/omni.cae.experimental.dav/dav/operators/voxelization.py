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

import dav
import numpy as np
import warp as wp
from dav.data_models.vtk.image_data import DataModel as image_data_model

logger = getLogger(__name__)


@dav.cached
def get_kernel(data_model: dav.DataModel, field_model_in, interpolator, field_model_out):
    @wp.kernel(enable_backward=False)
    def voxelize(
        ds: data_model.DatasetHandle,
        field_in: field_model_in.FieldHandle,
        image_ds: image_data_model.DatasetHandle,
        field_out: field_model_out.FieldHandle,
        bg_value: Any,
    ):
        out_point_idx = wp.tid()
        pos = image_data_model.DatasetAPI.get_point(
            image_ds, image_data_model.DatasetAPI.get_point_id_from_idx(image_ds, out_point_idx)
        )
        cell = data_model.DatasetAPI.find_cell_containing_point(ds, pos, data_model.CellAPI.empty())
        if data_model.CellAPI.is_valid(cell):
            # wp.printf("Position at index %d: (%f, %f, %f), cell: %d\n", out_point_idx, pos.x, pos.y, pos.z, data_model.CellAPI.get_cell_id(cell))
            value = interpolator.get(ds, field_in, cell, pos)
            field_model_out.FieldAPI.set(field_out, out_point_idx, value)
        else:
            field_model_out.FieldAPI.set(field_out, out_point_idx, bg_value)

    return voxelize


def compute(
    dataset: dav.DatasetLike,
    field_name: str,
    origin: wp.vec3f,
    dims: wp.vec3i,
    voxel_size: wp.vec3f,
    voxel_index_type: str = "ij",
    use_nanovdb: bool = True,
    output_field_name: str = "Volume",
) -> dav.DatasetLike:
    """
    Voxelize a field into a dataset.

    Args:
        dataset: The dataset to voxelize.
        field_name: Name of the field to voxelize.
        origin: The origin of the voxelization.
        dims: The dimensions of the voxelization.
        voxel_size: The size of the voxels.
        voxel_index_type: The type of voxel index to use (only used if use_nanovdb is True).
        use_nanovdb: Whether to use NanoVDB for the output field (True) or regular array (False).
        output_field_name: Name for the voxelized output field (default: "Volume").

    Returns:
        dav.DatasetLike: The voxelized dataset.

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

    device = dataset.device

    # Build cell locator if needed
    dataset.build_cell_locator()

    with dav.scoped_timer("voxelization.allocate_results"):
        # Create output field based on use_nanovdb flag
        bg_value = 0.0 if field_in.dtype == wp.float32 else wp.vec3f(0.0, 0.0, 0.0)
        if use_nanovdb:
            if field_in.dtype not in [wp.float32, wp.vec3f]:
                raise ValueError(f"Unsupported field dtype: {field_in.dtype} for NanoVDB voxelization")

            # use finite min float value for float32 and zero for vec3f
            origin_np = np.array(origin)
            voxel_size_np = np.array(voxel_size)
            min_np = np.around(origin_np / voxel_size_np).astype(np.int32)
            max_np = np.around((origin_np + (np.array(dims) - 1) * voxel_size_np) / voxel_size_np).astype(np.int32)

            # We use allocate_by_tiles since `warp.Volume.allocate()` is slow for large volumes.
            tile_min = min_np // 8
            tile_max = max_np // 8
            tiles_shape = tile_max - tile_min + 1
            tiles = wp.array((np.indices(tiles_shape).reshape(3, -1).T + tile_min) * 8, dtype=wp.vec3i, device=device)
            volume_out = wp.Volume.allocate_by_tiles(tiles, voxel_size_np, bg_value=bg_value, device=device)
            volume_out_origin = wp.vec3i(min_np.tolist())
            field_out = dav.Field.from_volume(
                volume_out,
                origin=volume_out_origin,
                dims=tuple(dims),
                association=dav.AssociationType.CELL,
                indexing=voxel_index_type,
            )
        else:
            # Create regular array field
            num_points = np.prod(dims).tolist()
            data_array = wp.zeros(num_points, dtype=field_in.dtype, device=device)
            field_out = dav.Field.from_array(data_array, dav.AssociationType.VERTEX)

        image_dataset_handle = image_data_model.DatasetHandle()
        image_dataset_handle.origin = wp.vec3f(*origin)
        image_dataset_handle.spacing = wp.vec3f(*voxel_size)
        image_dataset_handle.extent_min = wp.vec3i(0, 0, 0)
        image_dataset_handle.extent_max = wp.vec3i(dims - 1)
        assert image_data_model.DatasetAPI.get_num_points(image_dataset_handle) == np.prod(dims).tolist()

    with dav.scoped_timer("voxelization.get_kernel"):
        kernel = get_kernel(
            dataset.data_model,
            field_in.field_model,
            field_in.get_interpolated_field_api(dataset.data_model),
            field_out.field_model,
        )
    with dav.scoped_timer("voxelization.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(
            kernel,
            dim=np.prod(dims).tolist(),
            inputs=[dataset.handle, field_in.handle, image_dataset_handle, field_out.handle, bg_value],
            device=device,
        )

    # Create result dataset
    result_dataset_handle = image_data_model.DatasetHandle()
    result_dataset_handle.origin = wp.vec3f(*origin) - wp.vec3f(*voxel_size) / 2.0
    result_dataset_handle.spacing = wp.vec3f(*voxel_size)
    result_dataset_handle.extent_min = wp.vec3i(0, 0, 0)
    result_dataset_handle.extent_max = wp.vec3i(*dims)

    result_dataset = dav.Dataset(data_model=image_data_model, handle=result_dataset_handle, device=device)
    if use_nanovdb:
        result_field = dav.Field.from_volume(
            volume_out,
            origin=volume_out_origin,
            dims=tuple(dims),
            association=dav.AssociationType.VERTEX,
            indexing=voxel_index_type,
        )
    else:
        result_field = dav.Field.from_array(data_array, dav.AssociationType.CELL)

    result_dataset.add_field(output_field_name, result_field)

    return result_dataset
