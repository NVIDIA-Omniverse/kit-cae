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
This module provides an operator that marks cells according to whether they lie
inside an axis-aligned bounding box.

Three selection modes are supported:

- ``"all"``: a cell is selected iff every one of its vertices is inside the box.
- ``"any"``: a cell is selected iff at least one of its vertices is inside the box.
- ``"centroid"``: a cell is selected iff its geometric centroid is inside the box.
  Uses the dataset's cached ``cell_centers`` field (see ``Dataset.get_cached_field``),
  so repeated calls on the same dataset do not recompute centroids.

Two entry points are exposed:

- :func:`compute` adds a per-cell ``int32`` mask field (0 / 1) to a shallow copy
  of the input dataset.
- :func:`compute_indices` returns the compacted ``wp.int32`` array of selected
  cell indices, which plugs directly into
  ``dav.data_models.custom.cell_subset.create_dataset``.
"""

from logging import getLogger

import numpy as np
import warp as wp

import dav

logger = getLogger(__name__)

_VALID_MODES = ("all", "any", "centroid")


@dav.cached(aot="operators.cell_in_box")
def get_kernels(data_model: dav.DataModel):
    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_in_box_all_kernel")
    def mask_all_kernel(ds: data_model.DatasetHandle, box_min: wp.vec3f, box_max: wp.vec3f, mask: wp.array(dtype=wp.int32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        inside = wp.int32(0)
        if data_model.CellAPI.is_valid(cell):
            inside = wp.int32(1)
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt = data_model.DatasetAPI.get_point(ds, pt_id)
                if pt.x < box_min.x or pt.x > box_max.x or pt.y < box_min.y or pt.y > box_max.y or pt.z < box_min.z or pt.z > box_max.z:
                    inside = wp.int32(0)
        mask[cell_idx] = inside

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_in_box_any_kernel")
    def mask_any_kernel(ds: data_model.DatasetHandle, box_min: wp.vec3f, box_max: wp.vec3f, mask: wp.array(dtype=wp.int32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        inside = wp.int32(0)
        if data_model.CellAPI.is_valid(cell):
            num_points = data_model.CellAPI.get_num_points(cell, ds)
            for i in range(num_points):
                pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
                pt = data_model.DatasetAPI.get_point(ds, pt_id)
                if pt.x >= box_min.x and pt.x <= box_max.x and pt.y >= box_min.y and pt.y <= box_max.y and pt.z >= box_min.z and pt.z <= box_max.z:
                    inside = wp.int32(1)
        mask[cell_idx] = inside

    return mask_all_kernel, mask_any_kernel


@wp.kernel
def _mask_centroid_kernel(cell_centers: wp.array(dtype=wp.vec3f), box_min: wp.vec3f, box_max: wp.vec3f, mask: wp.array(dtype=wp.int32)):
    cell_idx = wp.tid()
    c = cell_centers[cell_idx]
    inside = wp.int32(1)
    if c.x < box_min.x or c.x > box_max.x or c.y < box_min.y or c.y > box_max.y or c.z < box_min.z or c.z > box_max.z:
        inside = wp.int32(0)
    mask[cell_idx] = inside


@wp.kernel
def _compact_kernel(mask: wp.array(dtype=wp.int32), scan_offsets: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if mask[i] != 0:
        indices[scan_offsets[i]] = i


def _to_vec3f(v) -> wp.vec3f:
    if isinstance(v, wp.vec3f):
        return v
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"box corner must have 3 components, got {arr.size}")
    return wp.vec3f(float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")


def _validate_box(box_min: wp.vec3f, box_max: wp.vec3f) -> None:
    if box_min.x > box_max.x or box_min.y > box_max.y or box_min.z > box_max.z:
        raise ValueError(f"box_min must be <= box_max component-wise, got min=({box_min.x}, {box_min.y}, {box_min.z}), max=({box_max.x}, {box_max.y}, {box_max.z})")


def _launch_mask(dataset: dav.DatasetLike, box_min: wp.vec3f, box_max: wp.vec3f, mode: str) -> wp.array:
    device = dataset.device
    num_cells = dataset.get_num_cells()
    mask = wp.zeros(num_cells, dtype=wp.int32, device=device)
    if num_cells == 0:
        return mask

    if mode == "centroid":
        cell_centers = dataset.get_cached_field("cell_centers").get_data()
        wp.launch(_mask_centroid_kernel, dim=num_cells, inputs=[cell_centers, box_min, box_max, mask], device=device)
    else:
        all_kernel, any_kernel = get_kernels(dataset.data_model)
        kernel = all_kernel if mode == "all" else any_kernel
        wp.launch(kernel, dim=num_cells, inputs=[dataset.handle, box_min, box_max, mask], device=device)
    return mask


def compute(dataset: dav.DatasetLike, box_min, box_max, mode: str = "all", field_name: str = "cell_in_box") -> dav.DatasetLike:
    """Compute a per-cell mask for whether each cell lies inside an axis-aligned box.

    Args:
        dataset: The input dataset.
        box_min: Minimum corner of the box. 3-component tuple / list / numpy array / ``wp.vec3f``.
        box_max: Maximum corner of the box. Same accepted forms as ``box_min``.
        mode: Selection criterion; one of ``"all"``, ``"any"``, or ``"centroid"``.
        field_name: Name for the output mask field (default ``"cell_in_box"``).

    Returns:
        A shallow copy of ``dataset`` with an ``int32`` per-cell mask field added
        (1 where the cell is selected, 0 otherwise).

    Raises:
        ValueError: If ``mode`` is not one of the three supported values, or if
            any component of ``box_min`` exceeds the corresponding component of
            ``box_max``.
    """
    _validate_mode(mode)
    box_min_v = _to_vec3f(box_min)
    box_max_v = _to_vec3f(box_max)
    _validate_box(box_min_v, box_max_v)

    with dav.scoped_timer(f"cell_in_box.compute_{mode}", cuda_filter=wp.TIMING_ALL):
        mask = _launch_mask(dataset, box_min_v, box_max_v, mode)

    field = dav.Field.from_array(mask, dav.AssociationType.CELL)
    result = dataset.shallow_copy()
    result.add_field(field_name, field)
    return result


def compute_indices(dataset: dav.DatasetLike, box_min, box_max, mode: str = "all") -> wp.array:
    """Return a compact ``wp.int32`` array of the cells selected by the box predicate.

    The returned array is on the same device as ``dataset`` and can be passed
    directly to ``dav.data_models.custom.cell_subset.create_dataset`` to produce
    a subset dataset over the selected cells.

    Args:
        dataset: The input dataset.
        box_min: Minimum corner of the box. See :func:`compute`.
        box_max: Maximum corner of the box. See :func:`compute`.
        mode: Selection criterion; one of ``"all"``, ``"any"``, or ``"centroid"``.

    Returns:
        A 1-D ``wp.array`` with ``dtype=wp.int32`` and length equal to the number
        of selected cells.

    Raises:
        ValueError: Same conditions as :func:`compute`.
    """
    _validate_mode(mode)
    box_min_v = _to_vec3f(box_min)
    box_max_v = _to_vec3f(box_max)
    _validate_box(box_min_v, box_max_v)

    device = dataset.device
    num_cells = dataset.get_num_cells()
    if num_cells == 0:
        return wp.zeros(0, dtype=wp.int32, device=device)

    with dav.scoped_timer(f"cell_in_box.compute_indices_{mode}", cuda_filter=wp.TIMING_ALL):
        mask = _launch_mask(dataset, box_min_v, box_max_v, mode)

        scan = wp.zeros(num_cells + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(mask, scan, inclusive=False, add_trailing_sum=True)
        num_selected = int(scan[-1:].numpy()[0])

        indices = wp.zeros(num_selected, dtype=wp.int32, device=device)
        if num_selected > 0:
            wp.launch(_compact_kernel, dim=num_cells, inputs=[mask, scan, indices], device=device)
    return indices


if dav.config.compile_kernels_aot:
    from dav.core import aot

    for data_model in aot.get_data_models(specialization="operators.cell_in_box.dataset"):
        logger.info(f"Compiling kernels for data model: {data_model}")
        all_kernel, any_kernel = get_kernels(data_model)
        wp.compile_aot_module(all_kernel.module, device=aot.get_devices())
        wp.compile_aot_module(any_kernel.module, device=aot.get_devices())
