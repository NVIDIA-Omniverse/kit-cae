# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from typing import Any

import numpy as np
import warp as wp

from .data_models.typing import DataModel


@wp.struct
class CellLocatorImpl:
    bvh_id: wp.uint64


class CellLocator:
    dataset: Any  # Data model specific dataset handle
    bvh: wp.Bvh

    def __init__(self, dataset: Any, bvh: wp.Bvh):
        self.dataset = dataset
        self.bvh = bvh

    def get_bvh_id(self) -> wp.uint64:
        return self.bvh.id


def build_cell_locator(dataModel: DataModel, dataset: Any, device: Any) -> CellLocator:
    """
    Build a cell locator for the given dataset based on its data model.
    Args:
        dataModel (DataModel): The data model defining dataset operations.
        dataset (Any): The dataset for which to build the cell locator (data model specific).
        device: The device to run the computation on.
    Returns:
        CellLocator: The constructed cell locator.
    """
    from .operators import cell_bounds

    min_bounds, max_bounds = cell_bounds.compute_from_data_model(dataModel, dataset, device)
    bvh = wp.Bvh(min_bounds, max_bounds)
    return CellLocator(dataset, bvh)


@wp.struct
class CellLinks:
    """
    Data structure for point-to-cell connectivity.
    For each point, stores which cells use that point.

    cell_ids: Flat array of cell IDs
    offsets: For each point, the starting index in cell_ids array
             offsets[i] to offsets[i+1] gives all cells using point i
    """

    cell_ids: wp.array(dtype=wp.int32)
    offsets: wp.array(dtype=wp.int32)


def get_build_cell_links_kernel(dataModel: DataModel):
    @wp.kernel(enable_backward=False)
    def build_cell_links(
        dataset: dataModel.DatasetHandle,
        cell_ids: wp.array(dtype=wp.int32),
        point_offsets: wp.array(dtype=wp.int32),
        counts: wp.array(dtype=wp.int32),
    ):
        cell_idx = wp.tid()
        cell_id = dataModel.DatasetAPI.get_cell_id_from_idx(dataset, cell_idx)
        cell = dataModel.DatasetAPI.get_cell(dataset, cell_id)
        if dataModel.CellAPI.is_valid(cell):
            num_points = dataModel.CellAPI.get_num_points(cell, dataset)
            for i in range(num_points):
                point_id = dataModel.CellAPI.get_point_id(cell, i, dataset)
                point_idx = dataModel.DatasetAPI.get_point_idx_from_id(dataset, point_id)

                offset = point_offsets[point_idx]
                loc = wp.atomic_add(counts, point_idx, 1)
                cell_ids[offset + loc] = cell_id

    return build_cell_links


def build_cell_links(dataModel: DataModel, dataset: Any, device: Any) -> CellLinks:
    """
    Build the cell links for the given dataset based on its data model.

    Creates a point-to-cell connectivity structure where for each point,
    we can efficiently query which cells use that point.

    Args:
        dataModel (DataModel): The data model defining dataset operations.
        dataset (Any): The dataset for which to build the cell links (data model specific).
        device: The device to run the computation on.
    Returns:
        CellLinks: The constructed cell links with:
            - offsets: array of size (num_points + 1) indicating where each point's cells start
            - cell_ids: flat array containing cell IDs that use each point
    """
    from .operators import point_cell_counts

    nb_cells = dataModel.DatasetAPI.get_num_cells(dataset)
    nb_points = dataModel.DatasetAPI.get_num_points(dataset)

    # Step 1: Compute how many cells use each point
    counts = wp.zeros((nb_points,), dtype=wp.int32, device=device)
    kernel = point_cell_counts.get_kernel(dataModel)
    wp.launch(kernel, dim=nb_cells, inputs=[dataset, counts], device=device)

    # Step 2: Convert counts to offsets (prefix sum)
    counts_np = counts.numpy()
    offsets_np = np.zeros(nb_points + 1, dtype=np.int32)
    offsets_np[0] = 0
    np.cumsum(counts_np, out=offsets_np[1:])

    # Step 3: Create warp arrays for offsets and cell_ids
    offsets = wp.array(offsets_np, dtype=wp.int32, device=device)
    cell_ids = wp.zeros((offsets_np[-1],), dtype=wp.int32, device=device)

    # Step 4: Reset counts to use as write positions
    counts.fill_(0)

    # Step 5: Fill in the cell IDs for each point
    kernel = get_build_cell_links_kernel(dataModel)
    wp.launch(kernel, dim=nb_cells, inputs=[dataset, cell_ids, offsets, counts], device=device)

    # Step 6: Create and return CellLinks struct
    cellLinks = CellLinks()
    cellLinks.cell_ids = cell_ids
    cellLinks.offsets = offsets
    return cellLinks
