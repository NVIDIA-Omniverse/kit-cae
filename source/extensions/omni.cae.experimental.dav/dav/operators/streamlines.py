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
This module provides streamline computation for vector fields.

The streamlines operator computes integral curves of a velocity field by advecting seed points
in both forward and backward directions. The result is a dataset containing polyline cells
representing the streamlines, along with time and cell index fields.
"""

from typing import Any

import dav
import numpy as np
import warp as wp

from . import advection


def compute(
    ds: dav.DatasetLike,
    velocity_field_name: str,
    seeds: dav.DatasetLike,
    initial_dt: float = 0.2,
    min_dt: float = 0.01,
    max_dt: float = 0.5,
    max_steps: int = 100,
    tolerance: float = 1e-5,
    tc_model: Any = None,
    tc_handle: Any = None,
) -> dav.DatasetLike:
    """
    Compute streamlines in both forward and backward directions.

    Args:
        ds: The dataset containing the vector field.
        velocity_field_name: Name of the velocity field.
        seeds: The seed points dataset.
        initial_dt: Initial time step for integration.
        min_dt: Minimum allowable time step.
        max_dt: Maximum allowable time step.
        max_steps: Maximum number of integration steps.
        tolerance: Error tolerance for adaptive stepping.
        tc_model: Custom termination condition model (Any)
        tc_handle: Custom termination condition handle (Any)
    Returns:
        dav.DatasetLike: The streamlines as a dataset with polylines.

    Raises:
        KeyError: If the specified velocity field is not found in the dataset.
    """
    with dav.scoped_timer("streamlines.advection"):
        forward, backward = advection.compute(
            ds,
            velocity_field_name,
            seeds,
            initial_dt,
            min_dt,
            max_dt,
            max_steps,
            tolerance,
            both_directions=True,
            tc_model=tc_model,
            tc_handle=tc_handle,
        )
        # wp.synchronize_device(ds.device)

    with dav.scoped_timer("streamlines.prepare_output"):
        # Combine forward and backward streamlines into polylines
        # Order: reverse (excluding seed)  + forward (including seed)
        # purge degenerate streamlines (i.e., length < 2)
        pts = [forward.positions.numpy(), backward.positions.numpy()]
        lengths = [forward.lengths.numpy(), backward.lengths.numpy()]
        times = [forward.times.numpy(), backward.times.numpy()]
        cell_idx = [forward.cell_idx.numpy(), backward.cell_idx.numpy()]

        total_lengths = lengths[0] + lengths[1] - 1  # exclude seed from backward
        valid_mask = total_lengths >= 2
        sum_lengths = np.sum(total_lengths[valid_mask])

        combined_positions = np.zeros((sum_lengths, 3), dtype=np.float32)
        combined_times = np.zeros((sum_lengths,), dtype=np.float32)
        combined_cell_idx = np.zeros((sum_lengths,), dtype=np.int32)
        combined_lengths = []
        offset = 0

        for i in range(lengths[0].shape[0]):
            if not valid_mask[i]:
                continue

            l_forward = lengths[0][i]
            l_reverse = lengths[1][i]
            total_l = l_forward + l_reverse - 1

            # add reverse points in reverse order (excluding the seed point at index 0)
            if l_reverse > 1:
                combined_positions[offset : offset + l_reverse - 1, :] = pts[1][i, l_reverse - 1 : 0 : -1, :]
                combined_times[offset : offset + l_reverse - 1] = times[1][i, l_reverse - 1 : 0 : -1]
                combined_cell_idx[offset : offset + l_reverse - 1] = cell_idx[1][i, l_reverse - 1 : 0 : -1]
                offset += l_reverse - 1

            # add forward points (including the seed point at index 0)
            combined_positions[offset : offset + l_forward, :] = pts[0][i, 0:l_forward, :]
            combined_times[offset : offset + l_forward] = times[0][i, 0:l_forward]
            combined_cell_idx[offset : offset + l_forward] = cell_idx[0][i, 0:l_forward]
            offset += l_forward

            combined_lengths.append(total_l)

        padded_combined_lengths = np.array([0] + combined_lengths, dtype=np.int32)
        combined_lengths = padded_combined_lengths[1:]  # exclude initial zero for cell offsets

        # Create output dataset
        from dav.data_models.vtk import unstructured_grid

        device = ds.device

        # TODO: don't like this as VTK data model needs too much extra info
        # Perhaps we introduce a new compact data model for streamlines and use that?
        dataset_handle = unstructured_grid.DatasetHandle()
        dataset_handle.points = wp.array(combined_positions, dtype=wp.vec3f, device=device, copy=False)
        dataset_handle.cell_types = wp.full(
            combined_lengths.shape[0], unstructured_grid.VTK_POLY_LINE, dtype=wp.int32, device=device
        )
        dataset_handle.cell_connectivity = wp.array(
            np.arange(combined_positions.shape[0], dtype=np.int32), dtype=wp.int32, device=device
        )
        dataset_handle.cell_offsets = wp.array(np.cumsum(padded_combined_lengths), dtype=wp.int32, device=device)

        output_ds = dav.Dataset(data_model=unstructured_grid.DataModel, handle=dataset_handle, device=device)
        times_array = wp.array(combined_times, dtype=wp.float32, device=device)
        times_field = dav.Field.from_array(times_array, dav.AssociationType.VERTEX)
        output_ds.fields["times"] = times_field

        cell_idx_array = wp.array(combined_cell_idx, dtype=wp.int32, device=device)
        cell_idx_field = dav.Field.from_array(cell_idx_array, dav.AssociationType.VERTEX)
        output_ds.fields["cell_idx"] = cell_idx_field

    return output_ds
