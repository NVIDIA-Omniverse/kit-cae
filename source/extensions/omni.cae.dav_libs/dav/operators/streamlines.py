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
in forward, backward, or both directions. The result is a dataset containing polyline cells
representing the streamlines, along with time and cell index fields.
"""

from typing import Any, Union

import numpy as np
import warp as wp

import dav
from dav.data_models.custom import curves

from . import advection
from .advection import Direction


def compute(
    ds: dav.DatasetLike,
    velocity_field_name: str,
    seeds: dav.DatasetLike,
    initial_dt: float = 0.2,
    min_dt: float = 0.01,
    max_dt: float = 0.5,
    max_steps: int = 100,
    tolerance: float = 1e-5,
    direction: Union[Direction, str] = Direction.BOTH,
    tc_model: Any = None,
    tc_handle: Any = None,
) -> dav.DatasetLike:
    """
    Compute streamlines in the specified direction(s).

    Args:
        ds: The dataset containing the vector field.
        velocity_field_name: Name of the velocity field.
        seeds: The seed points dataset.
        initial_dt: Initial time step for integration.
        min_dt: Minimum allowable time step.
        max_dt: Maximum allowable time step.
        max_steps: Maximum number of integration steps.
        tolerance: Error tolerance for adaptive stepping.
        direction: Direction for streamline computation - FORWARD, BACKWARD, or BOTH (Direction or str)
        tc_model: Custom termination condition model (Any)
        tc_handle: Custom termination condition handle (Any)
    Returns:
        dav.DatasetLike: The streamlines as a dataset with polylines, or None if no streamlines were computed.

    Raises:
        KeyError: If the specified velocity field is not found in the dataset.
        ValueError: If an invalid direction is provided.
    """
    # Convert string to Direction enum if needed
    if isinstance(direction, str):
        try:
            direction = Direction(direction.lower())
        except ValueError as err:
            raise ValueError(f"Invalid direction: '{direction}'. Must be one of: 'forward', 'backward', or 'both'.") from err

    with dav.scoped_timer("streamlines.advection"):
        results = advection.compute(ds, velocity_field_name, seeds, initial_dt, min_dt, max_dt, max_steps, tolerance, direction=direction, tc_model=tc_model, tc_handle=tc_handle)
        if direction in (Direction.FORWARD, Direction.BOTH):
            if not results[0]:
                raise RuntimeError(f"Expected forward result for {direction.value} direction")
        if direction in (Direction.BACKWARD, Direction.BOTH):
            if not results[1]:
                raise RuntimeError(f"Expected backward result for {direction.value} direction")
        results = [r for r in results if r is not None]
        assert results, f"Expected result for {direction.value} direction"

    with dav.scoped_timer("streamlines.prepare_output"):
        # Combine forward and backward streamlines into polylines based on direction
        # Order: reverse (excluding seed)  + forward (including seed) for BOTH
        # For FORWARD/BACKWARD only: just use that direction
        # purge degenerate streamlines (i.e., length < 2)

        # Handle different direction cases
        pts = [r.positions.numpy() for r in results]
        lengths = [r.lengths.numpy() for r in results]
        times = [r.times.numpy() for r in results]
        cell_idx = [r.cell_idx.numpy() for r in results]

        if direction == Direction.BOTH:
            # Combine forward and backward
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
        else:
            # FORWARD or BACKWARD only
            total_lengths = lengths[0]
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

                length = lengths[0][i]
                combined_positions[offset : offset + length, :] = pts[0][i, 0:length, :]
                combined_times[offset : offset + length] = times[0][i, 0:length]
                combined_cell_idx[offset : offset + length] = cell_idx[0][i, 0:length]
                offset += length

                combined_lengths.append(length)

        padded_combined_lengths = np.array([0] + combined_lengths, dtype=np.int32)
        combined_lengths = padded_combined_lengths[1:]  # exclude initial zero for cell offsets

        if combined_positions.shape[0] == 0:
            return None

        device = ds.device

        points = wp.array(combined_positions, dtype=wp.vec3f, device=device, copy=False)
        lengths = wp.array(combined_lengths, dtype=wp.int32, device=device, copy=False)

        # use the curves data model to create a dataset with points and line counts.
        output_ds = curves.create_dataset(points=points, curve_vertex_counts=lengths)

        times_array = wp.array(combined_times, dtype=wp.float32, device=device)
        times_field = dav.Field.from_array(times_array, dav.AssociationType.VERTEX)
        output_ds.add_field("times", times_field)

        cell_idx_array = wp.array(combined_cell_idx, dtype=wp.int32, device=device)
        cell_idx_field = dav.Field.from_array(cell_idx_array, dav.AssociationType.VERTEX)
        output_ds.add_field("cell_idx", cell_idx_field)

    return output_ds
