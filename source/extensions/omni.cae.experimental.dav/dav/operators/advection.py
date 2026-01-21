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
This module provides particle advection using adaptive Runge-Kutta integration.

The advection operator integrates seed points through a velocity field using a Cash-Karp
embedded RK4(5) method with adaptive time stepping. It returns raw advection results
containing positions, cell indices, times, and trajectory lengths.
"""

from typing import Any, Union

import dav
import warp as wp

# Constants for RK45 integrator
RK45_ITER_MAX = 100


@wp.struct
class AdvectionResult:
    """Class to hold the results of the advection computation."""

    """Positions of advected points."""
    positions: wp.array(ndim=2, dtype=wp.vec3f)

    """Index of the cells containing the advected points."""
    cell_idx: wp.array(ndim=2, dtype=wp.int32)

    """Integration times for each advected point."""
    times: wp.array(ndim=2, dtype=wp.float32)

    """Number of steps taken for each point."""
    lengths: wp.array(ndim=1, dtype=wp.int32)


@wp.struct
class Cursor:
    position: wp.vec3f
    cell_idx: wp.int32
    is_valid: wp.bool


class RKHelpers:
    """Cash-Karp coefficients for embedded Runge-Kutta 4(5) method."""

    # Cash-Karp a_{ij} coefficients
    @wp.func
    def dt1(dt0: float, k1: wp.vec3f) -> wp.vec3f:
        return k1 * (dt0 * (1.0 / 5.0))

    @wp.func
    def dt2(dt0: float, k1: wp.vec3f, k2: wp.vec3f) -> wp.vec3f:
        return k1 * (dt0 * (3.0 / 40.0)) + k2 * (dt0 * (9.0 / 40.0))

    @wp.func
    def dt3(dt0: float, k1: wp.vec3f, k2: wp.vec3f, k3: wp.vec3f) -> wp.vec3f:
        return k1 * (dt0 * (3.0 / 10.0)) - k2 * (dt0 * (9.0 / 10.0)) + k3 * (dt0 * (6.0 / 5.0))

    @wp.func
    def dt4(dt0: float, k1: wp.vec3f, k2: wp.vec3f, k3: wp.vec3f, k4: wp.vec3f) -> wp.vec3f:
        return (
            k1 * (dt0 * (-11.0 / 54.0))
            + k2 * (dt0 * (5.0 / 2.0))
            - k3 * (dt0 * (70.0 / 27.0))
            + k4 * (dt0 * (35.0 / 27.0))
        )

    @wp.func
    def dt5(dt0: float, k1: wp.vec3f, k2: wp.vec3f, k3: wp.vec3f, k4: wp.vec3f, k5: wp.vec3f) -> wp.vec3f:
        return (
            k1 * (dt0 * (1631.0 / 55296.0))
            + k2 * (dt0 * (175.0 / 512.0))
            + k3 * (dt0 * (575.0 / 13824.0))
            + k4 * (dt0 * (44275.0 / 110592.0))
            + k5 * (dt0 * (253.0 / 4096.0))
        )

    @wp.func
    def v4(k1: wp.vec3f, k3: wp.vec3f, k4: wp.vec3f, k5: wp.vec3f, k6: wp.vec3f) -> wp.vec3f:
        return (
            k1 * (2825.0 / 27648.0)
            + k3 * (18575.0 / 48384.0)
            + k4 * (13525.0 / 55296.0)
            + k5 * (277.0 / 14336.0)
            + k6 * (1.0 / 4.0)
        )

    @wp.func
    def v5(k1: wp.vec3f, k3: wp.vec3f, k4: wp.vec3f, k6: wp.vec3f) -> wp.vec3f:
        return k1 * (37.0 / 378.0) + k3 * (250.0 / 621.0) + k4 * (125.0 / 594.0) + k6 * (512.0 / 1771.0)


@dav.cached
def get_kernel(data_model: dav.DataModel, field_model, interpolator, seed_data_model: dav.DataModel, tc_model: Any):
    use_custom_termination_condition = wp.static(tc_model is not None)

    @wp.func
    def get_cell_idx(ds: data_model.DatasetHandle, cell_id: Any) -> wp.int32:
        return data_model.DatasetAPI.get_cell_idx_from_id(ds, cell_id)

    @wp.func
    def get_cell_id(ds: data_model.DatasetHandle, cell_idx: Any) -> wp.int32:
        return data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)

    @wp.func
    def create_cursor(position: wp.vec3f, ds: data_model.DatasetHandle, hint: data_model.CellHandle) -> Cursor:
        cursor = Cursor()
        cursor.position = position
        cell = data_model.DatasetAPI.find_cell_containing_point(ds, position, hint)
        if data_model.CellAPI.is_valid(cell):
            cursor.cell_idx = get_cell_idx(ds, data_model.CellAPI.get_cell_id(cell))
            cursor.is_valid = True
        else:
            cursor.cell_idx = -1
            cursor.is_valid = False
        return cursor

    @wp.func
    def advance(cursor: Cursor, offset: wp.vec3f, ds: data_model.DatasetHandle) -> Cursor:
        assert cursor.is_valid, "Cannot advance invalid cursor!"
        return create_cursor(
            cursor.position + offset, ds, data_model.DatasetAPI.get_cell(ds, get_cell_id(ds, cursor.cell_idx))
        )

    @wp.func
    def sample_velocity(ds: data_model.DatasetHandle, velocity: field_model.FieldHandle, cursor: Cursor) -> wp.vec3f:
        if not cursor.is_valid:
            return wp.vec3f(0.0)
        cell = data_model.DatasetAPI.get_cell(ds, get_cell_id(ds, cursor.cell_idx))
        val = interpolator.get(ds, velocity, cell, cursor.position)
        return wp.normalize(wp.vec3f(wp.float32(val.x), wp.float32(val.y), wp.float32(val.z)))

    @wp.func
    def sample_velocity_after_advancing(
        ds: data_model.DatasetHandle, velocity: field_model.FieldHandle, cursor: Cursor, dt: wp.vec3f
    ) -> wp.vec3f:
        next_cursor = advance(cursor, dt, ds)
        return sample_velocity(ds, velocity, next_cursor)

    @wp.func
    def rk_iteration(
        y0: Cursor, dt_used: float, dataset: data_model.DatasetHandle, velocity: field_model.FieldHandle, k1: wp.vec3f
    ) -> tuple[float, Cursor, wp.vec3f]:
        """Perform a single Cash-Karp Runge-Kutta 4(5) integration step.

        Uses the Cash-Karp embedded RK method with proper 4th and 5th order solutions.
        The error is estimated as the difference between the 4th and 5th order solutions.

        Args:
            y0: Current position and cell information (Cursor)
            dt_used: Time step (float)
            dataset: Dataset structure (data_model.DatasetHandle)
            velocity: Velocity field to sample from (FieldHandle)
            k1: First stage velocity (already computed)

        Returns:
            A tuple containing the following elements:
                - err: The estimated error of the step (float)
                - y5: The updated cursor position using 5th order solution (Cursor)
                - v_5: The weighted velocity for the 5th order solution (wp.vec3f)
        """
        assert y0.is_valid, "Cannot perform RK45 iteration from invalid cursor!"

        # Compute intermediate stages using Cash-Karp coefficients
        k2 = sample_velocity_after_advancing(dataset, velocity, y0, RKHelpers.dt1(dt_used, k1))
        k3 = sample_velocity_after_advancing(dataset, velocity, y0, RKHelpers.dt2(dt_used, k1, k2))
        k4 = sample_velocity_after_advancing(dataset, velocity, y0, RKHelpers.dt3(dt_used, k1, k2, k3))
        k5 = sample_velocity_after_advancing(dataset, velocity, y0, RKHelpers.dt4(dt_used, k1, k2, k3, k4))
        k6 = sample_velocity_after_advancing(dataset, velocity, y0, RKHelpers.dt5(dt_used, k1, k2, k3, k4, k5))

        # Cash-Karp 4th order solution (b coefficients, second row)
        v_4 = RKHelpers.v4(k1, k3, k4, k5, k6)
        y4 = advance(y0, dt_used * wp.normalize(v_4), dataset)  # normalization can be made configurable

        # Cash-Karp 5th order solution (b coefficients, first row)
        v_5 = RKHelpers.v5(k1, k3, k4, k6)
        y5 = advance(y0, dt_used * wp.normalize(v_5), dataset)  # normalization can be made configurable

        # Error estimate: difference between 4th and 5th order solutions
        # error is the distance between solutions
        err = wp.length(y5.position - y4.position)
        return err, y5, v_5

    @wp.func
    def rk45(
        cursor: Cursor,
        dt: float,
        min_dt: float,
        max_dt: float,
        tolerance: float,
        dataset: data_model.DatasetHandle,
        velocity: field_model.FieldHandle,
        max_iter: int,
    ) -> tuple[float, float, Cursor, wp.vec3f]:
        """Perform a single Runge-Kutta 4-5 integration step.

        Args:
            cursor: Current position and cell information (Cursor)
            dt: Proposed time step (float)
            min_dt: Minimum allowable time step (float)
            max_dt: Maximum allowable time step (float)
            tolerance: Error tolerance for adaptive stepping (float)
            ds: Dataset structure (data_model.DatasetHandle)
            velocity_field_in: Velocity field to sample from (FieldHandle)

        Returns:
            A tuple containing the following elements:
                - dt_used: The actual time step used (float)
                - dt_suggested: The suggested time step for the next iteration (float)
                - next: The updated cursor position (Cursor)
                - v_used: The velocity at the updated position (wp.vec3f)
        """
        dt_used = wp.float32(dt)
        dt_suggest = wp.float32(dt)  # Default suggestion is what was tried

        # Handle negative dt for reverse advection
        dt_sign = wp.sign(dt)
        dt_abs = wp.abs(dt_used)

        assert cursor.is_valid, "Cannot perform RK45 step from invalid cursor!"
        k1 = sample_velocity(dataset, velocity, cursor)

        # declare variables used in the loop
        for i in range(max_iter):  # limit to 10 iterations to avoid infinite loops
            err, y5, v_5 = rk_iteration(cursor, dt_used, dataset, velocity, k1)
            dt_abs = wp.abs(dt_used)
            if err > tolerance and dt_abs > min_dt:
                e_ratio = err / tolerance if tolerance > 0.0 else 0.0
                if e_ratio == 0.0:
                    dt_abs = min_dt
                elif e_ratio > 1.0:
                    dt_abs = 0.9 * dt_abs * wp.pow(e_ratio, -0.25)
                else:
                    dt_abs = 0.9 * dt_abs * wp.pow(e_ratio, -0.2)

                if dt_abs < min_dt:
                    dt_abs = min_dt
                    dt_suggest = dt_abs * dt_sign
                    break
                elif dt_abs > max_dt:
                    dt_abs = max_dt
                    dt_suggest = dt_abs * dt_sign
                    break
                else:
                    dt_abs = wp.clamp(dt_abs, min_dt, max_dt)
                    dt_used = dt_abs * dt_sign
                    # dt_suggest remains same until accept
            else:
                # Suggest larger/smaller future step based on error
                dt_suggest = dt_sign * wp.clamp(dt_abs * 1.2 if err < tolerance else dt_abs, min_dt, max_dt)
                # wp.printf("Streamline RK45 step accepted: err=%f, dt_used=%f, dt_suggest=%f, pos=(%f, %f, %f)\n", err, dt_used, dt_suggest, y5.pos.x, y5.pos.y, y5.pos.z)
                break

        return dt_used, dt_suggest, y5, v_5  # return last computed values even if not converged

    @wp.func
    def get_seed(seeds: seed_data_model.DatasetHandle, seed_idx: int) -> wp.vec3f:
        seed_id = seed_data_model.DatasetAPI.get_point_id_from_idx(seeds, seed_idx)
        seed_pos = seed_data_model.DatasetAPI.get_point(seeds, seed_id)
        return seed_pos

    @wp.kernel(enable_backward=False)
    def advection_kernel(
        ds: data_model.DatasetHandle,
        velocity_field_in: field_model.FieldHandle,
        cell_sizes: wp.array(dtype=wp.float32),
        seeds: seed_data_model.DatasetHandle,
        dt: float,
        dt_min: float,
        dt_max: float,
        tolerance: float,
        max_steps: int,
        r_forward: AdvectionResult,
        r_backward: AdvectionResult,
        tc_handle: Any,
    ):
        direction, seed_idx = wp.tid()

        # direction: 0 = forward, 1 = backward
        dt = dt if direction == 0 else -dt

        # Support negative dt for reverse direction advection
        dt_min = wp.abs(dt_min)
        dt_max = wp.abs(dt_max)

        step = wp.int32(0)
        propagation_time = wp.float32(0.0)
        dt_suggested = dt
        cursor = create_cursor(get_seed(seeds, seed_idx), ds, data_model.CellAPI.empty())

        result = r_forward if direction == 0 else r_backward
        while step < max_steps and cursor.is_valid:
            cell_size = cell_sizes[cursor.cell_idx]
            if cell_size <= 1e-10:
                # avoid advancing with too small cell size
                break

            # update result.
            result.positions[seed_idx, step] = cursor.position
            result.cell_idx[seed_idx, step] = cursor.cell_idx
            result.times[seed_idx, step] = propagation_time
            result.lengths[seed_idx] = step + 1
            step += 1

            dt_used, dt_suggested, next, v_used = rk45(
                cursor,
                dt_suggested * cell_size,
                dt_min * cell_size,
                dt_max * cell_size,
                tolerance,
                ds,
                velocity_field_in,
                RK45_ITER_MAX,
            )

            if wp.abs(dt_used) <= 1e-7:
                # Step size too small, terminate integration
                break

            if wp.static(use_custom_termination_condition):
                if cursor.is_valid and next.is_valid:
                    if tc_model.terminate(tc_handle, cursor, next, dt_used, v_used, ds):
                        break

            speed = wp.length(v_used)
            integration_time = dt_used / speed if speed > 0.0 else wp.float32(0.0)
            propagation_time += integration_time

            dt_used /= cell_size  # convert back to normalized dt
            dt_suggested /= cell_size  # convert back to normalized dt

            cursor = next

    return advection_kernel


def compute(
    dataset: dav.DatasetLike,
    velocity_field_name: str,
    seeds: dav.DatasetLike,
    initial_dt: float,
    min_dt: float,
    max_dt: float,
    max_steps: int,
    tolerance: float = 1e-5,
    both_directions: bool = True,
    tc_model: Any = None,
    tc_handle: Any = None,
) -> tuple[AdvectionResult, Union[AdvectionResult, None]]:
    """
    Advect seed points through a velocity field using adaptive Runge-Kutta 4-5 integrator.

    Args:
        dataset: The dataset containing the velocity field (Dataset)
        velocity_field_name: Name of the velocity field to advect through (str)
        seeds: The dataset containing seed points (Dataset)
        initial_dt: Initial time step for integration (float)
        min_dt: Minimum allowable time step (float)
        max_dt: Maximum allowable time step (float)
        max_steps: Maximum number of integration steps (int)
        tolerance: Error tolerance for adaptive stepping (float)
        both_directions: Whether to advect in both directions (bool)
        tc_model: Custom termination condition model (Any)
        tc_handle: Custom termination condition handle (Any)
    Returns:
        AdvectionResult: The result of the advection containing positions, lengths, and cell IDs in the forward direction.
        AdvectionResult: The result of the advection in the backward direction if both_directions is True otherwise None.

    Raises:
        KeyError: If the specified velocity field is not found in the dataset.
    """

    # TODO: for path-tracing, we should support accumulating result over multiple calls.
    # This would require passing in existing result arrays and their sizes and then extending them.
    if dataset.device != seeds.device:
        raise ValueError("Dataset and seeds must be on the same device.")

    # Get velocity field from dataset
    try:
        velocity_field_in = dataset.get_field(velocity_field_name)
    except KeyError:
        raise KeyError(
            f"Field '{velocity_field_name}' not found in dataset. Available fields: {list(dataset.fields.keys())}"
        ) from None

    device = dataset.device

    # Build cell locator for the dataset
    dataset.build_cell_locator()

    # Get cell sizes (computed and cached if needed)
    cell_sizes_field = dataset.get_cached_field("cell_sizes")

    nb_seeds = seeds.get_num_points()

    with dav.scoped_timer("advection.allocate_results"):
        forward_result = AdvectionResult()
        forward_result.positions = wp.empty((nb_seeds, max_steps), dtype=wp.vec3f, device=device)
        forward_result.cell_idx = wp.empty((nb_seeds, max_steps), dtype=wp.int32, device=device)
        forward_result.times = wp.empty((nb_seeds, max_steps), dtype=wp.float32, device=device)
        forward_result.lengths = wp.zeros(nb_seeds, dtype=wp.int32, device=device)

        backward_result = AdvectionResult()
        if both_directions:
            backward_result.positions = wp.empty((nb_seeds, max_steps), dtype=wp.vec3f, device=device)
            backward_result.cell_idx = wp.empty((nb_seeds, max_steps), dtype=wp.int32, device=device)
            backward_result.times = wp.empty((nb_seeds, max_steps), dtype=wp.float32, device=device)
            backward_result.lengths = wp.zeros(nb_seeds, dtype=wp.int32, device=device)

    with dav.scoped_timer("advection.get_kernel"):
        kernel = get_kernel(
            dataset.data_model,
            velocity_field_in.field_model,
            velocity_field_in.get_interpolated_field_api(dataset.data_model),
            seeds.data_model,
            tc_model,
        )
    with dav.scoped_timer("advection.launch", cuda_filter=wp.TIMING_ALL):
        wp.launch(
            kernel,
            dim=(2, nb_seeds) if both_directions else (1, nb_seeds),
            inputs=[
                dataset.handle,
                velocity_field_in.handle,
                cell_sizes_field.get_data(),
                seeds.handle,
                initial_dt,
                min_dt,
                max_dt,
                tolerance,
                max_steps,
                forward_result,
                backward_result,
                tc_handle if tc_model is not None else wp.int32(0),
            ],
            device=device,
            # block_dim=8,
        )
        return forward_result, backward_result if both_directions else None
