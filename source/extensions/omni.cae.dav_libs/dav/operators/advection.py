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

from enum import Enum
from logging import getLogger
from typing import Any, Union

import warp as wp

import dav

logger = getLogger(__name__)


class Direction(Enum):
    """Direction for particle advection."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"


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


# Type aliases for the RK stage matrix
_Mat6x6 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
_Mat6x3 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)
_Vec6f = wp.vec(length=6, dtype=wp.float32)

# Cash-Karp Butcher tableau: a_ij coefficients (lower-triangular, row-major).
# Row i gives the weights for stages k1..k_{i} used to compute the offset for stage k_{i+1}.
_CK_A = wp.constant(
    _Mat6x6(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / 5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 40.0,
        9.0 / 40.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 10.0,
        -9.0 / 10.0,
        6.0 / 5.0,
        0.0,
        0.0,
        0.0,
        -11.0 / 54.0,
        5.0 / 2.0,
        -70.0 / 27.0,
        35.0 / 27.0,
        0.0,
        0.0,
        1631.0 / 55296.0,
        175.0 / 512.0,
        575.0 / 13824.0,
        44275.0 / 110592.0,
        253.0 / 4096.0,
        0.0,
    )
)

# b coefficients for the 4th-order (b4) and 5th-order (b5) Cash-Karp solutions
_CK_B4 = wp.constant(_Vec6f(2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0))
_CK_B5 = wp.constant(_Vec6f(37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0))


@dav.func
def _km_get(km: _Mat6x3, i: int) -> wp.vec3f:
    return wp.vec3f(km[i, 0], km[i, 1], km[i, 2])


@dav.func
def _km_set(km: _Mat6x3, i: int, v: wp.vec3f) -> _Mat6x3:
    km[i, 0] = v[0]
    km[i, 1] = v[1]
    km[i, 2] = v[2]
    return km


@dav.func
def _ck_stage_offset(km: _Mat6x3, stage: int, dt: float) -> wp.vec3f:
    """Position offset for RK stage `stage`, computed from the Butcher tableau row."""
    result = wp.vec3f(0.0)
    for j in range(6):  # unrolled by Warp (Python int) — arithmetic only
        result += _CK_A[stage, j] * _km_get(km, j)
    return result * dt


@dav.func
def _ck_weighted_sum(km: _Mat6x3, b: _Vec6f) -> wp.vec3f:
    """Weighted combination of k-stages by coefficient vector b."""
    result = wp.vec3f(0.0)
    for j in range(6):  # unrolled by Warp (Python int) — arithmetic only
        result += b[j] * _km_get(km, j)
    return result


@dav.cached(aot="operators.advection", aot_roles={"data_model": "dataset", "seed_data_model": "seeds"})
def get_kernel(data_model: dav.DataModel, field_model: dav.FieldModel, interpolator: dav.InterpolatedFieldAPI, seed_data_model: dav.DataModel, tc_model: Any):
    use_custom_termination_condition = wp.static(tc_model is not None)
    tc_handle_type = Any if use_custom_termination_condition else wp.int32

    @dav.func
    def get_cell_idx(ds: data_model.DatasetHandle, cell_id: Any) -> wp.int32:
        return data_model.DatasetAPI.get_cell_idx_from_id(ds, cell_id)

    @dav.func
    def get_cell_id(ds: data_model.DatasetHandle, cell_idx: Any) -> wp.int32:
        return data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)

    @dav.func
    def create_cursor(position: wp.vec3f, ds: data_model.DatasetHandle, hint: data_model.CellHandle) -> Cursor:
        cursor = Cursor()
        cursor.position = position
        cell = data_model.CellLocatorAPI.find_cell_containing_point(ds, position, hint)
        if data_model.CellAPI.is_valid(cell):
            cursor.cell_idx = get_cell_idx(ds, data_model.CellAPI.get_cell_id(cell))
            cursor.is_valid = True
        else:
            cursor.cell_idx = -1
            cursor.is_valid = False
        return cursor

    @dav.func
    def advance(cursor: Cursor, offset: wp.vec3f, ds: data_model.DatasetHandle) -> Cursor:
        assert cursor.is_valid, "Cannot advance invalid cursor!"
        return create_cursor(cursor.position + offset, ds, data_model.DatasetAPI.get_cell(ds, get_cell_id(ds, cursor.cell_idx)))

    @dav.func
    def sample_velocity(ds: data_model.DatasetHandle, velocity: field_model.FieldHandle, cursor: Cursor) -> wp.vec3f:
        if not cursor.is_valid:
            return wp.vec3f(0.0)
        cell = data_model.DatasetAPI.get_cell(ds, get_cell_id(ds, cursor.cell_idx))
        val = interpolator.get(ds, velocity, cell, cursor.position)
        return wp.normalize(wp.vec3f(wp.float32(val.x), wp.float32(val.y), wp.float32(val.z)))

    @dav.func
    def sample_velocity_after_advancing(ds: data_model.DatasetHandle, velocity: field_model.FieldHandle, cursor: Cursor, dt: wp.vec3f) -> wp.vec3f:
        next_cursor = advance(cursor, dt, ds)
        return sample_velocity(ds, velocity, next_cursor)

    @dav.func
    def rk_iteration(y0: Cursor, dt_used: float, dataset: data_model.DatasetHandle, velocity: field_model.FieldHandle, k1: wp.vec3f) -> tuple[float, Cursor, wp.vec3f]:
        """Perform a single Cash-Karp Runge-Kutta 4(5) integration step.

        Uses the Cash-Karp embedded RK method with proper 4th and 5th order solutions.
        The error is estimated as the distance between the 4th and 5th order solution positions.

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

        # Compute k2-k6 via a runtime loop so sample_velocity_after_advancing has one
        # inlined call site in the generated CUDA binary instead of five.
        km = _Mat6x3()
        km = _km_set(km, 0, k1)
        n_stages = wp.int32(5)
        for i in range(n_stages):
            s = i + 1
            offset = _ck_stage_offset(km, s, dt_used)
            km = _km_set(km, s, sample_velocity_after_advancing(dataset, velocity, y0, offset))

        # Cash-Karp 4th and 5th order solution velocities
        v_4 = _ck_weighted_sum(km, _CK_B4)
        v_5 = _ck_weighted_sum(km, _CK_B5)

        # 5th order solution cursor (needs cell lookup for the returned cursor)
        y5 = advance(y0, dt_used * wp.normalize(v_5), dataset)  # normalization can be made configurable

        # 4th order position for error estimate only — no cell lookup needed
        pos4 = y0.position + dt_used * wp.normalize(v_4)
        err = wp.length(y5.position - pos4)

        return err, y5, v_5

    @dav.func
    def rk45(
        cursor: Cursor, dt: float, min_dt: float, max_dt: float, tolerance: float, dataset: data_model.DatasetHandle, velocity: field_model.FieldHandle, max_iter: int
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

    @dav.func
    def get_seed(seeds: seed_data_model.DatasetHandle, seed_idx: int) -> wp.vec3f:
        seed_id = seed_data_model.DatasetAPI.get_point_id_from_idx(seeds, seed_idx)
        seed_pos = seed_data_model.DatasetAPI.get_point(seeds, seed_id)
        return seed_pos

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_advection_kernel")
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
        tc_handle: tc_handle_type,
        direction_array: wp.array(dtype=wp.int32),
    ):
        direction_idx, seed_idx = wp.tid()
        direction = direction_array[direction_idx]

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

            dt_used, dt_suggested, next, v_used = rk45(cursor, dt_suggested * cell_size, dt_min * cell_size, dt_max * cell_size, tolerance, ds, velocity_field_in, RK45_ITER_MAX)

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
    direction: Union[Direction, str] = Direction.BOTH,
    tc_model: Any = None,
    tc_handle: Any = None,
) -> tuple[Union[AdvectionResult, None], Union[AdvectionResult, None]]:
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
        direction: Direction for advection - FORWARD, BACKWARD, or BOTH (Direction or str)
        tc_model: Custom termination condition model (Any)
        tc_handle: Custom termination condition handle (Any)
    Returns:
        tuple: A tuple of (forward_result, backward_result):
            - forward_result: AdvectionResult for forward direction (None if direction is BACKWARD)
            - backward_result: AdvectionResult for backward direction (None if direction is FORWARD)

    Raises:
        KeyError: If the specified velocity field is not found in the dataset.
        ValueError: If an invalid direction is provided.
    """

    # TODO: for path-tracing, we should support accumulating result over multiple calls.
    # This would require passing in existing result arrays and their sizes and then extending them.
    if dataset.device != seeds.device:
        raise ValueError("Dataset and seeds must be on the same device.")

    # Convert string to Direction enum if needed
    if isinstance(direction, str):
        try:
            direction = Direction(direction.lower())
        except ValueError as err:
            raise ValueError(f"Invalid direction: '{direction}'. Must be one of: 'forward', 'backward', or 'both'.") from err

    # Determine which directions to compute
    compute_forward = direction in (Direction.FORWARD, Direction.BOTH)
    compute_backward = direction in (Direction.BACKWARD, Direction.BOTH)

    # Get velocity field from dataset
    if not dataset.has_field(velocity_field_name):
        raise KeyError(f"Field '{velocity_field_name}' not found in dataset. Available fields: {dataset.get_field_names()}")

    velocity_field_in = dataset.get_field(velocity_field_name)
    device = dataset.device

    # Build cell locator for the dataset
    dataset.build_cell_locator()

    # Get cell sizes (computed and cached if needed)
    cell_sizes_field = dataset.get_cached_field("cell_sizes")

    nb_seeds = seeds.get_num_points()

    with dav.scoped_timer("advection.allocate_results"):
        forward_result = AdvectionResult()
        if compute_forward:
            forward_result.positions = wp.zeros((nb_seeds, max_steps), dtype=wp.vec3f, device=device)
            forward_result.cell_idx = wp.zeros((nb_seeds, max_steps), dtype=wp.int32, device=device)
            forward_result.times = wp.zeros((nb_seeds, max_steps), dtype=wp.float32, device=device)
            forward_result.lengths = wp.zeros(nb_seeds, dtype=wp.int32, device=device)

        backward_result = AdvectionResult()
        if compute_backward:
            backward_result.positions = wp.zeros((nb_seeds, max_steps), dtype=wp.vec3f, device=device)
            backward_result.cell_idx = wp.zeros((nb_seeds, max_steps), dtype=wp.int32, device=device)
            backward_result.times = wp.zeros((nb_seeds, max_steps), dtype=wp.float32, device=device)
            backward_result.lengths = wp.zeros(nb_seeds, dtype=wp.int32, device=device)

    with dav.scoped_timer("advection.get_kernel"):
        kernel = get_kernel(dataset.data_model, velocity_field_in.field_model, velocity_field_in.get_interpolated_field_api(dataset.data_model), seeds.data_model, tc_model)
    with dav.scoped_timer("advection.launch", cuda_filter=wp.TIMING_ALL):
        # Determine kernel dimension and direction array based on direction
        if direction == Direction.BOTH:
            dim = (2, nb_seeds)
            direction_array = wp.array([0, 1], dtype=wp.int32, device=device)
        elif direction == Direction.FORWARD:
            dim = (1, nb_seeds)
            direction_array = wp.array([0], dtype=wp.int32, device=device)
        else:  # Direction.BACKWARD
            dim = (1, nb_seeds)
            direction_array = wp.array([1], dtype=wp.int32, device=device)

        wp.launch(
            kernel,
            dim=dim,
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
                direction_array,
            ],
            device=device,
            # block_dim=8,
        )
        return (forward_result if compute_forward else None, backward_result if compute_backward else None)


if dav.config.compile_kernels_aot:
    from dav.core import aot
    from dav.fields import utils as field_utils

    vector_field_models = aot.get_vec3_field_models(specialization="operators.advection")
    seeds_data_models = aot.get_data_models(specialization="operators.advection.seeds")
    dataset_data_models = aot.get_data_models(specialization="operators.advection.dataset")

    for data_model in dataset_data_models:
        for field_model in vector_field_models:
            interpolated_field_api = field_utils.create_interpolated_field_api(data_model, field_model)

            for seed_data_model in seeds_data_models:
                logger.info(f"Compiling kernels for dataset data model: {data_model}, seed data model: {seed_data_model}, field model: {field_model}")

                kernel = get_kernel(data_model, field_model, interpolated_field_api, seed_data_model, None)
                wp.compile_aot_module(kernel.module, device=aot.get_devices())
