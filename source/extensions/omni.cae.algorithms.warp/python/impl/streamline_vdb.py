# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import carb.profiler
import warp as wp

logger = getLogger(__name__)


@wp.func
def sampleVolumeInternalField(volume: wp.uint64, point: wp.vec3f):

    point_uvw = wp.volume_world_to_index(volume, point)
    return wp.volume_sample(volume, point_uvw, wp.Volume.LINEAR, dtype=wp.vec3f)
    # return wp.volume_sample_f(volume, point_uvw, wp.Volume.LINEAR)


@wp.func
def norm3f(v: wp.vec3f):
    return wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@wp.func
def sampleVDB(volume0: wp.uint64, p: wp.vec3f):
    # save |velocity| as scalar
    vs = sampleVolumeInternalField(volume0, p)
    return (vs, norm3f(vs))


@wp.kernel
def advect_vector_field_kernel(
    volume: wp.uint64,  # bvh or vdb with volume factor (vf) 0.1
    initial_points: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    num_timesteps: wp.int32,
    streamlines_paths: wp.array(dtype=wp.vec3f),
    streamlines_scalars: wp.array(dtype=wp.float32),
    dt_MIN: wp.float32,
    dt_MAX: wp.float32,
    epsilon: wp.float32,
):

    tid = wp.tid()
    p = initial_points[tid]
    for it in range(num_timesteps):
        # bounding box is just a point +- epsilon

        streamlines_paths[tid * num_timesteps + it] = p
        (vs, scalar_v) = sampleVDB(volume, p)
        streamlines_scalars[tid * num_timesteps + it] = scalar_v

        # euler time-stepping
        # p = p + dt * vs

        # # rk4 time-stepping
        # # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
        # k1 = vs
        # (k2,_) = sampleVDB(volume0, p + dt/2.0*k1, vdb_vec3, streamlines_scalar_type)
        # (k3,_) = sampleVDB(volume0, p + dt/2.0*k2, vdb_vec3, streamlines_scalar_type)
        # (k4,_) = sampleVDB(volume0, p + dt*k3, vdb_vec3, streamlines_scalar_type)

        # # if it>0:
        # #     time[tid* num_timesteps + it] = time[tid* num_timesteps + it-1] + dt
        # # else:
        # #     time[tid* num_timesteps + it] = 0
        # p = p + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # rk45 adaptive time-stepping
        # a2, a3, a4, a5, a6 = 1/4, 3/8, 12/13, 1, 1/2
        # b21 = 1/4
        # b31, b32 = 3/32, 9/32
        # b41, b42, b43 = 1932/2197, -7200/2197, 7296/2197
        # b51, b52, b53, b54 = 439/216, -8, 3680/513, -845/4104
        # b61, b62, b63, b64, b65 = -8/27, 2, -3544/2565, 1859/4104, -11/40
        # c1, c3, c4, c5, c6 = 16/135, 6656/12825, 28561/56430, -9/50, 2/55
        # d1, d3, d4, d5 = 25/216, 1408/2565, 2197/4104, -1/5

        # Define coefficients
        a2, a3, a4, a5, a6 = 0.25, 0.375, 12.0 / 13.0, 1.0, 0.5
        b21 = 0.25
        b31, b32 = 3.0 / 32.0, 9.0 / 32.0
        b41, b42, b43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
        b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
        b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0
        c1, c3, c4, c5, c6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0
        d1, d3, d4, d5 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0

        # Compute the six function evaluations
        k1, _ = sampleVDB(volume, p)
        k2, _ = sampleVDB(volume, p + dt * (b21 * k1))
        k3, _ = sampleVDB(volume, p + dt * (b31 * k1 + b32 * k2))
        k4, _ = sampleVDB(volume, p + dt * (b41 * k1 + b42 * k2 + b43 * k3))
        k5, _ = sampleVDB(volume, p + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6, _ = sampleVDB(volume, p + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

        # Compute the 5th order and 4th order estimates
        p5 = p + dt * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
        p4 = p + dt * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5)

        # Compute the error estimate
        error = norm3f(p5 - p4)

        # use 5th order estimate
        # Wikipedia uses an average of p4&p5
        p = p5

        MIN_STEP_SIZE = dt_MIN
        MAX_STEP_SIZE = dt_MAX

        # grow step size
        if error < epsilon:

            if error == 0.0:
                # Limit step size increase
                dt_new = 2.0 * dt
            else:
                dt_new = 0.9 * dt * (epsilon / error) ** (1.0 / 5.0)
                # Limit step size increase
                dt_new = min(dt_new, 2.0 * dt)

            # Ensure step size is not too big
            dt_new = min(dt_new, MAX_STEP_SIZE)
            # wp.printf(" [%d:%d/%d] grow dt: error=%f, dt=%f, dt_new=%f (max=%f)\n", tid, it, num_timesteps, error, dt, dt_new, dt_MAX)

        # shrink step size
        else:

            dt_new = 0.9 * dt * (epsilon / error) ** (1.0 / 5.0)
            # wp.printf("[%d:%d/%d] shrink dt: error=%f, dt=%f, dt_new=%f\n", tid, it, num_timesteps, error, dt, dt_new)

            # Ensure step size is not too small
            dt_new = max(dt_new, MIN_STEP_SIZE)

        dt = dt_new


@carb.profiler.profile
def advect_vector_field(
    initial_points, vdb, dt=0.1, num_steps=5, dt_MIN=0.01, dt_MAX=1.0, rk45_epsilon=0.001, device=0
):
    carb.profiler.begin(1, "advect_vector_field pre warp kernel")
    wp.set_device(f"cuda:{device}")
    streamlines_paths = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.vec3f)
    streamlines_scalars = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.float32)

    volume = vdb
    gi = wp.Volume.get_grid_info(volume)
    logger.info("GridInfo=%s", gi)

    carb.profiler.end(1)
    carb.profiler.begin(1, "advect_vector_field warp kernel")
    wp.launch(
        advect_vector_field_kernel,
        dim=initial_points.shape,
        inputs=[
            volume.id,
            initial_points,
            dt,
            num_steps,
            streamlines_paths,
            streamlines_scalars,
            dt_MIN,
            dt_MAX,
            rk45_epsilon,
        ],
        device=f"cuda:{device}",
    )
    carb.profiler.end(1)
    return (streamlines_paths, streamlines_scalars)
