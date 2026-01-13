# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# import cupy as cp
import ctypes
import math

# from cuda import cuda
import os
from typing import Any

import carb.profiler
import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, Vt

from ..types import ElementType

# Proposed solution
# def tetraCoord_Dorian(A,B,C,D):
#     v1 = B-A ; v2 = C-A ; v3 = D-A
#     # mat defines an affine transform from the tetrahedron to the orthogonal system
#     mat = np.array((v1,v2,v3)).T # i think the transpose is wrong!
#     # The inverse matrix does the opposite (from orthogonal to tetrahedron)
#     M1 = np.linalg.inv(mat)
#     return(M1)

# def pointInside_Dorian(v1,v2,v3,v4,p):
#     # Find the transform matrix from orthogonal to tetrahedron system
#     M1=tetraCoord_Dorian(v1,v2,v3,v4)
#     # apply the transform to P
#     newp = M1.dot(p-v1)
#     # perform test
#     return (np.all(newp>=0) and np.all(newp <=1) and np.sum(newp)<=1)


# Gives a transform that yields a matrix that transforms a point into the reference coordinate system
# in other words, converts the points into barycentric coordinates (3D)
@wp.func
def Mref_tet(point: wp.vec3f, vertices: wp.array(dtype=wp.vec3f), idx_tet: wp.array(dtype=wp.int32), idx: wp.int32):

    a = vertices[idx_tet[4 * idx + 0]]
    b = vertices[idx_tet[4 * idx + 1]]
    c = vertices[idx_tet[4 * idx + 2]]
    d = vertices[idx_tet[4 * idx + 3]]

    v1 = b - a
    v2 = c - a
    v3 = d - a

    M = wp.inverse(wp.mat33(v1, v2, v3))

    return M


# matrix to convert input point into reference HEX coordinates
@wp.func
def Mref_hex(point: wp.vec3f, vertices: wp.array(dtype=wp.vec3f), idx_hex: wp.array(dtype=wp.int32), idx: wp.int32):

    # CGNS layout https://cgns.github.io/standard/SIDS/convention.html#id16
    #
    #             6          5
    #             +----------+
    #            /|         /|
    #           / |      4 / |
    #        7 /----------/  |
    #          |  |       |  |
    #          |  +-------|--+ 1
    #          | /2       | /
    #          |/         |/
    #          +----------+
    #         3           0

    a = vertices[idx_hex[8 * idx + 3]]
    b = vertices[idx_hex[8 * idx + 0]]
    c = vertices[idx_hex[8 * idx + 2]]
    d = vertices[idx_hex[8 * idx + 7]]

    v1 = b - a
    v2 = c - a
    v3 = d - a

    M = wp.inverse(wp.mat33(v1, v2, v3))

    return M


@wp.func
def point_in_tetrahedra(
    point: wp.vec3f, vertices: wp.array(dtype=wp.vec3f), idx_tet: wp.array(dtype=wp.int32), idx: wp.int32
):

    a = vertices[idx_tet[4 * idx + 0]]
    b = vertices[idx_tet[4 * idx + 1]]
    c = vertices[idx_tet[4 * idx + 2]]
    d = vertices[idx_tet[4 * idx + 3]]

    v1 = b - a
    v2 = c - a
    v3 = d - a

    M123 = wp.mat33(v1, v2, v3)
    M = wp.inverse(M123)

    pt = M @ (point - a)

    cond1 = pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0
    cond2 = pt[0] <= 1 and pt[1] <= 1 and pt[2] <= 1
    cond3 = (pt[0] + pt[1] + pt[2]) <= 1.0

    # print(f"({pt[0]},{pt[1]},{pt[2]})")
    # print("pt")
    # print(pt)
    return cond1 and cond2 and cond3


# @wp.kernel
# def test_point_in_tetrahedra(vertices: wp.array(dtype=wp.vec3f), idx_tet: wp.array(dtype=wp.int32)):
#     p = wp.vec3f(0.1, 0.1, 0.1)
#     p2 = wp.vec3f(1.1, 0.1, 0.1)
#     p3 = wp.vec3f(0.99, 0.99, 0.99)
#     if point_in_tetrahedra(p, vertices, idx_tet, 0) != True:
#         print("ERROR Test 'p' failed")
#     if point_in_tetrahedra(p2, vertices, idx_tet, 0) != False:
#         print("ERROR Test 'p2' failed")
#     if point_in_tetrahedra(p3, vertices, idx_tet, 1) != True:
#         print("ERROR Test 'p3' failed")


@wp.kernel
def test_point_in_hexahedra(vertices: wp.array(dtype=wp.vec3f), idx_elem: wp.array(dtype=wp.int32)):
    p = wp.vec3f(0.1, 0.1, 0.1)
    p2 = wp.vec3f(1.1, 0.1, 0.1)
    p3 = wp.vec3f(0.99, 0.99, 0.99)
    p4 = wp.vec3f(1.1, 0.1, 0.1)
    if point_in_hexahedra(p, vertices, idx_elem, 0) != True:
        print("ERROR Test 'p' failed")
    if point_in_hexahedra(p2, vertices, idx_elem, 0) != False:
        print("ERROR Test 'p2' failed")
    if point_in_hexahedra(p2, vertices, idx_elem, 1) != True:
        print("ERROR Test 'p2' failed")
    if point_in_hexahedra(p3, vertices, idx_elem, 0) != True:
        print("ERROR Test 'p3' failed")
    if point_in_hexahedra(p4, vertices, idx_elem, 1) != True:
        print("ERROR Test 'p4' failed")


@wp.func
def point_in_hexahedra(
    point: wp.vec3f, vertices: wp.array(dtype=wp.vec3f), idx_hex: wp.array(dtype=wp.int32), idx: wp.int32
):

    # CGNS layout https://cgns.github.io/standard/SIDS/convention.html#id16
    #
    #             6          5
    #             +----------+
    #            /|         /|
    #           / |      4 / |
    #        7 /----------/  |
    #          |  |       |  |
    #          |  +-------|--+ 1
    #          | /2       | /
    #          |/         |/
    #          +----------+
    #         3           0

    if idx * 8 > idx_hex.shape[0]:
        wp.printf("idx too big: %d > %d\n", idx * 8, idx_hex.shape[0])
        return False

    a = vertices[idx_hex[8 * idx + 3]]
    b = vertices[idx_hex[8 * idx + 0]]
    c = vertices[idx_hex[8 * idx + 2]]
    d = vertices[idx_hex[8 * idx + 7]]

    # wp.printf("[%d]: (%f,%f,%f)--(%f,%f,%f)--(%f,%f,%f)--(%f,%f,%f)\n",
    #           idx,
    #           a[0],a[1],a[2],
    #           b[0],b[1],b[2],
    #           c[0],c[1],c[2],
    #           d[0],d[1],d[2]
    #           )

    v1 = b - a
    v2 = c - a
    v3 = d - a

    M123 = wp.mat33(v1, v2, v3)
    M = wp.inverse(M123)

    pt = M @ (point - a)

    cond1 = pt[0] >= 0 and pt[1] >= 0 and pt[2] >= 0
    cond2 = pt[0] <= 1 and pt[1] <= 1 and pt[2] <= 1
    cond3 = (pt[0] + pt[1] + pt[2]) <= 3.0

    # print(f"({pt[0]},{pt[1]},{pt[2]})")
    # print("pt")
    # print(pt)
    return cond1 and cond2 and cond3


@wp.func
def interpolate_vector_field_tet(
    point: wp.vec3f,
    vertices: wp.array(dtype=wp.vec3f),
    idx_tet: wp.array(dtype=wp.int32),
    idx: wp.int32,
    vector_field: wp.array(dtype=Any),
):

    a = vertices[idx_tet[4 * idx + 0]]
    # b = vertices[idx_tet[4 * idx + 1]]
    # c = vertices[idx_tet[4 * idx + 2]]
    # d = vertices[idx_tet[4 * idx + 3]]

    M = Mref_tet(point, vertices, idx_tet, idx)

    pref = M @ (point - a)

    f0 = vector_field[idx_tet[4 * idx + 0]]
    f1 = vector_field[idx_tet[4 * idx + 1]]
    f2 = vector_field[idx_tet[4 * idx + 2]]
    f3 = vector_field[idx_tet[4 * idx + 3]]

    x = pref[0]
    y = pref[1]
    z = pref[2]
    # solution to vandermonde matrix (https://en.wikipedia.org/wiki/Vandermonde_matrix)
    fp = f0 + (f1 - f0) * x + (f2 - f0) * y + (f3 - f0) * z

    return fp


# CGNS layout https://cgns.github.io/standard/SIDS/convention.html#id16
#
#             6          5
#             +----------+
#            /|         /|
#           / |      4 / |
#        7 /----------/  |
#          |  |       |  |
#          |  +-------|--+ 1
#          | /2       | /
#          |/         |/
#          +----------+
#         3           0
@wp.func
def interpolate_vector_field_hex(
    point: wp.vec3f,
    vertices: wp.array(dtype=wp.vec3f),
    idx_hex: wp.array(dtype=wp.int32),
    idx: wp.int32,
    vector_field: wp.array(dtype=Any),
):

    a = vertices[idx_hex[8 * idx + 3]]

    M = Mref_hex(point, vertices, idx_hex, idx)

    pref = M @ (point - a)

    f0 = vector_field[idx_hex[8 * idx + 3]]
    f1 = vector_field[idx_hex[8 * idx + 2]]
    f2 = vector_field[idx_hex[8 * idx + 1]]
    f3 = vector_field[idx_hex[8 * idx + 0]]
    f4 = vector_field[idx_hex[8 * idx + 7]]
    f5 = vector_field[idx_hex[8 * idx + 4]]
    f6 = vector_field[idx_hex[8 * idx + 5]]
    f7 = vector_field[idx_hex[8 * idx + 6]]

    x = pref[0]
    y = pref[1]
    z = pref[2]

    # see compute_interpolation_hex() below for how to get the correct coefficients
    # solution to vandermonde matrix (https://en.wikipedia.org/wiki/Vandermonde_matrix)
    # computed using sympy
    fp = (
        f0
        + x * y * z * (-f0 + f1 - f2 + f3 + f4 - f5 + f6 - f7)
        + x * y * (f0 - f1 + f2 - f3)
        + x * z * (f0 - f3 - f4 + f5)
        + x * (-f0 + f3)
        + y * z * (f0 - f1 - f4 + f7)
        + y * (-f0 + f1)
        + z * (-f0 + f4)
    )

    return fp


# CGNS layout https://cgns.github.io/standard/SIDS/convention.html#id16
#
#             6          5
#             +----------+
#            /|         /|
#           / |      4 / |
#        7 /----------/  |
#          |  |       |  |
#          |  +-------|--+ 1
#          | /2       | /
#          |/         |/
#          +----------+
#         3           0
def compute_interpolation_hex():
    from sympy import Matrix, S, symbols
    from sympy.solvers.solveset import linsolve

    x, y, z, f0, f1, f2, f3, f4, f5, f6, f7 = symbols("x,y,z,f0,f1,f2,f3,f4,f5,f6,f7")
    # The linear interpolating polynomial. Needs 8 terms for 8 corners
    eq = [S.One, x, x * y, y, z, x * z, y * z, x * y * z]
    # 3,2,1,0,7,4,5,6
    corners = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    # vandermonde matrix, essentially 'eq' evaluated at each point
    V = Matrix([[eqi.subs({x: xi, y: yi, z: zi}) for eqi in eq] for (xi, yi, zi) in corners])
    b = Matrix([f0, f1, f2, f3, f4, f5, f6, f7])
    # solver solves V@x = b for 'x'. Basically solve for the right combo of
    # 'fn' so that the interpolation is correct at all the corners
    x = Matrix(linsolve((V, b)).args[0])
    interpolation_expr = Matrix(eq).T @ x
    print("interpolation = ", interpolation_expr)


@wp.func
def norm3f(v: wp.vec3f):
    return wp.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@wp.func
def sampleFEM(
    fem_volume: wp.uint64,
    fieldX: wp.array(dtype=wp.float32),
    fieldY: wp.array(dtype=wp.float32),
    fieldZ: wp.array(dtype=wp.float32),
    elem_type: wp.int32,  # 0 - tetrahedra, 1 - hexahedra
    vertices: wp.array(dtype=wp.vec3f),
    idx_elem: wp.array(dtype=wp.int32),
    prev_idx: wp.int32,
    p: wp.vec3f,
):

    # Try the same element as "last time", because RK45 will sample the volume several times per time step, we try to
    # avoid querying the BVH again by checking if the point is in the previously found element (as indexed by prev_idx),
    # unless prev_idx==-1, where we just skip this step.
    if prev_idx > -1:
        if elem_type == 10:  # TETRA_4 (CGNS)
            if point_in_tetrahedra(p, vertices, idx_elem, prev_idx):
                viX = interpolate_vector_field_tet(p, vertices, idx_elem, prev_idx, fieldX)
                viY = interpolate_vector_field_tet(p, vertices, idx_elem, prev_idx, fieldY)
                viZ = interpolate_vector_field_tet(p, vertices, idx_elem, prev_idx, fieldZ)
                vi = wp.vec3f(viX, viY, viZ)
                return (vi, prev_idx)
        elif elem_type == 17:  # HEXA_8 (CGNS)
            if point_in_hexahedra(p, vertices, idx_elem, prev_idx):
                viX = interpolate_vector_field_hex(p, vertices, idx_elem, prev_idx, fieldX)
                viY = interpolate_vector_field_hex(p, vertices, idx_elem, prev_idx, fieldY)
                viZ = interpolate_vector_field_hex(p, vertices, idx_elem, prev_idx, fieldZ)
                vi = wp.vec3f(viX, viY, viZ)
                return (vi, prev_idx)
        else:
            wp.printf("ERROR: Unsupported element type: %d!\n", elem_type)

    eps = wp.vec3f(1e-8, 1e-8, 1e-8)
    lower_b = p - eps
    upper_b = p + eps
    query = wp.bvh_query_aabb(fem_volume, lower_b, upper_b)
    candidate_idx = wp.int32(-1)
    found_idx = wp.int32(-1)

    while wp.bvh_query_next(query, candidate_idx):
        if elem_type == 10:  # TETRA_4 (CGNS)
            if point_in_tetrahedra(p, vertices, idx_elem, candidate_idx):
                found_idx = candidate_idx
                # wp.printf("FOUND tet %d\n", found_idx)
                viX = interpolate_vector_field_tet(p, vertices, idx_elem, found_idx, fieldX)
                viY = interpolate_vector_field_tet(p, vertices, idx_elem, found_idx, fieldY)
                viZ = interpolate_vector_field_tet(p, vertices, idx_elem, found_idx, fieldZ)
                vi = wp.vec3f(viX, viY, viZ)
                return (vi, candidate_idx)
        elif elem_type == 17:  # HEXA_8 (CGNS)
            if point_in_hexahedra(p, vertices, idx_elem, candidate_idx):
                found_idx = candidate_idx
                viX = interpolate_vector_field_hex(p, vertices, idx_elem, found_idx, fieldX)
                viY = interpolate_vector_field_hex(p, vertices, idx_elem, found_idx, fieldY)
                viZ = interpolate_vector_field_hex(p, vertices, idx_elem, found_idx, fieldZ)
                vi = wp.vec3f(viX, viY, viZ)
                return (vi, candidate_idx)

    if found_idx == -1:
        donothing = 0
        return (wp.vec3f(0.0, 0.0, 0.0), -1)


@wp.func
def sign(val: wp.float32):
    if val < 0.0:
        return -1.0
    else:
        return 1.0


@wp.kernel
def advect_vector_field_kernel(
    initial_points: wp.array(dtype=wp.vec3),
    dt: wp.float32,
    num_timesteps: wp.int32,
    vertices: wp.array(dtype=wp.vec3f),
    idx_elem: wp.array(dtype=wp.int32),
    elem_type: wp.int32,  # 0 - tetrahedra, 1 - hexahedra
    volume_fem: wp.uint64,
    fieldX: wp.array(dtype=wp.float32),
    fieldY: wp.array(dtype=wp.float32),
    fieldZ: wp.array(dtype=wp.float32),
    streamlines_paths: wp.array(dtype=wp.vec3f),
    streamlines_scalars: wp.array(dtype=wp.float32),
    streamlines_time: wp.array(dtype=wp.float32),
    dt_MIN: wp.float32,
    dt_MAX: wp.float32,
    epsilon: wp.float32,
):

    tid = wp.tid()
    p = initial_points[tid]

    # Define coefficients
    # a2, a3, a4, a5, a6 = 0.25, 0.375, 12.0/13.0, 1.0, 0.5 # not needed
    b21 = 0.25
    b31, b32 = 3.0 / 32.0, 9.0 / 32.0
    b41, b42, b43 = 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0
    b51, b52, b53, b54 = 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0
    b61, b62, b63, b64, b65 = -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0
    c1, c3, c4, c5, c6 = 16.0 / 135.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0
    d1, d3, d4, d5 = 25.0 / 216.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0

    MIN_STEP_SIZE = dt_MIN
    MAX_STEP_SIZE = dt_MAX

    time = wp.float32(0.0)
    prev_idx = wp.int32(-1)
    it = wp.int32(0)
    failed_tries = wp.int32(0)
    while it < num_timesteps:

        # Compute the six function evaluations
        k1, prev_idx = sampleFEM(volume_fem, fieldX, fieldY, fieldZ, elem_type, vertices, idx_elem, prev_idx, p)
        k2, prev_idx = sampleFEM(
            volume_fem, fieldX, fieldY, fieldZ, elem_type, vertices, idx_elem, prev_idx, p + dt * (b21 * k1)
        )
        k3, prev_idx = sampleFEM(
            volume_fem, fieldX, fieldY, fieldZ, elem_type, vertices, idx_elem, prev_idx, p + dt * (b31 * k1 + b32 * k2)
        )
        k4, prev_idx = sampleFEM(
            volume_fem,
            fieldX,
            fieldY,
            fieldZ,
            elem_type,
            vertices,
            idx_elem,
            prev_idx,
            p + dt * (b41 * k1 + b42 * k2 + b43 * k3),
        )
        k5, prev_idx = sampleFEM(
            volume_fem,
            fieldX,
            fieldY,
            fieldZ,
            elem_type,
            vertices,
            idx_elem,
            prev_idx,
            p + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4),
        )
        k6, prev_idx = sampleFEM(
            volume_fem,
            fieldX,
            fieldY,
            fieldZ,
            elem_type,
            vertices,
            idx_elem,
            prev_idx,
            p + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5),
        )

        # Compute the 5th order and 4th order estimates
        p5 = p + dt * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
        p4 = p + dt * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5)

        # Compute the error estimate
        error = norm3f(p5 - p4)

        # use 5th order estimate
        # Wikipedia uses an average of p4&p5

        if error < epsilon:
            # accept current time-step

            # save previous step
            streamlines_paths[tid * num_timesteps + it] = p
            streamlines_scalars[tid * num_timesteps + it] = norm3f(k1)  # k1 is field value at p
            streamlines_time[tid * num_timesteps + it] = time

            # new p is 5th order solution
            p = p5
            time += dt
            it += 1

            if error == 0.0:
                # Limit step size increase
                dt_new = 2.0 * dt
            else:
                dt_new = 0.9 * dt * (epsilon / error) ** (1.0 / 5.0)
                # Limit step size increase
                dt_new = min(abs(dt_new), abs(2.0 * dt))

            # Ensure step size is not too big
            dt_new = min(abs(dt_new), abs(MAX_STEP_SIZE))
            # wp.printf(" [%d:%d/%d] grow dt: error=%f, dt=%f, dt_new=%f (max=%f)\n", tid, it, num_timesteps, error, dt, dt_new, dt_MAX)

        else:
            # reject the step, adjust the stepsize, and rerun RK45 with smaller step
            dt_new = 0.9 * dt * (epsilon / error) ** (1.0 / 5.0)

            # if tid==0:
            #     wp.printf("[%d:%d/%d] shrink dt: error=%f, dt=%f, dt_new=%f\n", tid, it,
            #               num_timesteps, error, dt, dt_new)
            failed_tries += 1
            # Ensure step size is not too small
            dt_new = max(abs(dt_new), abs(MIN_STEP_SIZE))

            if failed_tries == 2:
                # save current step and make progress regardless of failure
                # wp.printf("[%d:%d/%d] failed to make progress: error=%f, dt=%f, dt_new=%f\n", tid, it,
                #           num_timesteps, error, dt, dt_new)
                streamlines_paths[tid * num_timesteps + it] = p
                streamlines_scalars[tid * num_timesteps + it] = norm3f(k1)  # k1 is field value at p
                streamlines_time[tid * num_timesteps + it] = time
                time += dt
                it += 1
                failed_tries = 0
        # adjust stepsize (up or down based on error).
        # ensure sign stays the same
        dt = sign(dt) * abs(dt_new)


@wp.kernel
def build_hexahedra_bb(
    vertices: wp.array(dtype=wp.vec3f),
    idx_hex: wp.array(dtype=wp.int32),
    lower_b: wp.array(dtype=wp.vec3),
    upper_b: wp.array(dtype=wp.vec3),
):

    tid = wp.tid()
    a = vertices[idx_hex[8 * tid + 0]]
    b = vertices[idx_hex[8 * tid + 1]]
    c = vertices[idx_hex[8 * tid + 2]]
    d = vertices[idx_hex[8 * tid + 3]]
    e = vertices[idx_hex[8 * tid + 4]]
    f = vertices[idx_hex[8 * tid + 5]]
    g = vertices[idx_hex[8 * tid + 6]]
    h = vertices[idx_hex[8 * tid + 7]]

    min_x = a.x
    min_x = min(min_x, b.x)
    min_x = min(min_x, c.x)
    min_x = min(min_x, d.x)
    min_x = min(min_x, d.x)
    min_x = min(min_x, e.x)
    min_x = min(min_x, f.x)
    min_x = min(min_x, g.x)
    min_x = min(min_x, h.x)

    min_y = a.y
    min_y = min(min_y, b.y)
    min_y = min(min_y, c.y)
    min_y = min(min_y, d.y)
    min_y = min(min_y, e.y)
    min_y = min(min_y, f.y)
    min_y = min(min_y, g.y)
    min_y = min(min_y, h.y)

    min_z = a.z
    min_z = min(min_z, b.z)
    min_z = min(min_z, c.z)
    min_z = min(min_z, d.z)
    min_z = min(min_z, e.z)
    min_z = min(min_z, f.z)
    min_z = min(min_z, g.z)
    min_z = min(min_z, h.z)

    max_x = a.x
    max_x = max(max_x, b.x)
    max_x = max(max_x, c.x)
    max_x = max(max_x, d.x)
    max_x = max(max_x, d.x)
    max_x = max(max_x, e.x)
    max_x = max(max_x, f.x)
    max_x = max(max_x, g.x)
    max_x = max(max_x, h.x)

    max_y = a.y
    max_y = max(max_y, b.y)
    max_y = max(max_y, c.y)
    max_y = max(max_y, d.y)
    max_y = max(max_y, e.y)
    max_y = max(max_y, f.y)
    max_y = max(max_y, g.y)
    max_y = max(max_y, h.y)

    max_z = a.z
    max_z = max(max_z, b.z)
    max_z = max(max_z, c.z)
    max_z = max(max_z, d.z)
    max_z = max(max_z, e.z)
    max_z = max(max_z, f.z)
    max_z = max(max_z, g.z)
    max_z = max(max_z, h.z)

    lower_b[tid] = wp.vec3(min_x, min_y, min_z)
    upper_b[tid] = wp.vec3(max_x, max_y, max_z)


@wp.kernel
def build_tetrahedra_bb(
    vertices: wp.array(dtype=wp.vec3f),
    idx_tet: wp.array(dtype=wp.int32),
    lower_b: wp.array(dtype=wp.vec3),
    upper_b: wp.array(dtype=wp.vec3),
):

    tid = wp.tid()
    a = vertices[idx_tet[4 * tid + 0]]
    b = vertices[idx_tet[4 * tid + 1]]
    c = vertices[idx_tet[4 * tid + 2]]
    d = vertices[idx_tet[4 * tid + 3]]

    min_x = a.x
    min_x = min(min_x, b.x)
    min_x = min(min_x, c.x)
    min_x = min(min_x, d.x)

    min_y = a.y
    min_y = min(min_y, b.y)
    min_y = min(min_y, c.y)
    min_y = min(min_y, d.y)

    min_z = a.z
    min_z = min(min_z, b.z)
    min_z = min(min_z, c.z)
    min_z = min(min_z, d.z)

    max_x = a.x
    max_x = max(max_x, b.x)
    max_x = max(max_x, c.x)
    max_x = max(max_x, d.x)

    max_y = a.y
    max_y = max(max_y, b.y)
    max_y = max(max_y, c.y)
    max_y = max(max_y, d.y)

    max_z = a.z
    max_z = max(max_z, b.z)
    max_z = max(max_z, c.z)
    max_z = max(max_z, d.z)

    lower_b[tid] = wp.vec3(min_x, min_y, min_z)
    upper_b[tid] = wp.vec3(max_x, max_y, max_z)


def build_tet_bvh(vertices, idx_tet):

    N = idx_tet.shape[0] // 4

    lower_b = wp.empty(N, dtype=wp.vec3)
    upper_b = wp.empty(N, dtype=wp.vec3)

    wp.launch(build_tetrahedra_bb, N, inputs=[vertices, idx_tet, lower_b, upper_b])

    mesh_bvh = wp.Bvh(lower_b, upper_b)

    return mesh_bvh


def build_hex_bvh(vertices, idx_hex):

    N = idx_hex.shape[0] // 8

    lower_b = wp.empty(N, dtype=wp.vec3)
    upper_b = wp.empty(N, dtype=wp.vec3)

    wp.launch(build_hexahedra_bb, N, inputs=[vertices, idx_hex, lower_b, upper_b])

    mesh_bvh = wp.Bvh(lower_b, upper_b)

    return mesh_bvh


@wp.kernel
def cgns_one_to_zero_index(old_indices: wp.array(dtype=wp.int32), new_indices: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    new_indices[tid] = old_indices[tid] - 1


@wp.kernel
def concatenate_fwd_bwd(
    array_fwd: wp.array(dtype=Any), array_bwd: wp.array(dtype=Any), array_bwdfwd: wp.array(dtype=Any), stride: wp.int32
):

    N_bwd = stride
    (tid, stride_id) = wp.tid()

    # concanenate fwd and bwd
    # fwd = [11,12,13,14]
    # bwd = [11,10,9,8]
    # -> bwdfwd = [8,9,10,11,12,13,14]
    # we assume that first element of fwd and bwd are evaluated at the same time, so have to skip 1 element of fwd
    # also: bwd field is evaluated in "backwards time", so we want to reverse the time for the bwd field

    if tid < N_bwd:
        # backward fields
        array_bwdfwd[(2 * stride - 1) * stride_id + tid] = array_bwd[
            stride * stride_id + N_bwd - 1 - tid
        ]  # ex: 4 - 1 - 0 = 3

    elif tid == N_bwd:
        # skip
        _ = 1 + 1
    else:
        # forward fields
        array_bwdfwd[(2 * stride - 1) * stride_id + tid - 1] = array_fwd[
            stride * stride_id + tid - N_bwd
        ]  # ex: 5 - 4 -> 1


# @carb.profiler.profile
def advect_vector_field(
    vertices,
    idx_elem_fortran_indices,
    elem_type,
    vector_field_XYZ,
    initial_points,
    dt=0.1,
    num_steps=5,
    dt_MIN=0.001,
    dt_MAX=1.0,
    rk45_epsilon=0.005,
    device=0,
):
    carb.profiler.begin(1, "advect_vector_field pre warp kernel")
    wp.set_device(f"cuda:{device}")
    streamlines_paths = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.vec3f)
    streamlines_scalars = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.float32)
    streamlines_time = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.float32)
    idx_elem = wp.zeros(idx_elem_fortran_indices.shape, dtype=idx_elem_fortran_indices.dtype)
    wp.launch(cgns_one_to_zero_index, idx_elem_fortran_indices.shape, inputs=[idx_elem_fortran_indices, idx_elem])

    if ElementType(elem_type) == ElementType.TETRA_4:
        volume = build_tet_bvh(vertices, idx_elem)
    elif ElementType(elem_type) == ElementType.HEXA_8:
        volume = build_hex_bvh(vertices, idx_elem)
    else:
        pts_per_elem = -2
        raise RuntimeError(f"Unsupported element type: {elem_type}:{type(elem_type)}")

    # return (streamlines_paths, streamlines_scalars, streamlines_time)
    carb.profiler.end(1)
    carb.profiler.begin(1, "advect_vector_field warp kernel")
    # print("Advecting kernel")
    wp.launch(
        advect_vector_field_kernel,
        dim=initial_points.shape,
        inputs=[
            initial_points,
            dt,
            num_steps,
            vertices,
            idx_elem,
            elem_type,
            volume.id,
            vector_field_XYZ[0],
            vector_field_XYZ[1],
            vector_field_XYZ[2],
            streamlines_paths,
            streamlines_scalars,
            streamlines_time,
            dt_MIN,
            dt_MAX,
            rk45_epsilon,
        ],
        device=f"cuda:{device}",
    )
    # wp.synchronize_device()
    # print("Done advecting forwards kernel")
    streamlines_paths_bwd = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.vec3f)
    streamlines_scalars_bwd = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.float32)
    streamlines_time_bwd = wp.zeros(initial_points.shape[0] * num_steps, dtype=wp.float32)
    # print("Advecting backwards kernel")
    wp.launch(
        advect_vector_field_kernel,
        dim=initial_points.shape,
        inputs=[
            initial_points,
            -dt,
            num_steps,
            vertices,
            idx_elem,
            elem_type,
            volume.id,
            vector_field_XYZ[0],
            vector_field_XYZ[1],
            vector_field_XYZ[2],
            streamlines_paths_bwd,
            streamlines_scalars_bwd,
            streamlines_time_bwd,
            -dt_MIN,
            -dt_MAX,
            rk45_epsilon,
        ],
        device=f"cuda:{device}",
    )
    # wp.synchronize_device()
    # print("Done advecting forwards kernel")
    # concatenate the backward and forward solutions in forward time.
    streamlines_paths_fwdbwd = wp.zeros(initial_points.shape[0] * (2 * num_steps - 1), dtype=wp.vec3f)
    streamlines_scalars_fwdbwd = wp.zeros(initial_points.shape[0] * (2 * num_steps - 1), dtype=wp.float32)
    streamlines_time_fwdbwd = wp.zeros(initial_points.shape[0] * (2 * num_steps - 1), dtype=wp.float32)
    carb.profiler.end(1)
    carb.profiler.begin(1, "concatenate fwd-bwd warp kernel")
    wp.launch(
        concatenate_fwd_bwd,
        (2 * num_steps, initial_points.shape[0]),
        inputs=[streamlines_paths, streamlines_paths_bwd, streamlines_paths_fwdbwd, num_steps],
    )
    wp.launch(
        concatenate_fwd_bwd,
        (2 * num_steps, initial_points.shape[0]),
        inputs=[streamlines_scalars, streamlines_scalars_bwd, streamlines_scalars_fwdbwd, num_steps],
    )
    wp.launch(
        concatenate_fwd_bwd,
        (2 * num_steps, initial_points.shape[0]),
        inputs=[streamlines_time, streamlines_time_bwd, streamlines_time_fwdbwd, num_steps],
    )
    wp.synchronize_device()
    # print("Done concatenating streamlines")
    carb.profiler.end(1)
    return (streamlines_paths_fwdbwd, streamlines_scalars_fwdbwd, streamlines_time_fwdbwd)


@wp.kernel
def bounding_box_kernel(
    points: wp.array(dtype=wp.vec3f), min_point: wp.array(dtype=wp.vec3f), max_point: wp.array(dtype=wp.vec3f)
):
    tid = wp.tid()

    # Use atomic operations to find min and max points
    wp.atomic_min(min_point, 0, points[tid])
    wp.atomic_max(max_point, 0, points[tid])


def bounding_box(points: wp.array(dtype=wp.vec3f), device=0):

    min_point = wp.zeros((1,), dtype=wp.vec3f, device=f"cuda:{device}")
    max_point = wp.zeros((1,), dtype=wp.vec3f, device=f"cuda:{device}")
    # Launch the kernel
    wp.launch(kernel=bounding_box_kernel, dim=points.shape[0], inputs=[points, min_point, max_point], device="cuda")

    return np.array(min_point.numpy().tolist() + max_point.numpy().tolist())


def get_hex_data():
    vertices_hexes = wp.array(
        [
            # CGNS layout
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(1.0, 1.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 1.0),
            wp.vec3f(1.0, 1.0, 1.0),
            wp.vec3f(0.0, 1.0, 1.0),
            wp.vec3f(0.0, 0.0, 1.0),
            wp.vec3f(2.0, 0.0, 0.0),
            wp.vec3f(2.0, 1.0, 0.0),
            wp.vec3f(2.0, 1.0, 1.0),
            wp.vec3f(2.0, 0.0, 1.0),
        ],
        dtype=wp.vec3f,
    )

    idx_hex = wp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 0, 11, 10, 5, 4], dtype=wp.int32)

    field_hex = wp.array(
        [
            wp.vec3f(0.5, 0.1, 0.1),  # 0
            wp.vec3f(0.5, -0.1, 0.1),  # 1
            wp.vec3f(0.5, -0.1, 0.1),  # 2
            wp.vec3f(0.5, 0.1, 0.1),  # 3
            wp.vec3f(0.5, 0.1, -0.1),  # 4
            wp.vec3f(0.5, -0.1, -0.1),  # 5
            wp.vec3f(0.5, -0.1, -0.1),  # 6
            wp.vec3f(0.5, 0.1, -0.1),  # 7
            wp.vec3f(0.5, 0.1, 0.1),  # 8
            wp.vec3f(0.5, -0.1, 0.1),  # 9
            wp.vec3f(0.5, -0.1, -0.1),  # 10
            wp.vec3f(0.5, 0.1, -0.1),  # 11
        ],
        dtype=wp.vec3f,
    )

    initial_points_hex = wp.array(
        [
            # wp.vec3f(0.001,0.5,0.5),
            wp.vec3f(0.25, 0.25, 0.25),
            wp.vec3f(0.25, 0.75, 0.25),
            wp.vec3f(0.25, 0.25, 0.75),
            wp.vec3f(0.25, 0.75, 0.75),
        ],
        dtype=wp.vec3f,
    )

    return vertices_hexes, idx_hex, field_hex, initial_points_hex


def test_tet():
    import cupy as cp

    vertices = wp.array(
        [
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(1.0, 0.0, 0.0),
            wp.vec3f(0.0, 1.0, 0.0),
            wp.vec3f(0.0, 0.0, 1.0),
            wp.vec3f(1.0, 1.0, 1.0),
        ],
        dtype=wp.vec3f,
    )

    idx_tet = wp.array([0, 1, 2, 3, 1, 2, 3, 4], dtype=wp.int32)

    save_tetmesh_vtk(vertices.numpy(), idx_tet.numpy())

    wp.launch(kernel=test_point_in_tetrahedra, dim=(1,), inputs=[vertices, idx_tet])
    # sys.exit(0)
    # p = wp.vec3f(0.1, 0.1, 0.1)
    # p2 = wp.vec3f(1.1, 0.1, 0.1)
    # p3 = wp.vec3f(0.5, 0.5, 0.5)
    # assert(point_in_tetrahedra(p, vertices, idx_tet) is True)
    # assert(point_in_tetrahedra(p2, vertices, idx_tet) is False)
    # print("p3=", point_in_tetrahedra(p2, vertices, idx_tet))

    N = 10000
    p3 = cp.random.random((N, 3), dtype=cp.float32)
    p3wp = wp.from_numpy(p3.get(), dtype=wp.vec3)

    results = wp.zeros(N, dtype=wp.int32)

    wp.launch(kernel=test_tetrahedra_cube_ratio, dim=(N,), inputs=[p3wp, vertices, idx_tet, results])

    print("Ratio=", cp.sum(cp.array(results)) / N, "vs ", 1 / 6)

    field = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=wp.float32)
    vector_field = wp.array(
        [
            wp.vec3f(1.0, 1.0, 1.0),
            wp.vec3f(2.0, 2.0, 2.0),
            wp.vec3f(3.0, 3.0, 3.0),
            wp.vec3f(4.0, 4.0, 4.0),
            wp.vec3f(5.0, 5.0, 5.0),
        ],
        dtype=wp.vec3f,
    )
    initial_points = wp.array([wp.vec3f(0.2, 0.2, 0.2), wp.vec3f(0.6, 0.6, 0.6)], dtype=wp.vec3f)
    # wp.launch(kernel=test_tetrahedra_interpolation, dim=(1,), inputs=[vertices, idx_tet, field])
    # wp.launch(kernel=test_tetrahedra_vec_interpolation, dim=(1,), inputs=[vertices, idx_tet, vector_field])

    # test_tet_bb(vertices, idx_tet)

    solutions = advect_vector_field(vertices, idx_tet, "TETRA_4", vector_field, initial_points, dt=0.01)


def test_hex():

    vertices_hexes, idx_hex, field_hex, initial_points_hex = get_hex_data()

    wp.launch(kernel=test_point_in_hexahedra, dim=(1,), inputs=[vertices_hexes, idx_hex])

    num_steps = 10
    (solutions_hex_steps, solutions_hex_scalars) = advect_vector_field(
        vertices_hexes, idx_hex, "HEXA_8", field_hex, initial_points_hex, dt=0.4, num_steps=num_steps
    )

    print(f"solutions_hex={solutions_hex_steps}")

    # stage = Usd.Stage.CreateNew('test_2hex_streamlines.usda')

    # idx_hull = generate_hull(idx_hex, "HEXA_8")
    # hull_mesh_to_stage(vertices_hexes.numpy(), idx_hull.numpy(), None, stage, "/World", 4)

    # print("solutions_hex=", solutions_hex)
    # make_basis_curve(stage, solutions_hex.numpy(), num_steps * np.ones(4), 4, width=0.02,path="/World/streamlines")

    # stage.Save()


if __name__ == "__main__":
    test_hex()
