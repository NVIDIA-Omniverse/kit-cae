# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import zlib
from logging import getLogger
from typing import Any, Union

import numpy as np
import warp as wp
from pxr import Gf, Usd
from warp.context import Device
from warp.types import DType, vector_types

from .bindings import IFieldArray
from .typing import FieldArrayLike

logger = getLogger(__name__)


def get_device(array: FieldArrayLike) -> Device:
    if isinstance(array, IFieldArray):
        if array.device_id == -1:
            return wp.get_device("cpu")
        else:
            return wp.get_cuda_device(array.device_id)
    elif isinstance(array, wp.array):
        return array.device
    elif hasattr(array, "__cuda_array_interface__"):
        # FIXME: we'll need to fix this to work correctly for multi-gpu
        return wp.get_cuda_device(0)
    elif hasattr(array, "__array_interface__"):
        return wp.get_device("cpu")
    raise RuntimeError("Cannot determine device %s!" % type(array))


def to_warp_dtype(array: FieldArrayLike) -> DType:
    """
    Returns the warp DType for the given array. This also handles 2D arrays
    by returning the appropriate vector type.
    """
    if isinstance(array, wp.array):
        return array.dtype
    elif isinstance(array, np.ndarray):
        scalar_dtype = wp.dtype_from_numpy(array.dtype)
    elif isinstance(array, IFieldArray):
        scalar_dtype = wp.dtype_from_numpy(array.dtype)
    else:
        raise RuntimeError("Cannot determine warp_dtype!")

    if array.ndim == 2:
        type_args = {"length": array.shape[1], "dtype": scalar_dtype}
        for vtype in vector_types:
            if vtype._wp_type_args_ == type_args:
                return vtype
    return scalar_dtype


def as_warp_array(array: FieldArrayLike) -> Union[wp.array, None]:
    """
    Returns a zero-copied warp.array from any object that supports
    the CUDA Array Interface or NumPy Array Interface.

    The returned array is hosted on the same device as the input array since
    this function does not copy the array.
    """
    if array is None:
        return None

    if isinstance(array, wp.array):
        return array

    return wp.array(data=array, copy=False, dtype=to_warp_dtype(array), device=get_device(array))


def as_numpy_array(array: FieldArrayLike) -> np.ndarray:
    if array is None:
        return None
    elif isinstance(array, np.ndarray):
        return array
    elif isinstance(array, wp.array):
        return array.numpy()
    else:
        device = get_device(array)
        if device.is_cpu:
            return np.asarray(array)
        else:
            return wp.array(array, copy=False, device=device).numpy()


def to_warp_array(array: FieldArrayLike, copy=False) -> wp.array:
    """
    Unlike as_warp_array, this function will does not change the `device`
    thus active device is used to create the warp.array.
    """
    if isinstance(array, wp.array):
        return array
    else:
        return wp.array(array, dtype=to_warp_dtype(array), copy=copy)


@wp.kernel
def _map_colors_kernel(
    input: wp.array(dtype=wp.float32),
    rgba_points: wp.array(ndim=2, dtype=wp.float32),
    x_points: wp.array(dtype=wp.float32),
    domain_min: wp.float32,
    domain_max: wp.float32,
    rgba: wp.array(ndim=2, dtype=wp.float32),
):
    tid = wp.tid()
    v = wp.clamp(input[tid], domain_min, domain_max)
    normalized_v = (v - domain_min) / (domain_max - domain_min)
    for i in range(x_points.shape[0] - 1):
        if normalized_v >= x_points[i] and normalized_v <= x_points[i + 1]:
            t = (normalized_v - x_points[i]) / (x_points[i + 1] - x_points[i])
            for c in range(4):
                rgba[tid][c] = wp.lerp(rgba_points[i][c], rgba_points[i + 1][c], t)
            break


def map_to_rgba(array: FieldArrayLike, colormap: Usd.Prim, timeCode: Usd.TimeCode) -> np.ndarray:
    input = wp.array(array, dtype=wp.float32)
    rgba = wp.zeros(shape=[array.shape[0], 4], dtype=wp.float32)

    rgba_points = wp.array(np.array(colormap.GetAttribute("rgbaPoints").Get(), dtype=np.float32))
    x_points = wp.array(np.array(colormap.GetAttribute("xPoints").Get(), dtype=np.float32))
    domain_min = colormap.GetAttribute("domain").Get(timeCode)[0]
    domain_max = colormap.GetAttribute("domain").Get(timeCode)[1]
    logger.info("[map_to_rgba]: using range (%f, %f)", domain_min, domain_max)

    wp.launch(
        _map_colors_kernel, dim=input.shape[0], inputs=[input, rgba_points, x_points, domain_min, domain_max, rgba]
    )

    return rgba.numpy()


def get_nanovdb(volume: wp.Volume) -> wp.array:
    """
    Volume.array() returns an array of dtype uint8 which can overflow for large volumes.
    This function is similar to that except returns a uint64 array instead which extends supportable
    volume size.
    """
    import ctypes

    buf = ctypes.c_void_p(0)
    size = ctypes.c_uint64(0)
    volume.runtime.core.volume_get_buffer_info(volume.id, ctypes.byref(buf), ctypes.byref(size))

    def deleter(_1, _2):
        vol = volume
        del vol

    return wp.array(ptr=buf.value, dtype=wp.uint64, shape=size.value // 8, device=volume.device, deleter=deleter)


def stack(arrays: list[FieldArrayLike]) -> IFieldArray:

    if len(arrays) == 0:
        return None

    if len(arrays) == 1:
        return arrays[0]

    device = get_device(arrays[0])

    if not all(get_device(a) == device for a in arrays):
        raise ValueError("All arrays must be on the same device")

    if not all(a.dtype == arrays[0].dtype for a in arrays):
        raise ValueError("All arrays must be of the same dtype")

    if not all(a.ndim == 1 for a in arrays):
        raise ValueError("All arrays must be of the same dimensionality (ndim == 1)")

    if not all(a.shape[0] == arrays[0].shape[0] for a in arrays):
        raise ValueError("All arrays must have the same length")

    if device.is_cpu:
        return IFieldArray.from_numpy(np.vstack(arrays).transpose())
    else:
        raise RuntimeError("Not implemented yet!")


def at(array: FieldArrayLike, index) -> Any:
    device = get_device(array)
    if device.is_cpu:
        return np.asarray(array)[index]
    else:
        wp_array = wp.array(array, copy=False, device=device)
        subarray = wp_array[index : index + 1]
        return subarray.numpy()[0]


def add(a: FieldArrayLike, value: Any) -> FieldArrayLike:
    device = get_device(a)
    if device.is_cpu:
        return np.asarray(a) + value
    else:
        raise RuntimeError("Not implemented yet!")


def lookup_index_0(array: FieldArrayLike, index_array: FieldArrayLike) -> FieldArrayLike:
    device = get_device(array)
    if device.is_cpu:
        return as_numpy_array(array)[as_numpy_array(index_array)]
    else:
        raise RuntimeError("Not implemented yet!")


def lookup_index_1(array: FieldArrayLike, index_array: FieldArrayLike) -> FieldArrayLike:
    device = get_device(array)
    if device.is_cpu:
        return as_numpy_array(array)[as_numpy_array(index_array) - 1]
    else:
        raise RuntimeError("Not implemented yet!")


def compute_quaternions_from_directions_usd(directions):
    """
    Given a NumPy array of shape (N, 3) representing 3D direction vectors,
    compute quaternions (w, x, y, z) using OpenUSD's Gf.Rotation.

    Parameters:
        directions (np.ndarray): Array of shape (N, 3) with direction vectors.

    Returns:
        np.ndarray: Array of shape (N, 4) containing quaternions as (w, x, y, z).
    """
    directions = np.array(directions, dtype=np.float32, copy=False)
    print(directions)

    # # Normalize direction vectors
    # norms = np.linalg.norm(directions, axis=1, keepdims=True)
    # directions = np.divide(directions, norms, out=np.zeros_like(directions), where=(norms != 0))

    default_forward = Gf.Vec3d(1, 0, 0)  # Reference direction X-axis
    # default_forward = Gf.Vec3d(0, 0, 1)  # Reference direction Z-axis

    quaternions = []
    for dir_vec in directions:
        target_direction = Gf.Vec3d(*dir_vec.tolist())  # Convert to Gf.Vec3f
        target_direction.Normalize()

        rotation = Gf.Rotation(target_direction, default_forward)  # Compute rotation
        quat = rotation.GetQuat()  # Get quaternion (returns Gf.Quatf)
        # quat.Normalize()

        # Convert to tuple (x, y, z, w) and store
        quaternions.append((*quat.GetImaginary(), quat.GetReal()))

    return np.array(quaternions, dtype=np.float32)  # Shape (N, 4)


def compute_quaternions_from_directions(directions: FieldArrayLike) -> np.ndarray:
    assert (
        directions.ndim == 2 and directions.shape[1] == 3
    ), f"Expected shape (N, 3), got {directions.shape}, {directions.dtype}"

    directions = as_numpy_array(directions).astype(np.float32, copy=False)

    # Normalize direction vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    mask = norms != 0

    inv_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=mask)
    half_vecs = directions * inv_norms
    half_vecs[:, 0] += 1.0

    half_norms = np.linalg.norm(half_vecs, axis=1, keepdims=True)
    half_vecs = np.divide(half_vecs, half_norms, out=np.zeros_like(half_vecs), where=(half_norms != 0))

    sine_axis = np.zeros_like(half_vecs)
    sine_axis[:, 1] = -half_vecs[:, 2]
    sine_axis[:, 2] = half_vecs[:, 1]
    cos_angle = half_vecs[:, 0]

    # note the stackign order. this is the order expected for Vt.QuathArrayFromBuffer
    return np.column_stack((sine_axis, cos_angle))


def checksum(array: FieldArrayLike) -> int:
    if hasattr(array, "__cuda_array_interface__"):
        raise RuntimeError("CUDA arrays are not supported!")
    else:
        return zlib.crc32(as_numpy_array(array).tobytes())
    # raise ValueError("Array does not support CUDA Array Interface or Array Interface!")
