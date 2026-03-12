# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import ctypes
import zlib
from logging import getLogger
from typing import Any, Union

import numpy as np
import warp as wp
from pxr import Gf, Usd
from usdrt import Vt as VtRT
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
    volume.runtime.core.wp_volume_get_buffer_info(volume.id, ctypes.byref(buf), ctypes.byref(size))

    def deleter(_1, _2):
        vol = volume
        del vol

    return wp.array(ptr=buf.value, dtype=wp.uint64, shape=size.value // 8, device=volume.device, deleter=deleter)


def get_nanovdb_as_field_array(volume: wp.Volume) -> IFieldArray:
    """
    Volume.array() returns an array of dtype uint8 which can overflow for large volumes.
    This function is similar to that except returns a uint64 array instead which extends supportable
    volume size.
    """

    class CAIBuffer:
        def __init__(self, volume: wp.Volume):
            logger.info(f"Creating CAIBuffer for volume {volume.id}")
            self._volume = volume

        def __del__(self):
            logger.info(f"CAIBuffer for volume {self._volume.id} is being deleted, releasing buffer reference.")

        @property
        def __cuda_array_interface__(self):
            try:
                buf = ctypes.c_void_p(0)
                size = ctypes.c_uint64(0)
                self._volume.runtime.core.wp_volume_get_buffer_info(
                    self._volume.id, ctypes.byref(buf), ctypes.byref(size)
                )

                return {
                    "version": 3,
                    "data": (buf.value, False),
                    "shape": (size.value // 8,),
                    "typestr": "|u8",
                    "strides": (8,),
                    "mask": None,
                    "stream": None,
                }
            except Exception as e:
                logger.exception(f"Failed to get __cuda_array_interface__ for volume: {e}")
                raise e

    buf = CAIBuffer(volume)
    return IFieldArray.from_array(buf)


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


def column_stack(arrays: list[FieldArrayLike]) -> FieldArrayLike:

    if len(arrays) == 0:
        return None

    if len(arrays) == 1:
        return arrays[0]

    device = get_device(arrays[0])

    if not all(get_device(a) == device for a in arrays):
        raise ValueError("All arrays must be on the same device")

    if device.is_cpu:
        return IFieldArray.from_numpy(np.column_stack(arrays))
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
    # print(directions)

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


def get_scalar_array(array_or_arrays: Union[FieldArrayLike, list[FieldArrayLike]]) -> FieldArrayLike:
    """Return a 1 component array. For multipe components arrays, this returns its magnitude."""

    if array_or_arrays is None:
        raise ValueError("Input array cannot be None!")

    if isinstance(array_or_arrays, list):
        np_array = as_numpy_array(column_stack(array_or_arrays))
    else:
        np_array = as_numpy_array(array_or_arrays)

    if np_array.ndim == 1:
        return np_array
    elif np_array.ndim == 2 and np_array.shape[1] == 1:
        return np_array.ravel()
    elif np_array.ndim == 2 and np_array.shape[1] > 1:
        # compute magnitudes
        return np.linalg.norm(np_array, axis=1)
    else:
        raise ValueError(f"Cannot convert array of shape {np_array.shape} to scalar array!")


@wp.func
def _remap_value(
    v: Any, domain_min: wp.float32, domain_max: wp.float32, data_range_min: wp.float32, data_range_max: wp.float32
) -> Any:
    cast_domain_min = type(v)(domain_min)
    cast_domain_max = type(v)(domain_max)
    v = wp.clamp(v, cast_domain_min, cast_domain_max)
    f32_normalized_v = wp.float32(v - cast_domain_min) / wp.float32(cast_domain_max - cast_domain_min)
    f32_out_v = f32_normalized_v * (data_range_max - data_range_min) + data_range_min
    return type(v)(f32_out_v)


@wp.func
def _remap_value_vector(
    v: Any, domain_min: wp.float32, domain_max: wp.float32, data_range_min: wp.float32, data_range_max: wp.float32
) -> Any:
    for comp in range(v.shape[1]):
        v[comp] = _remap_value(v[comp], domain_min, domain_max, data_range_min, data_range_max)
    return v


@wp.kernel
def _remap_array_kernel_scalar(
    input: wp.array(dtype=Any),
    domain_min: wp.float32,
    domain_max: wp.float32,
    data_range_min: wp.float32,
    data_range_max: wp.float32,
    output: wp.array(dtype=Any),
):
    tid = wp.tid()
    in_v = input[tid]
    output[tid] = _remap_value(in_v, domain_min, domain_max, data_range_min, data_range_max)


@wp.kernel
def _remap_array_kernel_vector(
    input: wp.array(dtype=Any),
    domain_min: wp.float32,
    domain_max: wp.float32,
    data_range_min: wp.float32,
    data_range_max: wp.float32,
    output: wp.array(dtype=Any),
):
    tid = wp.tid()
    output[tid] = _remap_value_vector(input[tid], domain_min, domain_max, data_range_min, data_range_max)


def remap_array(array: FieldArrayLike, domain: tuple[float, float], data_range: tuple[float, float]) -> FieldArrayLike:
    """
    Remaps an array from one domain to another. The values in array are clamped to the domain before remapping.
    If the domain is invalid (domain[0] >= domain[1]), the array is returned unchanged.
    """
    if domain[0] > domain[1]:
        logger.error(f"Invalid domain {domain}.")
        return array

    if array.ndim > 2:
        raise ValueError(f"Unsupported array of shape {array.shape}!")

    in_array = wp.array(array, copy=False, device=get_device(array))
    out_array = wp.zeros_like(in_array, device=in_array.device)
    wp.launch(
        _remap_array_kernel_scalar if in_array.ndim == 1 else _remap_array_kernel_vector,
        dim=in_array.shape[0],
        inputs=[in_array, domain[0], domain[1], data_range[0], data_range[1]],
        outputs=[out_array],
        device=in_array.device,
    )

    return IFieldArray.from_array(out_array)


def to_vtrt_array(farray: IFieldArray):
    """
    Converts a IFieldArray to a UsdRT.Array.
    """
    assert isinstance(farray, IFieldArray), f"Expected IFieldArray, got {type(farray)}"

    # VtRT.Array only supports __cuda_array_interface__ or ndarray (and not __numpy_array_interface__)
    # So we need to convert the IFieldArray to a numpy array explicitly here.
    if get_device(farray).is_cpu:
        cupy_or_np_array = farray.numpy()
    else:
        # cupy_or_np_array = farray
        # FORCE copy to CPU: this should not be necessary, but VtRt.Array
        # and direct GPU array copy is giving me issues for we pass through CPU for now.
        cupy_or_np_array = farray.to_device(-1).numpy()

    if farray.ndim == 1 or farray.ndim == 2 and farray.shape[1] == 1:
        if hasattr(cupy_or_np_array, "reshape"):
            cupy_or_np_array = cupy_or_np_array.reshape(-1, 1)

        # scalar array
        match farray.dtype:
            case np.int32:
                return VtRT.IntArray(cupy_or_np_array)
            case np.int64:
                return VtRT.Int64Array(cupy_or_np_array)
            case np.uint32:
                return VtRT.UIntArray(cupy_or_np_array)
            case np.uint64:
                return VtRT.UInt64Array(cupy_or_np_array)
            case np.float32:
                return VtRT.FloatArray(cupy_or_np_array)
            case np.float64:
                return VtRT.DoubleArray(cupy_or_np_array)
            case _:
                raise ValueError(f"Unsupported dtype {farray.dtype} for scalar array of shape {farray.shape}!")
    elif farray.ndim == 2 and farray.shape[1] == 2:
        # vector array
        match farray.dtype:
            case np.int32:
                return VtRT.Int2Array(cupy_or_np_array)
            case np.uint32:
                return VtRT.UInt2Array(cupy_or_np_array)
            case np.float32:
                return VtRT.Vec2fArray(cupy_or_np_array)
            case np.float64:
                return VtRT.Vec2dArray(cupy_or_np_array)
            case _:
                raise ValueError(f"Unsupported dtype {farray.dtype} for vector array of shape {farray.shape}!")
    elif farray.ndim == 2 and farray.shape[1] == 3:
        # vector array
        match farray.dtype:
            case np.int32:
                return VtRT.Vec3iArray(cupy_or_np_array)
            case np.float32:
                return VtRT.Vec3fArray(cupy_or_np_array)
            case np.float64:
                return VtRT.Vec3dArray(cupy_or_np_array)
            case _:
                raise ValueError(f"Unsupported dtype {farray.dtype} for vector array of shape {farray.shape}!")
    elif farray.ndim == 2 and farray.shape[1] == 4:
        # vector array
        match farray.dtype:
            case np.int32:
                return VtRT.Vec4iArray(cupy_or_np_array)
            case np.float32:
                return VtRT.Vec4fArray(cupy_or_np_array)
            case np.float64:
                return VtRT.Vec4dArray(cupy_or_np_array)
            case _:
                raise ValueError(f"Unsupported dtype {farray.dtype} for vector array of shape {farray.shape}!")
    else:
        raise ValueError(f"Unsupported array of shape {farray.shape}!")


@wp.kernel
def _as_type_kernel_scalar(
    input: wp.array(dtype=Any),
    output: wp.array(dtype=Any),
    ndim: int,
):
    tid = wp.tid()
    output[tid] = type(output[tid])(input[tid])


@wp.kernel
def _as_type_kernel_vector(
    input: wp.array(dtype=Any),
    output: wp.array(dtype=Any),
    ncomponents: int,
):
    tid = wp.tid()
    for i in range(ncomponents):
        output[tid][i] = type(output[tid][i])(input[tid][i])


def as_type(array: IFieldArray, dtype: np.dtype) -> IFieldArray:
    if array.dtype == dtype:
        return array
    assert array.ndim in [1, 2], f"Expected array of ndim 1 or 2, got {array.ndim}"

    wp_array = wp.array(array, copy=False, device=get_device(array))
    new_wp_array = wp.zeros(dtype=wp.dtype_from_numpy(dtype), shape=wp_array.shape, device=wp_array.device)
    wp.launch(
        _as_type_kernel_scalar if wp_array.ndim == 1 else _as_type_kernel_vector,
        dim=wp_array.shape[0],
        inputs=[wp_array, new_wp_array, wp_array.shape[1] if wp_array.ndim == 2 else 0],
    )
    return IFieldArray.from_array(new_wp_array)


@wp.kernel(enable_backward=False)
def _histogram_kernel(
    data: wp.array(dtype=Any),
    counts: wp.array(dtype=wp.int32),
    min_val: wp.float64,
    inv_bin_width: wp.float64,
    num_bins: wp.int32,
):
    tid = wp.tid()
    val = wp.float64(data[tid])
    bin_idx = wp.int32((val - min_val) * inv_bin_width)
    # Skip values outside the [min, max) range entirely
    if bin_idx >= wp.int32(0) and bin_idx < num_bins:
        wp.atomic_add(counts, bin_idx, wp.int32(1))


@wp.kernel(enable_backward=False)
def _sum_kernel(
    data: wp.array(dtype=Any),
    result: wp.array(dtype=wp.float64),
):
    tid = wp.tid()
    wp.atomic_add(result, 0, wp.float64(data[tid]))


def get_scalar_stats(array: FieldArrayLike, num_bins: int = 32) -> dict:
    """Compute histogram, mean, and approximate percentiles for a scalar array using Warp.

    Returns a dict with keys: "counts", "bin_edges", "mean", "min", "max",
    "median", "q1", "q2", "q3", "q4" (quartiles as (lo, hi) tuples).
    """
    device = get_device(array)
    wp_array = wp.array(array, copy=False, device=device)

    if wp_array.ndim == 2:
        wp_array = wp_array.reshape((wp_array.shape[0],))

    n = wp_array.shape[0]

    # Get min/max from the existing range utility (already Warp-based)
    ranges = get_componentwise_ranges(array)
    val_min, val_max = ranges[0]

    # Histogram via Warp
    bin_width = (val_max - val_min) / num_bins if val_max > val_min else 1.0
    inv_bin_width = 1.0 / bin_width if bin_width > 0 else 0.0
    wp_counts = wp.zeros((num_bins,), dtype=wp.int32, device=device)
    wp.launch(
        _histogram_kernel,
        dim=n,
        inputs=[wp_array, wp_counts, wp.float64(val_min), wp.float64(inv_bin_width), wp.int32(num_bins)],
        device=device,
    )
    counts = wp_counts.numpy()

    # Bin edges (computed on CPU, trivial)
    bin_edges = np.linspace(val_min, val_max, num_bins + 1)

    # Sum via Warp for mean
    wp_sum = wp.zeros((1,), dtype=wp.float64, device=device)
    wp.launch(_sum_kernel, dim=n, inputs=[wp_array, wp_sum], device=device)
    mean_val = float(wp_sum.numpy()[0]) / n

    # Approximate percentiles from cumulative histogram
    cumsum = np.cumsum(counts).astype(np.float64)

    def _percentile_from_hist(p):
        target = p * n
        idx = np.searchsorted(cumsum, target, side="left")
        idx = min(idx, num_bins - 1)
        # Linear interpolation within the bin
        prev_count = cumsum[idx - 1] if idx > 0 else 0.0
        bin_frac = (target - prev_count) / max(float(counts[idx]), 1.0)
        return float(bin_edges[idx] + bin_frac * bin_width)

    p25 = _percentile_from_hist(0.25)
    p50 = _percentile_from_hist(0.50)
    p75 = _percentile_from_hist(0.75)

    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "mean": mean_val,
        "min": val_min,
        "max": val_max,
        "median": p50,
        "q1": (val_min, p25),
        "q2": (p25, p50),
        "q3": (p50, p75),
        "q4": (p75, val_max),
    }


def compute_histogram(array: FieldArrayLike, num_bins: int, range_min: float, range_max: float) -> dict:
    """Compute histogram with a specified range using Warp.

    Returns dict with "counts" and "bin_edges".
    """
    device = get_device(array)
    wp_array = wp.array(array, copy=False, device=device)

    if wp_array.ndim == 2:
        wp_array = wp_array.reshape((wp_array.shape[0],))

    n = wp_array.shape[0]
    bin_width = (range_max - range_min) / num_bins if range_max > range_min else 1.0
    inv_bin_width = 1.0 / bin_width if bin_width > 0 else 0.0
    wp_counts = wp.zeros((num_bins,), dtype=wp.int32, device=device)
    wp.launch(
        _histogram_kernel,
        dim=n,
        inputs=[wp_array, wp_counts, wp.float64(range_min), wp.float64(inv_bin_width), wp.int32(num_bins)],
        device=device,
    )

    return {
        "counts": wp_counts.numpy().tolist(),
        "bin_edges": np.linspace(range_min, range_max, num_bins + 1).tolist(),
    }


def get_componentwise_ranges_kernel(ndim: int, ncomps: int):
    if wp.static(ndim == 1):

        # scalar array
        @wp.kernel(enable_backward=False)
        def _componentwise_ranges_kernel_scalar(
            input: wp.array(dtype=Any),
            output: wp.array(dtype=Any),
        ):
            tid = wp.tid()
            v = input[tid]
            wp.atomic_min(output, 0, v)
            wp.atomic_max(output, 1, v)

        return _componentwise_ranges_kernel_scalar
    else:

        @wp.kernel(enable_backward=False)
        def _componentwise_ranges_kernel_vector(
            input: wp.array(ndim=2, dtype=Any),
            output: wp.array(ndim=2, dtype=Any),
        ):
            tid = wp.tid()
            for comp in range(wp.static(ncomps)):
                v = input[tid][comp]
                wp.atomic_min(output, 0, comp, v)
                wp.atomic_max(output, 1, comp, v)

        return _componentwise_ranges_kernel_vector


def get_componentwise_ranges(array: FieldArrayLike) -> list[tuple[float, float]]:
    """
    Get the component-wise ranges of an array. For scalar arrays, this returns a list with a single tuple.
    For vector arrays, this returns a list of tuples corresponding to the range of each component.
    """
    wp_array = wp.array(array, copy=False, device=get_device(array))

    if wp_array.ndim == 1:
        ncomps = 1
    elif wp_array.ndim == 2:
        ncomps = wp_array.shape[1]
    else:
        raise ValueError(f"Unsupported array of shape {wp_array.shape}!")

    kernel = get_componentwise_ranges_kernel(wp_array.ndim, ncomps)

    zero_val = wp_array[0:1].numpy()
    wp_ranges = wp.array(np.concatenate((zero_val, zero_val), axis=0), device=wp_array.device)
    wp.launch(kernel, dim=wp_array.shape[0], inputs=[wp_array], outputs=[wp_ranges], device=wp_array.device)

    np_ranges = wp_ranges.numpy()
    ranges = []
    for comp in range(ncomps):
        comp_min = np_ranges[0][comp].item() if ncomps > 1 else np_ranges[0].item()
        comp_max = np_ranges[1][comp].item() if ncomps > 1 else np_ranges[1].item()
        ranges.append((comp_min, comp_max))

    return ranges
