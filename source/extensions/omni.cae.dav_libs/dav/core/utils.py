# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = [
    # Type specific utilities
    "get_limits",
    "get_scalar_dtype",
    "get_vector_dtype",
    "get_vector_length",
    "is_floating_point_dtype",
    "is_integral_dtype",
    "is_vector_dtype",
    # Array utils
    "array_scan",
    # Misc utils
    "get_div_by_i32_function",
    "get_mul_by_f32_function",
    "get_cast_function",
    "set_qualname",
    "func",
    "kernel",
    "assert_static_resolved",
]

from logging import getLogger

import numpy as np
import warp as wp
from warp import utils as wp_utils

from . import config

logger = getLogger(__name__)

# ======================================================================================================
# Utilities for working with data model dtypes, including vector dtypes.
# ======================================================================================================


def is_vector_dtype(wp_dtype):
    """Check if a wp_dtype is a vector dtype."""

    if not hasattr(wp_dtype, "_length_"):
        raise ValueError(f"Unsupported dtype: {wp_dtype}.")
    return wp_dtype._length_ > 1


def is_scalar_dtype(wp_dtype):
    """Check if a wp_dtype is a scalar dtype."""
    return not is_vector_dtype(wp_dtype)


def get_scalar_dtype(wp_dtype):
    """Get the scalar wp_dtype of a vector dtype."""
    if is_vector_dtype(wp_dtype):
        return wp_dtype._wp_scalar_type_
    else:
        return wp_dtype


def get_vector_dtype(dtype, length: int = 3):
    """Get the vector dtype of a scalar dtype for a given length. Length must be greater than 0.
    For length 1, the scalar dtype is returned.  This prefers to return the most specific vector
    dtype for the given length and dtype.  For example, `wp.vec3f` is preferred over `wp.vec(length=3, dtype=wp.float32)`
    for length 3 and `wp.float32`.
    """
    assert not is_vector_dtype(dtype), "Cannot get vector dtype of a vector dtype"
    assert length > 0, "Length must be greater than 0"
    if length == 1:
        return dtype
    elif length == 2:
        match dtype:
            case wp.float32:
                return wp.vec2f
            case wp.float64:
                return wp.vec2d
            case wp.int32:
                return wp.vec2i
            case wp.int64:
                return wp.vec2l
            case wp.uint32:
                return wp.vec2ui
            case wp.uint64:
                return wp.vec2ul
    elif length == 3:
        match dtype:
            case wp.float32:
                return wp.vec3f
            case wp.float64:
                return wp.vec3d
            case wp.int32:
                return wp.vec3i
            case wp.int64:
                return wp.vec3l
            case wp.uint32:
                return wp.vec3ui
            case wp.uint64:
                return wp.vec3ul
    elif length == 4:
        match dtype:
            case wp.float32:
                return wp.vec4f
            case wp.float64:
                return wp.vec4d
            case wp.int32:
                return wp.vec4i
            case wp.int64:
                return wp.vec4l
            case wp.uint32:
                return wp.vec4ui
            case wp.uint64:
                return wp.vec4ul
    else:
        return wp.vec(length=length, dtype=dtype)


def get_vector_length(dtype):
    """Get the length of a vector dtype."""
    if is_vector_dtype(dtype):
        return int(dtype._length_)
    else:
        return 1


def get_limits(dtype):
    """Get the limits of a dtype."""
    dtype = get_scalar_dtype(dtype)
    if dtype == wp.float32:
        finfo = np.finfo(np.float32)
        return (float(finfo.min), float(finfo.max))
    elif dtype == wp.float64:
        finfo = np.finfo(np.float64)
        return (float(finfo.min), float(finfo.max))
    elif dtype == wp.int32:
        iinfo = np.iinfo(np.int32)
        return (int(iinfo.min), int(iinfo.max))
    elif dtype == wp.int64:
        iinfo = np.iinfo(np.int64)
        return (int(iinfo.min), int(iinfo.max))
    elif dtype == wp.uint32:
        iinfo = np.iinfo(np.uint32)
        return (int(iinfo.min), int(iinfo.max))
    elif dtype == wp.uint64:
        iinfo = np.iinfo(np.uint64)
        return (int(iinfo.min), int(iinfo.max))
    else:
        raise ValueError(f"Unsupported dtype: {dtype} for limits")


def is_floating_point_dtype(dtype):
    """Check if a dtype is a floating point dtype."""
    dtype = get_scalar_dtype(dtype)
    return dtype in [wp.float32, wp.float64]


def is_integral_dtype(dtype):
    """Check if a dtype is an integral dtype."""
    dtype = get_scalar_dtype(dtype)
    return dtype in [wp.int32, wp.int64, wp.uint32, wp.uint64]


# ======================================================================================================
# Decorator utilities for Warp


def set_qualname(qualname: str):
    """
    Decorator to set the __qualname__ of a function or class. This is useful for improving the readability of function and class names in Warp kernels, which can otherwise have mangled names due to being defined inside other functions.

    Args:
        qualname: The qualified name to set on the decorated function or class.

    Returns:
        A decorator that sets the __qualname__ of the decorated function or class to the specified value.
    """

    def decorator(obj):
        obj.__qualname__ = qualname
        return obj

    return decorator


def assert_static_resolved(obj):
    """Guard decorator that raises RuntimeError at definition time if Warp failed to build
    the static evaluation context for the decorated function or kernel.

    Place this ABOVE @wp.func or @wp.kernel so it runs after Warp has processed the
    function and set the has_unresolved_static_expressions flag:

        @dav.utils.assert_static_resolved
        @func
        def my_func(cell: CellHandle) -> wp.bool:
            if wp.static(some_condition):
                ...

    Two distinct failure modes can set has_unresolved_static_expressions=True:

    1. get_static_evaluation_context() itself fails (e.g. a @wp.func method with an
       empty closure cell because it references its enclosing class by name mid-class-body).
       adj.static_expressions stays completely empty: no expression was even attempted.
       This causes non-deterministic module hashes and Warp binary cache misses.

    2. the evaluation context built fine but specific wp.static() expressions could not
       be resolved at decoration time because they depend on loop-local variables (e.g.
       `wp.static(ARRAY[idx])` where idx is a for-loop variable). Warp explicitly handles
       this at compile time; adj.static_expressions will be non-empty because the simpler
       wp.static() calls in the same function DID resolve.

    Both cause kernel hash to potentially change. Second affects the hash if unique kernels were
     invoked in different order across runs. So we avoid both in DAV.

    Raises:
        RuntimeError: If any adjoint has has_unresolved_static_expressions=True
    """

    def _check(adj, name):
        if adj is not None and adj.has_unresolved_static_expressions:
            raise RuntimeError(
                f"@wp.func/@wp.kernel '{name}' has unresolved wp.static() expressions and "
                f"adj.static_expressions is empty — Warp could not build the static evaluation "
                f"context. This causes non-deterministic kernel content hashes and Warp binary "
                f"cache misses. Ensure all closure variables referenced in the function body are "
                f"bound before @wp.func/@wp.kernel is applied (e.g. avoid referencing a class "
                f"from within its own @wp.func method bodies)."
            )

    # @wp.kernel: Kernel.adj is set directly on the kernel object.
    adj = getattr(obj, "adj", None)
    _check(adj, getattr(obj, "key", repr(obj)))

    # @wp.func: overloads each carry their own adj.
    for ovl in getattr(obj, "user_overloads", {}).values():
        _check(getattr(ovl, "adj", None), getattr(obj, "key", repr(obj)))

    return obj


def func(warp_func):
    """Composite decorator for Warp functions.

    Equivalent to applying:

        @dav.utils.assert_static_resolved
        @func
    """
    return assert_static_resolved(wp.func(warp_func))


def kernel(warp_kernel=None, /, **kwargs):
    """Composite decorator for Warp kernels.

    Equivalent to applying:

        @dav.utils.assert_static_resolved
        @wp.kernel(enable_backward=False, **kwargs)

    Can be used bare or with keyword arguments:

        @dav.kernel
        def my_kernel(...): ...

        @dav.kernel(module="unique")
        def my_kernel(...): ...
    """
    kwargs.setdefault("enable_backward", False)

    def decorator(fn):
        return assert_static_resolved(wp.kernel(**kwargs)(fn))

    if warp_kernel is not None:
        return decorator(warp_kernel)
    return decorator


# ======================================================================================================
# Array utilities


@kernel
def _array_scan_trailing_sum_kernel(array: wp.array(dtype=wp.int32), out_array: wp.array(dtype=wp.int32)):
    assert out_array.shape[0] == array.shape[0] + 1, "Output array must have one more element than input array for trailing sum"
    idx = array.shape[0]

    # Warp 1.9 does not support -ve indexing
    if idx > 0:
        out_array[idx] = out_array[idx - 1] + array[idx - 1]


def array_scan(array: wp.array(dtype=wp.int32), out_array: wp.array(dtype=wp.int32), inclusive: bool = True, add_trailing_sum: bool = False):
    """
    Perform an inclusive or exclusive scan on the input array and store the result in the output array.
    If `add_trailing_sum` is True, the output array must have one more element than the input array,
    and the last element of the output array will be the total sum of the input array.

    Args:
        array: Input array to scan
        out_array: Output array to store the result of the scan
        inclusive: If True, perform an inclusive scan. If False, perform an exclusive scan.
        add_trailing_sum: If True, add a trailing sum to the output array. Output array must have one more element than input array.

    Raises:
        ValueError: If output array size is not compatible with input array size and add_trailing_sum flag
    """
    assert array.size == out_array.size or (add_trailing_sum and array.size == out_array.size - 1)
    if inclusive and add_trailing_sum:
        raise ValueError("'add_trailing_sum' only makes sense with exclusive scan")

    if add_trailing_sum:
        wp_utils.array_scan(array, out_array[:-1], inclusive=False)
        wp.launch(_array_scan_trailing_sum_kernel, dim=1, inputs=[array, out_array], device=array.device)
    else:
        wp_utils.array_scan(array, out_array, inclusive=inclusive)


def get_div_by_i32_function(dividend_type):
    """
    Get a function that divides a value of the given type by an int32 divisor. This is useful for implementing
    operations like averaging where we want to divide by a count. The returned function will perform the division
    in a way that is appropriate for the given type, including handling vector types by dividing each component
    by the divisor.
    """
    ScalarType = get_scalar_dtype(dividend_type)

    @func
    def div_i32(dividend: dividend_type, divisor: wp.int32) -> dividend_type:
        return dividend / ScalarType(divisor)

    return div_i32


def get_mul_by_f32_function(dtype):
    """
    Get a function that multiplies a value of the given type by a float32 scalar. This is useful for implementing
    operations like averaging where we want to multiply by the reciprocal of a count. The returned function
    will perform the multiplication in a way that is appropriate for the given type, including handling vector types
    by multiplying each component by the scalar.
    """
    if is_vector_dtype(dtype):
        scalar_type = get_scalar_dtype(dtype)
        length = get_vector_length(dtype)

        @func
        def mul_f32(a: dtype, b: wp.float32) -> dtype:
            result = dtype(scalar_type(0))
            for i in range(length):
                result[i] = scalar_type(wp.float32(a[i]) * b)
            return result
    else:

        @func
        def mul_f32(a: dtype, b: wp.float32) -> dtype:
            return dtype(wp.float32(a) * b)

    return mul_f32


def get_cast_function(src_dtype, dst_dtype):
    """
    Get a function that casts a value from src_dtype to dst_dtype. This is useful for implementing
    operations that require type conversion. The returned function will perform the casting
    in a way that is appropriate for the given types, including handling vector types by casting each component.
    """
    if src_dtype == dst_dtype:

        @func
        def identity_cast(a: src_dtype) -> dst_dtype:
            """Identity cast when source and destination types are the same"""
            return a

        return identity_cast

    elif is_scalar_dtype(src_dtype) and is_scalar_dtype(dst_dtype):

        @func
        def cast_scalar_to_scalar(a: src_dtype) -> dst_dtype:
            """Cast a scalar to a scalar"""
            return dst_dtype(a)

        return cast_scalar_to_scalar

    elif is_vector_dtype(src_dtype) and is_scalar_dtype(dst_dtype):
        assert get_vector_length(src_dtype) > 1, "We never create vectors of length 1 in DAV"
        raise ValueError("Cannot cast a vector to a non-vector type")

    elif is_scalar_dtype(src_dtype) and is_vector_dtype(dst_dtype):
        dest_scalar_dtype = get_scalar_dtype(dst_dtype)

        @func
        def cast_scalar_to_vector(a: src_dtype) -> dst_dtype:
            """Cast a scalar to a vector by replicating the scalar across all components"""
            return dst_dtype(dest_scalar_dtype(a))

        return cast_scalar_to_vector

    elif is_vector_dtype(src_dtype) and is_vector_dtype(dst_dtype):
        assert get_vector_length(src_dtype) == get_vector_length(dst_dtype), "Cannot cast between vector types of different lengths"

        dest_scalar_dtype = get_scalar_dtype(dst_dtype)

        @func
        def cast_vector_to_vector(a: src_dtype) -> dst_dtype:
            """Cast a vector to a vector by casting each component"""
            result = dst_dtype(dest_scalar_dtype(0))
            for i in range(wp.static(get_vector_length(src_dtype))):
                result[i] = dest_scalar_dtype(a[i])
            return result

        return cast_vector_to_vector

    else:
        raise ValueError(f"Unsupported combination of source and destination dtypes for casting: {src_dtype} to {dst_dtype}")


if config.compile_kernels_aot:
    from . import aot

    logger.info("Compiling DAV Core utility kernels ...")
    wp.compile_aot_module(__name__, device=aot.get_devices())
