# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from functools import wraps


def cached(func):
    """Decorator to cache the result of a function based on input arguments.

    This decorator caches the result of a function based on the input arguments.
    The cache key is built from the function identity and the arguments.
    """

    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Build cache key from function identity and arguments
        cache_key = []
        cache_key.append(("args", tuple(id(arg) for arg in args)))
        cache_key.append(("kwargs", tuple((k, id(v)) for k, v in kwargs.items())))
        cache_key = tuple(cache_key)
        if cache_key not in cache:
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result
        else:
            return cache[cache_key]

    return wrapper
