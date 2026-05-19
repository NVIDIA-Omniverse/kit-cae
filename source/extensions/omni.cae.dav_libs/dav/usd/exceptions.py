# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Exceptions raised by :mod:`dav.usd`."""


class USDAdapterError(RuntimeError):
    """Base error raised by :mod:`dav.usd`.

    This covers runtime failures after an adapter has been selected, such as
    missing schema relationships or malformed data arrays on an otherwise
    supported prim.
    """


class UnsupportedPrimError(USDAdapterError):
    """Raised when no registered adapter can handle a prim.

    This usually means the prim is not annotated with one of the supported
    scientific schemas, or that the corresponding adapter has not yet been
    registered.
    """
