# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""OpenUSD adapters for DAV datasets and fields.

The :mod:`dav.usd` package provides the high-level bridge between OpenUSD
scientific dataset prims and DAV's in-memory dataset and field abstractions.
Its public API is intentionally small:

- :func:`dataset_from_prim` converts a supported prim into a
  :class:`dav.Dataset` or :class:`dav.DatasetCollection`
- :func:`dataset_from_stage` resolves a prim by path and converts it
- :func:`list_fields` reports which fields are available on a prim
- :func:`field_from_prim` loads one field, or combines multiple scalar arrays
  into a DAV vector field
- :func:`find_supported_prims` discovers all prims on a stage that one of the
  registered adapters can consume

Adapters are resolved through the module-level :data:`registry`.
"""

from .exceptions import UnsupportedPrimError, USDAdapterError

try:
    from pxr import Usd  # noqa: F401
except ImportError as _exc:
    raise USDAdapterError("dav.usd requires OpenUSD (pxr) at import time; install pxr to use this module") from _exc

from .api import dataset_from_prim, dataset_from_stage, field_from_prim, find_supported_prims, list_fields
from .registry import AdapterRegistry, registry
from .types import FieldInfo

__all__ = [
    "AdapterRegistry",
    "FieldInfo",
    "USDAdapterError",
    "UnsupportedPrimError",
    "dataset_from_prim",
    "dataset_from_stage",
    "field_from_prim",
    "find_supported_prims",
    "list_fields",
    "registry",
]
