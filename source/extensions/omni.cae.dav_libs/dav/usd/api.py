# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Public OpenUSD API for DAV.

These functions form the supported user-facing API for converting OpenUSD
scientific dataset prims into DAV objects. All conversion entry points are
``async`` because the underlying OpenUSD integrations may fetch array payloads
through asynchronous helpers.
"""

from collections.abc import Sequence

from pxr import Usd

from .exceptions import USDAdapterError
from .registry import registry


async def dataset_from_prim(prim: Usd.Prim, *, device: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Convert a supported USD prim into DAV.

    Args:
        prim: A USD prim supported by one of the registered adapters.
            Examples include CGNS zone/section prims and CAE mesh or point
            cloud prims.
        device: Warp device alias where DAV arrays should be created.
        time_code: USD time code used when reading time-sampled data. Defaults
            to ``Usd.TimeCode.Default()``.

    Returns:
        dav.Dataset | dav.DatasetCollection: The DAV representation of
        ``prim``.

    Raises:
        UnsupportedPrimError: If no registered adapter can handle ``prim``.
        USDAdapterError: If the prim is supported in principle but conversion
            fails because required schema data or runtime helpers are missing.
    """
    adapter = registry.get_adapter(prim)
    try:
        return await adapter.to_dataset(prim, device=device, time_code=time_code)
    except ImportError as exc:
        raise USDAdapterError(f"Required pxr schema module is not available: {exc}") from exc


async def dataset_from_stage(stage: Usd.Stage, prim_path: str, *, device: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Resolve a prim from a USD stage and convert it into DAV.

    This is a convenience wrapper around :func:`dataset_from_prim` for callers
    that have a stage and path string rather than a prim object.

    Args:
        stage: The USD stage to query.
        prim_path: Absolute prim path to resolve from ``stage``.
        device: Warp device alias where DAV arrays should be created.
        time_code: USD time code used when reading time-sampled data. Defaults
            to ``Usd.TimeCode.Default()``.

    Returns:
        dav.Dataset | dav.DatasetCollection: The DAV representation of the
        resolved prim.
    """
    prim = stage.GetPrimAtPath(prim_path)
    return await dataset_from_prim(prim, device=device, time_code=time_code)


async def list_fields(prim: Usd.Prim, *, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """List the DAV-visible fields exposed by a supported USD prim.

    Args:
        prim: A USD prim supported by one of the registered adapters.
        time_code: USD time code used when field metadata is time-dependent.
            Defaults to ``Usd.TimeCode.Default()``.

    Returns:
        list[dav.usd.FieldInfo]: One entry per available field. The ``name``
        value is the field identifier accepted by :func:`field_from_prim`.
    """
    adapter = registry.get_adapter(prim)
    try:
        return await adapter.list_fields(prim, time_code=time_code)
    except ImportError as exc:
        raise USDAdapterError(f"Required pxr schema module is not available: {exc}") from exc


async def field_from_prim(prim: Usd.Prim, field_names: str | Sequence[str], *, device: str, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """Load one field from a supported USD prim.

    ``field_names`` may be either a single field name or a sequence of names.
    When multiple names are provided, the adapter combines the corresponding
    scalar arrays into one DAV vector field using structure-of-arrays storage.

    Args:
        prim: A USD prim supported by one of the registered adapters.
        field_names: One field name, or multiple component field names to
            combine into a vector field.
        device: Warp device alias where the field arrays should be created.
        time_code: USD time code used when reading time-sampled data. Defaults
            to ``Usd.TimeCode.Default()``.

    Returns:
        dav.Field: The loaded field.

    Raises:
        UnsupportedPrimError: If no adapter can handle ``prim``.
        USDAdapterError: If the requested field cannot be resolved or loaded.
    """
    adapter = registry.get_adapter(prim)
    try:
        return await adapter.get_field(prim, field_names=field_names, device=device, time_code=time_code)
    except ImportError as exc:
        raise USDAdapterError(f"Required pxr schema module is not available: {exc}") from exc


def find_supported_prims(stage: Usd.Stage) -> list[Usd.Prim]:
    """Find all prims on a stage that one of the registered adapters supports.

    Args:
        stage: The USD stage to traverse.

    Returns:
        list[Usd.Prim]: All prims for which :data:`dav.usd.registry` can
        resolve an adapter.
    """
    supported = []
    for prim in stage.Traverse():
        if registry.find_adapter(prim) is not None:
            supported.append(prim)
    return supported
