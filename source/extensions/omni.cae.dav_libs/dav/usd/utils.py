# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared helpers for implementing USD-backed DAV adapters.

This module is the main support layer for adapter authors working inside
``dav.usd``. It intentionally avoids Kit-specific helpers and instead reads
scientific array payloads directly from authored USD attributes, using only
``pxr`` schema bindings plus ordinary USD prim and relationship APIs.

The helpers in this module are intended to keep schema-specific adapter code
small and declarative. In practice, most adapters only need to:

1. detect whether a prim carries the relevant API schema
2. load arrays through :func:`get_sci_array`, :func:`get_sci_points`, or
   :func:`get_sci_field`
3. resolve related prims with :func:`get_target_prim` or
   :func:`get_target_prims`
4. translate the resulting NumPy data into DAV datasets or fields

These helpers assume the scientific schemas follow the conventions established
by ``OmniSciArrayAPI`` and ``OmniSciFieldAPI``:

- array payloads are authored as USD attributes named
  ``omni:sci:array:<instance>:value``
- field metadata is expressed through ``OmniSciFieldAPI:<instance>``
- multi-apply API instances are discoverable through the prim's applied schema
  list
"""

import asyncio
from collections.abc import Sequence

import numpy as np
import warp as wp
from pxr import Usd

import dav

from .exceptions import USDAdapterError
from .types import FieldInfo


def sci_association_to_dav(association: str) -> dav.AssociationType:
    """Translate an OmniSci field association token into DAV's enum.

    Args:
        association: OmniSci association token string, typically one of
            ``"node"``, ``"element"``, or ``"none"``.

    Returns:
        dav.AssociationType: DAV association enum value.

    Raises:
        ValueError: If the token is not recognized.
    """
    if association == "node":
        return dav.AssociationType.VERTEX
    if association == "element":
        return dav.AssociationType.CELL
    if association == "none":
        return dav.AssociationType.NOT_SPECIFIED
    raise ValueError(f"Unsupported field association '{association}'")


def get_sci_field_infos(prim: Usd.Prim) -> list[FieldInfo]:
    """Collect ``FieldInfo`` records from ``OmniSciFieldAPI`` instances on a prim.

    Adapter authors can use this as the default implementation of
    ``PrimAdapter.list_fields()`` whenever a schema stores fields directly on
    the prim being adapted.

    Args:
        prim: USD prim carrying zero or more ``OmniSciFieldAPI`` instances.

    Returns:
        list[FieldInfo]: Metadata for each authored field instance.
    """
    from pxr import OmniSci

    field_infos = []
    for instance in get_instances(prim, "OmniSciFieldAPI"):
        field_api = OmniSci.FieldAPI(prim, instance)
        label = str(field_api.GetNameAttr().Get()) or instance
        association = str(field_api.GetAssociationAttr().Get()) if field_api.GetAssociationAttr() else "none"
        field_infos.append(FieldInfo(name=instance, label=label, association=sci_association_to_dav(association)))
    return field_infos


def has_sci_array(prim: Usd.Prim, instance: str) -> bool:
    """Return whether ``prim`` has a given ``OmniSciArrayAPI`` instance.

    Args:
        prim: USD prim to inspect.
        instance: Array instance name.

    Returns:
        bool: ``True`` when the multiple-apply API instance is present.
    """
    from pxr import OmniSci

    return prim.HasAPI(OmniSci.ArrayAPI, instance)


def has_sci_field(prim: Usd.Prim, instance: str) -> bool:
    """Return whether ``prim`` has a given ``OmniSciFieldAPI`` instance.

    Args:
        prim: USD prim to inspect.
        instance: Field instance name.

    Returns:
        bool: ``True`` when the multiple-apply API instance is present.
    """
    from pxr import OmniSci

    return prim.HasAPI(OmniSci.FieldAPI, instance)


def _normalize_array_instance_names(instance_or_instances: str | Sequence[str]) -> list[str]:
    """Normalize one or more OmniSci array instance names into a non-empty list."""
    if isinstance(instance_or_instances, str):
        names = [instance_or_instances]
    else:
        names = [str(instance) for instance in instance_or_instances if str(instance)]

    if not names:
        raise USDAdapterError("instance_or_instances cannot be empty")
    return names


def _infer_stacked_target_dtype(stacked: np.ndarray, *, component_count: int):
    """Infer a Warp vector dtype for stacked component arrays."""
    vector_dtypes = {
        (np.dtype(np.float32), 2): wp.vec2f,
        (np.dtype(np.float32), 3): wp.vec3f,
        (np.dtype(np.float32), 4): wp.vec4f,
        (np.dtype(np.int32), 2): wp.vec2i,
        (np.dtype(np.int32), 3): wp.vec3i,
        (np.dtype(np.int32), 4): wp.vec4i,
    }
    dtype = vector_dtypes.get((stacked.dtype, component_count))
    if dtype is None:
        raise USDAdapterError(f"Cannot infer a Warp vector dtype for {component_count} component arrays with dtype {stacked.dtype}; pass target_dtype explicitly")
    return dtype


async def get_sci_array(prim: Usd.Prim, instance_or_instances: str | Sequence[str], time_code: Usd.TimeCode, *, device: str, target_dtype=None):
    """Read one or more scientific array payloads from USD into a Warp array.

    This is the basic array-loading primitive for adapter implementations.
    It reads the authored attribute following the OmniSci naming convention:
    ``omni:sci:array:<instance>:value``.

    Args:
        prim: USD prim owning the array payload.
        instance_or_instances: One OmniSci array instance name, or multiple
            instance names to stack into one multi-component Warp array.
        time_code: USD time code to read.
        device: Warp device alias for the returned array.
        target_dtype: Optional Warp dtype to use for the returned array.

    Returns:
        wp.array: Warp array built from the authored USD array payload.

    Raises:
        USDAdapterError: If any requested array attribute is missing, has no
            value, or cannot be stacked into the requested output form.
    """
    instance_names = _normalize_array_instance_names(instance_or_instances)

    arrays = []
    for instance_name in instance_names:
        attr = prim.GetAttribute(f"omni:sci:array:{instance_name}:value")
        if not attr:
            raise USDAdapterError(f"Array '{instance_name}' not found on prim {prim.GetPath()}")

        value = await asyncio.to_thread(attr.Get, time_code)
        if value is None:
            raise USDAdapterError(f"Array '{instance_name}' has no value on prim {prim.GetPath()}")
        arrays.append(np.asarray(value))

    if len(arrays) == 1:
        if target_dtype is None and arrays[0].ndim == 2:
            target_dtype = _infer_stacked_target_dtype(arrays[0], component_count=arrays[0].shape[1])
        kwargs = {"device": device, "copy": False}
        if target_dtype is not None:
            kwargs["dtype"] = target_dtype
        return wp.array(arrays[0], **kwargs)

    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays[1:]):
        raise USDAdapterError(f"Array components {instance_names} do not share the same shape on prim {prim.GetPath()}")

    stacked = np.stack(arrays, axis=1)
    dtype = target_dtype if target_dtype is not None else _infer_stacked_target_dtype(stacked, component_count=len(instance_names))
    return wp.array(stacked, dtype=dtype, device=device, copy=False)


async def get_sci_points(prim: Usd.Prim, *, points_instance: str, x_instance: str, y_instance: str, z_instance: str, time_code: Usd.TimeCode, device: str):
    """Load points from either interleaved or split scientific arrays.

    Many scientific schemas use one of two point layouts:

    - interleaved coordinates, for example ``points`` as ``Nx3``
    - split coordinates, for example ``pointsX``, ``pointsY``, ``pointsZ``

    This helper checks for the interleaved form first, then falls back to the
    split form. It is a good default for point-cloud and mesh adapters.

    Args:
        prim: USD prim owning the point arrays.
        points_instance: Interleaved array instance name.
        x_instance: Split X-coordinate instance name.
        y_instance: Split Y-coordinate instance name.
        z_instance: Split Z-coordinate instance name.
        time_code: USD time code to read.
        device: Warp device alias for the returned DAV array.

    Returns:
        wp.array: ``wp.vec3f`` Warp array suitable for DAV dataset creation.
    """
    if has_sci_array(prim, points_instance):
        return await get_sci_array(prim, points_instance, time_code, device=device, target_dtype=wp.vec3f)
    elif has_sci_array(prim, x_instance) and has_sci_array(prim, y_instance) and has_sci_array(prim, z_instance):
        return await get_sci_array(prim, [x_instance, y_instance, z_instance], time_code, device=device, target_dtype=wp.vec3f)
    else:
        raise USDAdapterError(f"Prim {prim.GetPath()} is missing supported point arrays. Expected '{points_instance}' or '{x_instance}/{y_instance}/{z_instance}'.")


def _normalize_field_names(field_names: str | Sequence[str]) -> list[str]:
    """Normalize a single field name or sequence into a non-empty list."""
    if isinstance(field_names, str):
        return [field_names]
    return [field_name for field_name in field_names if field_name]


async def get_sci_field(prim: Usd.Prim, field_names: str | Sequence[str], *, device: str, time_code: Usd.TimeCode):
    """Load one field, or combine multiple field components into a DAV field.

    This helper is the default field-loading path for adapters whose field data
    lives directly on the prim being adapted. For multi-component requests, the
    field arrays are combined using :meth:`dav.Field.from_arrays`.

    Args:
        prim: USD prim carrying the field metadata and array payloads.
        field_names: One field name or a sequence of component names.
        device: Warp device alias for the returned field data.
        time_code: USD time code to read.

    Returns:
        dav.Field: Loaded DAV field.

    Raises:
        USDAdapterError: If any requested field is missing or if multiple
            components do not share the same association.
    """
    names = _normalize_field_names(field_names)
    if not names:
        raise USDAdapterError("field_names cannot be empty")

    from pxr import OmniSci

    arrays = []
    association = None
    for field_name in names:
        if not has_sci_field(prim, field_name):
            raise USDAdapterError(f"Field '{field_name}' not found on prim {prim.GetPath()}")

        field_api = OmniSci.FieldAPI(prim, field_name)
        association_str = field_api.GetAssociationAttr().Get(time_code) if field_api.GetAssociationAttr() else "none"
        current_association = sci_association_to_dav(str(association_str))
        if association is None:
            association = current_association
        elif association != current_association:
            raise USDAdapterError(f"Field components {names} do not share the same association on prim {prim.GetPath()}")

        arrays.append(await get_sci_array(prim, field_name, time_code, device=device))

    assert association is not None
    return dav.Field.from_arrays(arrays, association)


def get_instances(prim: Usd.Prim, api_schema_name: str) -> list[str]:
    """Return multiple-apply API instance names authored on ``prim``.

    Adapter authors can use this to enumerate ``OmniSciFieldAPI`` or
    ``OmniSciArrayAPI`` instances without depending on external helper
    packages.

    Args:
        prim: USD prim to inspect.
        api_schema_name: Multiple-apply API schema type name, such as
            ``"OmniSciFieldAPI"``.

    Returns:
        list[str]: Instance names authored for the requested schema.
    """
    registry = Usd.SchemaRegistry()
    instances = []
    for applied_schema in prim.GetAppliedSchemas():
        schema_name, instance_name = registry.GetTypeNameAndInstance(applied_schema)
        if instance_name and schema_name == api_schema_name:
            instances.append(instance_name)
    return instances


def get_target_prims(prim: Usd.Prim, relationship_name: str, *, quiet: bool = False) -> list[Usd.Prim]:
    """Resolve the target prims of a relationship on ``prim``.

    This is a lightweight replacement for the relationship helpers that often
    exist in Kit-side utility packages. It works with ordinary USD forwarded
    relationship targets and returns already-resolved prim objects.

    Args:
        prim: USD prim owning the relationship.
        relationship_name: Name of the relationship to resolve.
        quiet: When ``True``, return an empty list instead of raising if the
            relationship or its targets are missing.

    Returns:
        list[Any]: Resolved target prims, in authored target order.
    """
    rel = prim.GetRelationship(relationship_name)
    if not rel:
        if quiet:
            return []
        raise USDAdapterError(f"Missing relationship '{relationship_name}' on prim {prim.GetPath()}")

    target_paths = rel.GetForwardedTargets()
    if not target_paths:
        if quiet:
            return []
        raise USDAdapterError(f"Relationship '{relationship_name}' has no targets on prim {prim.GetPath()}")

    stage = prim.GetStage()
    target_prims = []
    for target_path in target_paths:
        target_prim = stage.GetPrimAtPath(target_path)
        if not target_prim:
            if quiet:
                return []
            raise USDAdapterError(f"Relationship '{relationship_name}' points to missing prim {target_path}")
        target_prims.append(target_prim)
    return target_prims


def get_target_prim(prim: Usd.Prim, relationship_name: str, *, quiet: bool = False) -> Usd.Prim | None:
    """Resolve the first target prim of a relationship on ``prim``.

    This is a convenience wrapper around :func:`get_target_prims` for schemas
    that model a one-to-one relationship but still author it as a USD
    relationship list.

    Args:
        prim: USD prim owning the relationship.
        relationship_name: Name of the relationship to resolve.
        quiet: When ``True``, return ``None`` instead of raising if the
            relationship or its targets are missing.

    Returns:
        Usd.Prim | None: First resolved target prim, or ``None`` when ``quiet``
        is enabled and no valid target is available.
    """
    target_prims = get_target_prims(prim, relationship_name, quiet=quiet)
    if not target_prims:
        return None
    return target_prims[0]
