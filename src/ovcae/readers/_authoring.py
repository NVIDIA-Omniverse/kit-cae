# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Author the holder prims this reader later resolves.

A holder is an untyped prim that payloads one native engineering file and
carries that file format's ``OmniSciFileFormatArgs*API`` schema. The CAE
file-format plug-ins compose the file into ``OmniSciDataset`` prims beneath it.
Only usd-core is needed; argument errors raise ``ValueError``.
"""

from __future__ import annotations

import re
from contextlib import contextmanager, nullcontext
from pathlib import PurePath

from pxr import Plug, Sdf, Tf, Usd

_API_PREFIX = "OmniSciFileFormatArgs"
_FORMAT_ALIASES = {"pythonproxy": "python"}


def import_dataset(stage, path, file_name, *, format=None, edit_target=None, **args):
    """Define a holder prim that payloads one scientific file.

    The CAE file-format plug-ins compose the file into typed ``OmniSciDataset``
    prims beneath ``path``; recipes select those prims directly (see
    :func:`dataset_prims`). ``file_name`` is authored as given, so pass a
    layer-relative string to keep the scene relocatable. ``format`` names the
    plug-in's format-arguments API by short id (``"cgns"``, ``"vtk"``, ...) and
    defaults to the format registered for the file extension. Keyword ``args``
    set that schema's ``omni:cae:format:*`` attributes by leaf name
    (``zoneName="Zone1"``, ``cacheMode="static"``) or Python name
    (``zone_name="Zone1"``); they become dynamic file-format arguments when the
    payload composes.
    """

    asset = file_name if isinstance(file_name, Sdf.AssetPath) else Sdf.AssetPath(str(file_name))
    if not asset.path:
        raise ValueError("file_name must be a non-empty path")
    api = _format_args_api(asset.path, format)
    if api is None and args:
        raise ValueError(f"{asset.path!r} has no format-arguments schema; cannot set {sorted(args)}")

    with _edit_context(stage, edit_target):
        prim_path = _as_prim_path(path)
        existing = stage.GetPrimAtPath(prim_path)
        if existing and existing.GetTypeName():
            raise ValueError(f"{prim_path} already exists with type {existing.GetTypeName()!r}; expected untyped")
        prim = stage.DefinePrim(prim_path)
        if api is not None:
            if not prim.ApplyAPI(api):
                raise ValueError(f"could not apply {api} to {prim_path}")
            for name, value in args.items():
                _set_format_argument(prim, api, name, value)
        if not prim.GetPayloads().AddPayload(asset.path):
            raise ValueError(f"could not add payload {asset.path!r} to {prim_path}")
    return prim


def dataset_prims(stage, holder):
    """Return the ``OmniSciDataset`` prims composed beneath a holder, in stage order.

    The list is empty when the payload did not compose, for example when the CAE
    file-format plug-ins are not installed in this process.
    """

    root = holder if isinstance(holder, Usd.Prim) else stage.GetPrimAtPath(_as_prim_path(holder))
    if not root or not root.IsValid():
        raise ValueError(f"dataset holder prim does not exist: {holder}")
    if not root.IsLoaded():
        root.Load()
    return [prim for prim in Usd.PrimRange(root) if prim.GetTypeName() == "OmniSciDataset"]


@contextmanager
def _edit_context(stage, edit_target):
    """Author into ``edit_target`` when given, otherwise into the stage's current target."""

    if edit_target is None:
        context = nullcontext()
    else:
        if isinstance(edit_target, Sdf.Layer):
            edit_target = Usd.EditTarget(edit_target)
        context = Usd.EditContext(stage, edit_target)
    with context:
        yield


def _as_prim_path(path) -> Sdf.Path:
    if isinstance(path, Usd.Prim):
        return path.GetPath()
    sdf_path = Sdf.Path(str(path))
    if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
        raise ValueError(f"expected an absolute prim path; got {path!r}")
    return sdf_path


def _format_args_apis() -> dict[str, str]:
    base = Tf.Type.FindByName("UsdAPISchemaBase")
    names = {}
    for tf_type in Plug.Registry().GetAllDerivedTypes(base):
        name = Usd.SchemaRegistry.GetSchemaTypeName(tf_type)
        if name.startswith(_API_PREFIX) and name.endswith("API"):
            names[name[len(_API_PREFIX) : -3].lower()] = name
    return names


def _format_args_api(asset_path: str, format) -> str | None:
    apis = _format_args_apis()
    if format is not None:
        key = _FORMAT_ALIASES.get(str(format).lower(), str(format).lower())
        try:
            return apis[key]
        except KeyError as exc:
            known = ", ".join(sorted(k for k in apis if k not in {"", "time", "streaming"}))
            raise ValueError(f"unknown format {format!r}; registered formats: {known or '<none>'}") from exc
    extension = PurePath(asset_path.split(":", 1)[0]).suffix.lstrip(".").lower()
    file_format = Sdf.FileFormat.FindByExtension(extension) if extension else None
    if not file_format or not file_format.formatId.startswith("OmniSci"):
        return None
    key = file_format.formatId.removeprefix("OmniSci").removesuffix("FileFormat").lower()
    return apis.get(_FORMAT_ALIASES.get(key, key))


def _set_format_argument(prim, api: str, name: str, value) -> None:
    """Validate ``value`` against the schema's attribute definition and author it."""

    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(api)
    attribute_name = _format_argument(definition, api, name)
    attribute = definition.GetAttributeDefinition(attribute_name)
    allowed = attribute.GetMetadata("allowedTokens")
    if allowed and str(value) not in allowed:
        raise ValueError(f"{name!r} must be one of {', '.join(allowed)}; got {value!r}")
    try:
        if not prim.GetAttribute(attribute_name).Set(value):
            raise ValueError(f"could not author {name!r} on {prim.GetPath()}")
    except Tf.ErrorException as exc:
        raise ValueError(f"{name!r} expects {attribute.GetTypeName()}; got {value!r}") from exc


def _format_argument(definition, api: str, name: str) -> str:
    """Map a leaf name (``zoneName``) or Python name (``zone_name``) to the schema's attribute."""

    properties = list(definition.GetPropertyNames()) if definition else []
    if name in properties:
        return name
    leaf = re.sub(r"_([a-z0-9])", lambda m: m.group(1).upper(), name)
    candidates = [prop for prop in properties if prop.rsplit(":", 1)[-1] in {name, leaf}]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        available = ", ".join(sorted(prop.rsplit(":", 1)[-1] for prop in properties))
        raise ValueError(f"{api} has no format argument {name!r}; available: {available}")
    raise ValueError(f"format argument {name!r} is ambiguous on {api}: {', '.join(candidates)}")
