# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Shared helpers for builders: paths, edit targets, role selections, and fields."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext

from . import schemas as S
from ._errors import AuthoringError
from ._usd import require_usd

Color = tuple[float, float, float]


def _pxr():
    return require_usd()


@contextmanager
def edit_context(stage, edit_target):
    """Author into ``edit_target`` when given, otherwise into the stage's current target."""

    pxr = _pxr()
    if edit_target is None:
        context = nullcontext()
    else:
        if isinstance(edit_target, pxr.Sdf.Layer):
            edit_target = pxr.Usd.EditTarget(edit_target)
        context = pxr.Usd.EditContext(stage, edit_target)
    with context:
        yield


def as_path(path):
    """Return a validated absolute prim path."""

    pxr = _pxr()
    if isinstance(path, pxr.Usd.Prim):
        return path.GetPath()
    sdf_path = pxr.Sdf.Path(str(path))
    if not sdf_path.IsAbsolutePath() or not sdf_path.IsPrimPath():
        raise AuthoringError(f"expected an absolute prim path; got {path!r}")
    return sdf_path


def resolve_prim(stage, prim_or_path, *, what: str):
    """Return an existing prim from a prim or path argument, raising if it is missing."""

    pxr = _pxr()
    is_prim = isinstance(prim_or_path, pxr.Usd.Prim)
    prim = prim_or_path if is_prim else stage.GetPrimAtPath(as_path(prim_or_path))
    if not prim or not prim.IsValid():
        raise AuthoringError(f"{what} prim does not exist: {prim_or_path}")
    return prim


def define_typed_prim(stage, path, type_name: str):
    """Create a typed prim at an unused composed path, without modifying existing prims."""

    path = as_path(path)
    existing = stage.GetPrimAtPath(path)
    if existing:
        raise AuthoringError(f"{path} already exists; choose an unused path to create a new prim")
    prim = stage.DefinePrim(path, type_name)
    if not prim:
        raise AuthoringError(f"could not define {type_name} prim at {path}")
    return prim


def set_optional(schema, **values) -> None:
    """Assign the typed attributes whose value is not ``None``, leaving schema fallbacks otherwise."""

    for attr, value in values.items():
        if value is not None:
            setattr(schema, attr, value)


def field_names(value) -> list[str]:
    """Normalize a field selection argument to a non-empty list of names."""

    if isinstance(value, str):
        names = [value]
    elif isinstance(value, Sequence):
        names = [str(item) for item in value]
    else:
        raise AuthoringError(f"field names must be a string or a sequence of strings; got {type(value).__name__}")
    if not names or any(not name for name in names):
        raise AuthoringError("field names must be non-empty")
    return names


def select_dataset(prim, role: str, target, *, identifier: str = ""):
    """Apply ``CaeVizDatasetSelectionAPI:<role>`` and author its target and identifier."""

    selection = S.DatasetSelectionAPI.apply(prim, role)
    selection.target = target
    if identifier:
        selection.identifier = str(identifier)
    return selection


def select_field(prim, role: str, names, *, input: str = "source", mode: str | None = None):
    """Apply ``CaeVizFieldSelectionAPI:<role>`` and author its field names."""

    selection = S.FieldSelectionAPI.apply(prim, role)
    selection.field_names = field_names(names)
    if input != "source":
        selection.input = input
    set_optional(selection, mode=mode)
    return selection


def set_display_color(prim, color: Color | None) -> None:
    """Author a constant displayColor primvar so a placeholder renders predictably."""

    if color is None:
        return
    pxr = _pxr()
    primvar = pxr.UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "displayColor", pxr.Sdf.ValueTypeNames.Color3fArray, pxr.UsdGeom.Tokens.constant
    )
    primvar.Set([pxr.Gf.Vec3f(*color)])
