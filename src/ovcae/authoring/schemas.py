# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed OmniCaeViz schema access on ``Usd.Prim``.

Importing this module registers :class:`UsdPrimBackend` with ``ovcae.schemas``
so every class in ``ovcae.schemas.typed`` accepts a ``Usd.Prim``::

    from ovcae.authoring import schemas as S

    faces = S.FacesAPI.apply(mesh_prim)  # refuses non-Mesh prims
    faces.external_only = False  # -> cae:viz:faces:externalOnly
    colors = S.FieldSelectionAPI.apply(mesh_prim, "colors")
    colors.field_names = ["Temperature"]  # -> Vt.TokenArray
    colors.mode = "vector_magnitude"  # validated against allowedTokens
    colors.attribute("mode")  # the Usd.Attribute when needed

Reads return pxr values (``Gf.Vec2f``, ``Vt.TokenArray``) when the property
exists on the prim and the plain schema default otherwise. For a schema
outside the bundle, such as a reader's file-format argument API,
``schema_from_registry(name)`` builds the class the same way.
"""

from __future__ import annotations

from collections.abc import Iterable

from ovcae.schemas import register_backend, schema_from_registry
from ovcae.schemas import typed as _typed

from ._errors import AuthoringError
from ._usd import require_usd

_SCALARS = {
    "bool": bool,
    "uchar": int,
    "int": int,
    "uint": int,
    "int64": int,
    "uint64": int,
    "half": float,
    "float": float,
    "double": float,
    "timecode": float,
    "token": str,
    "string": str,
}


class UsdPrimBackend:
    """Adapt a ``Usd.Prim`` to ``ovcae.schemas.PrimBackend``; schema misuse raises :class:`AuthoringError`."""

    error = AuthoringError

    def __init__(self, prim):
        self._prim = prim

    @property
    def prim(self):
        return self._prim

    @property
    def type_name(self) -> str:
        return str(self._prim.GetTypeName())

    @property
    def applied_schemas(self) -> tuple[str, ...]:
        return tuple(self._prim.GetAppliedSchemas())

    def apply(self, schema: str, instance: str | None) -> bool:
        return bool(self._prim.ApplyAPI(schema, instance) if instance else self._prim.ApplyAPI(schema))

    def get(self, name: str):
        attribute = self._prim.GetAttribute(name)
        return attribute.Get() if attribute else None

    def set(self, name: str, value, type_name: str) -> None:
        pxr = require_usd()
        attribute = self._prim.GetAttribute(name)
        if not attribute:
            attribute = self._prim.CreateAttribute(name, _value_type(pxr, type_name))
        if not attribute.Set(coerce(value, type_name)):
            raise AuthoringError(f"could not set {name!r} on {self._prim.GetPath()}")

    def clear(self, name: str) -> None:
        """Remove the property spec from the current edit target.

        ``Attribute.Clear()`` would leave an empty spec behind. Like every USD
        edit, this touches only the current edit target; an opinion authored in
        another layer remains and the read still resolves to it.
        """

        if self._prim.GetProperty(name):
            self._prim.RemoveProperty(name)

    def targets(self, name: str) -> tuple[str, ...]:
        relationship = self._prim.GetRelationship(name)
        return tuple(str(path) for path in relationship.GetTargets()) if relationship else ()

    def set_targets(self, name: str, targets: Iterable, *, kind: str) -> None:
        paths = [_target_path(target, kind, name) for target in targets]
        relationship = self._prim.GetRelationship(name) or self._prim.CreateRelationship(name)
        if not relationship.SetTargets(paths):
            raise AuthoringError(f"could not set relationship {name!r} on {self._prim.GetPath()}")

    def native(self, name: str):
        prop = self._prim.GetProperty(name)
        return prop if prop else None


def coerce(value, type_name: str):
    """Convert a plain Python value to the pxr value ``Sdf.ValueTypeNames.Find(type_name)`` stores.

    Raises :class:`AuthoringError` when ``value`` cannot become a ``type_name``.
    """

    pxr = require_usd()
    value_type = _value_type(pxr, type_name)
    try:
        return _coerce(pxr, value, value_type, type_name)
    except AuthoringError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        # Boost.Python's ArgumentError is a TypeError subclass.
        raise AuthoringError(f"cannot store {value!r} in a {type_name} attribute: {exc}") from exc


def _coerce(pxr, value, value_type, type_name: str):
    python_class = value_type.type.pythonClass
    base = type_name.removesuffix("[]")
    if python_class is None:
        if base == "asset":
            return value if isinstance(value, pxr.Sdf.AssetPath) else pxr.Sdf.AssetPath(str(value))
        scalar = _SCALARS.get(base)
        if scalar is bool and not isinstance(value, bool | int):
            raise AuthoringError(f"{type_name} attribute needs a bool; got {value!r}")
        return scalar(value) if scalar else value
    if isinstance(value, python_class):
        return value
    if value_type.isArray:
        # A lone string is one element, not a sequence of characters.
        return python_class([value] if isinstance(value, str) else list(value))
    if isinstance(value, str) or not isinstance(value, Iterable):
        return python_class(value)
    return python_class(*value)


def _value_type(pxr, type_name: str):
    value_type = pxr.Sdf.ValueTypeNames.Find(type_name)
    if not value_type:
        raise AuthoringError(f"unknown USD value type {type_name!r}")
    return value_type


def _target_path(target, kind: str, name: str):
    pxr = require_usd()
    if isinstance(target, pxr.Usd.Prim | pxr.Usd.Property):
        path = target.GetPath()
    else:
        path = target if isinstance(target, pxr.Sdf.Path) else pxr.Sdf.Path(str(target))
    if not path.IsAbsolutePath():
        raise AuthoringError(f"{name} targets must be absolute paths; got {target!r}")
    if kind == "prims" and not path.IsPrimPath():
        raise AuthoringError(f"{name} targets prims; got property path {path}")
    if kind == "properties" and not path.IsPropertyPath():
        raise AuthoringError(f"{name} targets properties such as /Prim.inputs:domain; got {path}")
    return path


def _adapt(prim):
    pxr = require_usd()
    return UsdPrimBackend(prim) if isinstance(prim, pxr.Usd.Prim) else None


register_backend(_adapt)


def __getattr__(name: str):
    """Resolve the typed classes lazily so importing this module performs no schema read.

    usd-core builds its schema registry on the first schema read and never
    consults a plug-in registered afterwards; building every class at import
    would make ``import ovcae.authoring`` that first read.
    """

    if name in _typed.__all__:
        return getattr(_typed, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [*_typed.__all__, "UsdPrimBackend", "coerce", "schema_from_registry"]
