# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed access to codeless API schemas through pluggable prim backends.

A typed class declares one :class:`Attribute` or :class:`Relationship`
descriptor per schema property; :func:`make_schema_class` builds such classes
from a description, and ``ovcae.schemas._registry`` feeds it usd-core's schema
definitions. This module holds every behaviour those declarations rely on and
imports no pxr: a backend adapts a concrete prim type (``Usd.Prim`` in
``ovcae.authoring``, later ``ovcae.scene.PrimView``) to the
:class:`PrimBackend` protocol and registers itself with :func:`register_backend`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

INSTANCE_PLACEHOLDER = "{instance}"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])([A-Z])")


class SchemaError(ValueError):
    """Raised for invalid schema usage: unknown schema, wrong prim type, bad value."""


@runtime_checkable
class PrimBackend(Protocol):
    """Minimal prim operations the typed layer needs; values are backend-native.

    ``get`` returns the resolved value including schema fallbacks, or ``None``
    when the property has no value. ``set`` receives the schema's type name so a
    backend can coerce plain Python values and must raise its ``error`` for values
    it cannot store. ``clear`` removes the authored opinion where the backend
    writes (for USD, the current edit target). Relationship targets are exchanged
    as absolute path strings, though
    a backend may accept richer inputs in ``set``. A backend may also expose an
    ``error`` attribute naming the :class:`SchemaError` subclass the typed layer
    should raise for misuse on its prims.
    """

    @property
    def type_name(self) -> str: ...

    @property
    def applied_schemas(self) -> tuple[str, ...]: ...

    def apply(self, schema: str, instance: str | None) -> bool: ...

    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: Any, type_name: str) -> None: ...

    def clear(self, name: str) -> None: ...

    def targets(self, name: str) -> tuple[str, ...]: ...

    def set_targets(self, name: str, targets: Iterable[Any], *, kind: str) -> None: ...

    def native(self, name: str) -> Any: ...


_adapters: list[Callable[[Any], PrimBackend | None]] = []


def register_backend(adapter: Callable[[Any], PrimBackend | None]) -> None:
    """Register ``adapter(prim) -> PrimBackend | None`` for a concrete prim type.

    Adapters are tried most-recently-registered first; returning ``None`` passes.
    """

    _adapters.insert(0, adapter)


def backend_for(prim: Any) -> PrimBackend:
    """Resolve the backend for ``prim``; objects implementing the protocol pass through."""

    for adapter in _adapters:
        backend = adapter(prim)
        if backend is not None:
            return backend
    if isinstance(prim, PrimBackend):
        return prim
    raise SchemaError(
        f"no schema backend for {type(prim).__name__}; import ovcae.authoring to use Usd.Prim, "
        "or pass an object implementing ovcae.schemas.PrimBackend"
    )


def snake_case(name: str) -> str:
    """``fieldNames`` -> ``field_names``; already-snake names pass through."""

    return _CAMEL_BOUNDARY.sub(r"_\1", name).lower()


class _Property:
    """Shared metadata and name resolution for attribute and relationship descriptors."""

    kind: ClassVar[str]

    def __init__(self, name: str, *, display_name: str = "", display_group: str = "", doc: str = ""):
        self.name = name
        self.display_name = display_name
        self.display_group = display_group
        self.doc = doc
        self.owner: type[AppliedSchema] | None = None
        self.attr = ""
        self.__doc__ = doc or None

    def __set_name__(self, owner, attr):
        self.owner = owner
        self.attr = attr

    @property
    def api_name(self) -> str:
        """The schema's short property name, e.g. ``externalOnly`` or ``fieldNames``."""

        return self.name.rsplit(":", 1)[-1]

    @property
    def name_template(self) -> str:
        """Full property name with ``{instance}`` standing in for a multiple-apply instance."""

        if self.owner is not None and self.owner.__kind__ == "multipleApply":
            return f"{self.owner.__namespace__}:{INSTANCE_PLACEHOLDER}:{self.name}"
        return self.name

    def resolve(self, instance: str | None = None) -> str:
        """Full property name for ``instance`` (required for multiple-apply schemas)."""

        if self.owner is not None and self.owner.__kind__ == "multipleApply":
            if not instance:
                raise SchemaError(f"{self.owner.__schema__} is multiple-apply; an instance name is required")
            return f"{self.owner.__namespace__}:{instance}:{self.name}"
        return self.name

    def __repr__(self) -> str:
        owner = self.owner.__name__ if self.owner else "?"
        return f"<{type(self).__name__} {owner}.{self.attr} -> {self.name_template}>"


class Attribute(_Property):
    """Descriptor for a schema attribute; reads fall back to the schema default.

    Setting ``None`` is rejected; ``del obj.attr`` clears the authored opinion.
    Token attributes validate against ``allowed_tokens`` when the schema declares them.
    """

    kind = "attribute"

    def __init__(
        self,
        name: str,
        type_name: str,
        *,
        default: Any = None,
        has_default: bool | None = None,
        allowed_tokens: tuple[str, ...] = (),
        variability: str = "Varying",
        display_name: str = "",
        display_group: str = "",
        doc: str = "",
    ):
        super().__init__(name, display_name=display_name, display_group=display_group, doc=doc)
        self.type_name = type_name
        self.default = default
        self.has_default = (default is not None) if has_default is None else has_default
        self.allowed_tokens = tuple(allowed_tokens)
        self.variability = variability

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = obj._backend.get(self.resolve(obj.instance))
        if value is None and self.has_default:
            return self.default
        return value

    def __set__(self, obj, value) -> None:
        if value is None:
            raise obj._error(f"cannot set {self.attr} to None; use `del` to clear the authored value")
        obj._require_applied()
        problem = self.problem(value)
        if problem:
            raise obj._error(problem)
        obj._backend.set(self.resolve(obj.instance), value, self.type_name)

    def __delete__(self, obj) -> None:
        obj._backend.clear(self.resolve(obj.instance))

    def problem(self, value) -> str | None:
        """Describe why ``value`` violates the schema's token constraints, or ``None``."""

        if not self.allowed_tokens:
            return None
        if isinstance(value, str) or not self.type_name.endswith("[]"):
            tokens = (value,)
        else:
            try:
                tokens = tuple(value)
            except TypeError:
                return f"{self.name_template} expects a token or a sequence of tokens; got {value!r}"
        bad = [str(token) for token in tokens if str(token) not in self.allowed_tokens]
        if not bad:
            return None
        return (
            f"{self.name_template} does not allow {', '.join(map(repr, bad))}; "
            f"expected one of {', '.join(self.allowed_tokens)}"
        )


class Relationship(_Property):
    """Descriptor for a schema relationship; reads yield a tuple of target path strings.

    Assign one target or an iterable of targets. ``targets`` documents what the
    relationship points at (``"prims"``, ``"properties"``, or ``"any"``) so the
    backend can validate the paths it is handed.
    """

    kind = "relationship"

    def __init__(
        self,
        name: str,
        *,
        targets: str = "prims",
        display_name: str = "",
        display_group: str = "",
        doc: str = "",
    ):
        super().__init__(name, display_name=display_name, display_group=display_group, doc=doc)
        if targets not in {"prims", "properties", "any"}:
            raise ValueError(f"targets must be 'prims', 'properties', or 'any'; got {targets!r}")
        self.targets = targets

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return tuple(obj._backend.targets(self.resolve(obj.instance)))

    def __set__(self, obj, value) -> None:
        if value is None:
            raise obj._error(f"cannot set {self.attr} to None; use `del` to clear the targets")
        obj._require_applied()
        items = [value] if isinstance(value, str) or not isinstance(value, Iterable) else list(value)
        obj._backend.set_targets(self.resolve(obj.instance), items, kind=self.targets)

    def __delete__(self, obj) -> None:
        obj._backend.clear(self.resolve(obj.instance))


class AppliedSchema:
    """Base of every typed schema class; instances wrap one prim (and instance name).

    ``bool(obj)`` reports whether the schema is applied. Use :meth:`apply` to
    apply and wrap, :meth:`get` to wrap only when applied, and the constructor to
    wrap unconditionally (reads then return schema defaults, writes raise until
    applied).
    """

    __schema__: ClassVar[str] = ""
    __kind__: ClassVar[str] = "singleApply"
    __namespace__: ClassVar[str | None] = None
    __can_only_apply_to__: ClassVar[tuple[str, ...]] = ()
    __schema_doc__: ClassVar[str] = ""

    def __init__(self, prim: Any, instance: str | None = None):
        if self.__kind__ == "multipleApply":
            if not instance:
                raise SchemaError(f"{self.__schema__} is multiple-apply; pass an instance name")
            if ":" in instance:
                raise SchemaError(f"instance names cannot contain ':'; got {instance!r}")
        elif instance is not None:
            raise SchemaError(f"{self.__schema__} is single-apply; it takes no instance name")
        self._prim = prim
        self._backend = backend_for(prim)
        self._instance = instance

    @property
    def prim(self) -> Any:
        return self._prim

    @property
    def instance(self) -> str | None:
        return self._instance

    @property
    def applied_name(self) -> str:
        """The token USD lists in ``apiSchemas``, e.g. ``CaeVizFieldSelectionAPI:colors``."""

        return f"{self.__schema__}:{self._instance}" if self._instance else self.__schema__

    def __bool__(self) -> bool:
        return self.applied_name in self._backend.applied_schemas

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._prim!r}{', ' + repr(self._instance) if self._instance else ''})"

    def _error(self, message: str) -> SchemaError:
        """The backend's preferred :class:`SchemaError` subclass, so authoring raises ``AuthoringError``."""

        error = getattr(self._backend, "error", SchemaError)
        return error(message)

    def _require_applied(self) -> None:
        if not self:
            raise self._error(f"{self.applied_name} is not applied to {self._prim!r}; call apply() first")

    @classmethod
    def can_apply_to(cls, type_name: str) -> bool:
        """Whether the schema may be applied to a prim of ``type_name`` (empty list allows all)."""

        return not cls.__can_only_apply_to__ or type_name in cls.__can_only_apply_to__

    @classmethod
    def apply(cls, prim: Any, instance: str | None = None, *, check: bool = True):
        """Apply the schema and return the wrapper.

        With ``check`` the prim type is validated against ``apiSchemaCanOnlyApplyTo``;
        pxr accepts mismatches silently for codeless schemas, so this is the only guard.
        """

        obj = cls(prim, instance)
        if check and not cls.can_apply_to(obj._backend.type_name):
            raise obj._error(
                f"{cls.__schema__} applies only to {', '.join(cls.__can_only_apply_to__)} prims; "
                f"{prim!r} is {obj._backend.type_name or 'untyped'}"
            )
        if not obj and not obj._backend.apply(cls.__schema__, instance):
            raise obj._error(f"could not apply {obj.applied_name} to {prim!r}")
        return obj

    @classmethod
    def get(cls, prim: Any, instance: str | None = None):
        """Return the wrapper when the schema is applied, otherwise ``None``."""

        obj = cls(prim, instance)
        return obj if obj else None

    @classmethod
    def instances(cls, prim: Any) -> dict[str, AppliedSchema]:
        """Applied instances of a multiple-apply schema keyed by instance name."""

        if cls.__kind__ != "multipleApply":
            raise SchemaError(f"{cls.__schema__} is single-apply and has no instances")
        prefix = f"{cls.__schema__}:"
        backend = backend_for(prim)
        return {
            token[len(prefix) :]: cls(prim, token[len(prefix) :])
            for token in backend.applied_schemas
            if token.startswith(prefix)
        }

    @classmethod
    def properties(cls) -> Mapping[str, Attribute | Relationship]:
        """Descriptors declared by this schema keyed by Python attribute name."""

        found = {}
        for klass in reversed(cls.__mro__):
            for attr, value in vars(klass).items():
                if isinstance(value, _Property):
                    found[attr] = value
        return MappingProxyType(found)

    @classmethod
    def property_names(cls, instance: str | None = None) -> tuple[str, ...]:
        """Full USD property names for ``instance``."""

        return tuple(item.resolve(instance) for item in cls.properties().values())

    def property_name(self, attr: str) -> str:
        """Full USD property name of the descriptor ``attr`` on this wrapper."""

        return self._descriptor(attr).resolve(self._instance)

    def attribute(self, attr: str) -> Any:
        """Backend-native handle for attribute ``attr`` (a ``Usd.Attribute`` for stages)."""

        descriptor = self._descriptor(attr)
        if descriptor.kind != "attribute":
            raise self._error(f"{attr} is a relationship on {self.__schema__}")
        return self._backend.native(descriptor.resolve(self._instance))

    def relationship(self, attr: str) -> Any:
        """Backend-native handle for relationship ``attr``."""

        descriptor = self._descriptor(attr)
        if descriptor.kind != "relationship":
            raise self._error(f"{attr} is an attribute on {self.__schema__}")
        return self._backend.native(descriptor.resolve(self._instance))

    def _descriptor(self, attr: str) -> _Property:
        try:
            return self.properties()[attr]
        except KeyError:
            raise self._error(f"{self.__schema__} has no property {attr!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.properties())


class SingleApplyAPI(AppliedSchema):
    __kind__ = "singleApply"


class MultipleApplyAPI(AppliedSchema):
    __kind__ = "multipleApply"


def make_schema_class(
    class_name: str,
    schema: str,
    *,
    kind: str,
    namespace: str | None = None,
    can_only_apply_to: Iterable[str] = (),
    doc: str = "",
    properties: Iterable[_Property] = (),
    module: str = __name__,
) -> type[AppliedSchema]:
    """Build a typed schema class from a description.

    ``properties`` become class attributes named by the snake case of each
    property's API name. usd-core-backed classes come from
    ``ovcae.schemas.schema_from_registry``; this is the constructor for any
    other description, such as a test double.
    """

    base = MultipleApplyAPI if kind == "multipleApply" else SingleApplyAPI
    properties = tuple(properties)
    if kind == "multipleApply" and properties and not namespace:
        raise SchemaError(f"multiple-apply schema {schema} needs a property namespace")
    body: dict[str, Any] = {
        "__schema__": schema,
        "__namespace__": namespace,
        "__can_only_apply_to__": tuple(can_only_apply_to),
        "__schema_doc__": doc,
        "__doc__": doc or None,
        "__module__": module,
    }
    for item in properties:
        body[snake_case(item.api_name)] = item
    return type(class_name, (base,), body)
