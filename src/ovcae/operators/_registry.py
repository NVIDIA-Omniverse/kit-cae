# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Class declarations and deterministic operator registries."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType, ModuleType

from .._errors import OvcaeError


class OperatorRegistrationError(OvcaeError, ValueError):
    """Raised for an invalid operator declaration or registration."""


class AmbiguousOperatorError(OvcaeError):
    """Raised when matching declarations have the same highest priority."""


@dataclass(frozen=True)
class OperatorDeclaration:
    """Validated class registration metadata shared by Session registry snapshots.

    Required selectors determine eligibility; optional selectors extend discoverable
    inputs without affecting matching. Higher priority wins among eligible classes.
    Dataset components are an immutable role-to-component mapping: None selects the
    legacy always-execute path, while a mapping enables schema-derived requests.
    """

    implementation: type
    required_schemas: frozenset[str]
    optional_schemas: frozenset[str]
    priority: int
    dataset_components: Mapping[str, frozenset[str]] | None = None

    def matches(self, applied_api_schemas: frozenset[str]) -> bool:
        """Match every required selector against applied APIs, including family wildcards."""
        return all(_selector_matches(selector, applied_api_schemas) for selector in self.required_schemas)


def operator(
    *,
    required_schemas: set[str] | frozenset[str],
    optional_schemas: set[str] | frozenset[str] = frozenset(),
    priority: int = 0,
    dataset_components: Mapping[str, set[str] | frozenset[str]] | None = None,
):
    """Validate and attach declaration metadata without registering the class globally.

    Copy selectors and component sets so later caller mutations cannot alter a
    Session's declaration. Component entries select topology/geometry requirements by
    input role; an exact entry overrides the optional * default during planning.
    Omitting dataset_components preserves legacy execution. Invalid selectors,
    components, roles, or classes raise OperatorRegistrationError.
    """

    required = _selectors(required_schemas, "required_schemas")
    optional = _selectors(optional_schemas, "optional_schemas")
    if not required:
        raise OperatorRegistrationError("an operator must require at least one API schema")

    components = None
    if dataset_components is not None:
        components = {}
        permitted = required | optional
        for role, values in dataset_components.items():
            values = frozenset(values)
            if not values <= {"topology", "geometry"}:
                raise OperatorRegistrationError(f"invalid dataset components for {role!r}")
            selector = f"CaeVizDatasetSelectionAPI:{role}"
            default_for_selected_roles = role == "*" and any(
                s.startswith("CaeVizDatasetSelectionAPI:") for s in permitted
            )
            if (
                not default_for_selected_roles
                and selector not in permitted
                and "CaeVizDatasetSelectionAPI:*" not in permitted
            ):
                raise OperatorRegistrationError(f"dataset role {role!r} is not selected by the declaration")
            components[str(role)] = values
        components = MappingProxyType(components)

    def decorate(cls: type) -> type:
        if not inspect.isclass(cls) or not callable(getattr(cls, "execute", None)):
            raise OperatorRegistrationError("@operator requires a class with execute(context)")
        if hasattr(cls, "__ovcae_operator_declaration__"):
            raise OperatorRegistrationError(f"operator class is already decorated: {cls.__qualname__}")
        cls.__ovcae_operator_declaration__ = OperatorDeclaration(cls, required, optional, int(priority), components)
        return cls

    return decorate


class OperatorRegistry:
    """Mutable declaration registry copied immutably into each Session."""

    def __init__(self) -> None:
        self._declarations: dict[type, OperatorDeclaration] = {}

    @classmethod
    def with_builtins(cls) -> OperatorRegistry:
        from ._builtins import Faces, Glyphs, IsoSurface, PlanarSlice, Points, Streamlines

        registry = cls()
        registry.register(Faces)
        registry.register(IsoSurface)
        registry.register(PlanarSlice)
        registry.register(Points)
        registry.register(Glyphs)
        registry.register(Streamlines)
        return registry

    def register(self, implementation: type) -> None:
        declaration = getattr(implementation, "__ovcae_operator_declaration__", None)
        if not isinstance(declaration, OperatorDeclaration):
            raise OperatorRegistrationError(f"operator class must be decorated before registration: {implementation!r}")
        self._declarations[implementation] = declaration

    def register_module(self, module: ModuleType) -> None:
        for value in vars(module).values():
            if inspect.isclass(value) and isinstance(
                getattr(value, "__ovcae_operator_declaration__", None), OperatorDeclaration
            ):
                self.register(value)

    def snapshot(self) -> RegistrySnapshot:
        return RegistrySnapshot(tuple(self._declarations.values()))


class RegistrySnapshot:
    def __init__(self, declarations: tuple[OperatorDeclaration, ...]):
        self._declarations = declarations

    def match(self, applied_api_schemas: frozenset[str]) -> OperatorDeclaration | None:
        matches = [item for item in self._declarations if item.matches(applied_api_schemas)]
        if not matches:
            return None
        matches.sort(
            key=lambda item: (
                -item.priority,
                item.implementation.__module__,
                item.implementation.__qualname__,
            )
        )
        highest = matches[0].priority
        tied = [item for item in matches if item.priority == highest]
        if len(tied) > 1:
            names = ", ".join(f"{item.implementation.__module__}.{item.implementation.__qualname__}" for item in tied)
            raise AmbiguousOperatorError(f"operator match is ambiguous at priority {highest}: {names}")
        return matches[0]


def _selectors(values, label: str) -> frozenset[str]:
    try:
        selectors = frozenset(str(value) for value in values)
    except TypeError as exc:
        raise OperatorRegistrationError(f"{label} must be an iterable of schema selectors") from exc
    if any(not value or ("*" in value and not value.endswith(":*")) for value in selectors):
        raise OperatorRegistrationError(f"{label} contains an invalid selector")
    return selectors


def _selector_matches(selector: str, applied: frozenset[str]) -> bool:
    if selector.endswith(":*"):
        prefix = selector[:-1]
        return any(value.startswith(prefix) and len(value) > len(prefix) for value in applied)
    return selector in applied
