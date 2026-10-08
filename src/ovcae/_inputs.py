# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""One immutable schema-derived request plan per operator Update."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .operators import InvalidRecipe
from .operators.utils import _apply_field_modes, _field_selection, _point_cloud_from_prim, _scene_input


@dataclass(frozen=True)
class InputSamples:
    """Native sample descriptions exposed to an operator for one input role.

    Topology and geometry each hold a provider description or None when unavailable
    or unrequested. The read-only fields mapping uses output field roles, not source
    array names. A multi-array selection preserves each contributing description.
    These are inspection metadata; complete comparison identities stay in InputPlan.
    """

    topology: object
    geometry: object
    fields: object


@dataclass(frozen=True)
class _Request:
    """Immutable loading requirements for one selected dataset role.

    The key and identifier select the source dataset; device and components describe
    the effective representation requirements. Selections bind output field roles
    to source arrays and interpretation modes. For scene inputs, key addresses a
    captured Points prim instead of scientific Frame Data. No payload is stored here.
    """

    role: str
    key: str
    identifier: str
    device: str
    components: frozenset[str]
    selections: tuple
    scene: bool

    @property
    def fields(self):
        """Translate field selections into the alias-to-source-names mapping expected by providers."""
        return {s.output_name: s.field_names for s in self.selections}


class InputPlan:
    """Shared description and loading plan for one annotated operator evaluation.

    Session builds this from the declared recipe and operator declaration before
    calling execute. Requests and matched field selections are reused by dataset
    loading, so the gate and operator cannot select different inputs. Read-only
    identities hold opaque comparison evidence; samples hold operator-facing native
    metadata. Missing evidence is represented by None and requires recomputation.

    Construction describes inputs without loading scientific arrays. The plan keeps
    no context or Frame Data reference and can therefore remain in an evaluation
    record after the execution capture expires. Resolution must use the originating
    Update context and its still-active lease.
    """

    def __init__(self, context, declaration):
        """Validate matched input roles and describe their effective dependencies.

        Expand exact/family schema selectors, associate fields through their input role,
        and combine declared components with representation requirements. Reject invalid
        relationships, unavailable field inputs, and unsupported selected preprocessing
        with InvalidRecipe. Ask the current frame provider for component evidence when
        available; scene inputs and legacy providers retain unknown evidence.
        """
        selectors = declaration.required_schemas | declaration.optional_schemas

        def selected(schema):
            return schema in selectors or f"{schema.split(':')[0]}:*" in selectors

        def required(schema):
            return schema in declaration.required_schemas or (
                f"{schema.split(':')[0]}:*" in declaration.required_schemas
            )

        prim = context.prim
        unsupported = [
            schema
            for schema in prim.applied_api_schemas
            if selected(schema)
            and schema.startswith("CaeVizDataset")
            and not schema.startswith("CaeVizDatasetSelectionAPI:")
        ]
        if unsupported:
            raise InvalidRecipe(f"dataset preprocessing is outside the prototype: {', '.join(unsupported)}")
        # A matched field instance names an output role; its input attribute
        # assigns it to a selected dataset role. Unmatched instances are not loaded.
        # An optionally-selected instance with no authored fieldNames was applied
        # but never given a source field, so it decodes to None and is dropped.
        self.selections = tuple(
            selection
            for selection in (
                _field_selection(
                    prim,
                    instance,
                    optional=not required(f"CaeVizFieldSelectionAPI:{instance}"),
                )
                for instance in prim.api_instances("CaeVizFieldSelectionAPI")
                if selected(f"CaeVizFieldSelectionAPI:{instance}")
            )
            if selection is not None
        )
        requests = {}
        for role in prim.api_instances("CaeVizDatasetSelectionAPI"):
            if not selected(f"CaeVizDatasetSelectionAPI:{role}"):
                continue
            components = declaration.dataset_components.get(role, declaration.dataset_components.get("*"))
            if components is None:
                raise InvalidRecipe(f"dataset_components does not cover input {role!r}")
            base = f"cae:viz:dataset_selection:{role}"
            targets = prim.get_relationship(f"{base}:target").targets
            if len(targets) != 1:
                raise InvalidRecipe(f"{base}:target must have exactly one target")
            key = str(targets[0])
            identifier = str(prim.get_attribute(f"{base}:identifier").get() or "")
            scene = _scene_input(context.stage.get_prim_at_path(key))
            selections = tuple(s for s in self.selections if s.input == role)
            if scene and (identifier or selections):
                raise InvalidRecipe(f"scene input {role!r} cannot use an identifier or scientific fields")
            # The prototype memory representation materializes both components.
            # Include those effective dependencies even for geometry-only algorithms.
            effective = components | {"topology", "geometry"} if not scene else components
            requests[role] = _Request(role, key, identifier, context.device, effective, selections, scene)
        for selection in self.selections:
            if selection.input not in requests:
                raise InvalidRecipe(f"field {selection.output_name!r} refers to unavailable input {selection.input!r}")
        self.requests = MappingProxyType(requests)
        # Keep comparison evidence separate from inspectable native metadata.
        # Include the request itself so selection/device/interpretation edits
        # cannot compare equal merely because backing sample versions match.
        identities, samples = {}, {}
        describe = getattr(context._frame_data._provider, "describe_component", None)
        for role, request in requests.items():
            evidence = {}
            native = {}
            for component in sorted(request.components):
                identity, sample = (
                    describe(request.key, request.identifier, component)
                    if describe and not request.scene
                    else (None, None)
                )
                evidence[component] = (request, identity) if identity is not None else None
                native[component] = sample
            field_samples = {}
            for selection in request.selections:
                parts = [
                    describe(request.key, request.identifier, f"field:{name}")
                    if describe and not request.scene
                    else (None, None)
                    for name in selection.field_names
                ]
                evidence[f"field:{selection.output_name}"] = (
                    (request, tuple(p[0] for p in parts)) if all(p[0] is not None for p in parts) else None
                )
                field_samples[selection.output_name] = parts[0][1] if len(parts) == 1 else tuple(p[1] for p in parts)
            identities[role] = MappingProxyType(evidence)
            samples[role] = InputSamples(
                native.get("topology"), native.get("geometry"), MappingProxyType(field_samples)
            )
        self.identities = MappingProxyType(identities)
        self.samples = MappingProxyType(samples)

    def resolve(self, context, role):
        """Materialize a declared role using the request already used for description.

        Convert captured scene Points or load scientific data under the current Frame
        Data lease, then apply selected field interpretation modes. Return a real
        simdata.Dataset with output field-role names. Unknown roles raise InvalidRecipe;
        provider resolution errors propagate. Any retention of loaded data is the
        provider's decision; this method holds nothing.
        """
        try:
            request = self.requests[role]
        except KeyError as exc:
            raise InvalidRecipe(f"input role {role!r} is not declared and available") from exc
        if request.scene:
            dataset = _point_cloud_from_prim(context.stage.get_prim_at_path(request.key), request.device)
        else:
            dataset = context._frame_data._resolve_dataset(
                request.key, identifier=request.identifier, device=request.device, fields=request.fields
            )
        _apply_field_modes(dataset, request.selections, request.fields)
        return dataset
