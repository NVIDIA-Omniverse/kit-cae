# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Reusable helpers for ovcae operator implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .._products import CurvesProduct, EmptyProduct, InstancerProduct, MeshProduct, PointsProduct, Primvar
from ..scene._plan import POINT_BASED_TYPES
from ._errors import InvalidRecipe


def primvar(name: str, values, interpolation: str) -> Primvar:
    """Describe one standard primvar for a returned operator product."""

    return Primvar(str(name), values, str(interpolation))


def mesh(
    *,
    points,
    face_vertex_counts,
    face_vertex_indices,
    primvars: Sequence[Primvar] = (),
) -> MeshProduct:
    return MeshProduct(points, face_vertex_counts, face_vertex_indices, tuple(primvars))


def mesh_from_surface(surface, *, exclude_fields: set[str] | frozenset[str] = frozenset()) -> MeshProduct:
    """Describe a Warp SimData surface mesh as a standard Mesh product."""

    import warp_simdata as simdata

    primvars = []
    for name in surface.get_field_names():
        if name in exclude_fields:
            continue
        field = surface.get_field(name)
        if field.association == simdata.AssociationType.NODE:
            interpolation = "vertex"
        elif field.association == simdata.AssociationType.ELEMENT:
            interpolation = "uniform"
        else:
            continue
        primvars.append(primvar(name, field.to_array(), interpolation))
    return mesh(
        points=surface.handle.points,
        face_vertex_counts=surface.handle.face_vertex_counts,
        face_vertex_indices=surface.handle.face_vertex_indices,
        primvars=primvars,
    )


def mesh_from_surfaces(
    surfaces,
    *,
    exclude_fields: set[str] | frozenset[str] = frozenset(),
) -> MeshProduct:
    """Merge one or more Warp SimData surfaces into one standard Mesh."""

    import warp_simdata as simdata

    surfaces = tuple(surface for surface in surfaces if surface.get_num_nodes() and surface.get_num_elems())
    if not surfaces:
        raise ValueError("at least one non-empty surface is required")
    if len(surfaces) == 1:
        return mesh_from_surface(surfaces[0], exclude_fields=exclude_fields)

    from ._compute import concatenate_arrays, merge_surface_geometry

    points, counts, indices = merge_surface_geometry(surfaces)
    common_fields = set(surfaces[0].get_field_names()).intersection(
        *(set(surface.get_field_names()) for surface in surfaces[1:])
    )
    primvars = []
    for name in sorted(common_fields - exclude_fields):
        fields = [surface.get_field(name) for surface in surfaces]
        association = fields[0].association
        if any(field.association != association for field in fields[1:]):
            continue
        if association == simdata.AssociationType.NODE:
            interpolation = "vertex"
        elif association == simdata.AssociationType.ELEMENT:
            interpolation = "uniform"
        else:
            continue
        primvars.append(
            primvar(
                name,
                concatenate_arrays([field.to_array() for field in fields]),
                interpolation,
            )
        )
    return mesh(
        points=points,
        face_vertex_counts=counts,
        face_vertex_indices=indices,
        primvars=primvars,
    )


def curves(
    *,
    points,
    curve_vertex_counts,
    widths: Primvar,
    primvars: Sequence[Primvar] = (),
    curve_type: str = "linear",
    basis: str | None = None,
    wrap: str = "nonperiodic",
) -> CurvesProduct:
    return CurvesProduct(
        points,
        curve_vertex_counts,
        widths,
        tuple(primvars),
        curve_type,
        basis,
        wrap,
    )


def points(
    *,
    points,
    widths: Primvar,
    primvars: Sequence[Primvar] = (),
) -> PointsProduct:
    return PointsProduct(points, widths, tuple(primvars))


def instancer(*, positions, proto_indices, scales, orientations, primvars: Sequence[Primvar] = ()) -> InstancerProduct:
    """Describe PointInstancer arrays; orientations are float16 quaternions in xyzw order.

    The application authors and populates the prototypes relationship and hierarchy.
    Positions/scales are float32 triples and proto_indices are int32. All arrays
    have one entry per instance, except orientations may be empty for identity.
    """
    return InstancerProduct(positions, proto_indices, scales, orientations, tuple(primvars))


def empty(reason: str = "operator produced no geometry") -> EmptyProduct:
    return EmptyProduct(reason)


def is_point_based(prim) -> bool:
    """True for a captured UsdGeomPointBased prim, whose ``points`` become a point-cloud dataset."""
    return prim is not None and prim.type_name in POINT_BASED_TYPES


def _scene_input(target) -> bool:
    """A present PointBased target is bounded scene geometry; anything else is Frame Data."""
    return is_point_based(target)


def get_input_dataset(
    context,
    *,
    role: str,
    topology: bool = True,
    geometry: bool = True,
    fields: Mapping[str, Sequence[str]] | set[str] | tuple[str, ...] = (),
):
    """Load an input eagerly and return a real simdata.Dataset.

    For annotated operators, resolve role through the Update's shared InputPlan,
    including field interpretation; repeated topology/geometry/fields arguments are
    rejected when they request nondefault behavior. Legacy operators derive loading
    arguments from the scene view here. That path retains historical compatibility:
    geometry is ignored and topology is not forwarded to the provider. Scene Points
    are converted from the scene view; scientific data loads under the Frame Data lease.
    """
    if context._inputs is not None:
        if fields or not topology or not geometry:
            raise InvalidRecipe(
                "declared inputs use get_input_dataset(context, role=...) without repeated requirements"
            )
        return context._inputs.resolve(context, role)
    del geometry
    relationship_name = f"cae:viz:dataset_selection:{role}:target"
    relationship = context.prim.get_relationship(relationship_name)
    if relationship is None or len(relationship.targets) != 1:
        raise InvalidRecipe(f"{relationship_name} must have exactly one target")
    if isinstance(fields, Mapping):
        requested = {
            str(name): (values,) if isinstance(values, str) else tuple(values) for name, values in fields.items()
        }
    else:
        selections = {item.output_name: item for item in get_field_selections(context, input=role)}
        requested = {}
        for name in fields:
            name = str(name)
            try:
                requested[name] = selections[name].field_names
            except KeyError as exc:
                raise InvalidRecipe(f"missing CaeVizFieldSelectionAPI:{name}") from exc
    identifier = str(context.prim.get_attribute(f"cae:viz:dataset_selection:{role}:identifier").get() or "")
    target_path = relationship.targets[0]
    target_prim = context.stage.get_prim_at_path(target_path)
    if _scene_input(target_prim):
        if identifier:
            raise InvalidRecipe(f"{relationship_name} targets scene geometry and cannot use an identifier")
        if requested:
            raise InvalidRecipe(f"{relationship_name} targets scene geometry and cannot provide selected fields")
        dataset = _point_cloud_from_prim(target_prim, context.device)
    else:
        dataset = context._frame_data._resolve_dataset(
            target_path,
            identifier=identifier,
            device=context.device,
            fields=requested,
        )
    _apply_field_modes(dataset, get_field_selections(context), requested)
    return dataset


def _point_cloud_from_prim(prim, device: str):
    import numpy as np
    import warp as wp
    from warp_simdata.data_models.custom import point_cloud

    if "points" not in prim.attribute_names():
        raise InvalidRecipe(f"PointBased target {prim.path} has no points")
    value = prim.get_attribute("points").get()
    if value is None:
        raise InvalidRecipe(f"PointBased target {prim.path} has no points")
    points = np.from_dlpack(value) if hasattr(value, "__dlpack__") else np.asarray(value)
    points = points.astype(np.float32, copy=False)
    if points.ndim == 1 and points.size % 3 == 0:
        points = points.reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] != 3:
        raise InvalidRecipe(f"PointBased target {prim.path} must provide an N x 3 points array")
    return point_cloud.create_dataset(wp.array(points, dtype=wp.vec3f, device=device))


@dataclass(frozen=True)
class FieldSelection:
    """One output field role and its schema-selected source arrays and interpretation.

    Input names the dataset-selection instance. Output_name is the field name exposed
    to operator code; field_names identifies contributing scientific arrays. Mode and
    component_index specify how those arrays are interpreted after materialization.
    """

    output_name: str
    field_names: tuple[str, ...]
    mode: str
    component_index: int
    input: str = "source"


def get_field_selections(context, *, input: str | None = None, exclude: set[str] | frozenset[str] = frozenset()):
    """Return matched field selections, optionally filtered by input role and output name.

    Annotated operators use the selections already validated in InputPlan. Legacy
    operators read applied field-selection schemas from the scene view. input=None
    includes all input roles; exclude contains output field-role names.
    """
    if context._inputs is not None:
        return tuple(
            s
            for s in context._inputs.selections
            if s.output_name not in exclude and (input is None or s.input == input)
        )
    selections = (
        _field_selection(context.prim, instance, optional=True)
        for instance in context.prim.api_instances("CaeVizFieldSelectionAPI")
        if instance not in exclude
    )
    return tuple(s for s in selections if s is not None and (input is None or s.input == input))


def _field_selection(prim, instance, *, optional: bool = False):
    """Decode one applied schema instance.

    An optional instance with no authored fieldNames was applied but never selected
    (the authoring layer's usual way to leave a role unset) and decodes to None
    instead of raising, so callers can filter it out like an unmatched instance.
    """
    base = f"cae:viz:field_selection:{instance}"
    input_role = str(prim.get_attribute(f"{base}:input").get() or "source")
    names = prim.get_attribute(f"{base}:fieldNames").get()
    if isinstance(names, str):
        names = (names,)
    names = tuple(str(name) for name in (names or ()))
    if not names:
        if optional:
            return None
        raise InvalidRecipe(f"field selection {instance!r} has no fieldNames")
    return FieldSelection(
        instance,
        names,
        str(prim.get_attribute(f"{base}:mode").get()),
        int(prim.get_attribute(f"{base}:componentIndex").get()),
        input_role,
    )


def probe_fields(source, result, *, selections: Sequence[FieldSelection]) -> tuple[Primvar, ...]:
    from warp_simdata.operators import probe

    primvars = []
    for selection in selections:
        name = selection.output_name
        if not result.has_field(name):
            probed = probe.compute(source, name, result, output_field_name=name)
            result.add_field(name, probed.get_field(name), warn_if_exists=False)
        primvars.append(primvar(name, result.get_field(name).to_array(), "vertex"))
    return tuple(primvars)


def resolve_widths(context, *, result, role: str, default: float) -> Primvar:
    import numpy as np

    if context.prim.has_api("CaeVizFieldSelectionAPI", instance=role) and result.has_field(role):
        return primvar("widths", result.get_field(role).to_array(), "vertex")
    return primvar("widths", np.asarray([default], dtype=np.float32), "constant")


def get_local_transform(prim):
    """Return one scene prim's local transform as a 4x4 NumPy matrix."""

    from ._compute import local_transform_matrix

    return local_transform_matrix(prim)


def _apply_field_modes(dataset, selections: Sequence[FieldSelection], requested: Mapping[str, Sequence[str]]) -> None:
    import warp_simdata as simdata

    for selection in selections:
        if selection.output_name not in requested:
            continue
        field = dataset.get_field(selection.output_name)
        if selection.mode == "vector_magnitude":
            if simdata.utils.is_vector_dtype(field.dtype):
                field = simdata.Field.from_field(field, magnitude=True)
        elif selection.mode == "selected_component":
            if simdata.utils.is_vector_dtype(field.dtype):
                field = simdata.Field.from_field(field, component=selection.component_index)
        elif selection.mode != "unchanged":
            raise InvalidRecipe(f"unsupported field selection mode {selection.mode!r}")
        dataset.add_field(selection.output_name, field, warn_if_exists=False)


def get_input_samples(context, *, role: str):
    """Return current native sample metadata for a declared input role.

    Descriptions come from the input plan; this performs no loading. Unknown provider
    evidence appears as None in the description. An unavailable role raises
    InvalidRecipe. Native times describe selected samples, not application time.
    """
    context.stage._require()
    if context._inputs is None:
        raise InvalidRecipe("sample inspection requires an operator with dataset_components")
    try:
        return context._inputs.samples[role]
    except KeyError as exc:
        raise InvalidRecipe(f"input role {role!r} is not available") from exc
