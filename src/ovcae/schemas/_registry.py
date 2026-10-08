# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed schema classes for whatever applied API schemas usd-core has registered.

usd-core's ``Usd.SchemaRegistry`` is the one source of schema knowledge: the
bundled OmniCaeViz schemas, a custom operator's codeless schema, and the
readers' file-format argument schemas all enter through ``Plug`` registration
and are answered from here. Classes are built once per schema on first use;
nothing is generated ahead of time. ``pxr`` is imported on call, never at
import.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from ._typed import AppliedSchema, Attribute, Relationship, SchemaError, make_schema_class

_INSTANCE = "__INSTANCE_NAME__"
_FLOAT_TYPES = ("float", "half", "color3f", "color4f", "point3f", "normal3f", "vector3f", "texCoord2f")


def register_usd_plugin(*directories: str | Path) -> tuple[str, ...]:
    """Register codeless schema plug-in directories with usd-core's ``Plug`` registry.

    With no arguments the bundled OmniCaeViz plug-in is registered. Returns the
    identifiers of the applied API schemas the directories declare, read from
    their ``plugInfo.json``, whether this call registered them or an earlier
    one did; ``Plug`` ignores a path it already knows.

    usd-core builds its schema registry once, on the first schema read in the
    process (creating a stage, applying a schema, asking for a definition), and
    a plug-in registered after that is never consulted. Register every custom
    schema first. This function itself performs no schema read. Raises
    ImportError when the ``pxr`` bindings are not installed.
    """

    from pxr import Plug

    from . import schema_resource_path

    paths = [str(Path(item).resolve()) for item in directories or (schema_resource_path(),)]
    Plug.Registry().RegisterPlugins(paths)
    identifiers = []
    for path in paths:
        plugins = [plugin for plugin in Plug.Registry().GetAllPlugins() if _plugin_owns(plugin, path)]
        if not plugins:
            raise SchemaError(f"no plugInfo.json registered from {path}")
        for plugin in plugins:
            identifiers.extend(_applied_api_schemas(plugin))
    return tuple(sorted(set(identifiers)))


def _plugin_owns(plugin, path: str) -> bool:
    return any(str(Path(item).resolve()) == path for item in (plugin.path, plugin.resourcePath) if item)


def _applied_api_schemas(plugin) -> list[str]:
    """Applied API schema identifiers a plug-in declares, from its metadata alone.

    Reading through ``Usd.SchemaRegistry`` would build the registry and freeze
    the set of known schemas, so this stays with what ``usdGenSchema`` wrote.
    """

    found = []
    for type_name, info in plugin.metadata.get("Types", {}).items():
        if info.get("schemaKind") not in {"singleApplyAPI", "multipleApplyAPI"}:
            continue
        identifier = info.get("schemaIdentifier") or info.get("alias", {}).get("UsdSchemaBase") or type_name
        found.append(str(identifier))
    return found


def find_schema(name: str) -> type[AppliedSchema] | None:
    """Return the typed class for applied API schema ``name``, or ``None`` when usd-core does not know it.

    A miss is not cached, so a schema registered later is found later.
    """

    from pxr import Usd

    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(name) is None:
        return None
    return schema_from_registry(name)


def is_usd_builtin_schema(name: str) -> bool:
    """Whether ``name`` is a schema usd-core itself ships, such as ``MaterialBindingAPI``.

    Built-in plug-ins are compiled into the bindings or live inside the ``pxr``
    package; anything registered from another directory is not built in.
    Unknown names are not built in either.
    """

    import pxr
    from pxr import Plug, Usd

    tf_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(name)
    if tf_type.isUnknown:
        return False
    plugin = Plug.Registry().GetPluginForType(tf_type)
    if plugin is None:
        return False
    return not plugin.path or Path(plugin.path).is_relative_to(Path(pxr.__file__).parent)


@cache
def schema_from_registry(name: str) -> type[AppliedSchema]:
    """Build the typed class for any applied API schema usd-core has registered, once.

    ``CaeVizFacesAPI`` becomes ``FacesAPI`` with one descriptor per declared
    property, snake-cased from the property's leaf name; ``OmniSci*`` and
    ``CaeViz*`` prefixes are dropped from the class name. Relationships whose
    name ends in ``includes`` target properties, as Kit-CAE's rescale wiring
    does. A multiple-apply schema's namespace is recovered from its property
    names, so a property-less one gets a class with no attributes and no
    namespace. Raises :class:`SchemaError` when usd-core does not know the
    schema.
    """

    from pxr import Sdf, Usd

    registry = Usd.SchemaRegistry()
    definition = registry.FindAppliedAPIPrimDefinition(name)
    if definition is None:
        raise SchemaError(f"{name!r} is not a registered applied API schema")
    multiple = registry.IsMultipleApplyAPISchema(name)
    namespace = None
    properties = []
    for full_name in definition.GetPropertyNames():
        spec = definition.GetSchemaPropertySpec(full_name)
        leaf = full_name
        if multiple:
            prefix, _, leaf = full_name.partition(f"{_INSTANCE}:")
            namespace = prefix.rstrip(":")
        common = {
            "display_name": spec.displayName,
            "display_group": spec.displayGroup,
            # usdGenSchema moves a codeless property's doc string into customData.
            "doc": spec.documentation or str(spec.customData.get("userDocBrief", "")),
        }
        if isinstance(spec, Sdf.RelationshipSpec):
            targets = "properties" if leaf.lower().endswith("includes") else "prims"
            properties.append(Relationship(leaf, targets=targets, **common))
        else:
            type_name = str(spec.typeName)
            default = spec.default
            properties.append(
                Attribute(
                    leaf,
                    type_name,
                    default=None if default is None else plain_value(default, type_name),
                    has_default=default is not None,
                    allowed_tokens=tuple(str(token) for token in (spec.allowedTokens or ())),
                    variability=str(spec.variability).removeprefix("Sdf.Variability"),
                    **common,
                )
            )
    can_apply = tuple(_usd_type_name(item) for item in Usd.SchemaRegistry.GetAPISchemaCanOnlyApplyToTypeNames(name))
    return make_schema_class(
        name.removeprefix("OmniSci").removeprefix("CaeViz"),
        name,
        kind="multipleApply" if multiple else "singleApply",
        namespace=namespace,
        can_only_apply_to=can_apply,
        doc=definition.GetDocumentation() if hasattr(definition, "GetDocumentation") else "",
        properties=properties,
        module="ovcae.schemas.typed",
    )


def _usd_type_name(name: str) -> str:
    """``UsdGeomMesh`` -> ``Mesh``; names pxr cannot resolve (Kit-only types) pass through."""

    from pxr import Tf, Usd

    tf_type = Tf.Type.FindByName(name)
    return name if tf_type.isUnknown else (Usd.SchemaRegistry.GetSchemaTypeName(tf_type) or name)


def plain_value(value, type_name: str = ""):
    """Reduce a pxr value to plain Python: scalars, ``str`` paths, and tuples for vectors and arrays.

    Float fallbacks are rounded to their shortest float32 decimal so a schema
    default of ``0.2`` does not read back as ``0.20000000298``.
    """

    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return float(f"{value:.7g}") if type_name.startswith(_FLOAT_TYPES) else value
    if hasattr(value, "path"):
        return str(value.path)
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        element = type_name.removesuffix("[]")
        return tuple(plain_value(item, element) for item in value)
    return str(value)


__all__ = [
    "find_schema",
    "is_usd_builtin_schema",
    "plain_value",
    "register_usd_plugin",
    "schema_from_registry",
]
