# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""One-call process initialization for ovcae's order-sensitive registrations."""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points
from pathlib import Path

from .materials import register_material_library
from .schemas import register_usd_plugin, schema_resource_path

READER_ENTRY_POINT_GROUP = "ovcae.readers"


def setup(*, strict: bool = False) -> None:
    """Perform every registration that must precede native initialization.

    Several steps have to run at the right moment in a process, and all are easy
    to miss or misorder:

    - Every installed reader must import before usd-core creates its first
      stage, because usd-core fixes its file-format registry then. Readers are
      discovered through the ``ovcae.readers`` entry-point group, so a reader
      from any wheel takes part; see :func:`load_readers`.
    - The codeless CAE schemas must be registered with usd-core and with
      OVStage before either reads a schema definition, so recipe prims resolve
      their API schemas on usd-core stages, their properties populate with
      fallbacks, and the Session finds the definitions its input scopes are
      built from. :func:`ovcae.register_schema_paths` does both.
    - :func:`ovcae.register_material_library` must run before
      ``ovrtx.Renderer()`` is constructed, because the MDL SDK reads its
      search paths once at initialization.

    ``setup()`` runs all of them in that order. Call it after
    ``ovrtx.register_schema_paths()`` (when rendering with OVRTX; that
    registration belongs to OVRTX and must come first) and before creating any
    stage or renderer. A reader whose optional dependencies are missing is
    skipped unless ``strict`` is true. Repeated calls are harmless. Importing
    ``ovcae`` still performs none of this; initialization stays explicit.
    """

    load_readers(strict=strict)
    register_schema_paths()
    register_material_library()


def load_readers(*, strict: bool = False) -> tuple[str, ...]:
    """Import every reader package declared in the ``ovcae.readers`` entry-point group.

    Importing a reader registers what usd-core needs to compose its datasets.
    Returns the names of the readers that loaded. A reader that raises
    ``ImportError`` is missing an optional dependency and is skipped with a
    warning, or re-raised when ``strict`` is true; any other error, such as a
    reader detecting that a stage was created before it could register, always
    propagates because it describes a process that cannot be repaired.
    """

    loaded = []
    for entry in entry_points(group=READER_ENTRY_POINT_GROUP):
        try:
            entry.load()
        except ImportError as exc:
            if strict:
                raise
            warnings.warn(f"ovcae reader {entry.name!r} is not available: {exc}", RuntimeWarning, stacklevel=2)
            continue
        loaded.append(entry.name)
    return tuple(loaded)


def codeless_schema_path() -> Path:
    """Return the resource directory containing the CAE codeless plug-in.

    The same directory :func:`ovcae.schemas.schema_resource_path` returns.
    """

    return schema_resource_path()


_REGISTERED_SCHEMAS: set[str] = set()


def register_schema_paths(*directories: str | Path) -> tuple[str, ...]:
    """Register codeless schema plug-in directories with usd-core and OVStage.

    With no arguments the bundled OmniCaeViz plug-in is registered. A custom
    operator registers the directory holding its own ``plugInfo.json`` and
    ``generatedSchema.usda`` the same way, before the process creates any
    stage; afterwards its schema is applied with fallbacks on usd-core stages,
    populated into OVStage by :func:`ovstage_population_desc`, and captured in
    the scene view its operator reads through. Returns the identifiers of the
    applied API schemas the directories declare. Repeating a registration is
    harmless.
    """

    import ovstage

    identifiers = register_usd_plugin(*directories)
    for directory in directories or (schema_resource_path(),):
        ovstage.population.register_usd_schemas(str(directory))
    _REGISTERED_SCHEMAS.update(identifiers)
    return identifiers


def registered_schemas() -> tuple[str, ...]:
    """Identifiers of the bundled applied API schemas plus every one :func:`register_schema_paths` added.

    The bundle is registered with usd-core on the way when nothing has done so
    yet, so the result always carries the OmniCaeViz schemas.
    """

    return tuple(sorted(set(register_usd_plugin()) | _REGISTERED_SCHEMAS))


def ovstage_population_desc(*, domains, schemas=None):
    """Describe standard-domain and CAE schema-property population for OVStage.

    ``schemas`` names the applied API schemas whose declared properties
    population carries into OVStage. It defaults to every schema registered
    through :func:`register_schema_paths` (or :func:`setup`), which includes
    the bundled OmniCaeViz schemas and any custom operator schemas registered
    the same way.
    """

    import ovstage

    schema_names = registered_schemas() if schemas is None else tuple(schemas)
    properties = ovstage.population.PropertyPredicate.declared_by_schema(*schema_names)
    return ovstage.population.Desc(
        domains=domains,
        selectors=(
            ovstage.population.Selector(
                prim_predicate=ovstage.population.PrimPredicate.has_property(properties),
                property_predicate=properties,
            ),
        ),
    )
