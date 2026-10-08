# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Bounded read-through access to application-stable OVStage inputs."""

from __future__ import annotations

import re

from .._errors import OvcaeError
from .._stage_io import read_value

Path = str
_PATH = re.compile(r"^/(?:[^/]+)(?:/[^/]+)*$")


class SceneViewError(OvcaeError):
    """Base class for operator scene access failures."""


class OutOfScopeError(SceneViewError, KeyError):
    """Raised for access outside the declared input scope."""


class ExpiredViewError(SceneViewError):
    """Raised when a scene handle is used after execution."""


def normalize_path(value: str) -> Path:
    if not isinstance(value, str) or not _PATH.fullmatch(value):
        raise ValueError(f"invalid absolute prim path: {value!r}")
    return value.rstrip("/") or "/"


class AttributeView:
    """An allowed property definition; get reads Stage and applies schema defaults.

    Returned values own their storage. Native read groups never outlive a get call,
    and returned arrays need no execution lease or custom DLPack wrapper.
    """

    def __init__(self, prim, definition):
        self._prim = prim
        self._definition = definition

    @property
    def name(self):
        self._prim.stage._require()
        return self._definition.name

    @property
    def type_name(self):
        self._prim.stage._require()
        return self._definition.type_name or ""

    def get(self):
        """Read the current committed value, copying arrays before native release."""
        stage = self._prim.stage
        stage._require()
        definition = self._definition
        value = stage._read(self._prim.path, definition.name, type_name=definition.type_name)
        if value is None and definition.has_default:
            value = _plain_default(definition.default)
        return value


class RelationshipView:
    """A declared relationship whose targets are read on demand."""

    def __init__(self, prim, name):
        self._prim = prim
        self._name = name

    @property
    def name(self):
        self._prim.stage._require()
        return self._name

    @property
    def targets(self):
        stage = self._prim.stage
        stage._require()
        value = stage._read(self._prim.path, self._name)
        if value is None:
            return ()
        return tuple(normalize_path(item) for item in ((value,) if isinstance(value, str) else value))


class PrimView:
    """Prim identity and allowed definitions, without stored property values.

    Type and schemas come from dependency discovery. Application sequencing keeps
    them and the read-through property values stable until Update.wait completes.
    """

    def __init__(self, stage, path, type_name, schemas, definitions):
        self._stage = stage
        self._path = path
        self._type_name = type_name
        self._schemas = tuple(schemas)
        self._definitions = definitions

    @property
    def stage(self):
        self._stage._require()
        return self._stage

    @property
    def path(self):
        self._stage._require()
        return self._path

    @property
    def type_name(self):
        self._stage._require()
        return self._type_name

    @property
    def applied_api_schemas(self):
        self._stage._require()
        return self._schemas

    def has_api(self, schema_name, *, instance=None):
        token = f"{schema_name}:{instance}" if instance is not None else schema_name
        schemas = self.applied_api_schemas
        return token in schemas or (instance is None and any(item.startswith(f"{token}:") for item in schemas))

    def api_instances(self, schema_name):
        prefix = f"{schema_name}:"
        return tuple(item[len(prefix) :] for item in self.applied_api_schemas if item.startswith(prefix))

    def attribute_names(self):
        """List allowed attributes, including currently absent/defaulted properties."""
        self._stage._require()
        return tuple(name for name, definition in self._definitions.items() if definition.kind != "relationship")

    def relationship_names(self):
        self._stage._require()
        return tuple(name for name, definition in self._definitions.items() if definition.kind == "relationship")

    def get_attribute(self, name):
        self._stage._require()
        if name not in self.attribute_names():
            raise OutOfScopeError(f"attribute {name!r} is outside the input scope of {self._path}")
        return AttributeView(self, self._definitions[name])

    def get_relationship(self, name):
        self._stage._require()
        if name not in self.relationship_names():
            raise OutOfScopeError(f"relationship {name!r} is outside the input scope of {self._path}")
        return RelationshipView(self, name)


class StageView:
    """Execution-scoped read-through view with a metadata-only dependency plan.

    Only explicit recipe/target prims and allowed properties are exposed. No values
    or native read groups are retained. Expiry invalidates handles, not owned values
    already returned to callers. Session alone constructs and expires this view.
    """

    def __init__(self, native, ordinal, *, paths=None):
        self._paths = paths
        self._owns_paths = paths is None
        self._queries = {}
        self._native = native
        self._ordinal = ordinal
        self._active = True
        self._prims = {}
        self.dependencies = ()
        self.unknown_schemas = ()

    def _require(self):
        if not self._active:
            raise ExpiredViewError("operator scene view has expired")

    @property
    def ordinal(self):
        self._require()
        return self._ordinal

    def get_prim_at_path(self, path):
        self._require()
        path = normalize_path(path)
        if path not in self._prims:
            raise OutOfScopeError(f"prim path {path!r} is outside the input scope")
        return self._prims[path]

    def _read(self, path, name, *, type_name=None):
        """Reuse one explicit-path query per prim, never retaining a native group.

        Session lends its dictionary. Standalone plans lazily own a dictionary
        instead. A failed query construction releases the path list immediately.
        """
        self._require()
        if self._paths is None:
            import ovstage

            self._paths = ovstage.PathDictionary(self._native)
        if path not in self._queries:
            plist = self._paths.create_path_list_from_strings([path])
            try:
                query = self._native.query_from_path_list(plist)
            except BaseException:
                self._paths.destroy_path_list(plist)
                raise
            self._queries[path] = (self._paths, plist, query)
        return read_value(self._native, path, name, self._ordinal, type_name=type_name, resources=self._queries[path])

    def _expire(self):
        """Release all queries/lists and any owned dictionary, then revoke handles."""
        if not self._active:
            return
        self._active = False
        try:
            # Attempt every release even when a native cleanup operation fails.
            from contextlib import ExitStack

            with ExitStack() as cleanup:
                if self._owns_paths and self._paths is not None:
                    cleanup.callback(self._paths.destroy)
                for paths, plist, query in self._queries.values():
                    cleanup.callback(paths.destroy_path_list, plist)
                    cleanup.callback(lambda q=query: q.release().wait())
        finally:
            self._queries.clear()
            self._paths = None
            self._native = None


def _plain_default(value):
    if isinstance(value, list):
        return tuple(_plain_default(item) for item in value)
    if isinstance(value, dict):
        return {key: _plain_default(item) for key, item in value.items()}
    return value
