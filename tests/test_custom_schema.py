# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""A custom operator reads its own schema's setting once the schema is registered with usd-core."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import ovstage
import pytest

import ovcae
from ovcae import operators
from ovcae.operators import utils as op_utils
from ovcae.schemas import register_usd_plugin

from ._ovstage_helpers import read_array, write_relationship, write_scalar, write_schemas, write_token
from ._simdata import tetrahedron

pytest.importorskip("warp_simdata")
pytest.importorskip("pxr")

SCHEMA_DIR = Path(__file__).parent / "data" / "scaled_points_schema"
SCHEMA = "CaeVizScaledPointsAPI"
SCALE = "cae:viz:scaled_points:scale"
UNREGISTERED = "CaeVizUnregisteredPointsAPI"
RECIPE = "/World/Recipes/Scaled"
SOURCE = "/World/Data"


@operators.operator(
    required_schemas={SCHEMA, "CaeVizDatasetSelectionAPI:source"},
    dataset_components={"source": {"geometry"}},
)
class ScaledPoints:
    """Publish the source points multiplied by the recipe's scale."""

    def execute(self, context):
        scale = float(context.prim.get_attribute(SCALE).get())
        source = op_utils.get_input_dataset(context, role="source")
        points = source.handle.points.numpy() * scale
        widths = op_utils.primvar("widths", np.asarray([0.1], dtype=np.float32), "constant")
        return op_utils.points(points=points, widths=widths)


@operators.operator(
    required_schemas={UNREGISTERED, "CaeVizDatasetSelectionAPI:source"},
    dataset_components={"source": {"geometry"}},
)
class UnregisteredPoints:
    """An operator bound to a schema no plug-in declares."""

    def execute(self, context):
        context.prim.get_attribute("cae:viz:unregistered_points:scale")
        return op_utils.empty("Points")


def _registry():
    registry = operators.OperatorRegistry.with_builtins()
    registry.register(ScaledPoints)
    registry.register(UnregisteredPoints)
    return registry


def _author(stage, schema, scale):
    write_schemas(stage, RECIPE, ["CaeVizOperatorAPI", schema, "CaeVizDatasetSelectionAPI:source"], 1)
    write_token(stage, RECIPE, "usd-prim-type", "Points", 1)
    write_scalar(stage, RECIPE, "cae:viz:operator:enabled", True, 1)
    write_relationship(stage, RECIPE, "cae:viz:dataset_selection:source:target", [SOURCE], 1)
    write_scalar(stage, RECIPE, SCALE, np.float32(scale), 1)
    stage.advance_write_floor(1).wait()


def _update(stage):
    source = ovcae.MemoryDataSource({SOURCE: tetrahedron("cpu")})
    with (
        source,
        ovcae.Session(stage, device="cpu", operators=_registry()) as session,
        source.at_time(0.0) as frame,
    ):
        return session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()


def test_register_usd_plugin_reports_the_custom_schema() -> None:
    assert register_usd_plugin(SCHEMA_DIR) == (SCHEMA,)
    from ovcae.schemas import find_schema

    klass = find_schema(SCHEMA)
    assert klass is not None and klass.scale.default == 1.0 and klass.scale.type_name == "float"


def test_custom_operator_reads_its_setting_once_the_schema_is_registered() -> None:
    register_usd_plugin(SCHEMA_DIR)
    stage = ovstage.Stage("ovcae.tests.custom-schema")
    try:
        _author(stage, SCHEMA, scale=3.0)
        report = _update(stage)
        assert report.committable and not report.diagnostics, report.diagnostics
        assert report.operators[0].status == "succeeded"
        stage.advance_write_floor(2).wait()
        points = np.asarray(read_array(stage, RECIPE, "points", 2)).reshape(-1, 3)
        np.testing.assert_allclose(points.max(axis=0), [3.0, 3.0, 3.0])
    finally:
        stage.destroy()


def test_unregistered_schema_is_reported_and_its_setting_not_captured() -> None:
    stage = ovstage.Stage("ovcae.tests.custom-schema-uncaptured")
    try:
        _author(stage, UNREGISTERED, scale=3.0)
        report = _update(stage)
        diagnostics = {diagnostic.code: diagnostic for diagnostic in report.diagnostics}
        assert "SCHEMA_NOT_CAPTURED" in diagnostics
        assert UNREGISTERED in diagnostics["SCHEMA_NOT_CAPTURED"].message
        assert report.operators[0].status != "succeeded"
    finally:
        stage.destroy()


def test_authoring_import_does_not_freeze_the_registry() -> None:
    """Importing ovcae.authoring must leave room to register a custom schema afterwards."""

    code = (
        "import ovcae, ovcae.authoring\n"
        "from pxr import Usd\n"
        f"assert ovcae.schemas.register_usd_plugin({str(SCHEMA_DIR)!r}) == ({SCHEMA!r},)\n"
        f"assert Usd.SchemaRegistry().FindAppliedAPIPrimDefinition({SCHEMA!r}) is not None\n"
        "from ovcae.authoring import schemas as S\n"
        "assert S.FacesAPI.__schema__ == 'CaeVizFacesAPI' and 'FacesAPI' in dir(S)\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_registered_schemas_always_include_the_bundle(monkeypatch) -> None:
    from ovcae import _setup

    monkeypatch.setattr(_setup, "_REGISTERED_SCHEMAS", {SCHEMA})
    names = ovcae.registered_schemas()
    assert SCHEMA in names and "CaeVizFacesAPI" in names and "CaeVizOperatorAPI" in names


def test_population_desc_takes_explicit_schemas() -> None:
    desc = ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL, schemas=("CaeVizFacesAPI", SCHEMA))
    assert desc is not None
    assert ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL) is not None
