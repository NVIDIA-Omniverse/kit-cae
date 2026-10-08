# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Typed schema access over Usd.Prim (usd-core only)."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, Vt

import ovcae.authoring as author
from ovcae.authoring import schemas as S
from ovcae.schemas import SchemaError, schema_from_registry


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    return stage


@pytest.fixture
def mesh(stage):
    return stage.DefinePrim("/World/Mesh", "Mesh")


def test_apply_get_and_truthiness(mesh) -> None:
    assert S.FacesAPI.get(mesh) is None
    assert not S.FacesAPI(mesh)
    faces = S.FacesAPI.apply(mesh)
    assert faces and S.FacesAPI.get(mesh)
    assert mesh.GetAppliedSchemas() == ["CaeVizFacesAPI"]
    assert faces.applied_name == "CaeVizFacesAPI" and faces.prim == mesh
    # Applying again is idempotent.
    S.FacesAPI.apply(mesh)
    assert mesh.GetAppliedSchemas() == ["CaeVizFacesAPI"]


def test_attributes_read_defaults_and_write_typed_values(mesh) -> None:
    faces = S.FacesAPI.apply(mesh)
    assert faces.external_only is True
    assert not mesh.GetAttribute("cae:viz:faces:externalOnly").IsAuthored()
    faces.external_only = False
    assert mesh.GetAttribute("cae:viz:faces:externalOnly").Get() is False
    del faces.external_only
    assert not mesh.GetAttribute("cae:viz:faces:externalOnly").IsAuthored()
    assert faces.external_only is True
    with pytest.raises(SchemaError, match="None"):
        faces.external_only = None

    colors = S.FieldSelectionAPI.apply(mesh, "colors")
    assert colors.field_names is None and colors.mode == "unchanged"
    colors.field_names = ["Temperature", "Pressure"]
    stored = mesh.GetAttribute("cae:viz:field_selection:colors:fieldNames").Get()
    assert isinstance(stored, Vt.TokenArray) and list(stored) == ["Temperature", "Pressure"]
    colors.field_names = "Temperature"
    assert list(colors.field_names) == ["Temperature"]

    mapping = S.FieldMappingAPI.apply(mesh, "widths")
    assert mapping.domain == (0.0, -1.0)
    mapping.range = (0.05, 0.1)
    assert mesh.GetAttribute("cae:viz:field_mapping:widths:range").Get() == Gf.Vec2f(0.05, 0.1)


def test_allowed_tokens_are_enforced(mesh) -> None:
    colors = S.FieldSelectionAPI.apply(mesh, "colors")
    colors.mode = "vector_magnitude"
    with pytest.raises(SchemaError, match="expected one of unchanged, vector_magnitude"):
        colors.mode = "sideways"
    assert colors.mode == "vector_magnitude"


def test_malformed_values_raise_authoring_errors(stage, mesh) -> None:
    mapping = S.FieldMappingAPI.apply(mesh, "widths")
    for bad in ((1.0, 2.0, 3.0), "abc", object()):
        with pytest.raises(author.AuthoringError, match="cannot store"):
            mapping.range = bad
    assert mapping.range == (0.1, 1.0)
    points = S.PointsAPI.apply(stage.DefinePrim("/World/Points", "Points"))
    with pytest.raises(author.AuthoringError, match="cannot store"):
        points.max_count = "abc"
    colors = S.FieldSelectionAPI.apply(mesh, "colors")
    with pytest.raises(author.AuthoringError, match="cannot store"):
        colors.field_names = 5
    with pytest.raises(author.AuthoringError, match="does not allow '5'"):
        colors.mode = 5
    assert S.StreamlinesAPI.direction.problem(object()) is not None  # scalar token: reported, not raised
    assert "expects a token" in S.Attribute("x", "token[]", allowed_tokens=("a",)).problem(5)
    with pytest.raises(author.AuthoringError, match="needs a bool"):
        S.FacesAPI.apply(mesh).external_only = "no"


def test_clear_is_scoped_to_the_edit_target(stage, mesh) -> None:
    layer = Sdf.Layer.CreateAnonymous("sub.usda")
    stage.GetRootLayer().subLayerPaths.append(layer.identifier)
    faces = S.FacesAPI.apply(mesh)
    with Usd.EditContext(stage, Usd.EditTarget(layer)):
        faces.external_only = False
    del faces.external_only  # root layer holds no opinion; the sublayer's stands
    assert faces.external_only is False
    with Usd.EditContext(stage, Usd.EditTarget(layer)):
        del faces.external_only
    assert faces.external_only is True


def test_apply_checks_prim_type(stage, mesh) -> None:
    points = stage.DefinePrim("/World/Points", "Points")
    with pytest.raises(SchemaError, match="applies only to BasisCurves prims"):
        S.StreamlinesAPI.apply(mesh)
    with pytest.raises(SchemaError, match="applies only to Mesh prims"):
        S.FacesAPI.apply(points)
    assert S.PointsAPI.apply(points)
    forced = S.StreamlinesAPI.apply(mesh, check=False)
    assert forced and "CaeVizStreamlinesAPI" in mesh.GetAppliedSchemas()
    assert S.StreamlinesAPI.can_apply_to("BasisCurves") and not S.StreamlinesAPI.can_apply_to("Mesh")
    assert S.FieldSelectionAPI.can_apply_to("Mesh")


def test_writes_require_an_applied_schema(mesh) -> None:
    faces = S.FacesAPI(mesh)
    with pytest.raises(SchemaError, match="not applied"):
        faces.external_only = False
    assert faces.external_only is True


def test_instance_handling(mesh) -> None:
    with pytest.raises(SchemaError, match="multiple-apply"):
        S.FieldSelectionAPI.apply(mesh)
    with pytest.raises(SchemaError, match="single-apply"):
        S.FacesAPI(mesh, "x")
    with pytest.raises(SchemaError, match="no instances"):
        S.FacesAPI.instances(mesh)
    with pytest.raises(SchemaError, match="cannot contain"):
        S.FieldSelectionAPI.apply(mesh, "a:b")
    S.FieldSelectionAPI.apply(mesh, "colors").field_names = ["T"]
    S.FieldSelectionAPI.apply(mesh, "widths")
    instances = S.FieldSelectionAPI.instances(mesh)
    assert sorted(instances) == ["colors", "widths"]
    assert list(instances["colors"].field_names) == ["T"] and instances["widths"].field_names is None
    assert instances["colors"].property_name("mode") == "cae:viz:field_selection:colors:mode"
    assert set(S.FieldSelectionAPI.property_names("colors")) >= {
        "cae:viz:field_selection:colors:input",
        "cae:viz:field_selection:colors:fieldNames",
    }


def test_relationships_accept_prims_paths_and_properties(stage, mesh) -> None:
    dataset = stage.DefinePrim("/World/Dataset", "OmniSciDataset")
    selection = S.DatasetSelectionAPI.apply(mesh, "source")
    assert selection.target == ()
    selection.target = dataset
    assert selection.target == ("/World/Dataset",)
    assert mesh.GetRelationship("cae:viz:dataset_selection:source:target").GetTargets() == [dataset.GetPath()]
    selection.target = ["/World/Dataset", Sdf.Path("/World/Mesh")]
    assert selection.target == ("/World/Dataset", "/World/Mesh")
    with pytest.raises(SchemaError, match="targets prims"):
        selection.target = "/World/Dataset.points"
    with pytest.raises(SchemaError, match="absolute"):
        selection.target = "Dataset"
    del selection.target
    assert selection.target == ()

    shader = stage.DefinePrim("/World/Mesh/Materials/ScalarColor/Shader", "Shader")
    domain = shader.CreateAttribute("inputs:domain", Sdf.ValueTypeNames.Float2)
    rescale = S.RescaleRangeAPI.apply(mesh, "colors")
    rescale.includes = [domain, "/World/Mesh/Materials/ScalarColor/Shader.inputs:other"]
    assert rescale.includes == (
        "/World/Mesh/Materials/ScalarColor/Shader.inputs:domain",
        "/World/Mesh/Materials/ScalarColor/Shader.inputs:other",
    )
    with pytest.raises(SchemaError, match="targets properties"):
        rescale.includes = [shader]
    assert rescale.rescale_mode == "clamp"
    rescale.rescale_mode = "grow"
    with pytest.raises(SchemaError):
        rescale.rescale_mode = "shrink"


def test_escape_hatches_and_introspection(mesh) -> None:
    faces = S.FacesAPI.apply(mesh)
    attribute = faces.attribute("external_only")
    assert isinstance(attribute, Usd.Attribute) and attribute.GetName() == "cae:viz:faces:externalOnly"
    selection = S.DatasetSelectionAPI.apply(mesh, "source")
    assert isinstance(selection.relationship("target"), Usd.Relationship)
    with pytest.raises(SchemaError, match="is a relationship"):
        selection.attribute("target")
    with pytest.raises(SchemaError, match="no property"):
        faces.attribute("nope")
    assert S.FacesAPI(mesh).attribute("external_only") is not None  # schema is applied above
    assert S.StreamlinesAPI(mesh).attribute("max_steps") is None  # not applied: no property yet

    descriptor = S.StreamlinesAPI.direction
    assert descriptor.allowed_tokens == ("forward", "backward", "both")
    assert descriptor.display_name == "Direction" and descriptor.default == "both"
    assert list(S.FacesAPI.properties()) == ["external_only"]
    assert S.FacesAPI.__schema_doc__.startswith("API schema when applied to UsdGeomMesh")
    assert schema_from_registry("CaeVizFacesAPI") is S.FacesAPI


def test_builders_are_readable_through_typed_classes(stage) -> None:
    readers = pytest.importorskip("ovcae.readers")
    holder = readers.import_dataset(stage, "/World/Assets/Mixer", "./StaticMixer.cgns")
    seeds = author.create_seed_points(stage, "/World/Controls/Seeds", [(0, 0, 0)])
    curves = author.create_streamlines(
        stage,
        "/World/Recipes/Streamlines",
        source=holder,
        seeds=seeds,
        velocities=("Vx", "Vy", "Vz"),
        colors="Temperature",
        direction="forward",
        max_steps=20,
    )
    recipe = S.StreamlinesAPI.get(curves)
    assert recipe.direction == "forward" and recipe.max_steps == 20
    assert recipe.min_step_size == pytest.approx(0.01)  # schema fallback, not authored
    assert S.OperatorAPI(curves).enabled is True
    assert S.DatasetSelectionAPI(curves, "seeds").target == (str(seeds.GetPath()),)
    fields = S.FieldSelectionAPI.instances(curves)
    assert list(fields["velocities"].field_names) == ["Vx", "Vy", "Vz"]
    assert fields["velocities"].mode == "unchanged"
    assert set(fields) == {"velocities", "colors", "opacity", "widths"}
    assert S.RescaleRangeAPI(curves, "widths").includes == (
        "/World/Recipes/Streamlines.cae:viz:field_mapping:widths:domain",
    )
    assert author.validate_recipe(curves) == ()

    recipe.direction = "backward"
    assert curves.GetAttribute("cae:viz:streamlines:direction").Get() == "backward"
    with pytest.raises(author.AuthoringError):
        author.create_streamlines(
            stage, "/World/Recipes/Bad", source=holder, seeds=seeds, velocities="V", direction="sideways"
        )


def test_validate_recipe_reports_problems(stage, mesh) -> None:
    assert author.validate_recipe(mesh) == (
        "CaeVizOperatorAPI is not applied",
        "no operator API applied; expected one of CaeVizGlyphsAPI, CaeVizFacesAPI, CaeVizIsoSurfaceAPI, "
        "CaeVizPlanarSliceAPI, "
        "CaeVizPointsAPI, CaeVizStreamlinesAPI",
    )
    S.OperatorAPI.apply(mesh)
    S.StreamlinesAPI.apply(mesh, check=False)
    S.FieldSelectionAPI.apply(mesh, "velocities")
    S.DatasetSelectionAPI.apply(mesh, "source")
    mesh.GetAttribute("cae:viz:streamlines:direction").Set("sideways")
    problems = author.validate_recipe(mesh)
    assert "CaeVizStreamlinesAPI requires a BasisCurves prim; got Mesh" in problems
    assert "CaeVizStreamlinesAPI requires CaeVizDatasetSelectionAPI:seeds" in problems
    assert "dataset role 'source' has no target" in problems
    assert "field role 'velocities' selects no fields" in problems
    assert any(item.startswith("cae:viz:streamlines:direction is 'sideways'") for item in problems)

    faces = author.create_faces(stage, "/World/Recipes/Faces", source=stage.DefinePrim("/World/D", "OmniSciDataset"))
    assert author.validate_recipe(faces) == ()
    S.PointsAPI.apply(faces, check=False)
    assert author.validate_recipe(faces) == (
        "more than one operator API applied: CaeVizFacesAPI, CaeVizPointsAPI",
        "CaeVizPointsAPI requires a Points prim; got Mesh",
    )


def test_registry_schemas_get_typed_classes(stage) -> None:
    pytest.importorskip("ovcae.readers")
    cgns = S.schema_from_registry("OmniSciFileFormatArgsCgnsAPI")
    assert cgns.__kind__ == "singleApply"
    names = cgns.properties()
    assert "zone_name" in names and names["zone_name"].api_name == "zoneName"
    holder = stage.DefinePrim("/World/Holder")
    arguments = cgns.apply(holder)
    arguments.zone_name = "Zone1"
    assert holder.GetAttribute("omni:cae:format:cgns:zoneName").Get() == "Zone1"
    assert S.schema_from_registry("OmniSciFileFormatArgsCgnsAPI") is cgns
    with pytest.raises(SchemaError, match="not a registered"):
        S.schema_from_registry("NopeAPI")


def test_registry_handles_property_less_multiple_apply_schemas(stage) -> None:
    dual = S.schema_from_registry("CaeVizDatasetDualAPI")
    assert dual.__kind__ == "multipleApply" and dual.properties() == {}
    holder = stage.DefinePrim("/World/Holder")
    assert dual.apply(holder, "source")
    assert holder.GetAppliedSchemas() == ["CaeVizDatasetDualAPI:source"]


def test_usd_backend_raises_authoring_errors(mesh) -> None:
    assert issubclass(author.AuthoringError, SchemaError)
    assert author.SchemaError is SchemaError
    with pytest.raises(author.AuthoringError, match="applies only to"):
        S.StreamlinesAPI.apply(mesh)
    with pytest.raises(author.AuthoringError, match="does not allow"):
        S.PlanarSliceAPI.apply(mesh).mode = "diagonal"
    with pytest.raises(author.AuthoringError, match="not applied"):
        S.FacesAPI(mesh).external_only = False


def test_typed_classes_need_a_backend_without_authoring() -> None:
    code = (
        "import sys\n"
        "from ovcae.schemas.typed import FacesAPI, SchemaError\n"
        "assert 'ovcae.authoring' not in sys.modules\n"
        "try:\n    FacesAPI(object())\nexcept SchemaError as exc:\n"
        "    assert 'import ovcae.authoring' in str(exc), exc\nelse:\n    raise SystemExit('expected SchemaError')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_duck_typed_backend_works_without_pxr_types() -> None:
    class Fake:
        def __init__(self):
            self.type_name = "Mesh"
            self.applied = []
            self.values = {}
            self.types = {}
            self.rels = {}

        @property
        def applied_schemas(self):
            return tuple(self.applied)

        def apply(self, schema, instance):
            self.applied.append(f"{schema}:{instance}" if instance else schema)
            return True

        def get(self, name):
            return self.values.get(name)

        def set(self, name, value, type_name):
            self.types[name] = type_name
            self.values[name] = value

        def clear(self, name):
            self.values.pop(name, None)

        def targets(self, name):
            return tuple(self.rels.get(name, ()))

        def set_targets(self, name, targets, *, kind):
            self.rels[name] = [str(item) for item in targets]

        def native(self, name):
            return name

    fake = Fake()
    faces = S.FacesAPI.apply(fake)
    faces.external_only = False
    assert fake.values == {"cae:viz:faces:externalOnly": False}
    assert fake.types == {"cae:viz:faces:externalOnly": "bool"}
    assert faces.external_only is False
    with pytest.raises(SchemaError, match="does not allow"):
        S.StreamlinesAPI.apply(fake, check=False).direction = "sideways"
    selection = S.DatasetSelectionAPI.apply(fake, "source")
    selection.target = "/D"
    assert selection.target == ("/D",)
