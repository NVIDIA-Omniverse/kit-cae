# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Glyphs authoring, real Session publication, device parity, and lifecycle coverage."""

from contextlib import contextmanager

import numpy as np
import ovstage
import pytest
import warp as wp
from pxr import Sdf, Usd, UsdGeom

import ovcae
import ovcae.authoring as author

from ._ovstage_helpers import read_array, write_scalar, write_schemas, write_token
from ._simdata import dataset_from_bundle

PATH = "/Glyphs"


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


@contextmanager
def glyph_pipeline(*, device="cpu", configure=None, materials=False, **kwargs):
    ovcae.setup()
    usd = Usd.Stage.CreateInMemory()
    source = usd.DefinePrim("/Data", "OmniSciDataset")
    prim = author.create_glyphs(usd, PATH, source=source, materials=materials, use_cell_points=False, **kwargs)
    if configure:
        configure(prim)
    stage = ovstage.Stage("ovcae.tests.glyphs")
    try:
        ovstage.population.open_usd_from_string_with_desc(
            stage,
            usd.GetRootLayer().ExportToString(),
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
        )
        stage.advance_write_floor(1).wait()
        with ovcae.Session(stage, device=device) as session:
            ordinal = 1

            def update(points, fields=None, **topology):
                nonlocal ordinal
                from ovcae._stage_io import write_floor

                ordinal = write_floor(stage) + 1
                bundle = {"points": np.asarray(points, np.float32).reshape(-1, 3), "fields": fields or {}, **topology}
                with (
                    ovcae.MemoryDataSource({"/Data": dataset_from_bundle(bundle, device)}) as source,
                    source.at_time(0) as frame,
                ):
                    report = session.update(input_ordinal=ordinal - 1, output_ordinal=ordinal, frame_data=frame).wait()
                assert report.committable, report.diagnostics
                stage.advance_write_floor(ordinal).wait()
                return report

            yield stage, prim, update
    finally:
        stage.destroy()


def field(values, association="node"):
    return {"association": association, "values": np.asarray(values)}


def test_instance_transforms_fields_and_empty_lifecycle(device):
    directions = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 1e-5, 0]], np.float32)
    points = np.arange(15, dtype=np.float32).reshape(-1, 3)
    with glyph_pipeline(device=device, shape="Arrow", scales="S", orientations="V", colors="T") as (stage, _, update):
        for count in (0, 5, 2, 0, 5):
            report = update(
                points[:count],
                {
                    "S": field(np.arange(count, dtype=np.float64)),
                    "V": field(directions[:count]),
                    "T": field(np.arange(count, dtype=np.float64)),
                },
            )
            assert not report.diagnostics
            ordinal = report.output_ordinal
            np.testing.assert_array_equal(read_array(stage, PATH, "positions", ordinal).reshape(-1, 3), points[:count])
            np.testing.assert_array_equal(read_array(stage, PATH, "protoIndices", ordinal), np.zeros(count, np.int32))
            np.testing.assert_array_equal(
                read_array(stage, PATH, "scales", ordinal).reshape(-1, 3),
                np.repeat(np.arange(count)[:, None], 3, axis=1),
            )
            np.testing.assert_array_equal(read_array(stage, PATH, "primvars:colors", ordinal), np.arange(count))
            quats = read_array(stage, PATH, "orientations", ordinal).reshape(-1, 4).astype(np.float32)
            assert quats.shape == (count, 4)
            np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1, atol=4e-4)
            # Independent quaternion-vector rotation oracle, including -X and zero.
            xyz, w = quats[:, :3], quats[:, 3:]
            x = np.tile([1, 0, 0], (count, 1))
            rotated = x + 2 * np.cross(xyz, np.cross(xyz, x) + w * x)
            expected = directions[:count].copy()
            if count > 3:
                expected[3] = [1, 0, 0]
            norms = np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(rotated, expected / norms, atol=8e-4)


def test_quaternions_vector_scales_mapping_and_removed_roles(device):
    def configure(prim):
        mapping = author.schemas.FieldMappingAPI.apply(prim, "scales")
        mapping.domain = (0, 2)
        mapping.range = (0.1, 0.5)

    points = np.zeros((2, 3), np.float32)
    with glyph_pipeline(
        device=device,
        scales="S",
        orientations="Q",
        orientations_mode="quaternion",
        colors="T",
        scale=0.25,
        configure=configure,
    ) as (stage, _prim, update):
        quats = np.asarray([[0, 0, 0, 1], [0, 0, 1, 0]], np.float64)
        report = update(
            points, {"S": field([[-1.0, 1.0, 3.0], [2.0, 0.0, 1.0]]), "Q": field(quats), "T": field([1.0, 2.0])}
        )
        assert not report.diagnostics
        n = report.output_ordinal
        np.testing.assert_allclose(
            read_array(stage, PATH, "scales", n).reshape(-1, 3), [[0.1, 0.3, 0.5], [0.5, 0.1, 0.3]], atol=1e-6
        )
        np.testing.assert_array_equal(read_array(stage, PATH, "orientations", n).reshape(-1, 4), quats)
        prototypes = read_array(stage, PATH, "prototypes", n).copy()
        write_schemas(stage, PATH, ["CaeVizOperatorAPI", "CaeVizGlyphsAPI", "CaeVizDatasetSelectionAPI:source"], n + 1)
        stage.advance_write_floor(n + 1).wait()
        report = update(points)
        assert not report.diagnostics
        n = report.output_ordinal
        np.testing.assert_allclose(read_array(stage, PATH, "scales", n), 0.25)
        assert read_array(stage, PATH, "orientations", n).size == 0
        assert read_array(stage, PATH, "primvars:colors", n) is None
        np.testing.assert_array_equal(read_array(stage, PATH, "prototypes", n), prototypes)
        write_scalar(stage, PATH, "cae:viz:operator:enabled", False, n + 1)
        stage.advance_write_floor(n + 1).wait()
        report = update(points)
        assert read_array(stage, PATH, "positions", report.output_ordinal).size == 0
        np.testing.assert_array_equal(read_array(stage, PATH, "prototypes", report.output_ordinal), prototypes)


def test_cells_decimation_and_element_fields(device):
    def configure(prim):
        author.schemas.GlyphsAPI.get(prim).use_cell_points = True

    with glyph_pipeline(device=device, max_count=2, colors="T", configure=configure) as (stage, _, update):
        points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]], np.float32)
        report = update(
            points,
            {"T": field([12.0], "element")},
            cell_types=np.array([10], np.uint8),
            connectivity_offsets=np.array([0, 4], np.int32),
            connectivity=np.array([0, 1, 2, 3], np.int32),
        )
        assert not report.diagnostics
        np.testing.assert_array_equal(
            read_array(stage, PATH, "positions", report.output_ordinal).reshape(-1, 3), points[[0, 2]]
        )
        np.testing.assert_allclose(read_array(stage, PATH, "primvars:colors", report.output_ordinal), [12, 12])


@pytest.mark.parametrize(
    "role,values,mode",
    [
        ("scales", [[1.0, 2.0]], "eulerAngles"),
        ("orientations", [[1.0, 2.0]], "eulerAngles"),
        ("orientations", [[1.0, 2.0, 3.0]], "quaternion"),
    ],
)
def test_invalid_component_counts_clear_previous_product(role, values, mode):
    with glyph_pipeline(**{role: "F"}, orientations_mode=mode) as (stage, _, update):
        good = [[0, 0, 0, 1]] if mode == "quaternion" else [[1, 0, 0]]
        first = update([[0, 0, 0]], {"F": field(good)})
        assert not first.diagnostics
        failed = update([[0, 0, 0]], {"F": field(values)})
        assert failed.operators[0].status == "failed"
        assert failed.diagnostics[0].code == "INVALID_RECIPE"
        assert read_array(stage, PATH, "positions", failed.output_ordinal).size == 0


@pytest.mark.parametrize("shape", ["Sphere", "Cone", "Arrow", "Custom"])
def test_authoring_prototypes_and_safe_initial_arrays(shape):
    usd = Usd.Stage.CreateInMemory()
    source = usd.DefinePrim("/Data", "OmniSciDataset")
    template = UsdGeom.Cube.Define(usd, "/Template").GetPrim()
    prim = author.create_glyphs(
        usd, PATH, source=source, shape=shape, prototype=template if shape == "Custom" else None
    )
    assert author.validate_recipe(prim) == ()
    instancer = UsdGeom.PointInstancer(prim)
    assert list(instancer.GetPositionsAttr().Get()) == []
    assert list(instancer.GetProtoIndicesAttr().Get()) == []
    assert list(instancer.GetOrientationsAttr().Get()) == []
    assert list(instancer.GetScalesAttr().Get()) == []
    root = usd.GetPrimAtPath(instancer.GetPrototypesRel().GetTargets()[0])
    assert root
    children = {p.GetName(): p for p in root.GetChildren()}
    if shape == "Arrow":
        assert set(children) == {"Cone", "Cylinder"}
        assert children["Cone"].GetAttribute("xformOp:translate").Get()[0] == 0.75
        assert children["Cylinder"].GetAttribute("radius").Get() == 0.15
    elif shape == "Custom":
        assert children["Template"].GetTypeName() == "Cube"
    else:
        assert children[shape].GetAttribute("radius").Get() == 0.5


def test_invalid_authoring_and_recipe_settings():
    usd = Usd.Stage.CreateInMemory()
    source = usd.DefinePrim("/Data", "OmniSciDataset")
    for kwargs in ({"shape": "unknown"}, {"shape": "Custom"}, {"prototype": source}):
        with pytest.raises(author.AuthoringError):
            author.create_glyphs(usd, PATH, source=source, **kwargs)
    with glyph_pipeline() as (stage, _, update):
        first = update([[0, 0, 0]])
        n = first.output_ordinal + 1
        write_scalar(stage, PATH, "cae:viz:glyphs:maxCount", np.int32(-1), n)
        stage.advance_write_floor(n).wait()
        report = update([[0, 0, 0]])
        assert report.diagnostics[0].code == "INVALID_RECIPE"
        write_scalar(stage, PATH, "cae:viz:glyphs:maxCount", np.int32(0), n + 2)
        write_token(stage, PATH, "cae:viz:glyphs:orientationsMode", "bad", n + 2)
        stage.advance_write_floor(n + 2).wait()
        assert update([[0, 0, 0]]).diagnostics[0].code == "INVALID_RECIPE"


@pytest.mark.parametrize(
    "domain,expected", [((0, 2), [0.1, 0.3, 0.5]), ((0, -1), [-1.0, 1.0, 3.0]), ((1, 1), [0.1, 0.1, 0.1])]
)
def test_scalar_column_scale_mapping(device, domain, expected):
    def configure(prim):
        mapping = author.schemas.FieldMappingAPI.apply(prim, "scales")
        mapping.domain = domain
        mapping.range = (0.1, 0.5)

    with glyph_pipeline(device=device, scales="S", configure=configure) as (stage, _, update):
        report = update(np.zeros((3, 3)), {"S": field(np.array([[-1], [1], [3]], np.int32))})
        assert not report.diagnostics
        np.testing.assert_allclose(
            read_array(stage, PATH, "scales", report.output_ordinal),
            np.repeat(np.asarray(expected)[:, None], 3, axis=1),
            atol=1e-6,
        )


def test_disabled_recipe_writes_nothing_after_the_first_clear(monkeypatch):
    import ovcae._session as session_module

    calls = []
    original = session_module.clear_instancer_product

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(session_module, "clear_instancer_product", counted)
    with glyph_pipeline() as (stage, _, update):
        first = update([[0, 0, 0]])
        n = first.output_ordinal + 1
        write_scalar(stage, PATH, "cae:viz:operator:enabled", False, n)
        stage.advance_write_floor(n).wait()
        for _ in range(3):
            report = update([[0, 0, 0]])
            assert report.operators[0].status == "skipped"
            assert read_array(stage, PATH, "positions", report.output_ordinal).size == 0
        assert len(calls) == 1
        write_scalar(stage, PATH, "cae:viz:operator:enabled", True, report.output_ordinal + 1)
        stage.advance_write_floor(report.output_ordinal + 1).wait()
        report = update([[0, 0, 0]])
        assert report.operators[0].status == "succeeded", report.diagnostics
        assert read_array(stage, PATH, "positions", report.output_ordinal).size == 3


@pytest.mark.parametrize("stronger_layer", [False, True])
def test_reauthoring_rejects_existing_glyphs_without_edits(stronger_layer):
    usd = Usd.Stage.CreateInMemory()
    source = usd.DefinePrim("/Data", "OmniSciDataset")
    author.create_glyphs(usd, PATH, source=source, shape="Arrow")
    edit_target = usd.GetRootLayer()
    if stronger_layer:
        edit_target = Sdf.Layer.CreateAnonymous("glyph-edits.usda")
        edit_target.subLayerPaths.append(usd.GetRootLayer().identifier)
        usd = Usd.Stage.Open(edit_target)
    layers = usd.GetLayerStack()
    before = [layer.ExportToString() for layer in layers]
    with pytest.raises(author.AuthoringError, match="already exists"):
        author.create_glyphs(usd, PATH, source="/Data", shape="Sphere", edit_target=edit_target)
    assert [layer.ExportToString() for layer in layers] == before
