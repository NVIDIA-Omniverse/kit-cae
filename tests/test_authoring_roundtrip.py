# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Authored recipes must be discovered and executed by ovcae unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest

import ovcae

from ._ovstage_helpers import read_array

readers = pytest.importorskip("ovcae.readers")
pytest.importorskip("warp")
pytest.importorskip("warp_simdata")
ovstage = pytest.importorskip("ovstage")

from pxr import Usd  # noqa: E402

import ovcae.authoring as author  # noqa: E402

CGNS = Path(__file__).parent / "data" / "StaticMixer.cgns"
VOLUME = "B1_P3"
BOUNDARY = "StaticMixer_Default"


def _author_scene(path: Path) -> dict[str, str]:
    stage = Usd.Stage.CreateNew(str(path))
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    holder = readers.import_dataset(stage, "/World/Assets/StaticMixer", str(CGNS.resolve()))
    datasets = {prim.GetName(): prim for prim in readers.dataset_prims(stage, holder)}
    volume, boundary = datasets[VOLUME], datasets[BOUNDARY]
    seeds = author.create_seed_sphere(
        stage, "/World/Controls/Seeds", bounds=((-2.5, -3.5, 0.0), (2.5, 2.5, 2.5)), count=8
    )
    recipes = {
        "faces": author.create_faces(stage, "/World/Recipes/Faces", source=boundary, colors="Temperature"),
        "points": author.create_points(
            stage, "/World/Recipes/Points", source=volume, colors="Temperature", max_count=500
        ),
        "glyphs": author.create_glyphs(
            stage,
            "/World/Recipes/Glyphs",
            source=volume,
            shape="Arrow",
            colors="Temperature",
            scales="Pressure",
            orientations=("VelocityX", "VelocityY", "VelocityZ"),
            max_count=32,
        ),
        "iso": author.create_iso_surface(
            stage, "/World/Recipes/Iso", source=volume, contour="Temperature", iso_value=300.0
        ),
        "slice": author.create_planar_slice(
            stage, "/World/Recipes/Slice", source=volume, colors="Temperature", mode="xyz"
        ),
        "streamlines": author.create_streamlines(
            stage,
            "/World/Recipes/Streamlines",
            source=volume,
            seeds=seeds,
            velocities=("VelocityX", "VelocityY", "VelocityZ"),
            colors="Temperature",
            max_steps=20,
        ),
    }
    stage.Save()
    return {name: str(prim.GetPath()) for name, prim in recipes.items()}


def test_authored_scene_executes_through_session(tmp_path: Path) -> None:
    scene = tmp_path / "authored.usda"
    recipes = _author_scene(scene)

    ovcae.register_schema_paths()
    stage = ovstage.Stage("ovcae.tests.authoring.roundtrip")
    source = readers.UsdDataSource.open(scene)
    try:
        ovstage.population.open_usd_with_desc(
            stage,
            str(scene),
            ordinal=1,
            time_code=0.0,
            descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
        )
        stage.advance_write_floor(1).wait()
        with ovcae.Session(stage, device="cpu") as session, source.at_time(0.0) as frame:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame).wait()
        assert report.committable, report.diagnostics
        statuses = {operator.operator_path: operator.status for operator in report.operators}
        assert statuses == dict.fromkeys(recipes.values(), "succeeded"), report.diagnostics
        stage.advance_write_floor(2).wait()

        for name in ("faces", "iso", "slice"):
            points = read_array(stage, recipes[name], "points", 2)
            assert points is not None and len(points.reshape(-1, 3)) > 0, name
            assert read_array(stage, recipes[name], "primvars:colors", 2) is not None, name
        assert len(read_array(stage, recipes["points"], "points", 2).reshape(-1, 3)) > 0
        glyph_count = len(read_array(stage, recipes["glyphs"], "positions", 2).reshape(-1, 3))
        assert 0 < glyph_count <= 32
        for attribute in ("scales", "orientations", "primvars:colors"):
            assert len(read_array(stage, recipes["glyphs"], attribute, 2)) == glyph_count
        counts = read_array(stage, recipes["streamlines"], "curveVertexCounts", 2)
        assert counts is not None and len(counts) > 0
    finally:
        source.close()
        stage.destroy()
