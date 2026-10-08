# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Unit tests for the ovcae.authoring sister package (usd-core only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom

import ovcae.authoring as author

MIXER = "/World/Assets/StaticMixer"
CGNS = Path(__file__).parent / "data" / "StaticMixer.cgns"


@pytest.fixture
def stage():
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    return stage


@pytest.fixture
def mixer(stage):
    readers = pytest.importorskip("ovcae.readers")
    return readers.import_dataset(stage, MIXER, "./StaticMixer.cgns")


def test_import_never_loads_runtime_dependencies() -> None:
    # The parent package always loads, but it is standard-library only; the
    # runtime's dependencies must not follow.
    code = (
        "import sys; import ovcae.authoring; "
        "assert not {'ovstage', 'warp', 'warp_simdata', 'numpy'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_default_recipe_needs_only_this_package() -> None:
    code = (
        "import ovcae.authoring as author\n"
        "from pxr import Usd\n"
        "stage = Usd.Stage.CreateInMemory()\n"
        "holder = stage.DefinePrim('/Data'); holder.GetPayloads().AddPayload('./sim.cgns')\n"
        "faces = author.create_faces(stage, '/Faces', source=holder)\n"
        "assert stage.GetPrimAtPath('/Faces/Materials'), 'default materials were not authored'\n"
        "author.create_points(stage, '/Points', source=stage.DefinePrim('/Custom/Descriptor', 'Scope'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_without_usd_core_raises_clear_error() -> None:
    code = (
        "import sys; sys.modules['pxr'] = None\n"
        "try:\n    import ovcae.authoring\nexcept ImportError as exc:\n"
        "    assert 'usd-core' in str(exc), exc\nelse:\n    raise SystemExit('expected ImportError')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_dataset_authors_payload_holder(stage) -> None:
    readers = pytest.importorskip("ovcae.readers")
    pytest.importorskip("cae_openusd_plugins")
    prim = readers.import_dataset(stage, MIXER, "./a.cgns", zoneName="Zone1", cacheMode="static")
    assert [payload.assetPath for payload in prim.GetMetadata("payload").prependedItems] == ["./a.cgns"]
    assert "OmniSciFileFormatArgsCgnsAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("omni:cae:format:cgns:zoneName").Get() == "Zone1"
    assert prim.GetAttribute("omni:cae:format:cacheMode").Get() == "static"
    with pytest.raises(ValueError, match="no format argument"):
        readers.import_dataset(stage, "/World/Assets/Bad", "./b.cgns", nonsense=1)
    with pytest.raises(ValueError, match="unknown format"):
        readers.import_dataset(stage, "/World/Assets/Bad", "./b.cgns", format="nope")
    with pytest.raises(ValueError, match="non-empty"):
        readers.import_dataset(stage, "/World/Assets/Bad", "")
    with pytest.raises(ValueError, match="must be one of all, static, none"):
        readers.import_dataset(stage, "/World/Assets/Bad", "./b.cgns", cacheMode="nonsense")
    with pytest.raises(ValueError, match="expects int"):
        readers.import_dataset(stage, "/World/Assets/Bad", "./b.cgns", int_size="four")
    plain = readers.import_dataset(stage, "/World/Assets/Plain", "./mesh.unknownext")
    assert plain.HasAuthoredPayloads() and not plain.GetAppliedSchemas()


def test_dataset_prims_lists_composed_datasets(stage) -> None:
    readers = pytest.importorskip("ovcae.readers")
    pytest.importorskip("cae_openusd_plugins")
    holder = readers.import_dataset(stage, MIXER, str(CGNS))
    names = {prim.GetName() for prim in readers.dataset_prims(stage, holder)}
    assert {"B1_P3", "StaticMixer_Default"} <= names
    faces = author.create_faces(stage, "/World/Recipes/Surface", source=f"{MIXER}/Base/StaticMixer/StaticMixer_Default")
    target = faces.GetRelationship("cae:viz:dataset_selection:source:target").GetTargets()[0]
    assert stage.GetPrimAtPath(target).GetTypeName() == "OmniSciDataset"


def test_faces_matches_hand_authored_recipe(stage, mixer) -> None:
    prim = author.create_faces(
        stage,
        "/World/Recipes/Surface",
        source=mixer,
        identifier="/Base/StaticMixer/StaticMixer_Default",
        colors="Temperature",
        external_only=True,
    )
    assert prim.GetTypeName() == "Mesh"
    assert prim.GetAppliedSchemas() == [
        "CaeVizOperatorAPI",
        "CaeVizFacesAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizFieldSelectionAPI:colors",
        "CaeVizRescaleRangeAPI:colors",
        "CaeVizFieldSelectionAPI:opacity",
        "CaeVizRescaleRangeAPI:opacity",
        "MaterialBindingAPI",
    ]
    assert prim.GetAttribute("cae:viz:operator:enabled").Get() is True
    assert prim.GetAttribute("cae:viz:faces:externalOnly").Get() is True
    assert prim.GetRelationship("cae:viz:dataset_selection:source:target").GetTargets() == [Sdf.Path(MIXER)]
    identifier = prim.GetAttribute("cae:viz:dataset_selection:source:identifier").Get()
    assert identifier == "/Base/StaticMixer/StaticMixer_Default"
    assert list(prim.GetAttribute("cae:viz:field_selection:colors:fieldNames").Get()) == ["Temperature"]
    # Placeholder geometry keeps renderers happy before the first evaluation.
    assert len(prim.GetAttribute("points").Get()) == 0
    assert prim.GetAttribute("subdivisionScheme").Get() == "none"
    assert UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor").GetInterpolation() == "constant"


def test_builders_only_author_supplied_values(stage, mixer) -> None:
    prim = author.create_faces(stage, "/World/Recipes/Surface", source=mixer, display_color=None)
    # The schema fallback is visible but nothing is authored for it.
    assert prim.GetAttribute("cae:viz:faces:externalOnly").Get() is True
    assert not prim.GetAttribute("cae:viz:faces:externalOnly").IsAuthored()
    assert "CaeVizFieldSelectionAPI:colors" not in prim.GetAppliedSchemas()
    assert not UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor").GetAttr().HasAuthoredValue()


def test_iso_surface_defaults_colors_to_contour(stage, mixer) -> None:
    prim = author.create_iso_surface(
        stage, "/World/Recipes/Iso", source=mixer, contour="Temperature", iso_value=300, identifier="/Base/X"
    )
    assert prim.GetAttribute("cae:viz:isoSurface:isoValue").Get() == pytest.approx(300.0)
    assert list(prim.GetAttribute("cae:viz:field_selection:contour:fieldNames").Get()) == ["Temperature"]
    assert list(prim.GetAttribute("cae:viz:field_selection:colors:fieldNames").Get()) == ["Temperature"]


def test_planar_slice_authors_transform_ops(stage, mixer) -> None:
    prim = author.create_planar_slice(
        stage, "/World/Recipes/Slice", source=mixer, colors="Temperature", origin=(0, 0, 1), mode="xz"
    )
    assert prim.GetAttribute("cae:viz:planarSlice:mode").Get() == "xz"
    assert list(UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get()) == [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
    ]
    assert tuple(prim.GetAttribute("xformOp:translate").Get()) == (0.0, 0.0, 1.0)


def test_points_and_streamlines_roles(stage, mixer) -> None:
    points = author.create_points(
        stage,
        "/World/Recipes/Points",
        source=mixer,
        colors="Temperature",
        widths="Pressure",
        max_count=4000,
        width=0.05,
    )
    assert points.GetTypeName() == "Points"
    assert points.GetAttribute("cae:viz:points:maxCount").Get() == 4000
    assert "CaeVizFieldSelectionAPI:widths" in points.GetAppliedSchemas()

    seeds = author.create_seed_points(stage, "/World/Controls/Seeds", [(0, 0, 0), (1, 1, 1)])
    assert seeds.GetAttribute("visibility").Get() == "invisible"
    curves = author.create_streamlines(
        stage,
        "/World/Recipes/Streamlines",
        source=MIXER,
        seeds="/World/Controls/Seeds",
        velocities=("VelocityX", "VelocityY", "VelocityZ"),
        direction="forward",
        max_steps=20,
        width=0.01,
    )
    assert curves.GetTypeName() == "BasisCurves"
    assert curves.GetRelationship("cae:viz:dataset_selection:seeds:target").GetTargets() == [seeds.GetPath()]
    assert list(curves.GetAttribute("cae:viz:field_selection:velocities:fieldNames").Get()) == [
        "VelocityX",
        "VelocityY",
        "VelocityZ",
    ]
    assert curves.GetAttribute("cae:viz:field_selection:velocities:mode").Get() == "unchanged"
    assert curves.GetAttribute("cae:viz:streamlines:maxSteps").Get() == 20
    assert curves.GetAttribute("widths").Get()[0] == pytest.approx(0.01)
    assert curves.GetAttribute("curveVertexCounts").Get()[0] == len(curves.GetAttribute("points").Get())


def test_invalid_inputs_raise_authoring_error(stage, mixer) -> None:
    seeds = author.create_seed_points(stage, "/World/Controls/Seeds", [(0, 0, 0)])
    with pytest.raises(author.AuthoringError, match="does not exist"):
        author.create_faces(stage, "/World/Recipes/X", source="/World/Missing")
    with pytest.raises(author.AuthoringError, match="PointBased prim"):
        author.create_streamlines(stage, "/World/Recipes/X", source=mixer, seeds=mixer, velocities="V")
    with pytest.raises(author.AuthoringError, match="already exists"):
        author.create_faces(stage, seeds.GetPath(), source=mixer)
    with pytest.raises(author.AuthoringError, match="absolute prim path"):
        author.create_faces(stage, "Recipes/Relative", source=mixer)
    with pytest.raises(ValueError, match="non-empty"):
        author.create_faces(stage, "/World/Recipes/X", source=mixer, colors=[])
    with pytest.raises(author.AuthoringError):
        author.create_seed_sphere(stage, "/World/Controls/Sphere", count=8)


def test_edit_target_routes_opinions_to_layer(stage, mixer) -> None:
    layer = Sdf.Layer.CreateAnonymous("edits.usda")
    stage.GetRootLayer().subLayerPaths.append(layer.identifier)
    author.create_faces(stage, "/World/Recipes/Session", source=mixer, edit_target=layer)
    author.create_seed_points(stage, "/World/Controls/S", [(0, 0, 0)], edit_target=Usd.EditTarget(layer))
    assert layer.GetPrimAtPath("/World/Recipes/Session")
    assert layer.GetPrimAtPath("/World/Controls/S")
    assert not stage.GetRootLayer().GetPrimAtPath("/World/Recipes/Session")
    assert stage.GetPrimAtPath("/World/Recipes/Session").HasAPI("CaeVizFacesAPI")


def test_controls_size_from_bounds(stage) -> None:
    bounds = author.Bounds((-2.0, -3.0, 0.0), (2.0, 2.0, 2.0))
    assert bounds.center == (0.0, -0.5, 1.0)
    sphere = author.create_seed_sphere(stage, "/World/Controls/Seeds", bounds=bounds, count=12)
    points = sphere.GetAttribute("points").Get()
    assert len(points) == 12
    radius = 0.05 * bounds.diagonal / 2.0
    for point in points:
        distance = sum((float(a) - b) ** 2 for a, b in zip(point, bounds.center, strict=True)) ** 0.5
        assert distance == pytest.approx(radius, rel=1e-5)

    box = author.create_bounding_box(stage, "/World/Controls/Box", ((-2, -3, 0), (2, 2, 2)))
    assert box.GetTypeName() == "BasisCurves"
    assert list(box.GetAttribute("curveVertexCounts").Get()) == [2] * 12
    assert len(box.GetAttribute("points").Get()) == 24
    with pytest.raises(author.AuthoringError, match="bounds"):
        author.create_bounding_box(stage, "/World/Controls/Bad", ((1, 1, 1), (0, 0, 0)))


def _shader(prim, name):
    return prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath(f"Materials/{name}/Shader"))


def test_faces_material_matches_kit_cae(stage, mixer) -> None:
    prim = author.create_faces(stage, "/World/Recipes/Surface", source=mixer, colors="Temperature")
    material = stage.GetPrimAtPath("/World/Recipes/Surface/Materials/ScalarColor")
    assert material.GetTypeName() == "Material"
    assert material.GetAttribute("omni:rtx:enableCutoutOpacity").Get() is True
    for output in ("surface", "displacement", "volume"):
        connection = material.GetAttribute(f"outputs:mdl:{output}").GetConnections()
        assert [str(path) for path in connection] == ["/World/Recipes/Surface/Materials/ScalarColor/Shader.outputs:out"]
    shader = _shader(prim, "ScalarColor")
    assert shader.GetAttribute("info:implementationSource").Get() == "sourceAsset"
    assert shader.GetAttribute("info:mdl:sourceAsset").Get().path == "cae/mdl/basic.mdl"
    assert shader.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() == "ScalarColor"
    assert shader.GetAttribute("outputs:out").GetMetadata("renderType") == "material"
    assert shader.GetAttribute("inputs:lut").Get().path == "cae/colormaps/gist_rainbow.png"
    assert shader.GetAttribute("inputs:opacity_lut").Get().path == "cae/colormaps/gist_gray.png"
    assert tuple(shader.GetAttribute("inputs:domain").Get()) == (0.0, -1.0)
    assert shader.GetAttribute("inputs:enable_coloring").Get() is False
    assert shader.GetAttribute("inputs:enable_opacity").Get() is False
    assert shader.GetAttribute("inputs:opacity").Get() == pytest.approx(1.0)
    assert tuple(shader.GetAttribute("inputs:opacity_domain").Get()) == (0.0, -1.0)

    binding = prim.GetRelationship("material:binding")
    assert binding.GetTargets() == [material.GetPath()]
    assert binding.GetMetadata("bindMaterialAs") == "weakerThanDescendants"
    shader_path = shader.GetPath()
    assert prim.GetRelationship("cae:viz:rescale_range:colors:includes").GetTargets() == [
        shader_path.AppendProperty("inputs:domain")
    ]
    assert prim.GetRelationship("cae:viz:rescale_range:colors:enableIncludes").GetTargets() == [
        shader_path.AppendProperty("inputs:enable_coloring")
    ]
    assert prim.GetRelationship("cae:viz:rescale_range:opacity:includes").GetTargets() == [
        shader_path.AppendProperty("inputs:opacity_domain")
    ]
    assert prim.GetRelationship("cae:viz:rescale_range:opacity:enableIncludes").GetTargets() == [
        shader_path.AppendProperty("inputs:enable_opacity")
    ]
    # The opacity role is selectable but nothing is selected, exactly as Kit-CAE authors it.
    assert not prim.GetAttribute("cae:viz:field_selection:opacity:fieldNames").IsAuthored()


def test_streamlines_and_slice_material_variants(stage, mixer) -> None:
    seeds = author.create_seed_points(stage, "/World/Controls/Seeds", [(0, 0, 0)])
    curves = author.create_streamlines(
        stage, "/World/Recipes/Streamlines", source=mixer, seeds=seeds, velocities=("Vx", "Vy", "Vz")
    )
    scalar, animated = _shader(curves, "ScalarColor"), _shader(curves, "AnimatedStreaks")
    assert animated.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() == "AnimatedStreaks"
    assert curves.GetRelationship("material:binding").GetTargets() == [scalar.GetParent().GetPath()]
    assert curves.GetRelationship("cae:viz:rescale_range:colors:includes").GetTargets() == [
        scalar.GetPath().AppendProperty("inputs:domain"),
        animated.GetPath().AppendProperty("inputs:domain"),
    ]
    # Only AnimatedStreaks is toggled; ScalarColor keeps the MDL default so it colors once ranged.
    assert curves.GetRelationship("cae:viz:rescale_range:colors:enableIncludes").GetTargets() == [
        animated.GetPath().AppendProperty("inputs:enable_coloring")
    ]
    assert not scalar.GetAttribute("inputs:enable_coloring")
    assert animated.GetAttribute("inputs:enable_coloring").Get() is False
    assert tuple(curves.GetAttribute("cae:viz:field_mapping:widths:domain").Get()) == (0.0, -1.0)
    assert curves.GetRelationship("cae:viz:rescale_range:widths:includes").GetTargets() == [
        curves.GetPath().AppendProperty("cae:viz:field_mapping:widths:domain")
    ]

    plane = author.create_planar_slice(stage, "/World/Recipes/Slice", source=mixer, colors="Temperature")
    unlit = _shader(plane, "UnlitScalarColor")
    assert unlit.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() == "UnlitScalarColor"
    assert plane.GetRelationship("material:binding").GetTargets() == [unlit.GetParent().GetPath()]
    assert plane.GetRelationship("cae:viz:rescale_range:colors:enableIncludes").GetTargets() == []
    assert not unlit.GetAttribute("inputs:enable_coloring")

    points = author.create_points(stage, "/World/Recipes/Points", source=mixer, width=0.05)
    assert tuple(points.GetAttribute("cae:viz:field_mapping:widths:range").Get()) == pytest.approx((0.05, 0.1))


def test_material_options_and_opt_out(stage, mixer) -> None:
    prim = author.create_faces(
        stage, "/World/Recipes/A", source=mixer, colors="Temperature", colormap="cividis", color_range=(290, 310)
    )
    shader = _shader(prim, "ScalarColor")
    assert shader.GetAttribute("inputs:lut").Get().path == "cae/colormaps/cividis.png"
    assert tuple(shader.GetAttribute("inputs:domain").Get()) == (290.0, 310.0)
    assert shader.GetAttribute("inputs:enable_coloring").Get() is True

    custom = author.create_faces(stage, "/World/Recipes/B", source=mixer, colormap="./luts/mine.png")
    assert _shader(custom, "ScalarColor").GetAttribute("inputs:lut").Get().path == "./luts/mine.png"

    with pytest.raises(author.AuthoringError, match="unknown bundled colormap"):
        author.create_faces(stage, "/World/Recipes/C", source=mixer, colormap="viridis")
    with pytest.raises(author.AuthoringError, match="min < max"):
        author.create_faces(stage, "/World/Recipes/D", source=mixer, color_range=(1, 1))

    bare = author.create_faces(stage, "/World/Recipes/E", source=mixer, materials=False)
    assert not stage.GetPrimAtPath("/World/Recipes/E/Materials")
    assert "MaterialBindingAPI" not in bare.GetAppliedSchemas()
    assert bare.GetAppliedSchemas() == ["CaeVizOperatorAPI", "CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"]


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (author.create_faces, {"source": "/Data"}),
        (author.create_points, {"source": "/Data"}),
        (author.create_iso_surface, {"source": "/Data", "contour": "T", "iso_value": 1}),
        (author.create_planar_slice, {"source": "/Data", "colors": "T"}),
        (author.create_streamlines, {"source": "/Data", "seeds": "/Seeds", "velocities": "V"}),
        (author.create_glyphs, {"source": "/Data"}),
        (author.create_seed_points, {"points": [(0, 0, 0)]}),
        (author.create_seed_sphere, {"center": (0, 0, 0), "radius": 1, "count": 4}),
        (author.create_bounding_box, {"bounds": ((0, 0, 0), (1, 1, 1))}),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_builders_reject_repeated_creation_without_edits(stage, builder, kwargs):
    stage.DefinePrim("/Data", "OmniSciDataset")
    author.create_seed_points(stage, "/Seeds", [(0, 0, 0)])
    builder(stage, "/Output", **kwargs)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(author.AuthoringError, match="already exists"):
        builder(stage, "/Output", **kwargs)
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("kind", ["untyped", "over", "inactive"])
def test_builder_rejects_existing_nondefined_or_inactive_prim(stage, kind):
    source = stage.DefinePrim("/Data", "OmniSciDataset")
    if kind == "over":
        stage.OverridePrim("/Output")
    else:
        prim = stage.DefinePrim("/Output")
        if kind == "inactive":
            prim.SetActive(False)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(author.AuthoringError, match="already exists"):
        author.create_faces(stage, "/Output", source=source)
    assert stage.GetRootLayer().ExportToString() == before
