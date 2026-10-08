# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Compare real Glyphs output with independently transformed ordinary geometry."""

import numpy as np
import pytest


@pytest.mark.parametrize("materials", [False, True])
def test_glyphs_render_like_reference_geometry(tmp_path, materials):
    ovrtx = pytest.importorskip("ovrtx")
    if not hasattr(ovrtx, "Renderer"):
        pytest.skip("OVRTX renderer is not installed")
    ovrtx.register_schema_paths()
    import warp as wp
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    import ovcae
    import ovcae.authoring as author
    import ovcae.materials
    from ovcae._stage_io import write_floor

    from ._ovstage_helpers import write_scalar
    from .test_glyphs import field, glyph_pipeline

    if not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")

    def configure(prim):
        stage = prim.GetStage()
        camera = UsdGeom.Camera.Define(stage, "/Camera")
        camera.CreateProjectionAttr("orthographic")
        camera.CreateHorizontalApertureAttr(80.0)
        camera.CreateVerticalApertureAttr(60.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100))
        camera.AddTranslateOp().Set((0, 0, 12))
        light = stage.DefinePrim("/Light", "DomeLight")
        light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
        product = stage.DefinePrim("/Render", "RenderProduct")
        product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(400, 300))
        product.CreateRelationship("camera").SetTargets(["/Camera"])
        product.CreateRelationship("orderedVars").SetTargets(["/Render/LdrColor"])
        var = stage.DefinePrim("/Render/LdrColor", "RenderVar")
        var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
        var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
        if materials:
            author.schemas.FieldSelectionAPI.apply(prim, "opacity").field_names = ("A",)
            shader = stage.GetPrimAtPath("/Glyphs/Materials/ScalarColor/Shader")
            shader.GetAttribute("inputs:enable_opacity").Set(True)
            shader.GetAttribute("inputs:opacity_domain").Set(Gf.Vec2f(0, 1))
        root = stage.GetPrimAtPath("/Glyphs/Prototypes/Xform")
        for child in root.GetChildren():
            UsdGeom.Gprim(child).CreateDisplayColorAttr([(0.05, 0.45, 0.9)])
        # Reference transforms are authored independently, in degrees, without
        # using the operator's direction-to-quaternion helper.
        for i, (x, angle) in enumerate(zip((-2, 0, 2), (0, 90, 180), strict=True)):
            reference = UsdGeom.Xform.Define(stage, f"/Reference{i}")
            reference.GetPrim().GetReferences().AddInternalReference(root.GetPath())
            reference.AddTranslateOp().Set((x, -1.3, 0))
            reference.AddRotateZOp().Set(float(angle))
            reference.AddScaleOp().Set((0.7, 0.7, 0.7))
            if materials:
                material = UsdShade.Material(stage.GetPrimAtPath("/Glyphs/Materials/ScalarColor"))
                UsdShade.MaterialBindingAPI.Apply(reference.GetPrim()).Bind(material)
                pv = UsdGeom.PrimvarsAPI(reference)
                pv.CreatePrimvar("colors", Sdf.ValueTypeNames.FloatArray, "constant").Set([[0.1, 0.5, 0.9][i]])
                pv.CreatePrimvar("opacity", Sdf.ValueTypeNames.FloatArray, "constant").Set([0.0 if i == 2 else 1.0])

    ovcae.materials.register_material_library()
    ovcae.setup()
    renderer = ovrtx.Renderer(
        ovrtx.RendererConfig(active_cuda_gpus="0", log_file_path=str(tmp_path / "renderer.log"), log_level="warning")
    )
    try:
        with glyph_pipeline(
            device="cuda:0",
            shape="Arrow",
            orientations="V",
            scales="S",
            configure=configure,
            materials=materials,
            colors="T" if materials else None,
            color_range=(0, 1) if materials else None,
        ) as (
            stage,
            _,
            update,
        ):
            renderer.attach_ovstage(stage)
            try:
                for count, enabled in ((0, True), (3, True), (0, True), (3, True), (3, False), (3, True)):
                    ordinal = write_floor(stage) + 1
                    write_scalar(stage, "/Glyphs", "cae:viz:operator:enabled", enabled, ordinal)
                    stage.advance_write_floor(ordinal).wait()
                    report = update(
                        np.asarray([[-2, 1.3, 0], [0, 1.3, 0], [2, 1.3, 0]])[:count],
                        {
                            "S": field(np.full(count, 0.7, np.float32)),
                            "T": field(np.asarray([0.1, 0.5, 0.9], np.float64)[:count]),
                            "A": field(np.asarray([1.0, 1.0, 0.0], np.float64)[:count]),
                            "V": field(np.asarray([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0]])[:count]),
                        },
                    )
                    assert not report.diagnostics
                    assert len(report.operators) == 1
                    assert report.operators[0].status == ("succeeded" if enabled else "skipped")
                    for _ in range(12):
                        products = renderer.step(
                            render_products={"/Render"}, delta_time=1 / 60, ordinal=report.output_ordinal
                        )
                        frame = products["/Render"].frames[0]
                        mapped = frame.render_vars["/Render/LdrColor"].map(device=ovrtx.Device.CPU)
                        pixels = np.from_dlpack(mapped).copy()
                        mapped.unmap()
                        del mapped, frame, products
                    np.save(tmp_path / f"glyphs-{report.output_ordinal}.npy", pixels)
                    for x in (-2, 0, 2):
                        cx = round(200 + x * 50)
                        top = pixels[55:115, cx - 43 : cx + 43, :3].astype(float)
                        bottom = pixels[185:245, cx - 43 : cx + 43, :3].astype(float)

                        def mask(patch):
                            background = np.median(patch[:3].reshape(-1, 3), axis=0)
                            return np.max(np.abs(patch - background), axis=-1) > 15

                        actual, expected = mask(top), mask(bottom)
                        if materials and x == 2:
                            assert expected.sum() < 5
                            assert actual.sum() < 5
                            continue
                        assert expected.sum() > 50
                        if count and enabled:
                            assert (actual & expected).sum() / (actual | expected).sum() > 0.9
                            assert np.abs(top - bottom)[actual | expected].mean() < 12
                        else:
                            assert actual.sum() < 5
            finally:
                renderer.detach_ovstage()
    finally:
        renderer.destroy()
