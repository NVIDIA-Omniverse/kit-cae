# %%
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
# tutorial-requires: gpu

# %% [markdown]
# # 3. Bring your own data
#
# Tutorials 1 and 2 opened scenes somebody else had authored. This one starts
# from a bare simulation result, a CGNS file, and ends with a scene like
# theirs, written from Python: the dataset composed into the scene, recipes
# for surfaces, an iso-surface, and streamlines, and a camera and render
# product to look at them. Along the way it does the two things every author
# has to do before writing a recipe: find out what the file contains, and
# look at the numbers so the color ranges and iso value make sense.
#
# It closes with two things the earlier tutorials skipped. The saved scene is
# reopened in a fresh pipeline and evaluated again to show that the scene and
# its referenced dataset reproduce the result, and a deliberate mistake shows
# what OVCAE tells you when a recipe cannot be evaluated.
#
# This tutorial requires an RTX-capable NVIDIA GPU and uses the device
# selected in tutorial 0 for evaluation and rendering.

# %% [markdown]
# ## Initialize
#
# The same order as before: OVRTX's schema paths, then `ovcae.setup()`, and
# only then any usd-core stage. This tutorial creates its own stage a few
# cells down, so the order matters here more than it did in tutorial 1.

# %%
import ovrtx

ovrtx.register_schema_paths()

import numpy as np
import warp as wp
from support import OUTPUT_DIR, asset, cuda_gpu_index, tutorial_device

import ovcae

ovcae.setup()
wp.config.log_level = wp.LOG_WARNING

device = tutorial_device()
print("Compute device:", device)

# %% [markdown]
# ## Compose the file into a scene
#
# A CAE dataset enters a scene as a **payload** on an otherwise empty prim,
# the holder. When usd-core composes the stage, the CAE OpenUSD plug-in for
# the file format reads the file's structure and produces one `OmniSciDataset`
# prim per zone or boundary beneath the holder, with the arrays and fields
# described by `OmniSci*API` schemas. Array data is read on demand when
# requested.
#
# `ovcae.readers.import_dataset` authors the holder and its format-arguments
# schema, and `dataset_prims` lists what composed beneath it. The file name is
# authored exactly as given; a path relative to the layer keeps the scene
# relocatable. Appendix A1 does this without OVCAE, to show what the plug-in
# itself provides.

# %%
import os

from pxr import Usd, UsdGeom

import ovcae.readers as readers

scene_path = OUTPUT_DIR / "my_aero_scene.usda"
scene_path.parent.mkdir(parents=True, exist_ok=True)
scene_path.unlink(missing_ok=True)  # start over each time the tutorial runs

stage = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # the solver worked in centimetres
world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)

cgns_path = asset("auto_aero_solver_result.cgns")
holder = readers.import_dataset(stage, "/World/Data", os.path.relpath(cgns_path, scene_path.parent))
datasets = {prim.GetName(): prim for prim in readers.dataset_prims(stage, holder)}
for name, prim in datasets.items():
    print(f"{prim.GetPath()!s:<48} {name}")

# %% [markdown]
# Seven datasets: the fluid volume, a polyhedral copy of it, the car body,
# and the wheel and seal boundaries. The fields live on a sibling
# `CGNSFlowSolution` prim, one `OmniSciFieldAPI` instance each, and datasets
# in the same zone share them.

# %%
zone = datasets["Fluid_Domain"].GetParent()
for prim in zone.GetChildren():
    if prim.GetTypeName() == "CGNSFlowSolution":
        fields = [
            schema.split(":", 1)[1] for schema in prim.GetAppliedSchemas() if schema.startswith("OmniSciFieldAPI:")
        ]
        print(f"{prim.GetPath()}: {', '.join(fields)}")

# %% [markdown]
# ## Look at the numbers
#
# A recipe needs an iso value and color ranges, and guessing them wastes a
# render cycle each time. Warp SimData loads a field from a dataset prim
# onto the compute device and reports its range directly. This is the one
# call the OVCAE Data Source makes on your behalf at evaluation time; here
# it answers the author's questions up front. Appendix A1 goes further with
# SimData's operators.

# %%
import warp_simdata.usd as simdata_usd

fluid = datasets["Fluid_Domain"]


def field_range(prim, *names: str) -> tuple[float, float]:
    """Load a field from a dataset prim and return its (min, max)."""
    field = simdata_usd.field_from_prim(prim, list(names), device=device, time_code=0.0)
    field = field[0] if isinstance(field, list) else field
    low, high = field.get_range()
    return float(low), float(high)


pressure_range = field_range(fluid, "Pressure")
vertical_velocity_range = field_range(fluid, "Velocity_2")
print(f"Pressure          {pressure_range[0]:9.1f} .. {pressure_range[1]:8.1f}")
print(f"Vertical velocity {vertical_velocity_range[0]:9.1f} .. {vertical_velocity_range[1]:8.1f}")

# %% [markdown]
# Both ranges are dominated by a few extreme cells, so mapping colors over
# the full range would leave most of the surface one color. The choices
# below come from where the bulk of the values sit: pressure from -800 to
# 400 for the body, an iso-surface at -200 (low pressure hugging the body),
# and vertical velocity from -15 to 15 to color it.

# %% [markdown]
# ## Author the recipes
#
# `ovcae.authoring` has one builder per operator. Each defines the recipe
# prim with the right type, applies the operator and selection schemas,
# authors placeholder geometry, and by default creates a colormap material
# under the recipe and wires it to the color selection. `source` is a
# dataset prim; `colors` names the field to color by; `color_range` fixes
# the colormap domain.
#
# The body is colored by pressure. The wheels and seals get no material and
# a dark display color. The iso-surface contours pressure and is colored by
# vertical velocity. The streamlines start from a sphere of seeds ahead of
# the car and are colored by speed; that last choice needs the velocity
# components combined into a magnitude, which the builder does not expose,
# so the typed schema wrapper sets the selection's mode afterwards. That is
# the general pattern: builders for the common case, schema wrappers for
# everything else.

# %%
import ovcae.authoring as author
from ovcae.authoring import schemas as S

stage.DefinePrim("/World/Recipes", "Scope")

body = author.create_faces(
    stage,
    "/World/Recipes/Body",
    source=datasets["body_Surface"],
    colors="Pressure",
    colormap="coolwarm",
    color_range=(-800.0, 400.0),
)
for name in ("front_wheels_Surface", "rear_wheels_Surface", "front_wheels_Seal", "rear_wheels_Seal"):
    author.create_faces(
        stage, f"/World/Recipes/{name}", source=datasets[name], materials=False, display_color=(0.05, 0.05, 0.05)
    )

iso_surface = author.create_iso_surface(
    stage,
    "/World/Recipes/IsoSurface",
    source=fluid,
    contour="Pressure",
    iso_value=-200.0,
    colors="Velocity_2",
    colormap="gist_rainbow",
    color_range=(-15.0, 15.0),
)

seeds = author.create_seed_sphere(stage, "/World/Controls/Seeds", center=(-2.0, 0.0, 0.5), radius=0.6, count=48)
streamlines = author.create_streamlines(
    stage,
    "/World/Recipes/Streamlines",
    source=fluid,
    seeds=seeds,
    velocities=("Velocity_0", "Velocity_1", "Velocity_2"),
    colors=("Velocity_0", "Velocity_1", "Velocity_2"),
    max_steps=200,
    width=0.01,
    colormap="turbo",
    color_range=(0.0, 40.0),
)
S.FieldSelectionAPI.get(streamlines, "colors").mode = "vector_magnitude"

for prim in (body, iso_surface, streamlines):
    problems = author.validate_recipe(prim)
    print(f"{prim.GetPath()!s:<32} {problems or 'ok'}")

# %% [markdown]
# `validate_recipe` checks structure: the operator schema is applied, the
# prim type matches, every required selection names something, tokens are
# allowed. It does not open the file, so it cannot know whether a field
# exists; that answer comes from evaluation, as the last section shows.

# %% [markdown]
# ## Add a camera and a render product
#
# Nothing in OVCAE authors these; they belong to the application, and they
# are ordinary USD. The camera pose is the one the earlier scenes used, a
# dome light gives the surfaces something to reflect, and the render product
# names the camera, the resolution, and the color output OVRTX should
# produce.

# %%
from pxr import Gf, Sdf, UsdLux, UsdRender

camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.CreateFocalLengthAttr(18.147562)
camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10000.0))
camera.AddTranslateOp().Set(Gf.Vec3d(-4.7114, -3.0067, 2.0345))
camera.AddRotateXYZOp().Set(Gf.Vec3f(74.92261, 0.0, -60.96017))

UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)

product = UsdRender.Product.Define(stage, "/Render/Camera")
product.CreateResolutionAttr(Gf.Vec2i(1280, 720))
product.CreateCameraRel().SetTargets([camera.GetPath()])
color_output = UsdRender.Var.Define(stage, "/Render/Camera/LdrColor")
color_output.CreateSourceNameAttr("LdrColor")
color_output.GetPrim().CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
product.CreateOrderedVarsRel().SetTargets([color_output.GetPath()])

stage.Save()
del stage
print("Saved", scene_path, f"({scene_path.stat().st_size} bytes)")

# %% [markdown]
# The whole scene is a small text file: a payload arc, prims with schemas and
# attributes, and seed points. Generated visualization geometry is not saved.
# Open it in an editor. Everything the evaluation will produce is derived
# from what is in there plus the CGNS file it points at.

# %% [markdown]
# ## Evaluate and render it
#
# The pipeline is tutorial 1's, packaged as a function because this
# tutorial runs it three times: on the scene just saved, on the same file
# reopened cold, and on a broken copy. It populates an OVStage from the
# file, evaluates every recipe, commits, optionally renders one frame, and
# reads back how many points each product published.

# %%
import ovstage
from support import read_attribute, save_png

RENDER_PRODUCT = "/Render/Camera"
PRODUCTS = ("Body", "IsoSurface", "Streamlines")


def evaluate_scene(path, *, png=None) -> dict[str, int]:
    """Populate, evaluate, commit, optionally render, and return points per product."""
    renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=cuda_gpu_index(device) or "0"))
    stage = ovstage.Stage("tutorial-03")
    ovstage.population.open_usd_with_desc(
        stage,
        str(path),
        ordinal=1,
        time_code=0.0,
        descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
    )
    stage.advance_write_floor(1).wait()
    data_source = ovcae.readers.UsdDataSource.open(path)
    session = ovcae.Session(stage, device=device)
    try:
        with data_source.at_time(0.0) as frame_data:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame_data).wait()
        for result in report.operators:
            print(f"  {result.operator_path:<36} {result.status:<10} {result.reason or ''}")
        for diagnostic in report.diagnostics:
            print(f"  [{diagnostic.severity}] {diagnostic.code} at {diagnostic.operator_path}:")
            print(f"      {diagnostic.message}")
        if not report.committable:
            raise RuntimeError("nothing to commit")
        stage.advance_write_floor(2).wait()
        counts = {}
        for name in PRODUCTS:
            points = read_attribute(stage, f"/World/Recipes/{name}", "points", 2)
            counts[name] = 0 if points is None else len(np.asarray(points))
        if png is not None:
            renderer.attach_ovstage(stage)
            for _ in range(2):  # the second step has everything uploaded
                pixels = render_frame(renderer)
            renderer.detach_ovstage()
            save_png(pixels, png)
        return counts
    finally:
        session.close()
        data_source.close()
        stage.destroy()
        renderer.destroy()


def render_frame(renderer) -> np.ndarray:
    products = renderer.step(render_products={RENDER_PRODUCT}, delta_time=0.0, ordinal=2)
    with (
        products[RENDER_PRODUCT]
        .frames[0]
        .render_vars[f"{RENDER_PRODUCT}/LdrColor"]
        .map(device=ovrtx.Device.CPU) as mapped
    ):
        return np.from_dlpack(mapped).copy()


png_path = OUTPUT_DIR / "03_bring_your_own_data.png"
print("Evaluating", scene_path.name)
first_counts = evaluate_scene(scene_path, png=png_path)
print("Points published:", first_counts)

if "get_ipython" in globals():
    from IPython.display import Image, display

    display(Image(filename=str(png_path)))

# %% [markdown]
# ## Reopen and reproduce
#
# The scene records intent, not results: no derived arrays were saved, only
# the payload, the selections, and the settings. Opening the file in a fresh
# pipeline and evaluating it again must give the same products back. That
# is what lets a workflow be shared as a small text file and re-run on
# another machine, or the same one next year.
#
# The counts are compared within a tolerance rather than exactly. GPU
# operators that build surfaces in parallel can emit a handful of triangles
# in a different order or resolve a degenerate cell differently between
# runs, so a few points in millions may differ. The geometry is the same;
# the check allows for that.

# %%
print("Evaluating", scene_path.name, "again in a fresh pipeline")
second_counts = evaluate_scene(scene_path)
for name in PRODUCTS:
    before, after = first_counts[name], second_counts[name]
    difference = abs(before - after) / max(before, 1)
    verdict = "reproduced" if difference < 1e-3 else "DIFFERENT"
    print(f"{name:<12} {before:>9} -> {after:>9}  {verdict}")

# %% [markdown]
# ## When it goes wrong
#
# Recipes are checked at two levels. `validate_recipe` catches structural
# mistakes while authoring. Anything that depends on the data, such as
# whether a field exists, can only be discovered by evaluating, and it comes
# back as a **diagnostic** on the update report: a severity, a code, the
# recipe it concerns, and a message meant to be acted on.
#
# To see one, this cell edits the saved scene through the typed schema
# wrapper, misspells the body's color field, and saves that as a separate
# file. Validation passes, because the structure is fine. Evaluation does
# not.

# %%
broken_path = OUTPUT_DIR / "my_aero_scene_broken.usda"
broken = Usd.Stage.Open(str(scene_path))
body_colors = S.FieldSelectionAPI.get(broken.GetPrimAtPath("/World/Recipes/Body"), "colors")
body_colors.field_names = ["Presure"]
print(
    "validate_recipe:", author.validate_recipe(broken.GetPrimAtPath("/World/Recipes/Body")) or "no structural problems"
)
broken.GetRootLayer().Export(str(broken_path))
del broken

print("Evaluating", broken_path.name)
broken_counts = evaluate_scene(broken_path)

# %% [markdown]
# The report is still **committable**: the other six recipes succeeded and
# their products were published, while the body reports `failed` and the
# diagnostic names the missing field and the zone it searched. An
# application shows that message and leaves the rest of the scene intact;
# the fix is the same edit in reverse, with the right spelling, followed by
# another update. Nothing was lost and nothing had to be rebuilt.

# %% [markdown]
# ## What just happened
#
# You went from a CGNS file to a scene of your own: composed the file with
# the CAE OpenUSD plug-ins through `import_dataset`, read field ranges with
# Warp SimData to choose sensible values, authored recipes with the
# builders and refined one with a schema wrapper, added a camera and render
# product in plain USD, and saved it all as a few kilobytes of text. The
# same file evaluated again from cold gave the same products, and a broken
# recipe explained itself.
#
# To stream this scene with tutorial 2, keep it in `build/tutorials` so its
# relative CGNS payload stays valid. Import `OUTPUT_DIR` from `support`, use
# `OUTPUT_DIR / "my_aero_scene.usda"` for `scene_path`, and set `ISO_SURFACE`
# to `"/World/Recipes/IsoSurface"`. The camera and render product paths
# already match. The additional Streamlines recipe also appears in update
# reports.
#
# Appendix A1 uses the plug-ins and Warp SimData without OVCAE or a renderer.
# It still uses an OpenUSD stage to access the native data.
