# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""One CAE workflow without a user interface: author it, run it, render it, save it, reopen it.

``app.py`` puts an OVUI window on top of this module and the tests drive it
directly. Nothing here imports OVUI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ovcae import UpdateError
from ovcae.app import UpdateLoop

LOGGER = logging.getLogger("ovui_minimal")

# The scene this example authors. Reopening checks for the iso-surface recipe.
ISO_SURFACE = "/World/Recipes/IsoSurface"
BOUNDARIES = "/World/Recipes/Boundaries"
CAMERA = "/World/Camera"
RENDER_PRODUCT = "/Render/Camera"
ISO_VALUE = "cae:viz:isoSurface:isoValue"
FOV_DEGREES = 50.0
# The camera authored with the scene, replaced by the pose the window saves.
PLACEHOLDER_CAMERA = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 10, 1]], dtype=np.float64)


def initialize() -> None:
    """Register OVRTX's schemas and OVCAE's readers, schemas, and materials before any stage exists."""
    import ovrtx

    ovrtx.register_schema_paths()
    import ovcae

    ovcae.setup()


@dataclass(frozen=True)
class Authored:
    """What the author chose: the dataset, the field, its range, and the starting iso value."""

    dataset: str
    field: str
    field_range: tuple[float, float]
    iso_value: float
    boundaries: tuple[str, ...]


def author_scene(data_file, scene_path, *, dataset=None, field=None, iso_value=None, device="cpu") -> Authored:
    """Compose ``data_file`` into a new scene at ``scene_path`` and author the recipes into it.

    The scene records intent only: a payload to the file, an iso-surface recipe
    on ``dataset`` (the first dataset in the file by default) contouring
    ``field`` (Temperature if the dataset has one, else its first field), one
    hidden Faces recipe per other
    dataset, a camera, a light, and a render product. No array is written. The
    field's range, read with Warp SimData, picks the starting iso value and
    the colormap domain.
    """
    import warp_simdata.usd as simdata_usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender

    import ovcae.authoring as author
    import ovcae.readers as readers

    initialize()
    data_file, scene_path = Path(data_file).resolve(), Path(scene_path).resolve()
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    holder = readers.import_dataset(stage, "/World/Data", os.path.relpath(data_file, scene_path.parent))
    datasets = {prim.GetName(): prim for prim in readers.dataset_prims(stage, holder)}
    if not datasets:
        raise ValueError(f"{data_file} composed no datasets; is its format supported by the CAE OpenUSD plug-ins?")
    if dataset is None:
        dataset = next(iter(datasets))
    if dataset not in datasets:
        raise ValueError(f"no dataset {dataset!r} in {data_file.name}; found {', '.join(datasets)}")
    volume = datasets[dataset]
    fields = [info.name for info in simdata_usd.list_fields(volume, time_code=Usd.TimeCode(0.0))]
    if not fields:
        raise ValueError(f"dataset {dataset!r} carries no fields to contour")
    if field is None:
        # Temperature reads well on the bundled static mixer; any other file gets its first field.
        field = "Temperature" if "Temperature" in fields else fields[0]
    if field not in fields:
        raise ValueError(f"no field {field!r} on {dataset!r}; found {', '.join(fields)}")
    loaded = simdata_usd.field_from_prim(volume, field, device=device, time_code=0.0)
    low, high = (float(v) for v in loaded.get_range())
    if iso_value is None:
        iso_value = (low + high) / 2

    stage.DefinePrim("/World/Recipes", "Scope")
    author.create_iso_surface(
        stage,
        ISO_SURFACE,
        source=volume,
        contour=field,
        iso_value=iso_value,
        colormap="coolwarm",
        color_range=(low, high),
    )
    stage.DefinePrim(BOUNDARIES, "Scope")
    boundaries = []
    for name, prim in datasets.items():
        if prim == volume:
            continue
        faces = author.create_faces(
            stage, f"{BOUNDARIES}/{name}", source=prim, materials=False, display_color=(0.55, 0.55, 0.55)
        )
        UsdGeom.Imageable(faces).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        boundaries.append(faces.GetPath().pathString)

    camera = UsdGeom.Camera.Define(stage, CAMERA)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100000.0))
    camera.AddTransformOp().Set(Gf.Matrix4d(PLACEHOLDER_CAMERA.tolist()))
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)
    product = UsdRender.Product.Define(stage, RENDER_PRODUCT)
    product.CreateResolutionAttr(Gf.Vec2i(1280, 720))
    product.CreateCameraRel().SetTargets([camera.GetPath()])
    color = UsdRender.Var.Define(stage, RENDER_PRODUCT + "/LdrColor")
    color.CreateSourceNameAttr("LdrColor")
    color.GetPrim().CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
    product.CreateOrderedVarsRel().SetTargets([color.GetPath()])
    stage.Save()
    return Authored(dataset, field, (low, high), float(iso_value), tuple(boundaries))


class Workflow:
    """Own the OVStage, the OVRTX renderer, and the update loop for one saved scene.

    Recipe edits go through :meth:`set_iso_value` and :meth:`show_boundaries`
    and re-evaluate; the camera goes through :meth:`render` and does not.
    :meth:`save` writes the current edits and camera back to the scene file,
    which is all that :meth:`open` needs to reproduce the picture.
    """

    def __init__(self, scene_path, *, device: str = "cuda:0") -> None:
        self.scene_path = Path(scene_path).resolve()
        self.device = device
        self.usd_stage = None
        self.stage = self.renderer = self.data_source = self.loop = None
        self.boundaries: tuple[str, ...] = ()
        self.field = ""
        self.field_range = (0.0, 1.0)
        self.up_axis = "Y"
        self.width, self.height = 1280, 720
        self.last_frame = None
        self.last_error = None
        self._camera = None
        self._viewport_signature = None
        self._attached = False

    # -- lifecycle -------------------------------------------------------------------

    def open(self) -> Workflow:
        """Populate the OVStage from the scene, evaluate it once, and attach the renderer."""
        if self.loop is not None:
            return self
        initialize()
        import ovrtx
        import ovstage
        from pxr import Tf, Usd, UsdGeom, UsdShade

        import ovcae
        import ovcae.readers
        from ovcae.authoring import schemas as S

        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)
        try:
            self.usd_stage = Usd.Stage.Open(str(self.scene_path))
        except Tf.ErrorException as exc:
            raise ValueError(f"{self.scene_path.name} is not a readable USD scene") from exc
        if self.usd_stage is None:
            raise ValueError(f"{self.scene_path.name} is not a readable USD scene")
        iso = self.usd_stage.GetPrimAtPath(ISO_SURFACE)
        if not iso or "CaeVizIsoSurfaceAPI" not in iso.GetAppliedSchemas():
            raise ValueError(
                f"{self.scene_path.name} was not authored by this example: no iso-surface at {ISO_SURFACE}"
            )
        self.field = str(S.FieldSelectionAPI.get(iso, "contour").field_names[0])
        shader = UsdShade.Shader(self.usd_stage.GetPrimAtPath(f"{ISO_SURFACE}/Materials/ScalarColor/Shader"))
        domain = shader.GetInput("domain").Get() if shader else None
        if domain is not None and domain[0] < domain[1]:
            self.field_range = (float(domain[0]), float(domain[1]))
        self.boundaries = tuple(
            child.GetPath().pathString for child in self.usd_stage.GetPrimAtPath(BOUNDARIES).GetChildren()
        )
        self.up_axis = str(UsdGeom.GetStageUpAxis(self.usd_stage))
        try:
            # The renderer reads the material library's search paths when it is
            # created, and population must happen after that or colormap
            # textures render grey. OVRTX is pinned to the compute GPU: with
            # two GPUs and a headless window its product copies fail otherwise.
            self.renderer = ovrtx.Renderer(ovrtx.RendererConfig(active_cuda_gpus=render_gpus(self.device)))
            self.stage = ovstage.Stage("ovcae.example.ovui-minimal")
            ovstage.population.open_usd_with_desc(
                self.stage,
                str(self.scene_path),
                ordinal=1,
                time_code=0.0,
                descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
            )
            self.stage.advance_write_floor(1).wait()
            self.data_source = ovcae.readers.UsdDataSource.open(self.scene_path)
            self.loop = UpdateLoop(self.stage, self.data_source, ordinal=1, device=self.device)
            # The pinned OVRTX build ignores ``primvars:displayColor`` on a mesh that
            # had no points when the renderer first saw it once its geometry arrives
            # from CUDA memory (ovcae issue #4), and operators publish from the
            # compute device, so a boundary hidden here would come back grey.
            # Evaluate hidden recipes once so their geometry exists before the
            # renderer attaches, then hide them again; a hidden recipe keeps its
            # products and is not evaluated again.
            hidden = [path for path in self.boundaries if self._authored_visibility(path) == "invisible"]
            self.loop.update({(path, "visibility"): "inherited" for path in hidden})
            if hidden:
                self.loop.update({(path, "visibility"): "invisible" for path in hidden})
            LOGGER.info("Attaching OVRTX; the first launch can take minutes to compile shaders")
            self.renderer.attach_ovstage(self.stage)
            self._attached = True
        except BaseException:
            self.close()
            raise
        return self

    def _authored_visibility(self, path: str) -> str:
        from pxr import UsdGeom

        return str(UsdGeom.Imageable(self.usd_stage.GetPrimAtPath(path)).GetVisibilityAttr().Get())

    def close(self) -> None:
        self.last_frame = None
        if self.loop is not None:
            self.loop.close()
            self.loop = None
        if self.data_source is not None:
            self.data_source.close()
            self.data_source = None
        if self.renderer is not None and self._attached:
            self.renderer.detach_ovstage()
            self._attached = False
        if self.stage is not None:
            self.stage.destroy()
            self.stage = None
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        self.usd_stage = None

    def __enter__(self) -> Workflow:
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- recipe edits ------------------------------------------------------------------

    @property
    def iso_value(self) -> float:
        return float(self.loop.value(ISO_SURFACE, ISO_VALUE))

    def set_iso_value(self, value: float) -> None:
        """Buffer a recipe edit; the loop submits it when no update is in flight."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("the iso value must be finite")
        self.loop.request({(ISO_SURFACE, ISO_VALUE): value})

    @property
    def boundaries_visible(self) -> bool:
        return bool(self.boundaries) and self.loop.value(self.boundaries[0], "visibility") != "invisible"

    def show_boundaries(self, visible: bool) -> None:
        """Visibility is a recipe edit: a hidden recipe is not evaluated, so showing one refreshes it."""
        token = "inherited" if visible else "invisible"
        self.loop.request({(path, "visibility"): token for path in self.boundaries})

    def pump(self):
        """Advance the loop without blocking; the report of an update that just committed, else None."""
        return self.loop.pump()

    @property
    def evaluating(self) -> bool:
        return self.loop.evaluating

    def await_evaluation(self, budget: float) -> bool:
        return self.loop.await_evaluation(budget)

    # -- camera and rendering -----------------------------------------------------------

    def render(self, width: int, height: int, view, projection) -> np.ndarray:
        """Write the viewport's camera and size as presentation edits, then render one frame.

        OVRTX cannot step while an update writes above the floor, so while one
        is in flight the previous frame is presented and the camera write waits.
        """
        if width <= 0 or height <= 0:
            raise ValueError("viewport dimensions must be positive")
        if self.loop.evaluating and self.loop.await_evaluation(0.0):
            self._flush()
        if self.loop.evaluating and self.last_frame is not None:
            return self.last_frame
        self._flush()
        world, intrinsics = viewport_camera(view, projection)
        signature = (width, height, world.tobytes(), intrinsics)
        if signature != self._viewport_signature:
            focal, horizontal, vertical, near, far = intrinsics
            edits = {
                (RENDER_PRODUCT, "resolution"): (width, height),
                (CAMERA, "omni:xform"): world,
                (CAMERA, "focalLength"): focal,
                (CAMERA, "horizontalAperture"): horizontal,
                (CAMERA, "verticalAperture"): vertical,
                (CAMERA, "clippingRange"): (near, far),
            }
            types = {
                (RENDER_PRODUCT, "resolution"): "int2",
                (CAMERA, "omni:xform"): "matrix4d",
                (CAMERA, "focalLength"): "float",
                (CAMERA, "horizontalAperture"): "float",
                (CAMERA, "verticalAperture"): "float",
                (CAMERA, "clippingRange"): "float2",
            }
            self.loop.write(edits, types=types)
            self.width, self.height = width, height
            self._camera = world
            self._viewport_signature = signature
        import ovrtx

        products = self.renderer.step(render_products={RENDER_PRODUCT}, delta_time=0.0, ordinal=self.loop.ordinal)
        frame = products[RENDER_PRODUCT].frames[0]
        mapped = frame.render_vars[f"{RENDER_PRODUCT}/LdrColor"].map(device=ovrtx.Device.CPU)
        try:
            self.last_frame = np.from_dlpack(mapped).copy()
        finally:
            mapped.unmap()
            del mapped, frame, products
        return self.last_frame

    def _flush(self) -> None:
        """Commit a finished update; an uncommittable one is reported, not raised, so the frame still renders."""
        try:
            self.loop.flush()
        except UpdateError as exc:
            LOGGER.error("%s", exc)
            self.last_error = str(exc)

    def bounds(self):
        """World bounds of every product this scene publishes, or None before anything exists."""
        self.loop.flush()
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for path in (ISO_SURFACE, *self.boundaries):
            points = self.loop.value(path, "points", type_name="point3f[]")
            if points is None or not len(points):
                continue
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                low, high = np.minimum(low, points.min(axis=0)), np.maximum(high, points.max(axis=0))
        return None if not np.isfinite(low).all() else (low, high)

    def overview_pose(self) -> dict:
        """A camera that frames every product, looking from the front-upper-right."""
        bounds = self.bounds()
        center, radius = (np.zeros(3), 5.0) if bounds is None else ((bounds[0] + bounds[1]) / 2, None)
        if radius is None:
            radius = max(float(np.linalg.norm(bounds[1] - bounds[0]) / 2), 0.01)
        direction = np.array((1.0, 0.65, 1.25) if self.up_axis == "Y" else (1.0, -1.25, 0.65))
        eye = center + direction / np.linalg.norm(direction) * radius * 3.2
        return dict(eye=tuple(eye), target=tuple(center), up_axis=self.up_axis, fov_degrees=FOV_DEGREES, prim_path="")

    def saved_pose(self) -> dict | None:
        """The pose a previous session saved into the scene, or None when the scene still has the placeholder."""
        from pxr import UsdGeom

        matrix = np.array(UsdGeom.Xformable(self.usd_stage.GetPrimAtPath(CAMERA)).GetLocalTransformation())
        if np.allclose(matrix, PLACEHOLDER_CAMERA):
            return None
        bounds = self.bounds()
        center = np.zeros(3) if bounds is None else (bounds[0] + bounds[1]) / 2
        return pose_from_matrix(matrix, center, up_axis=self.up_axis)

    # -- persistence -------------------------------------------------------------------

    def save(self) -> Path:
        """Write the current iso value, boundary visibility, and camera into the scene file."""
        from pxr import Gf, UsdGeom

        from ovcae.authoring import schemas as S

        self.loop.flush()
        S.IsoSurfaceAPI.get(self.usd_stage.GetPrimAtPath(ISO_SURFACE)).iso_value = self.iso_value
        token = UsdGeom.Tokens.inherited if self.boundaries_visible else UsdGeom.Tokens.invisible
        for path in self.boundaries:
            UsdGeom.Imageable(self.usd_stage.GetPrimAtPath(path)).GetVisibilityAttr().Set(token)
        if self._camera is not None:
            ops = UsdGeom.Xformable(self.usd_stage.GetPrimAtPath(CAMERA)).GetOrderedXformOps()
            ops[0].Set(Gf.Matrix4d(self._camera.tolist()))
        self.usd_stage.Save()
        return self.scene_path


def render_gpus(device: str) -> str | None:
    """Map a Warp device to OVRTX's ``active_cuda_gpus``: ``cuda:N`` pins GPU N, anything else leaves it alone."""
    name, _, index = str(device).partition(":")
    if name != "cuda":
        return None
    if index and not index.isdigit():
        raise ValueError(f"unsupported CUDA device: {device!r}")
    return index or "0"


def viewport_camera(view, projection) -> tuple[np.ndarray, tuple[float, ...]]:
    """Translate OVUI's column-vector view and perspective projection into USD camera values."""
    view = np.asarray(view, dtype=np.float64)
    projection = np.asarray(projection, dtype=np.float64)
    if view.shape != (4, 4) or projection.shape != (4, 4):
        raise ValueError("view and projection matrices must be 4x4")
    if not np.isfinite(view).all() or not np.isfinite(projection).all():
        raise ValueError("camera matrices must be finite")
    if projection[0, 0] <= 0 or projection[1, 1] <= 0 or not np.isclose(projection[3, 2], -1):
        raise ValueError("a perspective camera is required")
    if projection[2, 2] >= -1 or projection[2, 3] >= 0:
        raise ValueError("finite perspective clipping planes are required")
    near = projection[2, 3] / (projection[2, 2] - 1)
    far = projection[2, 3] / (projection[2, 2] + 1)
    focal_length = 28.0
    intrinsics = (focal_length, 2 * focal_length / projection[0, 0], 2 * focal_length / projection[1, 1], near, far)
    return np.ascontiguousarray(np.linalg.inv(view).T), intrinsics


def pose_from_matrix(matrix, center, *, up_axis: str = "Y") -> dict:
    """Turn a row-vector camera transform back into the eye and target a viewport camera takes.

    The camera looks down its local -Z. The target sits on that ray at the
    depth of ``center``, so orbiting from the restored pose pivots where the
    saved view was looking.
    """
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    eye = matrix[3, :3]
    forward = -matrix[2, :3]
    length = np.linalg.norm(forward)
    if length <= 0:
        raise ValueError("the camera transform has no viewing direction")
    forward = forward / length
    distance = float(np.dot(np.asarray(center, dtype=np.float64) - eye, forward))
    if distance <= 0:
        distance = 10.0
    return dict(
        eye=tuple(eye), target=tuple(eye + forward * distance), up_axis=up_axis, fov_degrees=FOV_DEGREES, prim_path=""
    )
