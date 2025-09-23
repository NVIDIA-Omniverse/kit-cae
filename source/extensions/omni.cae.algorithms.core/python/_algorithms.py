# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.data import array_utils, progress, usd_utils
from omni.cae.data.commands import ComputeBounds, ConvertToMesh, ConvertToPointCloud, GenerateStreamlines
from omni.cae.data.commands import Streamlines as StreamlinesT
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from usdrt import Sdf as SdfRt
from usdrt import Usd as UsdRt
from usdrt import UsdGeom as UsdGeomRt
from usdrt import Vt as VtRt

from .algorithm import Algorithm

logger = getLogger(__name__)


def get_shader(materials_pxr: Usd.Prim) -> UsdShade.Shader:
    shaders = []
    for child in materials_pxr.GetChildren():
        if child.IsA(UsdShade.Shader):
            return UsdShade.Shader(child)
    return None


def set_shader_input(material_prim, name, type, value):
    shader = get_shader(material_prim)
    if not shader:
        logger.error("No shaders were found under %s", material_prim)
        return
    else:
        shader.CreateInput(name, type).Set(value)


async def set_shader_domain(
    material_prim: Usd.Prim, datasetPrim: Usd.Prim, fieldName: str, timeCode: Usd.TimeCode, domain=None
):
    shader = get_shader(material_prim)
    if not shader:
        logger.error("No shaders were found under %s", material_prim)
        return

    inp = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
    if inp.Get() == None or inp.Get() == (0, -1):
        if domain is None:
            arrayPrim = usd_utils.get_target_prim(datasetPrim, f"field:{fieldName}")
            domain = await usd_utils.get_array_range(arrayPrim, timeCode, quiet=True)

        if domain:
            logger.info("setting domain to %s on %s (overriding %s)", domain, shader, inp.Get())
            inp.Set(domain)
    else:
        logger.info("domain already set to %s on %s (skipping)", inp.Get(), shader)


class BoundingBox(Algorithm):

    ns = "omni:cae:algorithms:boundingBox"

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsBoundingBoxAPI"])

    async def initialize(self):

        # check if already initialized, and if so, don't bother doing this.
        primT: UsdGeom.BasisCurves = UsdGeom.BasisCurves(self.prim)
        if primT.GetPointsAttr().HasValue():
            logger.info("Already initialized %s", self.prim)
            return await super().initialize()

        dataset_prims = usd_utils.get_target_prims(self.prim, f"{self.ns}:datasets")
        nb_datasets = len(dataset_prims)
        bbox = Gf.Range3d()
        for idx, p in enumerate(dataset_prims):
            with progress.ProgressContext(shift=idx / nb_datasets, scale=1.0 / nb_datasets):
                d_bbox: Gf.Range3d = await ComputeBounds.invoke(p, Usd.TimeCode.EarliestTime())
            if d_bbox.IsEmpty():
                logger.warning("Failed to get valid bounding box for %s. Skipping", p)
            else:
                bbox = bbox.UnionWith(d_bbox)

        if bbox.IsEmpty():
            logger.error("Failed to get valid bounding box for %s", dataset_prims)
        else:
            pts = [
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 0),
                (0, 1, 0),
                (0, 1, 0),
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (1, 1, 1),
                (0, 1, 1),
                (0, 1, 1),
                (0, 0, 1),
                (0, 0, 0),
                (0, 0, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
                (0, 1, 0),
                (0, 1, 1),
            ]
            coords = []
            min_pt = np.array(bbox.min)
            max_pt = np.array(bbox.max)
            delta = max_pt - min_pt
            for pt in pts:
                coord = min_pt + pt * delta
                coords.append(coord.tolist())

            primT.CreateExtentAttr().Set([bbox.min, bbox.max])
            primT.CreatePointsAttr().Set(coords)
            primT.CreateCurveVertexCountsAttr().Set([2] * 12)
        return await super().initialize()

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        width = usd_utils.get_attribute(self.prim, f"{self.ns}:width")
        primT: UsdGeom.BasisCurves = UsdGeom.BasisCurves(self.prim)
        primT.CreateWidthsAttr().Set([(width)])
        return True


class Points(Algorithm):
    ns = "omni:cae:algorithms:points"

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsPointsAPI"])

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        dataset_prim = usd_utils.get_target_prim(self.prim, f"{self.ns}:dataset")
        width = usd_utils.get_attribute(self.prim, f"{self.ns}:width")
        max_count = usd_utils.get_attribute(self.prim, f"{self.ns}:maxCount")

        colors_field = usd_utils.get_target_field_name(self.prim, f"{self.ns}:colors", dataset_prim, quiet=True)
        widths_field = usd_utils.get_target_field_name(self.prim, f"{self.ns}:widths", dataset_prim, quiet=True)

        fields = set()
        if colors_field:
            fields.add(colors_field)
        if widths_field:
            fields.add(widths_field)

        result = await ConvertToPointCloud.invoke(dataset_prim, list(fields), timeCode)
        points: np.ndarray = array_utils.as_numpy_array(result.points)

        if max_count > 0 and len(points) > max_count:
            # set a stride to limit the number of points
            stride = int(np.ceil(len(points) / max_count))
        else:
            stride = 1

        points = points[::stride] if stride > 1 else points

        primT: UsdGeomRt.Points = UsdGeomRt.Points(self.prim_rt)
        pvAPI = UsdGeomRt.PrimvarsAPI(primT.GetPrim())

        primT.GetPointsAttr().Set(VtRt.Vec3fArray(points))

        # exts = [np.amin(points, axis=0).tolist(), np.amax(points, axis=0).tolist()]
        scalar_pvar = pvAPI.GetPrimvar("scalar")
        widths_pvar = pvAPI.GetPrimvar("widths")

        if colors_field and colors_field in result.fields:
            colors = array_utils.as_numpy_array(result.fields[colors_field])
            colors = colors[::stride] if stride > 1 else colors
            assert colors.shape[0] == points.shape[0]
            scalar_pvar.Set(VtRt.FloatArray(colors.reshape(-1, 1)))
        else:
            # deactivate scalar coloring.
            scalar_pvar.Set(VtRt.FloatArray(np.full(points.shape[0], 0.0, dtype=np.float32).reshape(-1, 1)))
            # colors = np.full(points.shape[0], 0.0, dtype=np.float32)

        if material := self.get_material("ScalarColor"):
            if colors_field:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, True)
                await set_shader_domain(material, dataset_prim, colors_field, timeCode)
            else:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, False)
        else:
            logger.warning("No material found for ScalarColor")

        if widths_field and widths_field in result.fields:
            widths = array_utils.as_numpy_array(result.fields[widths_field])
            widths = widths[::stride] if stride > 1 else widths
            assert widths.shape[0] == points.shape[0]

            widths_domain = usd_utils.get_attribute(self.prim, f"{self.ns}:widthsDomain")
            widths_ramp = usd_utils.get_attribute(self.prim, f"{self.ns}:widthsRamp")

            # since ramp is specified as scale factor, we need to scale width by it.
            widths_ramp = np.array(widths_ramp) * width

            widths = np.clip(widths, widths_domain[0], widths_domain[1])
            widths = widths_ramp[0] + (widths - widths_domain[0]) / (widths_domain[1] - widths_domain[0]) * (
                widths_ramp[1] - widths_ramp[0]
            )
            widths_pvar.Set(VtRt.FloatArray(widths.reshape(-1, 1)))
        else:
            widths_pvar.Set(VtRt.FloatArray(np.full(points.shape[0], width, dtype=np.float32).reshape(-1, 1)))

        # set extents
        self.set_extent(Gf.Range3d(np.amin(points, axis=0).tolist(), np.amax(points, axis=0).tolist()))
        return True


class Glyphs(Algorithm):

    ns = "omni:cae:algorithms:glyphs"

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsGlyphsAPI"])

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        dataset_prim = usd_utils.get_target_prim(self.prim, f"{self.ns}:dataset")
        shape = usd_utils.get_attribute(self.prim, f"{self.ns}:shape")
        max_count = usd_utils.get_attribute(self.prim, f"{self.ns}:maxCount")

        orientation_fields = usd_utils.get_target_field_names(
            self.prim, f"{self.ns}:orientation", dataset_prim, quiet=True
        )
        color_field = usd_utils.get_target_field_name(self.prim, f"{self.ns}:colors", dataset_prim, quiet=True)

        fields = set()
        if color_field:
            fields.add(color_field)
        if orientation_fields:
            fields.update(orientation_fields)

        result = await ConvertToPointCloud.invoke(dataset_prim, list(fields), timeCode)
        points: np.ndarray = array_utils.as_numpy_array(result.points).astype(np.float32, copy=False)

        if max_count > 0 and len(points) > max_count:
            # set a stride to limit the number of points
            stride = int(np.ceil(len(points) / max_count))
        else:
            stride = 1

        points = points[::stride] if stride > 1 else points
        protoIndices = np.empty(points.shape[0], dtype=np.int32)
        if shape == "arrow":
            protoIndices.fill(0)
        elif shape == "cone":
            protoIndices.fill(1)
        elif shape == "sphere":
            protoIndices.fill(2)

        if len(orientation_fields) == 3:
            arrays = [array_utils.as_numpy_array(result.fields[i]) for i in orientation_fields]
            if stride > 1:
                arrays = [a[::stride] for a in arrays]
            orientations: np.ndarray = array_utils.as_numpy_array(array_utils.stack(arrays))
        elif len(orientation_fields) == 1:
            array = array_utils.as_numpy_array(result.fields[orientation_fields[0]])
            assert array.shape[0] == points.shape[0] and array.shape[1] == 3
            orientations: np.ndarray = array[::stride] if stride > 1 else array
        else:
            orientations = None

        if orientations is not None:
            quaternions = array_utils.compute_quaternions_from_directions(orientations)
        else:
            quaternions = None

        if color_field:
            scalars: np.ndarray = array_utils.as_numpy_array(result.fields[color_field])
            if stride > 1:
                scalars = scalars[::stride]
            assert scalars.shape[0] == points.shape[0]
        else:
            scalars = None

        primT = UsdGeomRt.PointInstancer(self.prim_rt)
        primvarsApi = UsdGeomRt.PrimvarsAPI(primT.GetPrim())

        primT.GetPositionsAttr().Set(VtRt.Vec3fArray(points))
        primT.GetProtoIndicesAttr().Set(VtRt.IntArray(protoIndices.reshape(-1, 1).astype(np.intc)))
        primT.GetOrientationsAttr().Set(VtRt.QuathArray(quaternions) if quaternions is not None else [])

        # have to create primvar here, creating in PXR and using here doesn't work.
        if scalar_pvar := primvarsApi.CreatePrimvar("scalar", SdfRt.ValueTypeNames.FloatArray, UsdGeomRt.Tokens.vertex):
            if color_field:
                scalar_pvar.Set(VtRt.FloatArray(scalars.reshape(-1, 1)))
            else:
                # deactivate scalar coloring.
                scalar_pvar.Set(VtRt.FloatArray(np.full(points.shape[0], 0.0, dtype=np.float32).reshape(-1, 1)))
        else:
            logger.error("Scalar primvar not found")

        if material := self.get_material("ScalarColor"):
            if color_field:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, True)
                await set_shader_domain(material, dataset_prim, color_field, timeCode)
            else:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, False)
        else:
            logger.warning("No material found for ScalarColor")

        # set extents
        self.set_extent(Gf.Range3d(np.amin(points, axis=0).tolist(), np.amax(points, axis=0).tolist()))
        return True


class ExternalFaces(Algorithm):
    ns = "omni:cae:algorithms:externalFaces"

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsExternalFacesAPI"])

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        dataset_prim = usd_utils.get_target_prim(self.prim, f"{self.ns}:dataset")
        colors_field = usd_utils.get_target_field_name(self.prim, f"{self.ns}:colors", dataset_prim, quiet=True)
        fields = [colors_field] if colors_field else []

        mesh: ConvertToMesh.Mesh = await ConvertToMesh.invoke(dataset_prim, fields, timeCode)

        mesh = mesh.numpy()

        meshT = UsdGeomRt.Mesh(self.prim_rt)
        primvarsApi = UsdGeomRt.PrimvarsAPI(self.prim_rt)

        meshT.CreatePointsAttr().Set(VtRt.Vec3fArray(mesh.points))
        meshT.CreateFaceVertexCountsAttr().Set(VtRt.IntArray(mesh.faceVertexCounts.reshape(-1, 1).astype(np.intc)))
        meshT.CreateFaceVertexIndicesAttr().Set(VtRt.IntArray(mesh.faceVertexIndices.reshape(-1, 1).astype(np.intc)))
        scalar_pvar = primvarsApi.GetPrimvar("scalar")

        if mesh.normals is not None:
            meshT.CreateNormalsAttr().Set(VtRt.Vec3fArray(mesh.normals))

        if colors_field and colors_field in mesh.fields:
            scalar = mesh.fields[colors_field]
            nb_scalars = scalar.shape[0]
            if nb_scalars == mesh.points.shape[0]:
                # "vertex": Values are interpolated between each vertex in the surface primitive. The basis function
                # of the surface is used for interpolation between vertices.
                scalar_pvar.SetInterpolation(UsdGeomRt.Tokens.vertex)
                scalar_pvar.Set(VtRt.FloatArray(scalar.reshape(-1, 1)))
            elif nb_scalars == mesh.faceVertexCounts.shape[0]:
                # "uniform": One value remains constant for each uv patch segment of the surface primitive
                # (which is a face for meshes).
                scalar_pvar.SetInterpolation(UsdGeomRt.Tokens.uniform)
                scalar_pvar.Set(VtRt.FloatArray(scalar.reshape(-1, 1)))
            else:
                logger.error(
                    "Invalid scalar shape %s (pts=%s, faces=%s)",
                    nb_scalars,
                    mesh.points.shape[0],
                    mesh.faceVertexCounts.shape[0],
                )
                scalar_pvar.SetInterpolation(UsdGeomRt.Tokens.constant)
                scalar_pvar.Set(VtRt.FloatArray([0.0]))
        else:
            scalar_pvar.SetInterpolation(UsdGeomRt.Tokens.constant)
            scalar_pvar.Set(VtRt.FloatArray([0.0]))

        if material := self.get_material("ScalarColor"):
            if colors_field and colors_field in mesh.fields:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, True)
                await set_shader_domain(material, dataset_prim, colors_field, timeCode)
            else:
                set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, False)
        else:
            logger.warning("No material found for ScalarColor")

        self.set_extent(mesh.extents)
        return True


class Streamlines(Algorithm):
    _xform_ops = [
        "omni:fabric:localMatrix",
    ]

    def __init__(self, prim: Usd.Prim):
        super().__init__(prim, ["CaeAlgorithmsStreamlinesAPI"])
        self._ns = "omni:cae:algorithms:streamlines"

        self._rel_tracker: usd_utils.ChangeTracker = usd_utils.ChangeTracker(self.stage)

        # xform ops
        for p in self._xform_ops:
            self._rel_tracker.TrackAttribute(p)

    def needs_update(self, timecode: Usd.TimeCode) -> bool:
        if super().needs_update(timecode):
            return True

        # if seeds were transformed, we need to reexecute.
        seeds_targets = self.prim.GetRelationship(f"{self._ns}:seeds").GetForwardedTargets()
        for t in seeds_targets:
            if self._rel_tracker.PrimChanged(t):
                return True

        return False

    @staticmethod
    def apply_xform(prim: Usd.Prim, coords: np.ndarray) -> np.ndarray:
        _xform_cache = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
        matrix: Gf.Matrix4d = _xform_cache.GetLocalTransformation(prim)[0]
        if matrix:
            coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
            coords_h = coords_h @ matrix
            coords = coords_h[:, :3]  # / coords_h[:, 3]
        return coords

    async def execute(self, timeCode: Usd.TimeCode = None, force: bool = True) -> None:
        await super().execute(timeCode, force)
        self._rel_tracker.ClearChanges()

    async def get_seeds(self, timeCode: Usd.TimeCode) -> wp.array:
        seeds_prim = usd_utils.get_target_prim(self.prim, f"{self._ns}:seeds")
        seeds_result = await ConvertToPointCloud.invoke(seeds_prim, [], timeCode)
        seeds = array_utils.as_numpy_array(seeds_result.points).astype(np.float32, copy=False)
        # apply xform
        xformed_seed_pts = Streamlines.apply_xform(seeds_prim, seeds)
        return wp.array(xformed_seed_pts, dtype=wp.vec3f, copy=False)

    async def execute_impl(self, timeCode: Usd.TimeCode) -> bool:
        logger.info("start executing streamlines")
        dX: float = usd_utils.get_attribute(self.prim, f"{self._ns}:dX")
        maxLength: int = usd_utils.get_attribute(self.prim, f"{self._ns}:maxLength")
        dataset_prim = usd_utils.get_target_prim(self.prim, f"{self._ns}:dataset")
        width: float = usd_utils.get_attribute(self.prim, f"{self._ns}:width")
        _ = usd_utils.get_target_prim(self.prim, f"{self._ns}:seeds")

        if dX <= 0.00001:
            raise RuntimeError(f"Invalid dX '{dX}'")
        if maxLength < 10:
            raise RuntimeError(f"Invalid maxLength '{maxLength}'")

        velocity_field_names = usd_utils.get_target_field_names(self.prim, f"{self._ns}:velocity", dataset_prim)
        color_field_name = usd_utils.get_target_field_name(self.prim, f"{self._ns}:colors", dataset_prim, quiet=True)
        primvars = usd_utils.get_target_field_names(
            self.prim, f"omni:cae:algorithms:primvars", dataset_prim, quiet=True
        )

        seeds = await self.get_seeds(timeCode)
        streamlines = await GenerateStreamlines.invoke(
            dataset_prim, seeds, velocity_field_names, color_field_name, dX, maxLength, timeCode, extra_fields=primvars
        )

        if streamlines is None:
            streamlines = StreamlinesT()
            streamlines.points = np.array(
                [
                    (0, 0, 0),
                    (0.01, 0, 0),
                    (0.02, 0, 0),
                    (0.03, 0, 0),
                    (0, 0, 0),
                    (0.01, 0, 0),
                    (0.02, 0, 0),
                    (0.03, 0, 0),
                ]
            )
            streamlines.curveVertexCounts = np.array([4, 4])

        streamlines = streamlines.numpy()

        scalars = streamlines.fields.get("scalar")
        time = streamlines.fields.get("time")
        rnd = np.random.default_rng(1986).random(streamlines.curveVertexCounts.shape[0], dtype=np.float32)

        basisCurvesT = UsdGeomRt.BasisCurves(self.prim_rt)
        basisCurvesT.CreateWidthsAttr().Set([(width)])
        basisCurvesT.CreatePointsAttr().Set(VtRt.Vec3fArray(streamlines.points))
        basisCurvesT.CreateCurveVertexCountsAttr().Set(
            VtRt.IntArray(streamlines.curveVertexCounts.reshape(-1, 1).astype(np.intc))
        )

        primvarsApi = UsdGeomRt.PrimvarsAPI(self.prim_rt)
        if scalars is not None:
            primvarsApi.GetPrimvar("scalar").Set(VtRt.FloatArray(scalars.reshape(-1, 1)))
        else:
            primvarsApi.GetPrimvar("scalar").Set(
                VtRt.FloatArray(np.full(streamlines.points.shape[0], 0.0, dtype=np.float32).reshape(-1, 1))
            )

        if material := self.get_material("ScalarColor"):
            set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, scalars is not None)
            if scalars is not None:
                await set_shader_domain(material, dataset_prim, color_field_name, timeCode)
        else:
            logger.warning("No material found for ScalarColor")

        if material := self.get_material("AnimatedStreaks"):
            set_shader_input(material, "enable_coloring", Sdf.ValueTypeNames.Bool, scalars is not None)
            if scalars is not None:
                await set_shader_domain(material, dataset_prim, color_field_name, timeCode)
        else:
            logger.warning("No material found for AnimatedStreaks")

        if time is not None:
            primvarsApi.GetPrimvar("time").Set(VtRt.FloatArray(time.reshape(-1, 1)))
        else:
            primvarsApi.GetPrimvar("time").Set(
                VtRt.FloatArray(np.full(streamlines.points.shape[0], 0.0, dtype=np.float32).reshape(-1, 1))
            )

        # generate random ids for each individual streamline
        primvarsApi.GetPrimvar("rnd").Set(VtRt.FloatArray(rnd.reshape(-1, 1)))

        # pass extra primvars
        for pvarname in primvars:
            field = streamlines.fields.get(pvarname)
            if field is not None:
                if field.shape[-1] == 3:
                    primvarsApi.CreatePrimvar(pvarname, SdfRt.ValueTypeNames.Float3Array, UsdGeomRt.Tokens.vertex).Set(
                        VtRt.Float3Array(field.reshape(-1, 3))
                    )
                elif field.shape[-1] == 1 or len(field.shape) == 1:
                    primvarsApi.CreatePrimvar(pvarname, SdfRt.ValueTypeNames.FloatArray, UsdGeomRt.Tokens.vertex).Set(
                        VtRt.FloatArray(field.reshape(-1, 1))
                    )
                else:
                    logger.warning("Unsupported primvar shape %s for %s", field.shape, pvarname)
            else:
                logger.warning("Primvar field %s not found", pvarname)

        # set extents
        if streamlines.points is not None and streamlines.points.shape[0] > 0:
            self.set_extent(
                Gf.Range3d(np.amin(streamlines.points, axis=0).tolist(), np.amax(streamlines.points, axis=0).tolist())
            )

        return True
