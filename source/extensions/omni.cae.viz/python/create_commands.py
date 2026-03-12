# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
r"""
This module contains the commands for creating the CAE Viz primitives.
"""


import asyncio
from logging import getLogger

import numpy as np
import omni.cae.dav as cae_dav
import omni.kit.commands
from omni.cae.data import progress, usd_utils
from omni.cae.schema import cae as cae
from omni.cae.schema import viz as cae_viz
from omni.kit.app import get_app
from omni.kit.property.usd import RelationshipTargetPicker
from omni.usd import get_context
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdVol, Vt

from . import settings

logger = getLogger(__name__)


def create_material(mtl_name: str, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
    omni.kit.commands.execute(
        "CreateMdlMaterialPrim", mtl_name=mtl_name, mtl_url="cae/mdl/basic.mdl", mtl_path=str(path), stage=stage
    )
    return stage.GetPrimAtPath(path)


def get_surface_shader(material_prim, render_context) -> UsdShade.Shader:
    material = UsdShade.Material(material_prim)
    return material.ComputeSurfaceSource(render_context)[0]


def setup_mdl_colormap(material: Usd.Prim, colormap_path: str):
    if shader := get_surface_shader(material, "mdl"):
        shader.CreateInput("lut", Sdf.ValueTypeNames.Asset).Set(colormap_path)
    else:
        logger.error("No surface shader found for material %s", material.GetPath())


def bind_material(prim: Usd.Prim, material: Usd.Prim):
    omni.kit.commands.execute(
        "BindMaterial",
        prim_path=str(prim.GetPath()),
        material_path=str(material.GetPath()),
        strength=UsdShade.Tokens.weakerThanDescendants,
    )


class DatasetHelper:
    """Helper class to compute the bounding box of a dataset."""

    dataset_paths: list[str]
    nb_points: int
    nb_cells: int
    bounds: Gf.Range3d

    @staticmethod
    @progress.progress_context("Pre-fetching datasets")
    async def init(stage: Usd.Stage, dataset_paths: list[str]) -> "DatasetHelper":
        from . import settings

        helper = DatasetHelper()
        helper.dataset_paths = dataset_paths
        helper.nb_points = 0
        helper.nb_cells = 0
        helper.bounds = Gf.Range3d()

        device = settings.get_default_bounding_box_device()
        use_point_bounds = settings.get_default_bounding_box_use_point_bounds()

        for dataset_path in dataset_paths:
            dataset_prim = stage.GetPrimAtPath(dataset_path)
            if not dataset_prim:
                raise RuntimeError("DataSet prim is invalid!")

            try:
                if dataset_prim.IsA(cae.DataSet):
                    dataset = await cae_dav.get_dataset(
                        dataset_prim,
                        timeCode=Usd.TimeCode.EarliestTime(),
                        device=device,
                        needs_topology=not use_point_bounds,
                        needs_geometry=True,
                    )
                    if use_point_bounds:
                        with progress.ProgressContext("Executing DAV [compute point bounds]"):
                            ds_bounds = dataset.get_bounds()
                    else:
                        with progress.ProgressContext("Executing DAV [compute cell bounds]"):
                            ds_bounds = (
                                dataset.get_cell_bounds() if dataset.get_num_cells() > 0 else dataset.get_bounds()
                            )
                    helper.nb_points += dataset.get_num_points()
                    helper.nb_cells += dataset.get_num_cells()
                    helper.bounds.UnionWith(
                        Gf.Range3d(
                            (ds_bounds[0][0], ds_bounds[0][1], ds_bounds[0][2]),
                            (ds_bounds[1][0], ds_bounds[1][1], ds_bounds[1][2]),
                        )
                    )
                else:
                    helper.bounds.UnionWith(usd_utils.get_bounds(dataset_prim))

            except usd_utils.QuietableException as e:
                logger.warning(f"Failed to get dataset for {dataset_path}: {e}")
            except Exception as e:
                logger.exception(f"Failed to get dataset for {dataset_path}: {e}", exc_info=True)
        return helper


def create_unit_sphere(
    resolution: int, radius: float = 1.0, center: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a sphere mesh.

    Args:
        resolution: int, the number of divisions along latitude and longitude.
        radius: float, the radius of the sphere (default: 1.0 for unit sphere).
        center: tuple of (x, y, z), the center of the sphere (default: origin).

    Returns:
        Tuple of (vertices, faces):
        - vertices: np.ndarray, shape (n, 3), the coordinates of the points on the surface.
        - faces: np.ndarray, shape (m,), flattened indices of the vertices forming triangular faces.
    """
    # Create a grid in spherical coordinates
    theta = np.linspace(0, np.pi, resolution)  # latitude (0 to pi)
    phi = np.linspace(0, 2 * np.pi, resolution)  # longitude (0 to 2*pi)

    # Create a meshgrid for spherical coordinates
    theta, phi = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]

    # Stack the coordinates into a single array of vertices
    vertices = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

    # Create faces by connecting the vertices in the grid
    faces = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            # Vertices of each quad
            v1 = i * resolution + j
            v2 = v1 + 1
            v3 = v1 + resolution
            v4 = v3 + 1

            # Create two triangular faces for each quad
            faces.append([v1, v2, v4])
            faces.append([v1, v4, v3])

    faces = np.array(faces, dtype=np.int32).ravel()

    return vertices, faces


class CreateCaeVizMeshPrim(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, prim_type: str, prim_path: str, resolution: int = 16, boundable_paths: list[str] = []):
        self._prim_type = prim_type
        self._prim_path = prim_path
        self._resolution = resolution
        self._boundable_paths = boundable_paths

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        center = (0.0, 0.0, 0.0)
        radius = 1.0

        if self._boundable_paths:
            helper = await DatasetHelper.init(stage, self._boundable_paths)
            if not helper.bounds.IsEmpty():
                center = helper.bounds.GetMidpoint()
                radius = 0.05 * helper.bounds.GetSize().GetLength() / 2.0

        prim = UsdGeom.Mesh.Define(stage, self._prim_path)

        if self._prim_type == "UnitSphere":
            coords, faces = create_unit_sphere(self._resolution)
            prim.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords))
            prim.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
            prim.CreateFaceVertexCountsAttr().Set(
                Vt.IntArray.FromNumpy(np.ones(faces.shape[0] // 3, dtype=np.int32) * 3)
            )
            prim.CreateExtentAttr().Set([(-1, -1, -1), (1, 1, 1)])

            # create a transform to position the sphere at the center of the bounds and scale it to the radius
            UsdGeom.XformCommonAPI(prim).SetRotate((0, 0, 0))
            UsdGeom.XformCommonAPI(prim).SetScale((radius, radius, radius))
            UsdGeom.XformCommonAPI(prim).SetTranslate(center)
        else:
            raise RuntimeError(f"Unsupported prim type: {self._prim_type}")


class CreateCaeVizStreamlines(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str, type="standard"):
        self._dataset_path = dataset_path
        self._prim_path = prim_path
        self._type = type

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        primT = UsdGeom.BasisCurves.Define(stage, self._prim_path)
        primT.CreatePointsAttr().Set([(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)])
        primT.CreateCurveVertexCountsAttr().Set([4])
        primT.CreateExtentAttr().Set([(0.0, 0.0, 0.0), (0.3, 0.05, 0.05)])

        # We can no longer add Widths attr since we need to use primvar
        # primT.CreateWidthsAttr().Set([0.025])

        primT.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        primT.CreateBasisAttr().Set(UsdGeom.Tokens.bspline)
        primT.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
        primT.CreateWrapAttr().Set(UsdGeom.Tokens.pinned)
        pv_api = UsdGeom.PrimvarsAPI(primT.GetPrim())
        pv_api.CreatePrimvar("widths", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set([0.025] * 4)

        # Apply schema
        prim = primT.GetPrim()
        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.StreamlinesAPI.Apply(prim)
        if self._type == "nanovdb":
            cae_viz.DatasetVoxelizationAPI.Apply(prim, "source")
            helper = await DatasetHelper.init(stage, [self._dataset_path])
            if helper.nb_cells <= 0:
                cae_viz.DatasetGaussianSplattingAPI.Apply(prim, "source")
        cae_viz.DatasetSelectionAPI.Apply(prim, "source")
        cae_viz.DatasetSelectionAPI.Apply(prim, "seeds")
        cae_viz.DatasetTransformingAPI.Apply(prim, "seeds")
        cae_viz.FieldSelectionAPI.Apply(prim, "velocities")
        cae_viz.FieldSelectionAPI.Apply(prim, "colors")

        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        # velocities: we want to preserve the vector components
        cae_viz.FieldSelectionAPI(prim, "velocities").CreateModeAttr().Set(cae_viz.Tokens.unchanged)

        # setup auto-scaling for colors
        cae_viz.RescaleRangeAPI.Apply(prim, "colors")
        rescale_range_api = cae_viz.RescaleRangeAPI(prim, "colors")

        # setup material
        scalar_material = create_material(
            "ScalarColor", stage, primT.GetPath().AppendChild("Materials").AppendChild("ScalarColor")
        )
        setup_mdl_colormap(scalar_material, "cae/colormaps/gist_rainbow.png")

        if shader := get_surface_shader(scalar_material, "mdl"):
            attr = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
            attr.Set((0, -1))
            rescale_range_api.CreateIncludesRel().AddTarget(attr.GetAttr().GetPath())

        animated_material = create_material(
            "AnimatedStreaks", stage, primT.GetPath().AppendChild("Materials").AppendChild("AnimatedStreaks")
        )
        setup_mdl_colormap(animated_material, "cae/colormaps/gist_rainbow.png")
        if shader := get_surface_shader(animated_material, "mdl"):
            attr = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
            attr.Set((0, -1))
            rescale_range_api.CreateIncludesRel().AddTarget(attr.GetAttr().GetPath())

            attr = shader.CreateInput("enable_coloring", Sdf.ValueTypeNames.Bool)
            attr.Set(False)
            rescale_range_api.CreateEnableIncludesRel().AddTarget(attr.GetAttr().GetPath())

        # setup auto-scaling for widths
        cae_viz.FieldSelectionAPI.Apply(prim, "widths")
        cae_viz.FieldMappingAPI.Apply(prim, "widths")
        field_mapping_api = cae_viz.FieldMappingAPI(prim, "widths")
        field_mapping_api.GetDomainAttr().Set((0, -1))

        cae_viz.RescaleRangeAPI.Apply(prim, "widths")
        rescale_range_api = cae_viz.RescaleRangeAPI(prim, "widths")
        rescale_range_api.CreateIncludesRel().AddTarget(field_mapping_api.GetDomainAttr().GetPath())

        bind_material(primT, scalar_material)

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(primT.GetPath()))


class CreateCaeVizPoints(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        # create prim, so user has immediate feedback on the action.
        primT = UsdGeom.Points.Define(stage, self._prim_path)
        primT.CreatePointsAttr()
        prim = primT.GetPrim()

        # This may take a while if the dataset is large (without blocking the UI)
        # Compute bounding box to set a good default width
        # Using 0.5% of the diagonal as a reasonable default (same as bounding box)
        helper = await DatasetHelper.init(stage, [self._dataset_path])
        if not helper.bounds.IsEmpty():
            min_pt = np.array(helper.bounds.GetMin())
            max_pt = np.array(helper.bounds.GetMax())
            diagonal = np.linalg.norm(max_pt - min_pt)
            default_width = diagonal * 0.0005
        else:
            logger.warning(
                "Failed to compute bounding box for dataset %s, using default width of 1.0", self._dataset_path
            )
            default_width = 0.01

        # Apply schema
        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.PointsAPI.Apply(prim)
        cae_viz.DatasetSelectionAPI.Apply(prim, "source")

        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        material = create_material(
            "ScalarColor", stage, primT.GetPath().AppendChild("Materials").AppendChild("ScalarColor")
        )
        setup_mdl_colormap(material, "cae/colormaps/gist_rainbow.png")

        # setup colors field selection and rescale range APIs
        cae_viz.FieldSelectionAPI.Apply(prim, "colors")
        cae_viz.RescaleRangeAPI.Apply(prim, "colors")
        if shader := get_surface_shader(material, "mdl"):
            # we want to rescale "inputs:domain" attribute of the shader to the range of the colors field
            attr = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
            attr.Set((0, -1))
            cae_viz.RescaleRangeAPI(prim, "colors").CreateIncludesRel().AddTarget(attr.GetAttr().GetPath())

            attr = shader.CreateInput("enable_coloring", Sdf.ValueTypeNames.Bool)
            attr.Set(False)
            cae_viz.RescaleRangeAPI(prim, "colors").CreateEnableIncludesRel().AddTarget(attr.GetAttr().GetPath())

        points_api = cae_viz.PointsAPI(prim)
        points_api.CreateWidthAttr().Set(default_width)
        points_api.CreateUseCellPointsAttr().Set(helper.nb_cells > 0)

        # setup widths field selection, mapping and rescale range APIs
        cae_viz.FieldSelectionAPI.Apply(prim, "widths")
        cae_viz.FieldMappingAPI.Apply(prim, "widths")
        widths_mapping_api = cae_viz.FieldMappingAPI(prim, "widths")
        # Set a good default range: map field values to [default_width, 2*default_width]
        # This gives a reasonable starting point that can be adjusted by the user
        widths_mapping_api.CreateDomainAttr().Set((0, -1))
        widths_mapping_api.CreateRangeAttr().Set((default_width, default_width * 2.0))

        cae_viz.RescaleRangeAPI.Apply(prim, "widths")
        widths_rescale_range_api = cae_viz.RescaleRangeAPI(prim, "widths")
        # we want to rescale "cae:viz:{widths}:domain" attribute to the range of the widths field
        widths_rescale_range_api.CreateIncludesRel().AddTarget(widths_mapping_api.GetDomainAttr().GetPath())

        bind_material(primT, material)

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(primT.GetPath()))


class CreateCaeVizFaces(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str, material_path: str | None = None):
        self._dataset_path = dataset_path
        self._prim_path = prim_path
        self._material_path = material_path

    def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        primT = UsdGeom.Mesh.Define(stage, self._prim_path)
        primT.CreatePointsAttr()
        primT.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)

        # Apply schemas
        prim = primT.GetPrim()
        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.FacesAPI.Apply(prim)
        cae_viz.DatasetSelectionAPI.Apply(prim, "source")

        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        reusing_material = self._material_path is not None
        if reusing_material:
            material = stage.GetPrimAtPath(self._material_path)
        else:
            material = create_material(
                "ScalarColor", stage, primT.GetPath().AppendChild("Materials").AppendChild("ScalarColor")
            )
            setup_mdl_colormap(material, "cae/colormaps/gist_rainbow.png")

        # setup colors field selection API (always needed)
        cae_viz.FieldSelectionAPI.Apply(prim, "colors")
        cae_viz.RescaleRangeAPI.Apply(prim, "colors")

        if not reusing_material:
            # setup rescale range API and wire shader inputs only for the owning prim
            if shader := get_surface_shader(material, "mdl"):
                # we want to rescale "inputs:domain" attribute of the shader to the range of the colors field
                attr = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
                attr.Set((0, -1))
                cae_viz.RescaleRangeAPI(prim, "colors").CreateIncludesRel().AddTarget(attr.GetAttr().GetPath())

                attr = shader.CreateInput("enable_coloring", Sdf.ValueTypeNames.Bool)
                attr.Set(False)
                cae_viz.RescaleRangeAPI(prim, "colors").CreateEnableIncludesRel().AddTarget(attr.GetAttr().GetPath())

        bind_material(primT, material)

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(primT.GetPath()))


class CreateCaeVizGlyphs(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str, shape="Sphere"):
        self._dataset_path = dataset_path
        self._prim_path = prim_path
        self._shape = shape

    async def do(self):
        if self._shape == "Custom":
            self._custom_prototype_template = await self._select_custom_prototype_template()
            if not self._custom_prototype_template:
                logger.error("No custom prototype template selected. Cannot execute command 'CreateCaeVizGlyphs'")
                return

        stage: Usd.Stage = get_context().get_stage()
        primT = UsdGeom.PointInstancer.Define(stage, self._prim_path)
        primT.CreatePositionsAttr([])
        primT.CreateProtoIndicesAttr([0])
        primT.CreateOrientationsAttr([])
        primT.CreateScalesAttr([])

        # create prototypes under an "over" prim to skip the prototypes in standard stage navigation
        # refer to OpenUSD documentation for PointInstancer for more details.
        protosPrim = stage.OverridePrim(primT.GetPath().AppendChild("Prototypes"))
        primT.CreatePrototypesRel().SetTargets([self._create_prototype(protosPrim).GetPath()])

        prim = primT.GetPrim()
        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.GlyphsAPI.Apply(prim)

        helper = await DatasetHelper.init(stage, [self._dataset_path])
        glyphs_api = cae_viz.GlyphsAPI(prim)
        glyphs_api.CreateUseCellPointsAttr().Set(helper.nb_cells > 0)

        cae_viz.DatasetSelectionAPI.Apply(prim, "source")
        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        material = create_material(
            "ScalarColor", stage, primT.GetPath().AppendChild("Materials").AppendChild("ScalarColor")
        )
        setup_mdl_colormap(material, "cae/colormaps/gist_rainbow.png")

        cae_viz.FieldSelectionAPI.Apply(prim, "colors")
        cae_viz.RescaleRangeAPI.Apply(prim, "colors")
        if shader := get_surface_shader(material, "mdl"):
            # we want to rescale "inputs:domain" attribute of the shader to the range of the colors field
            attr = shader.CreateInput("domain", Sdf.ValueTypeNames.Float2)
            attr.Set((0, -1))
            cae_viz.RescaleRangeAPI(prim, "colors").CreateIncludesRel().AddTarget(attr.GetAttr().GetPath())

            attr = shader.CreateInput("enable_coloring", Sdf.ValueTypeNames.Bool)
            attr.Set(False)
            cae_viz.RescaleRangeAPI(prim, "colors").CreateEnableIncludesRel().AddTarget(attr.GetAttr().GetPath())

        cae_viz.FieldSelectionAPI.Apply(prim, "scales")
        cae_viz.FieldSelectionAPI.Apply(prim, "orientations")

        bind_material(primT, material)

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(primT.GetPath()))

    async def _select_custom_prototype_template(self) -> str:
        stage: Usd.Stage = get_context().get_stage()
        picker = RelationshipTargetPicker(
            stage,
            [],
            None,
            {
                "model_window": True,
                "target_name": "Prototype",
                "target_plural_name": "Prototypes",
            },
        )
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        picker.show(1, lambda paths: future.set_result(paths[0] if paths else None))
        path = await future
        logger.warning("selected prototype: %s", path)
        return path

    def _create_prototype(self, protosPrim: Usd.Prim) -> Usd.Prim:
        stage = protosPrim.GetStage()
        xform = UsdGeom.Xform.Define(stage, protosPrim.GetPath().AppendChild("Xform"))
        if self._shape == "Sphere":
            sphere = UsdGeom.Sphere.Define(stage, xform.GetPath().AppendChild("Sphere"))
            sphere.CreateRadiusAttr().Set(0.5)
        elif self._shape == "Cone":
            cone = UsdGeom.Cone.Define(stage, xform.GetPath().AppendChild("Cone"))
            cone.CreateHeightAttr().Set(1.0)
            cone.CreateRadiusAttr().Set(0.5)
            cone.CreateAxisAttr().Set(UsdGeom.Tokens.x)
        elif self._shape == "Arrow":
            arrowCylinder = UsdGeom.Cylinder.Define(stage, xform.GetPath().AppendChild("Cylinder"))
            arrowCylinder.CreateHeightAttr().Set(0.5)
            arrowCylinder.CreateRadiusAttr().Set(0.15)
            arrowCylinder.CreateAxisAttr().Set(UsdGeom.Tokens.x)
            arrowCylinder.AddTranslateOp().Set((0.25, 0, 0))
            arrowCone = UsdGeom.Cone.Define(stage, xform.GetPath().AppendChild("Cone"))
            arrowCone.CreateHeightAttr().Set(0.5)
            arrowCone.CreateRadiusAttr().Set(0.3)
            arrowCone.CreateAxisAttr().Set(UsdGeom.Tokens.x)
            arrowCone.AddTranslateOp().Set((0.75, 0, 0))
        elif self._shape == "Custom":
            if not self._custom_prototype_template:
                raise RuntimeError("No custom prototype template selected. Cannot execute command 'CreateCaeVizGlyphs'")
            template_prim = stage.GetPrimAtPath(self._custom_prototype_template)
            if not template_prim:
                raise RuntimeError(
                    f"Custom prototype template prim at path {self._custom_prototype_template} not found"
                )
            ref_prim = stage.DefinePrim(xform.GetPath().AppendChild(template_prim.GetName()))
            ref_prim.GetReferences().AddInternalReference(template_prim.GetPath())
        else:
            raise RuntimeError(f"Unsupported shape: {self._shape}")
        return xform


class CreateCaeVizVolume(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str, type="irregular"):
        self._dataset_path = dataset_path
        self._prim_path = prim_path
        self._type = type

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        primV = UsdVol.Volume.Define(stage, self._prim_path)

        # # Apply schema
        prim = primV.GetPrim()
        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.IndeXVolumeAPI.Apply(prim)

        helper = await DatasetHelper.init(stage, [self._dataset_path])

        if self._type == "irregular":
            nvindex_type = "irregular_volume"
            if helper.nb_cells <= 0:
                logger.warning("No cells found in dataset. Volume rendering may not be correct.")
        else:
            nvindex_type = "vdb"
            cae_viz.DatasetVoxelizationAPI.Apply(prim, "source")

            if helper.nb_cells <= 0:
                cae_viz.DatasetGaussianSplattingAPI.Apply(prim, "source")

        cae_viz.DatasetSelectionAPI.Apply(prim, "source")
        cae_viz.FieldSelectionAPI.Apply(prim, "colors")

        # Apply NVINDEX schema attributes (since IndeX doesn't have an explicit schema,
        # we have to add these manually).
        prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool).Set(True)
        prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool).Set(False)
        prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token).Set(nvindex_type)
        prim.CreateAttribute("outputs:volume", Sdf.ValueTypeNames.Token)

        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        # add importer.
        importer = stage.DefinePrim(prim.GetPath().AppendChild("Importer"), "FieldAsset")
        primV.CreateFieldRelationship("importer", importer.GetPath())

        # setup material.
        material: UsdShade.Material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("Material"))
        colormap = self.define_colormap(stage, material.GetPath().AppendChild("Colormap"))
        # colormap.GetReferences().AddInternalReference(cr_colormap.GetPath())

        # add field loader compute task.
        loader: UsdShade.Shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("DataLoader"))
        loader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.id)
        loader.CreateIdAttr().Set("nv::omni::cae::index.PythonComputeTask")
        loader.CreateInput("enabled", Sdf.ValueTypeNames.Bool).Set(False)
        material.CreateOutput("nvindex:compute", Sdf.ValueTypeNames.Token).ConnectToSource(
            loader.CreateOutput("compute", Sdf.ValueTypeNames.Token)
        )

        # add XAC shader
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("VolumeShader"))
        # shader.GetPrim().GetReferences().AddInternalReference(cr_shader.GetPath())
        shader.SetSourceAsset("cae/xac/basic.xac", "xac")
        shader.CreateInput("colormap", Sdf.ValueTypeNames.Token).ConnectToSource(
            colormap.GetAttribute("outputs:colormap").GetPath()
        )
        if attr := shader.CreateInput("voxel_size", Sdf.ValueTypeNames.Float3).GetAttr():
            attr.Set((1, 1, 1))
            attr.SetCustomDataByKey("nvindex.param", 0)
            attr.SetDocumentation("Specifies the voxel size for the volume (if applicable).")
        if attr := shader.CreateInput("mode", Sdf.ValueTypeNames.Int).GetAttr():
            attr.Set(0)
            attr.SetCustomDataByKey("nvindex.param", 1)
            attr.SetDocumentation("Specifies the mode (attribute type). 0 is float, 1 is vector (vec3f).")
        if attr := shader.CreateInput("attrib_idx", Sdf.ValueTypeNames.Int2).GetAttr():
            attr.Set(Gf.Vec2i(0, -1))
            attr.SetCustomDataByKey("nvindex.param", 2)
            attr.SetDocumentation("Attribute indices: (current, next). Set next to -1 to disable interpolation.")
        if attr := shader.CreateInput("time_codes", Sdf.ValueTypeNames.Float3).GetAttr():
            attr.Set((0.0, 0.0, 0.0))
            attr.SetCustomDataByKey("nvindex.param", 3)
            attr.SetDocumentation("Time codes: (current, next, raw) for interpolation.")

        material.CreateOutput("nvindex:volume", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.CreateOutput("volume", Sdf.ValueTypeNames.Token)
        )

        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)

        # based on "colors" field, set things up so that we rescale the range of the colormap automatically.
        cae_viz.RescaleRangeAPI.Apply(prim, "colors")
        rescale_range_api = cae_viz.RescaleRangeAPI(prim, "colors")
        rescale_range_api.CreateIncludesRel().SetTargets([colormap.GetAttribute("domain").GetPath()])

        # based on "colors" field, set things up so that we configure the XAC shader automatically.
        cae_viz.ConfigureXACShaderAPI.Apply(prim, "colors")
        configure_xac_shader_api = cae_viz.ConfigureXACShaderAPI(prim, "colors")
        configure_xac_shader_api.CreateVoxelSizeIncludesRel().SetTargets(
            [shader.GetPrim().GetAttribute("inputs:voxel_size").GetPath()]
        )
        configure_xac_shader_api.CreateSampleModeIncludesRel().SetTargets(
            [shader.GetPrim().GetAttribute("inputs:mode").GetPath()]
        )
        configure_xac_shader_api.CreateAttribIdxIncludesRel().SetTargets(
            [shader.GetPrim().GetAttribute("inputs:attrib_idx").GetPath()]
        )
        configure_xac_shader_api.CreateTimeCodesIncludesRel().SetTargets(
            [shader.GetPrim().GetAttribute("inputs:time_codes").GetPath()]
        )

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(primV.GetPath()))

    @staticmethod
    def define_colormap(stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        """Defines a colormap shader at the given path."""
        colormap_prim = stage.DefinePrim(path, "Colormap")
        colormap_prim.CreateAttribute("outputs:colormap", Sdf.ValueTypeNames.Token)
        colormap_prim.CreateAttribute("colormapSource", Sdf.ValueTypeNames.String).Set("rgbaPoints")
        colormap_prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set(
            [(0.2, 0.3, 0.8, 0), (0.86, 0.86, 0.86, 0.5), (0.7, 0.01, 0.14, 1.0)]
        )
        colormap_prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0, 0.5, 1.0])
        colormap_prim.CreateAttribute("domain", Sdf.ValueTypeNames.Float2).Set((0, -1))
        colormap_prim.CreateAttribute("domainBoundaryMode", Sdf.ValueTypeNames.Token).Set("clampToEdge")
        return colormap_prim

    @staticmethod
    def define_opaque_colormap(stage: Usd.Stage, path: Sdf.Path, based_on_prim: Usd.Prim = None) -> Usd.Prim:
        """Defines an opaque colormap shader at the given path."""
        colormap_prim = stage.DefinePrim(path, "Colormap")
        colormap_prim.CreateAttribute("outputs:colormap", Sdf.ValueTypeNames.Token)
        colormap_prim.CreateAttribute("colormapSource", Sdf.ValueTypeNames.String).Set("rgbaPoints")
        colormap_prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set(
            [(0.2, 0.3, 0.8, 1), (0.86, 0.86, 0.86, 1), (0.7, 0.01, 0.14, 1)]
        )
        colormap_prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0, 0.5, 1.0])
        colormap_prim.CreateAttribute("domain", Sdf.ValueTypeNames.Float2).Set((0, -1))
        colormap_prim.CreateAttribute("domainBoundaryMode", Sdf.ValueTypeNames.Token).Set("clampToTransparent")

        if based_on_prim and based_on_prim.GetTypeName() == "Colormap":
            rgbaPoints = based_on_prim.GetAttribute("rgbaPoints").Get()
            xPoints = based_on_prim.GetAttribute("xPoints").Get()
            # change all alpha values to 1.0
            rgbaPoints = [(rgba[0], rgba[1], rgba[2], 1.0) for rgba in rgbaPoints]
            colormap_prim.GetAttribute("rgbaPoints").Set(rgbaPoints)
            colormap_prim.GetAttribute("xPoints").Set(xPoints)
        return colormap_prim


class CreateCaeVizVolumeSlice(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, volume_path: str, prim_path: str, shape: str):
        self._volume_path = volume_path
        self._prim_path = prim_path
        self._shape = shape

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()
        volume_prim = stage.GetPrimAtPath(self._volume_path)
        if not volume_prim:
            raise RuntimeError("Volume prim is invalid!")
        if not volume_prim.IsA(UsdVol.Volume):
            raise RuntimeError("Volume prim is not a UsdVol.Volume!")

        volume = UsdVol.Volume(volume_prim)
        bounds = usd_utils.get_bounds(volume_prim)

        if bounds.IsEmpty():
            raise RuntimeError("Volume bounds are empty! Are you sure the volume is valid?")

        # Populate mesh points and face vertex counts based on the type.
        prim = await self._create_mesh(bounds, stage)
        if prim is None:
            return

        if prim.IsA(UsdGeom.Mesh):
            meshes = [UsdGeom.Mesh(prim)]
        else:
            meshes = [UsdGeom.Mesh(child.GetPrim()) for child in prim.GetChildren() if child.IsA(UsdGeom.Mesh)]

        if not meshes:
            raise RuntimeError("No meshes created! Are you sure the shape is valid?")

        # Create material for slices.
        material: UsdShade.Material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("Material"))
        colormap = CreateCaeVizVolume.define_opaque_colormap(
            stage,
            material.GetPath().AppendChild("Colormap"),
            based_on_prim=volume_prim.GetChild("Material").GetChild("Colormap"),
        )

        # Define shader for slices that connects to the volume and the colormap.
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("SliceShader"))
        shader.SetSourceAsset("cae/xac/basic_slice.xac", "xac")
        shader.CreateInput("slot_0", Sdf.ValueTypeNames.Token).ConnectToSource(
            volume.GetPrim().GetAttribute("outputs:volume").GetPath()
        )
        shader.CreateInput("slot_1", Sdf.ValueTypeNames.Token).ConnectToSource(
            colormap.GetAttribute("outputs:colormap").GetPath()
        )

        # now add attributes to the shader that will be used to configure the XAC shader.
        if attr := shader.CreateInput("voxel_size", Sdf.ValueTypeNames.Float3).GetAttr():
            attr.Set((1, 1, 1))
            attr.SetCustomDataByKey("nvindex.param", 0)
            attr.SetDocumentation("Specifies the voxel size for the volume (if applicable).")
        if attr := shader.CreateInput("mode", Sdf.ValueTypeNames.Int).GetAttr():
            attr.Set(0)
            attr.SetCustomDataByKey("nvindex.param", 1)
            attr.SetDocumentation("Specifies the mode (attribute type). 0 is float, 1 is vector (vec3f).")
        if attr := shader.CreateInput("attrib_idx", Sdf.ValueTypeNames.Int2).GetAttr():
            attr.Set(Gf.Vec2i(0, -1))
            attr.SetCustomDataByKey("nvindex.param", 2)
            attr.SetDocumentation("Attribute indices: (current, next). Set next to -1 to disable interpolation.")
        if attr := shader.CreateInput("time_codes", Sdf.ValueTypeNames.Float3).GetAttr():
            attr.Set((0.0, 0.0, 0.0))
            attr.SetCustomDataByKey("nvindex.param", 3)
            attr.SetDocumentation("Time codes: (current, next, raw) for interpolation.")

        material.CreateSurfaceOutput("nvindex").ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )

        if volume_prim.HasAPI(cae_viz.RescaleRangeAPI, "colors"):
            rescale_range_api = cae_viz.RescaleRangeAPI(volume_prim, "colors")
            rescale_range_api.CreateIncludesRel().AddTarget(colormap.GetAttribute("domain").GetPath())
        else:
            logger.error("No RescaleRangeAPI found for colors field! Cannot auto-scale the colormap.")

        if volume_prim.HasAPI(cae_viz.ConfigureXACShaderAPI, "colors"):
            # Configure XAC shader for slices.
            configure_xac_shader_api = cae_viz.ConfigureXACShaderAPI(volume_prim, "colors")
            configure_xac_shader_api.CreateVoxelSizeIncludesRel().AddTarget(
                shader.GetPrim().GetAttribute("inputs:voxel_size").GetPath()
            )
            configure_xac_shader_api.CreateSampleModeIncludesRel().AddTarget(
                shader.GetPrim().GetAttribute("inputs:mode").GetPath()
            )
            configure_xac_shader_api.CreateAttribIdxIncludesRel().AddTarget(
                shader.GetPrim().GetAttribute("inputs:attrib_idx").GetPath()
            )
            configure_xac_shader_api.CreateTimeCodesIncludesRel().AddTarget(
                shader.GetPrim().GetAttribute("inputs:time_codes").GetPath()
            )
        else:
            logger.error("No ConfigureXACShaderAPI found for colors field! Cannot configure the XAC shader.")

        cae_viz.OperatorDependenciesAPI.Apply(volume_prim)
        operator_dependencies_api = cae_viz.OperatorDependenciesAPI(volume_prim)

        for mesh in meshes:
            mesh_prim = mesh.GetPrim()
            mesh_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            mesh_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("plane")

            operator_dependencies_api.CreateDependentsRel().AddTarget(mesh_prim.GetPath())

            UsdShade.MaterialBindingAPI.Apply(mesh_prim)
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

        logger.info("created '%s''", str(prim.GetPath()))

    async def _create_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage) -> Usd.Prim:
        if self._shape == "Plane":
            return self._create_plane_mesh(bounds, stage)
        elif self._shape == "Bi-Plane":
            return self._create_bi_plane_mesh(bounds, stage)
        elif self._shape == "Tri-Plane":
            return self._create_tri_plane_mesh(bounds, stage)
        elif self._shape == "Sphere":
            return self._create_sphere_mesh(bounds, stage)
        elif self._shape == "Custom":
            return await self._create_custom_mesh(bounds, stage)
        else:
            raise RuntimeError(f"Unsupported shape: {self._shape}")

    def _create_plane_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage) -> Usd.Prim:
        """create an X plane spanning the bounds of the volume centered at the center of the bounds."""
        center = bounds.GetMidpoint()
        width = bounds.GetSize()[0]
        length = bounds.GetSize()[2]
        points = [
            (center[0] - width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] + length / 2),
            (center[0] - width / 2, center[1], center[2] + length / 2),
        ]
        faces = [0, 1, 2, 3]
        primT = UsdGeom.Mesh.Define(stage, self._prim_path)
        primT.CreatePointsAttr().Set(points)
        primT.CreateFaceVertexCountsAttr().Set([4])
        primT.CreateFaceVertexIndicesAttr().Set(faces)
        primT.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        return primT.GetPrim()

    def _create_bi_plane_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage) -> Usd.Prim:
        """Create a bi-plane i.e. X and Y planes mesh spanning the bounds of the volume centered at the center of the bounds.
        Create as two UsdGeomMesh prims nested under an Xform"""
        center = bounds.GetMidpoint()
        width = bounds.GetSize()[0]
        length = bounds.GetSize()[2]
        height = bounds.GetSize()[1]

        # Plane 1: XY plane at center Z (perpendicular to Z axis)
        points1 = [
            (center[0] - width / 2, center[1] - height / 2, center[2]),
            (center[0] + width / 2, center[1] - height / 2, center[2]),
            (center[0] + width / 2, center[1] + height / 2, center[2]),
            (center[0] - width / 2, center[1] + height / 2, center[2]),
        ]
        faces1 = [0, 1, 2, 3]

        # Plane 2: XZ plane at center Y (perpendicular to Y axis)
        points2 = [
            (center[0] - width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] + length / 2),
            (center[0] - width / 2, center[1], center[2] + length / 2),
        ]
        faces2 = [0, 1, 2, 3]

        xform = UsdGeom.Xform.Define(stage, self._prim_path)
        xform_prim = xform.GetPrim()
        plane1 = UsdGeom.Mesh.Define(stage, xform_prim.GetPath().AppendChild("Z_Plane"))
        plane1.CreatePointsAttr().Set(points1)
        plane1.CreateFaceVertexCountsAttr().Set([4])
        plane1.CreateFaceVertexIndicesAttr().Set(faces1)
        plane1.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])

        plane2 = UsdGeom.Mesh.Define(stage, xform_prim.GetPath().AppendChild("X_Plane"))
        plane2.CreatePointsAttr().Set(points2)
        plane2.CreateFaceVertexCountsAttr().Set([4])
        plane2.CreateFaceVertexIndicesAttr().Set(faces2)
        plane2.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        return xform_prim

    def _create_tri_plane_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage) -> Usd.Prim:
        """Create a tri-plane i.e. X, Y and Z planes mesh spanning the bounds of the volume centered at the center of the bounds.
        Create as three UsdGeomMesh prims nested under an Xform"""
        center = bounds.GetMidpoint()
        width = bounds.GetSize()[0]
        length = bounds.GetSize()[2]
        height = bounds.GetSize()[1]

        # Plane 1: XY plane at center Z (perpendicular to Z axis)
        points1 = [
            (center[0] - width / 2, center[1] - height / 2, center[2]),
            (center[0] + width / 2, center[1] - height / 2, center[2]),
            (center[0] + width / 2, center[1] + height / 2, center[2]),
            (center[0] - width / 2, center[1] + height / 2, center[2]),
        ]
        faces1 = [0, 1, 2, 3]

        # Plane 2: XZ plane at center Y (perpendicular to Y axis)
        points2 = [
            (center[0] - width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] - length / 2),
            (center[0] + width / 2, center[1], center[2] + length / 2),
            (center[0] - width / 2, center[1], center[2] + length / 2),
        ]
        faces2 = [0, 1, 2, 3]

        # Plane 3: YZ plane at center X (perpendicular to X axis)
        points3 = [
            (center[0], center[1] - height / 2, center[2] - length / 2),
            (center[0], center[1] + height / 2, center[2] - length / 2),
            (center[0], center[1] + height / 2, center[2] + length / 2),
            (center[0], center[1] - height / 2, center[2] + length / 2),
        ]
        faces3 = [0, 1, 2, 3]

        xform = UsdGeom.Xform.Define(stage, self._prim_path)
        xform_prim = xform.GetPrim()
        plane1 = UsdGeom.Mesh.Define(stage, xform_prim.GetPath().AppendChild("Z_Plane"))
        plane1.CreatePointsAttr().Set(points1)
        plane1.CreateFaceVertexCountsAttr().Set([4])
        plane1.CreateFaceVertexIndicesAttr().Set(faces1)
        plane1.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        plane2 = UsdGeom.Mesh.Define(stage, xform_prim.GetPath().AppendChild("X_Plane"))
        plane2.CreatePointsAttr().Set(points2)
        plane2.CreateFaceVertexCountsAttr().Set([4])
        plane2.CreateFaceVertexIndicesAttr().Set(faces2)
        plane2.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        plane3 = UsdGeom.Mesh.Define(stage, xform_prim.GetPath().AppendChild("Y_Plane"))
        plane3.CreatePointsAttr().Set(points3)
        plane3.CreateFaceVertexCountsAttr().Set([4])
        plane3.CreateFaceVertexIndicesAttr().Set(faces3)
        plane3.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        return xform_prim

    def _create_sphere_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage, resolution: int = 32) -> Usd.Prim:
        """Create a sphere mesh centered at the center of the bounds.
        The radius is set to half the minimum dimension of the bounds.

        Args:
            bounds: Bounding box to center the sphere in
            stage: USD stage to create the mesh in
            resolution: Number of latitude/longitude divisions (default: 32)
        """
        center = bounds.GetMidpoint()
        size = bounds.GetSize()
        # Use half of the minimum dimension as radius to ensure sphere fits in bounds
        radius = min(size[0], size[1], size[2]) / 2.0

        # Generate sphere vertices and faces using the utility function
        vertices, faces = create_unit_sphere(resolution, radius=radius, center=tuple(center))

        # Convert faces to face vertex counts (all triangles)
        num_triangles = len(faces) // 3
        face_vertex_counts = [3] * num_triangles

        # Create the mesh
        sphere = UsdGeom.Mesh.Define(stage, self._prim_path)
        sphere.CreatePointsAttr().Set([tuple(v) for v in vertices])
        sphere.CreateFaceVertexCountsAttr().Set(face_vertex_counts)
        sphere.CreateFaceVertexIndicesAttr().Set(faces.tolist())
        sphere.CreateExtentAttr().Set([tuple(bounds.GetMin()), tuple(bounds.GetMax())])
        return sphere.GetPrim()

    async def _create_custom_mesh(self, bounds: Gf.Range3d, stage: Usd.Stage) -> Usd.Prim:
        """Prompt the user to select a custom mesh prim from the stage.

        Args:
            bounds: Bounding box (not used, kept for API consistency)
            stage: USD stage to search for mesh prims

        Returns:
            The selected mesh prim

        Raises:
            RuntimeError: If no prim is selected or the selected prim is not a UsdGeom.Mesh
        """
        # Use the relationship target picker to let the user select a prim
        picker = RelationshipTargetPicker(
            stage,
            [],
            None,
            {
                "model_window": True,
                "target_name": "Custom Mesh",
                "target_plural_name": "Custom Meshes",
            },
        )
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        picker.show(1, lambda paths: future.set_result(paths[0] if paths else None))
        selected_path = await future

        if not selected_path:
            raise RuntimeError("No custom mesh selected. Cannot execute command 'CreateCaeVizSeeds'")

        logger.info("selected custom mesh: %s", selected_path)

        # Verify the selected prim is a UsdGeom.Mesh
        selected_prim = stage.GetPrimAtPath(selected_path)
        if not selected_prim:
            raise RuntimeError(f"Custom mesh prim at path {selected_path} not found")

        if not selected_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"Selected prim at path {selected_path} is not a UsdGeom.Mesh (type: {selected_prim.GetTypeName()})"
            )

        primT = UsdGeom.Mesh.Define(stage, self._prim_path)
        primT.GetPrim().GetReferences().AddInternalReference(selected_prim.GetPath())
        primT.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        return primT.GetPrim()


class CreateCaeVizBoundingBox(omni.kit.commands.Command):
    """
    Unlike other visualization commands, this doesn't create an operator,
    instead directly computes the bounding box of the datasets and sets up a BasisCurve to render it.
    """

    def __init__(self, dataset_paths: list[str], prim_path: str):
        self._dataset_paths = dataset_paths
        self._prim_path = prim_path

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        helper = await DatasetHelper.init(stage, self._dataset_paths)
        if helper.bounds.IsEmpty():
            logger.error("Failed to compute bounding box for datasets %s", self._dataset_paths)
            return

        # Generate the 8 vertices of the bounding box using numpy
        min_pt = np.array(helper.bounds.GetMin())
        max_pt = np.array(helper.bounds.GetMax())

        # Define the 8 corners of the bounding box efficiently
        # Using binary representation: bit 0 = x, bit 1 = y, bit 2 = z
        corners = np.array(
            [
                [min_pt[0], min_pt[1], min_pt[2]],  # 0: 000
                [max_pt[0], min_pt[1], min_pt[2]],  # 1: 001
                [max_pt[0], max_pt[1], min_pt[2]],  # 2: 011
                [min_pt[0], max_pt[1], min_pt[2]],  # 3: 010
                [min_pt[0], min_pt[1], max_pt[2]],  # 4: 100
                [max_pt[0], min_pt[1], max_pt[2]],  # 5: 101
                [max_pt[0], max_pt[1], max_pt[2]],  # 6: 111
                [min_pt[0], max_pt[1], max_pt[2]],  # 7: 110
            ]
        )

        # Define the 12 edges of the bounding box
        # Each row represents an edge connecting two corner indices
        edge_indices = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # bottom face (z = min)
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # top face (z = max)
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # vertical edges
            ],
            dtype=np.int32,
        )

        # Reshape to get points for all edges: (12, 2, 3) -> (24, 3)
        edge_points = corners[edge_indices].reshape(-1, 3)

        primT = UsdGeom.BasisCurves.Define(stage, self._prim_path)
        primT.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(edge_points))
        # Each edge is a separate curve with 2 vertices (linear)
        primT.CreateCurveVertexCountsAttr().Set([2] * len(edge_indices))
        primT.CreateExtentAttr().Set([tuple(min_pt), tuple(max_pt)])

        # Set curve type to linear for straight edges
        primT.CreateTypeAttr().Set(UsdGeom.Tokens.linear)

        # Set a good width based on the diagonal of the bounding box
        # Using 0.2% of the diagonal as a reasonable default
        diagonal = np.linalg.norm(max_pt - min_pt)
        width = diagonal * 0.005
        primT.CreateWidthsAttr().Set([width])
        primT.SetWidthsInterpolation(UsdGeom.Tokens.constant)

        # setup pivot so the rotations and scales happen around the center of the bounding box
        pivot_pos = tuple(helper.bounds.GetMidpoint())
        xformable = UsdGeom.Xformable(primT.GetPrim())
        xformable.AddTranslateOp(opSuffix="pivot").Set(pivot_pos)
        xformable.AddTranslateOp().Set((0, 0, 0))
        xformable.AddRotateXYZOp()
        xformable.AddScaleOp().Set((1, 1, 1))
        xformable.AddTranslateOp(opSuffix="pivot", isInverseOp=True)


class CreateCaeVizFlowEnvironment(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, prim_path: str, layer_number: int):
        self._prim_path = prim_path
        self._layer_number = layer_number

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()
        scope = UsdGeom.Scope.Define(stage, self._prim_path)
        prim = scope.GetPrim()

        simulate_prim = omni.kit.commands.execute(
            "FlowCreatePrim", prim_path=self._prim_path + "/flowSimulate", type_name="FlowSimulate"
        )[1]
        offscreen_prim = omni.kit.commands.execute(
            "FlowCreatePrim", prim_path=self._prim_path + "/flowOffscreen", type_name="FlowOffscreen"
        )[1]
        render_prim = omni.kit.commands.execute(
            "FlowCreatePrim", prim_path=self._prim_path + "/flowRender", type_name="FlowRender"
        )[1]

        # wait till Flow has time to setup prims and properties fo the the prim types we just created.
        # Flow does this asynchronously, so we need to wait for it to complete.
        await get_app().next_update_async()

        simulate_prim.GetAttribute("layer").Set(self._layer_number)
        offscreen_prim.GetAttribute("layer").Set(self._layer_number)
        render_prim.GetAttribute("layer").Set(self._layer_number)

        if advection_prim := simulate_prim.GetChild("advection"):
            # Buoyancy affects the up-ward force applied to smoke particles
            # as they burn. We don't want smoke to be influenced by buoyancy, so we set it to 0.
            advection_prim.GetProperty("buoyancyPerTemp").Set(0.0)

            # Burn-Per-Temp is the amount of burn per temperature unit.
            # Setting it to max to ensure full burn regardless of temperature.
            advection_prim.GetProperty("burnPerTemp").Set(1000.0)

            # Fuel-Per-Temp is the amount of fuel per temperature unit.
            # Setting it to 0 to ensure the fuel once, injected, never depletes.
            # It may seem like that's what we want, but we need to kill the smoke eventually otherwise
            # it may linger too long when the emitter is moved. Setting it to 1.0 seems to be a good compromise.
            advection_prim.GetProperty("fuelPerBurn").Set(1.0)

            # ignition temperature is the temperature at which the fuel will ignite.
            # Setting it to 0 to ensure the fuel will ignite at any temperature.
            advection_prim.GetProperty("ignitionTemp").Set(0.0)
        else:
            logger.error(f"Missing {simulate_prim.GetPath().AppendChild('advection')} prim")

        if ray_march_prim := render_prim.GetChild("rayMarch"):
            # Attenuation is the amount of light that is absorbed by the scene. Default is very low.
            # Setting it to 3 makes it more opaque.
            ray_march_prim.GetProperty("attenuation").Set(3.0)
        else:
            logger.error(f"Missing {render_prim.GetPath().AppendChild('rayMarch')} prim")

        # setup colormap
        if colormap_prim := offscreen_prim.GetChild("colormap"):
            colormap_prim.GetProperty("colorScale").Set(10.0)
            colormap_prim.GetProperty("resolution").Set(256)
            colormap_prim.GetProperty("rgbaPoints").Set(
                [(0.2, 0.3, 0.8, 0.3), (1.0, 1.0, 0.0, 0.3), (0.7, 0.01, 0.14, 1.0)]
            )
            colormap_prim.GetProperty("xPoints").Set([0.0001, 0.5, 1.0])
            colormap_prim.GetProperty("colorScalePoints").Set([0.5, 1.0, 1.0])

        logger.info("created '%s''", str(prim.GetPath()))


class CreateCaeVizFlowDataSetEmitter(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, dataset_path: str, prim_path: str, layer_number: int, simulation_prim: Usd.Prim):
        self._dataset_path = dataset_path
        self._prim_path = prim_path
        self._layer_number = layer_number
        self._simulation_prim = simulation_prim

    def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        prim = omni.kit.commands.execute("FlowCreatePrim", prim_path=self._prim_path, type_name="FlowEmitterNanoVdb")[1]
        assert prim is not None, "Failed to create FlowEmitterNanoVdb prim"
        prim.GetAttribute("layer").Set(self._layer_number)

        # disable allocation in Flow.
        prim.GetAttribute("allocationScale").Set(0.0)

        # we don't what this to introduce any default data on any of the channels.
        prim.GetAttribute("burn").Set(0.0)
        prim.GetAttribute("divergence").Set(0.0)
        prim.GetAttribute("fuel").Set(0.0)
        prim.GetAttribute("smoke").Set(0.0)
        prim.GetAttribute("temperature").Set(0.0)
        prim.GetAttribute("velocity").Set((0.0, 0.0, 0.0))

        # Reset scale factors to 1.0 to avoid artifical scaling by default.
        # User should manually override, if needed.
        prim.GetAttribute("burnScale").Set(1.0)
        prim.GetAttribute("divergenceScale").Set(1.0)
        prim.GetAttribute("fuelScale").Set(1.0)
        prim.GetAttribute("smokeScale").Set(1.0)
        prim.GetAttribute("temperatureScale").Set(1.0)
        prim.GetAttribute("velocityScale").Set(1.0)

        # Set couple rate so that the simulation value simmediately snap to the
        # dataset values.
        prim.GetAttribute("coupleRateBurn").Set(0.0)
        prim.GetAttribute("coupleRateDivergence").Set(0.0)
        prim.GetAttribute("coupleRateFuel").Set(0.0)
        prim.GetAttribute("coupleRateSmoke").Set(0.0)
        prim.GetAttribute("coupleRateTemperature").Set(0.0)
        prim.GetAttribute("coupleRateVelocity").Set(0.0)

        # this guarantees velocity at the end of the frame will match dataset velocity
        prim.GetAttribute("applyPostPressure").Set(True)

        cae_viz.OperatorAPI.Apply(prim)
        cae_viz.DatasetSelectionAPI.Apply(prim, "source")
        cae_viz.ConfigureFlowEnvironmentAPI.Apply(prim, "source")

        source_selection_api = cae_viz.DatasetSelectionAPI(prim, "source")
        source_selection_api.CreateTargetRel().SetTargets({self._dataset_path})

        cae_viz.DatasetVoxelizationAPI.Apply(prim, "source")
        # inflate the bounds by 10% to ensure the dataset is padded with zeros
        # so that the volume streamlines are not truncated at the edges.
        cae_viz.DatasetVoxelizationAPI(prim, "source").CreateInflateBoundsAttr().Set(10.0)
        cae_viz.FieldSelectionAPI.Apply(prim, "velocities")
        cae_viz.FieldSelectionAPI.Apply(prim, "temperatures")
        cae_viz.RescaleRangeAPI.Apply(prim, "temperatures")

        rescale_range_api = cae_viz.RescaleRangeAPI(prim, "temperatures")
        if ray_march_prim := self._simulation_prim.GetPrimAtPath(("flowRender/rayMarch")):
            rescale_range_api.CreateMinIncludesRel().AddTarget(ray_march_prim.GetAttribute("colormapXMin").GetPath())
            rescale_range_api.CreateMaxIncludesRel().AddTarget(ray_march_prim.GetAttribute("colormapXMax").GetPath())
        else:
            logger.error(f"Missing {self._simulation_prim.GetPath().AppendChild('flowRender/rayMarch')} prim")

        if simulation_prim := self._simulation_prim.GetPrimAtPath("flowSimulate"):
            configure_flow_environment_api = cae_viz.ConfigureFlowEnvironmentAPI(prim, "source")
            configure_flow_environment_api.CreateDensityCellSizeIncludesRel().AddTarget(
                simulation_prim.GetAttribute("densityCellSize").GetPath()
            )
        else:
            logger.error(f"Missing {self._simulation_prim.GetPath().AppendChild('flowSimulate')} prim")

        # finally, set the enabled state based on settings
        cae_viz.OperatorAPI(prim).CreateEnabledAttr().Set(settings.get_default_operator_enabled())
        logger.info("created '%s''", str(prim.GetPath()))


class CreateCaeVizFlowSmokeInjector(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(
        self, boundable_paths: list[str], prim_path: str, layer_number: int, mode: str, simulation_prim: Usd.Prim
    ):
        self._boundable_paths = boundable_paths
        self._prim_path = prim_path
        self._layer_number = layer_number
        self._mode = mode
        self._simulation_prim = simulation_prim

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()
        helper = await DatasetHelper.init(stage, self._boundable_paths)
        bounds = helper.bounds
        if not bounds.IsEmpty():
            center = bounds.GetMidpoint()
            diagonal = np.linalg.norm(bounds.GetSize()).tolist()
            scale = np.array([diagonal, diagonal, diagonal]) * 0.02
        else:
            center = np.array([0.0, 0.0, 0.0])
            scale = np.array([1.0, 1.0, 1.0])

        if self._mode == "sphere":
            # Create a unit sphere mesh with resolution 16
            coords, faces = create_unit_sphere(16)
            shape = UsdGeom.Mesh.Define(stage, self._prim_path)
            shape.CreateExtentAttr().Set([(-1, -1, -1), (1, 1, 1)])
            shape.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords))
            shape.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces))
            shape.CreateFaceVertexCountsAttr().Set(
                Vt.IntArray.FromNumpy(np.ones(faces.shape[0] // 3, dtype=np.int32) * 3)
            )

            emitter = stage.DefinePrim(shape.GetPath().AppendChild("EmitterSphere"), "FlowEmitterSphere")
            emitter.GetAttribute("layer").Set(self._layer_number)
            emitter.GetAttribute("radius").Set(1.0)
            emitter.GetAttribute("radiusIsWorldSpace").Set(False)

        else:
            raise RuntimeError(f"Unsupported mode: {self._mode}")

        xform_api = UsdGeom.XformCommonAPI(shape)
        xform_api.SetTranslate(center)
        xform_api.SetRotate((0, 0, 0))
        xform_api.SetScale(scale.tolist())

        emitter.GetAttribute("coupleRateVelocity").Set(0.0)
        emitter.GetAttribute("coupleRateFuel").Set(120.0)
        emitter.GetAttribute("coupleRateTemperature").Set(1.0)
        emitter.GetAttribute("temperature").Set(1.0)
        emitter.GetAttribute("fuel").Set(1.0)

        logger.info("created '%s''", str(emitter.GetPath()))


class CreateCaeVizFlowBoundaryEmitter(omni.kit.commands.Command):
    ns: str = "cae:viz"

    def __init__(self, boundable_paths: list[str], prim_path: str, layer_number: int):
        self._boundable_paths = boundable_paths
        self._prim_path = prim_path
        self._layer_number = layer_number

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()

        # Compute bounding box of the dataset
        helper = await DatasetHelper.init(stage, self._boundable_paths)
        if helper.bounds.IsEmpty():
            logger.error("Failed to compute bounding box for boundable %s", self._boundable_paths)
            return

        min_pt = np.array(helper.bounds.GetMin())
        max_pt = np.array(helper.bounds.GetMax())
        center = np.array(helper.bounds.GetMidpoint())
        size = max_pt - min_pt

        # Nominal thickness for boundary emitters (2% of smallest dimension)
        thickness = np.min(size) * 0.1

        # Create parent Xform to hold all boundary emitters
        parent_xform = UsdGeom.Xform.Define(stage, self._prim_path)
        parent_xform.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # Define the 6 faces: -X, +X, -Y, +Y, -Z, +Z
        # Make faces 10% larger along face dimensions to ensure corners overlap and avoid leaks
        overlap_factor = 1.1
        faces = [
            {
                "name": "NegX",
                "center": (min_pt[0] - thickness / 2, center[1], center[2]),
                "half_size": (thickness / 2, size[1] / 2 * overlap_factor, size[2] / 2 * overlap_factor),
            },
            {
                "name": "PosX",
                "center": (max_pt[0] + thickness / 2, center[1], center[2]),
                "half_size": (thickness / 2, size[1] / 2 * overlap_factor, size[2] / 2 * overlap_factor),
            },
            {
                "name": "NegY",
                "center": (center[0], min_pt[1] - thickness / 2, center[2]),
                "half_size": (size[0] / 2 * overlap_factor, thickness / 2, size[2] / 2 * overlap_factor),
            },
            {
                "name": "PosY",
                "center": (center[0], max_pt[1] + thickness / 2, center[2]),
                "half_size": (size[0] / 2 * overlap_factor, thickness / 2, size[2] / 2 * overlap_factor),
            },
            {
                "name": "NegZ",
                "center": (center[0], center[1], min_pt[2] - thickness / 2),
                "half_size": (size[0] / 2 * overlap_factor, size[1] / 2 * overlap_factor, thickness / 2),
            },
            {
                "name": "PosZ",
                "center": (center[0], center[1], max_pt[2] + thickness / 2),
                "half_size": (size[0] / 2 * overlap_factor, size[1] / 2 * overlap_factor, thickness / 2),
            },
        ]

        for face in faces:
            # Create a cube mesh for visualization
            cube_path = parent_xform.GetPath().AppendChild(f"Boundary_{face['name']}")
            cube = UsdGeom.Cube.Define(stage, cube_path)
            cube.CreateSizeAttr().Set(1.0)  # Unit cube

            # Transform the cube to match the face position and size
            xform_api = UsdGeom.XformCommonAPI(cube)
            xform_api.SetTranslate(face["center"])
            xform_api.SetScale((face["half_size"][0] * 2, face["half_size"][1] * 2, face["half_size"][2] * 2))

            # Create FlowEmitterBox under the cube
            emitter_path = cube.GetPath().AppendChild("EmitterBox")
            emitter = omni.kit.commands.execute(
                "FlowCreatePrim", prim_path=str(emitter_path), type_name="FlowEmitterBox"
            )[1]

            if emitter:
                emitter.GetAttribute("allocationScale").Set(0.0)
                emitter.GetAttribute("applyPostPressure").Set(True)
                emitter.GetAttribute("layer").Set(self._layer_number)
                emitter.GetAttribute("halfSize").Set(Gf.Vec3f(0.5, 0.5, 0.5))  # Half size in local space

                # Set velocity to zero (solid boundary)
                emitter.GetAttribute("velocity").Set((0.0, 0.0, 0.0))
                emitter.GetAttribute("coupleRateVelocity").Set(1200)

                # Disable other fields
                for field in ["fuel", "temperature", "smoke", "burn", "divergence"]:
                    emitter.GetAttribute(f"coupleRate{field[0].upper()}{field[1:]}").Set(1200.0)
                    emitter.GetAttribute(f"{field}").Set(0.0)

            else:
                logger.error(f"Failed to create FlowEmitterBox at {emitter_path}")

        logger.info("created '%s' with 6 boundary emitters", str(parent_xform.GetPath()))
