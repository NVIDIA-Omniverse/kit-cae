# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = [
    "CreateCaeFlowDataSetEmitter",
    "CreateCaeFlowEnvironment",
    "CreateCaeFlowSmoker",
]

from logging import getLogger

from omni.cae.data import settings
from omni.kit.commands import Command
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom

logger = getLogger(__name__)


class CreateCaeFlowEnvironment(Command):

    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()
        path = self._prim_path

        scope = stage.DefinePrim(path, "CaeFlowEnvironment")

        simulate = stage.DefinePrim(scope.GetPath().AppendChild("flowSimulate"), "FlowSimulate")
        simulate.CreateAttribute("autoCellSize", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
        simulate.CreateAttribute("densityCellSize", Sdf.ValueTypeNames.Float, custom=False).Set(
            0.01
        )  # we need to make this dependent on data bounds

        advection = stage.DefinePrim(simulate.GetPath().AppendChild("advection"), "FlowAdvectionCombustionParams")
        advection.CreateAttribute("gravity", Sdf.ValueTypeNames.Float3, custom=False).Set((0, 0, 0))

        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("velocity"), "FlowAdvectionChannelParams")
        channelParams.CreateAttribute("damping", Sdf.ValueTypeNames.Float, custom=False).Set(0.01)
        channelParams.CreateAttribute("fade", Sdf.ValueTypeNames.Float, custom=False).Set(1.0)
        channelParams.CreateAttribute("secondOrderBlendFactor", Sdf.ValueTypeNames.Float, custom=False).Set(0.5)

        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("divergence"), "FlowAdvectionChannelParams")
        channelParams.CreateAttribute("damping", Sdf.ValueTypeNames.Float, custom=False).Set(0.01)
        channelParams.CreateAttribute("fade", Sdf.ValueTypeNames.Float, custom=False).Set(1.0)
        channelParams.CreateAttribute("secondOrderBlendFactor", Sdf.ValueTypeNames.Float, custom=False).Set(0.5)

        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("temperature"), "FlowAdvectionChannelParams")
        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("fuel"), "FlowAdvectionChannelParams")
        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("burn"), "FlowAdvectionChannelParams")

        channelParams = stage.DefinePrim(advection.GetPath().AppendChild("smoke"), "FlowAdvectionChannelParams")
        channelParams.CreateAttribute("damping", Sdf.ValueTypeNames.Float, custom=False).Set(0.3)
        channelParams.CreateAttribute("fade", Sdf.ValueTypeNames.Float, custom=False).Set(0.65)

        vorticityParams = stage.DefinePrim(simulate.GetPath().AppendChild("vorticity"), "FlowVorticityParams")
        # we don't want too much turbulence, but a little bit helps with the effect.
        vorticityParams.CreateAttribute("forceScale", Sdf.ValueTypeNames.Float, custom=False).Set(0.1)

        stage.DefinePrim(simulate.GetPath().AppendChild("pressure"), "FlowPressureParams")
        stage.DefinePrim(simulate.GetPath().AppendChild("summaryAllocate"), "FlowSummaryAllocateParams")
        stage.DefinePrim(simulate.GetPath().AppendChild("nanoVdbExport"), "FlowSparseNanoVdbExportParams")

        offscreen = stage.DefinePrim(scope.GetPath().AppendChild("flowOffscreen"), "FlowOffscreen")
        stage.DefinePrim(offscreen.GetPath().AppendChild("shadow"), "FlowShadowParams")
        self._create_colormap(stage, offscreen.GetPath().AppendChild("colormap"))
        stage.DefinePrim(offscreen.GetPath().AppendChild("debugVolume"), "FlowDebugVolumeParams")

        render = stage.DefinePrim(scope.GetPath().AppendChild("flowRender"), "FlowRender")
        marchParams = stage.DefinePrim(render.GetPath().AppendChild("rayMarch"), "FlowRayMarchParams")
        marchParams.CreateAttribute("attenuation", Sdf.ValueTypeNames.Float, custom=False).Set(3.0)
        stage.DefinePrim(marchParams.GetPath().AppendChild("cloud"), "FlowRayMarchCloudParams")

        logger.info("created '%s''", scope)

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None

    def _create_colormap(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        colormap_prim = stage.DefinePrim(path, "FlowRayMarchColormapParams")
        colormap_prim.CreateAttribute("colorScale", Sdf.ValueTypeNames.Float, custom=False).Set(10.0)
        colormap_prim.CreateAttribute("resolution", Sdf.ValueTypeNames.Int, custom=False).Set(256)
        colormap_prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array, custom=False).Set(
            [(0.2, 0.3, 0.8, 0.3), (1.0, 1.0, 0.0, 0.3), (0.7, 0.01, 0.14, 1.0)]
        )
        colormap_prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray, custom=False).Set([0.05, 0.5, 1.0])
        colormap_prim.CreateAttribute("colorScalePoints", Sdf.ValueTypeNames.FloatArray, custom=False).Set(
            [0.5, 1.0, 1.0]
        )
        return colormap_prim


class CreateCaeFlowDataSetEmitter(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()
        dataset_prim = stage.GetPrimAtPath(self._dataset_path)
        if not dataset_prim:
            raise RuntimeError("DataSet prim is invalid!")

        prim = stage.DefinePrim(self._prim_path, "FlowEmitterNanoVdb")
        # turn off fuel so this emitter does not emit smoke.
        prim.CreateAttribute("fuel", Sdf.ValueTypeNames.Float, custom=False).Set(0.0)
        prim.CreateAttribute("smoke", Sdf.ValueTypeNames.Float, custom=False).Set(0.0)
        # when set to 0, I get some giggle in the volume streamlines otherwise they just stop are seem frozen
        prim.CreateAttribute("allocationScale", Sdf.ValueTypeNames.Float, custom=False).Set(0.0)
        ns = "omni:cae:flow:emitter"
        prim.AddAppliedSchema("CaeFlowDataSetEmitterAPI")
        prim.CreateRelationship(f"{ns}:dataset", custom=False).SetTargets({dataset_prim.GetPath()})
        prim.CreateRelationship(f"{ns}:velocity", custom=False).SetTargets({})
        prim.CreateRelationship(f"{ns}:colors", custom=False).SetTargets({})
        prim.CreateAttribute(f"{ns}:maxResolution", Sdf.ValueTypeNames.Int, custom=False).Set(
            settings.get_default_max_voxel_grid_resolution()
        )
        logger.info("created '%s''", str(prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None


class CreateCaeFlowSmoker(Command):

    def __init__(self, prim_path: str):
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()

        # create a xform
        xform = UsdGeom.Xform.Define(stage, self._prim_path)

        # scale to a decent shape so it's visible
        xformAPI = UsdGeom.XformCommonAPI(xform)
        xformAPI.SetTranslate((0, 0, 0))
        xformAPI.SetRotate((0, 0, 0))
        xformAPI.SetScale((1, 1, 1))

        # create nested xform emitter
        emitter = stage.DefinePrim(xform.GetPath().AppendChild("EmitterSphere"), "FlowEmitterSphere")
        # move the emitter to center of the base-plate for the cone.
        emitter.CreateAttribute("position", Sdf.ValueTypeNames.Float3, custom=False).Set((0.0, 0, 0))
        emitter.CreateAttribute("velocity", Sdf.ValueTypeNames.Float3, custom=False).Set((0.0, 0, 10.0))
        emitter.CreateAttribute("velocityIsWorldSpace", Sdf.ValueTypeNames.Bool, custom=False).Set(True)
        # we don't want the probe to induce any flow by itself.
        emitter.CreateAttribute("coupleRateVelocity", Sdf.ValueTypeNames.Float, custom=False).Set(0.0)

        emitter.CreateAttribute("radius", Sdf.ValueTypeNames.Float, custom=False).Set(0.5)
        # emitter.CreateAttribute("velocity", Sdf.ValueTypeNames.Float3, custom=False).Set((0, 0, 0))
        emitter.CreateAttribute("coupleRateFuel", Sdf.ValueTypeNames.Float, custom=False).Set(50.0)
        # this ensures that we scaling of cone will also scale the
        emitter.CreateAttribute("radiusIsWorldSpace", Sdf.ValueTypeNames.Bool, custom=False).Set(False)

        logger.info("created '%s''", str(xform.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None
