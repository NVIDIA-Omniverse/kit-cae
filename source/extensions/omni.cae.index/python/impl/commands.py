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
    "CreateCaeIndeXSlice",
    "CreateCaeIndeXVolume",
    "CreateCaeNanoVdbIndeXVolume",
]

from logging import getLogger

from omni.cae.data import settings
from omni.kit.commands import Command
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdVol

from .algorithms import IrregularVolumeHelper, NanoVdbHelper, SliceMaterialHelper, VolumeMaterialHelper

logger = getLogger(__name__)


class CreateCaeIndeXSlice(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()

        path = Sdf.Path(self._prim_path)
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()

        prim.AddAppliedSchema("CaeIndeXSliceAPI")
        prim.CreateRelationship("omni:cae:index:slice:dataset", custom=False).SetTargets({self._dataset_path})
        prim.CreateRelationship("omni:cae:index:slice:field", custom=False).SetTargets({})

        mesh = UsdGeom.Mesh.Define(stage, path.AppendChild("Plane"))
        mesh_prim = mesh.GetPrim()

        # setup mesh for IndeX slice rendering
        mesh_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(1)
        mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        mesh_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("plane")

        points = [(0, -0.5, -0.5), (0, -0.5, 0.5), (0, 0.5, 0.5), (0, 0.5, -0.5)]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        volume_prim = IrregularVolumeHelper.define_volume(stage, path.AppendChild("Volume"))
        volume = UsdVol.Volume(volume_prim)
        volume.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        material: UsdShade.Material = UsdShade.Material.Define(stage, path.AppendChild("Material"))
        SliceMaterialHelper.define_shader(material, volume)

        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
        logger.info("created '%s''", str(mesh_prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None


class CreateCaeIndeXNanoVdbSlice(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        ns = "omni:cae:index:slice"
        stage: Usd.Stage = get_context().get_stage()

        path = Sdf.Path(self._prim_path)
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()

        prim.AddAppliedSchema("CaeIndeXNanoVdbSliceAPI")
        prim.CreateRelationship(f"{ns}:dataset", custom=False).SetTargets({self._dataset_path})
        prim.CreateRelationship(f"{ns}:field", custom=False).SetTargets({})
        prim.CreateRelationship(f"{ns}:field", custom=False).SetTargets({})  # this is necessary with 106.4
        prim.CreateRelationship(f"{ns}:roi", custom=False).SetTargets({})  # this is necessary with 106.4
        prim.CreateAttribute(f"{ns}:maxResolution", Sdf.ValueTypeNames.Int, custom=False).Set(
            settings.get_default_max_voxel_grid_resolution()
        )

        mesh = UsdGeom.Mesh.Define(stage, path.AppendChild("Plane"))
        mesh_prim = mesh.GetPrim()

        # setup mesh for IndeX slice rendering
        mesh_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(1)
        mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        mesh_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("plane")

        points = [(0, -0.5, -0.5), (0, -0.5, 0.5), (0, 0.5, 0.5), (0, 0.5, -0.5)]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        volume_prim = NanoVdbHelper.define_volume(stage, path.AppendChild("Volume"))
        volume = UsdVol.Volume(volume_prim)
        volume.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

        # technically, we should not need the volume shader since we don't intend to render the volume,
        # however, the compute task doesn't seem to execute without it.
        VolumeMaterialHelper.define_shader(NanoVdbHelper.get_material(volume_prim))

        # setup material for IndeX slice rendering.
        material: UsdShade.Material = UsdShade.Material.Define(stage, path.AppendChild("Material"))
        SliceMaterialHelper.define_shader(material, volume)

        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)

        logger.info("created '%s''", str(prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None


class CreateCaeIndeXVolumeSlice(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._volume_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()
        volume_prim = stage.GetPrimAtPath(self._volume_path)
        if not volume_prim:
            raise RuntimeError("Volume prim is invalid!")
        if not volume_prim.IsA(UsdVol.Volume):
            raise RuntimeError("Volume prim is not a UsdVol.Volume!")

        volume = UsdVol.Volume(volume_prim)

        ns = "omni:cae:index:slice"
        path = Sdf.Path(self._prim_path)
        prim = UsdGeom.Scope.Define(stage, path).GetPrim()
        prim.AddAppliedSchema("CaeIndeXVolumeSliceAPI")
        prim.CreateRelationship(f"{ns}:volume", custom=False).SetTargets({self._volume_path})

        mesh = UsdGeom.Mesh.Define(stage, path.AppendChild("Plane"))
        mesh_prim = mesh.GetPrim()

        # setup mesh for IndeX slice rendering
        mesh_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(1)
        mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
        mesh_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("plane")

        points = [(0, -0.5, -0.5), (0, -0.5, 0.5), (0, 0.5, 0.5), (0, 0.5, -0.5)]
        mesh.CreatePointsAttr().Set(points)
        mesh.CreateFaceVertexCountsAttr().Set([4])
        mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)

        material: UsdShade.Material = UsdShade.Material.Define(stage, path.AppendChild("Material"))
        SliceMaterialHelper.define_shader(material, volume)

        UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
        logger.info("created '%s''", str(mesh_prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None


class CreateCaeIndeXVolume(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()
        dataset_prim = stage.GetPrimAtPath(self._dataset_path)
        if not dataset_prim:
            raise RuntimeError("DataSet prim is invalid!")

        prim = IrregularVolumeHelper.define_volume(stage, self._prim_path)

        ns = "omni:cae:index:volume"
        prim.AddAppliedSchema("CaeIndeXVolumeAPI")
        prim.CreateRelationship(f"{ns}:dataset", custom=False).SetTargets({dataset_prim.GetPath()})
        prim.CreateRelationship(f"{ns}:field", custom=False).SetTargets(
            {}
        )  # this is necessary with 106.4 for change tracking to work

        # add materials.
        material: UsdShade.Material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("Material"))
        VolumeMaterialHelper.define_shader(material)

        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)
        logger.info("created '%s''", str(prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None


class CreateCaeNanoVdbIndeXVolume(Command):

    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        stage: Usd.Stage = get_context().get_stage()
        dataset_prim = stage.GetPrimAtPath(self._dataset_path)
        if not dataset_prim:
            raise RuntimeError("DataSet prim is invalid!")

        ns = "omni:cae:index:nvdb"
        prim = NanoVdbHelper.define_volume(stage, self._prim_path)
        prim.AddAppliedSchema("CaeIndeXNanoVdbVolumeAPI")
        prim.CreateRelationship(f"{ns}:dataset", custom=False).SetTargets({dataset_prim.GetPath()})
        prim.CreateRelationship(f"{ns}:field", custom=False).SetTargets({})  # this is necessary with 106.4
        prim.CreateRelationship(f"{ns}:roi", custom=False).SetTargets({})  # this is necessary with 106.4
        prim.CreateAttribute(f"{ns}:maxResolution", Sdf.ValueTypeNames.Int, custom=False).Set(
            settings.get_default_max_voxel_grid_resolution()
        )
        prim.CreateAttribute(f"{ns}:temporalInterpolation", Sdf.ValueTypeNames.Bool, custom=False).Set(False)

        material: UsdShade.Material = NanoVdbHelper.get_material(prim)
        VolumeMaterialHelper.define_shader(material)

        logger.info("created '%s''", str(prim.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None
