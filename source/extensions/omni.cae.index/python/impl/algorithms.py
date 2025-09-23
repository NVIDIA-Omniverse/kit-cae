# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

__all__ = ["Volume", "NanoVdbVolume"]

import asyncio
from logging import getLogger
from typing import Any

# from pxr import Sdf, Usd, UsdShade, UsdVol, Vt
from omni.cae.algorithms.core import Algorithm
from omni.cae.data import IJKExtents, cache, usd_utils
from omni.cae.data.commands import ComputeIJKExtents, Voxelize
from omni.cae.data.impl.command_types import ComputeBounds
from omni.kit.async_engine import run_coroutine
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdVol, Vt

logger = getLogger(__name__)


_execution_tag: float = 1.0


class IrregularVolumeHelper:

    IMPORTER_RELATION_NAME = "importer"

    @staticmethod
    def get_dataset(prim: Usd.Prim) -> Usd.Prim:
        """Returns the dataset prim for the given irregular volume prim."""
        return usd_utils.get_target_prim(prim, "omni:cae:index:volume:dataset", quiet=True)

    @staticmethod
    def get_fields(prim: Usd.Prim) -> list[Usd.Prim]:
        """Returns the field prims for the given irregular volume prim."""
        return usd_utils.get_target_prims(prim, "omni:cae:index:volume:field", quiet=True)

    @staticmethod
    def define_volume(stage: Usd.Stage, path: str) -> Usd.Prim:
        """Creates an irregular volume prim with the given path."""
        volume = UsdVol.Volume.Define(stage, path)
        volume_prim = volume.GetPrim()

        # setup empty xform (adding xform later on causes random issues)
        xformAPI = UsdGeom.XformCommonAPI(volume_prim)
        xformAPI.SetTranslate((0, 0, 0))
        xformAPI.SetRotate((0, 0, 0))
        xformAPI.SetScale((1, 1, 1))

        # skip RTX rendering for this volume
        volume_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(1)
        # skip IndeX too until it's all setup for rendering to avoid premature errors
        volume_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(0)
        # set the type to irregular volume
        volume_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("irregular_volume")
        # create the volume output attribute
        volume_prim.CreateAttribute("outputs:volume", Sdf.ValueTypeNames.Token, custom=False)

        # the irregular volume importer is specified as a FieldAsset prim
        fieldPrim = stage.DefinePrim(volume_prim.GetPath().AppendChild("Importer"), "FieldAsset")
        # fieldPrim.SetCustomDataByKey("nvindex.importerSettings", {
        #     "importer": Vt.Token("nv::omni::cae::index.CaeDataSetImporter"),
        #     # other properties are setup later since they depend on the dataset / field selection.
        # })
        volume.CreateFieldRelationship(IrregularVolumeHelper.IMPORTER_RELATION_NAME, fieldPrim.GetPath())
        return volume_prim

    @staticmethod
    def get_importer(prim: Usd.Prim) -> Usd.Prim:
        """Returns the FieldAsset importer prim for the given volume prim."""
        return usd_utils.get_target_prim(prim, f"field:{IrregularVolumeHelper.IMPORTER_RELATION_NAME}")

    @staticmethod
    def update(
        volume_prim: Usd.Prim,
        dataset_prim: Usd.Prim,
        field_prims: list[Usd.Prim],
        bounds: Gf.Range3d,
        timeCode: Usd.TimeCode,
    ) -> None:
        """Updates the irregular volume prim with the given dataset and field names."""

        global _execution_tag
        volume = UsdVol.Volume(volume_prim)

        # update volume extents; this is necessary for IndeX
        volume.CreateExtentAttr().Set([bounds.GetMin(), bounds.GetMax()])

        # now, update importer
        importer = IrregularVolumeHelper.get_importer(volume_prim)
        settings = {
            "importer": Vt.Token("nv::omni::cae::index.CaeDataSetImporter"),
            "mesh": Vt.Token(str(dataset_prim.GetPath())),
            "timeCode": Vt.Double(timeCode.GetValue()),
        }
        for idx, field in enumerate(field_prims):
            settings[f"field:{idx}"] = Vt.Token(usd_utils.get_field_name(dataset_prim, field))
        settings["nb_fields"] = Vt.Int(len(field_prims))
        logger.info("Setting importer settings for %s: %s", volume_prim.GetPath(), settings)
        importer.SetCustomDataByKey("nvindex.importerSettings", settings)

        # enable IndeX compositing since now we have the importer set.
        volume_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(1)

        # bug in IndeX results in importer not being updated when the importerSettings change.
        # so we have to set a dummy attribute to force the update.
        volume_prim.CreateAttribute("nvindex:timestep", Sdf.ValueTypeNames.Double, custom=True).Set(_execution_tag)
        _execution_tag += 1.0


class NanoVdbHelper:
    """Helper class for NanoVDB volumes."""

    IMPORTER_PATH = "Importer"
    IMPORTER_RELATION_NAME = "importer"
    MATERIAL_PATH = "Material"
    DATA_LOADER_PATH = "DataLoader"

    @staticmethod
    def get_dataset(prim: Usd.Prim) -> Usd.Prim:
        """Returns the dataset prim for the given NanoVDB volume prim."""
        return usd_utils.get_target_prim(prim, "omni:cae:index:nvdb:dataset", quiet=True)

    @staticmethod
    def get_fields(prim: Usd.Prim) -> list[Usd.Prim]:
        """Returns the field prims for the given NanoVDB volume prim."""
        return usd_utils.get_target_prims(prim, "omni:cae:index:nvdb:field", quiet=True)

    @staticmethod
    def define_volume(stage: Usd.Stage, path: str) -> Usd.Prim:
        """Creates a NanoVDB volume prim with the given path.
        Unlike IrregularVolumeHelper, this also setups a Material since a material is needed to define the
        IndeX compute task that imports the NanoVDB dataset."""
        volume = UsdVol.Volume.Define(stage, path)
        volume_prim = volume.GetPrim()

        # setup empty xform (adding xform later on causes random issues)
        # xformAPI = UsdGeom.XformCommonAPI(volume_prim)
        # xformAPI.SetTranslate((0, 0, 0))
        # xformAPI.SetRotate((0, 0, 0))
        # xformAPI.SetScale((1, 1, 1))

        volume_prim.CreateAttribute("omni:rtx:skip", Sdf.ValueTypeNames.Bool, custom=True).Set(1)
        volume_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(0)
        volume_prim.CreateAttribute("nvindex:type", Sdf.ValueTypeNames.Token, custom=True).Set("vdb")
        volume_prim.CreateAttribute("outputs:volume", Sdf.ValueTypeNames.Token, custom=False)

        # the NanoVDB importer is specified as a FieldAsset prim; this is a empty importer.
        importer_prim = stage.DefinePrim(volume_prim.GetPath().AppendChild(NanoVdbHelper.IMPORTER_PATH), "FieldAsset")

        importer_prim.SetCustomDataByKey(
            "nvindex.importerSettings",
            {
                "importer": Vt.Token("nv::index::plugin::openvdb_integration.NanoVDB_empty_init_importer"),
                "nb_attributes": Vt.Int(1),
            },
        )
        volume.CreateFieldRelationship(NanoVdbHelper.IMPORTER_RELATION_NAME, importer_prim.GetPath())

        # For NanoVDB, we need to create compute task to import the data.
        compute_material: UsdShade.Material = UsdShade.Material.Define(
            stage, volume_prim.GetPath().AppendChild(NanoVdbHelper.MATERIAL_PATH)
        )
        loader: UsdShade.Shader = UsdShade.Shader.Define(
            stage, compute_material.GetPath().AppendChild(NanoVdbHelper.DATA_LOADER_PATH)
        )
        loader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.id)
        loader.CreateIdAttr().Set("nv::omni::cae::index.CaeDataSetNanoVdbFetchTechnique")
        loader.CreateInput("enabled", Sdf.ValueTypeNames.Bool).Set(False)
        compute_material.CreateOutput("nvindex:compute", Sdf.ValueTypeNames.Token).ConnectToSource(
            loader.CreateOutput("compute", Sdf.ValueTypeNames.Token)
        )

        UsdShade.MaterialBindingAPI.Apply(volume_prim)
        UsdShade.MaterialBindingAPI(volume_prim).Bind(compute_material)
        return volume_prim

    @staticmethod
    def get_material(prim: Usd.Prim) -> UsdShade.Material:
        """Returns the Material prim for the given NanoVDB volume prim."""
        if material_prim := prim.GetPrimAtPath(NanoVdbHelper.MATERIAL_PATH):
            return UsdShade.Material(material_prim)
        else:
            raise RuntimeError(f"Material not found for NanoVDB volume prim {prim.GetPath()}")

    @staticmethod
    def get_loader(prim: Usd.Prim) -> UsdShade.Shader:
        """Returns the DataLoader shader for the given NanoVDB volume prim."""
        if loader_prim := prim.GetPrimAtPath(f"{NanoVdbHelper.MATERIAL_PATH}/{NanoVdbHelper.DATA_LOADER_PATH}"):
            return UsdShade.Shader(loader_prim)
        else:
            raise RuntimeError(f"DataLoader not found for NanoVDB volume prim {prim.GetPath()}")

    def update(volume_prim: Usd.Prim, cache_key: str, enable_interpolation: bool, ijk_extents: IJKExtents) -> None:
        """Updates the NanoVDB volume prim with the given cache key and extents."""
        global _execution_tag

        volume: UsdVol.Volume = UsdVol.Volume(volume_prim)

        # update volume extents; this is necessary for IndeX
        volume.CreateExtentAttr().Set([ijk_extents.min, ijk_extents.max])

        # update importer
        if importer := usd_utils.get_target_prim(
            volume_prim, f"field:{NanoVdbHelper.IMPORTER_RELATION_NAME}", quiet=True
        ):
            importer.SetCustomDataByKey(
                "nvindex.importerSettings",
                {
                    "importer": Vt.Token("nv::index::plugin::openvdb_integration.NanoVDB_empty_init_importer"),
                    "nb_attributes": Vt.Int(2 if enable_interpolation else 1),
                },
            )
        else:
            raise RuntimeError(f"Importer not found for volume prim {volume_prim.GetPath()}")

        # update nanovdb loader compute task.
        if loader := NanoVdbHelper.get_loader(volume_prim):
            loader.CreateInput("prim", Sdf.ValueTypeNames.String).Set(str(volume_prim.GetPath()))
            loader.CreateInput("cache_key", Sdf.ValueTypeNames.String).Set(cache_key)
            loader.CreateInput("enabled", Sdf.ValueTypeNames.Bool).Set(True)
            loader.CreateInput("enable_interpolation", Sdf.ValueTypeNames.Bool).Set(enable_interpolation)
            # this tag is just used to force a recompute of the compute task.
            loader.CreateInput("execution_tag", Sdf.ValueTypeNames.Int).Set(int(_execution_tag))
            _execution_tag += 1
        else:
            raise RuntimeError(f"DataLoader not found for volume prim {volume_prim.GetPath()}")

        # enable IndeX compositing since now we have the importer set.
        volume_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(1)

    def update_scale(volume_prim: Usd.Prim, scale: Gf.Vec3d) -> None:
        """Updates the scale of the NanoVDB volume prim based on the extents."""
        # if not volume_prim.GetAttribute("xformOp:scale").HasAuthoredValue():
        if True:
            logger.info("Updating scale of %s to %s", volume_prim.GetPath(), scale)
            xformAPI = UsdGeom.XformCommonAPI(volume_prim)
            xformAPI.SetTranslate((0, 0, 0))
            xformAPI.SetRotate((0, 0, 0))
            xformAPI.SetScale(Gf.Vec3f(scale[0], scale[1], scale[2]))


class ColormapHelper:
    """Helper class for defining colormap shaders."""

    @staticmethod
    def define_colormap(stage: Usd.Stage, path: Sdf.Path) -> UsdShade.Shader:
        """Defines a colormap shader at the given path."""
        colormap_prim = stage.DefinePrim(path, "Colormap")
        colormap_prim.CreateAttribute("outputs:colormap", Sdf.ValueTypeNames.Token)
        colormap_prim.CreateAttribute("colormapSource", Sdf.ValueTypeNames.String).Set("rgbaPoints")
        colormap_prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set(
            [(0.2, 0.3, 0.8, 0), (0.86, 0.86, 0.86, 0.5), (0.7, 0.01, 0.14, 1.0)]
        )
        colormap_prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0, 0.5, 1.0])
        colormap_prim.CreateAttribute("domain", Sdf.ValueTypeNames.Float2).Set((0, 1))
        colormap_prim.CreateAttribute("domainBoundaryMode", Sdf.ValueTypeNames.Token).Set("clampToEdge")
        return colormap_prim

    @staticmethod
    def define_opaque_colormap(stage: Usd.Stage, path: Sdf.Path) -> UsdShade.Shader:
        """Defines an opaque colormap shader at the given path."""
        colormap_prim = stage.DefinePrim(path, "Colormap")
        colormap_prim.CreateAttribute("outputs:colormap", Sdf.ValueTypeNames.Token)
        colormap_prim.CreateAttribute("colormapSource", Sdf.ValueTypeNames.String).Set("rgbaPoints")
        colormap_prim.CreateAttribute("rgbaPoints", Sdf.ValueTypeNames.Float4Array).Set(
            [(0.2, 0.3, 0.8, 1), (0.86, 0.86, 0.86, 1), (0.7, 0.01, 0.14, 1)]
        )
        colormap_prim.CreateAttribute("xPoints", Sdf.ValueTypeNames.FloatArray).Set([0, 0.5, 1.0])
        colormap_prim.CreateAttribute("domain", Sdf.ValueTypeNames.Float2).Set((0, 1))
        colormap_prim.CreateAttribute("domainBoundaryMode", Sdf.ValueTypeNames.Token).Set("clampToTransparent")
        return colormap_prim


class VolumeMaterialHelper:

    COLORMAP_PATH = "Colormap"
    SHADER_PATH = "VolumeShader"

    @staticmethod
    def define_shader_input(
        shader: UsdShade.Shader,
        name: str,
        value_type: Sdf.ValueTypeName,
        default_value: Any,
        nvindex_param: int = None,
        doc: str = None,
    ) -> UsdShade.Input:
        """Creates a shader input with the given name, value type, and default value."""
        input_attr = shader.CreateInput(name, value_type)
        input_attr.Set(default_value)
        if nvindex_param is not None:
            input_attr.GetAttr().SetCustomDataByKey("nvindex.param", nvindex_param)
        if doc is not None:
            input_attr.GetAttr().SetDocumentation(doc)
        return input_attr

    @staticmethod
    def define_shader(material: UsdShade.Material) -> UsdShade.Shader:
        """Defines a shader for the given material."""
        stage = material.GetPrim().GetStage()
        colormap = ColormapHelper.define_colormap(
            stage, material.GetPrim().GetPath().AppendChild(VolumeMaterialHelper.COLORMAP_PATH)
        )

        shader = UsdShade.Shader.Define(
            stage, material.GetPrim().GetPath().AppendChild(VolumeMaterialHelper.SHADER_PATH)
        )
        shader.SetSourceAsset("xac/cae_volume.xac", "xac")
        shader.CreateInput("colormap", Sdf.ValueTypeNames.Token).ConnectToSource(
            colormap.GetAttribute("outputs:colormap").GetPath()
        )

        VolumeMaterialHelper.define_shader_input(
            shader,
            "fraction",
            Sdf.ValueTypeNames.Float,
            0.0,
            nvindex_param=0,
            doc="LERP fraction between two time samples",
        )
        VolumeMaterialHelper.define_shader_input(
            shader, "mode", Sdf.ValueTypeNames.Int, 0, nvindex_param=1, doc="Sampler data mode (0: float, 1: float3)"
        )

        material.CreateOutput("nvindex:volume", Sdf.ValueTypeNames.Token).ConnectToSource(
            shader.CreateOutput("volume", Sdf.ValueTypeNames.Token)
        )
        return shader


class SliceMaterialHelper:
    """Helper class for defining slice materials."""

    COLORMAP_PATH = "Colormap"
    SHADER_PATH = "SliceShader"

    @staticmethod
    def define_shader(material: UsdShade.Material, volume: UsdVol.Volume) -> UsdShade.Shader:
        """Defines a shader for the given material."""
        stage = material.GetPrim().GetStage()
        shader = UsdShade.Shader.Define(
            stage, material.GetPrim().GetPath().AppendChild(SliceMaterialHelper.SHADER_PATH)
        )
        shader.SetSourceAsset("xac/cae_slice.xac", "xac")
        material.CreateSurfaceOutput("nvindex").ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )

        colormap = ColormapHelper.define_opaque_colormap(
            stage, material.GetPath().AppendChild(SliceMaterialHelper.COLORMAP_PATH)
        )
        shader.CreateInput("slot_0", Sdf.ValueTypeNames.Token).ConnectToSource(
            volume.GetPrim().GetAttribute("outputs:volume").GetPath()
        )
        shader.CreateInput("slot_1", Sdf.ValueTypeNames.Token).ConnectToSource(
            colormap.GetAttribute("outputs:colormap").GetPath()
        )

        return shader


class SliceHelper:
    """Helper class for updating slice mesh transform."""

    @staticmethod
    def update_xform(mesh_prim: Usd.Prim, bds: Gf.Range3d) -> None:
        """Updates the slice mesh transform based on the bounds."""

        if (
            not mesh_prim.GetAttribute("xformOp:scale").HasAuthoredValue()
            and not mesh_prim.GetAttribute("xformOp:translate").HasAuthoredValue()
            and not mesh_prim.GetAttribute("xformOp:rotateXYZ").HasAuthoredValue()
        ):
            size = bds.GetSize()
            xformAPI = UsdGeom.XformCommonAPI(mesh_prim)
            xformAPI.SetTranslate(bds.GetMidpoint())
            xformAPI.SetRotate((0, 0, 0))
            xformAPI.SetScale((size[0], size[1], size[2]))


class Slice(Algorithm):
    def __init__(self, prim: Usd.Prim) -> None:
        super().__init__(prim, ["CaeIndeXSliceAPI"])

    async def execute_impl(self, timeCode: Usd.TimeCode) -> None:
        prim: Usd.Prim = self.prim
        dataset_prim = usd_utils.get_target_prim(prim, "omni:cae:index:slice:dataset")
        field_prims: list[Usd.Prim] = usd_utils.get_target_prims(prim, "omni:cae:index:slice:field")

        mesh_prim = prim.GetPrimAtPath("Plane")
        if not mesh_prim:
            raise RuntimeError(f"Plane prims not found for slice {prim.GetPath()}")

        volume_prim: Usd.Prim = prim.GetPrimAtPath("Volume")
        if not volume_prim:
            raise RuntimeError(f"Volume prim not found for slice {prim.GetPath()}")

        bds = await ComputeBounds.invoke(dataset_prim, Usd.TimeCode.EarliestTime())
        SliceHelper.update_xform(mesh_prim, bds)

        with self.edit_context:
            IrregularVolumeHelper.update(volume_prim, dataset_prim, field_prims, bds, timeCode)

            mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(1)

        # Let's do the colormap default setting happening in the root layer
        # so it can be overridden by the user and just acts as a default value.
        if len(field_prims) == 1:
            await usd_utils.apply_dataset_colormap_range(
                field_prims[0], prim.GetPrimAtPath("Material/Colormap"), timeCode
            )


class NanoVdbSlice(Algorithm):
    _xform_ops = [
        "omni:fabric:localMatrix",
    ]

    def __init__(self, prim: Usd.Prim) -> None:
        super().__init__(prim, ["CaeIndeXNanoVdbSliceAPI"])
        self._rel_tracker: usd_utils.ChangeTracker = usd_utils.ChangeTracker(self.stage)
        self._force_update: asyncio.Task = None
        self._cache_key: str = f"nvdb_slice: {id(self)}"

        # xform ops
        for p in self._xform_ops:
            self._rel_tracker.TrackAttribute(p)

    def release(self):
        self.clear_cache()
        return super().release()

    def clear_cache(self):
        cache.remove(self._cache_key)

    def uses_temporal_cache(self) -> bool:
        # this algorithm uses a temporal cache, thus no need to call execute on previously executed timestep.
        return True

    def needs_update(self, timeCode: Usd.TimeCode) -> bool:
        if super().needs_update(timeCode):
            return True

        # if roi was transformed, we need to reexecute.
        # If ROI changes, then the changes to USD prim are a bit extensive and trigger quite of bit
        # of changes in IndeXUsd causing runtime errors due to mismatch between the two. So
        # it's best to delay the update until the interaction has stabilized. we achieve this
        # using a coroutine that will force an update after 0.1 seconds (unless it gets canceled due
        # to a new interaction event).
        roi_targets = self.prim.GetRelationship("omni:cae:index:slice:roi").GetForwardedTargets()
        for t in roi_targets:
            if self._rel_tracker.PrimChanged(t):
                if self._force_update is None or self._force_update.done():
                    # logger.error("enqueue force update")
                    self._force_update = run_coroutine(self.force_update())
                self._rel_tracker.ClearChanges()
        return False

    def sync_time(self, timeCode: Usd.TimeCode) -> None:
        super().sync_time(timeCode)

        tc = self.get_time_code(timeCode)
        volume_prim = self.prim.GetPrimAtPath("Volume")
        if volume_prim:
            # we use Double to avoid precision loss with timecode.
            volume_prim.CreateAttribute("nvindex:timestep", Sdf.ValueTypeNames.Double, custom=True).Set(tc.GetValue())
            volume_prim.SetCustomDataByKey("omni.cae.index:timestep_0", tc.GetValue())

    async def force_update(self):
        # logger.error("start sleep force update")
        await asyncio.sleep(0.5)
        # logger.error("end sleep force update")
        run_coroutine(self.execute())

    async def execute_impl(self, timeCode: Usd.TimeCode) -> None:
        if self._force_update:
            self._force_update.cancel()
            self._force_update = None

        prim: Usd.Prim = self.prim
        dataset_prim: Usd.Prim = usd_utils.get_target_prim(prim, "omni:cae:index:slice:dataset")
        field_prims: list[Usd.Prim] = usd_utils.get_target_prims(prim, "omni:cae:index:slice:field")
        field_names: list[str] = [usd_utils.get_field_name(dataset_prim, field) for field in field_prims]

        mesh_prim = prim.GetPrimAtPath("Plane")
        if not mesh_prim:
            raise RuntimeError(f"Plane prims not found for slice {prim.GetPath()}")

        volume_prim: Usd.Prim = prim.GetPrimAtPath("Volume")
        if not volume_prim:
            raise RuntimeError(f"Volume prim not found for slice {prim.GetPath()}")

        bounds = await ComputeBounds.invoke(dataset_prim, Usd.TimeCode.EarliestTime())
        SliceHelper.update_xform(mesh_prim, bounds)

        ijk_extents: IJKExtents = await self.get_structured_extents(timeCode)
        logger.info("ijk extents: %s", ijk_extents)

        wp_volume = await Voxelize.invoke(
            dataset_prim,
            fields=field_names,
            bbox=ijk_extents.getRange(),
            voxel_size=ijk_extents.spacing[0],
            device_ordinal=0,
            timeCode=timeCode,
        )
        assert wp_volume is not None, "Voxelization failed"
        cache.put(self._cache_key, wp_volume, sourcePrims=[], consumerPrims=[prim], force=True, timeCode=timeCode)

        # setting scale in Root layer.
        NanoVdbHelper.update_scale(volume_prim, ijk_extents.spacing)

        with self.edit_context:
            NanoVdbHelper.update(volume_prim, self._cache_key, enable_interpolation=False, ijk_extents=ijk_extents)

        # last step: enable IndeX compositing.
        mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(1)

        # Let's do the colormap default setting happening in the root layer
        # so it can be overridden by the user and just acts as a default value.
        if len(field_prims) == 1:
            await usd_utils.apply_dataset_colormap_range(
                field_prims[0], prim.GetPrimAtPath("Material/Colormap"), timeCode
            )

    async def get_structured_extents(self, timeCode: Usd.TimeCode) -> IJKExtents:
        dataset: Usd.Prim = usd_utils.get_target_prim(self.prim, "omni:cae:index:slice:dataset")
        max_resolution = usd_utils.get_attribute(self.prim, "omni:cae:index:slice:maxResolution")
        roi_prim: Usd.Prim = usd_utils.get_target_prim(self.prim, "omni:cae:index:slice:roi", quiet=True)
        roi = usd_utils.get_bounds(roi_prim, timeCode, quiet=True)
        logger.info("Using ROI: %s", roi)
        ijk_extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset, max_dims=Gf.Vec3i([max_resolution] * 3), roi=roi, timeCode=timeCode
        )
        return ijk_extents


class VolumeSlice(Algorithm):

    def __init__(self, prim: Usd.Prim) -> None:
        super().__init__(prim, ["CaeIndeXSliceAPI"])

    def get_requirements(self):
        reqs = super().get_requirements()
        if volume_prim := usd_utils.get_target_prim(self.prim, "omni:cae:index:slice:volume", quiet=True):
            reqs.add(str(volume_prim.GetPath()))
        return reqs

    async def execute_impl(self, timeCode: Usd.TimeCode) -> None:
        ns = "omni:cae:index:slice"
        prim: Usd.Prim = self.prim

        mesh_prim = prim.GetPrimAtPath("Plane")
        if not mesh_prim:
            raise RuntimeError(f"Plane prims not found for slice {prim.GetPath()}")

        volume_prim: Usd.Prim = usd_utils.get_target_prim(prim, f"{ns}:volume")
        dataset_prim: Usd.Prim = NanoVdbHelper.get_dataset(volume_prim) or IrregularVolumeHelper.get_dataset(
            volume_prim
        )
        if not dataset_prim:
            raise usd_utils.QuietableException(f"Dataset prim not found for slice {prim.GetPath()}")

        field_prims: list[Usd.Prim] = NanoVdbHelper.get_fields(volume_prim) or IrregularVolumeHelper.get_fields(
            volume_prim
        )
        if not field_prims:
            raise usd_utils.QuietableException(f"Field prims not found for slice {prim.GetPath()}")

        # we use Volume bounds to update the slice mesh transform.
        bds = usd_utils.get_bounds(volume_prim)
        SliceHelper.update_xform(mesh_prim, bds)

        mesh_prim.CreateAttribute("nvindex:composite", Sdf.ValueTypeNames.Bool, custom=True).Set(1)

        # Let's do the colormap default setting happening in the root layer
        # so it can be overridden by the user and just acts as a default value.
        if len(field_prims) == 1:
            await usd_utils.apply_dataset_colormap_range(
                field_prims[0], prim.GetPrimAtPath("Material/Colormap"), timeCode
            )


class Volume(Algorithm):

    INDEX_RENDER_SETTINGS = {
        "diagnosticsMode": Vt.Int(4),
        "diagnosticsFlags": Vt.Int(8),
        "samplingMode": Vt.Int(1),
        "samplingReferenceSegmentLength": Vt.Float(1),
        "samplingSegmentLength": Vt.Double(0.75),
    }

    def __init__(self, prim: Usd.Prim) -> None:
        super().__init__(prim, ["CaeIndeXVolumeAPI"])
        # todo: track properties on dataset?

    async def execute_impl(self, timeCode: Usd.TimeCode) -> None:
        ns = "omni:cae:index:volume"
        prim: Usd.Prim = self.prim
        dataset_prim: Usd.Prim = usd_utils.get_target_prim(prim, f"{ns}:dataset")
        field_prims: Usd.Prim = usd_utils.get_target_prims(prim, f"{ns}:field")

        # ComputeBounds will be caching the extent computed for this prim.
        bounds = await ComputeBounds.invoke(dataset_prim, timeCode)
        with self.edit_context:
            IrregularVolumeHelper.update(prim, dataset_prim, field_prims, bounds, timeCode)

        # Let's do the colormap default setting happening in the root layer
        # so it can be overridden by the user and just acts as a default value.
        await usd_utils.apply_dataset_colormap_range(field_prims[0], prim.GetPrimAtPath("Material/Colormap"), timeCode)


class NanoVdbVolume(Algorithm):
    _xform_ops = [
        "omni:fabric:localMatrix",
    ]

    def __init__(self, prim: Usd.Prim) -> None:
        super().__init__(prim, ["CaeIndeXNanoVdbVolumeAPI"])
        self._rel_tracker: usd_utils.ChangeTracker = usd_utils.ChangeTracker(self.stage)
        self._force_update: asyncio.Task = None
        self._cache_key: str = f"nvdb_volume: {id(self)}"

        # xform ops
        for p in self._xform_ops:
            self._rel_tracker.TrackAttribute(p)

    def release(self):
        self.clear_cache()
        return super().release()

    def clear_cache(self):
        cache.remove(self._cache_key)

    def uses_temporal_cache(self) -> bool:
        # this algorithm uses a temporal cache, thus no need to call execute on previously executed timestep.
        return True

    def needs_update(self, timeCode: Usd.TimeCode) -> bool:
        if super().needs_update(timeCode):
            return True

        # if roi was transformed, we need to reexecute.
        # If ROI changes, then the changes to USD prim are a bit extensive and trigger quite of bit
        # of changes in IndeXUsd causing runtime errors due to mismatch between the two. So
        # it's best to delay the update until the interaction has stabilized. we achieve this
        # using a coroutine that will force an update after 0.1 seconds (unless it gets canceled due
        # to a new interaction event).
        roi_targets = self.prim.GetRelationship("omni:cae:index:nvdb:roi").GetForwardedTargets()
        for t in roi_targets:
            if self._rel_tracker.PrimChanged(t):
                if self._force_update is None or self._force_update.done():
                    # logger.error("enqueue force update")
                    self._force_update = run_coroutine(self.force_update())
                self._rel_tracker.ClearChanges()
        return False

    def needs_update_for_time(self, timeCode: Usd.TimeCode) -> bool:
        times = self.get_execution_timecodes(timeCode)
        return super().needs_update_for_time(times[0]) or super().needs_update_for_time(times[-1])

    def get_execution_timecodes(self, timeCode: Usd.TimeCode) -> tuple[Usd.TimeCode, Usd.TimeCode]:
        if usd_utils.get_attribute(self.prim, "omni:cae:index:nvdb:temporalInterpolation", quiet=True):
            codes = self.get_bracketing_time_codes(timeCode)
            # logger.info("execution time codes %s => (%s, %s)", Algorithm.get_current_timecode(), codes[0], codes[1])
            return codes
        else:
            tc = self.get_time_code(timeCode)
            # logger.info("execution time codes %s => (%s, )", Algorithm.get_current_timecode(), tc)
            return (tc,)

    async def force_update(self):
        # logger.error("start sleep force update")
        await asyncio.sleep(0.5)
        # logger.error("end sleep force update")
        run_coroutine(self.execute())

    def sync_time(self, timeCode: Usd.TimeCode) -> None:
        super().sync_time(timeCode)

        with self.edit_context:
            timeCodes = self.get_execution_timecodes(timeCode)

            # we use Double to avoid precision loss with timecode.
            self.prim.CreateAttribute("nvindex:timestep", Sdf.ValueTypeNames.Double, custom=True).Set(
                timeCodes[0].GetValue()
            )

            # used by CaeDataSetVoxelizedNanoVdbFetchTechnique / CaeDataSetNanoVdbFetchTechnique
            # for temporal interpolation
            self.prim.SetCustomDataByKey("omni.cae.index:timestep_0", timeCodes[0].GetValue())
            self.prim.SetCustomDataByKey("omni.cae.index:timestep_1", timeCodes[-1].GetValue())

            if len(timeCodes) == 2 and timeCodes[0] < timeCodes[1]:
                fraction = (timeCode.GetValue() - timeCodes[0].GetValue()) / (
                    timeCodes[1].GetValue() - timeCodes[0].GetValue()
                )
            else:
                fraction = 0.0

            if shaderPrim := usd_utils.get_target_prim(self.prim, "cae:shaderLERP", quiet=True):
                shader: UsdShade.Shader = UsdShade.Shader(shaderPrim)
                shader.GetInput("fraction").Set(fraction)
            if shaderPrim := usd_utils.get_target_prim(self.prim, "cae:shader", quiet=True):
                shader: UsdShade.Shader = UsdShade.Shader(shaderPrim)
                shader.GetInput("fraction").Set(fraction)

    async def execute_impl(self, timeCode: Usd.TimeCode):
        if self._force_update:
            self._force_update.cancel()
            self._force_update = None

        prim: Usd.Prim = self.prim
        dataset_prim: Usd.Prim = usd_utils.get_target_prim(prim, "omni:cae:index:nvdb:dataset")
        field_prim: Usd.Prim = usd_utils.get_target_prim(prim, "omni:cae:index:nvdb:field")
        enable_interpolation: bool = usd_utils.get_attribute(prim, "omni:cae:index:nvdb:temporalInterpolation")

        # we have to process data to determine a few attributes ijk extents and spacing/scale
        # since with Kit 106.5, IndeXUsd does not support timesamples for extents/scale, we
        # we will use the earliest timesample to compute the extents.
        ijk_extents = await self.get_structured_extents(Usd.TimeCode.EarliestTime())

        fields = [usd_utils.get_field_name(dataset_prim, field_prim)]
        codes = self.get_execution_timecodes(timeCode)
        logger.info("executing time codes %s", f"{codes[0]} -> {codes[-1]}")

        mode = 0  # float
        for tc in codes:
            # we trigger voxelization here so that this potentially slow steps happens before rendering passes
            volume = await Voxelize.invoke(
                dataset_prim,
                fields=fields,
                bbox=ijk_extents.getRange(),
                voxel_size=ijk_extents.spacing[0],
                device_ordinal=0,
                timeCode=tc,
            )
            assert volume is not None

            if volume.get_grid_info().type_str == "float":
                mode = 0
            elif volume.get_grid_info().type_str.lower() == "vec3f":
                mode = 1
            else:
                raise RuntimeError(f"Unsupported grid type: {volume.get_grid_info().type_str}")

            # FIXME: consumerPrim = [ prim ] causes cache to be cleared immediately! need to fix that.
            # FIXME: sourcePrims = [ dataset_prim, field_prim ] causes cache to be cleared when FPS changes since "/" is
            # in ResycnedPaths. For now, avoid this by setting sourcePrims = [].
            cache.put(self._cache_key, volume, sourcePrims=[], consumerPrims=[], force=True, timeCode=tc)

            # this is necessary to avoid re-executing the algorithm for the same timeCode(s)
            self._executed_time_codes.add(tc)

        # setting scale in Root layer.
        NanoVdbHelper.update_scale(prim, ijk_extents.spacing)

        with self.edit_context:
            NanoVdbHelper.update(prim, self._cache_key, enable_interpolation, ijk_extents)

            # set the shader mode
            if shader_prim := prim.GetPrimAtPath("Material/VolumeShader"):
                shader: UsdShade.Shader = UsdShade.Shader(shader_prim)
                shader.CreateInput("mode", Sdf.ValueTypeNames.Int).Set(mode)

        # Let's do the colormap default setting happening in the root layer
        # so it can be overridden by the user and just acts as a default value.
        await usd_utils.apply_dataset_colormap_range(field_prim, prim.GetPrimAtPath("Material/Colormap"), timeCode)

        # reset change tracker
        self._rel_tracker.ClearChanges()

    async def get_structured_extents(self, timeCode: Usd.TimeCode) -> IJKExtents:
        dataset: Usd.Prim = usd_utils.get_target_prim(self.prim, "omni:cae:index:nvdb:dataset")
        max_resolution = usd_utils.get_attribute(self.prim, "omni:cae:index:nvdb:maxResolution")

        roi_prim: Usd.Prim = usd_utils.get_target_prim(self.prim, "omni:cae:index:nvdb:roi", quiet=True)
        roi = usd_utils.get_bounds(roi_prim, timeCode, quiet=True)
        logger.info("Using ROI: %s", roi)
        ijk_extents: IJKExtents = await ComputeIJKExtents.invoke(
            dataset, max_dims=Gf.Vec3i([max_resolution] * 3), roi=roi, timeCode=timeCode
        )
        return ijk_extents
