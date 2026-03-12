# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

from omni.cae.algorithms.core import bind_material, create_material
from omni.cae.data import settings
from omni.kit.commands import Command
from omni.usd import get_context
from pxr import Sdf, Usd, UsdGeom

logger = getLogger(__name__)


class CreateCaeNanoVdbStreamlines(Command):
    def __init__(self, dataset_path: str, prim_path: str):
        self._dataset_path = dataset_path
        self._prim_path = prim_path

    def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)
        stage: Usd.Stage = get_context().get_stage()
        dataset_prim = stage.GetPrimAtPath(self._dataset_path)
        if not dataset_prim:
            raise RuntimeError("DataSet prim is invalid!")
        width = 0.025
        primT = UsdGeom.BasisCurves.Define(stage, self._prim_path)
        primT.CreatePointsAttr().Set([(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)])
        primT.CreateCurveVertexCountsAttr().Set([4])
        primT.CreateExtentAttr().Set([(0.0, 0.0, 0.0), (0.3, 0.05, 0.05)])
        primT.CreateWidthsAttr().Set([(width)])

        primT.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        primT.CreateBasisAttr().Set(UsdGeom.Tokens.bspline)
        primT.CreateTypeAttr().Set(UsdGeom.Tokens.cubic)
        primT.CreateWrapAttr().Set(UsdGeom.Tokens.pinned)

        # apply schema
        prim = primT.GetPrim()
        prim.AddAppliedSchema("CaeAlgorithmsWarpStreamlinesAPI")
        ns = "omni:cae:warp:streamlines"
        prim.CreateRelationship(f"{ns}:dataset", custom=False).SetTargets({dataset_prim.GetPath()})
        prim.CreateRelationship(f"{ns}:velocity", custom=False).SetTargets({})
        # prim.CreateAttribute(f"{ns}:velocity", Sdf.ValueTypeNames.StringArray, custom=False)
        prim.CreateAttribute(f"{ns}:dX", Sdf.ValueTypeNames.Float, custom=False).Set(0.5)
        prim.CreateAttribute(f"{ns}:maxLength", Sdf.ValueTypeNames.Int, custom=False).Set(50)
        prim.CreateRelationship(f"{ns}:seeds", custom=False).SetTargets({})
        prim.CreateRelationship(f"{ns}:colors", custom=False).SetTargets({})
        prim.CreateRelationship(f"{ns}:roi", custom=False).SetTargets({})
        prim.CreateAttribute(f"{ns}:maxResolution", Sdf.ValueTypeNames.Int, custom=False).Set(
            settings.get_default_max_voxel_grid_resolution()
        )
        prim.CreateAttribute(f"{ns}:width", Sdf.ValueTypeNames.Float, custom=False).Set(width)

        # primvars for the basis curve
        pvAPI = UsdGeom.PrimvarsAPI(primT.GetPrim())

        # primvars needed by shader
        pvAPI.CreatePrimvar("scalar", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set([0.0, 0.1, 0.2, 0.3])

        # setup material
        material = create_material(
            "ScalarColor", stage, primT.GetPath().AppendChild("Materials").AppendChild("ScalarColor")
        )
        bind_material(primT, material)

        logger.info("created '%s''", str(primT.GetPath()))

    def undo(self):
        if self._prim_path:
            stage: Usd.Stage = get_context().get_stage()
            stage.RemovePrim(self._prim_path)
            self._prim_path = None
