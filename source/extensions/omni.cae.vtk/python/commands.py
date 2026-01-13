# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from omni.cae.data import cache, commands, usd_utils
from omni.cae.schema import cae
from omni.kit.commands import Command
from pxr import Usd
from vtkmodules.numpy_interface import dataset_adapter as dsa

from .impl import utils


class ConvertToVTKDataSet(Command):

    class Params:
        dataset: Usd.Prim
        fields: list[str]

    def __init__(self, params: Params, timeCode: Usd.TimeCode) -> None:
        self._params: ConvertToVTKDataSet.Params = params
        self._timeCode: Usd.TimeCode = timeCode

    @property
    def params(self) -> Params:
        return self._params

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(
        cls, dataset: Usd.Prim, fields: list[str], forcePointData: bool, timeCode: Usd.TimeCode
    ) -> dsa.DataSet:

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet" % dataset)

        # validate that fields are field relationships on dataset.
        field_prims = []
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s if not a field name on %s" % (f, dataset))
            field_prims += usd_utils.get_target_prims(dataset, f"field:{f}")

        cache_key = {
            "label": "ConvertToVTKDataSet",
            "dataset": str(dataset.GetPath()),
            "fields": str(fields),
            "forcePointData": forcePointData,
        }

        cache_state = {}
        vtk_dataset = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if vtk_dataset is None:
            params = ConvertToVTKDataSet.Params()
            params.dataset = dataset
            params.fields = fields
            vtk_dataset = await commands.execute(cls.__name__, dataset, params=params, timeCode=timeCode)
            if forcePointData and len(fields) > 0:
                vtk_dataset = utils.cell_data_to_point_data(vtk_dataset)
            if vtk_dataset:
                cache.put(
                    str(cache_key),
                    vtk_dataset,
                    state=cache_state,
                    sourcePrims=[dataset] + field_prims,
                    timeCode=timeCode,
                )
        return vtk_dataset
