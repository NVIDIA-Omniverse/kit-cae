# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from omni.cae.data import commands
from omni.cae.schema import cae
from omni.kit.commands import Command
from pxr import Usd

from .._omni_cae_index import Bbox_float32, IIrregular_volume_subset


class CreateIrregularVolumeSubset(Command):

    def __init__(
        self,
        dataset: Usd.Prim,
        fields: list[str],
        bbox: Bbox_float32,
        subset: IIrregular_volume_subset,
        timeCode: Usd.TimeCode,
    ):
        self._dataset: Usd.Prim = dataset
        self._fields: list[str] = fields
        self._bbox: Bbox_float32 = bbox
        self._subset: IIrregular_volume_subset = subset
        self._timeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        return self._dataset

    @property
    def fields(self) -> list[str]:
        return self._fields

    @property
    def bbox(self) -> Bbox_float32:
        return self._bbox

    @property
    def subset(self) -> IIrregular_volume_subset:
        return self._subset

    @property
    def timeCode(self) -> Usd.TimeCode:
        return self._timeCode

    @classmethod
    async def invoke(
        cls,
        dataset: Usd.Prim,
        fields: list[str],
        timeCode: Usd.TimeCode,
        bbox: Bbox_float32,
        subset: IIrregular_volume_subset,
    ):

        # validate args
        if not dataset.IsA(cae.DataSet):
            raise ValueError("%s must be a OmniCae.DataSet" % dataset)

        # validate that fields are field relationships on dataset.
        for f in fields:
            if not dataset.HasRelationship(f"field:{f}"):
                raise ValueError("%s if not a field name on %s" % (f, dataset))

        return await commands.execute(
            cls.__name__, dataset, dataset=dataset, fields=fields, bbox=bbox, subset=subset, timeCode=timeCode
        )
