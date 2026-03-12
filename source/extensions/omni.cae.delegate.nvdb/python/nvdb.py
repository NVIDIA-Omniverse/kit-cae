# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["NvdbDataDelegate"]

from logging import getLogger

import numpy as np
import warp as wp
from omni.cae.data import array_utils
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import nvdb
from omni.client import get_local_file
from pxr import Usd

logger = getLogger(__name__)


class NvdbDataDelegate(DataDelegateBase):

    def __init__(self, extId: str):
        super().__init__(extId)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode):
        primT = nvdb.FieldArray(prim)
        fileNames = primT.GetFileNamesAttr().Get(time)
        if not fileNames:
            return None
        if len(fileNames) > 1:
            raise ValueError(f"NvdbFieldArray only supports a single .nvdb file; {len(fileNames)} were provided.")
        file_path = get_local_file(fileNames[0].resolvedPath)[1]
        with open(file_path, "rb") as f:
            volume = wp.Volume.load_from_nvdb(f)
            array = (
                volume.array().numpy().view(np.uint32)
            )  # Ensure consistent dtype; NanoVDB may have different internal types.
            return array

    def can_provide(self, prim: Usd.Prim) -> bool:
        return prim and prim.IsValid() and prim.IsA(nvdb.FieldArray)
