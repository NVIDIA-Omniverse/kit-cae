# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["NPZDataDelegate"]

from logging import getLogger

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.client import get_local_file
from pxr import Usd

logger = getLogger(__name__)


# TODO:
# * use OmniCAEFile to avoid re-reading npz file for each array.


def parse_one_slice(spec: str):
    # spec like "1:5", ":", "1::2", "-3:", "::2"
    parts = spec.split(":")
    if len(parts) == 1:
        # single index like "3" -> int
        return int(parts[0])

    # fill [start, stop, step] with None where empty
    start_s, stop_s, step_s = (parts + [None] * 3)[:3]

    def conv(x):
        return None if x is None or x == "" else int(x)

    return slice(conv(start_s), conv(stop_s), conv(step_s))


def parse_slice_string(s: str):
    # e.g. "1:5, ::2, -3:" -> (slice(...), slice(...), slice(...))
    specs = [part.strip() for part in s.split(",")]
    return tuple(parse_one_slice(p) for p in specs)


class NPZDataDelegate(DataDelegateBase):

    def __init__(self, extId: str):
        super().__init__(extId)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        primT = cae.NumPyFieldArray(prim)
        arrayName = primT.GetArrayNameAttr().Get(time)
        fileNames = primT.GetFileNamesAttr().Get(time)
        allowPickle = primT.GetAllowPickleAttr().Get(time)
        slice = primT.GetSliceAttr().Get(time)
        ts = primT.GetTsAttr().Get(time)
        arrays = []
        for f in fileNames:
            dataset = np.load(get_local_file(f.resolvedPath)[1], allow_pickle=allowPickle)
            if isinstance(dataset, np.ndarray) and dataset.dtype == object:
                dataset = dataset.item(0)
            if arrayName in dataset.keys():
                array = dataset[arrayName]
                # this seems funny, but for pickled files, the np array type
                # checks in DataDelegateBindingsPython.cpp fail so we do this explicit conversion
                # here to avoid that issue.
                if array.dtype == np.int32:
                    array = array.astype(np.int32)
                elif array.dtype == np.int64:
                    array = array.astype(np.int64)
                elif array.dtype == np.uint32:
                    array = array.astype(np.uint32)
                elif array.dtype == np.uint64:
                    array = array.astype(np.uint64)
                elif array.dtype == np.float32:
                    array = array.astype(np.float32)
                elif array.dtype == np.float64:
                    array = array.astype(np.float64)
                if slice:
                    array = array[np.s_[parse_slice_string(slice.format(ts=ts))]]
                arrays.append(array)
        return np.concatenate(arrays) if arrays else None

    def can_provide(self, prim: Usd.Prim) -> bool:
        return prim and prim.IsValid() and prim.IsA(cae.NumPyFieldArray)
