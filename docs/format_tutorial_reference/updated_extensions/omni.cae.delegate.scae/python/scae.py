# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import json
import os
from logging import getLogger
from typing import Any

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.client import get_local_file
from pxr import Usd

__all__ = ["ScaeDataDelegate"]

logger = getLogger(__name__)


def _parse_one_slice(spec: str):
    parts = spec.split(":")
    if len(parts) == 1:
        return int(parts[0])
    start_s, stop_s, step_s = (parts + [None] * 3)[:3]

    def _to_int(value):
        return None if value is None or value == "" else int(value)

    return slice(_to_int(start_s), _to_int(stop_s), _to_int(step_s))


def _parse_slice_string(value: str):
    return tuple(_parse_one_slice(part.strip()) for part in value.split(","))


def _resolve_local_path(asset_path: str) -> str | None:
    if not asset_path:
        return None
    try:
        return get_local_file(asset_path)[1]
    except Exception:
        logger.exception("Failed to resolve local path for asset '%s'", asset_path)
        return None


def _read_manifest(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except Exception:
        logger.exception("Failed to read Scae manifest '%s'", path)
        return None
    if not isinstance(data, dict):
        logger.error("Scae manifest '%s' is not a JSON object", path)
        return None
    if not isinstance(data.get("arrays"), dict):
        logger.error("Scae manifest '%s' missing 'arrays' object", path)
        return None
    return data


def _read_array_from_manifest(manifest: dict[str, Any], manifest_path: str, array_name: str) -> np.ndarray | None:
    array_spec = manifest.get("arrays", {}).get(array_name)
    if not isinstance(array_spec, dict):
        return None

    binary_file = manifest.get("binary_file")
    if not isinstance(binary_file, str) or not binary_file:
        logger.error("Manifest '%s' missing valid 'binary_file'", manifest_path)
        return None

    binary_path = (
        binary_file if os.path.isabs(binary_file) else os.path.join(os.path.dirname(manifest_path), binary_file)
    )
    if not os.path.exists(binary_path):
        logger.error("Scae binary payload does not exist: %s", binary_path)
        return None

    try:
        dtype = np.dtype(array_spec["dtype"])
        shape = tuple(int(dim) for dim in array_spec["shape"])
        offset_bytes = int(array_spec.get("offset_bytes", 0))
    except Exception:
        logger.exception("Invalid array specification '%s' in '%s'", array_name, manifest_path)
        return None

    count = int(np.prod(shape, dtype=np.int64))
    if count <= 0:
        logger.error("Array '%s' has non-positive element count in '%s'", array_name, manifest_path)
        return None

    try:
        data = np.fromfile(binary_path, dtype=dtype, count=count, offset=offset_bytes)
    except Exception:
        logger.exception("Failed reading array '%s' from '%s'", array_name, binary_path)
        return None

    if data.size != count:
        logger.error("Array '%s' in '%s' has %d values, expected %d", array_name, binary_path, data.size, count)
        return None

    return data.reshape(shape)


class ScaeDataDelegate(DataDelegateBase):
    def __init__(self, ext_id: str):
        super().__init__(ext_id)

    def can_provide(self, prim: Usd.Prim) -> bool:
        return prim and prim.IsValid() and prim.IsA(cae.ScaeFieldArray)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray | None:
        prim_t = cae.ScaeFieldArray(prim)
        array_name = prim_t.GetArrayNameAttr().Get(time)
        file_names = prim_t.GetFileNamesAttr().Get(time)
        slice_expr = prim_t.GetSliceAttr().Get(time)
        ts = prim_t.GetTsAttr().Get(time)

        if not array_name or not file_names:
            return None

        arrays = []
        for file_name in file_names:
            asset_path = getattr(file_name, "resolvedPath", None) or str(file_name)
            manifest_path = _resolve_local_path(asset_path)
            if not manifest_path:
                continue
            manifest = _read_manifest(manifest_path)
            if manifest is None:
                continue

            data = _read_array_from_manifest(manifest, manifest_path, array_name)
            if data is None:
                continue

            if slice_expr:
                try:
                    data = data[np.s_[_parse_slice_string(slice_expr.format(ts=ts))]]
                except Exception:
                    logger.exception("Failed applying slice '%s' for '%s'", slice_expr, array_name)
                    continue

            arrays.append(data)

        if not arrays:
            return None
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays)
