# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Building blocks for implementing a custom :class:`DataSource`.

A custom source subclasses :class:`DataSource`, claims keys with ``resolves``,
and returns a per-lease frame provider from ``frame_at_time``. These helpers
cover the parts every file-backed source repeats: resolving an asset path
against the layer that authored it, selecting the held sample from a timeline,
and turning NumPy arrays into a Warp SimData dataset with fields.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._data import CompositeDataSource, DataSource, FrameData
from ._errors import DatasetResolutionError
from ._usd_time import DatasetSample, held_time

__all__ = [
    "CompositeDataSource",
    "DataSource",
    "DatasetSample",
    "FrameData",
    "add_fields",
    "dataset_from_arrays",
    "held_time",
    "resolve_asset_path",
]


def resolve_asset_path(layer, asset) -> str:
    """Resolve an ``Sdf.AssetPath`` or string relative to ``layer``; empty when unresolvable."""

    from pxr import Sdf

    resolved = getattr(asset, "resolvedPath", "")
    if resolved:
        return str(resolved)
    raw = str(getattr(asset, "path", asset))
    return str(Sdf.ComputeAssetPathRelativeToLayer(layer, raw)) if raw else ""


def dataset_from_arrays(value: Mapping[str, Any], device: str):
    """Build a SimData dataset from ``points`` and, for cells, VTK-style connectivity.

    ``value`` holds ``points`` (N x 3) and optionally ``cell_types``,
    ``connectivity_offsets``, and ``connectivity``; without cells the result is a
    point cloud. Fields are attached separately with :func:`add_fields`.
    """

    try:
        import numpy as np
        import warp as wp
        from warp_simdata.data_models.custom import point_cloud
        from warp_simdata.data_models.vtk import unstructured_grid

        points = wp.array(np.asarray(value["points"], dtype=np.float32), dtype=wp.vec3f, device=device)
        if "cell_types" not in value:
            return point_cloud.create_dataset(points)
        cell_types = wp.array(np.asarray(value["cell_types"], dtype=np.uint8), dtype=wp.uint8, device=device)
        offsets = wp.array(np.asarray(value["connectivity_offsets"], dtype=np.int32), dtype=wp.int32, device=device)
        connectivity = wp.array(np.asarray(value["connectivity"], dtype=np.int32), dtype=wp.int32, device=device)
        return unstructured_grid.create_dataset(points, cell_types, offsets, connectivity)
    except KeyError as exc:
        raise DatasetResolutionError(f"in-memory array dataset is missing {exc.args[0]!r}") from exc
    except Exception as exc:
        raise DatasetResolutionError(f"could not construct in-memory dataset: {exc}") from exc


def add_fields(dataset, value, fields: Mapping[str, Sequence[str]], device: str) -> None:
    """Attach requested fields from ``value["fields"]`` under their output names.

    Each entry of ``value["fields"]`` maps a source array name to a mapping with
    ``values`` and ``association`` (``node``/``vertex`` or
    ``element``/``cell``/``uniform``). ``fields`` maps output names to one source
    name each. Values may be scalar arrays or arrays with one to four components;
    four-component arrays support xyzw quaternion fields.
    """

    if not fields or not isinstance(value, Mapping):
        return
    available = value.get("fields", {})
    for output_name, names in fields.items():
        if len(names) != 1 or names[0] not in available:
            raise DatasetResolutionError(f"in-memory field selection {output_name!r} could not resolve {list(names)!r}")
        descriptor = available[names[0]]
        try:
            import numpy as np
            import warp as wp
            import warp_simdata as simdata

            association_name = str(descriptor["association"]).lower()
            associations = {
                "node": simdata.AssociationType.NODE,
                "vertex": simdata.AssociationType.NODE,
                "element": simdata.AssociationType.ELEMENT,
                "cell": simdata.AssociationType.ELEMENT,
                "uniform": simdata.AssociationType.ELEMENT,
            }
            try:
                association = associations[association_name]
            except KeyError as exc:
                raise ValueError(f"unsupported field association {association_name!r}") from exc
            array = np.asarray(descriptor["values"])
            if array.ndim == 2 and array.shape[1] == 1:
                array = array.reshape(-1)
            warp_array = wp.array(array, dtype=_warp_dtype(wp, array), device=device)
            dataset.add_field(output_name, simdata.Field.from_array(warp_array, association), warn_if_exists=False)
        except Exception as exc:
            raise DatasetResolutionError(f"could not construct in-memory field {output_name!r}: {exc}") from exc


def _warp_dtype(wp, array):
    import numpy as np

    scalar = {
        np.dtype(np.float32): wp.float32,
        np.dtype(np.float64): wp.float64,
        np.dtype(np.int32): wp.int32,
        np.dtype(np.int64): wp.int64,
    }.get(array.dtype)
    if scalar is None:
        raise TypeError(f"unsupported field dtype {array.dtype}")
    if array.ndim == 1:
        return scalar
    if array.ndim == 2 and array.shape[1] in (2, 3, 4):
        return wp.types.vector(length=array.shape[1], dtype=scalar)
    raise TypeError(f"unsupported field shape {array.shape}")
