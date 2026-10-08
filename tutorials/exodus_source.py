# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The Exodus Data Source appendix A2 writes cell by cell, packaged for later appendices.

A2 explains this code: a ``VtkExternalDatasetAPI`` descriptor prim names the
file, the identifier is a data assembly node path, and the frame provider
reports identities and loads datasets. Register the schema in
``vtk_external_dataset_schema/`` with usd-core before the first stage, then
construct ``ExodusDataSource`` over the usd-core stage the scene was authored
on. Keep this module in step with A2.
"""

from __future__ import annotations

import numpy as np
import vtk
import warp as wp
from vtk.util.numpy_support import vtk_to_numpy

import ovcae

API = "VtkExternalDatasetAPI"
FILE_NAMES = "vtk:fileNames"

import os

from pxr import Usd
from warp_simdata import AssociationType, Field
from warp_simdata.data_models.vtk import unstructured_grid

from ovcae.data import DatasetSample, held_time, resolve_asset_path


def _arrays(grid) -> dict:
    """Host arrays of one unstructured grid, in the layout the ``vtk`` data model takes."""
    point_data = grid.GetPointData()
    return {
        "points": vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32),
        "cell_types": vtk_to_numpy(grid.GetCellTypes()).astype(np.uint8),
        "cell_offsets": vtk_to_numpy(grid.GetCells().GetOffsetsArray()).astype(np.int32),
        "connectivity": vtk_to_numpy(grid.GetCells().GetConnectivityArray()).astype(np.int32),
        "fields": {
            point_data.GetArrayName(i): vtk_to_numpy(point_data.GetArray(i)).astype(np.float32)
            for i in range(point_data.GetNumberOfArrays())
        },
    }


def _one_grid(partitioned) -> vtk.vtkUnstructuredGrid | None:
    """The partitions of one partitioned dataset as a single unstructured grid; None if it is empty."""
    partitions = [partitioned.GetPartition(i) for i in range(partitioned.GetNumberOfPartitions())]
    partitions = [part for part in partitions if part is not None and part.GetNumberOfCells()]
    if not partitions:
        return None
    if len(partitions) == 1 and partitions[0].IsA("vtkUnstructuredGrid"):
        return partitions[0]
    append = vtk.vtkAppendFilter()
    for part in partitions:
        append.AddInputData(part)
    append.Update()
    return append.GetOutput()


def read_exodus(path: str) -> dict[str, dict]:
    """Read an Exodus II file into host arrays per dataset, keyed by data assembly node path."""
    reader = vtk.vtkExodusIIReader()
    reader.SetFileName(path)
    reader.UpdateInformation()
    for kind in (reader.ELEM_BLOCK, reader.SIDE_SET):
        for index in range(reader.GetNumberOfObjects(kind)):
            reader.SetObjectStatus(kind, index, 1)
    for index in range(reader.GetNumberOfObjectArrays(reader.NODAL)):
        reader.SetObjectArrayStatus(reader.NODAL, reader.GetObjectArrayName(reader.NODAL, index), 1)

    convert = vtk.vtkConvertToPartitionedDataSetCollection()
    convert.SetInputConnection(reader.GetOutputPort())
    convert.Update()
    collection = convert.GetOutput()
    assembly = collection.GetDataAssembly()

    datasets = {}
    pending = [assembly.GetRootNode()]
    while pending:  # every assembly node that holds exactly one non-empty partitioned dataset is selectable
        node = pending.pop()
        pending.extend(assembly.GetChildNodes(node, False))
        indices = list(assembly.GetDataSetIndices(node, False))
        if len(indices) != 1:
            continue
        grid = _one_grid(collection.GetPartitionedDataSet(indices[0]))
        if grid is not None:
            datasets[assembly.GetNodePath(node)] = _arrays(grid) | {
                "label": assembly.GetAttributeOrDefault(node, "label", "")
            }
    return datasets


class ExodusDataSource(ovcae.DataSource):
    """Serve ``VtkExternalDatasetAPI`` descriptors on a usd-core stage by reading their files with VTK."""

    def __init__(self, stage):
        super().__init__()
        self._stage = stage
        self._cache: dict[tuple[str, int], dict[str, dict]] = {}

    def resolves(self, key: str) -> bool:
        prim = self._stage.GetPrimAtPath(str(key)) if self._stage else None
        return bool(prim) and prim.HasAPI(API)

    def frame_at_time(self, time_seconds: float):
        return _ExodusFrame(self, float(time_seconds))

    def release_resources(self) -> None:
        self._stage = None
        self._cache.clear()

    # Helpers the frame provider uses.

    def select_file(self, key: str, time_seconds: float) -> tuple[str, DatasetSample]:
        """The file the descriptor names at ``time_seconds``, under held sampling."""
        prim = self._stage.GetPrimAtPath(str(key))
        if not prim or not prim.HasAPI(API):
            raise ovcae.DatasetResolutionError(f"{key} is not a {API} descriptor")
        attribute = prim.GetAttribute(FILE_NAMES)
        rate = self._stage.GetTimeCodesPerSecond()
        sample = held_time(tuple(attribute.GetTimeSamples()), time_seconds * rate)
        values = attribute.Get(Usd.TimeCode(sample) if sample is not None else Usd.TimeCode.Default()) or ()
        if len(values) != 1:
            raise ovcae.DatasetResolutionError(f"{key} must name exactly one file per sample; got {len(values)}")
        path = resolve_asset_path(self._stage.GetRootLayer(), values[0])
        if not path or not os.path.isfile(path):
            raise ovcae.DatasetResolutionError(f"{key} names a missing file {values[0].path!r}")
        return path, DatasetSample(None if sample is None else sample / rate)

    def _datasets(self, path: str) -> dict[str, dict]:
        """The file's datasets, read once per (path, modification time) so a rewritten file is re-read."""
        key = (path, os.stat(path).st_mtime_ns)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != path}  # drop the stale version
            self._cache[key] = read_exodus(path)
        return self._cache[key]

    def identifiers(self, path: str) -> tuple[str, ...]:
        """The data assembly node paths the file at ``path`` offers as identifiers."""
        return tuple(sorted(self._datasets(path)))

    def arrays(self, path: str, identifier: str) -> dict:
        identifiers = self.identifiers(path)
        if identifier not in identifiers:
            problem = (
                "is a composite dataset and needs an identifier naming one of its datasets"
                if not identifier
                else f"has no dataset {identifier!r}"
            )
            raise ovcae.DatasetResolutionError(
                f"{path} {problem}. Valid identifiers: {', '.join(repr(i) for i in identifiers)}"
            )
        return self._datasets(path)[identifier]


class _ExodusFrame:
    """One lease on the source: the file selection is fixed at construction."""

    def __init__(self, source: ExodusDataSource, time_seconds: float):
        self._source = source
        self._time = time_seconds

    def describe_component(self, key: str, identifier: str, component: str):
        path, sample = self._source.select_file(key, self._time)
        identity = (path, identifier, component, os.stat(path).st_mtime_ns, sample.time)
        return identity, sample

    def resolve_dataset(self, key: str, *, identifier: str, device: str, fields):
        path, _sample = self._source.select_file(key, self._time)
        arrays = self._source.arrays(path, identifier)
        dataset = unstructured_grid.create_dataset(
            wp.array(arrays["points"], dtype=wp.vec3f, device=device),
            wp.array(arrays["cell_types"], dtype=wp.uint8, device=device),
            wp.array(arrays["cell_offsets"], dtype=wp.int32, device=device),
            wp.array(arrays["connectivity"], dtype=wp.int32, device=device),
        )
        for role, names in fields.items():
            if len(names) != 1 or names[0] not in arrays["fields"]:
                available = ", ".join(sorted(arrays["fields"]))
                raise ovcae.DatasetResolutionError(f"{role!r} must name one of {available}; got {list(names)}")
            values = arrays["fields"][names[0]]
            dtype = wp.vec3f if values.ndim == 2 else wp.float32
            dataset.add_field(
                role, Field.from_array(wp.array(values, dtype=dtype, device=device), AssociationType.NODE)
            )
        return dataset
