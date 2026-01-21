# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utilities for converting VTK datasets to DAV datasets."""

import logging
from typing import Union

import warp as wp
from dav.dataset import Dataset, DatasetCollection
from dav.field import Field
from dav.fields import AssociationType
from vtkmodules.numpy_interface import dataset_adapter as dsa
from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModel import (
    vtkDataSet,
    vtkImageData,
    vtkMultiBlockDataSet,
    vtkPartitionedDataSet,
    vtkPartitionedDataSetCollection,
    vtkPointSet,
    vtkRectilinearGrid,
    vtkStructuredGrid,
    vtkUniformGridAMR,
    vtkUnstructuredGrid,
)
from vtkmodules.vtkFiltersCore import vtkConvertToPartitionedDataSetCollection

logger = logging.getLogger(__name__)


def vtk_to_dataset(
    vtk_dataset: vtkDataSet, device: Union[str, wp.context.Device, None] = None
) -> dict[str, Union[Dataset, DatasetCollection]]:
    """Convert a VTK dataset to a DAV dataset with appropriate data model.

    Args:
        vtk_dataset: VTK dataset object (vtkDataSet or composite dataset)
        device: Device to create the dataset on. If None, uses Warp's current device context.

    Returns:
        dict[str, Union[Dataset, DatasetCollection]]: Dictionary mapping names to datasets or collections.
        - For simple datasets: Returns {"dataset": dav.Dataset}
        - For vtkPartitionedDataSet: Returns {"dataset": dav.DatasetCollection}
        - For vtkMultiBlockDataSet/vtkPartitionedDataSetCollection: Returns {"BlockName": dav.DatasetCollection, ...}

    Raises:
        ValueError: If the dataset type is not supported
        TypeError: If the input is not a valid VTK dataset

    Supported types:
        - vtkStructuredGrid, vtkUnstructuredGrid, vtkImageData
        - vtkPointSet (converted to unstructured grid with points only, no cells)
        - vtkPolyData (converted to unstructured grid with points only, no cells)
        - vtkPartitionedDataSet
        - vtkPartitionedDataSetCollection
        - vtkMultiBlockDataSet (converted to PartitionedDataSetCollection)
        - vtkUniformGridAMR (raises ValueError - not yet supported)
    """
    if vtk_dataset is None:
        raise TypeError("vtk_dataset cannot be None")

    # Use Warp's current device if not specified
    if device is None:
        device = wp.get_device()

    # Check for AMR datasets
    if isinstance(vtk_dataset, vtkUniformGridAMR):
        raise ValueError("vtkUniformGridAMR is not yet supported. AMR support will be added in a future release.")

    # Handle vtkMultiBlockDataSet by converting to vtkPartitionedDataSetCollection
    if isinstance(vtk_dataset, vtkMultiBlockDataSet):
        logger.info("Converting vtkMultiBlockDataSet to vtkPartitionedDataSetCollection")
        converter = vtkConvertToPartitionedDataSetCollection()
        converter.SetInputDataObject(vtk_dataset)
        converter.Update()
        vtk_dataset = converter.GetOutput()
        # Fall through to handle as PartitionedDataSetCollection

    # Handle vtkPartitionedDataSetCollection - return dict[str, DatasetCollection]
    if isinstance(vtk_dataset, vtkPartitionedDataSetCollection):
        result = {}
        num_pds = vtk_dataset.GetNumberOfPartitionedDataSets()

        logger.info("Processing vtkPartitionedDataSetCollection with %d partitioned datasets", num_pds)

        for i in range(num_pds):
            pds = vtk_dataset.GetPartitionedDataSet(i)
            if pds is None:
                logger.warning("Skipping null partitioned dataset at index %d", i)
                continue

            # Get the name for this partitioned dataset
            metadata = vtk_dataset.GetMetaData(i)
            name = metadata.Get(vtk_dataset.NAME()) if metadata and metadata.Has(vtk_dataset.NAME()) else f"Block_{i}"

            try:
                # Convert the partitioned dataset to a DatasetCollection
                datasets = []
                for j in range(pds.GetNumberOfPartitions()):
                    partition = pds.GetPartition(j)
                    if partition is not None:
                        try:
                            converted = _convert_single_dataset(partition, device=device)
                            datasets.append(converted)
                        except (ValueError, TypeError) as e:
                            logger.warning("Skipping partition %d in dataset '%s': %s", j, name, e)
                            continue

                if datasets:
                    result[name] = DatasetCollection.from_datasets(datasets)
                else:
                    logger.warning("No valid partitions found in dataset '%s'", name)
            except Exception as e:
                logger.warning("Failed to convert partitioned dataset '%s': %s", name, e)
                continue

        if not result:
            raise ValueError("PartitionedDataSetCollection contains no convertible datasets")

        return result

    # Handle vtkPartitionedDataSet - return dict with single DatasetCollection
    elif isinstance(vtk_dataset, vtkPartitionedDataSet):
        datasets = []
        num_partitions = vtk_dataset.GetNumberOfPartitions()

        logger.info("Processing vtkPartitionedDataSet with %d partitions", num_partitions)

        for i in range(num_partitions):
            partition = vtk_dataset.GetPartition(i)
            if partition is not None:
                try:
                    converted = _convert_single_dataset(partition, device=device)
                    datasets.append(converted)
                except (ValueError, TypeError) as e:
                    logger.warning("Skipping partition %d: %s", i, e)
                    continue

        if not datasets:
            raise ValueError("PartitionedDataSet contains no convertible partitions")

        return {"dataset": DatasetCollection.from_datasets(datasets)}

    # Handle single datasets - return dict with single Dataset
    else:
        return {"dataset": _convert_single_dataset(vtk_dataset, device=device)}


def _convert_single_dataset(vtk_dataset: vtkDataSet, device: Union[str, wp.context.Device, None] = None) -> Dataset:
    """Convert a single VTK dataset to a DAV Dataset.

    Args:
        vtk_dataset: Single VTK dataset (not composite)
        device: Device to create the dataset on

    Returns:
        Dataset: Converted DAV dataset

    Raises:
        ValueError: If the dataset type is not supported
    """
    # Wrap with numpy interface for easier access
    vtk_data = dsa.WrapDataObject(vtk_dataset)

    # Handle single datasets
    if isinstance(vtk_dataset, vtkStructuredGrid):
        from dav.data_models.vtk import structured_grid

        ds_handle = structured_grid.DatasetHandle()
        ds_handle.points = wp.array(vtk_data.Points, dtype=wp.vec3f, device=device)

        # Get extent from the structured grid
        extent = vtk_dataset.GetExtent()
        ds_handle.extent_min = wp.vec3i(extent[0], extent[2], extent[4])
        ds_handle.extent_max = wp.vec3i(extent[1], extent[3], extent[5])
        ds_handle.cell_bvh_id = 0  # Will be built when needed

        dataset = Dataset(data_model=structured_grid.DataModel, handle=ds_handle, device=device)

    elif isinstance(vtk_dataset, vtkUnstructuredGrid):
        from dav.data_models.vtk import unstructured_grid

        ds_handle = unstructured_grid.DatasetHandle()
        ds_handle.points = wp.array(vtk_data.Points, dtype=wp.vec3f, device=device)
        ds_handle.cell_types = wp.array(
            numpy_support.vtk_to_numpy(vtk_dataset.GetCellTypesArray()), dtype=wp.int32, device=device
        )
        ds_handle.cell_offsets = wp.array(
            numpy_support.vtk_to_numpy(vtk_dataset.GetCells().GetOffsetsArray()), dtype=wp.int32, device=device
        )
        ds_handle.cell_connectivity = wp.array(
            numpy_support.vtk_to_numpy(vtk_dataset.GetCells().GetConnectivityArray()), dtype=wp.int32, device=device
        )

        dataset = Dataset(data_model=unstructured_grid.DataModel, handle=ds_handle, device=device)

    elif isinstance(vtk_dataset, vtkImageData):
        from dav.data_models.vtk import image_data

        ds_handle = image_data.DatasetHandle()
        ds_handle.origin = wp.vec3f(vtk_dataset.GetOrigin())
        ds_handle.spacing = wp.vec3f(vtk_dataset.GetSpacing())
        extent = vtk_dataset.GetExtent()
        ds_handle.extent_min = wp.vec3i(extent[0], extent[2], extent[4])
        ds_handle.extent_max = wp.vec3i(extent[1], extent[3], extent[5])

        dataset = Dataset(data_model=image_data.DataModel, handle=ds_handle, device=device)

    elif isinstance(vtk_dataset, vtkPointSet):
        # Generic vtkPointSet (or vtkPolyData) - create unstructured grid with points only
        from dav.data_models.vtk import unstructured_grid

        logger.warning(
            "Converting %s to unstructured grid with points only (no cells). Full support for this dataset type is not yet implemented.",
            type(vtk_dataset).__name__,
        )

        ds_handle = unstructured_grid.DatasetHandle()
        ds_handle.points = wp.array(vtk_data.Points, dtype=wp.vec3f, device=device)
        # Create empty cell arrays
        ds_handle.cell_types = wp.empty(0, dtype=wp.int32, device=device)
        ds_handle.cell_offsets = wp.empty(0, dtype=wp.int32, device=device)
        ds_handle.cell_connectivity = wp.empty(0, dtype=wp.int32, device=device)

        dataset = Dataset(data_model=unstructured_grid.DataModel, handle=ds_handle, device=device)

    elif isinstance(vtk_dataset, vtkRectilinearGrid):
        raise ValueError(
            "vtkRectilinearGrid is not yet supported. Supported types: "
            "vtkStructuredGrid, vtkUnstructuredGrid, vtkImageData, and composite datasets."
        )

    else:
        raise ValueError(
            f"Unsupported VTK data type: {type(vtk_dataset).__name__}. "
            "Supported types: vtkStructuredGrid, vtkUnstructuredGrid, vtkImageData, "
            "vtkMultiBlockDataSet, vtkPartitionedDataSet, vtkPartitionedDataSetCollection."
        )

    # Load point data arrays
    point_data = vtk_data.PointData
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)

        # Determine if scalar or vector
        num_components = array.GetNumberOfComponents()
        if num_components == 1:
            field = Field.from_array(wp.array(array, dtype=wp.float32, device=device), AssociationType.VERTEX)
        elif num_components == 3:
            field = Field.from_array(wp.array(array, dtype=wp.vec3f, device=device), AssociationType.VERTEX)
        else:
            # Skip arrays with unsupported number of components
            logger.warning(
                "Skipping point data array '%s' with %d components (only 1 or 3 supported)", array_name, num_components
            )
            continue

        dataset.fields[array_name] = field

    # Load cell data arrays
    cell_data = vtk_data.CellData
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        array = cell_data.GetArray(i)

        # Determine if scalar or vector
        num_components = array.GetNumberOfComponents()
        if num_components == 1:
            field = Field.from_array(wp.array(array, dtype=wp.float32, device=device), AssociationType.CELL)
        elif num_components == 3:
            field = Field.from_array(wp.array(array, dtype=wp.vec3f, device=device), AssociationType.CELL)
        else:
            # Skip arrays with unsupported number of components
            logger.warning(
                "Skipping cell data array '%s' with %d components (only 1 or 3 supported)", array_name, num_components
            )
            continue

        dataset.fields[array_name] = field

    return dataset
