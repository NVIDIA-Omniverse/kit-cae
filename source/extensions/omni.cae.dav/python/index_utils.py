# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Utilities for converting DAV datasets to NVIDIA IndeX mesh storage format.

This module provides functions to extract face connectivity information from
DAV datasets and populate IndeX Mesh_storage structures using Warp kernels
that work with the DAV data model protocol.
"""

from logging import getLogger
from typing import Tuple

import dav
import numpy as np
import warp as wp
from omni.cae.index import bindings as index_bindings

logger = getLogger(__name__)


@dav.cached
def _get_count_faces_kernel(data_model: dav.DataModel):
    """Get cached kernel for counting faces per cell."""

    @wp.kernel(enable_backward=False, module="unique")
    def count_faces_kernel(
        ds: data_model.DatasetHandle, face_counts: wp.array(dtype=wp.int32), face_vtx_counts: wp.array(dtype=wp.int32)
    ):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            wp.atomic_add(face_counts, 0, wp.int32(num_faces))

            # Count vertices for all faces in this cell
            total_face_vtx = wp.int32(0)
            for face_idx in range(num_faces):
                num_face_pts = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
                total_face_vtx += num_face_pts

            wp.atomic_add(face_vtx_counts, 0, wp.int32(total_face_vtx))

    return count_faces_kernel


@dav.cached
def _get_copy_vertices_kernel(data_model: dav.DataModel):
    """Get cached kernel for copying vertices."""

    @wp.kernel(enable_backward=False, module="unique")
    def copy_vertices_kernel(ds: data_model.DatasetHandle, vertices: wp.array(ndim=1, dtype=wp.vec3f)):
        pt_idx = wp.tid()
        pt_id = data_model.DatasetAPI.get_point_id_from_idx(ds, pt_idx)
        pt = data_model.DatasetAPI.get_point(ds, pt_id)
        vertices[pt_idx] = pt

    return copy_vertices_kernel


@dav.cached
def _get_populate_cells_kernel(data_model: dav.DataModel):
    """Get cached kernel for populating cell face counts."""

    @wp.kernel(enable_backward=False, module="unique")
    def populate_cells_kernel(ds: data_model.DatasetHandle, cells: wp.array(ndim=2, dtype=wp.uint32)):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            cells[cell_idx, 0] = wp.uint32(num_faces)
        else:
            cells[cell_idx, 0] = wp.uint32(0)

    return populate_cells_kernel


@dav.cached
def _get_populate_faces_kernel(data_model: dav.DataModel):
    """Get cached kernel for populating face vertex counts."""

    @wp.kernel(enable_backward=False, module="unique")
    def populate_faces_kernel(
        ds: data_model.DatasetHandle, cells: wp.array(ndim=2, dtype=wp.uint32), faces: wp.array(ndim=2, dtype=wp.uint32)
    ):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            face_start_idx = wp.int32(cells[cell_idx, 1])

            for face_idx in range(num_faces):
                num_face_pts = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
                global_face_idx = face_start_idx + face_idx
                faces[global_face_idx, 0] = wp.uint32(num_face_pts)

    return populate_faces_kernel


@dav.cached
def _get_populate_face_vtx_indices_kernel(data_model: dav.DataModel):
    """Get cached kernel for populating face vertex indices."""

    @wp.kernel(enable_backward=False, module="unique")
    def populate_face_vtx_indices_kernel(
        ds: data_model.DatasetHandle,
        cells: wp.array(ndim=2, dtype=wp.uint32),
        faces: wp.array(ndim=2, dtype=wp.uint32),
        face_vtx_indices: wp.array(dtype=wp.uint32),
    ):
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            face_start_idx = wp.int32(cells[cell_idx, 1])

            for face_idx in range(num_faces):
                global_face_idx = face_start_idx + face_idx
                num_face_pts = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
                vtx_start_idx = wp.int32(faces[global_face_idx, 1])

                for local_idx in range(num_face_pts):
                    pt_id = data_model.CellAPI.get_face_point_id(cell, face_idx, local_idx, ds)
                    # Convert point ID to point index
                    pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
                    face_vtx_indices[vtx_start_idx + local_idx] = wp.uint32(pt_idx)

    return populate_face_vtx_indices_kernel


def compute_face_summary(dataset: dav.Dataset) -> Tuple[int, int]:
    """
    Compute the total number of faces and face vertices in a dataset.

    This scans all cells and counts:
    - Total number of faces across all cells
    - Total number of face vertex indices needed

    Args:
        dataset: The DAV dataset to analyze

    Returns:
        Tuple of (total_faces, total_face_vertices)

    Raises:
        ValueError: If the dataset has no cells
    """
    data_model = dataset.data_model
    device = dataset.device

    num_cells = dataset.get_num_cells()
    if num_cells == 0:
        raise ValueError("Cannot compute face summary for dataset with no cells")

    # Get cached kernel
    count_faces_kernel = _get_count_faces_kernel(data_model)

    face_counts = wp.zeros(1, dtype=wp.int32, device=device)
    face_vtx_counts = wp.zeros(1, dtype=wp.int32, device=device)

    wp.launch(count_faces_kernel, dim=num_cells, inputs=[dataset.handle, face_counts, face_vtx_counts], device=device)
    wp.synchronize_device(device)

    # Sum up the totals
    total_faces = int(face_counts.numpy()[0])
    total_face_vertices = int(face_vtx_counts.numpy()[0])

    return total_faces, total_face_vertices


def populate_mesh_storage(dataset: dav.Dataset, storage: index_bindings.Mesh_storage):
    """
    Populate IndeX Mesh_storage from a DAV dataset.

    This extracts the mesh topology and geometry from the dataset and fills
    the IndeX storage arrays with:
    - Vertex positions
    - Cell face counts and offsets
    - Face vertex counts and offsets
    - Face vertex indices
    - Cell face indices

    The implementation uses Warp kernels that work with the DAV data model protocol,
    allowing it to work with any data model (VTK, CGNS, etc.) that implements the
    protocol's CellAPI and DatasetAPI interfaces.

    Simple operations (prefix sum, sequential indices) are performed using numpy
    directly for better performance on CPU. Data-model-dependent operations use
    Warp kernels to leverage the protocol.

    Note: Currently does not share faces between cells - each cell has its own
    set of face definitions. This simplifies the implementation but uses more memory.

    Args:
        dataset: The DAV dataset containing the mesh (must be on CPU device)
        storage: The IndeX Mesh_storage to populate (must be pre-allocated with
                 generate_mesh_storage)

    Raises:
        ValueError: If the dataset is not on CPU device
    """
    data_model = dataset.data_model
    device = dataset.device

    # Only CPU device is supported
    if device != "cpu":
        raise ValueError(f"populate_mesh_storage only supports CPU device, got '{device}'")

    num_points = dataset.get_num_points()
    num_cells = dataset.get_num_cells()

    # Get face summary to know array sizes
    nb_faces, nb_face_vertices = compute_face_summary(dataset)

    # Create parameters structure
    params = index_bindings.Mesh_parameters()
    params.nb_vertices = num_points
    params.nb_cells = num_cells
    params.nb_faces = nb_faces
    params.nb_face_vtx_indices = nb_face_vertices
    params.nb_cell_face_indices = nb_faces  # No face sharing

    logger.info(
        f"Populating mesh storage: {num_points:,} vertices, {num_cells:,} cells, {nb_faces:,} faces, {nb_face_vertices:,} face vertices"
    )

    # 1. Populate vertices
    _populate_vertices(dataset, storage, params)

    # 2. Populate cells (face counts and start indices)
    _populate_cells(dataset, storage, params)

    # 3. Populate cell face indices
    _populate_cell_face_indices(dataset, storage, params)

    # 4. Populate faces (vertex counts and start indices)
    _populate_faces(dataset, storage, params)

    # 5. Populate face vertex indices
    _populate_face_vtx_indices(dataset, storage, params)


def _populate_vertices(
    dataset: dav.Dataset, storage: index_bindings.Mesh_storage, params: index_bindings.Mesh_parameters
):
    """Populate vertex positions from dataset."""
    data_model = dataset.data_model
    device = dataset.device

    # Get cached kernel
    copy_vertices_kernel = _get_copy_vertices_kernel(data_model)

    vertices_np = storage.get_vertices(params)
    vertices_wp = wp.array(vertices_np, dtype=wp.vec3f, copy=False, device=device)

    wp.launch(copy_vertices_kernel, dim=params.nb_vertices, inputs=[dataset.handle, vertices_wp], device=device)
    wp.synchronize_device(device)


def _populate_cells(dataset: dav.Dataset, storage: index_bindings.Mesh_storage, params: index_bindings.Mesh_parameters):
    """Populate cell array (face counts per cell)."""
    data_model = dataset.data_model
    device = dataset.device

    # Get cached kernel
    populate_cells_kernel = _get_populate_cells_kernel(data_model)

    cells_np = storage.get_cells(params)
    cells_wp = wp.array(cells_np, dtype=wp.uint32, copy=False, device=device)

    # Fill face counts
    wp.launch(populate_cells_kernel, dim=params.nb_cells, inputs=[dataset.handle, cells_wp], device=device)
    wp.synchronize_device(device)

    # Compute start indices (prefix sum) using numpy
    cells_np[:, 1] = 0
    cells_np[1:, 1] = np.cumsum(cells_np[:-1, 0])


def _populate_cell_face_indices(
    dataset: dav.Dataset, storage: index_bindings.Mesh_storage, params: index_bindings.Mesh_parameters
):
    """Populate cell face indices array."""
    # Since we're not sharing faces, cell face indices are just sequential
    cell_face_indices_np = storage.get_cell_face_indices(params)
    cell_face_indices_np[:] = np.arange(params.nb_cell_face_indices, dtype=np.uint32)


def _populate_faces(dataset: dav.Dataset, storage: index_bindings.Mesh_storage, params: index_bindings.Mesh_parameters):
    """Populate faces array (vertex counts per face)."""
    data_model = dataset.data_model
    device = dataset.device

    # Get cached kernel
    populate_faces_kernel = _get_populate_faces_kernel(data_model)

    cells_np = storage.get_cells(params)
    faces_np = storage.get_faces(params)

    cells_wp = wp.array(cells_np, dtype=wp.uint32, copy=False, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.uint32, copy=False, device=device)

    # Fill vertex counts
    wp.launch(populate_faces_kernel, dim=params.nb_cells, inputs=[dataset.handle, cells_wp, faces_wp], device=device)
    wp.synchronize_device(device)

    # Compute start indices (prefix sum) using numpy
    faces_np[:, 1] = 0
    faces_np[1:, 1] = np.cumsum(faces_np[:-1, 0])


def _populate_face_vtx_indices(
    dataset: dav.Dataset, storage: index_bindings.Mesh_storage, params: index_bindings.Mesh_parameters
):
    """Populate face vertex indices array."""
    data_model = dataset.data_model
    device = dataset.device

    # Get cached kernel
    populate_face_vtx_indices_kernel = _get_populate_face_vtx_indices_kernel(data_model)

    cells_np = storage.get_cells(params)
    faces_np = storage.get_faces(params)
    face_vtx_indices_np = storage.get_face_vtx_indices(params)

    cells_wp = wp.array(cells_np, dtype=wp.uint32, copy=False, device=device)
    faces_wp = wp.array(faces_np, dtype=wp.uint32, copy=False, device=device)
    face_vtx_indices_wp = wp.array(face_vtx_indices_np, dtype=wp.uint32, copy=False, device=device)

    wp.launch(
        populate_face_vtx_indices_kernel,
        dim=params.nb_cells,
        inputs=[dataset.handle, cells_wp, faces_wp, face_vtx_indices_wp],
        device=device,
    )
    wp.synchronize_device(device)


def allocate_attribute_storage(
    dataset: dav.Dataset,
    subset: index_bindings.IIrregular_volume_subset,
    field_names: list[str] = None,
    start_index: int = 0,
) -> list[dict]:
    """
    Allocate IndeX attribute storage for field data without populating it.

    This allocates storage space for attributes but does not copy field data.
    Useful when IndeX compute programs will populate the data later, or for
    setting up storage that will be filled by other means.

    Args:
        dataset: The DAV dataset containing the mesh and fields (must be on CPU device)
        subset: The IndeX irregular volume subset to allocate attributes for
        field_names: List of field names to allocate. If None, uses all fields from dataset.
        start_index: Starting attribute index (default: 0). Useful for allocating multiple timesteps.

    Returns:
        List of dictionaries containing allocation info for each field:
        - 'name': field name
        - 'index': attribute index
        - 'type': attribute type enum
        - 'affiliation': attribute affiliation enum
        - 'num_values': number of values
        - 'num_components': number of components (1-4)

    Raises:
        ValueError: If the dataset is not on CPU device
    """
    device = dataset.device

    # Only CPU device is supported
    if device != "cpu":
        raise ValueError(f"allocate_attribute_storage only supports CPU device, got '{device}'")

    # Get field names to process
    if field_names is None:
        field_names = list(dataset.get_field_names())

    if not field_names:
        logger.info("No fields to allocate in attribute storage")
        return []

    logger.info(f"Allocating storage for {len(field_names)} field(s), starting at index {start_index}")

    # Attribute types for scalar, vec2, vec3, vec4
    attrib_types = [
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_2,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_3,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_4,
    ]

    allocated_attributes = []

    for idx, field_name in enumerate(field_names):
        attr_index = start_index + idx

        if not dataset.has_field(field_name):
            logger.warning(f"Field '{field_name}' not found in dataset, skipping")
            continue

        field = dataset.get_field(field_name)

        # Get the field data to determine shape (but don't copy it)
        data = field.get_data().numpy()

        # Determine affiliation (vertex or cell)
        if field.association == dav.AssociationType.VERTEX:
            affiliation = index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX
        elif field.association == dav.AssociationType.CELL:
            affiliation = index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_CELL
        else:
            logger.error(f"Unsupported field association '{field.association}' for field '{field_name}', skipping")
            continue

        # Check if array shape is supported (scalar or vector up to 4 components)
        if data.ndim == 1:
            # Scalar field
            num_components = 1
        elif data.ndim == 2 and data.shape[1] <= 4:
            # Vector field
            num_components = data.shape[1]
        else:
            logger.error(f"Unsupported array shape {data.shape} for field '{field_name}', skipping")
            continue

        # Create attribute parameters
        params = index_bindings.Attribute_parameters()
        params.type = attrib_types[num_components - 1]
        params.affiliation = affiliation
        params.nb_attrib_values = data.shape[0]

        logger.info(
            f"  Field '{field_name}': {num_components} component(s), "
            f"{'vertex' if affiliation == index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX else 'cell'} data, "
            f"{params.nb_attrib_values:,} values (allocated, not populated)"
        )

        # Generate attribute storage (but don't populate it)
        storage = subset.generate_attribute_storage(attr_index, params)
        if storage is None:
            logger.error(f"Failed to generate attribute storage for field '{field_name}'")
            continue

        attrib_values = storage.get_attrib_values(params)
        attrib_values[:] = 300.0

        # Record allocation info
        allocated_attributes.append(
            {
                "name": field_name,
                "index": attr_index,
                "type": params.type,
                "affiliation": affiliation,
                "num_values": params.nb_attrib_values,
                "num_components": num_components,
            }
        )

    logger.info(f"Attribute storage allocation complete: {len(allocated_attributes)} field(s) allocated")
    return allocated_attributes


def fill_attribute_storage(
    dataset: dav.Dataset,
    subset: index_bindings.IIrregular_volume_subset,
    field_names: list[str] = None,
    start_index: int = 0,
):
    """
    Fill previously allocated IndeX attribute storage with field data from DAV dataset.

    This populates attribute storage that was already allocated (e.g., via allocate_attribute_storage).
    Copies field data from the dataset into the IndeX attribute storage arrays.

    Args:
        dataset: The DAV dataset containing the mesh and fields (must be on CPU device)
        subset: The IndeX irregular volume subset with pre-allocated attributes
        field_names: List of field names to populate. If None, uses all fields from dataset.
        start_index: Starting attribute index (default: 0). Useful if IDs occupy indices 0-1.

    Raises:
        ValueError: If the dataset is not on CPU device
    """
    device = dataset.device

    # Only CPU device is supported
    if device != "cpu":
        raise ValueError(f"fill_attribute_storage only supports CPU device, got '{device}'")

    # Get field names to process
    if field_names is None:
        field_names = list(dataset.get_field_names())

    if not field_names:
        logger.info("No fields to fill in attribute storage")
        return

    logger.info(f"Filling storage for {len(field_names)} field(s), starting at index {start_index}")

    # Attribute types for scalar, vec2, vec3, vec4
    attrib_types = [
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_2,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_3,
        index_bindings.Attribute_type.ATTRIB_TYPE_FLOAT32_4,
    ]

    for idx, field_name in enumerate(field_names):
        attr_index = start_index + idx

        if not dataset.has_field(field_name):
            logger.warning(f"Field '{field_name}' not found in dataset, skipping")
            continue

        field = dataset.get_field(field_name)

        # Get the field data as numpy array
        data = field.get_data().numpy()

        # Determine affiliation (vertex or cell)
        if field.association == dav.AssociationType.VERTEX:
            affiliation = index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX
        elif field.association == dav.AssociationType.CELL:
            affiliation = index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_CELL
        else:
            logger.error(f"Unsupported field association '{field.association}' for field '{field_name}', skipping")
            continue

        # Check if array shape is supported (scalar or vector up to 4 components)
        if data.ndim == 1:
            # Scalar field
            num_components = 1
        elif data.ndim == 2 and data.shape[1] <= 4:
            # Vector field
            num_components = data.shape[1]
        else:
            logger.error(f"Unsupported array shape {data.shape} for field '{field_name}', skipping")
            continue

        # Create attribute parameters (needed to access storage)
        params = subset.get_attribute_parameters(attr_index)

        if params is None:
            logger.error(f"Failed to get attribute parameters at index {attr_index} for field '{field_name}'")
            continue
        if params.type != attrib_types[num_components - 1]:
            logger.error(f"Attribute type mismatch at index {attr_index} for field '{field_name}'")
            continue
        if params.affiliation != affiliation:
            logger.error(f"Attribute affiliation mismatch at index {attr_index} for field '{field_name}'")
            continue
        if params.nb_attrib_values != data.shape[0]:
            logger.error(f"Attribute number of values mismatch at index {attr_index} for field '{field_name}'")
            continue

        logger.info(
            f"  Field '{field_name}' at index {attr_index}: {num_components} component(s), "
            f"{'vertex' if affiliation == index_bindings.Attribute_affiliation.ATTRIB_AFFIL_PER_VERTEX else 'cell'} data, "
            f"{params.nb_attrib_values:,} values"
        )

        # Get the pre-allocated attribute storage
        storage = subset.get_attribute(attr_index)
        if storage is None:
            logger.error(f"Failed to get attribute storage at index {attr_index} for field '{field_name}'")
            continue

        # Get storage array and copy data
        try:
            attrib_values = storage.get_attrib_values(params)
            np.copyto(attrib_values, data, casting="same_kind")
            subset.sync_device_storage(attr_index)
        except Exception as e:
            logger.error(f"Failed to copy data for field '{field_name}': {e}")
            continue

    logger.debug("Attribute storage fill complete")
