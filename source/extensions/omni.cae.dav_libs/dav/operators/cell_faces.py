# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
r"""
Cell Faces Operator
===================

This module extracts the boundary faces from 3D volumetric cells and creates
a surface mesh dataset following the UsdGeomMesh specification.

The operator iterates over all cells in the input dataset, extracts their faces,
and generates a surface mesh with unique vertices and face connectivity.

Key Features:
- Extracts faces from 3D volumetric cells (tetrahedra, hexahedra, etc.)
- Creates UsdGeomMesh-compatible surface mesh data model
- Handles cell fields by mapping them to face fields
- Supports multiple data models (VTK unstructured grid, CGNS, etc.)

Usage:
    ```python
    # Extract all cell faces to a surface mesh
    surface_dataset = cell_faces.compute(volume_dataset)
    ```
"""

from collections import namedtuple
from logging import getLogger

import warp as wp

import dav
from dav.data_models.custom import surface_mesh

logger = getLogger(__name__)

CellFacesKernels = namedtuple("CellFacesKernels", ["per_cell_face_counts", "per_face_vtx_counts", "extract_faces", "extract_points", "check_face_counts"])


@dav.cached(aot="operators.cell_faces")
def get_kernels(data_model: dav.DataModel) -> CellFacesKernels:
    """Generate kernels for counting, extracting, and validating cell faces."""

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_faces_per_cell_face_counts_kernel")
    def per_cell_face_counts_kernel(ds: data_model.DatasetHandle, face_counts_per_cell: wp.array(dtype=wp.int32)):
        """Store per-cell face counts."""
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            face_counts_per_cell[cell_idx] = num_faces

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_faces_per_face_vtx_counts_kernel")
    def per_face_vtx_counts_kernel(
        ds: data_model.DatasetHandle, nface_offsets: wp.array(dtype=wp.int32), vtx_counts_per_face: wp.array(dtype=wp.int32), face_to_cell: wp.array(dtype=wp.int32)
    ):
        """Store per-face vertex counts."""
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            face_start = nface_offsets[cell_idx]
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            for face_idx in range(num_faces):
                num_pts = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
                vtx_counts_per_face[face_start + face_idx] = num_pts
                face_to_cell[face_start + face_idx] = cell_idx

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_faces_extract_faces_kernel")
    def extract_faces_kernel(
        ds: data_model.DatasetHandle, nface_offsets: wp.array(dtype=wp.int32), ngon_offsets: wp.array(dtype=wp.int32), face_vertex_indices: wp.array(dtype=wp.int32)
    ):
        """Extract face connectivity using original point IDs."""
        cell_idx = wp.tid()
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)
        if data_model.CellAPI.is_valid(cell):
            face_start = nface_offsets[cell_idx]
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            for face_idx in range(num_faces):
                num_pts = data_model.CellAPI.get_face_num_points(cell, face_idx, ds)
                face_offset = ngon_offsets[face_start + face_idx]
                for local_idx in range(num_pts):
                    pt_id = data_model.CellAPI.get_face_point_id(cell, face_idx, local_idx, ds)
                    pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
                    face_vertex_indices[face_offset + local_idx] = pt_idx

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_faces_extract_points_kernel")
    def extract_points_kernel(
        ds: data_model.DatasetHandle,
        usage_mask: wp.array(dtype=wp.int32),
        scan_offsets: wp.array(dtype=wp.int32),
        points: wp.array(dtype=wp.vec3f),
        pt_idx_map: wp.array(dtype=wp.int32),
    ):
        """Extract all points from the original dataset."""
        pt_idx = wp.tid()
        if usage_mask[pt_idx] == 1:
            new_idx = scan_offsets[pt_idx]
            pt_id = data_model.DatasetAPI.get_point_id_from_idx(ds, pt_idx)
            points[new_idx] = data_model.DatasetAPI.get_point(ds, pt_id)
            pt_idx_map[new_idx] = pt_idx

    @dav.kernel(module="unique")
    @dav.utils.set_qualname("dav_cell_faces_check_face_counts_kernel")
    def check_face_counts_kernel(ds: data_model.DatasetHandle, sample_indices: wp.array(dtype=wp.int32), face_counts: wp.array(dtype=wp.int32)):
        """Check face counts for sampled cells (used by external_only validation)."""
        idx = wp.tid()
        cell_idx = sample_indices[idx]
        cell_id = data_model.DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
        cell = data_model.DatasetAPI.get_cell(ds, cell_id)

        if data_model.CellAPI.is_valid(cell):
            num_faces = data_model.CellAPI.get_num_faces(cell, ds)
            face_counts[idx] = num_faces

    return CellFacesKernels(per_cell_face_counts_kernel, per_face_vtx_counts_kernel, extract_faces_kernel, extract_points_kernel, check_face_counts_kernel)


def _validate_external_only_support(dataset: dav.DatasetLike, num_cells: int):
    """
    Validate that external_only option is supported for this dataset.

    Uses random sampling to check if cells have at most 1 face. If any sampled
    cell has more than 1 face, external_only filtering is not yet supported.

    Args:
        dataset: The dataset to validate
        num_cells: Total number of cells in the dataset

    Raises:
        NotImplementedError: If any sampled cell has more than 1 face
    """
    import numpy as np

    # Sample at most 10% of cells or 100 cells, whichever is smaller
    sample_size = min(max(1, num_cells // 10), 100)

    # Generate random cell indices to sample
    if num_cells <= sample_size:
        # Sample all cells if dataset is small
        sample_indices = list(range(num_cells))
    else:
        # Deterministic random sampling for reproducibility
        rng = np.random.default_rng(99)
        sample_indices = rng.choice(num_cells, size=sample_size, replace=False).tolist()

    logger.info(f"Validating external_only support by sampling {len(sample_indices)} of {num_cells} cells")

    # Get validation kernel
    check_face_counts_kernel = get_kernels(dataset.data_model).check_face_counts

    # Run the check on sampled cells
    device = dataset.device
    sample_indices_array = wp.array(sample_indices, dtype=wp.int32, device=device)
    face_counts = wp.zeros(len(sample_indices), dtype=wp.int32, device=device)

    wp.launch(check_face_counts_kernel, dim=len(sample_indices), inputs=[dataset.handle, sample_indices_array, face_counts], device=device)

    # Check results
    face_counts_np = face_counts.numpy()
    max_faces = face_counts_np.max()

    if max_faces > 1:
        raise NotImplementedError(
            f"external_only=True is not supported for this dataset. "
            f"Found cells with {max_faces} faces in sampled cells. "
            f"External face filtering for volumetric meshes is not yet implemented."
        )

    logger.info(f"Validation passed: all sampled cells have at most 1 face (max={max_faces})")


@dav.kernel
def _mark_used_points_kernel(face_vertex_indices: wp.array(dtype=wp.int32), usage_mask: wp.array(dtype=wp.int32)):
    """Mark points that are used by faces."""
    idx = wp.tid()
    pt_idx = face_vertex_indices[idx]
    usage_mask[pt_idx] = 1  # race condition is fine here since all threads write the same value


@dav.kernel
def _remap_ids_kernel(face_vertex_indices: wp.array(dtype=wp.int32), scan_offsets: wp.array(dtype=wp.int32)):
    """Remap point IDs to a compact range of vertex IDs used in the output mesh."""
    idx = wp.tid()
    pt_idx = face_vertex_indices[idx]
    new_id = scan_offsets[pt_idx]
    face_vertex_indices[idx] = new_id


def compute(dataset: dav.DatasetLike, external_only: bool = False) -> dav.Dataset:
    """
    Extract all cell faces and create a surface mesh dataset.

    This operator iterates over all 3D volumetric cells in the input dataset,
    extracts their boundary faces, and creates a new surface mesh dataset
    following the UsdGeomMesh specification.

    Args:
        dataset (dav.DatasetLike): Input dataset containing 3D volumetric cells.
        external_only (bool): If True, extract only external faces (boundary faces).
                              Currently only supported for datasets where all cells
                              have at most 1 face (i.e., they are already surface meshes).
                              Default: False (extract all faces).

    Returns:
        dav.Dataset: A new surface mesh dataset with the extracted faces.
                     The dataset uses the surface_mesh.DataModel.

    Raises:
        ValueError: If the input dataset has no cells with faces.
        NotImplementedError: If external_only=True and the dataset has cells with
                           multiple faces (internal face filtering not yet implemented).

    Note:
        - Point clouds and 2D surface meshes have no 3D cell faces (get_num_faces returns 0)
        - The output dataset contains a "cell_idx" field mapping faces to original cells
        - The output dataset contains a "point_idx" field mapping output points to original point indices
        - When external_only=True, a random sample of cells is checked to validate
          that all cells have at most 1 face before proceeding

    Example:
        >>> import dav
        >>> from dav.operators import cell_faces
        >>> # Load a volumetric dataset
        >>> volume_ds = dav.io.read_vtk("volume.vtu")
        >>> # Extract surface mesh
        >>> surface_ds = cell_faces.compute(volume_ds)
        >>> print(f"Extracted {surface_ds.get_num_cells()} faces")
        >>> # Extract only external faces (if dataset is already a surface)
        >>> external_faces = cell_faces.compute(volume_ds, external_only=True)
    """
    device = dataset.device
    num_cells = dataset.get_num_cells()

    if num_cells == 0:
        raise ValueError("Input dataset has no cells")

    # Validate external_only option with random sampling
    if external_only:
        _validate_external_only_support(dataset, num_cells)

    logger.info(f"Extracting faces from {num_cells} cells")

    # Get kernels
    kernels = get_kernels(dataset.data_model)

    # We'll use SIDS polyhedral data model to extract faces; essentially,
    # we're reverse building the nface/ngon structure from the cells.

    # Build build the nface block

    # Step 1: Store per-cell face counts
    with dav.scoped_timer("cell_faces.count"):
        face_counts_per_cell = wp.zeros(num_cells, dtype=wp.int32, device=device)
        wp.launch(kernels.per_cell_face_counts, dim=num_cells, inputs=[dataset.handle, face_counts_per_cell], device=device)

        nface_offsets = wp.zeros(num_cells + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(face_counts_per_cell, nface_offsets, inclusive=False, add_trailing_sum=True)
        del face_counts_per_cell

        nb_faces = nface_offsets[-1:].numpy()[0].tolist()
        if nb_faces == 0:
            raise ValueError("No faces found in dataset cells (input may be point cloud or 2D surface mesh)")

        vtx_counts_per_face = wp.zeros(nb_faces, dtype=wp.int32, device=device)
        face_to_cell = wp.zeros(nb_faces, dtype=wp.int32, device=device)
        wp.launch(kernels.per_face_vtx_counts, dim=num_cells, inputs=[dataset.handle, nface_offsets, vtx_counts_per_face, face_to_cell], device=device)

        ngon_offsets = wp.zeros(nb_faces + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(vtx_counts_per_face, ngon_offsets, inclusive=False, add_trailing_sum=True)
        del vtx_counts_per_face

        nb_face_vertices = ngon_offsets[-1:].numpy()[0].tolist()

        face_vertex_indices = wp.zeros(nb_face_vertices, dtype=wp.int32, device=device)
        wp.launch(kernels.extract_faces, dim=num_cells, inputs=[dataset.handle, nface_offsets, ngon_offsets, face_vertex_indices], device=device)

    logger.info(f"Found {nb_faces} faces with {nb_face_vertices} vertices")

    # Step 2: Get all points from original dataset
    with dav.scoped_timer("cell_faces.extract_points"):
        # Extract all point coordinates from the original dataset
        num_points = dataset.get_num_points()

        usage_mask = wp.zeros(num_points, dtype=wp.int32, device=device)
        wp.launch(_mark_used_points_kernel, dim=nb_face_vertices, inputs=[face_vertex_indices, usage_mask], device=device)

        scan_offsets = wp.zeros(num_points + 1, dtype=wp.int32, device=device)
        dav.utils.array_scan(usage_mask, scan_offsets, inclusive=False, add_trailing_sum=True)
        num_used_points = scan_offsets[-1:].numpy()[0].item()
        logger.info(f"{num_used_points} unique points are used by faces")

        new_points = wp.zeros(num_used_points, dtype=wp.vec3f, device=device)
        pt_idx_map = wp.zeros(num_used_points, dtype=wp.int32, device=device)
        wp.launch(kernels.extract_points, dim=num_points, inputs=[dataset.handle, usage_mask, scan_offsets, new_points, pt_idx_map], device=device)

        wp.launch(_remap_ids_kernel, dim=nb_face_vertices, inputs=[face_vertex_indices, scan_offsets], device=device)

    # Step 3: Create surface mesh dataset
    with dav.scoped_timer("cell_faces.create_surface_mesh"):
        surface_dataset = surface_mesh.create_dataset(new_points, face_vertex_indices, face_vertex_offsets=ngon_offsets)

    # Step 4: Add face-to-cell mapping field
    with dav.scoped_timer("cell_faces.add_fields"):
        cell_idx_field = dav.Field.from_array(face_to_cell, dav.AssociationType.CELL)
        surface_dataset.add_field("cell_idx", cell_idx_field)

        pt_idx_field = dav.Field.from_array(pt_idx_map, dav.AssociationType.VERTEX)
        surface_dataset.add_field("point_idx", pt_idx_field)

    logger.info(f"Created surface mesh with {num_used_points} vertices and {nb_faces} faces")
    return surface_dataset


if dav.config.compile_kernels_aot:
    from dav.core import aot

    for data_model in aot.get_data_models(specialization="operators.cell_faces.dataset"):
        logger.info(f"Compiling kernels for data model: {data_model}")
        for kernel in get_kernels(data_model):
            wp.compile_aot_module(kernel.module, device=aot.get_devices())

    # compile static kernels
    wp.compile_aot_module(__name__, device=aot.get_devices())
