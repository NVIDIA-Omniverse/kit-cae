# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from logging import getLogger

import warp as wp

import dav
from dav.data_models import utils as data_models_utils

logger = getLogger(__name__)


@dav.kernel
def count_cell_faces_kernel(owner: wp.array(dtype=wp.int32), neighbour: wp.array(dtype=wp.int32), cell_face_counts: wp.array(dtype=wp.int32)):
    """Count how many faces each cell has.

    Handles two neighbour formats:
    1. Short format: neighbour.shape[0] < owner.shape[0] (only internal faces)
       - Internal faces: face_idx < neighbour.shape[0]
       - Boundary faces: face_idx >= neighbour.shape[0]
    2. Full format: neighbour.shape[0] == owner.shape[0]
       - Internal faces: neighbour[face_idx] >= 0 and neighbour[face_idx] != owner[face_idx]
       - Boundary faces: neighbour[face_idx] < 0 or neighbour[face_idx] == owner[face_idx]
    """
    face_idx = wp.tid()
    owner_cell = owner[face_idx]

    # Add face to owner cell (every face has an owner)
    if owner_cell >= 0 and owner_cell < cell_face_counts.shape[0]:
        wp.atomic_add(cell_face_counts, owner_cell, 1)

    # Check if this is an internal face and add to neighbour cell
    if face_idx < neighbour.shape[0]:
        neighbour_cell = neighbour[face_idx]
        # Internal face if neighbour is valid and different from owner
        if neighbour_cell >= 0 and neighbour_cell != owner_cell and neighbour_cell < cell_face_counts.shape[0]:
            wp.atomic_add(cell_face_counts, neighbour_cell, 1)


@dav.kernel
def fill_cell_faces_kernel(
    owner: wp.array(dtype=wp.int32),
    neighbour: wp.array(dtype=wp.int32),
    cell_face_offsets: wp.array(dtype=wp.int32),
    cell_faces: wp.array(dtype=wp.int32),
    cell_face_counts: wp.array(dtype=wp.int32),
):
    """Fill cell_faces array with face indices for each cell.

    Handles two neighbour formats:
    1. Short format: face_idx < neighbour.shape[0] => check neighbour[face_idx]
    2. Full format: face_idx < neighbour.shape[0] (always true) => check neighbour[face_idx]

    For internal faces (valid neighbour != owner), add to both owner and neighbour.
    For boundary faces, add only to owner.
    """
    face_idx = wp.tid()
    owner_cell = owner[face_idx]

    # Add face to owner cell (every face has an owner)
    if owner_cell >= 0 and owner_cell < cell_face_counts.shape[0]:
        offset = wp.atomic_add(cell_face_counts, owner_cell, 1)
        cell_faces[cell_face_offsets[owner_cell] + offset] = face_idx

    # Check if this is an internal face and add to neighbour cell
    if face_idx < neighbour.shape[0]:
        neighbour_cell = neighbour[face_idx]
        if neighbour_cell >= 0 and neighbour_cell != owner_cell and neighbour_cell < cell_face_counts.shape[0]:
            offset = wp.atomic_add(cell_face_counts, neighbour_cell, 1)
            cell_faces[cell_face_offsets[neighbour_cell] + offset] = face_idx


def compute_cell_face_connectivity(owner: wp.array, neighbour: wp.array, nb_cells: int = None) -> tuple[wp.array, wp.array]:
    """Compute cell-face connectivity from owner/neighbour arrays.

    OpenFOAM polyMesh provides owner/neighbour arrays that map faces to cells,
    but we need the inverse: which faces belong to each cell. This function
    computes cell_faces and cell_face_offsets from owner/neighbour.

    Args:
        owner: Array of cell IDs that own each face (size = num_faces, includes all faces)
        neighbour: Array of neighbor cell IDs (size <= num_faces).
                   Two formats supported:
                   1. Short format: size = num_internal_faces (< num_faces), only internal neighbors
                   2. Full format: size = num_faces, with -1 or owner_id for boundary faces
        nb_cells: Number of cells (optional, will be computed from max(owner) if not provided)

    Returns:
        tuple: (cell_faces, cell_face_offsets)
            - cell_faces: Flattened array of face indices for all cells
            - cell_face_offsets: Offsets into cell_faces (size = num_cells + 1)

    Algorithm:
        1. Determine number of cells from max(owner) if not provided
        2. Count faces per cell:
           - For all faces: add to owner cell
           - For faces with valid neighbour (>= 0 and != owner): also add to neighbour cell
        3. Create offsets array via exclusive scan
        4. Fill cell_faces by iterating through faces and adding to appropriate cells

    Note:
        Internal faces are added to BOTH the owner cell and the neighbour cell,
        so they appear twice in the cell_faces array.
        Boundary faces only appear in their owner cell.
    """
    # Compute number of cells if not provided
    if nb_cells is None:
        _, max_cell_id = data_models_utils.get_scalar_min_max(owner)
        nb_cells = int(max_cell_id + 1)
    else:
        nb_cells = int(nb_cells)

    if nb_cells <= 0:
        raise ValueError("nb_cells must be positive")

    nb_faces = int(owner.shape[0])  # number of all faces (internal and boundary)

    # Count how many faces each cell has
    cell_face_counts = wp.zeros(nb_cells, dtype=wp.int32, device=owner.device)
    wp.launch(count_cell_faces_kernel, dim=nb_faces, inputs=[owner, neighbour, cell_face_counts], device=owner.device)

    # Compute offsets via exclusive scan
    cell_face_offsets = wp.zeros(nb_cells + 1, dtype=wp.int32, device=owner.device)
    dav.utils.array_scan(cell_face_counts, cell_face_offsets, inclusive=False, add_trailing_sum=True)

    # Fill cell_faces array
    cell_face_counts.fill_(0)  # Reset counts to use as write positions
    nb_cell_faces = cell_face_offsets[-1:].numpy().item()
    cell_faces = wp.zeros(nb_cell_faces, dtype=wp.int32, device=owner.device)
    wp.launch(fill_cell_faces_kernel, dim=nb_faces, inputs=[owner, neighbour, cell_face_offsets, cell_faces, cell_face_counts], device=owner.device)
    return cell_faces, cell_face_offsets


# ================================================================================================================================
# Utility function to compute unique cell-point connectivity from face-based connectivity.
# ================================================================================================================================


@dav.kernel
def _count_unique_cell_points_kernel(
    cell_faces: wp.array(dtype=wp.int32),
    cell_face_offsets: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    face_offsets: wp.array(dtype=wp.int32),
    cell_point_counts: wp.array(dtype=wp.int32),
):
    """Count the number of unique points for each cell.

    Iterates over all faces of a cell, collecting point IDs and deduplicating
    using linear scan with a fixed-size scratch vector (O(n^2) per cell, but
    n is small for typical polyhedral cells).
    """
    cell_idx = wp.tid()
    cf_start = cell_face_offsets[cell_idx]
    cf_end = cell_face_offsets[cell_idx + 1]

    count = wp.int32(0)
    seen = wp.vec(length=dav.config.max_points_per_cell, dtype=wp.int32)

    for fi in range(cf_end - cf_start):
        global_face_id = cell_faces[cf_start + fi]
        f_start = face_offsets[global_face_id]
        f_end = face_offsets[global_face_id + 1]

        for pi in range(f_end - f_start):
            point_id = faces[f_start + pi]

            already_seen = wp.bool(False)
            for si in range(count):
                if seen[si] == point_id:
                    already_seen = True
                    break

            if not already_seen:
                assert count < dav.config.max_points_per_cell, "Exceeded maximum points per cell"
                if count < dav.config.max_points_per_cell:
                    seen[count] = point_id
                    count += 1

    cell_point_counts[cell_idx] = count


@dav.kernel
def _fill_unique_cell_points_kernel(
    cell_faces: wp.array(dtype=wp.int32),
    cell_face_offsets: wp.array(dtype=wp.int32),
    faces: wp.array(dtype=wp.int32),
    face_offsets: wp.array(dtype=wp.int32),
    cell_points_offsets: wp.array(dtype=wp.int32),
    cell_points: wp.array(dtype=wp.int32),
):
    """Fill cell_points with unique point IDs for each cell.

    Same deduplication logic as the counting kernel, but writes the results
    into the output array at the correct offsets.
    """
    cell_idx = wp.tid()
    cf_start = cell_face_offsets[cell_idx]
    cf_end = cell_face_offsets[cell_idx + 1]

    offset = cell_points_offsets[cell_idx]
    count = wp.int32(0)

    for fi in range(cf_end - cf_start):
        global_face_id = cell_faces[cf_start + fi]
        f_start = face_offsets[global_face_id]
        f_end = face_offsets[global_face_id + 1]

        for pi in range(f_end - f_start):
            point_id = faces[f_start + pi]

            already_seen = wp.bool(False)
            for si in range(count):
                if cell_points[offset + si] == point_id:
                    already_seen = True
                    break

            if not already_seen:
                assert count < dav.config.max_points_per_cell, "Exceeded maximum points per cell"
                assert offset + count < cell_points_offsets[cell_idx + 1], "Output cell_points array overflow"
                cell_points[offset + count] = point_id
                count += 1


def populate_cell_point_connectivity(handle):
    """Compute unique cell-point connectivity and populate handle.cell_points / handle.cell_points_offsets.

    For each cell, iterates through every face and collects unique point IDs
    (deduplicating vertices shared between faces).

    Algorithm:
        1. Count unique points per cell.
        2. Convert counts to offsets via exclusive scan with trailing sum.
        3. Fill the flattened cell_points array with unique point IDs per cell.

    Args:
        handle: A partially-filled DatasetHandle with cell_faces, cell_face_offsets,
                faces, and face_offsets already populated.

    Side-effects:
        Sets handle.cell_points and handle.cell_points_offsets.
    """
    nb_cells = handle.cell_face_offsets.shape[0] - 1
    device = handle.faces.device

    # Step 1: count unique points per cell
    cell_point_counts = wp.zeros(nb_cells, dtype=wp.int32, device=device)
    wp.launch(
        _count_unique_cell_points_kernel,
        dim=nb_cells,
        inputs=[handle.cell_faces, handle.cell_face_offsets, handle.faces, handle.face_offsets],
        outputs=[cell_point_counts],
        device=device,
    )

    # Step 2: exclusive scan → offsets
    cell_points_offsets = wp.zeros(nb_cells + 1, dtype=wp.int32, device=device)
    dav.utils.array_scan(cell_point_counts, cell_points_offsets, inclusive=False, add_trailing_sum=True)

    # Step 3: fill cell_points
    total_points = int(cell_points_offsets.numpy()[-1])
    cell_points = wp.zeros(total_points, dtype=wp.int32, device=device)
    wp.launch(
        _fill_unique_cell_points_kernel,
        dim=nb_cells,
        inputs=[handle.cell_faces, handle.cell_face_offsets, handle.faces, handle.face_offsets, cell_points_offsets],
        outputs=[cell_points],
        device=device,
    )

    handle.cell_points = cell_points
    handle.cell_points_offsets = cell_points_offsets


@dav.kernel
def _compute_face_centers_kernel(faces: wp.array(dtype=wp.int32), face_offsets: wp.array(dtype=wp.int32), points: wp.array(dtype=wp.vec3f), face_centers: wp.array(dtype=wp.vec3f)):
    """Compute face centers by averaging the coordinates of the points that make up each face."""
    face_idx = wp.tid()
    f_start = face_offsets[face_idx]
    f_end = face_offsets[face_idx + 1]

    center = wp.vec3f(0.0)
    count = wp.float32(0.0)

    for pi in range(f_end - f_start):
        point_id = faces[f_start + pi]
        point_coords = points[point_id]
        center += point_coords
        count += 1.0

    if count > 0.0:
        center /= count

    face_centers[face_idx] = center


def populate_face_centers(handle):
    """Compute face centers and populate handle.face_centers."""
    nb_faces = handle.faces.shape[0]
    device = handle.faces.device

    nb_faces = handle.face_offsets.shape[0] - 1
    assert handle.face_centers.shape[0] == nb_faces, "face_centers array must have the same length as number of faces"

    wp.launch(_compute_face_centers_kernel, dim=nb_faces, inputs=[handle.faces, handle.face_offsets, handle.points], outputs=[handle.face_centers], device=device)


if dav.config.compile_kernels_aot:
    from dav.core import aot

    logger.info("Compiling OpenFOAM utility kernels ...")
    wp.compile_aot_module(__name__, device=aot.get_devices())
