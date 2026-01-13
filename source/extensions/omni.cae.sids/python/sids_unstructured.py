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
from typing import Any, Union

import numpy as np
import warp as wp
from omni.cae.data import IFieldArray, array_utils, progress, usd_utils
from omni.cae.data.typing import FieldArrayLike
from omni.cae.schema import cae, sids
from pxr import Usd

from .impl import kernels
from .impl.kernels import Elements_t
from .types import ElementType, get_vertex_count, get_vertex_counts

logger = getLogger(__name__)

_i32_limit = 2**31 - 1


def ceildiv(a, b):
    return -(a // -b)


def check_limit(val):
    i32_max = 2**31 - 1
    if val >= _i32_limit:
        raise RuntimeError("Currently quantities must fit in i32 range (%d >= %d)!" % (val, i32_max))


fa_int_types = (wp.int32, wp.uint32, wp.int64, wp.uint64)
fa_real_types = (wp.float32, wp.float64)
fa_types = fa_int_types + fa_real_types


def _get_unstructured_prim(prim) -> Usd.Prim:
    if not prim:
        raise usd_utils.QuietableException("Empty prim")
    if not prim.IsA(cae.DataSet):
        raise usd_utils.QuietableException(f"{prim} not a CaeDataSet")
    if not prim.HasAPI(sids.UnstructuredAPI):
        raise usd_utils.QuietableException(f"{prim} does not support CaeSidsUnstructuredAPI")
    return prim


@wp.kernel()
def _compute_start_offsets_kernel(
    offsets: wp.array(dtype=Any), conn: wp.array(dtype=Any), vertex_counts: wp.array(dtype=wp.int32), nb_cells: wp.int32
):
    """
    Kernel used to generate StartOffsets array when one is missing. While it's required by SIDS,
    datasets can sometimes have this array missing. Hence, we need to generate it at runtime.
    """
    pos = conn.dtype(0)
    for i in range(nb_cells):
        offsets[i] = pos

        e_type = wp.int32(conn[wp.int32(pos)])
        num_verts = vertex_counts[e_type]
        pos += conn.dtype(1) + conn.dtype(num_verts)
    offsets[nb_cells] = pos


# define kernel overloads for int types
for T in fa_int_types:
    wp.overload(_compute_start_offsets_kernel, {"offsets": wp.array(dtype=T), "conn": wp.array(dtype=T)})


def _compute_start_offsets(conn: FieldArrayLike, nb_cells) -> wp.array(dtype=wp.int32):
    """For MIXED element connectivity arrays, this can compute start offsets when missing."""
    with wp.ScopedDevice(device=array_utils.get_device(conn)):
        wp_conn = wp.array(conn, copy=False)
        wp_vertex_counts = wp.array(get_vertex_counts(), dtype=wp.int32, copy=False)
        wp_offsets = wp.empty(nb_cells + 1, dtype=array_utils.to_warp_dtype(conn))
        # this is not parallelizable, we still use Warp to speed up iteration
        wp.launch(_compute_start_offsets_kernel, 1, inputs=[wp_offsets, wp_conn, wp_vertex_counts, nb_cells])
        return wp_offsets


def get_element_type(prim) -> ElementType:
    """Returns the element type of the given prim."""
    prim = _get_unstructured_prim(prim)
    return ElementType[str(usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementType))]


def get_element_range(prim) -> tuple[int, int]:
    """Returns the element range of the given prim."""
    prim = _get_unstructured_prim(prim)
    element_range_start = usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementRangeStart)
    element_range_end = usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementRangeEnd)
    return element_range_start, element_range_end


async def get_section(prim: Usd.Prim, timeCode=Usd.TimeCode.EarliestTime(), limit=_i32_limit) -> Elements_t:
    """
    Convenience function to get a single section from a dataset. Raises an exception if there are
    multiple sections or no sections are found.
    """
    sections = await get_sections(prim, timeCode, limit)
    if len(sections) == 0:
        raise RuntimeError("No sections found in dataset")
    if len(sections) > 1:
        raise RuntimeError("Multiple sections not supported yet")
    return sections[0]


@progress.progress_context("Reading connectivity data (SIDS)")
async def get_sections(prim: Usd.Prim, timeCode=Usd.TimeCode.EarliestTime(), limit=_i32_limit) -> list[Elements_t]:
    prim = _get_unstructured_prim(prim)

    e_type = ElementType[str(usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementType))]
    element_range_start = usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementRangeStart, timeCode)
    element_range_end = usd_utils.get_attribute(prim, sids.Tokens.caeSidsElementRangeEnd, timeCode)
    if element_range_end < element_range_start or element_range_start == 0:
        raise usd_utils.QuietableException(f"Invalid element range ({element_range_start}, {element_range_end})")

    check_limit(element_range_start)
    check_limit(element_range_end)
    nb_elems = element_range_end - element_range_start + 1

    with progress.ProgressContext(scale=0.5):
        conn: np.ndarray = (
            await usd_utils.get_array_from_relationship(prim, sids.Tokens.caeSidsElementConnectivity, timeCode)
        ).numpy()

    # while len(conn) maybe greater than _i32_limit, the ids in the connectivity array must fit this limit.
    check_limit(np.amax(conn))

    sections = []
    if e_type == ElementType.MIXED or e_type == ElementType.NGON_n or e_type == ElementType.NFACE_n:
        with progress.ProgressContext(shift=0.5, scale=0.5):
            start_offset: Union[None, IFieldArray] = await usd_utils.get_array_from_relationship(
                prim, sids.Tokens.caeSidsElementStartOffset, timeCode, quiet=(e_type == ElementType.MIXED)
            )

        if start_offset is None:
            # SIDS required MIXED to have startoffset, however StaticMixer.cgns, a sample dataset provided by CGNS
            # has no start offset. So we compute it explicitly. Since this is not usual, we don't bother handling
            # large dataset for that case.
            logger.warning(
                "MIXED elements without start offsets. Computing start offsets. Please reconsider adding start offsets to the dataset."
            )
            # for now, I am not handling large arrays for this case as this case is very rare in practice given that SIDS
            # requres StartOffsets to be present.
            check_limit(conn.shape[0])
            start_offset = _compute_start_offsets(conn, nb_elems).numpy()
        else:
            assert isinstance(start_offset, IFieldArray)
            start_offset = start_offset.numpy()

        def _in_limit(chunks):
            nb_elems_per_chunk = ceildiv(nb_elems, chunks)
            count = start_offset[nb_elems_per_chunk - 1] - start_offset[0]
            if count >= limit:
                return False
            return True

        nb_chunks = 1
        while not _in_limit(nb_chunks):
            nb_chunks += 1
        logger.info("splitting into %d chunks", nb_chunks)

        nb_elems_per_chunk = ceildiv(nb_elems, nb_chunks)
        for idx in range(nb_chunks):
            start_idx = idx * nb_elems_per_chunk
            end_idx = min(start_idx + nb_elems_per_chunk, nb_elems) - 1

            offset_start = int(start_offset[start_idx])
            offset_end = int(start_offset[end_idx + 1] - 1)

            section = Elements_t()
            section.elementType = e_type.value
            section.elementRange = wp.vec2i(element_range_start + start_idx, element_range_start + end_idx)
            section.elementConnectivity = wp.array(conn[offset_start : offset_end + 1], dtype=wp.int32, copy=False)
            section.elementStartOffset = wp.array(
                start_offset[start_idx : end_idx + 2], dtype=wp.int64, copy=False
            )  # +2 to include the sentinel
            section.elementStartOffsetShift = offset_start
            sections.append(section)

        if e_type == ElementType.NFACE_n:
            # with NFACE, the element range seems to be off for datasets I am seeing.
            # From what I can understand, the element range should be [1, nb_elems]
            # but it begins after all NFACE_n element blocks ids. So we just adjust it.
            nface_e0 = sections[0].elementRange[0]
            for section in sections:
                section.elementRange[0] = section.elementRange[0] - nface_e0 + 1
                section.elementRange[1] = section.elementRange[1] - nface_e0 + 1
    else:
        # uniform
        nb_ids_per_elem = get_vertex_count(e_type)
        assert nb_ids_per_elem > 0
        assert conn.shape[0] == nb_elems * nb_ids_per_elem

        nb_chunks = 1
        while (ceildiv(nb_elems, nb_chunks) * nb_ids_per_elem) >= limit:
            nb_chunks += 1
        logger.info("splitting %s into %d chunks", prim, nb_chunks)

        nb_elems_per_chunk = ceildiv(nb_elems, nb_chunks)
        for idx in range(nb_chunks):
            start_idx = idx * nb_elems_per_chunk
            end_idx = min((idx + 1) * nb_elems_per_chunk - 1, nb_elems - 1)

            offset_start = start_idx * nb_ids_per_elem
            offset_end = (end_idx + 1) * nb_ids_per_elem - 1

            section = Elements_t()
            section.elementType = e_type.value
            section.elementRange = wp.vec2i(element_range_start + start_idx, element_range_start + end_idx)
            section.elementConnectivity = wp.array(conn[offset_start : offset_end + 1], dtype=wp.int32, copy=False)
            section.elementStartOffset = None
            section.elementStartOffsetShift = offset_start
            sections.append(section)

    return sections


@progress.progress_context("Computing grid coordinates ids (SIDS)")
async def get_grid_coordinates_ids(prim, timeCode, limit=_i32_limit, nb_verts=None) -> wp.array(dtype=wp.int32):
    """
    For an element block, returns the unique ids for the vertices used by elements in that block.
    For NFACE_n blocks, this will require processing of related NGON_n blocks as well.
    """
    e_type = get_element_type(prim)
    with progress.ProgressContext(scale=0.1):
        nb_verts = await get_nb_grid_coordinates(prim, timeCode) if nb_verts is None else nb_verts
        check_limit(nb_verts)

    if e_type == ElementType.NFACE_n:
        ngon_prims = usd_utils.get_target_prims(prim, sids.Tokens.caeSidsNgons)
        assert len(ngon_prims) > 0

    with progress.ProgressContext(shift=0.1, scale=0.9):
        mask = wp.zeros(nb_verts, dtype=wp.bool)
        sections = await get_sections(prim, timeCode, limit)
        for section in sections:
            if e_type == ElementType.NFACE_n:
                for ngon in ngon_prims:
                    logger.warning("processing %s", ngon)
                    ngon_sections = await get_sections(ngon, timeCode, limit)
                    for ngon_section in ngon_sections:
                        wp.launch(
                            kernels.compute_pt_mask_nface,
                            dim=section.elementRange[1] - section.elementRange[0] + 1,
                            inputs=[section, ngon_section, mask],
                        )
            else:
                wp.launch(
                    kernels.compute_pt_mask,
                    dim=section.elementRange[1] - section.elementRange[0] + 1,
                    inputs=[section, mask],
                )
        ids = np.arange(1, nb_verts + 1, dtype=np.int32)
        return wp.array(ids[mask], dtype=wp.int32, copy=False)


@progress.progress_context("Getting grid coordinates (SIDS)")
async def get_grid_coordinates(prim, timeCode, limit=_i32_limit, ids: wp.array = None) -> wp.array(dtype=wp.vec3f):
    with progress.ProgressContext(scale=0.4):
        g_coords: IFieldArray = await usd_utils.get_vecN_from_relationship(
            prim, sids.Tokens.caeSidsGridCoordinates, 3, timeCode
        )

    with progress.ProgressContext(shift=0.4, scale=0.4):
        ids: wp.array = (
            await get_grid_coordinates_ids(prim, timeCode, limit, nb_verts=g_coords.shape[0]) if ids is None else ids
        )

    g_coords_device = array_utils.get_device(g_coords)
    if not ids.device.is_cpu or not g_coords_device.is_cpu:
        raise RuntimeError("GPU arrays not supported yet")
    return wp.array(g_coords.numpy()[ids.numpy() - 1], dtype=wp.vec3f, copy=False)


@progress.progress_context("Getting grid coordinates (SIDS)")
async def get_original_grid_coordinates(prim, timeCode) -> wp.array(dtype=wp.vec3f):
    g_coords = await usd_utils.get_vecN_from_relationship(prim, sids.Tokens.caeSidsGridCoordinates, 3, timeCode)
    return wp.array(g_coords, dtype=wp.vec3f, copy=False)


@progress.progress_context("Computing number of vertices (SIDS)")
async def get_nb_grid_coordinates(prim, timeCode) -> int:
    prim = _get_unstructured_prim(prim)
    targets = usd_utils.get_target_prims(prim, sids.Tokens.caeSidsGridCoordinates)
    if len(targets) == 0:
        raise usd_utils.QuietableException("grid coordinates not found")
    coord0 = await usd_utils.get_array(targets[0], timeCode)
    return coord0.shape[0]


@progress.progress_context("Computing cell centers (SIDS)")
async def get_cell_centers(prim, timeCode, limit=_i32_limit) -> wp.array(dtype=wp.vec3f):
    e_type = get_element_type(prim)
    sections = await get_sections(prim, timeCode, limit)
    g_coords = await usd_utils.get_vecN_from_relationship(prim, sids.Tokens.caeSidsGridCoordinates, 3, timeCode)
    check_limit(g_coords.shape[0])

    nb_cells = sections[-1].elementRange[1] - sections[0].elementRange[0] + 1
    check_limit(nb_cells)

    centers = wp.zeros(shape=nb_cells, dtype=wp.vec3f)
    counts = wp.zeros(shape=nb_cells, dtype=wp.int32)
    g_coords = wp.array(g_coords, dtype=wp.vec3f, copy=False)

    if e_type == ElementType.NFACE_n:
        ngon_prims = usd_utils.get_target_prims(prim, sids.Tokens.caeSidsNgons)
        assert len(ngon_prims) > 0

    for section in sections:
        if e_type == ElementType.NFACE_n:
            for ngon in ngon_prims:
                logger.warning("processing %s", ngon)
                ngon_sections = await get_sections(ngon, timeCode, limit)
                for ngon_section in ngon_sections:
                    wp.launch(
                        kernels.compute_centroid_accumulate_nface,
                        dim=section.elementRange[1] - section.elementRange[0] + 1,
                        inputs=[section, ngon_section, g_coords, centers, counts],
                    )
        else:
            wp.launch(
                kernels.compute_centroid_accumulate,
                dim=section.elementRange[1] - section.elementRange[0] + 1,
                inputs=[section, g_coords, centers, counts],
            )
    wp.launch(kernels.compute_centroids_reduce, dim=nb_cells, inputs=[centers, counts])
    return centers


def get_mixed_types(section: Elements_t) -> wp.array(dtype=wp.int32):
    assert section.elementType == ElementType.MIXED.value
    if not section.elementConnectivity.device.is_cpu or not section.elementStartOffset.device.is_cpu:
        raise RuntimeError("GPU arrays not supported yet")
    elementConnectivity = section.elementConnectivity.numpy()
    offset = section.elementStartOffset.numpy() - section.elementStartOffsetShift
    return wp.array(elementConnectivity[offset[:-1]], dtype=wp.int32, copy=False)


@progress.progress_context("Converting cell data to point pata (SIDS)")
async def get_cell_field_as_point(
    prim: Usd.Prim, fieldPrims: list[Usd.Prim], nb_points=None, timeCode=Usd.TimeCode.EarliestTime()
) -> list[wp.array]:
    prim = _get_unstructured_prim(prim)
    fieldPrims = list(fieldPrims)

    if len(fieldPrims) == 0:
        return []

    if nb_points is None:
        with progress.ProgressContext("Computing number of points", scale=0.1):
            nb_points = await get_nb_grid_coordinates(prim, timeCode)

    for fieldPrim in fieldPrims:
        if usd_utils.get_attribute(fieldPrim, cae.Tokens.fieldAssociation, timeCode) != cae.Tokens.cell:
            raise usd_utils.QuietableException("field association must be cell (%s)" % fieldPrim)

    e_type = get_element_type(prim)
    with progress.ProgressContext("Reading cell data", shift=0.1, scale=0.1):
        cell_data = [array_utils.to_warp_array(a, copy=False) for a in await usd_utils.get_arrays(fieldPrims, timeCode)]
    nb_elems = cell_data[0].shape[0]
    check_limit(nb_elems)

    pt_data = [wp.zeros(nb_points, dtype=cd.dtype) for cd in cell_data]
    counts = wp.zeros(nb_points, dtype=wp.int32)

    max_nb_arrays = 3
    nb_arrays = len(cell_data)
    if nb_arrays > max_nb_arrays:
        raise usd_utils.QuietableException("too many cell arrays (%d > %d)" % (nb_arrays, max_nb_arrays))

    for idx in range(nb_arrays, max_nb_arrays):
        cell_data.append(wp.zeros(1, dtype=wp.float32))
        pt_data.append(wp.zeros(1, dtype=wp.float32))

    if e_type == ElementType.NFACE_n:
        ngon_prims = usd_utils.get_target_prims(prim, sids.Tokens.caeSidsNgons)
        assert len(ngon_prims) > 0

    with progress.ProgressContext(shift=0.2, scale=0.1):
        sections = await get_sections(prim, timeCode)

    if not sections:
        return pt_data[:nb_arrays]

    if nb_elems != (sections[-1].elementRange[1] - sections[0].elementRange[0] + 1):
        # sanity check to ensure the flow solution indeed refers to the elements of this grid
        raise usd_utils.QuietableException("cell field must have the same number of elements as the grid")

    logger.info(f"nface sections: {len(sections)}")
    for idx, section in enumerate(sections):
        logger.info(f"  section #{idx}")
        logger.info(f"    elementRange {section.elementRange[0]:,}, {section.elementRange[1]:,}")
        logger.info(f"    elementStartOffsetShift: {section.elementStartOffsetShift:,}")

    with progress.ProgressContext("Computing point data", shift=0.3, scale=0.65):
        if e_type == ElementType.NFACE_n:
            nb_ngons = len(ngon_prims)
            for ngon_idx, ngon in enumerate(ngon_prims):
                with progress.ProgressContext(
                    f"Processing ngon {ngon}", shift=ngon_idx / nb_ngons, scale=1 / nb_ngons
                ) as pctx:
                    ngon_sections = await get_sections(ngon, timeCode)
                    nb_ngons_sections = len(ngon_sections)
                    pctx.notify(1 / (nb_ngons_sections + 1))
                    for idx, ngon_section in enumerate(ngon_sections):
                        pctx.notify((1 + idx) / (nb_ngons_sections + 1))
                        logger.info("   processing ngon section #%d of %d", (idx + 1), len(ngon_sections))
                        # print("section: ", str(section.elementRange))
                        # print("ngon_section: ", str(ngon_section.elementRange))
                        for fs_idx, section in enumerate(sections):
                            logger.info("    on nface section #%d of %d", (fs_idx + 1), len(sections))
                            wp.launch(
                                kernels.cell_2_point_data_nface,
                                dim=section.elementRange[1] - section.elementRange[0] + 1,
                                inputs=[section, ngon_section, counts, nb_arrays] + cell_data + pt_data,
                            )
        else:
            # non-polyhedral elements
            for section in sections:
                wp.launch(
                    kernels.cell_2_point_data,
                    dim=section.elementRange[1] - section.elementRange[0] + 1,
                    inputs=[section, counts, nb_arrays] + cell_data + pt_data,
                )

    # print("ptArray = ", ptArray.numpy())
    # print("counts = ", counts.numpy())
    with progress.ProgressContext(shift=0.95, scale=0.05):
        for array in pt_data[:nb_arrays]:
            if array.dtype == wp.vec3f:
                wp.launch(kernels.cell_2_point_data_reduce_vec3f, dim=nb_points, inputs=[array, counts])
            else:
                wp.launch(kernels.cell_2_point_data_reduce_1d, dim=nb_points, inputs=[array, counts])

    return pt_data[:nb_arrays]
