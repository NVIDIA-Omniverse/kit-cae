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
Cell Subset Data Model
======================

This data model wraps an existing (parent) dataset and exposes only a selected
subset of its cells, while continuing to share the parent's full point set.
It is a parameterized data model: :func:`get_data_model` takes the parent's
data model and returns a new one that forwards most cell queries to it.

Use Cases
---------
- Masking / thresholding: expose only cells that pass a predicate (e.g. a
  quality filter or a scalar threshold) without copying point data.
- Extract-by-id: present an arbitrary ordered selection of parent cells as a
  standalone dataset.
- Zero-copy subsetting: the resulting dataset reuses the parent's point
  buffers and per-cell connectivity through delegation.

Semantics
---------
- **Cell IDs in the subset** are contiguous ``0 .. N-1`` where ``N`` is the
  length of ``inner_cell_indexes``. Subset ``cell_id`` equals subset
  ``cell_idx``.
- **``inner_cell_indexes[i]`` stores a parent cell *index*** (not a parent
  cell id). It is mapped through the parent's
  ``DatasetAPI.get_cell_id_from_idx`` before being stored on the cell handle.
  For data models where id == idx this is transparent; for models where they
  differ (e.g. SIDS), the array is still interpreted as indices.
- **Point set is shared.** ``get_num_points`` / ``get_point`` /
  ``get_point_id_from_idx`` all delegate to the parent unchanged. No point
  remapping is performed, so the subset exposes the parent's full point
  array even though some points may no longer be referenced by any cell.
- **Locators are subset-local.** The BVH / cell-links are built over the
  subset's cells and are independent from any locators on the parent.

Type System
-----------
- **Subset CellIdHandle / PointIdHandle**: ``wp.int32``.
- **Subset CellHandle** carries both the subset-local ``cell_id`` and the
  parent's ``inner_cell_id`` so that forwarded calls can reach into the
  parent cheaply.

Lifetime
--------
:func:`create_dataset` attaches the parent ``dav.Dataset`` as a keyword
argument on the returned dataset. This keeps the parent (and therefore the
underlying point / connectivity buffers) alive for the lifetime of the
subset.
"""

__all__ = ["create_dataset", "get_data_model"]

from typing import Any

import warp as wp

import dav
from dav import locators


@dav.cached
def get_data_model(inner_data_model):
    """Return a cell subset data model parameterized by ``inner_data_model``.

    The returned data model exposes a subset of the parent's cells (selected
    by an ``inner_cell_indexes`` array on the :class:`DatasetHandle`) while
    forwarding cell-level and point-level queries to ``inner_data_model``.

    This function is cached with :func:`dav.cached`, so requesting the same
    ``inner_data_model`` twice returns the same ``DataModel`` class (which
    matters for kernel caching and struct identity).

    Args:
        inner_data_model: The parent dataset's data model. Must define
            ``DatasetHandle``, ``CellIdHandle``, ``DatasetAPI``, ``CellAPI``,
            and ``CellLocatorAPI`` following the DAV data-model protocol.

    Returns:
        The cell-subset ``DataModel`` class parameterized by
        ``inner_data_model``.
    """

    @wp.struct
    class DatasetHandle:
        inner_cell_indexes: wp.array(dtype=wp.int32)
        """Parent-cell indices selected by this subset, ordered. Length defines the subset's cell count."""
        inner_handle: inner_data_model.DatasetHandle
        """Handle to the parent dataset; all point and per-cell-topology data live here."""
        cell_bvh_id: wp.uint64
        """BVH id for the subset-local cell locator (0 until ``build_cell_locator`` succeeds)."""
        cell_links: locators.CellLinks
        """Subset-local point-to-cell connectivity (empty until ``build_cell_links`` succeeds)."""

    @wp.struct
    class CellHandle:
        cell_id: wp.int32
        """Subset-local cell id (contiguous, 0..N-1)."""
        inner_cell_id: inner_data_model.CellIdHandle
        """Corresponding cell id in the parent dataset (what the inner model's APIs expect)."""

    class CellAPI:
        """Cell-level API for the subset; topology queries delegate to the parent."""

        @staticmethod
        @dav.func
        def is_valid(cell: CellHandle) -> wp.bool:
            """Return True if the subset cell handle is valid."""
            return cell.cell_id >= 0

        @staticmethod
        @dav.func
        def empty() -> CellHandle:
            """Return a sentinel invalid subset cell handle."""
            cell = CellHandle()
            cell.cell_id = -1
            return cell

        @staticmethod
        @dav.func
        def get_cell_id(cell: CellHandle) -> wp.int32:
            """Return the subset-local cell id."""
            return cell.cell_id

        @staticmethod
        @dav.func
        def get_num_points(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            """Number of points in the cell (delegates to the parent cell)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellAPI.get_num_points(inner_cell, ds.inner_handle)

        @staticmethod
        @dav.func
        def get_point_id(cell: CellHandle, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Point id for a cell-local vertex index (returns a parent point id, since the point set is shared)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellAPI.get_point_id(inner_cell, local_idx, ds.inner_handle)

        @staticmethod
        @dav.func
        def get_num_faces(cell: CellHandle, ds: DatasetHandle) -> wp.int32:
            """Number of faces in the cell (delegates to the parent cell)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellAPI.get_num_faces(inner_cell, ds.inner_handle)

        @staticmethod
        @dav.func
        def get_face_num_points(cell: CellHandle, face_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Number of points on a given face of the cell (delegates to the parent cell)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellAPI.get_face_num_points(inner_cell, face_idx, ds.inner_handle)

        @staticmethod
        @dav.func
        def get_face_point_id(cell: CellHandle, face_idx: wp.int32, local_idx: wp.int32, ds: DatasetHandle) -> wp.int32:
            """Point id for a face-local vertex index (delegates to the parent cell)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellAPI.get_face_point_id(inner_cell, face_idx, local_idx, ds.inner_handle)

    class DatasetAPI:
        """Dataset-level API for the subset.

        Cell ids and indices are identical in the subset (contiguous 0..N-1),
        so the id/idx mapping functions are the identity. Point queries are
        delegated directly to the parent.
        """

        @staticmethod
        @wp.func
        def get_cell_id_from_idx(ds: DatasetHandle, idx: wp.int32) -> wp.int32:
            """Map a subset cell index to its cell id (identity; they are equal in the subset)."""
            return idx

        @staticmethod
        @wp.func
        def get_cell_idx_from_id(ds: DatasetHandle, id: wp.int32) -> wp.int32:
            """Map a subset cell id to its index (identity; they are equal in the subset)."""
            return id

        @staticmethod
        @wp.func
        def get_cell(ds: DatasetHandle, id: wp.int32) -> CellHandle:
            """Build a cell handle for a subset cell id, resolving the parent cell id via ``inner_cell_indexes``."""
            assert id >= 0 and id < ds.inner_cell_indexes.shape[0], "id is invalid"
            cell_handle = CellHandle()
            cell_handle.cell_id = id
            cell_handle.inner_cell_id = inner_data_model.DatasetAPI.get_cell_id_from_idx(ds.inner_handle, ds.inner_cell_indexes[id])
            return cell_handle

        @staticmethod
        @wp.func
        def get_num_cells(ds: DatasetHandle) -> wp.int32:
            """Number of cells in the subset (length of ``inner_cell_indexes``)."""
            return ds.inner_cell_indexes.shape[0]

        @staticmethod
        @wp.func
        def get_num_points(ds: DatasetHandle) -> wp.int32:
            """Number of points — same as the parent, since the point set is shared."""
            return inner_data_model.DatasetAPI.get_num_points(ds.inner_handle)

        @staticmethod
        @wp.func
        def get_point_id_from_idx(ds: DatasetHandle, local_idx: wp.int32) -> wp.int32:
            """Delegate to the parent's point-index-to-id mapping."""
            return inner_data_model.DatasetAPI.get_point_id_from_idx(ds.inner_handle, local_idx)

        @staticmethod
        @wp.func
        def get_point_idx_from_id(ds: DatasetHandle, point_id: wp.int32) -> wp.int32:
            """Delegate to the parent's point-id-to-index mapping."""
            return inner_data_model.DatasetAPI.get_point_idx_from_id(ds.inner_handle, point_id)

        @staticmethod
        @wp.func
        def get_point(ds: DatasetHandle, point_id: wp.int32) -> wp.vec3f:
            """Fetch a point position from the parent dataset."""
            return inner_data_model.DatasetAPI.get_point(ds.inner_handle, point_id)

        @staticmethod
        def build_cell_locator(data_model, ds: DatasetHandle, device=None):
            """Build a subset-local cell BVH.

            On success, ``ds.cell_bvh_id`` is set to the new BVH id; on
            failure it is reset to 0. Returns ``(success, locator)``.
            """
            locator = locators.build_cell_locator(data_model, ds, device)
            if locator is not None:
                ds.cell_bvh_id = locator.get_bvh_id()
                return (True, locator)

            ds.cell_bvh_id = 0
            return (False, None)

        @staticmethod
        def build_cell_links(data_model, ds: DatasetHandle, device=None) -> tuple[bool, Any]:
            """Build subset-local point-to-cell links. Returns ``(success, links)``."""
            cell_links = locators.build_cell_links(data_model, ds, device)
            if cell_links is not None:
                ds.cell_links = cell_links
                return (True, cell_links)

            ds.cell_links = None
            return (False, None)

    @dav.func
    def _point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
        """Shared helper: test containment by resolving the parent cell and delegating."""
        inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
        return inner_data_model.CellLocatorAPI.point_in_cell(ds.inner_handle, point, inner_cell)

    class CellLocatorAPI:
        """Locator API for the subset.

        Point containment and interpolation delegate to the parent's cell
        locator. ``find_cell_containing_point`` uses the subset-local BVH
        (which indexes only the selected cells) and must be built via
        :meth:`DatasetAPI.build_cell_locator` before being called.
        """

        @staticmethod
        @dav.func
        def evaluate_position(ds: DatasetHandle, position: wp.vec3f, cell: CellHandle) -> dav.CellWeights:
            """Interpolation weights at ``position`` within ``cell`` (delegated to the parent)."""
            inner_cell = inner_data_model.DatasetAPI.get_cell(ds.inner_handle, cell.inner_cell_id)
            return inner_data_model.CellLocatorAPI.evaluate_position(ds.inner_handle, position, inner_cell)

        @staticmethod
        @dav.func
        def point_in_cell(ds: DatasetHandle, point: wp.vec3f, cell: CellHandle) -> wp.bool:
            """Whether ``point`` lies inside ``cell`` (delegated to the parent)."""
            return _point_in_cell(ds, point, cell)

        @staticmethod
        @dav.func
        def find_cell_containing_point(ds: DatasetHandle, position: wp.vec3f, hint: CellHandle) -> CellHandle:
            """Locate the subset cell containing ``position``.

            Tries ``hint`` first if valid, then queries the subset BVH and
            tests each candidate with the parent's point-in-cell. Returns
            :meth:`CellAPI.empty` if no cell contains the point. Requires
            :meth:`DatasetAPI.build_cell_locator` to have been called.
            """
            assert ds.cell_bvh_id != 0, "Cell locator not built for dataset"

            if CellAPI.is_valid(hint) and _point_in_cell(ds, position, hint):
                return hint

            radius = wp.vec3f(1.0e-6, 1.0e-6, 1.0e-6)
            query = wp.bvh_query_aabb(ds.cell_bvh_id, position - radius, position + radius)
            cell_idx = wp.int32(-1)
            while wp.bvh_query_next(query, cell_idx):
                cell_id = DatasetAPI.get_cell_id_from_idx(ds, cell_idx)
                cell = DatasetAPI.get_cell(ds, cell_id)
                assert CellAPI.is_valid(cell), "Queried cell from BVH is not valid"
                if _point_in_cell(ds, position, cell):
                    return cell

            return CellAPI.empty()

    CellLinksAPI = locators.get_cell_links_api(emptyPointId=wp.int32(0), emptyCellId=wp.int32(0), DatasetHandle=DatasetHandle, DatasetAPI=DatasetAPI)

    class DataModelMeta(type):
        def __repr__(cls):
            return f"DataModel (Cell Subset, inner:{inner_data_model})"

    class DataModel(metaclass=DataModelMeta):
        """Cell Subset data model parameterized by an inner (parent) data model."""

        pass

    DataModel.DatasetHandle = DatasetHandle
    DataModel.CellHandle = CellHandle
    DataModel.CellIdHandle = wp.int32
    DataModel.PointIdHandle = wp.int32
    DataModel.DatasetAPI = DatasetAPI
    DataModel.CellAPI = CellAPI
    DataModel.CellLinksAPI = CellLinksAPI
    DataModel.CellLocatorAPI = CellLocatorAPI

    return DataModel


def create_dataset(parent_dataset: dav.Dataset, cell_indexes: wp.array) -> dav.Dataset:
    """Create a dataset that exposes only the selected parent cells.

    The returned dataset shares the parent's point set and forwards cell
    topology to it. ``cell_indexes`` is interpreted as parent *cell indices*
    (not ids); the subset's cell ids will be ``0..len(cell_indexes)-1`` in
    the order given. The parent dataset is retained on the returned
    ``dav.Dataset`` to keep its buffers alive.

    Args:
        parent_dataset: The dataset whose cells are being subsetted.
        cell_indexes: 1-D ``wp.int32`` array of parent cell indices to
            expose. Must live on the same device as ``parent_dataset``.

    Returns:
        A ``dav.Dataset`` backed by a cell-subset data model parameterized
        by ``parent_dataset.data_model``.

    Raises:
        TypeError: If ``parent_dataset`` is not a ``dav.Dataset``.
        ValueError: If ``cell_indexes`` is not 1-D ``wp.int32``, or if the
            two inputs are on different devices.

    Example:
        >>> import warp as wp
        >>> from dav.data_models.custom import cell_subset
        >>> indices = wp.array([0, 2, 5], dtype=wp.int32, device=parent.device)
        >>> subset = cell_subset.create_dataset(parent, indices)
        >>> subset.get_num_cells()
        3
    """
    if cell_indexes is None or cell_indexes.ndim != 1:
        raise ValueError("cell_indexes must be a 1-D array")
    if cell_indexes.dtype != wp.int32:
        raise ValueError("cell_indexes must be of dtype wp.int32")
    if parent_dataset is None or not isinstance(parent_dataset, dav.Dataset):
        raise TypeError("parent_dataset must be a dav.Dataset")
    if parent_dataset.device != cell_indexes.device:
        raise ValueError("parent_dataset and cell_indexes must be on the same device")

    data_model = get_data_model(parent_dataset.data_model)
    handle = data_model.DatasetHandle()
    handle.inner_handle = parent_dataset.handle
    handle.inner_cell_indexes = cell_indexes
    handle.cell_bvh_id = 0
    handle.cell_links = locators.CellLinks()

    return dav.Dataset(data_model, handle=handle, device=parent_dataset.device, parent_dataset=parent_dataset)
