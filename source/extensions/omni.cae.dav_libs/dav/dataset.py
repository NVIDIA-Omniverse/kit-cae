# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = ["Dataset", "DatasetCollection"]


from logging import getLogger
from typing import Any

import warp as wp

from .data_models.collection import get_collection_data_model
from .data_models.typing import DataModel, DatasetHandle
from .locators import CellLocator
from .typing import FieldLike

logger = getLogger(__name__)


class FieldsMixin:
    _fields: dict[str, FieldLike]

    def _init_fields(self):
        """Initialize fields dictionary."""
        self._fields = {}

    def reset_fields(self):
        """Reset the fields dictionary to an empty dictionary.

        This is useful when you want to clear the fields dictionary and start fresh.
        """
        self._fields = {}

    def add_field(self, name: str, field: FieldLike, warn_if_exists: bool = True):
        """Add a field to the dataset.

        Args:
            name: Name of the field
            field: Field object to add
            warn_if_exists: If True, warn when replacing an existing field (default: True)
        """
        if not field:
            raise ValueError(f"Field '{name}' is None")
        if field.device != self.device:
            raise ValueError(f"Field '{name}' is on device '{field.device}', but dataset is on device '{self.device}'")
        if warn_if_exists and name in self._fields:
            logger.warning(f"Replacing existing field: '{name}'")
        self._fields[name] = field

    def get_field(self, name: str) -> FieldLike:
        """Get a field from the dataset.

        Args:
            name: Name of the field

        Returns:
            The requested field if it exists, otherwise None.
        """
        return self._fields.get(name)

    def remove_field(self, name: str):
        """Remove a field from the dataset.

        Args:
            name: Name of the field
        """
        del self._fields[name]

    def has_field(self, name: str) -> bool:
        """Check if a field exists in the dataset.

        Args:
            name: Name of the field
        """
        return name in self._fields

    def get_field_names(self) -> list[str]:
        """Get the names of all fields in the dataset.

        Returns:
            list[str]: A list of field names.
        """
        return list(self._fields.keys())

    @staticmethod
    def pass_fields(source: "FieldsMixin", target: "FieldsMixin"):
        """Pass the fields from a source dataset to a target dataset.

        Fields are passed by copying the dictionary of fields from the source to the target.
        The source and target datasets don't share the same dictionary of fields, but they share the same fields.
        This allows operators to return new Dataset instances with computed fields without modifying the
        input dataset.
        """
        if source.device != target.device:
            raise ValueError(f"Source dataset is on device '{source.device}', but target dataset is on device '{target.device}'")

        target._fields = source._fields.copy()


class CachedFieldsMixin:
    """Mixin providing cached field functionality.

    This mixin provides a caching mechanism for computed fields that are
    expensive to generate (e.g., cell sizes, cell centers). Cached fields
    are computed on-demand and stored for reuse.
    """

    _cached_fields: dict[str, FieldLike]

    def _init_cached_fields(self):
        """Initialize cached fields storage and register default field generators."""
        self._cached_fields = {}

    def get_cached_field(self, name: str) -> FieldLike:
        """Get a cached computed field, generating it if not available.

        Cached fields are computed properties of the dataset like cell sizes,
        cell centers, etc. They are computed once and cached for efficiency.

        Args:
            name: Name of the cached field (e.g., 'cell_sizes', 'cell_centers')

        Returns:
            FieldLike: The cached field

        Raises:
            ValueError: If the field name is not recognized
        """
        # Check if already cached
        if name in self._cached_fields:
            return self._cached_fields[name]

        # Generate the field
        if name == "cell_sizes":
            field = self._generate_cell_sizes(self)
        elif name == "cell_centers":
            field = self._generate_cell_centers(self)
        else:
            raise ValueError(f"Unknown cached field: {name}. Available fields: cell_sizes, cell_centers")

        self._cached_fields[name] = field
        return field

    def _generate_cell_sizes(self, instance) -> FieldLike:
        """Generate cell sizes field.

        Cell size is defined as the length of the diagonal of the cell's bounding box.

        Returns:
            FieldLike: Cell-associated scalar field containing cell sizes
        """
        from .operators import cell_sizes

        # Call compute which returns a new instance with the field
        result = cell_sizes.compute(instance, field_name="cell_sizes")
        field = result.get_field("cell_sizes")

        return field

    def _generate_cell_centers(self, instance) -> FieldLike:
        """Generate cell centers field.

        Cell center is the geometric centroid (average of vertex positions).

        Returns:
            Field: Cell-associated vector field containing cell centers
        """
        from .operators import centroid

        # Call compute which returns a new instance with the field
        result = centroid.compute(instance, field_name="cell_centers")
        field = result.get_field("cell_centers")

        return field

    @staticmethod
    def pass_cached_fields(source: "CachedFieldsMixin", target: "CachedFieldsMixin"):
        """Pass the cached fields from a source dataset to a target dataset.

        Cached fields are passed by referencing the same dictionary of cached fields between the source and target.
        That way all copies of the dataset will share the same cached fields.
        This avoids recomputing cached fields for each copy.
        """
        if source.device != target.device:
            raise ValueError(f"Source dataset is on device '{source.device}', but target dataset is on device '{target.device}'")

        target._cached_fields = source._cached_fields


class AccelerationStructuresMixin:
    """Mixin providing acceleration structures functionality.

    This mixin provides a mechanism for building and passing acceleration structures between datasets.
    """

    _accel_structs: dict[str, Any]

    def _init_accel_structs(self):
        """Initialize acceleration structures storage and register default acceleration structure generators."""
        self._accel_structs = {}

    def has_accel_struct(self, name: str) -> bool:
        """Check if an acceleration structure is present."""
        return name in self._accel_structs

    def get_accel_struct(self, name: str) -> Any:
        """Get an acceleration structure by name."""
        return self._accel_structs.get(name)

    def set_accel_struct(self, name: str, accel_struct: Any | None):
        """Set an acceleration structure by name."""
        self._accel_structs[name] = accel_struct

    @staticmethod
    def pass_accel_structs(source: "AccelerationStructuresMixin", target: "AccelerationStructuresMixin"):
        """Pass the acceleration structures from a source dataset to a target dataset.

        Acceleration structures are passed by referencing the same dictionary of acceleration structures between the source and target.
        That way all copies of the dataset will share the same acceleration structures.
        This avoids rebuilding acceleration structures for each copy.
        """
        if source.device != target.device:
            raise ValueError(f"Source dataset is on device '{source.device}', but target dataset is on device '{target.device}'")

        target._accel_structs = source._accel_structs


class Dataset(FieldsMixin, CachedFieldsMixin, AccelerationStructuresMixin):
    """
    Class representing a dataset in DAV. This follows the DatasetLike protocol.
    It's a container for a Data Model specific dataset handle and a few acceleration structures
    like cell locators, cell-sizes, etc. It also contains a dictionary of fields.
    """

    handle: DatasetHandle
    data_model: DataModel
    device: str

    def __init__(self, data_model: DataModel, handle: DatasetHandle, device: str | None = None, **kwargs):
        """Initialize a Dataset.

        Args:
            data_model: The data model defining dataset operations
            handle: The dataset structure (data model specific handle)
            device: Device to create the dataset on. If None, uses current Warp device.
        """
        self.handle = handle
        self.data_model = data_model
        self._kwargs = kwargs

        # Get device - use provided device or current Warp device
        if device is None:
            self.device = wp.get_device().alias
        else:
            self.device = device

        # Initialize mixins
        self._init_fields()
        self._init_cached_fields()
        self._init_accel_structs()

    def __getattr__(self, name: str) -> Any:
        if name in self._kwargs:
            return self._kwargs[name]
        raise AttributeError(f"Dataset has no attribute '{name}'")

    def build_cell_locator(self):
        """Build the cell locator for the dataset using the dataset's device."""
        if not self.has_accel_struct("cell_locator"):
            status, handle = self.data_model.DatasetAPI.build_cell_locator(self.data_model, self.handle, self.device)
            if not status:
                raise RuntimeError("Failed to build cell locator for dataset.")
            self.set_accel_struct("cell_locator", handle)

    def build_cell_links(self):
        """Build the cell links for the dataset using the dataset's device."""
        if not self.has_accel_struct("cell_links"):
            status, handle = self.data_model.DatasetAPI.build_cell_links(self.data_model, self.handle, self.device)
            if not status:
                raise RuntimeError("Failed to build cell links for dataset.")
            self.set_accel_struct("cell_links", handle)

    def get_num_points(self) -> int:
        """Get the number of points in the dataset."""
        return self.data_model.DatasetAPI.get_num_points(self.handle)

    def get_num_cells(self) -> int:
        """Get the number of cells in the dataset."""
        return self.data_model.DatasetAPI.get_num_cells(self.handle)

    def get_bounds(self) -> tuple[wp.vec3f, wp.vec3f]:
        """Get the bounding box of the dataset as (min_bounds, max_bounds) vectors.

        The bounds are computed once and cached for efficiency.

        Returns:
            tuple[wp.vec3f, wp.vec3f]: A tuple of (bounds_min, bounds_max) as warp vec3f values.
        """
        if not self.has_accel_struct("bounds"):
            from .operators import bounds

            logger.info("Computing dataset bounds")
            result = bounds.compute(self)
            bounds_min_field = result.get_field("bounds_min")
            bounds_max_field = result.get_field("bounds_max")

            # Extract the actual vec3f values from the fields
            bounds_min = bounds_min_field.get_data().numpy()[0]
            bounds_max = bounds_max_field.get_data().numpy()[0]

            # Convert to wp.vec3f
            bounds_min_vec = wp.vec3f(bounds_min[0], bounds_min[1], bounds_min[2])
            bounds_max_vec = wp.vec3f(bounds_max[0], bounds_max[1], bounds_max[2])

            self.set_accel_struct("bounds", (bounds_min_vec, bounds_max_vec))

        return self.get_accel_struct("bounds")

    def get_cell_bounds(self) -> tuple[wp.vec3f, wp.vec3f]:
        """
        Get the bounds of all cells in the dataset.
        """
        if not self.has_accel_struct("cell_bounds"):
            from .operators import cell_bounds

            logger.info("Computing dataset cell bounds")
            result = cell_bounds.compute(self)
            cell_bounds_min_field = result.get_field("cell_bounds_min")
            cell_bounds_max_field = result.get_field("cell_bounds_max")

            # Extract the actual vec3f values from the fields
            cell_bounds_min = wp.vec3f(cell_bounds_min_field.get_range(0)[0], cell_bounds_min_field.get_range(1)[0], cell_bounds_min_field.get_range(2)[0])
            cell_bounds_max = wp.vec3f(cell_bounds_max_field.get_range(0)[1], cell_bounds_max_field.get_range(1)[1], cell_bounds_max_field.get_range(2)[1])
            self.set_accel_struct("cell_bounds", (cell_bounds_min, cell_bounds_max))

        return self.get_accel_struct("cell_bounds")

    def shallow_copy(self, pass_fields: bool = True) -> "Dataset":
        """Create a shallow copy of this dataset.

        The copy shares the underlying handle, data_model, and cell_locator,
        but has its own separate fields dictionary if pass_fields is True. This allows operators to
        return new Dataset instances with computed fields without modifying the
        input dataset.

        Args:
            pass_fields: If True, pass the fields dictionary to the clone. This allows operators to
                return new Dataset instances with computed fields without modifying the
                input dataset.
                If False, the clone will have an empty fields dictionary.
                Default is True.

        Returns:
            Dataset: A new Dataset instance sharing the underlying data
        """
        clone = Dataset(self.data_model, self.handle, self.device, **self._kwargs)
        AccelerationStructuresMixin.pass_accel_structs(source=self, target=clone)
        CachedFieldsMixin.pass_cached_fields(source=self, target=clone)
        if pass_fields:
            FieldsMixin.pass_fields(source=self, target=clone)
        return clone


class DatasetCollection(FieldsMixin, CachedFieldsMixin, AccelerationStructuresMixin):
    """Collection of datasets that can be treated as a single unified dataset.

    This class wraps multiple Dataset instances into a collection with a unified API.
    All datasets must use the same data model and reside on the same device.

    Example usage:
        ```python
        from dav.dataset import Dataset, DatasetCollection

        # Create individual datasets
        dataset1 = Dataset(data_model, handle1, "cuda:0")
        dataset2 = Dataset(data_model, handle2, "cuda:0")

        # Create collection
        collection = DatasetCollection.from_datasets([dataset1, dataset2])
        ```
    """

    handle: DatasetHandle
    data_model: DataModel
    device: str
    base_data_model: DataModel  # The underlying data model for pieces
    datasets: list[Dataset]
    _kwargs: dict[str, Any]

    def __init__(self, handle: DatasetHandle, data_model: DataModel, base_data_model: DataModel, datasets: list[Dataset], device: str, **kwargs):
        """Initialize a DatasetCollection.

        Args:
            handle: Collection dataset handle
            data_model: Collection data model
            base_data_model: Base data model for individual pieces
            datasets: List of Dataset instances
            device: Device where datasets reside

        Note:
            Typically you should use from_datasets() instead of calling this directly.
        """
        self.handle = handle
        self.data_model = data_model
        self.base_data_model = base_data_model
        self.datasets = datasets
        self.device = device
        self._kwargs = kwargs

        # Initialize mixins
        self._init_fields()
        self._init_cached_fields()
        self._init_accel_structs()

    @staticmethod
    def from_datasets(datasets: list[Dataset]) -> "DatasetCollection":
        """Create a DatasetCollection from a list of Dataset instances.

        This method automatically creates FieldCollection objects for fields that are present
        in all pieces with matching dtypes and associations. Fields that are missing from
        some pieces or have mismatched properties are skipped with a warning.

        Args:
            datasets: List of Dataset instances (must all use the same data model and device)

        Returns:
            DatasetCollection wrapping the provided datasets with collected fields

        Raises:
            ValueError: If datasets list is empty, datasets use different models, or different devices

        Example:
            >>> dataset1 = Dataset(data_model, handle1, "cuda:0")
            >>> dataset2 = Dataset(data_model, handle2, "cuda:0")
            >>> collection = DatasetCollection.from_datasets([dataset1, dataset2])
        """
        if not datasets:
            raise ValueError("Cannot create DatasetCollection from empty list of datasets")

        # Get base data model and device from first dataset
        base_data_model = datasets[0].data_model
        device = datasets[0].device

        # Verify all datasets use the same model and device
        for i, dataset in enumerate(datasets[1:], 1):
            if dataset.data_model is not base_data_model:
                raise ValueError(f"All datasets must use the same data model. Dataset 0 and dataset {i} use different data models.")
            if dataset.device != device:
                raise ValueError(f"All datasets must be on the same device. Dataset 0 is on {device}, dataset {i} is on {dataset.device}")

        # Get collection data model
        collection_data_model = get_collection_data_model(base_data_model)

        # Create collection handle
        coll_handle = collection_data_model.DatasetHandle()
        coll_handle.pieces = wp.array([ds.handle for ds in datasets], dtype=base_data_model.DatasetHandle, device=device)
        coll_handle.piece_bvh_id = 0  # Will be set when piece_locator is built

        # Create collection
        collection = DatasetCollection(handle=coll_handle, data_model=collection_data_model, base_data_model=base_data_model, datasets=datasets, device=device)

        # Collect fields from all pieces
        collection._collect_fields_from_pieces(datasets)

        return collection

    def _collect_fields_from_pieces(self, datasets: list[Dataset]):
        """Collect fields from all pieces and create field collections.

        For each field name that exists in all pieces with matching properties,
        create a collection field and add it to the collection.

        Args:
            datasets: List of Dataset instances

        Note:
            Fields are only collected if they are present in ALL pieces and have
            matching dtypes, associations, and field models. Mismatched or missing
            fields are skipped with a warning.
        """
        from .field import Field

        if not datasets:
            return

        # Get all field names from the first dataset
        first_field_names = datasets[0].get_field_names()
        if not first_field_names:
            return

        # For each field name in the first dataset
        for field_name in first_field_names:
            try:
                # Collect the field from each piece
                piece_fields = []
                all_valid = True
                for i, dataset in enumerate(datasets):
                    if not dataset.has_field(field_name):
                        logger.warning(f"Skipping field '{field_name}': missing from piece {i}")
                        all_valid = False
                        break

                    piece_field = dataset.get_field(field_name)
                    piece_fields.append(piece_field)

                if all_valid and len(piece_fields) == len(datasets):
                    # Create collection field from the collected fields
                    field_collection = Field.from_fields(piece_fields)
                    self.add_field(field_name, field_collection, warn_if_exists=False)
                    logger.debug(f"Created field collection for field '{field_name}'")

            except Exception as e:
                logger.warning(f"Failed to create field collection for field '{field_name}': {e}")

    def _update_handle(self):
        """Update the handle with the new pieces (needed since dataset.build_cell_locator() may modify the handle)"""
        self.handle.pieces = wp.array([x.handle for x in self.datasets], dtype=self.base_data_model.DatasetHandle, device=self.device)
        piece_locator = self.get_accel_struct("piece_locator")
        self.handle.piece_bvh_id = piece_locator.get_bvh_id() if piece_locator is not None else 0

    def get_num_cells(self) -> int:
        """Get the number of cells in the collection."""
        return int(sum([x.get_num_cells() for x in self.datasets]))

    def get_num_points(self) -> int:
        """Get the number of points in the collection."""
        return int(sum([x.get_num_points() for x in self.datasets]))

    def get_bounds(self) -> tuple[wp.vec3f, wp.vec3f]:
        """Get the bounding box of the dataset collection as (min_bounds, max_bounds) vectors.

        The bounds are computed once and cached for efficiency. The collection bounds
        are computed by combining the bounds of all individual datasets.

        Returns:
            tuple[wp.vec3f, wp.vec3f]: A tuple of (bounds_min, bounds_max) as warp vec3f values.
        """
        if not self.has_accel_struct("bounds"):
            logger.info("Computing dataset collection bounds")
            # Get bounds from each dataset
            all_min_bounds = []
            all_max_bounds = []
            for dataset in self.datasets:
                bounds_min, bounds_max = dataset.get_bounds()
                all_min_bounds.append([bounds_min[0], bounds_min[1], bounds_min[2]])
                all_max_bounds.append([bounds_max[0], bounds_max[1], bounds_max[2]])

            # Compute overall min and max
            import numpy as np

            overall_min = np.minimum.reduce(all_min_bounds)
            overall_max = np.maximum.reduce(all_max_bounds)

            # Convert to wp.vec3f
            bounds_min_vec = wp.vec3f(overall_min[0], overall_min[1], overall_min[2])
            bounds_max_vec = wp.vec3f(overall_max[0], overall_max[1], overall_max[2])

            self.set_accel_struct("bounds", (bounds_min_vec, bounds_max_vec))

        return self.get_accel_struct("bounds")

    def shallow_copy(self, pass_fields: bool = True) -> "DatasetCollection":
        """Create a shallow copy of this dataset collection.

        The copy shares the underlying handle, data_model, and piece_locator,
        but has its own separate fields dictionary if pass_fields is True. This allows operators to
        return new DatasetCollection instances with computed fields without modifying the
        input dataset collection.

        Args:
            pass_fields: If True, pass the fields dictionary to the clone. This allows operators to
                return new DatasetCollection instances with computed fields without modifying the
                input dataset collection.
                If False, the clone will have an empty fields dictionary.
                Default is True.

        Returns:
            DatasetCollection: A new DatasetCollection instance sharing the underlying data
        """
        clone = DatasetCollection(handle=self.handle, data_model=self.data_model, base_data_model=self.base_data_model, datasets=self.datasets, device=self.device, **self._kwargs)
        AccelerationStructuresMixin.pass_accel_structs(source=self, target=clone)
        CachedFieldsMixin.pass_cached_fields(source=self, target=clone)
        if pass_fields:
            FieldsMixin.pass_fields(source=self, target=clone)
        return clone

    def build_cell_locator(self):
        """Build the cell locator for the collection."""

        min_bounds = []
        max_bounds = []
        if not self.has_accel_struct("piece_locator"):
            # call build_cell_locator for each dataset
            for dataset in self.datasets:
                dataset.build_cell_locator()
                bounds_min, bounds_max = dataset.get_bounds()
                min_bounds.append([bounds_min[0], bounds_min[1], bounds_min[2]])
                max_bounds.append([bounds_max[0], bounds_max[1], bounds_max[2]])

            min_bounds = wp.array(min_bounds, dtype=wp.vec3f, device=self.device)
            max_bounds = wp.array(max_bounds, dtype=wp.vec3f, device=self.device)
            piece_bvh = wp.Bvh(min_bounds, max_bounds)
            self.set_accel_struct("piece_locator", CellLocator(self.handle, piece_bvh))

            # update the handle with the new piece locator
            self._update_handle()

    def build_cell_links(self):
        """Build the cell links for the collection."""
        if not self.has_accel_struct("cell_links"):
            for dataset in self.datasets:
                dataset.build_cell_links()
            self.set_accel_struct("cell_links", None)
            self._update_handle()
