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


from collections.abc import Callable
from logging import getLogger
from typing import Any

import warp as wp

from .data_models.collection import get_collection_data_model
from .data_models.typing import DataModel, DatasetHandle
from .locators import CellLocator
from .typing import FieldLike

logger = getLogger(__name__)


class _FieldsMixin:
    fields: dict[str, FieldLike]

    def _init_fields(self):
        """Initialize fields dictionary."""
        self.fields = {}

    def add_field(self, name: str, field: FieldLike, warn_if_exists: bool = True):
        """Add a field to the dataset.

        Args:
            name: Name of the field
            field: Field object to add
            warn_if_exists: If True, warn when replacing an existing field (default: True)
        """
        if warn_if_exists and name in self.fields:
            logger.warning(f"Replacing existing field: '{name}'")
        self.fields[name] = field

    def get_field(self, name: str) -> FieldLike:
        """Get a field from the dataset.

        Args:
            name: Name of the field

        Returns:
            The requested field

        Raises:
            KeyError: If field not found
        """
        return self.fields[name]


class _CachedFieldsMixin:
    """Mixin providing cached field functionality.

    This mixin provides a caching mechanism for computed fields that are
    expensive to generate (e.g., cell sizes, cell centers). Cached fields
    are computed on-demand and stored for reuse.
    """

    _cached_fields: dict[str, FieldLike]
    _field_generators: dict[str, Callable[..., FieldLike]]

    def _init_cached_fields(self):
        """Initialize cached fields storage and register default field generators."""
        self._cached_fields = {}
        self._field_generators = {}

        # Register default field generators
        self._register_field_generator("cell_sizes", self._generate_cell_sizes)
        self._register_field_generator("cell_centers", self._generate_cell_centers)

    def _register_field_generator(self, name: str, generator: Callable[..., FieldLike]):
        """Register a field generator for a cached field.

        Args:
            name: Name of the cached field
            generator: Function that generates the field (takes self as argument)
        """
        self._field_generators[name] = generator

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

        # Check if we have a generator for this field
        if name not in self._field_generators:
            raise ValueError(f"Unknown cached field: {name}. Available fields: {list(self._field_generators.keys())}")

        # Generate and cache the field
        logger.info(f"Computing cached field: {name}")
        field = self._field_generators[name](self)
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


class Dataset(_FieldsMixin, _CachedFieldsMixin):
    """
    Class representing a dataset in DAV. This follows the DatasetLike protocol.
    It's a container for a Data Model specific dataset handle and a few acceleration structures
    like cell locators, cell-sizes, etc. It also contains a dictionary of fields.
    """

    handle: DatasetHandle
    data_model: DataModel
    device: str
    _cell_locator: Any | None
    _cell_links: Any | None
    _cell_locator_built: bool
    _cell_links_built: bool
    _cached_bounds: tuple[wp.vec3f, wp.vec3f] | None

    def __init__(self, data_model: DataModel, handle: DatasetHandle, device: str | None = None, **kwargs):
        """Initialize a Dataset.

        Args:
            data_model: The data model defining dataset operations
            handle: The dataset structure (data model specific handle)
            device: Device to create the dataset on. If None, uses current Warp device.
        """
        self.handle = handle
        self.data_model = data_model
        self._cell_locator = None
        self._cell_links = None
        self._cell_locator_built = False
        self._cell_links_built = False
        self._cached_bounds = None
        self._kwargs = kwargs

        # Get device - use provided device or current Warp device
        if device is None:
            self.device = wp.get_device().alias
        else:
            self.device = device

        # Initialize mixins
        self._init_fields()
        self._init_cached_fields()

    def __getattr__(self, name: str) -> Any:
        if name in self._kwargs:
            return self._kwargs[name]
        raise AttributeError(f"Dataset has no attribute '{name}'")

    def build_cell_locator(self):
        """Build the cell locator for the dataset using the dataset's device."""
        if not self._cell_locator_built:
            status, handle = self.data_model.DatasetAPI.build_cell_locator(self.data_model, self.handle, self.device)
            if not status:
                raise RuntimeError("Failed to build cell locator for dataset.")
            self._cell_locator = handle
            self._cell_locator_built = True

    def build_cell_links(self):
        """Build the cell links for the dataset using the dataset's device."""
        if not self._cell_links_built:
            status, handle = self.data_model.DatasetAPI.build_cell_links(self.data_model, self.handle, self.device)
            if not status:
                raise RuntimeError("Failed to build cell links for dataset.")
            self._cell_links = handle
            self._cell_links_built = True

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
        if self._cached_bounds is None:
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

            self._cached_bounds = (bounds_min_vec, bounds_max_vec)

        return self._cached_bounds

    def shallow_copy(self) -> "Dataset":
        """Create a shallow copy of this dataset.

        The copy shares the underlying handle, data_model, and cell_locator,
        but has its own separate fields dictionary. This allows operators to
        return new Dataset instances with computed fields without modifying the
        input dataset.

        Returns:
            Dataset: A new Dataset instance sharing the underlying data
        """
        copy = Dataset(self.data_model, self.handle, self.device)
        copy._cell_locator = self._cell_locator
        copy._cell_locator_built = self._cell_locator_built
        copy._cell_links = self._cell_links
        copy._cell_links_built = self._cell_links_built
        # Copy existing fields (shallow copy of the dict)
        copy.fields = self.fields.copy()
        copy._kwargs = self._kwargs.copy()
        return copy


class DatasetCollection(_FieldsMixin, _CachedFieldsMixin):
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
    _piece_locator: CellLocator | None
    _cell_links_built: bool
    _cached_bounds: tuple[wp.vec3f, wp.vec3f] | None

    def __init__(
        self,
        handle: DatasetHandle,
        data_model: DataModel,
        base_data_model: DataModel,
        datasets: list[Dataset],
        device: str,
    ):
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
        self._piece_locator = None
        self._cell_links_built = False
        self._cached_bounds = None

        # Initialize mixins
        self._init_fields()
        self._init_cached_fields()

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
                raise ValueError(
                    f"All datasets must use the same data model. Dataset 0 and dataset {i} use different data models."
                )
            if dataset.device != device:
                raise ValueError(
                    f"All datasets must be on the same device. Dataset 0 is on {device}, dataset {i} is on {dataset.device}"
                )

        # Get collection data model
        collection_data_model = get_collection_data_model(base_data_model)

        # Create collection handle
        coll_handle = collection_data_model.DatasetHandle()
        coll_handle.pieces = wp.array(
            [ds.handle for ds in datasets], dtype=base_data_model.DatasetHandle, device=device
        )
        coll_handle.piece_bvh_id = 0  # Will be set when piece_locator is built

        # Create collection
        collection = DatasetCollection(
            handle=coll_handle,
            data_model=collection_data_model,
            base_data_model=base_data_model,
            datasets=datasets,
            device=device,
        )

        # Collect fields from all pieces
        collection._collect_fields_from_pieces(datasets)

        return collection

    def _collect_fields_from_pieces(self, datasets: list[Dataset]):
        """Collect fields from all pieces and create FieldCollections.

        For each field name that exists in all pieces with matching properties,
        create a FieldCollection and add it to the collection.

        Args:
            datasets: List of Dataset instances

        Note:
            Fields are only collected if they are present in ALL pieces and have
            matching dtypes, associations, and field models. Mismatched or missing
            fields are skipped with a warning.
        """
        from .field import FieldCollection

        if not datasets:
            return

        # Get all field names from the first dataset
        first_fields = datasets[0].fields
        if not first_fields:
            return

        # For each field name in the first dataset
        for field_name in first_fields:
            try:
                # Collect the field from each piece
                piece_fields = []
                all_valid = True
                for i, dataset in enumerate(datasets):
                    if field_name not in dataset.fields:
                        logger.warning(f"Skipping field '{field_name}': missing from piece {i}")
                        all_valid = False
                        break

                    piece_field = dataset.fields[field_name]
                    piece_fields.append(piece_field)

                if all_valid and len(piece_fields) == len(datasets):
                    # Create FieldCollection from the collected fields
                    field_collection = FieldCollection.from_fields(piece_fields)
                    self.add_field(field_name, field_collection, warn_if_exists=False)
                    logger.debug(f"Created FieldCollection for field '{field_name}'")

            except Exception as e:
                logger.warning(f"Failed to create FieldCollection for field '{field_name}': {e}")

    def _update_handle(self):
        """Update the handle with the new pieces (needed since dataset.build_cell_locator() may modify the handle)"""
        self.handle.pieces = wp.array(
            [x.handle for x in self.datasets], dtype=self.base_data_model.DatasetHandle, device=self.device
        )
        self.handle.piece_bvh_id = self._piece_locator.get_bvh_id() if self._piece_locator is not None else 0

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
        if self._cached_bounds is None:
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

            self._cached_bounds = (bounds_min_vec, bounds_max_vec)

        return self._cached_bounds

    def shallow_copy(self) -> "DatasetCollection":
        """Create a shallow copy of this dataset collection.

        The copy shares the underlying handle, data_model, and piece_locator,
        but has its own separate fields dictionary. This allows operators to
        return new DatasetCollection instances with computed fields without modifying the
        input dataset collection.

        Returns:
            DatasetCollection: A new DatasetCollection instance sharing the underlying data
        """
        copy = DatasetCollection(
            handle=self.handle,
            data_model=self.data_model,
            base_data_model=self.base_data_model,
            datasets=self.datasets,
            device=self.device,
        )
        copy._piece_locator = self._piece_locator
        copy._cell_links_built = self._cell_links_built
        copy.fields = self.fields.copy()
        return copy

    def build_cell_locator(self):
        """Build the cell locator for the collection."""
        from .locators import CellLocator

        min_bounds = []
        max_bounds = []
        if self._piece_locator is None:
            # call build_cell_locator for each dataset
            for dataset in self.datasets:
                dataset.build_cell_locator()
                bounds_min, bounds_max = dataset.get_bounds()
                min_bounds.append([bounds_min[0], bounds_min[1], bounds_min[2]])
                max_bounds.append([bounds_max[0], bounds_max[1], bounds_max[2]])

            min_bounds = wp.array(min_bounds, dtype=wp.vec3f, device=self.device)
            max_bounds = wp.array(max_bounds, dtype=wp.vec3f, device=self.device)
            piece_bvh = wp.Bvh(min_bounds, max_bounds)
            self._piece_locator = CellLocator(self.handle, piece_bvh)

            # update the handle with the new piece locator
            self._update_handle()

    def build_cell_links(self):
        """Build the cell links for the collection."""
        if not self._cell_links_built:
            for dataset in self.datasets:
                dataset.build_cell_links()
            self._update_handle()
            self._cell_links_built = True
