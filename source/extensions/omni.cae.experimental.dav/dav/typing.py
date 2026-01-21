# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["DatasetLike", "FieldLike"]

from typing import Any, Protocol, runtime_checkable

import warp as wp

from .data_models.typing import DataModel, DatasetHandle
from .fields.typing import AssociationType, FieldHandle, FieldModel, InterpolatedFieldAPI


@runtime_checkable
class FieldLike(Protocol):
    """
    Protocol defining the interface for fields in DAV operations.

    The field protocol is intended for assembling data fields in the Python
    code (and not in Warp kernels). Hence, none of its methods should be
    annotated with `@wp.func` or called from Warp kernels.
    """

    handle: FieldHandle
    """The handle for the field."""

    field_model: FieldModel
    """The field model for the field."""

    device: str
    """The device for the field."""

    dtype: Any
    """The dtype for an element in the field."""

    association: AssociationType
    """The association type of the field."""

    def get_data(self) -> Any:
        """Get the underlying data containing field data.

        Returns:
            Any: The underlying data (wp.array, list[wp.array], wp.Volume, etc.)
        """
        ...

    def get_interpolated_field_api(self, data_model: DataModel) -> type[InterpolatedFieldAPI]:
        """Generate an interpolated field API for this field for a given data model.

        Args:
            data_model: Data model to generate API for

        Returns:
            InterpolatedFieldAPI class with get() method for field interpolation
        """
        ...


@runtime_checkable
class DatasetLike(Protocol):
    """Protocol defining the interface for datasets in DAV operations.

    The dataset protocol is intended for assembling data datasets in the Python
    code (and not in Warp kernels). Hence, none of its methods should be
    annotated with `@wp.func` or called from Warp kernels.
    """

    handle: DatasetHandle
    """The handle for the dataset."""

    data_model: DataModel
    """The data model for the dataset."""

    device: str
    """The device for the dataset."""

    def build_cell_locator(self):
        """Build the cell locator for the dataset."""
        ...

    def build_cell_links(self):
        """Build the cell links for the dataset."""
        ...

    def get_num_cells(self) -> int:
        """Get the number of cells in the dataset."""
        ...

    def get_num_points(self) -> int:
        """Get the number of points in the dataset."""
        ...

    def get_bounds(self) -> tuple[wp.vec3f, wp.vec3f]:
        """Get the bounding box of the dataset as (min_bounds, max_bounds) vectors."""
        ...

    def shallow_copy(self) -> "DatasetLike":
        """Create a shallow copy of the dataset."""
        ...

    def add_field(self, name: str, field: FieldLike, warn_if_exists: bool = True):
        """Add a field to the dataset."""
        ...

    def get_field(self, name: str) -> FieldLike:
        """Get a field from the dataset."""
        ...

    def get_cached_field(self, name: str) -> FieldLike:
        """Get a cached field from the dataset.

        Cached fields are computed properties of the dataset like cell sizes,
        cell centers, etc. They are computed once and cached for efficiency.

        Args:
            name: Name of the cached field (e.g., 'cell_sizes', 'cell_centers')

        Returns:
            FieldLike: The cached field.
        """
        ...
