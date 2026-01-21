# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = ["Field", "FieldCollection"]

from typing import Any

import warp as wp

from ._helpers import cached
from .data_models.typing import DataModel
from .fields.typing import AssociationType, FieldHandle, FieldModel, InterpolatedFieldAPI


@cached
def _create_interpolated_field_api(data_model: DataModel, field_model: FieldModel) -> type[InterpolatedFieldAPI]:
    """Create an interpolated field API for a given data model and field model combination.

    This is an internal function used by Field and FieldCollection classes.

    This function generates a specialized InterpolatedFieldAPI class that works with
    the provided data model and field model. It handles both cell-associated and
    vertex-associated fields with proper interpolation.

    Args:
        data_model: The data model defining dataset operations
        field_model: The field model for field operations (may be a collection model)

    Returns:
        InterpolatedFieldAPI class with get() method for field interpolation
    """
    if hasattr(data_model.DatasetAPI, "create_interpolated_field_api"):
        # Defer to data model to create the interpolated field API, if the data model implements it.
        return data_model.DatasetAPI.create_interpolated_field_api(field_model)

    @wp.func
    def get_cell_value(ds: data_model.DatasetHandle, field: field_model.FieldHandle, cell: data_model.CellHandle):
        cell_id = data_model.CellAPI.get_cell_id(cell)
        cell_idx = data_model.DatasetAPI.get_cell_idx_from_id(ds, cell_id)
        return field_model.FieldAPI.get(field, cell_idx)

    @wp.func
    def get_point_value(
        ds: data_model.DatasetHandle, field: field_model.FieldHandle, cell: data_model.CellHandle, position: wp.vec3f
    ):
        i_cell = data_model.DatasetAPI.evaluate_position(ds, position, cell)
        nb_pts = data_model.CellAPI.get_num_points(cell, ds)
        value = field_model.FieldAPI.zero()

        for i in range(nb_pts):
            weight = data_model.InterpolatedCellAPI.get_weight(i_cell, i)
            pt_id = data_model.CellAPI.get_point_id(cell, i, ds)
            pt_idx = data_model.DatasetAPI.get_point_idx_from_id(ds, pt_id)
            pt_value = field_model.FieldAPI.get(field, pt_idx)
            value += pt_value * type(field_model.FieldAPI.zero_s())(weight)

        return value

    class GenericInterpolatedFieldAPI:
        @staticmethod
        @wp.func
        def get(
            ds: data_model.DatasetHandle,
            field: field_model.FieldHandle,
            cell: data_model.CellHandle,
            position: wp.vec3f,
        ):
            if not data_model.CellAPI.is_valid(cell):
                wp.printf("Cell is not valid\n")
                return field_model.FieldAPI.zero()
            elif field_model.FieldAPI.get_association(field) == wp.static(AssociationType.VERTEX.value):
                return get_point_value(ds, field, cell, position)
            elif field_model.FieldAPI.get_association(field) == wp.static(AssociationType.CELL.value):
                return get_cell_value(ds, field, cell)
            else:
                wp.printf("Unsupported association type: %d\n", field_model.FieldAPI.get_association(field))
                return field_model.FieldAPI.zero()

    return GenericInterpolatedFieldAPI


class Field:
    handle: FieldHandle
    field_model: FieldModel
    data: Any
    device: Any
    dtype: Any

    def __init__(self, handle: FieldHandle, field_model: FieldModel, data: Any, dtype: Any, device: Any):
        self.handle = handle
        self.field_model = field_model
        self.data = data
        self.dtype = dtype
        self.device = device

    @property
    def association(self) -> AssociationType:
        """Get the field association type.

        Returns:
            AssociationType: The association type (VERTEX or CELL)
        """
        return AssociationType(self.handle.association)

    def get_data(self) -> Any:
        """Get the underlying data containing field data.

        Returns:
            Any: The underlying data (wp.array, list[wp.array], wp.Volume, etc.)
        """
        return self.data

    @staticmethod
    def from_array(data: wp.array, association: AssociationType) -> "Field":
        """Create a Field from a single warp array.

        This method supports:
        - Scalar fields: data is 1D array of scalar type (float32, float64, int32)
        - Vector fields (AoS): data is 1D array of vector type (vec3f, vec3d, vec3i)

        Args:
            data: Warp array containing the field data
            association: Field association (VERTEX or CELL)

        Returns:
            Field: A new Field instance

        Raises:
            ValueError: If data dtype is not supported

        Example:
            >>> import warp as wp
            >>> from dav.field import Field
            >>> from dav.fields import AssociationType
            >>>
            >>> # Scalar field
            >>> data = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
            >>> field = Field.from_array(data, AssociationType.VERTEX)
            >>>
            >>> # Vector field (AoS)
            >>> data = wp.array([[1,2,3], [4,5,6]], dtype=wp.vec3f)
            >>> field = Field.from_array(data, AssociationType.CELL)
        """
        from . import fields

        # Get device from array
        device = data.device.alias

        # Get value dtype from array
        value_dtype = data.dtype

        # Get field model (storage dtype same as value dtype for single array)
        field_model = fields.array.get_field_model(value_dtype, storage_dtype=value_dtype)

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        handle.data = data

        return Field(handle=handle, field_model=field_model, data=data, dtype=value_dtype, device=device)

    @staticmethod
    def from_arrays(data: list[wp.array], association: AssociationType) -> "Field":
        """Create a Field from multiple warp arrays (SoA storage).

        This method creates vector fields using Structure-of-Arrays (SoA) storage
        where each component is stored in a separate array. All arrays must have
        the same dtype (the scalar component type), length, and device.

        Args:
            data: List of warp arrays (must be 3 arrays for vec3 types)
            association: Field association (VERTEX or CELL)

        Returns:
            Field: A new Field instance with SoA storage

        Raises:
            ValueError: If arrays have different dtypes, lengths, devices, or invalid count

        Example:
            >>> import warp as wp
            >>> from dav.field import Field
            >>> from dav.fields import AssociationType
            >>>
            >>> # Vector field with SoA storage
            >>> x = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
            >>> y = wp.array([4.0, 5.0, 6.0], dtype=wp.float32)
            >>> z = wp.array([7.0, 8.0, 9.0], dtype=wp.float32)
            >>> field = Field.from_arrays([x, y, z], AssociationType.VERTEX)
        """
        from . import fields

        if not data:
            raise ValueError("data list cannot be empty")

        if len(data) != 3:
            raise ValueError(f"Expected 3 arrays for vector field, got {len(data)}")

        # Get device from first array
        device = data[0].device.alias

        # Verify all arrays have same dtype
        scalar_dtype = data[0].dtype
        for i, arr in enumerate(data[1:], 1):
            if arr.dtype != scalar_dtype:
                raise ValueError(
                    f"All arrays must have the same dtype. Array 0 has dtype {scalar_dtype}, array {i} has dtype {arr.dtype}"
                )

        # Verify all arrays have same length
        length = data[0].shape[0]
        for i, arr in enumerate(data[1:], 1):
            if arr.shape[0] != length:
                raise ValueError(
                    f"All arrays must have the same length. Array 0 has length {length}, array {i} has length {arr.shape[0]}"
                )

        # Verify all arrays are on the same device
        for i, arr in enumerate(data[1:], 1):
            if arr.device.alias != device:
                raise ValueError(
                    f"All arrays must be on the same device. Array 0 is on {device}, array {i} is on {arr.device.alias}"
                )

        # Determine value dtype (vec3 type from scalar type)
        if scalar_dtype == wp.float32:
            value_dtype = wp.vec3f
        elif scalar_dtype == wp.float64:
            value_dtype = wp.vec3d
        elif scalar_dtype == wp.int32:
            value_dtype = wp.vec3i
        else:
            raise ValueError(f"Unsupported scalar dtype: {scalar_dtype}. Supported types: float32, float64, int32")

        # Get field model (SoA: storage dtype is scalar, value dtype is vector)
        field_model = fields.array.get_field_model(value_dtype, storage_dtype=scalar_dtype)

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        handle.data = data[0]
        handle.data_1 = data[1]
        handle.data_2 = data[2]

        return Field(handle=handle, field_model=field_model, data=data, dtype=value_dtype, device=device)

    @staticmethod
    def from_volume(
        volume: wp.Volume,
        dims: wp.vec3i,
        association: AssociationType,
        indexing: str = "xy",
        origin: wp.vec3i = wp.vec3i(0, 0, 0),
    ) -> "Field":
        """Create a Field from a warp Volume (NanoVDB).

        Args:
            volume: Warp Volume containing NanoVDB data
            dims: Volume dimensions (Ni, Nj, Nk)
            association: Field association (VERTEX or CELL)
            indexing: Index ordering - "ij" (Fortran) or "xy" (Cartesian). Default: "xy"
            origin: Volume origin (Ni, Nj, Nk). Default: (0, 0, 0)

        Returns:
            Field: A new Field instance backed by NanoVDB

        Raises:
            ValueError: If volume dtype is not supported or indexing is invalid

        Example:
            >>> import warp as wp
            >>> from dav.field import Field
            >>> from dav.fields import AssociationType
            >>>
            >>> # Scalar volume field
            >>> volume = wp.Volume.allocate(min=[0,0,0], max=[32,32,32], voxel_size=1.0)
            >>> field = Field.from_volume(volume, (0, 0, 0), (32, 32, 32), AssociationType.VERTEX)
        """
        from . import fields

        # Get volume dtype
        if not hasattr(volume, "dtype"):
            raise ValueError("Volume must have a dtype attribute")

        dtype = volume.dtype
        if dtype not in [wp.float32, wp.vec3f]:
            raise ValueError(f"Unsupported dtype: {dtype}. NanoVDB only supports wp.float32 and wp.vec3f")

        if indexing not in ["ij", "xy"]:
            raise ValueError(f"Invalid indexing: {indexing}. Must be 'ij' or 'xy'")

        device = volume.device.alias if hasattr(volume, "device") else wp.get_device().alias

        # Get field model for NanoVDB
        field_model = fields.nanovdb.get_field_model(dtype, indexing)

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        handle.volume_id = volume.id
        handle.dims = wp.vec3i(*dims)
        handle.origin = wp.vec3i(*origin)

        return Field(handle=handle, field_model=field_model, data=volume, dtype=dtype, device=device)

    def get_interpolated_field_api(self, data_model: DataModel) -> type[InterpolatedFieldAPI]:
        """Generate an interpolated field API for this field.

        Args:
            data_model: Data model to generate API for

        Returns:
            InterpolatedFieldAPI class with get() method for field interpolation
        """
        return _create_interpolated_field_api(data_model, self.field_model)


class FieldCollection:
    """Collection of field pieces that can be treated as a single unified field.

    This class wraps a collection field model and provides methods to:
    - Create a collection from a list of Field instances
    - Access the underlying collection field handle and model
    - Generate interpolated field APIs for use with collection data models

    Example usage:
        ```python
        import warp as wp
        from dav.fields import array
        from dav.field import Field, FieldCollection

        # Create individual fields
        field1 = Field(handle1, model1, "cuda:0")
        field2 = Field(handle2, model2, "cuda:0")

        # Create collection
        collection = FieldCollection.from_fields([field1, field2])

        # Use with collection data model
        interpolated_api = collection.get_interpolated_field_api(collection_data_model)
        ```
    """

    handle: FieldHandle
    field_model: FieldModel  # This is the collection field model
    dtype: Any
    base_field_model: FieldModel  # The underlying field model for pieces
    device: Any

    def __init__(
        self, handle: FieldHandle, field_model: FieldModel, base_field_model: FieldModel, dtype: Any, device: Any
    ):
        """Initialize a FieldCollection.

        Args:
            handle: Collection field handle
            field_model: Collection field model
            base_field_model: Base field model for individual pieces
            dtype: Dtype of the field
            device: Device where fields reside
        """
        self.handle = handle
        self.field_model = field_model
        self.base_field_model = base_field_model
        self.dtype = dtype
        self.device = device

    @property
    def association(self) -> AssociationType:
        """Get the association type of the field collection."""
        return AssociationType(self.handle.association)

    @staticmethod
    def from_fields(fields: list[Field]) -> "FieldCollection":
        """Create a FieldCollection from a list of Field instances.

        Args:
            fields: List of Field instances (must all use the same field model and device)

        Returns:
            FieldCollection wrapping the provided fields

        Raises:
            ValueError: If fields list is empty, fields use different models, or different devices

        Example:
            >>> field1 = Field(handle1, model, "cuda:0")
            >>> field2 = Field(handle2, model, "cuda:0")
            >>> collection = FieldCollection.from_fields([field1, field2])
        """
        if not fields:
            raise ValueError("Cannot create FieldCollection from empty list of fields")

        # Get base field model and device from first field
        base_field_model = fields[0].field_model
        device = fields[0].device
        association = fields[0].association
        value_dtype = fields[0].dtype

        # Verify all fields use the same model and device
        for i, field in enumerate(fields[1:], 1):
            if field.field_model.FieldAPI is not base_field_model.FieldAPI:
                raise ValueError(
                    f"All fields must use the same field model. Field 0 and field {i} use different field APIs."
                )
            if field.field_model.FieldHandle is not base_field_model.FieldHandle:
                raise ValueError(
                    f"All fields must use the same field handle model. Field 0 and field {i} use different field handle models."
                )
            if field.device != device:
                raise ValueError(
                    f"All fields must be on the same device. Field 0 is on {device}, field {i} is on {field.device}"
                )
            if field.dtype != value_dtype:
                raise ValueError(
                    f"All fields must have the same dtype. Field 0 has dtype {value_dtype}, field {i} has dtype {field.dtype}"
                )
            if field.association != association:
                raise ValueError(
                    f"All fields must have the same association. "
                    f"Field 0 has association {association}, field {i} has association {field.association}"
                )

        # Import collection module
        from .fields import collection

        # Get collection field model
        collection_field_model = collection.get_field_model(base_field_model)

        # Create collection field handle
        piece_handles = [field.handle for field in fields]

        # Create warp array of piece handles
        pieces_array = wp.array(piece_handles, dtype=base_field_model.FieldHandle, device=device)

        # Create collection handle
        coll_handle = collection_field_model.FieldHandle()
        coll_handle.association = association.value
        coll_handle.pieces = pieces_array

        return FieldCollection(
            handle=coll_handle,
            field_model=collection_field_model,
            base_field_model=base_field_model,
            dtype=value_dtype,
            device=device,
        )

    def get_interpolated_field_api(self, data_model: DataModel) -> type[InterpolatedFieldAPI]:
        """Generate an interpolated field API for this collection.

        This method creates a specialized interpolated field API that works with
        collection data models. It handles mapping between global and piece-local
        indices automatically.

        Args:
            data_model: Collection data model to generate API for

        Returns:
            InterpolatedFieldAPI class with get() method for field interpolation

        Example:
            >>> from dav.data_models import collection
            >>> coll_data_model = collection.get_collection_data_model(base_data_model)
            >>> interp_api = field_collection.get_interpolated_field_api(coll_data_model)
        """
        return _create_interpolated_field_api(data_model, self.field_model)

    def get_data(self) -> Any:
        """Get the underlying data for this field collection.

        Field collections contain multiple field pieces, each with their own data.
        There is no single unified data array. To access individual piece data,
        iterate through the collection's pieces and call get_data() on each.

        Raises:
            NotImplementedError: Always, as collections don't have unified data

        Note:
            To get data from individual pieces in a collection, you need to reconstruct
            the original Field objects and call get_data() on them. This is not
            currently exposed through the FieldCollection API.
        """
        raise NotImplementedError(
            "FieldCollection does not have a unified get_data() method. "
            "Field collections contain multiple field pieces, each with separate data. "
            "Access individual field pieces to retrieve their data."
        )
