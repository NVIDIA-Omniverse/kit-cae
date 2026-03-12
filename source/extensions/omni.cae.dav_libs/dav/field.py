# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = ["Field"]

from typing import Any

import warp as wp

from . import core
from .data_models import DataModel
from .fields import AssociationType, FieldHandle, FieldModel, InterpolatedFieldAPI


class Field:
    handle: FieldHandle
    field_model: FieldModel
    data: Any
    device: Any
    dtype: Any
    size: int

    def __init__(self, handle: FieldHandle, field_model: FieldModel, data: Any, dtype: Any, device: Any, size: int):
        self.handle = handle
        self.field_model = field_model
        self.data = data
        self.dtype = dtype
        self.device = device
        self.size = size
        self._range_cache = None  # Cache for all range computations

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
        from .fields import array as field_array

        # Get device from array
        device = data.device.alias

        # Get field model (storage dtype same as value dtype for single array)
        field_model = field_array.get_field_model_AoS(core.utils.get_scalar_dtype(data.dtype), length=core.utils.get_vector_length(data.dtype))

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        handle.data = data

        return Field(handle=handle, field_model=field_model, data=data, dtype=data.dtype, device=device, size=data.shape[0])

    @staticmethod
    def from_arrays(data: list[wp.array], association: AssociationType) -> "Field":
        """Create a Field from multiple warp arrays (SoA storage).

        This method creates vector fields using Structure-of-Arrays (SoA) storage
        where each component is stored in a separate array. All arrays must have
        the same dtype (the scalar component type), length, and device.

        Args:
            data: List of warp arrays
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
        from .fields import array as field_array

        if not data:
            raise ValueError("data list cannot be empty")

        if len(data) == 1:
            return Field.from_array(data[0], association)

        # Get device from first array
        device = data[0].device.alias

        # Verify all arrays have same dtype
        scalar_dtype = data[0].dtype
        if core.utils.is_vector_dtype(scalar_dtype):
            raise ValueError(f"Unsupported dtype: {scalar_dtype}. Vector dtypes are not supported for SoA fields")

        for i, arr in enumerate(data[1:], 1):
            if arr.dtype != scalar_dtype:
                raise ValueError(f"All arrays must have the same dtype. Array 0 has dtype {scalar_dtype}, array {i} has dtype {arr.dtype}")

        # Verify all arrays have same length
        length = data[0].shape[0]
        for i, arr in enumerate(data[1:], 1):
            if arr.shape[0] != length:
                raise ValueError(f"All arrays must have the same length. Array 0 has length {length}, array {i} has length {arr.shape[0]}")

        # Verify all arrays are on the same device
        for i, arr in enumerate(data[1:], 1):
            if arr.device.alias != device:
                raise ValueError(f"All arrays must be on the same device. Array 0 is on {device}, array {i} is on {arr.device.alias}")

        # Determine value dtype
        value_dtype = core.utils.get_vector_dtype(scalar_dtype, length=len(data))

        # Get field model (SoA: storage dtype is scalar, value dtype is vector)
        field_model = field_array.get_field_model_SoA(scalar_dtype, length=len(data))

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        field_model.FieldHandle._set_data(handle, data, device)

        return Field(handle=handle, field_model=field_model, data=data, dtype=value_dtype, device=device, size=length)

    @staticmethod
    def from_volume(volume: wp.Volume, dims: wp.vec3i, association: AssociationType, origin: wp.vec3i = wp.vec3i(0, 0, 0)) -> "Field":
        """Create a Field from a warp Volume (NanoVDB).

        Args:
            volume: Warp Volume containing NanoVDB data
            dims: Volume dimensions (Ni, Nj, Nk)
            association: Field association (VERTEX or CELL)
            origin: Volume origin (Ni, Nj, Nk). Default: (0, 0, 0)

        Returns:
            Field: A new Field instance backed by NanoVDB

        Raises:
            ValueError: If volume dtype is not supported

        Note:
            NanoVDB fields always use 'ij' (Fortran/column-major) indexing where the first dimension varies fastest.

        Example:
            >>> import warp as wp
            >>> from dav import Field, AssociationType
            >>>
            >>> # Scalar volume field
            >>> volume = wp.Volume.allocate(min=[0,0,0], max=[32,32,32], voxel_size=1.0)
            >>> field = Field.from_volume(volume, dims=(32, 32, 32), association=AssociationType.VERTEX, origin=(0, 0, 0))
        """
        from .fields import nanovdb as field_nanovdb

        # Get volume dtype
        if not hasattr(volume, "dtype"):
            raise ValueError("Volume must have a dtype attribute")

        dtype = volume.dtype
        if dtype not in [wp.float32, wp.vec3f]:
            raise ValueError(f"Unsupported dtype: {dtype}. NanoVDB only supports wp.float32 and wp.vec3f")

        device = volume.device.alias if hasattr(volume, "device") else wp.get_device().alias

        # Get field model for NanoVDB
        field_model = field_nanovdb.get_field_model(dtype)

        # Create field handle
        handle = field_model.FieldHandle()
        handle.association = association.value
        handle.volume_id = volume.id
        handle.dims = wp.vec3i(*dims)
        handle.origin = wp.vec3i(*origin)

        # Compute size from dims
        size = dims[0] * dims[1] * dims[2]

        return Field(handle=handle, field_model=field_model, data=volume, dtype=dtype, device=device, size=size)

    @staticmethod
    def from_field(field: "Field", *, component: int | None = None, magnitude: bool = False) -> "Field":
        """Create a read-only scalar Field by reducing a vector Field.

        Either extracts a single component or computes the vector magnitude.
        Exactly one of ``component`` or ``magnitude`` must be specified.

        Args:
            field: A vector Field to reduce.
            component: 0-based index of the component to extract, or ``None``.
            magnitude: If ``True``, compute the L2 norm (magnitude) of each vector value.
                Only supported for floating-point vector fields.

        Returns:
            Field: A new read-only scalar Field derived from *field*.

        Example:
            >>> velocity = Field.from_array(arr, AssociationType.VERTEX)
            >>> vx    = Field.from_field(velocity, component=0)
            >>> speed = Field.from_field(velocity, magnitude=True)
        """
        from .fields.vector_reduced import get_field_model_vector_reduced

        reduced_model = get_field_model_vector_reduced(field.field_model, component=component, magnitude=magnitude)

        handle = reduced_model.FieldHandle()
        handle.association = field.handle.association
        handle.inner = field.handle

        scalar_dtype = core.utils.get_scalar_dtype(field.dtype)
        return Field(handle=handle, field_model=reduced_model, data=field, dtype=scalar_dtype, device=field.device, size=field.size)

    @staticmethod
    def from_fields(fields: list["Field"]) -> "Field":
        """Create a Field from a list of Field instances (collection).

        This method creates a collection field that wraps multiple field pieces,
        treating them as a single unified field. All fields must use the same
        field model, device, dtype, and association.

        Args:
            fields: List of Field instances (must all use the same field model and device)

        Returns:
            Field: A new Field instance wrapping the provided fields as a collection

        Raises:
            ValueError: If fields list is empty, fields use different models, or different devices

        Example:
            >>> import warp as wp
            >>> from dav import Field, AssociationType
            >>>
            >>> # Create individual fields
            >>> data1 = wp.array([1.0, 2.0], dtype=wp.float32)
            >>> data2 = wp.array([3.0, 4.0, 5.0], dtype=wp.float32)
            >>> field1 = Field.from_array(data1, AssociationType.VERTEX)
            >>> field2 = Field.from_array(data2, AssociationType.VERTEX)
            >>>
            >>> # Create collection field
            >>> collection_field = Field.from_fields([field1, field2])
        """
        if not fields:
            raise ValueError("Cannot create Field collection from empty list of fields")

        # Get base field model and device from first field
        base_field_model = fields[0].field_model
        device = fields[0].device
        association = fields[0].association
        value_dtype = fields[0].dtype

        # Verify all fields use the same model and device
        for i, field in enumerate(fields[1:], 1):
            if field.field_model.FieldAPI is not base_field_model.FieldAPI:
                raise ValueError(f"All fields must use the same field model. Field 0 and field {i} use different field APIs.")
            if field.field_model.FieldHandle is not base_field_model.FieldHandle:
                raise ValueError(f"All fields must use the same field handle model. Field 0 and field {i} use different field handle models.")
            if field.device != device:
                raise ValueError(f"All fields must be on the same device. Field 0 is on {device}, field {i} is on {field.device}")
            if field.dtype != value_dtype:
                raise ValueError(f"All fields must have the same dtype. Field 0 has dtype {value_dtype}, field {i} has dtype {field.dtype}")
            if field.association != association:
                raise ValueError(f"All fields must have the same association. Field 0 has association {association}, field {i} has association {field.association}")

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

        size = sum(field.size for field in fields)
        return Field(handle=coll_handle, field_model=collection_field_model, data=fields, dtype=value_dtype, device=device, size=size)

    def get_range(self, component: int | None = None) -> tuple[float, float]:
        """Get the range (min, max) of field values.

        This method computes and caches the range of field values. For efficiency,
        all ranges (all components and magnitude) are computed in a single pass
        on the first call, then cached for subsequent calls.

        Args:
            component: For vector fields, specify component index (0, 1, 2, ...) to get
                      the range of that component. If None, returns magnitude range.

                      For scalar fields, must be None or 0 and in both cases returns
                      the range of the scalar field.

                      Default: None

        Returns:
            tuple[float, float]: (min_value, max_value)

        Raises:
            ValueError: If component is invalid for the field type
            NotImplementedError: If field type doesn't support range computation

        Example:
            >>> import warp as wp
            >>> from dav import Field, AssociationType
            >>>
            >>> # Scalar field
            >>> data = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
            >>> field = Field.from_array(data, AssociationType.VERTEX)
            >>> min_val, max_val = field.get_range()  # Returns (1.0, 3.0)
            >>>
            >>> # Vector field - get X component range
            >>> vec_data = wp.array([wp.vec3f(1,2,3), wp.vec3f(4,5,6)], dtype=wp.vec3f)
            >>> vec_field = Field.from_array(vec_data, AssociationType.VERTEX)
            >>> min_x, max_x = vec_field.get_range(component=0)  # Returns (1.0, 4.0)
            >>>
            >>> # Vector field - get magnitude range
            >>> min_mag, max_mag = vec_field.get_range()  # Returns magnitude range
        """
        # Check if we need to compute ranges
        if self._range_cache is None:
            self._range_cache = self._compute_all_ranges()

        length = core.utils.get_vector_length(self.dtype)

        # Validate component parameter
        if component is not None and (component < 0 or component >= length):
            raise ValueError(f"Invalid component {component}. Valid components are 0 to {length - 1}, or None for magnitude.")

        # Return the appropriate range from cache
        if length == 1:
            return self._range_cache["components"][0][0], self._range_cache["components"][0][1]
        elif component is None:
            # For both scalar and vector fields, return magnitude range
            return self._range_cache["magnitude"][0], self._range_cache["magnitude"][1]
        else:
            # Return specific component range
            return self._range_cache["components"][component][0], self._range_cache["components"][component][1]

    def _compute_all_ranges(self) -> dict:
        """Compute all ranges (components and magnitude) in a single pass.

        This method should compute ranges efficiently in a single kernel launch.
        Use warp kernels with atomic operations to compute min/max values.

        Returns:
            dict: Dictionary with range tuples for all components and/or magnitude.
                  For scalar fields: {"components": [(min, max)], "magnitude": (min, max)}
                  For vector fields: {"components": [(min, max), (min, max), (min, max)], "magnitude": (min, max)}

                  All ranges are returned as float64 values.
        """
        from .fields import utils as field_utils

        nb_elements = self.size
        min_value, max_value = core.utils.get_limits(self.dtype)
        length = core.utils.get_vector_length(self.dtype)

        # Initialize magnitude range: [max, min] so atomic_min/max work correctly
        # Use float64 for magnitude range
        wp_mag = wp.array([float(max_value), float(min_value)], dtype=wp.float64, device=self.device)

        # Initialize component ranges as 2D array: [length, 2] with [max, min] for each component
        # Use float64 for component ranges
        wp_component_ranges = wp.array([[float(max_value), float(min_value)]] * length, dtype=wp.float64, device=self.device)

        kernel = field_utils.get_compute_field_range_kernel(self.field_model)
        wp.launch(kernel, dim=nb_elements, inputs=(self.handle, wp_component_ranges, wp_mag), device=self.device)
        return {"components": wp_component_ranges.numpy().tolist(), "magnitude": wp_mag.numpy().tolist()}

    def get_interpolated_field_api(self, data_model: DataModel) -> type[InterpolatedFieldAPI]:
        """Generate an interpolated field API for this field.

        Args:
            data_model: Data model to generate API for

        Returns:
            InterpolatedFieldAPI class with get() method for field interpolation
        """
        from .fields import utils as field_utils

        return field_utils.create_interpolated_field_api(data_model, self.field_model)

    def to_array(self) -> wp.array:
        """Convert the field to a warp array in AoS (Array-of-Structures) format.

        For fields already stored as warp arrays (created via `from_array`), this returns
        the underlying array directly without copying. For fields with other storage layouts
        (SoA, NanoVDB volumes, or collections), this creates a new array and copies the data
        using the FieldAPI.

        Returns:
            wp.array: A warp array containing the field data in AoS format.
                     For scalar fields, returns a 1D array of scalar values.
                     For vector fields, returns a 1D array of vector values (vec3f, vec3d, vec3i).

        Example:
            >>> import warp as wp
            >>> from dav import Field, AssociationType
            >>>
            >>> # AoS field - returns the same array (no copy)
            >>> data = wp.array([1.0, 2.0, 3.0], dtype=wp.float32)
            >>> field = Field.from_array(data, AssociationType.VERTEX)
            >>> result = field.to_array()
            >>> assert result is data  # Same object
            >>>
            >>> # SoA field - creates a new array (copy needed)
            >>> x = wp.array([1.0, 2.0], dtype=wp.float32)
            >>> y = wp.array([3.0, 4.0], dtype=wp.float32)
            >>> z = wp.array([5.0, 6.0], dtype=wp.float32)
            >>> field = Field.from_arrays([x, y, z], AssociationType.VERTEX)
            >>> result = field.to_array()
            >>> assert result.dtype == wp.vec3f  # Converted to AoS format
        """
        from .fields import utils as field_utils

        if isinstance(self.data, wp.array):
            return self.data

        result = wp.zeros(self.size, dtype=self.dtype, device=self.device)
        wp.launch(field_utils.get_copy_field_kernel(self.field_model, self.dtype), dim=self.size, inputs=(self.handle, result), device=self.device)
        return result
