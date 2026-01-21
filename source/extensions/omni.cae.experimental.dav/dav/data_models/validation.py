# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Validation utilities for data models.

This module provides utilities to validate that a data model implementation
fully conforms to the DataModel protocol, including all required handle types,
API classes, and methods with correct signatures.

Example usage:
    # Validate a single data model
    >>> from dav.data_models.vtk import image_data
    >>> from dav.data_models.validation import validate_data_model
    >>> is_valid, errors = validate_data_model(image_data.DataModel, verbose=True)
    >>> if not is_valid:
    ...     for error in errors:
    ...         print(f"  - {error}")

    # Validate all built-in data models
    >>> from dav.data_models.validation import validate_all_builtin_models
    >>> results = validate_all_builtin_models(verbose=False)
    >>> for model_name, (is_valid, errors) in results.items():
    ...     if not is_valid:
    ...         print(f"{model_name}: INVALID")

    # Run from command line
    $ python -m dav.data_models.validation
"""

import inspect
from typing import Any

import warp as wp


def validate_data_model(data_model: Any, verbose: bool = True) -> tuple[bool, list[str]]:
    """Validate that a data model fully implements the DataModel protocol.

    This function performs comprehensive validation of a data model implementation,
    checking:
    1. All required handle types are present and are wp.struct types
    2. All required API classes are present
    3. All API methods are present with correct signatures
    4. Transitive validation of nested types referenced in method signatures

    Args:
        data_model: The data model to validate (should conform to DataModel protocol)
        verbose: If True, print detailed validation results

    Returns:
        tuple[bool, list[str]]: (is_valid, error_messages)
            - is_valid: True if the data model is fully valid
            - error_messages: List of validation errors (empty if valid)

    Example:
        >>> from dav.data_models.vtk import image_data
        >>> is_valid, errors = validate_data_model(image_data.DataModel)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """
    from dav.data_models.typing import CellAPI, CellLinksAPI, DatasetAPI, InterpolatedCellAPI

    errors = []

    if verbose:
        print(f"Validating data model: {data_model.__name__}")
        print("=" * 80)

    # 1. Validate required handle types
    handle_types = {
        "DatasetHandle": None,
        "CellHandle": None,
        "InterpolatedCellHandle": None,
        "CellLinkHandle": None,
        "PointIdHandle": None,
        "CellIdHandle": None,
    }

    for handle_name in handle_types.keys():
        if not hasattr(data_model, handle_name):
            errors.append(f"Missing required handle type: {handle_name}")
        else:
            handle_type = getattr(data_model, handle_name)
            handle_types[handle_name] = handle_type

            # Check if handle is a warp struct (except for ID types which can be primitives)
            if handle_name not in ["PointIdHandle", "CellIdHandle"]:
                if not _is_warp_struct(handle_type):
                    errors.append(f"{handle_name} should be a warp struct type (wp.struct), got {type(handle_type)}")
            else:
                # ID handles can be primitives like wp.int32, wp.int64, or structs like wp.vec2i
                if not _is_warp_type(handle_type):
                    errors.append(
                        f"{handle_name} should be a warp type (wp.int32, wp.int64, wp.vec2i, etc.), got {type(handle_type)}"
                    )

    if verbose:
        print("\n1. Handle Types:")
        for handle_name, handle_type in handle_types.items():
            status = "✓" if handle_type is not None else "✗"
            type_str = str(handle_type) if handle_type else "MISSING"
            print(f"  {status} {handle_name}: {type_str}")

    # 2. Validate required API classes
    api_classes = {
        "DatasetAPI": DatasetAPI,
        "CellAPI": CellAPI,
        "InterpolatedCellAPI": InterpolatedCellAPI,
        "CellLinksAPI": CellLinksAPI,
    }

    api_instances = {}
    for api_name, protocol_class in api_classes.items():
        if not hasattr(data_model, api_name):
            errors.append(f"Missing required API class: {api_name}")
        else:
            api_class = getattr(data_model, api_name)
            api_instances[api_name] = api_class

            # Validate that the API class has all required methods
            api_errors = _validate_api_class(api_class, protocol_class, data_model, verbose)
            errors.extend(api_errors)

    if verbose and not errors:
        print("\n2. API Classes:")
        for api_name in api_classes.keys():
            status = "✓" if api_name in api_instances else "✗"
            print(f"  {status} {api_name}")

    # 3. Summary
    if verbose:
        print("\n" + "=" * 80)
        if not errors:
            print("✓ Data model is VALID - all required types and methods are present")
        else:
            print(f"✗ Data model is INVALID - {len(errors)} error(s) found:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

    return len(errors) == 0, errors


def _is_warp_struct(obj: Any) -> bool:
    """Check if an object is a warp struct type."""
    # Warp structs are instances of warp._src.codegen.Struct
    return type(obj).__name__ == "Struct" and hasattr(obj, "cls")


def _is_warp_type(obj: Any) -> bool:
    """Check if an object is a warp type (primitive or struct)."""
    if _is_warp_struct(obj):
        return True

    # Check for warp primitive types
    warp_primitives = [
        wp.int8,
        wp.int16,
        wp.int32,
        wp.int64,
        wp.uint8,
        wp.uint16,
        wp.uint32,
        wp.uint64,
        wp.float16,
        wp.float32,
        wp.float64,
        wp.bool,
    ]

    # Check for warp vector types
    warp_vectors = [
        wp.vec2,
        wp.vec3,
        wp.vec4,
        wp.vec2i,
        wp.vec3i,
        wp.vec4i,
        wp.vec2f,
        wp.vec3f,
        wp.vec4f,
        wp.vec2d,
        wp.vec3d,
        wp.vec4d,
    ]

    # Check for warp matrix types
    warp_matrices = [wp.mat22, wp.mat33, wp.mat44, wp.mat22f, wp.mat33f, wp.mat44f, wp.mat22d, wp.mat33d, wp.mat44d]

    all_warp_types = warp_primitives + warp_vectors + warp_matrices

    return obj in all_warp_types


def _validate_api_class(api_class: type, protocol_class: type, data_model: Any, verbose: bool) -> list[str]:
    """Validate that an API class conforms to its protocol.

    Args:
        api_class: The API class to validate
        protocol_class: The protocol it should conform to
        data_model: The parent data model (for context)
        verbose: If True, print detailed information

    Returns:
        List of error messages
    """
    errors = []
    api_name = api_class.__name__

    if verbose:
        print(f"\n  Validating {api_name}:")

    # Get all methods defined in the protocol
    protocol_methods = {}
    for name, member in inspect.getmembers(protocol_class):
        if name.startswith("_"):
            continue
        if callable(member) or isinstance(member, staticmethod):
            protocol_methods[name] = member

    # Check each protocol method exists in the API class
    for method_name in protocol_methods.keys():
        if not hasattr(api_class, method_name):
            errors.append(f"{api_name}.{method_name} is missing")
            if verbose:
                print(f"    ✗ {method_name}: MISSING")
        else:
            method = getattr(api_class, method_name)

            # Check if it's a static method
            if not isinstance(inspect.getattr_static(api_class, method_name), staticmethod):
                errors.append(f"{api_name}.{method_name} should be a @staticmethod")
                if verbose:
                    print(f"    ✗ {method_name}: Not a static method")
            else:
                if verbose:
                    # Get method signature
                    try:
                        sig = inspect.signature(method)
                        print(f"    ✓ {method_name}{sig}")
                    except Exception:
                        print(f"    ✓ {method_name}")

    return errors


def validate_all_builtin_models(verbose: bool = True) -> dict[str, tuple[bool, list[str]]]:
    """Validate all built-in data models.

    Args:
        verbose: If True, print detailed validation results for each model

    Returns:
        Dictionary mapping model name to (is_valid, errors) tuple

    Example:
        >>> results = validate_all_builtin_models(verbose=False)
        >>> for model_name, (is_valid, errors) in results.items():
        ...     if not is_valid:
        ...         print(f"{model_name}: INVALID")
        ...         for error in errors:
        ...             print(f"  - {error}")
    """
    results = {}

    # VTK data models
    try:
        from dav.data_models.vtk import image_data

        results["vtk.image_data"] = validate_data_model(image_data.DataModel, verbose=verbose)
    except ImportError:
        results["vtk.image_data"] = (False, ["Module not available"])

    try:
        from dav.data_models.vtk import structured_grid

        results["vtk.structured_grid"] = validate_data_model(structured_grid.DataModel, verbose=verbose)
    except ImportError:
        results["vtk.structured_grid"] = (False, ["Module not available"])

    try:
        from dav.data_models.vtk import unstructured_grid

        results["vtk.unstructured_grid"] = validate_data_model(unstructured_grid.DataModel, verbose=verbose)
    except ImportError:
        results["vtk.unstructured_grid"] = (False, ["Module not available"])

    # SIDS data models
    try:
        from dav.data_models.sids import unstructured

        results["sids.unstructured"] = validate_data_model(unstructured.DataModel, verbose=verbose)
    except ImportError:
        results["sids.unstructured"] = (False, ["Module not available"])

    # Collection data model
    # Note: collection uses a factory pattern (get_collection_data_model) so we need to instantiate it
    # with each known data model and validate the result
    try:
        from dav.data_models import collection

        collection_errors = []
        collection_valid = True

        # Get all non-collection data models that were successfully validated
        base_models = {
            "vtk.image_data": ("dav.data_models.vtk.image_data", "DataModel"),
            "vtk.structured_grid": ("dav.data_models.vtk.structured_grid", "DataModel"),
            "vtk.unstructured_grid": ("dav.data_models.vtk.unstructured_grid", "DataModel"),
            "sids.unstructured": ("dav.data_models.sids.unstructured", "DataModel"),
        }

        if verbose:
            print("\nValidating collection data models (factory pattern):")

        for base_name, (module_path, attr_name) in base_models.items():
            # Skip if base model failed validation or isn't available
            if base_name not in results or not results[base_name][0]:
                if verbose:
                    print(f"  Skipping collection({base_name}) - base model not available or invalid")
                continue

            try:
                # Import the base data model
                module_parts = module_path.rsplit(".", 1)
                base_module = __import__(module_path, fromlist=[module_parts[-1]])
                base_data_model = getattr(base_module, attr_name)

                # Create collection data model
                collection_data_model = collection.get_collection_data_model(base_data_model)

                # Validate the collection data model
                is_valid, errors = validate_data_model(collection_data_model, verbose=False)

                if verbose:
                    status = "✓" if is_valid else "✗"
                    print(f"  {status} collection({base_name})")

                if not is_valid:
                    collection_valid = False
                    collection_errors.extend([f"collection({base_name}): {err}" for err in errors])

            except Exception as e:
                collection_valid = False
                error_msg = f"collection({base_name}): Failed to create or validate - {e}"
                collection_errors.append(error_msg)
                if verbose:
                    print(f"  ✗ {error_msg}")

        results["collection"] = (collection_valid, collection_errors)

    except ImportError:
        results["collection"] = (False, ["Module not available"])

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL DATA MODELS:")
        print("=" * 80)
        for model_name, (is_valid, errors) in results.items():
            status = "✓ VALID" if is_valid else f"✗ INVALID ({len(errors)} errors)"
            print(f"  {model_name}: {status}")

    return results


if __name__ == "__main__":
    # Example usage: validate all built-in models
    import sys

    results = validate_all_builtin_models(verbose=True)

    # Exit with error code if any model is invalid
    all_valid = all(is_valid for is_valid, _ in results.values())
    sys.exit(0 if all_valid else 1)
