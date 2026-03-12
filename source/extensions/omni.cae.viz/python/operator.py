# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Operator registration system for CAE visualization operators.

This module provides a decorator for marking and validating operator classes,
along with functions to register/unregister operators from modules. This allows
for proper lifecycle management in extensions.

The operator system works in two phases:
1. **Marking**: The @operator decorator validates and marks classes as operators
2. **Registration**: The register_module_operators() function collects marked
   operators from a module and registers them with the system

This separation allows extensions to register operators on startup and cleanly
unregister them on shutdown.
"""

import bisect
import inspect
import weakref
from logging import getLogger
from types import ModuleType
from typing import Any, Callable, List, Type, TypeVar

__all__ = ["operator", "register_module_operators", "unregister_module_operators", "get_operators"]

logger = getLogger(__name__)

# Type variable for the operator class
T = TypeVar("T")

# Global registry of operator class types with metadata
# Each entry is a tuple: (weakref_to_class_type, priority, decoration_order)
_registered_operators: List[tuple[weakref.ref, int, int]] = []

# Counter for tracking decoration order (incremented when @operator is applied)
_decoration_counter = 0

# Marker attribute name used to identify operator classes
_OPERATOR_MARKER = "__is_cae_operator__"


def operator(
    priority: int = 0, supports_temporal: bool = False, tick_on_time_change: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to validate and mark a class as an operator with a priority.

    This decorator validates that the class has the required attributes and methods,
    then marks it as an operator class with the specified priority. The class is NOT
    registered at this point - registration happens later via register_module_operators().

    Operators with higher priority values are returned first by get_operators().
    For operators with the same priority, operators decorated later are returned first
    (reverse decoration order).

    Parameters
    ----------
    priority : int, optional
        The priority of this operator. Higher values are returned first by
        get_operators(). Default is 0.
    supports_temporal : bool, optional
        Whether this operator supports temporal field interpolation via
        CaeVizOperatorTemporalAPI. When True, the operator can receive
        ExecutionContext to optimize temporal-only updates. Default is False.
    tick_on_time_change : bool, optional
        Whether to call on_time_changed() hook on every time change, even for
        already-executed timecodes. Only applies when supports_temporal=True.
        Default is False. When True, the operator must implement the
        on_time_changed(prim, timeCode, device) method.

    Returns
    -------
    Callable[[Type[T]], Type[T]]
        A decorator function that validates and marks the operator class

    Raises
    ------
    TypeError
        If the class is missing required attributes or they are of incorrect type
    AttributeError
        If the class is missing required attributes

    Examples
    --------
    >>> @operator(priority=10)
    ... class HighPriorityOperator:
    ...     prim_type = "CaeDataSet"
    ...     api_schemas = {"CaeVizStreamlinesAPI"}
    ...     optional_api_schemas = set()
    ...
    ...     def exec(self, prim, **kwargs):
    ...         pass
    >>>
    >>> @operator()  # Default priority = 0
    ... class NormalPriorityOperator:
    ...     prim_type = "CaeDataSet"
    ...     api_schemas = {"CaeDataAPI"}
    ...     optional_api_schemas = set()
    ...
    ...     def exec(self, prim, **kwargs):
    ...         pass
    >>>
    >>> @operator(supports_temporal=True, tick_on_time_change=True)
    ... class TemporalOperator:
    ...     prim_type = "Volume"
    ...     api_schemas = {"CaeVizIndeXVolumeAPI"}
    ...     optional_api_schemas = set()
    ...
    ...     async def exec(self, prim, device, context):
    ...         if context.is_full_rebuild_needed():
    ...             # Full rebuild (structural change or initial)
    ...             pass
    ...         elif context.is_temporal_update():
    ...             # Fast path for temporal updates
    ...             pass
    ...
    ...     async def on_time_changed(self, prim, device, context):
    ...         # Called on EVERY time change when tick_on_time_change=True
    ...         # Use context.timecode, context.raw_timecode, context.next_time_code
    ...         pass

    Notes
    -----
    The decorated class is only marked, not registered. To register operators,
    call register_module_operators() with the module containing the operator classes.

    Required Class Attributes
    -------------------------
    - exec : callable
        Method that executes the operator's visualization logic.
        Signature: exec(self, prim: Usd.Prim, timeCode: Usd.TimeCode, device: str, context: ExecutionContext = None) -> Any

    - prim_type : str
        The USD prim type name that this operator handles
        (e.g., "CaeDataSet", "Mesh")

    - api_schemas : set[str]
        Set of required API schema names that must be applied to the prim.
        All schemas in this set must be present for the operator to match.

    - optional_api_schemas : set[str]
        Set of optional API schema names. These schemas are not required
        but may influence the operator's behavior if present.

    See Also
    --------
    register_module_operators : Register all marked operators from a module
    unregister_module_operators : Unregister operators from a module
    get_operators : Get all registered operators sorted by priority
    """

    def decorator(cls: Type[T]) -> Type[T]:
        global _decoration_counter

        # Validate exec method
        if not hasattr(cls, "exec"):
            raise AttributeError(f"Operator class '{cls.__name__}' must have an 'exec' method")

        if not callable(cls.exec):
            raise TypeError(f"Operator class '{cls.__name__}': 'exec' attribute must be callable")

        # Validate prim_type attribute
        if not hasattr(cls, "prim_type"):
            raise AttributeError(f"Operator class '{cls.__name__}' must have a 'prim_type' attribute")

        if not isinstance(cls.prim_type, str):
            raise TypeError(
                f"Operator class '{cls.__name__}': 'prim_type' must be a string, " f"got {type(cls.prim_type).__name__}"
            )

        # Validate api_schemas attribute
        if not hasattr(cls, "api_schemas"):
            raise AttributeError(f"Operator class '{cls.__name__}' must have an 'api_schemas' attribute")

        if not isinstance(cls.api_schemas, set):
            raise TypeError(
                f"Operator class '{cls.__name__}': 'api_schemas' must be a set, "
                f"got {type(cls.api_schemas).__name__}"
            )

        if not all(isinstance(schema, str) for schema in cls.api_schemas):
            raise TypeError(f"Operator class '{cls.__name__}': 'api_schemas' must be a set of strings")

        # Validate optional_api_schemas attribute
        if not hasattr(cls, "optional_api_schemas"):
            raise AttributeError(f"Operator class '{cls.__name__}' must have an 'optional_api_schemas' attribute")

        if not isinstance(cls.optional_api_schemas, set):
            raise TypeError(
                f"Operator class '{cls.__name__}': 'optional_api_schemas' must be a set, "
                f"got {type(cls.optional_api_schemas).__name__}"
            )

        if not all(isinstance(schema, str) for schema in cls.optional_api_schemas):
            raise TypeError(f"Operator class '{cls.__name__}': 'optional_api_schemas' must be a set of strings")

        # Mark the class as an operator and store its priority, decoration order, and temporal support
        setattr(cls, _OPERATOR_MARKER, True)
        setattr(cls, "__operator_priority__", priority)
        setattr(cls, "__operator_decoration_order__", _decoration_counter)
        setattr(cls, "__supports_temporal__", supports_temporal)
        setattr(cls, "__tick_on_time_change__", tick_on_time_change)
        _decoration_counter += 1

        return cls

    return decorator


def _prune_dead_references():
    """Remove entries with dead weak references from the registry."""
    global _registered_operators
    _registered_operators = [op_data for op_data in _registered_operators if op_data[0]() is not None]


def register_module_operators(module: ModuleType) -> int:
    """
    Register all operator classes from a module.

    This function scans the given module for classes marked with the @operator
    decorator and registers them (as class types, not instances) in the global
    operator registry in priority order (maintained during insertion).

    This should typically be called in Extension.on_startup().

    Parameters
    ----------
    module : Module
        The module to scan for operator classes. All classes marked with @operator
        will be registered.

    Returns
    -------
    int
        The number of operators registered from this module

    Examples
    --------
    >>> from omni.cae.viz.impl import streamlines
    >>>
    >>> # In Extension.on_startup():
    >>> count = register_module_operators(streamlines)
    >>> print(f"Registered {count} operators")

    >>> # In Extension.on_shutdown():
    >>> unregister_module_operators(streamlines)

    Notes
    -----
    - Each operator class is registered as a class type (not instantiated)
    - Weak references are used to avoid preventing garbage collection
    - Operators are inserted in priority order (maintained for O(1) retrieval)
    - Already registered operator classes from this module won't be duplicated
    - Use get_operators() to retrieve operator class types in priority order

    See Also
    --------
    unregister_module_operators : Unregister operators from a module
    operator : Mark a class as an operator with a priority
    get_operators : Get operator class types in priority order (no sorting needed)
    """
    # Prune dead references before registering new ones
    _prune_dead_references()

    registered_count = 0

    # Get all classes from the module
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Check if this is a marked operator class
        if hasattr(obj, _OPERATOR_MARKER) and getattr(obj, _OPERATOR_MARKER):
            # Check if already registered (avoid duplicates)
            # Dereference weak refs to check
            if not any(op_data[0]() is obj for op_data in _registered_operators if op_data[0]() is not None):
                # Get the priority and decoration order from the class
                priority = getattr(obj, "__operator_priority__", 0)
                decoration_order = getattr(obj, "__operator_decoration_order__", 0)

                logger.debug(
                    f"Registering operator {obj.__name__} from module {module.__name__} "
                    f"with priority {priority}, decoration_order {decoration_order}"
                )

                # Create weak reference to the class type
                class_weakref = weakref.ref(obj)

                # Create the operator data tuple using decoration_order
                operator_data = (class_weakref, priority, decoration_order)

                # Insert in sorted position using bisect
                # Sort key: (-priority, -decoration_order) for descending priority
                # and reverse decoration order (later decorated operators come first)
                # We use bisect_right to maintain stable ordering
                insertion_key = (-priority, -decoration_order)

                # Find insertion point using binary search
                insert_pos = bisect.bisect_right(_registered_operators, insertion_key, key=lambda x: (-x[1], -x[2]))

                _registered_operators.insert(insert_pos, operator_data)
                registered_count += 1

    return registered_count


def unregister_module_operators(module: ModuleType) -> int:
    """
    Unregister all operator class types that came from a module.

    This function removes all registered operator class types whose classes are
    defined in the given module. This should typically be called in
    Extension.on_shutdown() to cleanly unregister operators.

    Parameters
    ----------
    module : Module
        The module whose operators should be unregistered

    Returns
    -------
    int
        The number of operators unregistered

    Examples
    --------
    >>> from omni.cae.viz.impl import streamlines
    >>>
    >>> # In Extension.on_startup():
    >>> register_module_operators(streamlines)
    >>>
    >>> # In Extension.on_shutdown():
    >>> count = unregister_module_operators(streamlines)
    >>> print(f"Unregistered {count} operators")

    See Also
    --------
    register_module_operators : Register operators from a module
    """
    global _registered_operators

    # Prune dead references first
    _prune_dead_references()

    # Get all operator classes from the module
    module_operator_classes = set()
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if hasattr(obj, _OPERATOR_MARKER) and getattr(obj, _OPERATOR_MARKER):
            module_operator_classes.add(obj)

    # Remove class types that match
    original_count = len(_registered_operators)
    _registered_operators = [
        op_data for op_data in _registered_operators if op_data[0]() not in module_operator_classes
    ]

    return original_count - len(_registered_operators)


def get_operators() -> List[Type[Any]]:
    """
    Get all registered operator class types in priority order.

    Operators are returned in descending priority order (higher priority first).
    For operators with the same priority, operators decorated later are returned first
    (reverse decoration order).

    The list is maintained in sorted order during insertion, so this is an O(1)
    operation (no sorting needed). Dead weak references are automatically pruned.

    Returns
    -------
    List[Type[Any]]
        List of all operator class types that have been registered via
        register_module_operators(), already sorted by priority (highest first).
        Weak references are dereferenced, so callers don't need to worry about that.

    Examples
    --------
    >>> operator_classes = get_operators()
    >>> for op_class in operator_classes:
    ...     print(f"Operator: {op_class.__name__}, Type: {op_class.prim_type}")

    Notes
    -----
    The returned list is a copy, so modifying it won't affect the
    internal registry. Dead weak references are automatically filtered out.

    See Also
    --------
    register_module_operators : Register operators from a module (maintains sort order)
    operator : Mark a class as an operator with a priority
    """
    # Prune dead references and dereference weak refs
    _prune_dead_references()

    # Return just the operator class types (dereference weak refs)
    # The list is already sorted during insertion
    return [op_data[0]() for op_data in _registered_operators]
