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
Change tracking system for monitoring USD prim schema property changes.

Overview
--------
The ChangeTracker monitors changes to USD prims that have specific schemas applied
to them. It uses USD's Tf.Notice system to efficiently track property changes and
maintains a record of which properties have changed for each schema on each prim.

Optionally, it can also track time-based changes by monitoring the timeline and
detecting when time-sampled attribute values would yield different results.

Key Features
------------
- **Schema-based filtering**: Only tracks changes for prims with schemas matching
  specified regex patterns (e.g., "^Cae" to track all CAE-related schemas)
- **Multi-apply schema support**: Handles both single-apply and multi-apply API schemas,
  including instance-specific tracking
- **Efficient**: Uses two-pass processing to minimize overhead and only tracks
  properties that belong to matching schemas
- **Property caching**: Caches schema property definitions to avoid repeated lookups
- **Time-based tracking (optional)**: Efficiently detects when animated attributes
  would yield different values due to timeline changes

Usage
-----
Basic usage::

    from pxr import Usd
    from omni.cae.viz.impl.change_tracker import ChangeTracker

    # Create or get a USD stage
    stage = Usd.Stage.Open("path/to/stage.usd")

    # Create a tracker for CAE schemas
    tracker = ChangeTracker(stage, schema_regexs=[r"^Cae"])

    # Make some changes to the stage
    prim = stage.GetPrimAtPath("/World/MyDataset")
    prim.GetAttribute("someAttr").Set("new_value")

    # Check if a prim's schema properties changed
    if tracker.prim_changed(prim, ["CaeDataSet"]):
        print("CaeDataSet properties changed!")

    # Check if a specific attribute changed
    if tracker.attr_changed(prim, "someAttr"):
        print("someAttr changed!")

    # Clear changes after processing
    tracker.clear_changes(prim, ["CaeDataSet"])

    # Clean up when done
    tracker.disable()

Multi-apply Schema Example::

    # Track changes to specific instances of a multi-apply schema
    prim = stage.GetPrimAtPath("/World/MyPrim")

    # Check for changes to a specific instance
    if tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:velocity"]):
        print("Velocity field selection changed!")

    # Check for changes to any instance of the schema
    if tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI"]):
        print("Some field selection changed!")

Pattern Matching Example::

    # Track multiple schema families
    tracker = ChangeTracker(stage, schema_regexs=[r"^Cae", r"^UsdGeom"])

    # Now tracks both CAE schemas and UsdGeom schemas
    if tracker.prim_changed(prim, ["CaeDataSet", "UsdGeomMesh"]):
        print("Either CaeDataSet or UsdGeomMesh properties changed!")

Implementation Notes
--------------------
- The tracker uses Tf.Notice.Register to listen for Usd.Notice.ObjectsChanged events
- Only property paths (not prim paths or metadata) trigger tracking
- Property changes are matched against schema definitions obtained from Usd.SchemaRegistry
- For multi-apply schemas, the tracker handles __INSTANCE_NAME__ substitution automatically
- Changes are stored per-prim, per-schema, per-property for fine-grained control

Performance Considerations
--------------------------
- Schema property definitions are cached after first lookup
- Only prims with matching schemas are processed (early filtering)
- Two-pass algorithm: first collect changes, then process by prim
- Minimal memory footprint: only changed properties are tracked
"""

__all__ = ["ChangeTracker"]

import logging
import re
from typing import Dict, Optional, Set, Tuple

import carb
import omni.timeline
from carb.eventdispatcher import get_eventdispatcher
from pxr import Sdf, Tf, Usd

logger = logging.getLogger(__name__)


class ChangeTracker:
    """
    Tracks changes to USD prim properties for specified schema patterns.

    This class monitors USD stage changes using Tf.Notice and maintains a record of
    which properties have changed for prims that have schemas matching the configured
    regex patterns. It's designed to be efficient and only track relevant changes.

    The tracker automatically registers for USD change notifications when created and
    can be enabled/disabled as needed. Changes are tracked per-prim, per-schema, and
    can be queried and cleared individually or in bulk.

    **Important:** When the tracker is created, all prims are treated as initially dirty
    since their state is unknown at creation time. Prims remain dirty until explicitly
    cleared via clear_changes() or clear_all_changes(). This ensures that existing prims
    on the stage are processed at least once.

    Parameters
    ----------
    stage : Usd.Stage
        The USD stage to monitor for changes
    schema_regexs : list[str], optional
        List of regex patterns to match schema names to track (default: [r"^Cae"])
        Examples: [r"^Cae"] tracks all CAE schemas, [r"^CaeVtk"] tracks only VTK schemas
    track_time_changes : bool, optional
        Whether to track time-based changes to animated attributes (default: False)
        When enabled, monitors timeline changes and detects when time-sampled
        attribute values would yield different results

    Attributes
    ----------
    _schema_patterns : list[re.Pattern]
        Compiled regex patterns for schema matching
    _enabled : bool
        Whether change tracking is currently enabled
    _stage : Usd.Stage
        The USD stage being monitored
    _notice_listener : Tf.Notice.Listener
        The registered notice listener for ObjectsChanged events
    _prim_schema_changes : dict
        Nested dictionary tracking changes: {prim_path: {schema: {property: True}}}
    _schema_properties_cache : dict
        Cache of schema property names to avoid repeated lookups
    _track_time_changes : bool
        Whether time-based change tracking is enabled
    _timeline : omni.timeline.ITimeline
        Timeline interface for monitoring time changes
    _timeline_subscription : carb.Subscription
        Subscription to timeline events
    _current_time_code : Usd.TimeCode
        The last known timeline time as a TimeCode
    _time_varying_cache : dict
        Cache of time-varying attribute info: {prim_path: {attr_name: (has_samples, last_bracket)}}

    Examples
    --------
    >>> from pxr import Usd
    >>> from omni.cae.viz.impl.change_tracker import ChangeTracker
    >>>
    >>> stage = Usd.Stage.CreateInMemory()
    >>> tracker = ChangeTracker(stage, schema_regexs=[r"^Cae"])
    >>>
    >>> # Create a prim and make changes
    >>> prim = stage.DefinePrim("/World/Data", "CaeDataSet")
    >>> prim.CreateAttribute("myAttr", Sdf.ValueTypeNames.String).Set("value")
    >>>
    >>> # Check for changes
    >>> if tracker.prim_changed(prim, ["CaeDataSet"]):
    ...     print("Dataset changed!")
    >>>
    >>> # Clear changes after processing
    >>> tracker.clear_changes(prim)
    >>>
    >>> # Disable when done
    >>> tracker.disable()

    See Also
    --------
    Usd.Notice.ObjectsChanged : The USD notice type used for change detection
    Tf.Notice.Register : The USD notification registration system
    """

    def __init__(
        self,
        stage: Usd.Stage,
        schema_regexs: list[str] = [r"^Cae"],
        track_time_changes: bool = False,
        debug_logging: bool = False,
    ):
        """
        Initialize the change tracker and register for USD notifications.

        The tracker will immediately start monitoring changes on the provided stage
        for any schemas matching the provided regex patterns. By default, it tracks
        all schemas starting with "Cae".

        Parameters
        ----------
        stage : Usd.Stage
            The USD stage to track changes on. Must be a valid stage.
        schema_regexs : list[str], optional
            List of regex patterns to match schema names to track.
            Default is [r"^Cae"] which matches all schemas starting with "Cae".

            Examples:
            - [r"^Cae"] - Track all CAE schemas
            - [r"^CaeVtk"] - Track only VTK-related CAE schemas
            - [r"^Cae", r"^UsdGeom"] - Track both CAE and UsdGeom schemas
        track_time_changes : bool, optional
            Whether to track time-based changes to animated attributes.
            Default is False. When enabled, the tracker monitors timeline changes
            and detects when time-sampled attribute values would yield different
            results. This uses efficient time sample bracketing to avoid evaluating
            actual attribute values.
        debug_logging : bool, optional
            Enable verbose debug logging for troubleshooting. Default is False.
            When enabled, logs detailed information about change tracking operations.

        Notes
        -----
        - The tracker is enabled by default upon creation
        - A Tf.Notice listener is automatically registered for ObjectsChanged events
        - Schema property definitions are cached for performance
        - Only property changes (not metadata) are tracked
        - **All prims are initially treated as dirty** since their state is unknown at
          tracker creation time. They remain dirty until explicitly cleared via
          clear_changes() or clear_all_changes()
        - If track_time_changes is enabled, timeline events are also monitored
        """
        self._schema_patterns = [re.compile(pattern) for pattern in schema_regexs]
        self._enabled = True
        self._stage: Optional[Usd.Stage] = stage
        self._notice_listener = None
        self._debug_logging = debug_logging

        # Track changes: {prim_path: {schema_name: {attr_name: True}}}
        self._prim_schema_changes: Dict[str, Dict[str, Dict[str, bool]]] = {}

        # Track resynced prim paths (these and all descendants are "dirty")
        self._resynced_paths: Set[Sdf.Path] = set()

        # Track explicitly cleared paths (to override resynced ancestor state)
        self._cleared_paths: Set[Sdf.Path] = set()

        # Treat all prims as initially dirty (since we don't know their state at creation time)
        # Once a prim is cleared, it's no longer considered initially dirty
        self._treat_all_as_initially_dirty: bool = True

        # Cache of schema properties to avoid repeated lookups
        self._schema_properties_cache: Dict[str, Set[str]] = {}

        # Cache of relationship properties per schema: {schema_name: set of relationship property names}
        self._schema_relationships_cache: Dict[str, Set[str]] = {}

        # Time-based change tracking (optional)
        self._track_time_changes = track_time_changes
        self._timeline = None
        self._timeline_subscription = None
        self._current_time_code = Usd.TimeCode.Default()
        # Cache: {prim_path: {attr_name: (has_samples, last_bracket)}}
        self._time_varying_cache: Dict[str, Dict[str, Tuple[bool, Optional[Tuple[float, float]]]]] = {}

        # Register for USD change notifications
        if self._stage:
            self._notice_listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_objects_changed, self._stage)

        # Setup time tracking if enabled
        if self._track_time_changes:
            self._setup_time_tracking()

        logger.info("ChangeTracker initialized with patterns: %s, time_tracking: %s", schema_regexs, track_time_changes)

    def __del__(self):
        """Cleanup resources."""
        self.disable()

    def enable(self):
        """
        Enable change tracking.

        This method is provided for completeness but currently has no effect
        since the tracker is enabled by default upon creation. Disabling and
        re-enabling would require re-registering for notifications.
        """
        if self._enabled:
            return

        self._enabled = True
        logger.info("ChangeTracker enabled")

    def disable(self):
        """
        Disable change tracking and cleanup resources.

        This method:
        - Revokes the USD notice listener to stop receiving change notifications
        - Unsubscribes from timeline events if time tracking was enabled
        - Clears all tracked change data
        - Clears the schema property cache

        After calling disable(), the tracker will no longer receive or store
        change notifications. The tracker should be disabled when no longer needed
        to free up resources.

        Notes
        -----
        This method is also called automatically in __del__ for cleanup.
        """
        if not self._enabled:
            return

        self._enabled = False

        # Revoke notice listener
        if self._notice_listener:
            self._notice_listener.Revoke()
            self._notice_listener = None

        # Unsubscribe from timeline
        if self._timeline_subscription:
            self._timeline_subscription = None
        self._timeline = None

        # Clear all tracked data
        self._prim_schema_changes.clear()
        self._resynced_paths.clear()
        self._cleared_paths.clear()
        self._schema_properties_cache.clear()
        self._schema_relationships_cache.clear()
        self._time_varying_cache.clear()

        logger.info("ChangeTracker disabled")

    def is_enabled(self) -> bool:
        """
        Check if change tracking is currently enabled.

        Returns
        -------
        bool
            True if the tracker is enabled and receiving notifications, False otherwise
        """
        return self._enabled

    def _debug_log(self, msg: str, *args):
        """
        Log a debug message if debug logging is enabled.

        Args:
            msg: The log message format string
            *args: Arguments to format the message
        """
        if self._debug_logging:
            logger.warning(msg, *args)

    def _get_schema_properties(self, schema_name: str) -> Set[str]:
        """
        Get the set of property names for a given schema (cached).

        Args:
            schema_name: The schema name (without instance)

        Returns:
            Set of property names defined by the schema
        """
        # Check cache first
        if schema_name in self._schema_properties_cache:
            return self._schema_properties_cache[schema_name]

        registry = Usd.SchemaRegistry()

        # Try to find as applied API first, then as concrete type
        defn = registry.FindAppliedAPIPrimDefinition(schema_name)
        if not defn:
            defn = registry.FindConcretePrimDefinition(schema_name)

        if not defn:
            logger.warning("Schema definition not found for: %s", schema_name)
            self._schema_properties_cache[schema_name] = set()
            self._schema_relationships_cache[schema_name] = set()
            return set()

        properties = set(defn.GetPropertyNames())
        self._schema_properties_cache[schema_name] = properties

        # Also cache which properties are relationships
        relationships = set()
        for prop_name in properties:
            prop = defn.GetPropertyDefinition(prop_name)
            if prop and prop.IsRelationship():
                relationships.add(prop_name)
        self._schema_relationships_cache[schema_name] = relationships

        return properties

    def _get_schema_relationships(self, schema_name: str) -> Set[str]:
        """
        Get the set of relationship property names for a given schema (cached).

        Args:
            schema_name: The schema name (without instance)

        Returns:
            Set of relationship property names defined by the schema
        """
        # Ensure properties are cached (which also caches relationships)
        if schema_name not in self._schema_relationships_cache:
            self._get_schema_properties(schema_name)

        return self._schema_relationships_cache.get(schema_name, set())

    def _matches_schema_patterns(self, schema_name: str) -> bool:
        """Check if a schema name matches any of our tracking patterns."""
        for pattern in self._schema_patterns:
            if pattern.match(schema_name):
                return True
        return False

    def _check_relationship_targets_changed(
        self,
        prim: Usd.Prim,
        schemas: list[str] = None,
        visited: Set[str] = None,
        last_time_code: Usd.TimeCode = None,
        current_time_code: Usd.TimeCode = None,
    ) -> bool:
        """
        Check if any relationship targets have changed for a prim's schemas.

        This recursively checks relationship targets to see if they have changes.
        Uses a visited set to detect and break cycles.

        Args:
            prim: The prim to check
            schemas: Optional list of schema names to check (None = all tracked schemas)
            visited: Set of already-visited prim paths (for cycle detection)
            last_time_code: Optional last checked time code for time-based change detection
            current_time_code: Optional current time code to compare against (defaults to self._current_time_code)

        Returns:
            True if any relationship target has changes, False otherwise
        """
        if not prim or not prim.IsValid():
            return False

        prim_path_str = str(prim.GetPath())

        self._debug_log(
            "_check_relationship_targets_changed(%s, schemas=%s, visited=%s)", prim.GetName(), schemas, visited
        )

        # Initialize visited set on first call
        if visited is None:
            visited = set()

        # Detect cycles
        if prim_path_str in visited:
            self._debug_log("  %s already visited (cycle detected)", prim.GetName())
            return False

        visited.add(prim_path_str)

        try:
            # Get applied schemas
            applied_schemas = list(prim.GetAppliedSchemas())
            type_name = prim.GetTypeName()
            if type_name:
                applied_schemas.append(str(type_name))

            self._debug_log("  %s applied_schemas: %s", prim.GetName(), applied_schemas)

            # Filter for matching schemas
            matching_schemas = [s for s in applied_schemas if self._matches_schema_patterns(s.split(":")[0])]
            self._debug_log("  %s matching_schemas: %s", prim.GetName(), matching_schemas)

            # If schemas specified, further filter
            if schemas:
                filtered_schemas = []
                for applied_schema in matching_schemas:
                    for requested_schema in schemas:
                        # Check exact match or base schema match
                        if applied_schema == requested_schema or applied_schema.split(":")[0] == requested_schema:
                            filtered_schemas.append(applied_schema)
                            break
                matching_schemas = filtered_schemas
                self._debug_log("  %s filtered_schemas: %s", prim.GetName(), matching_schemas)

            if not matching_schemas:
                self._debug_log("  %s: No matching schemas, returning False", prim.GetName())
                return False

            # Check each matching schema for relationship changes
            for applied_schema in matching_schemas:
                schema_name, instance_name = self._parse_schema_name(applied_schema)

                # Get relationship properties for this schema
                relationship_props = self._get_schema_relationships(schema_name)
                self._debug_log("  %s: schema=%s, relationships=%s", prim.GetName(), applied_schema, relationship_props)

                if not relationship_props:
                    continue

                # Handle multi-apply instance substitution
                if instance_name:
                    relationship_props = {
                        prop.replace("__INSTANCE_NAME__", instance_name) for prop in relationship_props
                    }

                # Check each relationship
                for rel_name in relationship_props:
                    rel = prim.GetRelationship(rel_name)
                    if not rel or not rel.IsValid():
                        continue

                    # Get relationship targets
                    targets = rel.GetTargets()
                    self._debug_log("    %s.%s targets: %s", prim.GetName(), rel_name, targets)
                    if not targets:
                        continue

                    # Check each target
                    for target_path in targets:
                        self._debug_log("      Checking target: %s", target_path)
                        if target_path.IsPropertyPath():
                            # Target is an attribute - check if the owning prim's attribute changed
                            target_prim_path = target_path.GetPrimPath()
                            target_prim = self._stage.GetPrimAtPath(target_prim_path)
                            if target_prim and target_prim.IsValid():
                                # Check if the specific attribute changed
                                attr_name = target_path.name
                                self._debug_log("        Checking attr %s.%s", target_prim.GetName(), attr_name)
                                if self._attr_changed_internal(
                                    target_prim, attr_name, visited, last_time_code, current_time_code
                                ):
                                    self._debug_log("        -> Attribute changed!")
                                    return True
                        elif target_path.IsPrimPath():
                            # Target is a prim - check if it changed (recursively)
                            target_prim = self._stage.GetPrimAtPath(target_path)
                            if target_prim and target_prim.IsValid():
                                # Recursively check the target prim
                                self._debug_log("        Recursively checking prim %s", target_prim.GetName())
                                if self._prim_changed_internal(
                                    target_prim, None, visited, last_time_code, current_time_code
                                ):
                                    self._debug_log("        -> Prim changed!")
                                    return True

            self._debug_log("  %s: No relationship target changes found", prim.GetName())
            return False
        finally:
            # Remove from visited set when we're done with this branch
            visited.discard(prim_path_str)

    def _setup_time_tracking(self):
        """Setup timeline monitoring for time-based change tracking."""
        try:
            self._timeline = omni.timeline.get_timeline_interface()
            if self._timeline:
                # Get initial time as TimeCode
                self._current_time_code = Usd.TimeCode(
                    round(self._timeline.get_current_time() * self._timeline.get_time_codes_per_seconds())
                )
                event_dispatcher = get_eventdispatcher()
                self._timeline_subscription = event_dispatcher.observe_event(
                    observer_name="ChangeTracker_Timeline",
                    filter=self._timeline.get_event_key(),
                    event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
                    on_event=lambda event: self._on_timeline_event(event),
                )
                self._debug_log("Timeline tracking enabled, initial time: %s", self._current_time_code)
            else:
                logger.warning("Timeline interface not available, time tracking disabled")
        except Exception as e:
            logger.error("Failed to setup time tracking: %s", e, exc_info=True)
            self._track_time_changes = False

    def _get_time_sample_bracket(self, attr: Usd.Attribute, time_code: Usd.TimeCode) -> Optional[Tuple[float, float]]:
        """
        Get the bracketing time samples for an attribute at a given time.

        Returns None if attribute has no time samples, or tuple of (lower, upper) times.
        """
        if not attr or not attr.IsValid():
            return None

        try:
            # Check if attribute has any time samples
            time_samples = attr.GetTimeSamples()
            if not time_samples:
                return None

            # Get bracketing samples
            lower, upper = attr.GetBracketingTimeSamples(time_code.GetValue())
            return (lower, upper)
        except Exception as e:
            self._debug_log("Error getting time samples for %s: %s", attr.GetPath(), e)
            return None

    def _check_prim_for_time_changes(self, prim: Usd.Prim, last_time_code: Usd.TimeCode, new_time_code: Usd.TimeCode):
        """
        Check a prim for time-varying attribute changes between two time codes.

        This efficiently checks if any time-sampled attributes would have
        different values at the new time vs the old time by comparing
        time sample brackets.

        Args:
            prim: The prim to check
            last_time_code: The previous time code to compare from
            new_time_code: The new time code to compare to
        """
        if not prim or not prim.IsValid():
            return

        prim_path_str = str(prim.GetPath())

        # Get applied schemas
        applied_schemas = list(prim.GetAppliedSchemas())
        type_name = prim.GetTypeName()
        if type_name:
            applied_schemas.append(str(type_name))

        # Filter for matching schemas
        matching_schemas = [s for s in applied_schemas if self._matches_schema_patterns(s.split(":")[0])]
        if not matching_schemas:
            return

        # Initialize caches if needed
        if prim_path_str not in self._time_varying_cache:
            self._time_varying_cache[prim_path_str] = {}

        # Check each matching schema
        for applied_schema in matching_schemas:
            schema_name, instance_name = self._parse_schema_name(applied_schema)
            schema_properties = self._get_schema_properties(schema_name)

            if not schema_properties:
                continue

            # Handle multi-apply instance substitution
            if instance_name:
                schema_properties = {prop.replace("__INSTANCE_NAME__", instance_name) for prop in schema_properties}

            schema_key = applied_schema

            # Check each property
            for prop_name in schema_properties:
                attr = prim.GetAttribute(prop_name)
                if not attr or not attr.IsValid():
                    continue

                # Check cache for time-varying status and last bracket
                cache_key = prop_name
                if cache_key in self._time_varying_cache[prim_path_str]:
                    has_samples, _ = self._time_varying_cache[prim_path_str][cache_key]
                    if not has_samples:
                        continue  # Not time-varying, skip
                else:
                    # First time seeing this attribute, check if time-varying
                    has_samples = len(attr.GetTimeSamples()) > 0
                    self._time_varying_cache[prim_path_str][cache_key] = (has_samples, None)
                    if not has_samples:
                        continue

                # Get brackets at both times
                last_bracket = self._get_time_sample_bracket(attr, last_time_code)
                new_bracket = self._get_time_sample_bracket(attr, new_time_code)

                # Check if bracket changed (meaning interpolation would differ)
                if last_bracket != new_bracket:
                    # Mark as changed
                    if prim_path_str not in self._prim_schema_changes:
                        self._prim_schema_changes[prim_path_str] = {}
                    if schema_key not in self._prim_schema_changes[prim_path_str]:
                        self._prim_schema_changes[prim_path_str][schema_key] = {}

                    self._prim_schema_changes[prim_path_str][schema_key][prop_name] = True
                    self._debug_log(
                        "Time-varying property changed: %s[%s].%s (bracket: %s -> %s)",
                        prim_path_str,
                        schema_key,
                        prop_name,
                        last_bracket,
                        new_bracket,
                    )

    def _on_timeline_event(self, event: carb.events.IEvent):
        """Handle timeline events for time-based change tracking."""
        try:
            if not self._enabled or not self._stage or not self._track_time_changes:
                return

            # Get current time as TimeCode
            new_time_code = Usd.TimeCode(
                round(self._timeline.get_current_time() * self._timeline.get_time_codes_per_seconds())
            )

            # Check if time actually changed
            if self._current_time_code == new_time_code:
                return

            old_time_code = self._current_time_code
            self._current_time_code = new_time_code

            self._debug_log("Timeline time changed: %s -> %s", old_time_code, new_time_code)

            # Don't check prims here - do it lazily when prim_changed() is called
            # This is much more efficient since we only check prims that are actually queried

        except Exception as e:
            logger.error("Error in _on_timeline_event: %s", e, exc_info=True)

    def _has_resynced_ancestor(self, prim_path: Sdf.Path) -> bool:
        """
        Check if prim_path or any of its ancestors is in the resynced paths set.

        Args:
            prim_path: The prim path to check

        Returns:
            True if the path or any ancestor was resynced, False otherwise
        """
        path = prim_path
        while not path.isEmpty:
            if path in self._resynced_paths:
                return True
            path = path.GetParentPath()
        return False

    def _parse_schema_name(self, applied_schema: str) -> Tuple[str, Optional[str]]:
        """
        Parse an applied schema name to extract schema name and instance.

        Args:
            applied_schema: The applied schema string (e.g., "CaeVizFieldSelectionAPI:velocity")

        Returns:
            Tuple of (schema_name, instance_name)
        """
        try:
            registry = Usd.SchemaRegistry()
            schema_name, instance_name = registry.GetTypeNameAndInstance(applied_schema)
            return str(schema_name), str(instance_name) if instance_name else None
        except Exception as e:
            logger.warning("Failed to parse schema name '%s': %s", applied_schema, e)
            # Fallback to simple split
            if ":" in applied_schema:
                parts = applied_schema.split(":", 1)
                return parts[0], parts[1]
            return applied_schema, None

    def _on_objects_changed(self, notice: Usd.Notice.ObjectsChanged, sender: Usd.Stage):
        """
        Called when objects on the stage change (via Tf.Notice).

        Args:
            notice: The ObjectsChanged notice
            sender: The stage that sent the notice
        """
        try:
            if not self._stage or sender != self._stage:
                return

            changed_paths = []

            # Handle resynced paths - these indicate entire subtrees are "dirty"
            resynced_paths = notice.GetResyncedPaths()
            for path in resynced_paths:
                if path.IsPrimPath():
                    self._resynced_paths.add(path)
                    self._debug_log("Resynced path (subtree dirty): %s", path)

                    # Remove this path and all descendants from cleared paths
                    # since resync makes them dirty again
                    cleared_to_remove = []
                    for cleared_path in self._cleared_paths:
                        # Check if cleared_path is the resynced path or has it as prefix (descendant)
                        if cleared_path == path or cleared_path.HasPrefix(path):
                            cleared_to_remove.append(cleared_path)

                    for cleared_path in cleared_to_remove:
                        self._cleared_paths.remove(cleared_path)
                        self._debug_log("Removed from cleared paths due to resync: %s", cleared_path)

                    # Invalidate time cache for resynced prims (structure may have changed)
                    if self._track_time_changes:
                        prim_path_str = str(path)
                        if prim_path_str in self._time_varying_cache:
                            del self._time_varying_cache[prim_path_str]
                        self._debug_log("Invalidated time cache due to resync: %s", prim_path_str)
                elif path.IsPropertyPath():
                    # implies a new property was added; tread it as a changed property
                    changed_paths.append(path)

            # Get info-only changed paths for detailed property tracking
            changed_paths += notice.GetChangedInfoOnlyPaths()

            # Track prims we've already processed and their changed properties
            # {prim_path_str: set of property names}
            prim_changed_properties: Dict[str, Set[str]] = {}

            # First pass: collect all changed properties per prim
            for path in changed_paths:
                if path.IsPropertyPath():
                    # This is a property/attribute change
                    prim_path = path.GetPrimPath()
                    prim_path_str = str(prim_path)
                    property_name = path.name

                    if prim_path_str not in prim_changed_properties:
                        prim_changed_properties[prim_path_str] = set()
                    prim_changed_properties[prim_path_str].add(property_name)

            # Second pass: process each prim with changes
            for prim_path_str, changed_property_names in prim_changed_properties.items():
                prim_path = Sdf.Path(prim_path_str)

                # Get the prim
                prim = self._stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    continue

                # Get applied schemas and type schema
                applied_schemas = list(prim.GetAppliedSchemas())
                type_name = prim.GetTypeName()
                if type_name:
                    applied_schemas.append(str(type_name))

                # Filter for schemas matching our patterns - do this early to skip unnecessary work
                matching_schemas = [s for s in applied_schemas if self._matches_schema_patterns(s.split(":")[0])]

                # Skip processing if no matching schemas
                if not matching_schemas:
                    continue

                # Ensure we have a changes dict for this prim
                if prim_path_str not in self._prim_schema_changes:
                    self._prim_schema_changes[prim_path_str] = {}

                # Check each matching schema
                for applied_schema in matching_schemas:
                    schema_name, instance_name = self._parse_schema_name(applied_schema)

                    # Get properties for this schema
                    # NOTE: for multi-apply schemas, schema property names include `:__INSTANCE_NAME__:`
                    schema_properties = self._get_schema_properties(schema_name)

                    if not schema_properties:
                        continue

                    if instance_name:
                        schema_properties = {
                            prop.replace("__INSTANCE_NAME__", instance_name) for prop in schema_properties
                        }

                    # Build schema key (with or without instance)
                    schema_key = applied_schema

                    # Check if any of the schema's properties are in the changed properties
                    for property_name in changed_property_names:
                        # Check if this property corresponds to a schema property
                        if property_name in schema_properties:
                            # Initialize schema changes dict if needed
                            if schema_key not in self._prim_schema_changes[prim_path_str]:
                                self._prim_schema_changes[prim_path_str][schema_key] = {}

                            self._prim_schema_changes[prim_path_str][schema_key][property_name] = True
                            self._debug_log(
                                "Schema property changed: %s[%s].%s", prim_path_str, schema_key, property_name
                            )
        except Exception as e:
            logger.error("Error in _on_objects_changed: %s", e, exc_info=True)

    def _prim_changed_internal(
        self,
        prim: Usd.Prim,
        schemas: list[str] = None,
        visited: Set[str] = None,
        last_time_code: Usd.TimeCode = None,
        current_time_code: Usd.TimeCode = None,
    ) -> bool:
        """
        Internal version of prim_changed that accepts a visited set for cycle detection.

        Args:
            prim: The prim to check
            schemas: Optional list of schema names
            visited: Set of visited prim paths (for cycle detection during relationship traversal)
            last_time_code: Optional last checked time code for time-based change detection
            current_time_code: Optional current time code to compare against (defaults to self._current_time_code)

        Returns:
            True if prim has changes, False otherwise
        """
        self._debug_log("Checking %s for changes", prim.GetName() if prim else "None")
        if not prim or not prim.IsValid():
            return False

        prim_path_str = str(prim.GetPath())
        prim_path = prim.GetPath()

        # Use provided current_time_code or fall back to internal state
        effective_current_time = current_time_code if current_time_code is not None else self._current_time_code

        self._debug_log(
            "  prim_path: %s, schemas: %s, visited: %s, last_time: %s", prim_path_str, schemas, visited, last_time_code
        )

        # If time tracking is enabled and last_time_code provided, check for time-based changes
        if self._track_time_changes and last_time_code is not None:
            # Check if timeline time changed since last check
            if last_time_code != effective_current_time:
                self._debug_log(
                    "  Checking time changes for %s: last=%s, current=%s",
                    prim.GetName(),
                    last_time_code,
                    effective_current_time,
                )
                # Timeline changed since last check, evaluate time changes
                self._check_prim_for_time_changes(prim, last_time_code, effective_current_time)

        # First check if this path was explicitly cleared (overrides resync state and initial dirty state)
        if prim_path in self._cleared_paths:
            # Check only the detailed property changes (ignore resynced ancestors and initial state)
            if prim_path_str not in self._prim_schema_changes:
                self._debug_log("  %s cleared but no changes tracked", prim.GetName())
                prim_changes = {}
            else:
                prim_changes = self._prim_schema_changes[prim_path_str]
                self._debug_log("  %s cleared, has changes: %s", prim.GetName(), list(prim_changes.keys()))
        else:
            # If treating all as initially dirty and this prim hasn't been cleared yet, return True
            if self._treat_all_as_initially_dirty:
                self._debug_log("  %s treated as initially dirty", prim.GetName())
                return True

            # Check if this prim or any ancestor was resynced (entire subtree is dirty)
            if self._has_resynced_ancestor(prim_path):
                self._debug_log("  %s has resynced ancestor", prim.GetName())
                return True

            # Check detailed property changes
            if prim_path_str not in self._prim_schema_changes:
                self._debug_log("  %s has no tracked changes", prim.GetName())
                prim_changes = {}
            else:
                prim_changes = self._prim_schema_changes[prim_path_str]
                self._debug_log("  %s has changes: %s", prim.GetName(), list(prim_changes.keys()))

        # If no schemas specified, check if ANY tracked schema has changes
        if not schemas:
            self._debug_log("  %s: Checking all schemas", prim.GetName())
            for schema_changes in prim_changes.values():
                if schema_changes:  # If any property changed in this schema
                    self._debug_log("  %s: Found direct property changes", prim.GetName())
                    return True
            # Don't return False yet - check relationships below
            self._debug_log("  %s: No direct changes, checking relationships...", prim.GetName())
        else:
            # Check specific schemas
            self._debug_log("  %s: Checking specific schemas: %s", prim.GetName(), schemas)
            for schema in schemas:
                # Check exact match first (with instance name if provided)
                if schema in prim_changes and prim_changes[schema]:
                    self._debug_log("  %s: Found changes in %s", prim.GetName(), schema)
                    return True

                # If no instance name, check all instances of this schema
                if ":" not in schema:
                    for tracked_schema in prim_changes:
                        base_schema = tracked_schema.split(":")[0]
                        if base_schema == schema and prim_changes[tracked_schema]:
                            self._debug_log("  %s: Found changes in %s", prim.GetName(), tracked_schema)
                            return True
            self._debug_log("  %s: No direct schema changes, checking relationships...", prim.GetName())

        # Also check relationship targets (recursively)
        self._debug_log("  %s: Calling _check_relationship_targets_changed", prim.GetName())
        if self._check_relationship_targets_changed(prim, schemas, visited, last_time_code, current_time_code):
            self._debug_log("  %s: Relationship targets changed!", prim.GetName())
            return True

        self._debug_log("  %s: No changes found", prim.GetName())
        return False

    def prim_changed(
        self,
        prim: Usd.Prim,
        schemas: list[str] = None,
        last_time_code: Usd.TimeCode = None,
        current_time_code: Usd.TimeCode = None,
    ) -> bool:
        """
        Check if any properties of the specified schemas have changed on a prim.

        This method returns True if any property belonging to one of the specified
        schemas has changed on the given prim since the last time changes were cleared.
        This includes both structural changes (from USD notices), time-based changes
        (if time tracking is enabled), and relationship target changes.

        Parameters
        ----------
        prim : Usd.Prim
            The USD prim to check for changes
        schemas : list[str], optional
            List of schema names to check. Can include:
            - Type schemas: "CaeDataSet"
            - Single-apply API schemas: "CaePointCloudAPI"
            - Multi-apply with instance: "CaeVizFieldSelectionAPI:velocity"
            - Multi-apply without instance: "CaeVizFieldSelectionAPI" (matches any instance)
            If None or empty, checks if ANY tracked schema on the prim has changed.
        last_time_code : Usd.TimeCode, optional
            The last time code that was checked. If provided and time tracking is enabled,
            will check for time-based changes between last_time_code and current_time_code.
            If None, only structural changes are checked.
        current_time_code : Usd.TimeCode, optional
            The current time code to compare against. If not provided, uses the timeline's
            current time code (automatically tracked). This allows explicit control over
            time comparisons without relying on timeline state.

        Returns
        -------
        bool
            True if any property of the specified schemas has changed (structurally,
            due to time, or relationship targets), False otherwise

        Notes
        -----
        - For multi-apply schemas without an instance name, returns True if ANY
          instance of that schema has changed
        - Only tracks properties that are defined by the schema (not all attributes)
        - Changes persist until explicitly cleared with clear_changes()
        - If schemas is None or empty, returns True if ANY tracked schema changed
        - If time tracking is enabled, also checks for time-based changes on first call
        - Relationship targets are checked recursively, with cycle detection
        - If current_time_code is provided, it overrides the timeline's current time

        Examples
        --------
        >>> # Check if a typed prim's properties changed
        >>> if tracker.prim_changed(prim, ["CaeDataSet"]):
        ...     print("Dataset properties changed")
        >>>
        >>> # Check for time-based changes from last_time to current timeline time
        >>> if tracker.prim_changed(prim, ["CaeDataSet"], last_time_code=last_time):
        ...     print("Dataset changed due to time or structure")
        >>>
        >>> # Explicit time comparison: check changes from time A to time B
        >>> if tracker.prim_changed(prim, ["CaeDataSet"], last_time_code=time_a, current_time_code=time_b):
        ...     print("Dataset would change when moving from time A to time B")
        >>>
        >>> # Check for specific multi-apply instance
        >>> if tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI:velocity"]):
        ...     print("Velocity field selection changed")
        >>>
        >>> # Check for any instance of a multi-apply schema
        >>> if tracker.prim_changed(prim, ["CaeVizFieldSelectionAPI"]):
        ...     print("Some field selection changed")
        >>>
        >>> # Check multiple schemas at once
        >>> if tracker.prim_changed(prim, ["CaeDataSet", "CaePointCloudAPI"]):
        ...     print("Either schema changed")
        >>>
        >>> # Check if ANY tracked schema changed on this prim
        >>> if tracker.prim_changed(prim):
        ...     print("Some tracked schema changed")
        """
        # Check direct changes AND relationship targets
        # Pass None as visited to start fresh cycle detection
        # Pass time codes through for time-based checking
        return self._prim_changed_internal(
            prim, schemas, visited=None, last_time_code=last_time_code, current_time_code=current_time_code
        )

    def _attr_changed_internal(
        self,
        prim: Usd.Prim,
        attr_name: str,
        visited: Set[str] = None,
        last_time_code: Usd.TimeCode = None,
        current_time_code: Usd.TimeCode = None,
    ) -> bool:
        """
        Internal version of attr_changed that accepts a visited set for cycle detection.

        Args:
            prim: The prim to check
            attr_name: The attribute name
            visited: Set of visited prim paths (for cycle detection during relationship traversal)
            last_time_code: Optional last checked time code for time-based change detection
            current_time_code: Optional current time code to compare against (defaults to self._current_time_code)

        Returns:
            True if attribute has changes, False otherwise
        """
        if not prim or not prim.IsValid():
            return False

        prim_path_str = str(prim.GetPath())

        # Use provided current_time_code or fall back to internal state
        effective_current_time = current_time_code if current_time_code is not None else self._current_time_code

        # If time tracking is enabled and last_time_code provided, check for time changes
        if self._track_time_changes and last_time_code is not None:
            if last_time_code != effective_current_time:
                self._check_prim_for_time_changes(prim, last_time_code, effective_current_time)

        if prim_path_str not in self._prim_schema_changes:
            return False

        # Check if this attribute appears in any of the tracked schemas for this prim
        for schema_changes in self._prim_schema_changes[prim_path_str].values():
            if attr_name in schema_changes:
                return True

        return False

    def attr_changed(
        self,
        prim: Usd.Prim,
        attr_name: str,
        last_time_code: Usd.TimeCode = None,
        current_time_code: Usd.TimeCode = None,
    ) -> bool:
        """
        Check if a specific attribute has changed on a prim.

        This method returns True if the specified attribute has changed on the prim,
        but ONLY if that attribute belongs to a schema matching the tracker's patterns.
        Attributes not belonging to tracked schemas will not be detected.

        For relationships, this checks if any of the relationship's recursive targets
        have changed, with cycle detection.

        Parameters
        ----------
        prim : Usd.Prim
            The USD prim to check
        attr_name : str
            The name of the attribute or relationship to check for changes
        last_time_code : Usd.TimeCode, optional
            The last time code that was checked. If provided and time tracking is enabled,
            will check for time-based changes between last_time_code and current_time_code.
            If None, only structural changes are checked.
        current_time_code : Usd.TimeCode, optional
            The current time code to compare against. If not provided, uses the timeline's
            current time code (automatically tracked). This allows explicit control over
            time comparisons without relying on timeline state.

        Returns
        -------
        bool
            True if the attribute changed and belongs to a tracked schema, or if it's
            a relationship and any of its targets changed, False otherwise

        Notes
        -----
        - Only attributes defined by schemas matching the configured patterns are tracked
        - Attributes not part of any schema definition are not tracked
        - For multi-apply schemas, the attribute name should include the instance
          (e.g., "cae:viz:target:velocity" not just "target")
        - For relationships, recursively checks if targets changed (with cycle detection)
        - If current_time_code is provided, it overrides the timeline's current time

        Examples
        --------
        >>> # Check if a specific attribute changed
        >>> if tracker.attr_changed(prim, "fileNames"):
        ...     print("fileNames attribute changed")
        >>>
        >>> # Check for time-based changes from last_time to current timeline time
        >>> if tracker.attr_changed(prim, "fileNames", last_time_code=last_time):
        ...     print("fileNames changed due to time or structure")
        >>>
        >>> # Explicit time comparison: check changes from time A to time B
        >>> if tracker.attr_changed(prim, "fileNames", last_time_code=time_a, current_time_code=time_b):
        ...     print("fileNames would change when moving from time A to time B")
        >>>
        >>> # For multi-apply schema attributes
        >>> if tracker.attr_changed(prim, "cae:viz:target:velocity"):
        ...     print("Velocity target changed")
        >>>
        >>> # For relationships (checks if targets changed)
        >>> if tracker.attr_changed(prim, "coordinates"):
        ...     print("Coordinates relationship target changed")
        """
        # Check direct attribute change first
        if self._attr_changed_internal(
            prim, attr_name, visited=None, last_time_code=last_time_code, current_time_code=current_time_code
        ):
            return True

        # If it's a relationship, check if any targets changed
        if not prim or not prim.IsValid():
            return False

        rel = prim.GetRelationship(attr_name)
        if rel and rel.IsValid():
            # Initialize visited set for cycle detection
            visited = set()
            prim_path_str = str(prim.GetPath())
            visited.add(prim_path_str)

            # Get relationship targets
            targets = rel.GetTargets()
            if targets:
                for target_path in targets:
                    if target_path.IsPropertyPath():
                        # Target is an attribute
                        target_prim_path = target_path.GetPrimPath()
                        target_prim = self._stage.GetPrimAtPath(target_prim_path)
                        if target_prim and target_prim.IsValid():
                            target_attr_name = target_path.name
                            if self._attr_changed_internal(
                                target_prim, target_attr_name, visited, last_time_code, current_time_code
                            ):
                                return True
                    elif target_path.IsPrimPath():
                        # Target is a prim
                        target_prim = self._stage.GetPrimAtPath(target_path)
                        if target_prim and target_prim.IsValid():
                            if self._prim_changed_internal(
                                target_prim, None, visited, last_time_code, current_time_code
                            ):
                                return True

        return False

    def clear_changes(self, prim: Usd.Prim, schema_names: list[str] = None):
        """
        Clear tracked changes for a prim.

        This method removes change records for the specified prim. You can either
        clear changes for specific schemas or clear all changes for the prim.

        Parameters
        ----------
        prim : Usd.Prim
            The USD prim to clear changes for
        schema_names : list[str], optional
            List of schema names to clear changes for. If None, clears all changes
            for the prim. Schema names can include instances for multi-apply schemas.

        Notes
        -----
        - For multi-apply schemas without instance names, clears all instances
        - After clearing, prim_changed() and attr_changed() will return False
          until new changes occur
        - If prim structure changes (attributes added/removed), the time-varying
          cache for that prim is invalidated

        Examples
        --------
        >>> # Clear all changes for a prim
        >>> tracker.clear_changes(prim)
        >>>
        >>> # Clear changes for specific schemas
        >>> tracker.clear_changes(prim, ["CaeDataSet", "CaePointCloudAPI"])
        >>>
        >>> # Clear a specific multi-apply instance
        >>> tracker.clear_changes(prim, ["CaeVizFieldSelectionAPI:velocity"])
        >>>
        >>> # Clear all instances of a multi-apply schema
        >>> tracker.clear_changes(prim, ["CaeVizFieldSelectionAPI"])
        """
        if not prim or not prim.IsValid():
            return

        prim_path_str = str(prim.GetPath())
        prim_path = prim.GetPath()

        if schema_names is None:
            # Clear all changes for this prim
            if prim_path_str in self._prim_schema_changes:
                del self._prim_schema_changes[prim_path_str]

            # Remove from resynced paths if present
            self._resynced_paths.discard(prim_path)

            # Add to cleared paths to override any resynced ancestor
            self._cleared_paths.add(prim_path)

            self._debug_log("Cleared all changes for prim: %s", prim_path)
        else:
            # Clear changes for specific schemas
            if prim_path_str in self._prim_schema_changes:
                for schema_name in schema_names:
                    # Clear exact matches
                    if schema_name in self._prim_schema_changes[prim_path_str]:
                        del self._prim_schema_changes[prim_path_str][schema_name]
                        self._debug_log("Cleared changes for %s[%s]", prim_path_str, schema_name)

                    # Clear all instances if no instance name provided
                    if ":" not in schema_name:
                        to_remove = []
                        for tracked_schema in self._prim_schema_changes[prim_path_str]:
                            if tracked_schema.split(":")[0] == schema_name:
                                to_remove.append(tracked_schema)
                        for schema in to_remove:
                            del self._prim_schema_changes[prim_path_str][schema]
                            self._debug_log("Cleared changes for %s[%s]", prim_path_str, schema)

            # When clearing specific schemas, still add to cleared paths to override resynced ancestor
            self._cleared_paths.add(prim_path)

            # Remove from resynced paths if present
            self._resynced_paths.discard(prim_path)

    def has_changes(self) -> bool:
        """
        Check if there are any tracked changes.

        This method returns True if the tracker has recorded any changes for any
        prim since the last time changes were cleared. This includes both property
        changes and resynced paths (which indicate entire subtree changes).

        Returns
        -------
        bool
            True if any changes are tracked, False otherwise

        Notes
        -----
        - Returns True if there are any property changes in tracked schemas
        - Returns True if there are any resynced paths (full subtree changes)
        - Returns False if no changes have occurred or all changes have been cleared
        - This is useful for determining if processing is needed before iterating
          through specific prims

        Examples
        --------
        >>> # Only process changes if there are any
        >>> if tracker.has_changes():
        ...     for prim_path in get_all_prims_to_check():
        ...         prim = stage.GetPrimAtPath(prim_path)
        ...         if tracker.prim_changed(prim):
        ...             process_prim_changes(prim)
        ...     tracker.clear_all_changes()
        >>>
        >>> # Skip expensive operations if nothing changed
        >>> if not tracker.has_changes():
        ...     return  # Nothing to do
        """
        # Check if we're treating all as initially dirty
        if self._treat_all_as_initially_dirty:
            return True

        # Check if there are any resynced paths (full subtree changes)
        if self._resynced_paths:
            return True

        # Check if there are any property changes
        if self._prim_schema_changes:
            # Make sure there's at least one non-empty schema change dict
            for schema_changes in self._prim_schema_changes.values():
                if schema_changes:  # If this prim has any schema changes
                    return True

        return False

    def clear_all_changes(self):
        """
        Clear all tracked changes for all prims.

        This removes all change records from the tracker, including resynced paths
        and cleared paths. After calling this method, prim_changed() and attr_changed()
        will return False for all prims until new changes occur.

        This also clears the initial "all dirty" state, meaning prims will only be
        considered dirty if they have actual tracked changes.

        This is useful for resetting the tracker's state after processing a batch
        of changes or at specific checkpoints in your application.

        Examples
        --------
        >>> # Process all changes, then clear
        >>> for prim_path in get_all_prims_to_check():
        ...     prim = stage.GetPrimAtPath(prim_path)
        ...     if tracker.prim_changed(prim, ["CaeDataSet"]):
        ...         process_dataset_changes(prim)
        >>>
        >>> # Clear all changes after processing
        >>> tracker.clear_all_changes()
        """
        self._prim_schema_changes.clear()
        self._resynced_paths.clear()
        self._cleared_paths.clear()
        self._treat_all_as_initially_dirty = False
        logger.debug("All changes cleared")

    def invalidate_time_cache(self, prim: Usd.Prim = None):
        """
        Invalidate the time-varying attribute cache.

        Call this when prim structure changes (e.g., attributes added/removed) to
        force re-checking of time sample information. This is automatically called
        when structural changes are detected via USD notices.

        Parameters
        ----------
        prim : Usd.Prim, optional
            The prim to invalidate cache for. If None, clears entire cache.

        Notes
        -----
        - Generally not needed as structural changes are detected automatically
        - Useful if you manually modify attributes outside USD's change notification system

        Examples
        --------
        >>> # Invalidate cache for a specific prim
        >>> tracker.invalidate_time_cache(prim)
        >>>
        >>> # Invalidate entire cache
        >>> tracker.invalidate_time_cache()
        """
        if not self._track_time_changes:
            return

        if prim is None:
            self._time_varying_cache.clear()
            self._debug_log("Cleared entire time-varying cache")
        else:
            prim_path_str = str(prim.GetPath())
            if prim_path_str in self._time_varying_cache:
                del self._time_varying_cache[prim_path_str]
            self._debug_log("Cleared time-varying cache for: %s", prim_path_str)
