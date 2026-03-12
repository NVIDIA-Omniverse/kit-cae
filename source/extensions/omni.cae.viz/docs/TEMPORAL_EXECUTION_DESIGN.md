# Temporal Execution Design for CaeVizOperatorTemporalAPI

## Overview

This document describes the implementation of temporal execution optimization for CAE visualization operators. The design allows operators to skip expensive rebuilds when only the time changes, significantly improving interactive playback performance.

## Current Implementation Status

✅ **Fully Implemented**:
- `ExecutionContext` with reason, timecodes (snapped, raw, next), and device
- `ExecutionReason` enum (INITIAL, STRUCTURAL_CHANGE, TEMPORAL_UPDATE, TEMPORAL_TICK)
- Generator-based `_build_execution_context()` yielding multiple contexts per frame
- Operator decorator flags: `supports_temporal` and `tick_on_time_change`
- Temporal state tracking in controller with `executed_timecodes` set
- `on_time_changed()` hook for lightweight updates (called even when exec skips)
- Field interpolation support with dual-timestep attribute allocation
- Both `IrregularVolume` and `NanoVDBVolume` support temporal mode
- Timecode snapping via `get_bracketing_time_codes()`
- All operators migrated to use `ExecutionContext` (required parameter)

## Core Concept

When `CaeVizOperatorTemporalAPI` is applied to an operator prim with `enableFieldInterpolation=true`, the controller intelligently decides when to execute the operator:

1. **Structural Change**: Prim properties changed (independent of time) → **Execute (full rebuild)**
2. **New Timecode**: Time changed to a value not seen since last structural change → **Execute (temporal update only)**
3. **Repeat Timecode**: Same timecode as before → **Skip execution**

## Implementation

### 1. ExecutionContext

**File**: `python/execution_context.py`

New dataclass that provides operators with execution reason information:

```python
class ExecutionReason(Enum):
    INITIAL = "initial"                # First execution
    STRUCTURAL_CHANGE = "structural"   # Time-independent changes
    TEMPORAL_UPDATE = "temporal"       # Time-only change (first time at this timecode)
    TEMPORAL_TICK = "temporal_tick"    # Time changed (minimal update only)

@dataclass
class ExecutionContext:
    reason: ExecutionReason
    timecode: Usd.TimeCode              # Snapped timecode (from get_bracketing_time_codes)
    raw_timecode: Usd.TimeCode          # Original timeline timecode
    next_time_code: Optional[Usd.TimeCode]  # Next bracketing timecode (if available)
    device: str

    def is_full_rebuild_needed(self) -> bool:
        """True if full rebuild needed, False for temporal-only updates."""
        return self.reason != ExecutionReason.TEMPORAL_UPDATE

    def is_temporal_update(self) -> bool:
        """True if this is a temporal-only update."""
        return self.reason == ExecutionReason.TEMPORAL_UPDATE

    def is_temporal_tick(self) -> bool:
        """True if this is a minimal time tick."""
        return self.reason == ExecutionReason.TEMPORAL_TICK
```

### 2. Operator Decorator Enhancement

**File**: `python/operator.py`

Added `supports_temporal` and `tick_on_time_change` parameters to `@operator()` decorator:

```python
@operator(supports_temporal=True, tick_on_time_change=True)
class NanoVDBVolume(VolumeBase):
    async def exec(self, prim, device, context: ExecutionContext):
        """
        Main execution method. Context provides reason, timecodes, and device.

        Note: timeCode is accessed via context.timecode (snapped) or context.raw_timecode
        """
        if context.is_temporal_update():
            # Fast path: only update fields (first time at this timecode)
            await self._exec_temporal_update(prim, device, context)
        else:
            # Full rebuild
            await super().exec(prim, device, context)

    async def on_time_changed(self, prim, device, context: ExecutionContext):
        """
        Lightweight hook called on EVERY time change when tick_on_time_change=True.

        Called regardless of whether exec() runs. Keep this VERY fast!

        When enableFieldInterpolation=True: Called for ALL raw timecodes
        When enableFieldInterpolation=False: Called only when snapped timecode changes

        Use cases:
        - Update cached time references
        - Refresh display parameters
        - Update shader uniform values
        - Update DataLoader time parameters
        """
        # Update DataLoader to trigger IndeX ComputeTask
        with viz_utils.edit_context(prim):
            if loader := prim.GetChild("Material").GetChild("DataLoader"):
                s_loader = UsdShade.Shader(loader)
                s_loader.CreateInput("params_time_code", ...).Set(str(context.timecode))
                s_loader.CreateInput("params_next_time_code", ...).Set(str(context.next_time_code))
```

### 3. Controller Logic

**File**: `python/controller.py`

#### Temporal State Tracking

Extended `_last_execution_cache` with temporal state:

```python
{
    "timecode": Usd.TimeCode,
    "raw_timecode": Usd.TimeCode,
    "device": str,
    "operator_class": Type,
    "temporal_state": {
        "executed_timecodes": set()  # Set of float timecode values
    } or None
}
```

#### Execution Decision Logic

New method `_build_execution_context()` is a **generator** that yields one or more `ExecutionContext` objects:

1. **Check forced execution conditions**:
   - Operator class changed
   - Device changed
   - Input transform changed
   - Prim changed (time-independent via `ChangeTracker.prim_changed()`)
   → Clear temporal state, yield STRUCTURAL_CHANGE context

2. **Check if temporal mode applies**:
   - Operator must have `supports_temporal=True`
   - `CaeVizOperatorTemporalAPI` may be applied with `enableFieldInterpolation=True`

3. **Timecode snapping**: All timecodes are snapped using `get_bracketing_time_codes()` to align with actual time samples

4. **If temporal mode active**:
   - Check if `current_time_code` has been executed → If not, yield TEMPORAL_UPDATE
   - If `enableFieldInterpolation=True` and `next_time_code` exists:
     * Check if `next_time_code` has been executed → If not, yield another TEMPORAL_UPDATE
   - After exec contexts, check if tick is needed:
     * If `tick_on_time_change=True`:
       - With interpolation: Yield TEMPORAL_TICK if `raw_timecode` changed
       - Without interpolation: Yield TEMPORAL_TICK if snapped `timecode` changed

5. **Generator pattern**: The controller iterates over yielded contexts and executes accordingly:
   - `exec()` is called for STRUCTURAL_CHANGE and TEMPORAL_UPDATE contexts
   - `on_time_changed()` is called for TEMPORAL_TICK contexts

**Key behavior:**
- Multiple contexts can be yielded per call (e.g., exec for current, exec for next, then tick)
- Tick always fires after exec contexts (if applicable)
- Errors in one context don't prevent subsequent contexts from executing

### 4. Change Tracking Integration

The design leverages `ChangeTracker`'s existing API:

- `prim_changed(prim)` → Returns True for **time-independent** changes only
- `prim_changed(prim, last_time_code=X, current_time_code=Y)` → Returns True for **any** change

This clean separation makes the temporal logic straightforward.

## Operator Implementation Example

### NanoVDBVolume (supports temporal with tick)

**File**: `python/index_volume.py`

```python
@operator(supports_temporal=True, tick_on_time_change=True)
class NanoVDBVolume(VolumeBase):
    async def exec(self, prim, device, context: ExecutionContext):
        """Full execution - builds volume structure and loads initial fields."""
        return await super().exec(prim, device, context)

    # Inherits on_time_changed() from VolumeBase:
    async def on_time_changed(self, prim, device, context: ExecutionContext):
        """
        Called on every time change to update DataLoader time parameters.
        Triggers IndeX ComputeTask to refresh field data.
        """
        with viz_utils.edit_context(prim):
            if loader := prim.GetChild("Material").GetChild("DataLoader"):
                s_loader = UsdShade.Shader(loader)
                s_loader.CreateInput("params_time_code", ...).Set(str(context.timecode))
                s_loader.CreateInput("params_next_time_code", ...).Set(str(context.next_time_code))

    def get_nvindex_type(self) -> str:
        return "vdb"

    def setup_importer(self, importer, volume_prim, source_dataset, enable_field_interpolation):
        # If enable_field_interpolation=True, allocate storage for TWO timesteps
        nb_attributes = len(source_dataset.get_field_names())
        if enable_field_interpolation:
            nb_attributes *= 2  # Double the attributes for current + next timestep

        importer.SetCustomDataByKey("nvindex.importerSettings", {
            "importer": "nv::index::IDistributed_sparse_volume_importer",
            "nb_attributes": nb_attributes,
            # ... other settings
        })
```

### IrregularVolume (supports temporal with tick)

```python
@operator(supports_temporal=True, tick_on_time_change=True)
class IrregularVolume(VolumeBase):
    async def exec(self, prim, device, context: ExecutionContext):
        """Always runs on CPU for irregular volumes."""
        if device != "cpu":
            logger.warning(f"Executing IrregularVolume operator on CPU")
            device = "cpu"
        return await super().exec(prim, device, context)

    # Inherits on_time_changed() from VolumeBase

    def get_nvindex_type(self) -> str:
        return "irregular_volume"

    def setup_importer(self, importer, volume_prim, source_dataset, enable_field_interpolation):
        # Pass enable_field_interpolation to importer for attribute allocation
        importer.SetCustomDataByKey("nvindex.importerSettings", {
            "importer": "nv::omni::cae::index.PythonImporter",
            "params_enable_field_interpolation": enable_field_interpolation,
            # ... other settings
        })

class IndeXImporter_irregular_volume(IndeXBase):
    def create_subset(self, bbox, factory):
        # ... mesh setup ...

        # Allocate attributes for current timestep
        dav_index_utils.allocate_attribute_storage(
            source_dataset, subset, instance_names, start_index=0
        )

        # If temporal interpolation enabled, allocate for next timestep too
        if self.params.get("enable_field_interpolation", False):
            num_fields = len(instance_names)
            dav_index_utils.allocate_attribute_storage(
                source_dataset, subset, instance_names, start_index=num_fields
            )

        return subset

class IndeXComputeTask_irregular_volume(IndeXBase):
    def launch_compute(self, dst_buffer):
        # Collect datasets (current + next if available)
        datasets = [self.get_source_dataset()]
        if self.has_next_time_code():
            datasets.append(self.get_next_source_dataset())

        subset = dst_buffer.get_distributed_data_subset()
        instance_names = usd_utils.get_instances(self.prim, "CaeVizFieldSelectionAPI")

        # Fill attributes with offset for each timestep
        for ds_idx, dataset in enumerate(datasets):
            start_index = ds_idx * len(instance_names)
            dav_index_utils.fill_attribute_storage(
                dataset, subset, instance_names, start_index
            )
```

## Operator Migration

All operators updated to use `ExecutionContext` (required parameter, not optional):

- `streamlines.py`: `Streamlines` - Updated signature, no temporal support
- `points.py`: `Points`, `Glyphs` - Updated signature, no temporal support
- `flow_emitters.py`: `FlowNanoVDBEmitter` - Updated signature, no temporal support
- `index_volume.py`: `IrregularVolume`, `NanoVDBVolume` - Full temporal support with `tick_on_time_change=True`

**Key changes:**
- Removed `timeCode` parameter from `exec()` signatures
- Added required `context: ExecutionContext` parameter to `exec()`
- Operators access timecode via `context.timecode` (snapped) or `context.raw_timecode`
- Added `on_time_changed()` method to `VolumeBase` for temporal tick handling

## Benefits

1. **Performance**: Temporal updates skip expensive rebuilds (mesh/volume structure, importer setup)
2. **Clean API**: Operators explicitly declare temporal support and tick behavior
3. **Automatic tracking**: Controller handles all state management transparently
4. **Safe fallback**: Structural changes always trigger full rebuild
5. **Simple adoption**: Non-temporal operators just pass through context
6. **Field interpolation**: Support for rendering between time samples with dual-timestep attributes
7. **Generator pattern**: Flexible execution flow supporting multiple contexts per frame
8. **Separation of concerns**: `exec()` for full/temporal builds, `on_time_changed()` for lightweight updates

## Future Enhancements (Not Implemented)

These were discussed but deferred:

1. **Memory management**: Bound `executed_timecodes` set size to prevent unbounded growth
2. **timeWindow attribute**: Schema has it, but usage TBD (could limit temporal cache size)
3. **Performance stats**: Track and report execution metrics (rebuild vs temporal update counts)
4. **Auto-clearing**: Clear state on large time jumps or memory pressure
5. **Per-field temporal control**: Allow different fields to have different temporal behaviors

## Testing Recommendations

1. **Basic temporal flow**:
   - Apply `CaeVizOperatorTemporalAPI` to NanoVDBVolume or IrregularVolume prim
   - Set `enableFieldInterpolation = true`
   - Scrub timeline - should see temporal execution logs
   - Verify only one execution per unique timecode
   - Check that `on_time_changed()` is called on every timeline movement

2. **Structural change resets**:
   - Change a property (e.g., field selection)
   - Verify temporal state is cleared (see "STRUCTURAL_CHANGE" in logs)
   - Scrub timeline again - should re-execute all timecodes

3. **Toggle interpolation**:
   - Toggle `enableFieldInterpolation` off/on
   - Verify temporal state is cleared (treated as structural change)
   - With interpolation on: Verify two sets of attributes are allocated in importer
   - With interpolation off: Verify only one set of attributes

4. **Field interpolation**:
   - Enable `enableFieldInterpolation`
   - Scrub between time samples
   - Verify `params_next_time_code` is set in DataLoader
   - Verify ComputeTask fetches and fills data for both current and next timestep

5. **Tick behavior**:
   - With `enableFieldInterpolation=True`: Verify tick fires on every raw timeline movement
   - With `enableFieldInterpolation=False`: Verify tick fires only when snapped timecode changes
   - Verify `on_time_changed()` updates DataLoader parameters correctly

6. **Non-temporal operators**:
   - Test operators without temporal support (Streamlines, Points, Glyphs)
   - Verify they still work correctly with ExecutionContext parameter
   - Verify they execute on every time change

## Schema Reference

```usda
class "CaeVizOperatorTemporalAPI" (
    inherits = </APISchemaBase>
    customData = {
        token apiSchemaType = "singleApply"
    }
)
{
    bool cae:viz:temporal:enableFieldInterpolation = false (
        displayName = "Enable Field Interpolation"
    )

    int cae:viz:temporal:timeWindow = 2 (
        displayName = "Time Window"
        doc = "Specifies the time sample window (future use)"
    )
}
```

## Implementation Files

- `python/execution_context.py` - New: ExecutionContext and ExecutionReason
- `python/operator.py` - Modified: Added supports_temporal and tick_on_time_change parameters
- `python/controller.py` - Modified: Added _build_execution_context() generator, temporal tracking, execute per context
- `python/index_volume.py` - Modified: Added temporal support to both IrregularVolume and NanoVDBVolume, moved on_time_changed() to VolumeBase
- `python/streamlines.py` - Modified: Updated exec() signature to use ExecutionContext
- `python/points.py` - Modified: Updated exec() signature to use ExecutionContext
- `python/flow_emitters.py` - Modified: Updated exec() signature to use ExecutionContext
- `omni.cae.dav/python/index_utils.py` - Modified: Added start_index parameter to allocate_attribute_storage()

## Key Implementation Details

### Timecode Handling

All timecode snapping happens in `_build_execution_context()` via `get_bracketing_time_codes()`:
- `context.timecode`: Snapped to actual time sample (current)
- `context.raw_timecode`: Original timeline position
- `context.next_time_code`: Snapped to next time sample (if available, for interpolation)

Operators should always use `context.timecode` for data loading and should pass `context.next_time_code` to systems that need interpolation support.

### Attribute Storage for Interpolation

When `enableFieldInterpolation=True`:
1. **Importer** (`create_subset`): Allocates attributes for 2 timesteps
   - Indices 0..N-1: Current timestep fields
   - Indices N..2N-1: Next timestep fields
2. **ComputeTask** (`launch_compute`): Fills both sets with actual data
   - Fetches current dataset → fills indices 0..N-1
   - Fetches next dataset → fills indices N..2N-1
3. **Shader/Renderer**: Can interpolate between the two timesteps based on `raw_timecode`

### Error Handling

Each execution context is handled independently:
- Errors in one context don't prevent subsequent contexts from executing
- Cache is updated after each context (success or failure) to maintain consistency
- Last successful execution determines visibility in the scene
