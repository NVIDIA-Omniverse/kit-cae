# AI Agent Guide: Temporal Execution System

**Purpose**: This document trains AI agents to understand and modify the temporal execution system for CAE visualization operators.

**Last Updated**: January 2026

---

## System Overview

The temporal execution system optimizes operator performance during timeline playback by:
1. Tracking which timecodes have been executed since the last structural change
2. Skipping expensive rebuilds when only time changes
3. Supporting field interpolation between time samples for smooth rendering
4. Providing lightweight tick hooks for minimal per-frame updates

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Controller._execute_operator()                                  │
│  └─> _build_execution_context() [GENERATOR]                     │
│       ├─> Yields: ExecutionContext(STRUCTURAL_CHANGE)           │
│       ├─> Yields: ExecutionContext(TEMPORAL_UPDATE) [current]   │
│       ├─> Yields: ExecutionContext(TEMPORAL_UPDATE) [next]      │
│       └─> Yields: ExecutionContext(TEMPORAL_TICK)               │
└─────────────────────────────────────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────────┐
│ For each ExecutionContext yielded:                              │
│  ├─> If reason is exec-worthy: Call operator.exec(context)      │
│  ├─> If reason is TEMPORAL_TICK: Call operator.on_time_changed()│
│  └─> Update cache after each context (success or failure)       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Data Structures

### ExecutionContext (python/execution_context.py)

```python
@dataclass
class ExecutionContext:
    reason: ExecutionReason              # WHY we're executing
    timecode: Usd.TimeCode               # SNAPPED to actual time sample (current)
    raw_timecode: Usd.TimeCode           # Original timeline position
    next_time_code: Optional[Usd.TimeCode]  # SNAPPED to next sample (for interpolation)
    device: str                          # "cpu" or "cuda:0", etc.
```

**Critical Rules:**
- `timecode` and `next_time_code` are ALWAYS snapped via `get_bracketing_time_codes()`
- `raw_timecode` is the original timeline position (not snapped)
- Operators should use `timecode` for data loading, `next_time_code` for interpolation
- Never add `is_temporal_mode` - that was removed

### ExecutionReason (python/execution_context.py)

```python
class ExecutionReason(Enum):
    INITIAL = "initial"              # First execution ever
    STRUCTURAL_CHANGE = "structural" # Non-temporal change (properties, inputs, etc.)
    TEMPORAL_UPDATE = "temporal"     # Time changed, first time at this timecode
    TEMPORAL_TICK = "temporal_tick"  # Lightweight update (may or may not have exec'd)
```

### Execution Cache (python/controller.py)

```python
_last_execution_cache[prim_path] = {
    "timecode": Usd.TimeCode,           # Last snapped timecode
    "raw_timecode": Usd.TimeCode,       # Last raw timecode
    "device": str,
    "operator_class": Type,
    "temporal_state": {
        "executed_timecodes": set()     # Set of float values (snapped)
    } or None
}
```

**Critical Rules:**
- `temporal_state` is always initialized (even for non-temporal operators) on first execution
- `temporal_state` is cleared on ANY structural change (including API toggle)
- `executed_timecodes` stores SNAPPED timecode values as floats
- **For temporal operators**: Accumulates all executed timecodes (multi-value set)
- **For non-temporal operators**: Cleared on each new timecode (single-value cache via `.clear()`)
- Cache is updated after EACH context execution (in finally block)

## Critical Implementation Patterns

### Pattern 1: Generator for Multiple Contexts

`_build_execution_context()` is a **GENERATOR** that can yield 0-N contexts:

```python
def _build_execution_context(...) -> Iterator[ExecutionContext]:
    # May yield: exec(current), exec(next), tick
    # May yield: exec(structural)
    # May yield: tick only

    if structural_change:
        yield ExecutionContext(STRUCTURAL_CHANGE, ...)
    elif supports_temporal:
        # Temporal operators: only execute on FIRST-TIME-SEEN timecodes
        if current_timecode not in executed_timecodes:
            yield ExecutionContext(TEMPORAL_UPDATE, timecode=current, ...)
        if interpolation_enabled and next_timecode not in executed_timecodes:
            yield ExecutionContext(TEMPORAL_UPDATE, timecode=next, ...)
    else:
        # Non-temporal operators: execute on EVERY timecode CHANGE
        if current_timecode != last_timecode:
            executed_timecodes.clear()  # Keep only most recent
            executed_timecodes.add(current_timecode)
            yield ExecutionContext(TEMPORAL_UPDATE, ...)

    if tick_on_time_change and should_tick:
        yield ExecutionContext(TEMPORAL_TICK, ...)
```

**Critical Rules:**
- ALWAYS yield exec contexts before tick contexts
- Each yielded context is handled independently (errors don't stop subsequent contexts)
- Use `elif` between structural and temporal branches (mutually exclusive on same iteration)
- **Non-temporal operators execute on every timecode CHANGE** (not every frame if staying on same timecode)
- **Temporal operators skip already-seen timecodes** (optimization for scrubbing)
- Never return None - use empty generator if nothing to do

### Pattern 2: next_time_code and enableFieldInterpolation

**CRITICAL:** The `next_time_code` in `ExecutionContext` is set to `None` if `enableFieldInterpolation` is `False`:

```python
# In _build_execution_context():
enable_field_interpolation = (
    prim.HasAPI(cae_viz.OperatorTemporalAPI)
    and cae_viz.OperatorTemporalAPI(prim).GetEnableFieldInterpolationAttr().Get()
)
if not enable_field_interpolation:
    # Prevent interpolation even if next sample data is available
    next_time_code = None
```

**Why this matters:**
- **Without interpolation disabled**: Even if USD has a next time sample, operators should NOT interpolate
- **Affects ALL operators**: Including those like `Faces` that might have next sample data available
- **Single source of truth**: Controller enforces this centrally so operators don't need to check

**Critical Rule:** NEVER set `next_time_code` outside of `_build_execution_context()`. The flag check MUST happen before any yielding.

### Pattern 3: Tick Firing Logic

The logic for `TEMPORAL_TICK` depends on `enableFieldInterpolation`:

```python
# WITH interpolation:
if enable_field_interpolation:
    needs_tick = executed or (raw_timecode changed)
    # Fires on EVERY raw timeline movement

# WITHOUT interpolation:
else:
    needs_tick = (snapped_timecode changed)
    # Fires only when landing on a DIFFERENT snapped timecode
```

**Why this matters:**
- With interpolation: User scrubs between samples → tick fires every frame → shader interpolates
- Without interpolation: User scrubs between samples → tick only when crossing sample boundaries

**Critical Rule:** Always check `last_execution["timecode"]` for snapped comparison, `last_execution["raw_timecode"]` for raw comparison

### Pattern 4: Operator Signatures

All operator methods use `ExecutionContext` (REQUIRED parameter, not optional):

```python
# CORRECT:
async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
    timecode = context.timecode  # Use this for data loading

    # Check if interpolation is enabled
    if context.next_time_code is not None:
        # Interpolation is enabled AND there's a next sample
        # Load data for both t0 and t0+1
        pass
    else:
        # Either interpolation disabled or no next sample available
        # Load data for current time only
        pass

# WRONG (old pattern):
async def exec(self, prim: Usd.Prim, timeCode: Usd.TimeCode, device: str):
    # NO - timeCode parameter was removed
```

**Critical Rules:**
- Never add `timeCode` parameter - it was removed
- Context is REQUIRED, not Optional
- Access timecode via `context.timecode`, `context.raw_timecode`, or `context.next_time_code`
- **ALWAYS check `context.next_time_code is not None`** before using it - it's `None` when interpolation is disabled
- Don't check `enableFieldInterpolation` in operators - the controller already did this

### Pattern 5: Field Interpolation Attribute Allocation

When `enableFieldInterpolation=True`, allocate double attributes:

```python
# In Importer.create_subset():
instance_names = [...field names...]

# Current timestep
allocate_attribute_storage(dataset, subset, instance_names, start_index=0)

# Next timestep (if interpolation enabled)
if enable_field_interpolation:
    num_fields = len(instance_names)
    allocate_attribute_storage(dataset, subset, instance_names, start_index=num_fields)
```

**Memory layout:**
```
Attributes:  [0] [1] [2] ... [N-1]  [N] [N+1] ... [2N-1]
             └─ Current timestep ─┘  └─ Next timestep ─┘
```

**Critical Rules:**
- Always use `start_index` parameter for offsetting
- Importer allocates based on `params["enable_field_interpolation"]`
- ComputeTask fills based on `has_next_time_code()`
- Use same `dataset` twice for allocation (structure only, not data)
- **The controller ensures `context.next_time_code` is `None` when interpolation is disabled**, so operators don't need to double-check the flag

### Pattern 6: on_time_changed() Hook

```python
async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
    """
    Called when tick_on_time_change=True.
    Keep VERY fast - no expensive operations!
    """
    # Update DataLoader time parameters to trigger rendering updates
    with viz_utils.edit_context(prim):
        if loader := prim.GetChild("Material").GetChild("DataLoader"):
            s_loader = UsdShade.Shader(loader)
            s_loader.CreateInput("params_time_code", ...).Set(str(context.timecode))
            # context.next_time_code may be None if interpolation is disabled
            if context.next_time_code is not None:
                s_loader.CreateInput("params_next_time_code", ...).Set(str(context.next_time_code))
            else:
                # Clear or use sentinel value
                s_loader.CreateInput("params_next_time_code", ...).Set("")
```

**Critical Rules:**
- Only called if `@operator(tick_on_time_change=True)`
- Called AFTER any exec() contexts
- Called even if exec() was skipped
- Must be async (return awaitable)
- Lives on `VolumeBase`, not individual volume classes
- **Always check `context.next_time_code is not None`** before using it

## Common Modification Scenarios

### Scenario 1: Add New Operator with Temporal Support

```python
@operator(supports_temporal=True, tick_on_time_change=True)
class MyOperator:
    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        if context.is_full_rebuild_needed():
            # Full rebuild: STRUCTURAL_CHANGE or INITIAL
            await self._full_build(prim, device, context)
        else:
            # Temporal update: Just update time-varying data
            await self._update_fields(prim, device, context)

    async def on_time_changed(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        # Minimal update - just refresh display parameters
        pass
```

### Scenario 2: Add Field to ExecutionContext

**DO:**
1. Add field to `@dataclass` in `execution_context.py`
2. Update docstring and type hints
3. Update ALL instantiations in `controller.py` (search for `ExecutionContext(`)
4. Update examples in this guide and `TEMPORAL_EXECUTION_DESIGN.md`

**DON'T:**
- Add computed fields that could be methods (prefer `def is_foo()` over `foo: bool`)
- Add fields that duplicate existing info (like `is_temporal_mode` - use `reason` instead)

### Scenario 3: Modify Tick Logic

**Location:** `controller.py` → `_build_execution_context()` → tick section

**Key considerations:**
- Must check `tick_on_time_change` flag from operator class
- Must distinguish interpolation on/off behavior
- Must access correct cached timecode (`last_execution["timecode"]` vs `last_execution["raw_timecode"]`)
- Always yield, never return

### Scenario 4: Add Interpolation Support to New Volume Type

**Steps:**
1. Update `setup_importer()` to accept `enable_field_interpolation: bool` parameter
2. Pass flag through to importer via `params_enable_field_interpolation`
3. In Importer `create_subset()`: Check `self.params.get("enable_field_interpolation")` and allocate 2x attributes
4. In ComputeTask `launch_compute()`: Check `has_next_time_code()` and fill 2x datasets
5. Ensure `exec()` sets both `params_time_code` and `params_next_time_code` in DataLoader

## Critical Invariants (NEVER VIOLATE)

1. **Timecode snapping is centralized**: ALL snapping happens in `_build_execution_context()` via `get_bracketing_time_codes()`
   - Never snap timecodes in operators
   - Never call `get_bracketing_time_codes()` outside controller

2. **Interpolation control is centralized**: The `next_time_code` is set to `None` in `_build_execution_context()` if `enableFieldInterpolation` is `False`
   - This happens BEFORE any context yielding
   - Operators should check `context.next_time_code is not None` rather than checking the API flag
   - This prevents accidental interpolation when the feature is disabled but next sample data exists

3. **Generator pattern**: `_build_execution_context()` must ALWAYS be a generator
   - Never convert to returning a single context or list
   - Use `yield` not `return`

4. **Cache updates are in finally**: Cache updates happen in finally block of loop
   - Never skip cache update
   - Update happens whether execution succeeds or fails

5. **Temporal state lifecycle**:
   - Created on first execution (for both temporal AND non-temporal operators)
   - Cleared on ANY structural change (including API toggle)
   - For temporal operators: Accumulates all executed timecodes
   - For non-temporal operators: Cleared on each new timecode (via `.clear()`)
   - Never manually cleared elsewhere

6. **Attribute indexing for interpolation**:
   - Current timestep: indices `[0, N-1]`
   - Next timestep: indices `[N, 2N-1]`
   - Never interleave or use different offsets

7. **ExecutionContext is immutable**: It's a dataclass with no setters
   - Never modify fields after creation
   - Create new context if values need to change

8. **Tick always fires after exec**: In the generator, yield exec contexts first, then tick
   - Never yield tick before exec
   - This ensures rendering updates happen after data updates

## Common Pitfalls

### Pitfall 1: Using `set(timecode.GetValue())`

**WRONG:**
```python
temporal_state = {"executed_timecodes": set(current_time_code.GetValue())}
```

**RIGHT:**
```python
temporal_state = {"executed_timecodes": {current_time_code.GetValue()}}
```

**Why:** `set(float)` fails - use set literal `{value}` instead

### Pitfall 2: Checking Wrong Timecode for Tick

**WRONG:**
```python
# Without interpolation
needs_tick = executed  # NO - always true when we exec
```

**RIGHT:**
```python
# Without interpolation
needs_tick = current_time_code.GetValue() != last_execution["timecode"].GetValue()
```

**Why:** We need to detect *timecode change*, not *execution*. These differ in edge cases (cache corruption, etc.)

### Pitfall 3: Forgetting start_index

**WRONG:**
```python
allocate_attribute_storage(dataset, subset, instance_names)  # Defaults to start_index=0
allocate_attribute_storage(dataset, subset, instance_names)  # OVERWRITES index 0!
```

**RIGHT:**
```python
allocate_attribute_storage(dataset, subset, instance_names, start_index=0)
allocate_attribute_storage(dataset, subset, instance_names, start_index=len(instance_names))
```

### Pitfall 4: Caching Next Dataset in exec()

**WRONG:**
```python
async def exec(...):
    source_dataset = await self.get_source(prim, context.timecode, device)
    cache[...][str(context.timecode)] = source_dataset

    # DON'T fetch and cache next here!
    if context.next_time_code:
        next_dataset = await self.get_source(prim, context.next_time_code, device)
        cache[...][str(context.next_time_code)] = next_dataset
```

**RIGHT:**
```python
# Let ComputeTask fetch next dataset when it actually needs it
def launch_compute():
    datasets = [self.get_source_dataset()]  # From cache
    if self.has_next_time_code():
        datasets.append(self.get_next_source_dataset())  # Fetches now
```

**Why:** Separation of concerns - exec() sets up structure, ComputeTask fills data

### Pitfall 5: Non-Temporal Operators Not Executing on Time Change

**WRONG:**
```python
def _build_execution_context(...):
    if structural_change:
        yield ...

    if supports_temporal:  # BUG: Non-temporal operators never execute!
        if new_timecode:
            yield ...
```

**RIGHT:**
```python
def _build_execution_context(...):
    if structural_change:
        yield ...
    elif supports_temporal:  # Note: elif
        if new_timecode:
            yield ...
    else:
        # Non-temporal: execute on every timecode CHANGE
        if current_timecode != last_timecode:
            executed_timecodes.clear()  # Single-value cache
            executed_timecodes.add(current_timecode)
            yield ...
```

**Why:** Non-temporal operators must execute on every timecode change (different behavior from temporal). The `clear()` converts the set into a single-value cache that only tracks "current timecode" without accumulating history.

### Pitfall 6: Using next_time_code Without Checking for None

**WRONG:**
```python
async def exec(self, prim, device, context):
    # Assumes next_time_code is always valid
    next_dataset = await get_dataset(context.next_time_code)  # CRASHES if None!
```

**RIGHT:**
```python
async def exec(self, prim, device, context):
    current_dataset = await get_dataset(context.timecode)

    if context.next_time_code is not None:
        next_dataset = await get_dataset(context.next_time_code)
        # Allocate for both timesteps
    else:
        # Interpolation disabled or no next sample
        # Allocate for current only
```

**Why:** The controller sets `next_time_code = None` when `enableFieldInterpolation=False`, even if USD has a next sample. This prevents accidental interpolation in operators like `Faces` that shouldn't interpolate when the feature is disabled.

### Pitfall 7: Checking enableFieldInterpolation in Operators

**WRONG:**
```python
async def exec(self, prim, device, context):
    # DON'T check the API flag - controller already did this
    if prim.HasAPI(cae_viz.OperatorTemporalAPI):
        enable_interp = cae_viz.OperatorTemporalAPI(prim).GetEnableFieldInterpolationAttr().Get()
        if enable_interp:
            # Load next dataset
            pass
```

**RIGHT:**
```python
async def exec(self, prim, device, context):
    # Controller already checked - just use next_time_code
    if context.next_time_code is not None:
        # Interpolation is enabled AND there's a next sample
        # Controller guarantees this
        pass
```

**Why:** The controller is the single source of truth for interpolation. It checks the flag and sets `next_time_code` accordingly. Operators should trust this and not re-check the flag.

### Pitfall 8: Returning None from Generator

**WRONG:**
```python
def _build_execution_context(...):
    if not should_execute:
        return None  # NO - breaks iteration
```

**RIGHT:**
```python
def _build_execution_context(...):
    if not should_execute:
        return  # Empty generator - nothing to yield
```

## Temporal vs Non-Temporal Operator Behavior

This is a **critical distinction** that determines execution frequency:

### Temporal Operators (`supports_temporal=True`)

**Behavior:**
- Execute ONCE per unique timecode (since last structural change)
- Accumulate all executed timecodes in a set
- Skip re-execution when returning to previously seen timecodes

**Use case:** Expensive operations that can cache results per timestep
- Volume rendering setup
- Field data loading
- Mesh construction from temporal data

**Cache behavior:**
```python
temporal_state["executed_timecodes"] = {0.0, 1.0, 2.0, 5.0, ...}  # Grows over time
# Scrubbing back to 1.0 → SKIP (already executed)
```

### Non-Temporal Operators (`supports_temporal=False`)

**Behavior:**
- Execute on EVERY timecode CHANGE (even if returning to previously seen frame)
- Only track the MOST RECENT timecode (single-value cache)
- Always execute when moving to a different timecode

**Use case:** Operations on non-temporal data or operations that must always update
- Static geometry operators
- Property-driven computations
- Legacy operators not designed for temporal optimization

**Cache behavior:**
```python
temporal_state["executed_timecodes"] = {2.0}  # Only current timecode
# Moving to 1.0 → executed_timecodes.clear() → {1.0} → EXECUTE
# Moving back to 2.0 → executed_timecodes.clear() → {2.0} → EXECUTE (even though we were here before)
```

**Key insight:** The `.clear()` operation ensures non-temporal operators don't accumulate history, converting the set into a "last timecode" tracker.

## File Modification Checklist

When modifying the temporal system, check:

- [ ] `execution_context.py`: Updated dataclass and enum if adding new concepts
- [ ] `controller.py`: Updated `_build_execution_context()` generator logic
- [ ] `controller.py`: Updated cache structure if adding new tracked state
- [ ] `operator.py`: Updated decorator if adding new flags
- [ ] `index_volume.py`: Updated operators if changing signatures
- [ ] `index_utils.py`: Updated attribute functions if changing allocation
- [ ] `TEMPORAL_EXECUTION_DESIGN.md`: Updated documentation
- [ ] `AI_TEMPORAL_SYSTEM_GUIDE.md`: Updated this guide

## Testing Strategy

When making changes, always test:

1. **Structural change resets temporal state**
   - Change property → verify temporal_state cleared → scrub timeline → verify re-execution

2. **Interpolation on/off**
   - Toggle `enableFieldInterpolation` → verify attribute count (1x vs 2x)
   - Verify tick behavior changes (every frame vs on samples)
   - **Verify `next_time_code` is `None` when disabled, even if next sample exists**
   - Test with temporal dataset where next samples are definitely available

3. **Multiple contexts per frame**
   - With interpolation on, verify both current and next timecodes execute (first time)
   - Verify tick always fires after exec

4. **Error recovery**
   - Cause error in first context → verify second context still executes
   - Verify cache updates even on failure

5. **Non-temporal operators**
   - Verify they execute on EVERY timecode change (not cached)
   - Verify they DON'T re-execute when staying on same timecode
   - Verify `executed_timecodes` only contains current timecode (single value)
   - Test: Move timeline 0→1→2→1→2 → should execute at each step (5 times)
   - **Verify `next_time_code` is correctly set to `None` when interpolation is disabled**

6. **Multiple operators with independent interpolation settings**
   - Create two operators on same dataset, one with interpolation enabled, one disabled
   - Verify each gets correct `next_time_code` based on their own settings
   - Verify they don't interfere with each other's execution

## Key Design Decisions (Historical Context)

### Why Generator Pattern?
- Allows yielding 0-N contexts per call (flexible)
- Enables "exec, exec, tick" flow for interpolation
- Better error isolation (handle each context separately)

### Why Separate on_time_changed()?
- `exec()` is for building/updating structure and data
- `on_time_changed()` is for lightweight per-frame updates
- Allows tick on every frame without re-executing expensive logic

### Why next_time_code in ExecutionContext?
- Needed by operators to pass to rendering system
- Enables interpolation at render time
- Single source of truth (controller computes once)

### Why Remove is_temporal_mode?
- Redundant - can check `reason != STRUCTURAL_CHANGE`
- Less state = less confusion
- ExecutionReason is more explicit

### Why Tick Logic Differs for Interpolation On/Off?
- With interpolation: Need smooth scrubbing → tick every raw frame
- Without interpolation: Jump between samples → tick only on sample change
- Matches user expectation of continuous vs discrete time

## Glossary

- **Snapped timecode**: Timecode aligned to actual time sample (via `get_bracketing_time_codes()`)
- **Raw timecode**: Original timeline position from USD
- **Bracketing timecodes**: Current and next time samples surrounding raw timecode
- **Temporal state**: Cache tracking which timecodes have been executed since structural change
- **Structural change**: Any non-temporal change (properties, inputs, API settings)
- **Temporal update**: First execution at a new timecode (no structural changes)
- **Temporal tick**: Lightweight update when timecode changes (may or may not have exec'd)
- **Field interpolation**: Rendering technique blending two timesteps for smooth playback
- **Attribute storage**: IndeX rendering engine's storage for field data per vertex/cell

## Quick Reference: Key Functions

```python
# Controller (python/controller.py)
def _build_execution_context(...) -> Iterator[ExecutionContext]
    # Yields 0-N ExecutionContext objects based on change analysis

def _execute_operator(...)
    # Iterates over contexts from _build_execution_context()
    # Calls exec() or on_time_changed() per context

# Operator (python/operator.py)
def operator(priority: int = 0, supports_temporal: bool = False, tick_on_time_change: bool = False)
    # Decorator that sets __supports_temporal__ and __tick_on_time_change__ on class

# Utils (python/usd_utils.py)
def get_bracketing_time_codes(prim: Usd.Prim, timeCode: Usd.TimeCode) -> List[Usd.TimeCode]
    # Returns [current] or [current, next] based on time samples

# Index Utils (omni.cae.dav/python/index_utils.py)
def allocate_attribute_storage(dataset, subset, field_names, start_index=0)
    # Allocates IndeX attribute storage with optional offset

def fill_attribute_storage(dataset, subset, field_names, start_index=0)
    # Fills pre-allocated IndeX attribute storage with field data
```

## Contact & Questions

If this guide is unclear or you encounter edge cases not covered:
- Refer to `TEMPORAL_EXECUTION_DESIGN.md` for high-level design
- Check git history for `controller.py` to see evolution of logic
- Search for "TEMPORAL DEBUG" logs in controller for runtime behavior

---

**Remember**: This is a well-tested, production system. When in doubt, maintain existing patterns rather than introducing new ones.
