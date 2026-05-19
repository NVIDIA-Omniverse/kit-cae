# Kit-CAE API Reference

## Imports

```python
import asyncio
import omni.kit.app
import omni.renderer_capture
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.usd import get_context
from pxr import Usd, UsdGeom
```

All `omni.*` imports MUST be at top level, never inside functions.

## Import APIs

```python
from omni.cae.importer.vtk import import_to_stage      # VTK
await import_to_stage(path, prim_path)

from omni.cae.importer.cgns import import_to_stage     # CGNS
await import_to_stage(path, prim_path, time_scale=1.0, time_offset=0.0, time_source="TimeStep")

from omni.cae.importer.ensight import import_to_stage  # EnSight
await import_to_stage(path, prim_path, time_scale=1.0, time_offset=0.0)

from omni.cae.importer.npz import import_to_stage      # NumPy
await import_to_stage(path, prim_path, schema_type="SIDS Unstructured")
```

Stage paths after import → see `formats.md`.

## Stage Discovery

Always use USDRT queries (not `stage.Traverse()`). USDRT queries Fabric directly
and automatically returns derived types.

```python
import usdrt
from pxr import Tf
from omni.cae.schema import cae

fabric_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

dataset_type = Tf.Type.Find(cae.DataSet)
datasets = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(dataset_type.typeName)]

field_base_type = Tf.Type.Find(cae.FieldArray)
fields = [p.GetString() for p in fabric_stage.GetPrimsWithTypeName(field_base_type.typeName)]

# Filter out coordinate datasets
mesh_datasets = [d for d in datasets if "GridCoordinates" not in d]

# Filter to data fields (skip internal mesh arrays)
containers = ["PointData", "CellData", "FlowSolution", "Flow_Solution",
              "SolutionCellCenter", "SolutionVertex", "Variables", "NumPyArrays"]
data_fields = [f for f in fields
               if any(f"/{c}/" in f or f.endswith(f"/{c}") for c in containers)]
if not data_fields:
    data_fields = fields  # custom format fallback
```

## Visualization Commands

Pattern: `await execute_command("<Name>", ...)` then bind fields via `cae_viz.FieldSelectionAPI`.

> **⚠ Parameter naming matters:** `CreateCaeVizBoundingBox` takes `dataset_paths=` (plural, a **list**).
> `CreateCaeVizVolume` takes `dataset_path=` (singular, a **string**). Both require `prim_path=`.
> Mixing these up is a common source of `TypeError` or silent failures.

### Operator I/O reference

Every viz operator ultimately reads from a `cae.DataSet`, but the source kwarg
and the prim that owns field bindings differ. This table is the source of truth
for "what's applied to what":

| Operator | Source kwarg | Reads from | Field bindings on |
|---|---|---|---|
| `CreateCaeVizBoundingBox` | `dataset_paths=` (list) | `cae.DataSet` (bounds only) | — |
| `CreateCaeVizVolume` | `dataset_path=` | `cae.DataSet` | this prim |
| `CreateCaeVizVolumeSlice` | `volume_path=` | `cae.DataSet` (via source volume) | source volume |
| `CreateCaeVizPlanarSlice` | `dataset_path=` | `cae.DataSet` | this prim |
| `CreateCaeVizFaces` | `dataset_path=` | `cae.DataSet` | this prim |
| `CreateCaeVizPoints` | `dataset_path=` | `cae.DataSet` | this prim |
| `CreateCaeVizGlyphs` | `dataset_path=` | `cae.DataSet` | this prim |
| `CreateCaeVizStreamlines` | `dataset_path=` | `cae.DataSet` | this prim |

`CreateCaeVizVolumeSlice` is the only operator that requires another operator
(the source volume) as an intermediary — the slice has no field-binding APIs of
its own. All others author bindings directly on the prim they create.

### Bounding Box (always create for framing)
```python
await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
```

### Volume
```python
await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path,
                      type="vdb")  # or "irregular" for unstructured grids with cell topology
cae_viz.FieldSelectionAPI(stage.GetPrimAtPath(vol_path), "colors").CreateTargetRel().SetTargets([field_path])
```

**Type selection:**
- `type="irregular"` — unstructured grids with cell connectivity (e.g., NPZ with `element_connectivity`, CGNS unstructured). Uses IndeX native irregular volume renderer. **Fails on cell-less datasets** (surface-only CGNS, point clouds, …) with `ValueError: Cannot compute face summary for dataset with no cells` — fall back to `vdb`.
- `type="vdb"` — structured grids, large datasets, point clouds without topology. Voxelizes to NanoVDB. When `nb_cells <= 0`, the command auto-applies `DatasetGaussianSplattingAPI` so cell-less data still renders as a splatted cloud.

Gaussian splatting radius (point clouds / cell-less fallback): `cae_viz.DatasetGaussianSplattingAPI(viz_prim, "source").CreateRadiusFactorAttr().Set(4.0)` — try 4–8 and tune visually.

ROI: create second bounding box, scale it, wire via `cae_viz.DatasetVoxelizationAPI(vol_prim, "source").CreateRoiRel().SetTargets([roi_path])`

### Faces
```python
await execute_command("CreateCaeVizFaces", dataset_path=dataset_path, prim_path=faces_path)
cae_viz.FieldSelectionAPI(stage.GetPrimAtPath(faces_path), "colors").CreateTargetRel().SetTargets([field_path])
```

### Points
```python
await execute_command("CreateCaeVizPoints", dataset_path=dataset_path, prim_path=points_path)
viz_prim = stage.GetPrimAtPath(points_path)
cae_viz.PointsAPI(viz_prim).CreateWidthAttr().Set(0.5)
cae_viz.FieldSelectionAPI(viz_prim, "colors").CreateTargetRel().SetTargets([field_path])
```

Variable width:
```python
cae_viz.FieldMappingAPI.Apply(viz_prim, "widths")
cae_viz.FieldMappingAPI(viz_prim, "widths").CreateRangeAttr().Set((0.01, 0.3))
cae_viz.FieldMappingAPI(viz_prim, "widths").CreateDomainAttr().Set((field_min, field_max))
cae_viz.FieldSelectionAPI(viz_prim, "widths").CreateTargetRel().SetTargets([width_field_path])
```

### Glyphs
```python
await execute_command("CreateCaeVizGlyphs", dataset_path=dataset_path, prim_path=glyphs_path,
                      shape="Arrow")  # or "Cone", "Sphere"
viz_prim = stage.GetPrimAtPath(glyphs_path)
cae_viz.FieldSelectionAPI(viz_prim, "colors").CreateTargetRel().SetTargets([scalar_field])
cae_viz.FieldSelectionAPI(viz_prim, "orientations").CreateTargetRel().SetTargets([vx, vy, vz])
```

Constant scale (default 1.0):
```python
cae_viz.GlyphsAPI(viz_prim).CreateScaleAttr().Set(0.3)  # smaller glyphs when combined with volume
```

Field-driven scaling:
```python
cae_viz.FieldMappingAPI.Apply(viz_prim, "scales")
cae_viz.FieldMappingAPI(viz_prim, "scales").CreateRangeAttr().Set((0.001, 0.1))
cae_viz.FieldMappingAPI(viz_prim, "scales").CreateDomainAttr().Set((field_min, field_max))
cae_viz.FieldSelectionAPI(viz_prim, "scales").CreateTargetRel().SetTargets([scale_field])
```

### Slice

Two distinct commands — pick by what you have:

- **`CreateCaeVizVolumeSlice(volume_path=, prim_path=, shape=)`** — slices an existing volume. Bindings on the volume; slice has own colormap.
- **`CreateCaeVizPlanarSlice(dataset_path=, prim_path=, type=)`** — independent dataset-bound slice; binds/rescales like other operators. `type` ∈ `{"standard","nanovdb"}`. Material: `<prim_path>/Materials/SliceTexture` (plural), default `gist_rainbow`.

```python
# shape: "Plane" | "Bi-Plane" | "Tri-Plane" | "Sphere" | "Custom"
await execute_command("CreateCaeVizVolumeSlice", volume_path=vol_path, prim_path=slice_path, shape="Plane")
```

VolumeSlice specifics:

- Volume holds the dataset rel + `FieldSelectionAPI`/`RescaleRangeAPI`/`ConfigureXACShaderAPI` (all on `"colors"`). Slice has none of these.
- Slice colormap at `<slice_path>/Material/Colormap` copies the volume's `rgbaPoints`/`xPoints` (alphas → 1.0); `domain` starts at `(0,-1)` sentinel; `domainBoundaryMode` defaults `"clampToTransparent"`. None of those three are inherited.
- The volume's `RescaleRangeAPI("colors")` Includes rel is wired to **both** colormap domains — disable for fixed, leave enabled for auto-fit.
- Multi-plane shapes author named children: Bi-Plane → `Z_Plane`+`X_Plane`; Tri-Plane → three. Single Plane: `<slice_path>` is the mesh. All Xformable.
- Even on multi-plane shapes the colormap is a single shared prim at `<slice_path>/Material/Colormap` — **not** under each plane child. One `domain` / `rgbaPoints` / `domainBoundaryMode` stamp covers all planes.

Fixed-domain pattern (boundary mode per the volume/slice rule — see
cae-visualization SKILL.md § Boundary mode):

```python
await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=vol_path, type="vdb")
vol = stage.GetPrimAtPath(vol_path)
cae_viz.FieldSelectionAPI(vol, "colors").CreateTargetRel().SetTargets([field_path])
await execute_command("CreateCaeVizVolumeSlice", volume_path=vol_path, prim_path=slice_path, shape="Plane")
cae_viz.RescaleRangeAPI(vol, "colors").CreateRescaleModeAttr().Set("disable")  # freezes both
for cm_path, boundary in (
    (f"{vol_path}/Material/Colormap", "clampToTransparent"),  # volume rule
    (f"{slice_path}/Material/Colormap", "clampToEdge"),       # slice rule
):
    cm = stage.GetPrimAtPath(cm_path)
    cm.GetAttribute("domain").Set(Gf.Vec2f(min_val, max_val))
    cm.GetAttribute("domainBoundaryMode").Set(boundary)
```

Skip the disable + manual sets to keep auto-fit; manual `domain` writes are overwritten on the next rescale tick otherwise.

### Streamlines
```python
await execute_command("CreateCaeVizStreamlines", dataset_path=dataset_path, prim_path=viz_path, type="standard")
await execute_command("CreateCaeVizMeshPrim", prim_type="UnitSphere", prim_path=seed_path)
await execute_command("TransformPrimSRT", path=seed_path, new_scale=[0.2, 0.2, 0.2])
await wait_for_update()

viz_prim = stage.GetPrimAtPath(viz_path)
cae_viz.StreamlinesAPI(viz_prim).GetDirectionAttr().Set(cae_viz.Tokens.forward)
cae_viz.DatasetSelectionAPI(viz_prim, "seeds").GetTargetRel().SetTargets([seed_path])
cae_viz.FieldSelectionAPI(viz_prim, "velocities").GetTargetRel().SetTargets(vel_field_paths)
cae_viz.FieldSelectionAPI(viz_prim, "colors").CreateTargetRel().SetTargets([color_field])
```

### Flow (animated smoke/particles)
```python
import omni.timeline
await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
await execute_command("CreateCaeVizFlowEnvironment", prim_path=flow_env_path, layer_number=0)
await execute_command("CreateCaeVizFlowSmokeInjector",
    boundable_paths=[bbox_path], prim_path=smoke_path, layer_number=0,
    mode="sphere", simulation_prim=flow_env)
await execute_command("CreateCaeVizFlowBoundaryEmitter",
    boundable_paths=[bbox_path], prim_path=boundary_path, layer_number=0)
await execute_command("CreateCaeVizFlowDataSetEmitter",
    dataset_path=dataset_path, prim_path=emitter_path, layer_number=0, simulation_prim=flow_env)
cae_viz.FieldSelectionAPI(stage.GetPrimAtPath(emitter_path), "velocities").CreateTargetRel().SetTargets([vel_field])
omni.timeline.get_timeline_interface().play()
```

## NPZ Field Association Fix

Required after NPZ import as `"SIDS Unstructured"` (not needed for `"Point Cloud"`):

```python
for child in stage.GetPrimAtPath("/World/<name>/NumPyArrays").GetAllChildren():
    cae.FieldArray(child).CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
```

## Velocity Components

Solvers store velocity as separate scalars (VelocityX/Y/Z, Velocity_0/1/2, U/V/W).
Pass all three:
```python
cae_viz.FieldSelectionAPI(viz_prim, "velocities").GetTargetRel().SetTargets([v0, v1, v2])
```

Single vector array (N×3): pass one path.

Discovery:
```python
vel_prefixes = ["Velocity", "velocity", "U", "V"]
vel_fields = [f for f in data_fields
              if any(f.rsplit("/", 1)[-1].startswith(p) for p in vel_prefixes)]
```

## Coloring Modes

- Scalar (N,1): colors by value
- Vector (N,3): colors by magnitude
- Three scalars: interpreted as vector, colors by magnitude

## Compatibility Matrix

| Viz | ImageData | Unstructured | Structured | PolyData | CGNS | EnSight | NPZ |
|-----|-----------|-------------|------------|----------|------|---------|-----|
| Volume (vdb) | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| Volume (irregular) | — | ✓ | ✓ | — | ✓ | ✓ | ✓¹ |
| Faces | — | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Points | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Glyphs | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Streamlines | — | ✓ᵛ | ✓ᵛ | — | ✓ᵛ | ✓ᵛ | ✓ᵛ¹ |
| Slice | via vol | via vol | via vol | — | via vol | via vol | via vol |

ᵛ = requires velocity field. ¹ = requires field association fix.

## Field Statistics & Data Query

Programmatic access to the same data as UI's CAE Insights panel.

```python
from omni.cae.data import array_utils, usd_utils
from pxr import Usd

prim = stage.GetPrimAtPath(field_path)
farray = await usd_utils.get_array(prim, Usd.TimeCode.EarliestTime())

# Metadata
print(farray.dtype, farray.shape, farray.ndim)

# Ranges
ranges = array_utils.get_componentwise_ranges(farray)  # [(min, max), ...]

# Scalar statistics (ndim==1 only)
stats = array_utils.get_scalar_stats(farray, num_bins=32)
# Keys: min, max, mean, median, q1, q2, q3, q4, counts, bin_edges

# Custom histogram range
hist = array_utils.compute_histogram(farray, num_bins=32, range_min=-300.0, range_max=300.0)
```

### Via USD relationship (dataset-level)

```python
from omni.cae.data.range_utils import get_range
min_val, max_val = await get_range(dataset_prim, "Pressure")
min_mag, max_mag = await get_range(dataset_prim, ["Velocity_0", "Velocity_1", "Velocity_2"])
```

## Prim Visibility

```python
UsdGeom.Imageable(prim).MakeInvisible()
UsdGeom.Imageable(prim).MakeVisible()
```

Hide all scene lights:
```python
for prim in stage.Traverse():
    if prim.GetTypeName() in ("DistantLight", "DomeLight", "SphereLight", "RectLight"):
        UsdGeom.Imageable(prim).MakeInvisible()
```

## Common Pitfalls

- `stage.GetPrimAtPath()` never returns `None` — always returns a Prim (possibly invalid). Use `prim.IsValid()`, and wrap in `bool()` for JSON: `bool(prim and prim.IsValid())`
- First `capture_viewport_to_file` needs 120+ settle frames (shader compilation). Subsequent: 10-30.
- `Gf.Vec2f(np_min, np_max)` fails with numpy types — cast: `Gf.Vec2f(float(np_min), float(np_max))`
- `attr.Set((1, 2, 3))` on a typed vector attribute (e.g. `int3`, `float3`) fails with a type mismatch because Python tuples become `Gf.Vec3d` by default. Always wrap with the attribute's exact type: `Gf.Vec3i(...)` for integer vectors (IJK extents, counts), `Gf.Vec3f(...)` for single-precision spacings/origins.
- A `UsdShadeInput` (from `shader.GetInput("name")`) can go stale across a long `await wait_for_update(...)` if the viz controller rebuilds the material prim. If you see `UsdExpiredPrimAccessError` after the wait, don't cache the input — re-fetch each access via `stage.GetPrimAtPath(shader_path).GetAttribute(f"inputs:{name}")`.
- Authored USD state is not always rendered state. Some CAE operators generate
  renderable data through IndeX, Fabric, or USDRT. A prim can have the expected
  transform or visibility samples while the viewport does not change. If that
  happens, capture frames, test the same operator statically, and prefer updating
  the current/default attribute from a Kit timeline or update callback over
  relying on time-sampled USD attributes.

## Screenshot (Swapchain — includes UI)

```python
await frame_prims([bbox_path], zoom=0.9)
for _ in range(600):
    await app.next_update_async()
cap = omni.renderer_capture.acquire_renderer_capture_interface()
cap.capture_next_frame_swapchain(output_path)
await app.next_update_async()
```

For clean render-only output, use `capture_viewport_to_file` (see `capture-api.md`).

## Script Template

```python
import asyncio
import os

import omni.kit.app
import omni.renderer_capture
import omni.usd
from omni.cae.data.commands import execute_command
from omni.cae.importer.<format> import import_to_stage
from omni.cae.schema import cae
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.usd import get_context
from pxr import Usd, UsdGeom

async def main():
    app = omni.kit.app.get_app()
    await import_to_stage("<FILE>", "/World/<name>")
    stage = get_context().get_stage()

    # Discover paths (see Stage Discovery) or use known paths (see formats.md)
    dataset_path = "/World/<name>/..."
    field_path = "/World/<name>/..."

    bbox_path = "/World/CAE/BoundingBox"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # Create viz, bind fields (see Visualization Commands)
    # ...

    await frame_prims([bbox_path], zoom=0.9)
    for _ in range(600):
        await app.next_update_async()

    omni.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain("<OUT>.png")
    await app.next_update_async()
    print("RENDER_COMPLETE")

    app.post_quit()
    for _ in range(10):
        await app.next_update_async()
    os._exit(0)

if __name__ == "__main__":
    asyncio.ensure_future(main())
```
