# OmniCaeViz Schemas

These schemas are the "language" that Kit-CAE uses to describe CAE visualization operators inside a USD stage. They live in `source/schemas/shared/omniCaeViz/schema.usda` and are consumed at runtime by the `omni.cae.viz` extension.

---

## Background: What is a USD schema?

USD (Universal Scene Description) organises a scene as a tree of **prims** (short for *primitives*), each of which can carry typed **attributes** and **relationships**. A prim might be a mesh, a curve, a volume, a camera, or just an empty container.

An **API schema** is a reusable bundle of attributes/relationships that can be *applied* to any prim — much like a mixin or interface. When you apply `CaeVizOperatorAPI` to a prim, you are saying: *"this prim is a CAE visualization operator."* The runtime finds those prims and knows what to do with them.

There are two flavours of API schema used here:

| Flavour | What it means |
|---------|--------------|
| **Single-apply** | Can only be applied once per prim (e.g., `CaeVizFacesAPI`). |
| **Multiple-apply** | Can be applied multiple times with different *instance names*, like labelled slots (e.g., `CaeVizDatasetSelectionAPI:source` and `CaeVizDatasetSelectionAPI:seeds` on the same prim). |

The instance name is the colon-separated suffix and tells the runtime what *role* that particular application plays.

---

## Mental model

Think of a visualization operator as a **recipe card** attached to a USD prim:

```
Prim (BasisCurves)
 ├── CaeVizOperatorAPI          ← "I am an operator, please execute me"
 ├── CaeVizStreamlinesAPI       ← "I specifically compute streamlines"
 ├── DatasetSelectionAPI:source ─→ /Scene/FluidDataset    (the CFD data)
 ├── DatasetSelectionAPI:seeds  ─→ /Scene/SeedPoints       (where to start)
 ├── FieldSelectionAPI:velocities ─→ /Scene/FluidDataset/velocity
 └── FieldSelectionAPI:colors   ─→ /Scene/FluidDataset/pressure
```

The runtime (`omni.cae.viz`) walks the stage, finds every prim that has `CaeVizOperatorAPI`, reads the rest of the recipe, runs the computation, and writes the results back to the prim's geometry attributes (the curves, points, mesh, or volume that USD then renders).

---

## Schema groups

### 1. Operator lifecycle schemas

These tell the runtime *whether* and *how* to run an operator.

| Schema | Type | Purpose |
|--------|------|---------|
| `CaeVizOperatorAPI` | Single-apply | **Required on every operator prim.** Marks the prim as a CAE operator. Has an `enabled` toggle and a `device` selector (`auto`, `cpu`, `cuda:0`–`cuda:3`). |
| `CaeVizOperatorDebuggingAPI` | Single-apply | Optional add-on that activates timing logs — useful for profiling. Experimental. |
| `CaeVizOperatorDependenciesAPI` | Single-apply | Declares that other operators depend on this one, so the scheduler can order execution. Experimental. |
| `CaeVizOperatorTemporalAPI` | Single-apply | Controls time-aware execution: pin the operator to a fixed time (`useLockedTime`), or enable sub-sample **field interpolation** for smooth animation playback. |

### 2. Dataset input schemas

These select and pre-process the CAE datasets that feed into an operator.

| Schema | Type | Purpose |
|--------|------|---------|
| `CaeVizDatasetSelectionAPI` | Multiple-apply | Picks a `CaeDataSet` prim as input. The instance name is the role (`source`, `seeds`, …). |
| `CaeVizDatasetTransformingAPI` | Multiple-apply | Controls whether the dataset's world-space transform is applied before processing. |
| `CaeVizDatasetVoxelizationAPI` | Multiple-apply | Converts an unstructured (e.g., FEM/CFD) dataset into a regular **NanoVDB** voxel grid. You choose the grid resolution either by maximum voxel count (`maxResolution`) or an explicit voxel size. |
| `CaeVizDatasetGaussianSplattingAPI` | Multiple-apply | Before voxelization, splats point/cell data using Gaussian kernels. Useful for sparse data like particle clouds. |
| `CaeVizDatasetTemporalTraitsAPI` | Multiple-apply | Annotates a dataset with knowledge about its temporal behaviour (`static` / `varying` / `undefined` for both topology and geometry). The runtime uses this to skip unnecessary recomputation between frames. |

### 3. Field selection and processing schemas

CAE datasets contain many **fields** (pressure, velocity, temperature, …). These schemas pick which fields to use and how to interpret them.

| Schema | Type | Purpose |
|--------|------|---------|
| `CaeVizFieldSelectionAPI` | Multiple-apply | Points at one or more field prims in the dataset. The instance name is the role (`velocities`, `colors`, `widths`, …). The `mode` attribute controls whether to pass values unchanged, compute the **vector magnitude**, or extract a single **component**. |
| `CaeVizFieldMappingAPI` | Multiple-apply | Remaps the field's numeric range to a desired output range. If the domain is left invalid, it is auto-computed from the data minimum/maximum. |
| `CaeVizFieldThresholdingAPI` | Multiple-apply | Masks out values outside a specified range, filling them with a configurable background value. Useful for isolating regions of interest. |

### 4. Algorithm schemas

Each of these corresponds to a specific visualization technique. They are applied to a standard USD geometry prim (mesh, curves, points, volume) and the runtime fills that prim with the computed geometry.

| Schema | Applied to | What it produces |
|--------|-----------|-----------------|
| `CaeVizFacesAPI` | `UsdGeomMesh` | Extracts the **surface faces** of a volumetric dataset (e.g., the outer skin of a CFD mesh, or all internal faces). |
| `CaeVizStreamlinesAPI` | `UsdGeomBasisCurves` | Traces **streamlines** (or pathlines) through a velocity field seeded from a seed dataset. Exposes integration parameters: step sizes, direction, velocity threshold, tolerance. |
| `CaeVizPointsAPI` | `UsdGeomPoints` | Renders the **point cloud** of a dataset, optionally limited to points actually used by cells. |
| `CaeVizGlyphsAPI` | `UsdGeomPointInstancer` | Places oriented **glyphs** (arrows, spheres, …) at each point. Orientation can be expressed as Euler angles or quaternions. |
| `CaeVizIndeXVolumeAPI` | `UsdVolVolume` | Marks the volume for rendering with **NVIDIA IndeX** — a high-quality in-situ volume renderer. |

### 5. Automation / wiring schemas

These schemas are plumbing: they wire computed data (field ranges, voxel sizes) back into other USD attributes automatically, so the rest of the scene stays consistent without manual updates.

| Schema | Purpose |
|--------|---------|
| `CaeVizRescaleRangeAPI` | After execution, updates target properties (e.g., a colormap range slider) to match the actual field min/max. Supports `clamp` (don't go beyond existing range) and `grow` (expand if data exceeds range) modes. |
| `CaeVizConfigureXACShaderAPI` | Pushes voxel size, field type, attribute indices, and time codes into an XAC volume shader so it stays in sync with the operator output. |
| `CaeVizConfigureFlowEnvironmentAPI` | Sets the particle density cell size of an NVIDIA Flow emitter to match the voxel grid produced by the operator. |

---

## End-to-end example: surface extraction

The following USD fragment shows a complete surface extraction operator applied to a mesh prim:

```usda
def Mesh "SurfaceMesh" (
    prepend apiSchemas = [
        "CaeVizOperatorAPI",
        "CaeVizFacesAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizFieldSelectionAPI:colors",
    ]
)
{
    # Operator lifecycle
    bool cae:viz:operator:enabled = true
    token cae:viz:operator:device = "auto"

    # Which dataset to extract faces from
    rel cae:viz:dataset_selection:source:target = </Scene/FluidMesh>

    # Which field to use as the face colour
    rel cae:viz:field_selection:colors:target = </Scene/FluidMesh/pressure>

    # Extract only the outer skin of the mesh
    uniform bool cae:viz:faces:externalOnly = true
}
```

At runtime the `omni.cae.viz` extension finds this prim, reads the relationships and attributes, runs the face-extraction algorithm on `FluidMesh`, and fills the mesh geometry (points, face counts, indices) with the result. The `pressure` field values are written as a `primvar:colors` attribute, which the renderer picks up for shading.

---

## End-to-end example: streamlines with field colouring

```usda
def BasisCurves "Streamlines" (
    prepend apiSchemas = [
        "CaeVizOperatorAPI",
        "CaeVizStreamlinesAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizDatasetSelectionAPI:seeds",
        "CaeVizDatasetVoxelizationAPI:source",
        "CaeVizFieldSelectionAPI:velocities",
        "CaeVizFieldSelectionAPI:colors",
        "CaeVizFieldMappingAPI:colors",
    ]
)
{
    bool cae:viz:operator:enabled = true

    # The CFD volume mesh
    rel cae:viz:dataset_selection:source:target = </Scene/FluidVolume>
    # Seed points (e.g., a rake of points at the inlet)
    rel cae:viz:dataset_selection:seeds:target = </Scene/InletRake>

    # Voxelize the source dataset into a 256-cell grid for integration
    token cae:viz:dataset_voxelization:source:voxelSizeMode = "maxResolution"
    int   cae:viz:dataset_voxelization:source:maxResolution = 256

    # Use the velocity field for integration
    rel cae:viz:field_selection:velocities:target = </Scene/FluidVolume/velocity>

    # Colour by pressure, mapped to [0, 1]
    rel cae:viz:field_selection:colors:target = </Scene/FluidVolume/pressure>
    float2 cae:viz:field_mapping:colors:domain = (0.0, 100000.0)   # Pa
    float2 cae:viz:field_mapping:colors:range  = (0.0, 1.0)

    # Integration parameters
    uniform float cae:viz:streamlines:initialStepSize = 0.1
    uniform float cae:viz:streamlines:maxStepSize     = 0.5
    uniform int   cae:viz:streamlines:maxSteps        = 300
    uniform token cae:viz:streamlines:direction       = "forward"
}
```

---

## How schemas relate to each other

```
CaeVizOperatorAPI              ← required on every operator prim
 │
 ├── CaeVizOperatorTemporalAPI         (optional: time control)
 ├── CaeVizOperatorDebuggingAPI        (optional: profiling)
 │
 ├── CaeVizDatasetSelectionAPI:*       (1+ dataset inputs)
 │    ├── CaeVizDatasetVoxelizationAPI:*   (optional: convert to voxel grid)
 │    ├── CaeVizDatasetGaussianSplattingAPI:*  (optional: Gaussian splat first)
 │    ├── CaeVizDatasetTransformingAPI:*   (optional: apply world transform)
 │    └── CaeVizDatasetTemporalTraitsAPI:* (optional: hint about time behaviour)
 │
 ├── CaeVizFieldSelectionAPI:*         (0+ field inputs)
 │    ├── CaeVizFieldMappingAPI:*          (optional: remap numeric range)
 │    └── CaeVizFieldThresholdingAPI:*     (optional: mask by value)
 │
 ├── One algorithm schema:
 │    CaeVizFacesAPI | CaeVizStreamlinesAPI | CaeVizPointsAPI
 │    CaeVizGlyphsAPI | CaeVizIndeXVolumeAPI
 │
 └── Automation schemas (optional):
      CaeVizRescaleRangeAPI
      CaeVizConfigureXACShaderAPI
      CaeVizConfigureFlowEnvironmentAPI
```

---

## Key design choices

**Why API schemas instead of typed prims?**
API schemas compose cleanly. A single prim can simultaneously be a USD `BasisCurves` (so the renderer knows how to draw it) *and* a CAE streamline operator (so our runtime knows how to fill it). There is no need for a parallel hierarchy of CAE-specific prim types.

**Why multiple-apply for datasets and fields?**
An operator often needs more than one input. Streamlines need a source volume *and* a seed dataset. The instance name (`source`, `seeds`) acts as a named port, making the intent explicit and letting the operator code look up each input by its role.

**Why relationships instead of string paths?**
USD relationships (`rel`) are first-class references that the stage tracks. They survive renames, composition arcs, and referencing. A plain string attribute containing a path would break whenever a prim moves.

**Why write results back to the prim's geometry?**
The output is regular USD geometry (`BasisCurves`, `Mesh`, `Points`, `Volume`). Any USD-aware renderer — NVIDIA Omniverse RTX, Hydra, offline renderers — can display it without knowing anything about Kit-CAE. The schemas are purely an *authoring* convention; at render time the prim looks like ordinary geometry.
