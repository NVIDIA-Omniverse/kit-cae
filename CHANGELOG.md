# Release notes

## 0.1.0 (unreleased)

The first release. OVCAE is a standalone CAE visualization library for Linux
x86-64 and Python 3.12: it evaluates visualization recipes authored in OpenUSD
against simulation data in its native formats and publishes the resulting
geometry into an OVStage for any renderer. See
[installation](docs/installation.md) for the dependency ranges and
installation requirements.

### Fixes

- Stagger animated streamline heads with deterministic per-curve phase offsets
  generated on the compute device.

### What it does

- Evaluate Faces, Points, Glyphs, Iso Surface, Planar Slice, and Streamlines
  recipes on CPU or CUDA, publishing standard geometry onto their recipe
  prims. Glyphs offers Sphere, Cone, Arrow, and Custom prototypes with
  scientific primvars and explicit scale mapping.
- Integrate with an application-owned OVStage: asynchronous updates,
  completion polling, immutable reports with per-operator status and
  diagnostics, and commits the application controls ordinal by ordinal.
- Reuse unchanged committed products. USD sources use held sampling and keep
  the latest loaded sample per dataset and device; a `MemoryDataSource`
  publish marks only its key changed. See the
  [integration guide](docs/integration-guide.md#change-detection) for the
  checking-frequency requirement and the cases that execute conservatively.
- Preserve a hidden recipe's geometry while suspending its evaluation, and
  remove generated attributes when a recipe is disabled or removed.
- Read datasets in their native formats with `ovcae.readers.UsdDataSource`
  through the CAE OpenUSD Plugins, serve in-memory Warp SimData datasets
  with `MemoryDataSource`, or combine sources with deterministic key
  resolution. The runtime defines the Data Source contract and knows no
  dataset representation; each representation ships as a reader package.
- Extend the pipeline with registered custom operators that declare their
  inputs, read the scene through a bounded view, and query what changed. A
  custom operator carries its own settings in a codeless schema registered
  with `ovcae.register_schema_paths`; usd-core's schema registry is the one
  source of schema definitions.
- Author recipe scenes with `ovcae.authoring`: recipe builders, controls,
  field selections, edit targets, and material bindings, plus
  `ovcae.readers.import_dataset` to payload a source file onto a holder prim.
  Typed schema classes, built from the registry on first use, read and edit
  the codeless schemas with defaults, token validation, and prim-type checks.
- Resolve the bundled MDL materials and colormap textures (including
  `coolwarm` and `turbo`) through `ovcae.materials`; `ovcae.setup()`
  initializes readers, schemas, and materials in one call. Float64 primvars
  and widths are converted to float32 for rendering.
- Run an interactive application on `ovcae.app.UpdateLoop`: it owns the
  Session and the ordinal, writes recipe edits and runs an update, and writes
  presentation edits such as a camera move with no update. `ovcae.stream`
  puts the loop on a render thread and streams OVRTX frames to a browser
  over WebRTC.

### Viewer

- CAE Viewer uses a single viewport with picking, transform gizmos, undo/redo,
  frame/reset controls, and FPS/evaluation annotations. The Stage Browser and
  Operator Inspector have been removed; author visibility and recipe parameters
  in USD or notebooks. Viewport interaction no longer refreshes a complete
  hierarchy snapshot during CAE updates.

### Packages and extras

One `ovcae` wheel with the runtime at the top level and the subpackages
`ovcae.readers`, `ovcae.authoring`, `ovcae.schemas`, `ovcae.materials`,
`ovcae.app`, and `ovcae.stream`. `import ovcae` loads only the standard
library and registers nothing. usd-core is a dependency of the wheel; the
`usd` extra adds the CAE OpenUSD Plugins for `ovcae.readers`, and the
`stream` extra adds OVStream for `ovcae.stream`. OVRTX is installed
separately.

### Examples and tutorials

- The minimal OVUI example is the desktop reference workflow: it authors an
  iso-surface recipe from a CGNS file, shows the live OVRTX result in an
  OVUI viewport, moves the iso value with a slider, saves the scene, and
  reproduces the picture when the scene is reopened. Its tests cover the
  round trip headlessly.
- The CAE Viewer example opens recipe scenes, edits operator settings,
  controls visibility, drags seed transforms, renders with OVRTX, and
  captures images, on the GPU selected by `--device`.
- Tutorials 0 to 5, executable notebooks committed as Jupytext percent-format
  Python and run end to end in CI, cover the environment, the pipeline on an
  aerodynamics scene, a live browser view with controls, authoring from a
  CGNS file, feeding data from memory, and a custom operator with its own
  schema. Appendices cover the native libraries alone, a custom Data Source
  over VTK, and an operator built on VTK. Their data is checked in under
  `tutorials/data` through Git LFS.
- Guides for application developers: installation, getting started,
  integration, authoring, operators, readers, packages, examples, and
  troubleshooting.

### Compatibility notes

- OVCAE source uses Kit-CAE's NVIDIA proprietary license headers.
  `LICENSE.md`, `PRODUCT_TERMS_OMNIVERSE`, and `NOTICE.md` ship in the wheel
  and sdist. Third-party materials and dependencies retain their own terms.
- All dependencies resolve from public PyPI, including the example and CI
  environments. Warp SimData requires `>=0.2.0,<0.3`; CAE OpenUSD Plugins
  requires `>=0.1.2,<0.2` with the `usd` extra. OVRTX is installed separately
  from PyPI when rendering is needed.
- The usd-core range is `>=25.11,<=26.8`, matching the combined plug-in's
  supported runtime payloads.
- Field selections use `fieldNames` and an `input` role. Dataset selections
  retain their `target` relationship.
- Automatic material range rescaling is not supported. Supply known material
  ranges with `color_range`.
- Streamlines re-execute on every update because scene seeds carry no
  comparison identity; treat camera and other presentation edits as such in
  the application so they do not trigger updates.
