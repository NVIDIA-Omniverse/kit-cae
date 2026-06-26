# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.1.1]

### Changed

- **Kit SDK 110.1.2 support**: Added Kit SDK 110.1.2 to `select_kit_version` and promoted it to the
  default Kit version. The release includes refreshed 110.1.2 app, streaming, and Kit-CAE templates,
  plus the matching Packman/repo tooling updates needed for the newer Kit SDK.

### Fixed

- **Warp 1.13 compatibility** (`omni.cae.data`): Updated CAE data array helpers to avoid removed
  `warp.context.Device`, `warp.types.DType`, and `warp.types.vector_types` APIs, allowing
  `omni.cae.kit` and the bundled CAE extensions to start with `omni.warp.core` 1.13.
- **Bundled DAV runtime for Warp 1.13** (`omni.cae.dav_libs`): Updated the bundled DAV snapshot and
  submodule branch to the Kit-CAE v2 line used for Warp 1.13-compatible DAV code.

## [2.1.0]

### Added

- **Cell-centered fields on CGNS NGON_n sections** (`omni.cae.file_format.cgns` 1.2.0, `omni.cae.dav` 1.2.0):
  The CGNS reader now creates `field:*` relationships for cell-centered fields on NGON_n sections, and
  `CaeSidsUnstructuredGetField` remaps each NGON face to a referencing NFACE_n volume cell so face-based
  visualization can color by volume cell data.
- **ROI-based cell subsetting for operators** (`omni.cae.schema` 1.6.0, `omni.cae.viz` 1.5.1,
  `omni.cae.context_menu` 2.4.0): New `CaeVizDatasetSubsetAPI` multi-apply schema restricts an
  operator's input dataset to cells inside an ROI prim's axis-aligned bounds. The schema exposes a
  `roi` relationship, a `mode` selector (`all`/`any`/`centroid` — how cell vertices must relate to
  the box), and an `inflateBounds` percentage. `get_input_dataset` runs the subset step before
  voxelization via the new `cell_in_box` operator and `cell_subset` data model in DAV. The API is
  auto-applied on the `source` instance of newly created non-voxelized operators (Points, Faces,
  Glyphs — only when the dataset has cells; Streamlines and PlanarSlice in standard mode; Volume in
  irregular mode), and is also available as an **Add API > CAE > Dataset Subset** context menu entry.
- **`omni.cae.startup` extension**: New extension that reads an optional `exts."omni.cae.startup".usdFile`
  setting and, when set, waits for the renderer to deliver its first frame before closing the empty startup
  stage and opening the specified USD file.
- **Automatic colormap LUT publishing** (`omni.cae.viz`): The IndeX volume renderer consumes `Colormap` prims
  directly, but MDL shaders used for surfaces and streamlines require a 1-D texture asset for their LUT input.
  The new `ColormapTextureManager` service bridges this gap — it monitors every `Colormap` prim that has
  `CaeVizColormapTextureAPI` applied and publishes each one as a `dynamic://` texture. The URL is derived from
  a stable `identifier` attribute authored on the prim at creation time
  (e.g. `dynamic://cae_colormap_<uuid>`), so it survives prim relocation via USD composition. Textures are
  regenerated automatically whenever the prim's control points change, keeping IndeX and MDL shaders in sync
  with no manual baking step. See the [Reference Guide](docs/Misc.md#colormap-textures) for usage details.

- **PlanarSlice operator**: New `omni.cae.viz` operator that extracts a texture-mapped planar slice from any CAE
  dataset using `dav.operators.probe` — no IndeX license required.  Supports free-moving, single-axis,
  dual-axis, and tri-axis (`xyz`) modes with configurable texture resolution and colormap.
- **`CaeVizPlanarSliceAPI` schema**: New USD single-apply API schema (in `omni.cae.schema`) capturing
  `textureResolution` and `mode` attributes for planar slice prims.
- **Planar slice context menu entry**: Right-clicking a CAE dataset now offers a *Planar Slice* option in the
  Operators submenu (`omni.cae.context_menu`), with a dialog to choose `standard` or `nanovdb` execution.
- **Copy LUT Texture URL context menu entry** (`omni.cae.context_menu`): Right-clicking a `Colormap` prim now
  shows a *Copy LUT Texture URL* option that copies the `dynamic://` texture URL for that prim to the
  clipboard, making it easy to wire up MDL shader LUT inputs without constructing the path by hand.
- **Agent skills for Kit-CAE** (`skills/`): Bundled Claude Code, Codex, OpenClaw, and Cursor-compatible
  skills that let coding agents drive Kit-CAE programmatically — `cae-core` (shared foundation),
  `cae-visualization` (data import + viz operators), `cae-capture` (clean PNG/EXR/MP4 output), and
  `cae-streaming` (run Kit-CAE as a WebRTC-streamed app controllable via data-channel messages).
  Point your agent's skill search path at `<kit-cae-dir>/skills/` to use them — see the README's
  *Agent Skills* section for a copy-paste onboarding prompt.

### Changed

- **Schema extension packaging and loading** (`omni.cae.schema`): USD schema artifacts now install under a
  dedicated `usd/` subtree, with plugin resources and native libraries in `usd/plugin/<SchemaName>` and
  generated Python modules exposed through a shared `usd/python/` tree. The schema extension discovers and
  registers copied USD plugins at startup instead of relying on Kit `[[native.library]]` entries.
- **Schema Python modules published under `pxr`** (`omni.cae.schema` 1.5.0): Generated schema Python
  packages now live under `usd/python/pxr/`, and the extension ships a local `pxr/__init__.py` that turns
  `pxr` into a namespace package. Schemas can be imported as
  `from pxr import OmniCae, OmniCaeSids, OmniCaeVtk, ...`, matching Pixar's USD schema convention.
  `omni/cae/schema/*.py` wrappers and the format onboarding tutorial were updated to the new import form.
- **Precise cache invalidation for DAV datasets**: `ConvertToDAVDataSet.invoke` now stores results with a
  `PrimWatch`-based guard (`cache.put_ex`) so cached DAV datasets are invalidated precisely when their source
  prim changes (`omni.cae.dav`).
- **Multi-apply schema support in `PrimWatch`**: `PrimWatch` schemas list now accepts
  `(class_or_str, instance_name)` tuples, enabling fine-grained property filtering against specific multi-apply
  schema instances (`omni.cae.data`).
- **Self-transform watching in the controller**: Operators can now declare
  `CaeVizDatasetTransformingAPI:self` to re-execute automatically when their own prim's transform changes
  (`omni.cae.viz`).
- **Kit SDK 110.1.1 default**: The default Kit version is now 110.1.1.

### Fixed

- **PlanarSlice material shading** (`omni.cae.viz` 1.5.1): `SliceTexture` now emits the sampled slice color by
  default so planar slice colors are less dependent on surrounding scene lighting.
- **Stale intermediate caches on `FieldArray` changes** (`omni.cae.data` 2.2.0): `put_ex()` now
  auto-expands `PrimWatch` prims to include related `CaeDataSet`/`CaeFieldArray` prims, so
  changing a field array's data correctly invalidates any cached result that depended on it.
- Fixed incomplete caching across the `get_input_dataset` call chain — intermediate results were previously
  not cached, causing redundant recomputation on each operator execution (`omni.cae.data`).

## [2.0.1]

### Fixed

- **EnSight variable parsing**: Fixed variable names being read as empty strings when multiple spaces
  separate the variable name from the filename in the `VARIABLE` section (`omni.cae.delegate.ensight`).
- **EnSight per-element variable reader**: Fixed assertion failure when a `.case`/`.encas` var file lists
  element type sections in a different order than the geometry file (`omni.cae.delegate.ensight`).
- **EnSight `.encas` support**: Importer now accepts `.encas` as an alias for `.case` files
  (`omni.cae.importer.ensight`).
- **EnSight quoted filenames**: Geometry and variable filenames wrapped in double-quotes (e.g. from Fluent
  exports) are now parsed correctly, including filenames that contain spaces (`omni.cae.delegate.ensight`).

## [2.0.0]

### Added

- **OpenFOAM format support**: Full import pipeline with data delegate and importer
- **NanoVDB data delegate**: Roundtrip support for NanoVDB datasets via `CaeNanoVDBFieldArrayAPI` schema
- **Biplane/Triplane slice**: New triplane option for the Slice operator
- **Multi-seed streamlines**: Streamlines support multiple seeds natively
- **Operator execution events**: `EVT_OPERATOR_BEGIN` / `EVT_OPERATOR_END` events for synchronizing external workflows (e.g., movie capture)
- **Insights panel enhancements**: Quartile, histogram, and statistical analysis for field data
- **Expression variables**: New extension (`omni.cae.exVars`) to set expression variables from command-line arguments
- **Format onboarding tutorial**: End-to-end guide for integrating custom data formats into the OpenUSD schema and delegate model
- **CaeViz schema documentation**: Visualization operator authoring guide with examples and design rationale
- **`omni.cae.bundle` meta-extension**: Single dependency that brings in all Kit-CAE extensions with pre-configured IndeX, Flow, and Hydra settings
- **Ahead-of-time kernel compilation**: Tool and settings for precompiling Warp/DAV kernels to reduce first-run latency

### Changed

- **Kit SDK 109.0.3**: Updated to Kit 109.0.3
- **Selectable Kit SDK version**: Build against different Kit SDK versions (108.0.0 through 109.0.3) using `select_kit_version` tool without code changes
- **Simplified build**: Removed CMake dependency; schemas now build as part of `repo.sh build` (no separate `repo.sh schema` step)
- **Extensions reorganized by role**: Extensions renamed to clearly reflect function (delegates, importers, viz, UI); legacy code moved to `source/legacy_extensions/`
- **Schemas reorganized**: Split into `source/schemas/shared/` (core, viz) and `source/schemas/formats/` (per-format)
- **Unified field selection**: All visualization operators use `CaeVizFieldSelectionAPI` (multiple-apply schema) for consistent field selection
- **Prim-authored voxelization**: Voxelization settings now authored on USD prims rather than local preferences, ensuring the same stage produces the same result on any machine
- **GPU-accelerated computation**: Bounding boxes, field ranges, and histograms computed on GPU via Warp kernels instead of NumPy
- **64-to-32-bit auto-downconversion**: Float64/int64 arrays automatically downconverted to 32-bit at load time (configurable)
- **Fine-grained cache invalidation**: Cache supports 4 invalidation modes (`any`, `update`, `resync`, `delete`) and schema-filtered property watching
- **Default Z-up axis**: Changed default up axis to Z to match CAE industry convention
- **EnSight Gold enhancements**: Polyhedral element support, metadata caching to avoid repeated file parsing, and import progress reporting
- **VTK enhancements**: Custom VTU reader for faster reads, polyhedra support, and improved reader caching
- **Static geometry support for Faces**: Avoids reprocessing unchanging mesh topology on temporal datasets
- **Ubuntu 24.04 and VS2026 support**: Platform compatibility updates

### Fixed

- Point-based streamline computation
- CGNS NGon_n prims no longer get cell-centered data associated incorrectly
- IndeX race conditions in NanoVDB path
- VTK reader caching scoped per-prim instead of per-filename

## [1.5.0]

- Update to Kit 109.0.1

## [1.4.0]

- Added support to using irregular volume rendering, i.e. Volume (IndeX), for VTK unstructured
  datasets.
- Fixed kit web streaming dependencies for Kit 108 in  `omni.cae_streaming.kit` app.

## [1.3.4]

- Fixed typo in Flow algorithms causing runtime errors

## [1.3.3]

* Fixed API error in NanoVDBHelper.
* Fixed bug in DataSetEmitter which resulted in Root layer being populated with nanovdb values.

## [1.3.2]

* Fix issue on Windows when passing int arrays to UsdRt

## [1.3.1]

* Backwards compatibility issue introduced in 1.3.0: fix bug causing errors when volume stages
  did not have `Material/Colormap` prim present.

## [1.3.0]

## Changes

* Points, Glyphs, External Faces now support coloring by vectors. When coloring with vectors, the
  vector magnitude is used for coloring.
* Cleaned up code for resetting color ranges for colormaps, and domains on MDL shaders. The ranges are automatically
  reset if value is invalid (i.e. min > max) or if the field used to color with is changed.

## [1.2.0]

### Changes

* Streamlines now supports passing arbitrary fields as primvars to the shaders.

### Bug Fixes

* Ensured IndeX algorithms use proper edit-layer when updating attributes on prims during execution
  to avoid clobbering root layer.

## [1.1.0]

### Changes

* Updates to use support Kit 108.0.0

### Bug fixes

* Fixed inability to correctly locate shaders and imported files when using nucleus stages by ensure
  correct scheme (`file:`) is added to such paths.
* Fixed texture PNGs to be non-lfs, avoiding the need to install `git lfs` for basic operation.

## [1.0.0]

### Changes

* CGNS and NPZ datasets can now be imported from Nucleus. Stages referring to CGNS, HDF5 and NPZ assets hosted on Nucleus
  or other supported services are also supported.
* Consolidated XAC shader code for volume rendering of unstructured grid and NanoVDB volumes into a single shader. The shader
  now also supports rendering using magnitude for vector arrays.
* `Slice` algorithm has been refactored to support both creating a `Slice` on existing volume as well as directly on
  a `DataSet`. Schema for Slice has changed subsequently older stages will need to be recreated.
* Split `omni.cae.data.cgns` extension into `omni.cae.cgns`, `omni.cae.file_format.cgns`, and `omni.cae.hdf5` for
  clarity. New extensions `omni.cae.cgns_libs` and `omni.cae.hdf5_libs` now handle packaging and loading CGNS and HDF5 libraries,
  respectively.
* `omni.cae.data.npz`, `omni.cae.data.ensight`, `omni.cae.data.vtk` have been renamed to `omni.cae.npz`, `omni.cae.ensight`,
  and `omni.cae.vtk` for consistency and brevity. Similarly, `omni.cae.utils.sids` has also been renamed to `omni.cae.sids`.

### Bug fixes

* Fixed coding error in EnSight importer causing import failures when importing files with nsided elements.
* `Points` algorithm now uses a fixed `0.001` as the default value for width and no longer uses "Default Point Width"
  setting. This avoids freeze / hang if user forget to change default setting for large point clouds.
* `Slice` no longer resets user specified mesh points or transforms. The initialization of the Slice prim only happens if the
  properties are not already set.
* `Volume (IndeX)` was not correctly updating for temporal datasets. Fixed that.


## [1.0.0-RC1]

* Kit version updated to 107.3. Includes changes to dependencies and toolchain as needed for this Kit version change.
* HDF5 and CGNS versions updated to 1.14.6 and 4.5 respectively.
* Algorithms now use USDRT APIs to update USD prims. USDRT requires Fabric Scene Delegate (FSD) is enabled. Hence Kit-CAE
  now enables FSD by default.
* All algorithms schemas moved to separate extension (`omni.cae.algorithms.schema`). This ensures that this extension
  and hence the USD schemas can be loaded at correct time. This was causing the schemas to not be functional
  in certain cases due to changes in extension loading order.
* IndeX Volume supports passing multiple fields. All passed fields can be accessed in custom XAC shaders. Example
  `LERPMaterial` demonstrates how an XAC shader can be used to interpolate between two fields.
* Added limited support for VTK unstructured files (`.vtu`, `.vtk`). `OmniCaeVtk` has been extended to add define a new
  API schema for VTK unstructured grids (`CaeVtkUnstructuredGridAPI`).
* Default search path for USD Asset resolver is updated to include locations for shaders provided. Referring to
  `cae_materials.mdl`, for example, automatically loads the material provided by the loaded Kit-CAE extension.
* `Points` algorithm now uses same approach for scalar coloring as `External Faces` i.e. uses a MDL shader and primvars
  for mapping scalars to colors.
* PIP packages for VTK, h5py, etc. are no longer downloaded automatically.
  Updated instructions in the [README.md](./README.md) file indicate how to manually download these packages.

## [1.0.0-beta.11]

* Added time support for algorithms. Most algorithms, including Points, Volume, External Faces, now support processing temporal
  datasets.
* NanoVDB Volume now supports temporal interpolation for fields between time samples.
* Added support to Points algorithm for mapping a field array to point widths.
* External Faces now support coloring using a field array.
* Data delegate support added for HDF5 arrays enabling importing HDF5 datasets that are not CGNS.

## [1.0.0-beta.5]

* Added support for VS2022.
* Updated code to use Python `asyncio` for non-blocking data reading and execution.
* Removed `cupy` dependency to minimize runtime issues.
* Added Slice algorithms for slicing through volumetric data.
* Added Warp implementation for generating streamlines using NanoVDB / voxelized data.
* Voxelization now supports region-of-interest (ROI) which can be specified interactively.
* Added support for dense volumetric datasets imported from `.vtk` or `.vti` files.
* Added Glyphs algorithm for rendering arrows/cones/spheres at point locations in a point cloud.
* Streamlines uses MDL shader for scalar coloring instead of using `displayColor` on BasisCurve prim.
* Disabling Fabric Scene Delegate (FSD) until future release.

## [0.1.0-alpha]

* Initial release to limited customers.
