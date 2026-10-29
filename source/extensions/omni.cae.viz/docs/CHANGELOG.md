# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.6.2]

### Fixed
- Operator failure logs now preserve chained exception causes so actionable
  converter and adapter diagnostics remain visible.

## [2.6.1]

### Fixed
- Planar-slice RT geometry now retains its parent transform when output buffers
  swap and publishes transformed world extents, preventing slices under
  transformed parents from disappearing when moved.

## [2.6.0]

### Added
- Added independent field-driven opacity mapping to the MDL materials used by
  Faces, Points, Glyphs, Iso Surfaces, Streamlines, and Planar Slices. Each
  operator authors a separate `opacity` field selection, rescale range, LUT,
  multiplier, and enable control while retaining a fully opaque default.
- `ColormapTextureManager` now publishes a stable grayscale opacity LUT derived
  from each Colormap's alpha channel alongside its RGBA color LUT. Added
  `CreateCaeVizColormap` for creating a texture-ready Colormap directly.
- Added experimental `CaeVizIndeXAxisymmetricVolumeAPI` support and a dedicated
  axisymmetric operator for FLASH AMR. NVIDIA IndeX renders one multi-attribute
  NanoVDB whose coarse first attribute establishes the revolved domain while
  its XAC shader samples sparse native-resolution refinement attributes,
  avoiding angular replication and preserving each leaf block at its authored
  AMR refinement. The shader clips the analytical revolution to the authored
  angle range, and visited timesteps retain their native payloads for fast
  re-adoption. Ray sampling defaults to the finest native cell spacing.

### Changed
- Planar slices now use `warp_simdata.operators.slice` to extract colored triangle geometry directly
  from volume cells. Multi-plane direction modes batch their extraction through `compute_many`.
  The prim transform positions the selected plane or planes, and double-buffered RT output keeps
  the previous complete surfaces visible while replacements are computed.
- Runtime warnings emitted during SimData slice and iso-surface extraction are
  routed through Kit's warning log channel instead of Python stderr.

### Fixed
- Planar-slice creation now requests the explicit dual representation for
  axisymmetric FLASH datasets, providing the element shapes required for
  triangle extraction.

### Removed
- Removed the probe-grid, dynamic-texture, mask-texture, and `SliceTexture` MDL
  implementation. Free, axis-aligned, and multi-plane direction modes now extract
  colored triangle geometry with the SimData slice operator and render it with the
  unlit scalar-color material on both front and back faces.

## [2.5.0]

### Added
- Iso-surface and streamlines integration tests now cover derived scalar and
  vector fields supplied through the shared SimData input pipeline.
- Added `CaeVizDatasetAxisymmetricRepresentationAPI` processing to the shared input
  pipeline. FLASH inputs use concrete native or dual representations with authored angular cells
  and angle ranges, including cache invalidation when those properties change.
- Iso-surface creation applies axisymmetric representation controls automatically for supported
  datasets alongside the existing dual request.
- Iso-surface output caching is scoped by device and timecode, reuses results for temporal updates,
  and evicts them for full rebuilds through the operator execution context instead of maintaining
  a separate state snapshot and schema-specific watch list.

## [2.4.0]

### Added
- Added experimental `CaeVizDatasetDualAPI` processing. Iso-surface creation automatically authors
  the `source` instance for dual-capable inputs, and dataset plus field loading consistently uses
  the requested representation with representation-aware caching.
- Added time-code-aware caching of generated iso-surface datasets when the reconstructed input,
  iso-value, and requested output fields are unchanged.

### Fixed
- Iso-surfaces with no generated points or elements are hidden until a subsequent execution
  produces renderable geometry, preventing stale surfaces from remaining visible.

## [2.3.0]

### Added
- Added the `UnlitScalarColor` MDL material, which maps the `colors` scalar primvar through an
  optional LUT and emits the resulting color independently of scene lighting.
- Added a SimData-backed iso-surface operator, creation command, scalar-color material wiring, and
  cached field-association conversion in `get_input_dataset`. Cell contour and output fields are
  averaged onto points before triangular surface extraction.

## [2.2.0]

### Added
- Added support for `CaeVizDatasetVoronoiPointCloudAPI`, converting selected input datasets to
  SimData Voronoi point-cloud datasets before optional voxelization and carrying selected fields
  across as node fields.
- Point-cloud dataset model APIs are now resolved in applied-schema order, so Voronoi conversion
  and Gaussian splatting are mutually exclusive and deterministic when both APIs are present.

## [2.1.0]

### Added
- Time and change-tracking machinery now follows USD-plugin-backed `OmniSciDataset` prims, so
  operators driven by USD plugin schemas (reservoir, NPZ, NanoVDB, Trimesh, …) re-execute on the
  correct timecodes and prim mutations.
- Operator execution now reports progress to the Kit progress UI via the shared progress service, so
  long-running operators surface their stage in the status line.
- New `change_tracker` and `listener` helpers in the controller decouple stage event handling from
  operator execution, simplifying re-entry and stage transitions.

### Changed
- Switched runtime imports from `omni.cae.data` to `omni.cae.core` to track the core/legacy split.
- Tightened operator test utilities and fabric synchronization, removing flakiness in
  controller/faces/index_volume/points/slice/streamlines test suites.

### Fixed
- Stabilized volume temporal interpolation tests against renderer warm-up and stage update timing.
- Operators using `CaeVizDatasetSubsetAPI.roi` or `CaeVizDatasetVoxelizationAPI.roi` now re-execute
  when the ROI target prim's transform changes, keeping subset and voxelized inputs in sync with
  interactive bounding-box edits.

## [2.0.0]

### Changed
- Removed the dependency on the legacy CAE material-library extension; CAE MDL assets and LUT textures are packaged
  directly with `omni.cae.viz`.
- Added an explicit `omni.flowusd` dependency for Flow prim authoring commands.
- Documentation now describes the SimData-backed visualization operator stack as the primary processing path.

## [1.5.1]

### Fixed
- `SliceTexture` now emits the sampled slice color by default so planar slice colors are less
  dependent on surrounding scene lighting.

## [1.5.0]

### Added
- `get_input_dataset` now honors `CaeVizDatasetSubsetAPI` on the selected instance. When applied, the
  input dataset is restricted to cells inside the ROI prim's bounds (optionally inflated by
  `inflateBounds`) before voxelization, using the `cell_in_box` operator and the `cell_subset` data
  model. Parent fields are carried across via `cae_simdata.pass_fields` using an ephemeral `cell_idx`
  indirection.
- New create-command wiring auto-applies `CaeVizDatasetSubsetAPI:source` when creating non-voxelized
  operators: Points, Faces, and Glyphs (only when the input dataset has cells); Streamlines and
  PlanarSlice in standard mode; and Volume in irregular mode. Voxelized paths are untouched since
  voxelization already scopes the ROI.
- `CreateCaeVizFaces` is now async so it can query cell count before deciding whether to apply the
  subset API.

## [1.4.0]

### Fixed
- `PlanarSlice`: pre-create all RT quad prims (invisible) before the data fetch so the renderer
  discovers them in Fabric on the first (possibly failed) exec. Previously, prims were only created
  on the first *successful* exec; because the renderer needs one full cycle to register newly created
  Fabric prims, the slice was invisible until a second exec ran.
- `RtSubPrimGuard` registry and stage-update subscription promoted to class attributes (`_registry`,
  `_stage_sub`). The subscription calls `clear_all()` on every stage attach and detach, so guards
  referencing prims from a previous stage are revoked before any new guards are registered. Previously
  the module-level registry was never cleared on stage transitions, causing `register()` to silently
  skip re-registration for prim paths reused across stages and leaving RT sub-prims alive after their
  source prim was deleted on the new stage.

## [1.3.0]

### Changed
- `ColormapTextureManager` now requires `CaeVizColormapTextureAPI` to be applied to a `Colormap` prim before
  managing it. The texture URL is `dynamic://cae_colormap_<identifier>` where `identifier` is the UUID stored
  in `cae:viz:colormapTexture:identifier`, making it stable under USD prim relocation.
- `get_dynamic_url_for_identifier(identifier)` replaces the removed path-based helpers as the primary public
  API for constructing texture URLs.

## [1.2.0]

### Added
- `ColormapTextureManager`: a stage-scoped service that monitors USD `Colormap` prims and publishes each one
  as a dynamic LUT texture. Texture URL derived from a stable `identifier` attribute via `CaeVizColormapTextureAPI`.
- Unit tests for LUT generation, texture naming, prim discovery, updates, and deletion cleanup
  (`tests/test_colormap_texture_manager.py`).

## [1.1.0]

### Added
- `PlanarSlice` operator: texture-mapped planar slice extracted from CAE datasets using `simdata.operators.probe`.
  Supports `free`, single-axis (`x`/`y`/`z`), dual-axis (`xy`/`xz`/`yz`), and tri-axis (`xyz`) modes.
  Does not require IndeX.
- `CreateCaeVizPlanarSlice` command: creates a `UsdGeomMesh` with `CaeVizPlanarSliceAPI` applied, wired up to
  a `SliceTexture` MDL material and a configurable colormap.
- `RtSubPrimGuard` utility: keeps RT-only sub-prims in sync with a primary USD prim's visibility and
  deactivation state, and removes them on prim deletion.
- Unit and integration tests (`tests/test_slice.py`).

### Changed
- Controller now handles `CaeVizDatasetTransformingAPI:self`, allowing operators to re-execute when their own
  prim transform changes (required by `PlanarSlice`).

## [1.0.0] - 2025-11-30

### Added
- Initial release of omni.cae.viz extension
- Support for OmniCaeViz USD schemas
- Integration with omni.cae.schema and omni.cae.data
- Basic extension infrastructure for CAE visualization
