# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.5.1]

### Fixed
- `SliceTexture` now emits the sampled slice color by default so planar slice colors are less
  dependent on surrounding scene lighting.

## [1.5.0]

### Added
- `get_input_dataset` now honors `CaeVizDatasetSubsetAPI` on the selected instance. When applied, the
  input dataset is restricted to cells inside the ROI prim's bounds (optionally inflated by
  `inflateBounds`) before voxelization, using the `cell_in_box` operator and the `cell_subset` data
  model. Parent fields are carried across via `cae_dav.pass_fields` using an ephemeral `cell_idx`
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
- `PlanarSlice` operator: texture-mapped planar slice extracted from CAE datasets using `dav.operators.probe`.
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
