# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.4.0]

### Added
- **Add API > CAE > Dataset Subset** context menu entry: applies `CaeVizDatasetSubsetAPI` to a selected
  prim. Instance-name suggestions come from existing `CaeVizDatasetSelectionAPI` applications, matching
  the other per-selection dataset APIs.

## [2.3.0]

### Changed
- **Copy LUT Texture URL** now only appears for `Colormap` prims that have `CaeVizColormapTextureAPI` applied
  (previously shown for all `Colormap` prims).

### Added
- **Add API > CAE > Colormap Texture** context menu entry: applies `CaeVizColormapTextureAPI` to a selected
  `Colormap` prim and stamps a stable UUID identifier used by `ColormapTextureManager` to publish the dynamic
  LUT texture.

## [2.2.0]

### Added
- **Copy LUT Texture URL** context menu entry: right-clicking a `Colormap` prim in the stage now shows a
  *Copy LUT Texture URL* option that copies the `dynamic://` texture URL for that prim to the clipboard.

## [2.1.0]

- Added `OperatorsPlanarSlice` context menu action for creating a `PlanarSlice` operator on selected CAE datasets.
  Opens a dialog to choose between `standard` (mesh-based probe) and `nanovdb` (voxelized) execution types.
- `TypeSelectionDialog` API updated: `options`/`default_index`/`field_label` constructor arguments replaced by a
  `selections` list of `(label, choices)` pairs, enabling multi-field configuration dialogs.

## [2.0.0]

- Refactored to add support for CAE Operator. All CAE Algorithm specific options
  are marked as legacy and are only shown when legacy UI is enabled through settings.

## [1.0.0]

- Initial version.
