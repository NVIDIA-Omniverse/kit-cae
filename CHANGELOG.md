# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
