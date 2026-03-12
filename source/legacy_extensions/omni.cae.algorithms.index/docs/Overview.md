# Overview

`omni.cae.algorithms.index` provides CAE-specific IndeX visualization algorithms and commands,
built on top of the `omni.cae.index` bindings extension.

**C++ plugin** (`omni.cae.algorithms.index.plugin`):
- `CaeDataSetImporter`: imports a CAE DataSet prim into an NVIndex irregular volume by delegating
  to `omni.cae.algorithms.index.impl.helpers.CaeDataSetImporter`.
- `CaeDataSetNanoVdbFetchTechnique`: compute technique that voxelizes CAE data into NanoVDB format.

**Python module** (`omni.cae.algorithms.index`):
- `impl.algorithms`: `Slice`, `NanoVdbSlice`, `VolumeSlice`, `Volume`, `NanoVdbVolume` Algorithm
  subclasses registered with `omni.cae.algorithms.core.Factory`.
- `impl.commands`: Kit commands for creating IndeX slices and volumes in the USD stage.
- `impl.helpers`: Python-side bridge classes called by the C++ importers.
- `commands`: re-exports `CreateIrregularVolumeSubset` command type.
