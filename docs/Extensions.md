# Extensions Overview

Kit-CAE is composed of modular Omniverse extensions organised into the following categories.

| Category | Extensions |
|----------|-----------|
| [USD Schemas](#usd-schemas) | `omni.cae.schema` |
| [Data Infrastructure](#data-infrastructure) | `omni.cae.data`, `omni.cae.dav` |
| [Data Delegates](#data-delegates) | `omni.cae.delegate.*` |
| [File Formats & Importers](#file-formats--importers) | `omni.cae.file_format.cgns`, `omni.cae.importer.*` |
| [Visualization](#visualization) | `omni.cae.viz`, `omni.cae.index` |
| [UI](#ui) | `omni.cae.context_menu`, `omni.cae.property.bundle`, `omni.cae.widget.stage_icons` |
| [Utilities](#utilities) | `omni.cae.bundle`, `omni.cae.exVars`, `omni.cae.testing`, `omni.cae.pip_prebundle` |
| [Native Libraries](#native-libraries) | `omni.cae.cgns_libs`, `omni.cae.dav_libs`, `omni.cae.hdf5_libs`, `omni.cae.vtk_libs` |

---

## USD Schemas

#### [`omni.cae.schema`](../source/extensions/omni.cae.schema/)

Loads the CAE USD schemas into Omniverse. USD plugins must be registered early during initialisation; this extension handles that lifecycle. It exposes the core `CaeDataSet` and `CaeFieldArray` prim types plus all associated API schemas described in [USD Schemas](./UsdSchemas.md).

---

## Data Infrastructure

#### [`omni.cae.data`](../source/extensions/omni.cae.data/)

Core extension providing the **Data Delegate API** — an extensible mechanism for reading data from `CaeFieldArray` prims. Extensions register delegates for specific `CaeFieldArray` subtypes; `omni.cae.data` dispatches read requests to the right delegate at runtime. See [Data Delegate API](./DataDelegate.md).

#### [`omni.cae.dav`](../source/extensions/omni.cae.dav/)

Provides DAV-specific data processing algorithms and operators. DAV is the internal compute library used by `omni.cae.viz` for CAE algorithms (streamlines, face extraction, etc.).

---

## Data Delegates

These extensions register delegates with `omni.cae.data` to handle reading data from specific `CaeFieldArray` subtypes and file formats.

#### [`omni.cae.delegate.cgns`](../source/extensions/omni.cae.delegate.cgns/)

Reads data from `CaeCgnsFieldArray` prims — field arrays stored in CGNS (`.cgns`) files.

#### [`omni.cae.delegate.hdf5`](../source/extensions/omni.cae.delegate.hdf5/)

Reads data from `CaeHdf5FieldArray` prims — field arrays stored in HDF5 files.

#### [`omni.cae.delegate.npz`](../source/extensions/omni.cae.delegate.npz/)

Reads data from `CaeNumPyFieldArray` prims — field arrays stored in NumPy `.npy` / `.npz` files. Pure-Python implementation.

#### [`omni.cae.delegate.vtk`](../source/extensions/omni.cae.delegate.vtk/)

Reads data from VTK field arrays stored in `.vti`, `.vtu`, `.vts`, `.vtp`, and `.vtk` files.

#### [`omni.cae.delegate.ensight`](../source/extensions/omni.cae.delegate.ensight/)

Reads data from EnSight Gold datasets.

#### [`omni.cae.delegate.openfoam`](../source/extensions/omni.cae.delegate.openfoam/)

Reads data from OpenFOAM mesh and field files.

#### [`omni.cae.delegate.trimesh`](../source/extensions/omni.cae.delegate.trimesh/)

Reads surface mesh formats (STL, OBJ, PLY, OFF, GLTF/GLB, and others) via the `trimesh` Python library.

#### [`omni.cae.delegate.edem`](../source/extensions/omni.cae.delegate.edem/)

Reads EDEM particle simulation datasets from HDF5 files.

---

## File Formats & Importers

#### [`omni.cae.file_format.cgns`](../source/extensions/omni.cae.file_format.cgns/)

USD **file format plugin** that allows `.cgns` files to be opened directly in a USD stage. When a CGNS file is referenced, USD calls this plugin to produce the corresponding prim hierarchy with `CaeCgnsFieldArray` prims.

### Importers

These extensions add entries to the **File → Import** menu for their respective formats. Each importer creates the appropriate `CaeDataSet` and `CaeFieldArray` prim hierarchy in the active stage.

#### [`omni.cae.importer.cgns`](../source/extensions/omni.cae.importer.cgns/)

Imports CGNS (`.cgns`) files.

#### [`omni.cae.importer.vtk`](../source/extensions/omni.cae.importer.vtk/)

Imports VTK files (`.vtk`, `.vti`, `.vtu`).

#### [`omni.cae.importer.ensight`](../source/extensions/omni.cae.importer.ensight/)

Imports EnSight Gold CASE (`.case`) files.

#### [`omni.cae.importer.npz`](../source/extensions/omni.cae.importer.npz/)

Imports NumPy `.npz` / `.npy` files, typically used for point cloud data.

#### [`omni.cae.importer.openfoam`](../source/extensions/omni.cae.importer.openfoam/)

Imports OpenFOAM case directories.

#### [`omni.cae.importer.edem`](../source/extensions/omni.cae.importer.edem/)

Imports EDEM particle simulation datasets.

---

## Visualization

#### [`omni.cae.viz`](../source/extensions/omni.cae.viz/)

The **CAE visualization operator runtime**. Monitors the USD stage for prims with `CaeVizOperatorAPI` applied, then executes the corresponding visualization algorithm — surface extraction, streamlines, points, glyphs, or volume rendering. Algorithms write their results back as standard USD geometry prims so any Hydra renderer can display them.

See [CAE Viz Schemas](./CaeVizSchemas.md) for the full schema reference and authoring guide.

#### [`omni.cae.index`](../source/extensions/omni.cae.index/)

Provides NVIDIA IndeX Python bindings and generic compute tasks for IndeX-based volume rendering. Used by the `CaeVizIndeXVolumeAPI` operator in `omni.cae.viz`.

---

## UI

#### [`omni.cae.context_menu`](../source/extensions/omni.cae.context_menu/)

Adds CAE-specific entries to the Stage widget context menu, giving users one-click access to common CAE operations (e.g. adding visualization operators to a dataset prim).

#### [`omni.cae.property.bundle`](../source/extensions/omni.cae.property.bundle/)

Custom property panel widgets for CAE schema attributes — provides richer UI controls (e.g. colour pickers, range sliders) instead of generic USD property editors.

#### [`omni.cae.widget.stage_icons`](../source/extensions/omni.cae.widget.stage_icons/)

Custom icons for CAE prim types (`CaeDataSet`, `CaeFieldArray`, etc.) in the Stage widget.

---

## Utilities

#### [`omni.cae.bundle`](../source/extensions/omni.cae.bundle/)

Meta-extension that depends on all Kit-CAE extensions. Enabling this single extension brings in the full CAE stack — useful as the single dependency in application `.kit` files.

#### [`omni.cae.exVars`](../source/extensions/omni.cae.exVars/)

Reads `expressionVariables` from the command line and injects them into all session layers at stage load. Useful for parameterising USD files (e.g. data paths) at launch time without editing assets.

#### [`omni.cae.testing`](../source/extensions/omni.cae.testing/)

Shared testing utilities and fixtures for Kit-CAE extension test suites.

#### [`omni.cae.pip_prebundle`](../source/extensions/omni.cae.pip_prebundle/)

Bundles Python pip packages required by other CAE extensions so they are available offline without a live pip install.

---

## Native Libraries

These are internal extensions that ship compiled native libraries. They have no public Python API and exist only to satisfy dependencies of other extensions.

| Extension | Libraries provided |
|-----------|-------------------|
| [`omni.cae.cgns_libs`](../source/extensions/omni.cae.cgns_libs/) | CGNS C library |
| [`omni.cae.hdf5_libs`](../source/extensions/omni.cae.hdf5_libs/) | HDF5 C library |
| [`omni.cae.dav_libs`](../source/extensions/omni.cae.dav_libs/) | DAV compute library |
| [`omni.cae.vtk_libs`](../source/extensions/omni.cae.vtk_libs/) | VTK libraries |

---

## Legacy Extensions

The following extensions under `source/legacy_extensions/` are considered legacy and will be removed in a future release. They implement an older algorithm operator pattern that has been superseded by `omni.cae.viz` and the OmniCaeViz schemas.

- `omni.cae.algorithms.core`
- `omni.cae.algorithms.index`
- `omni.cae.algorithms.schema`
- `omni.cae.algorithms.warp`
- `omni.cae.experimental.dav`
- `omni.cae.flow`
- `omni.cae.material_library`
- `omni.cae.schema.simh`
- `omni.cae.sids`
- `omni.cae.simh`
- `omni.cae.vtk`
