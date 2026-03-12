# Overview

`omni.cae.index` provides the foundation layer for NVIndex integration in CAE:

- **Python bindings** (`_omni_cae_index`): exposes NVIndex C++ types to Python so that algorithm
  code can populate `IIrregular_volume_subset` and related interfaces without writing C++.
- **PythonImporter**: an `IDistributed_data_import_callback` whose `create()` method
  delegates to a caller-supplied Python class/module.
- **PythonComputeTask**: an `IDistributed_compute_technique` whose `launch_compute()` method
  delegates to a caller-supplied Python class/module.

Both generic classes are registered with NVIndex's importer/interface factory under the
`nv::omni::cae::index` namespace.

For the CAE-specific `CaeDataSetImporter` and `CaeDataSetNanoVdbFetchTechnique`, as well as
visualization algorithms and Kit commands, see `omni.cae.algorithms.index`.
