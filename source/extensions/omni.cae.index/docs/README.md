# omni.cae.index

IndeX Python bindings and generic Python-based importers and compute tasks for CAE data.

This extension provides:
- `_omni_cae_index`: pybind11 bindings exposing IndeX C++ types to Python
  (`IIrregular_volume_subset`, `IData_subset_factory`, `Bbox_float32`, etc.)
- `PythonImporter`: generic NVIndex importer that delegates to a Python class
- `PythonComputeTask`: generic NVIndex compute technique that delegates to a Python class

For CAE visualization algorithms and commands, see `omni.cae.algorithms.index`.
