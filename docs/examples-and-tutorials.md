# Examples and tutorials

Tutorials explain the APIs step by step; application examples show
integrations you can adapt.

OVCAE workflows require an RTX-capable NVIDIA GPU, including for headless
rendering.

## Getting started

Start with the [getting-started guide](getting-started.md): author a recipe,
evaluate in-memory arrays on the GPU, and publish the result to OVStage,
without a renderer.

Then try the [minimal OVUI workflow](../examples/ovui-minimal/README.md)
to explore native data in a desktop application and see how the libraries
fit together.

## Application examples

Each application has its own environment. Follow its README for setup and execution.

| Example | What it shows |
| --- | --- |
| [Minimal OVUI workflow](../examples/ovui-minimal/README.md) | A two-file OVUI and OVRTX application: open CGNS data, edit an iso-surface, save the scene, and reopen it. |
| [CAE Viewer](../examples/cae-viewer/README.md) | A desktop viewport with picking, transform gizmos, undo, status annotations, and screenshots. Includes a scene and CGNS dataset. |

## Tutorials

### Notebook sequence

Work through the numbered tutorials in order. See the
[tutorials README](../tutorials/README.md) for setup and execution in
JupyterLab or an editor.

| Tutorial | What you will learn |
| --- | --- |
| [0. Check your environment](../tutorials/00_check_environment.py) | Check the kernel, dependency versions, plug-in registration, and compute device. |
| [1. See it working](../tutorials/01_see_it_working.py) | Populate an OVStage from an aerodynamics scene, evaluate surface and streamline recipes, and render with OVRTX. |
| [2. Make it live](../tutorials/02_interactive_stream.py) | Use `UpdateLoop` and `LiveView` to stream the rendered view through OVStream, orbit the camera, and edit an iso-surface in the browser. |
| [3. Bring your own data](../tutorials/03_bring_your_own_data.py) | Compose a CGNS file, author recipes and a render rig, save and reopen the scene, and inspect a bad-field diagnostic. |
| [4. Feed it live data](../tutorials/04_feed_it_live_data.py) | Publish a changing field through `MemoryDataSource` and see a recipe recompute after publication and reuse its result otherwise. |
| [5. Extend it](../tutorials/05_extend_it.py) | Add a Warp by Vector operator, register its custom schema, author a recipe, and edit its scale. |

### Additional tutorials

A1 stands alone. A2 builds on tutorials 2 and 3; A3 builds on A2 and tutorial 5.

| Tutorial | What you will learn |
| --- | --- |
| [A1. Native data without OVCAE](../tutorials/A1_native_data_headless.py) | Use CAE OpenUSD Plugins and Warp SimData to process CGNS data with faces and iso-surface operators on CPU or GPU, without OVCAE or a renderer. |
| [A2. A Data Source of your own](../tutorials/A2_custom_data_source.py) | Read Exodus II with VTK through a custom Data Source and use built-in operators in a live view. |
| [A3. An operator built on VTK](../tutorials/A3_vtk_operator.py) | Wrap VTK's clip and surface filters as an operator with a custom schema and live slider. |

See the [operators](operators.md) and [readers](readers.md) guides for the
extension contracts.
