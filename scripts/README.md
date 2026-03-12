# Example Scripts

Sample scripts demonstrating Kit-CAE data import, visualization, and processing
workflows. Each script can be executed on launch:

```bash
./repo.sh launch -n omni.cae.kit -- --exec scripts/<script_name>.py
```

> **Note**: `example_headsq_vti.py` requires VTK pip dependencies and the
> VTK-enabled application variant (`omni.cae_vtk.kit`).
> See [Build Instructions](../docs/Build.md#optional-dependencies) for setup.

## CGNS — StaticMixer.cgns

| Script | Description |
|--------|-------------|
| `example_bounding_box.py` | Bounding box creation around an imported dataset |
| `example_faces.py` | Surface mesh face visualization with field coloring |
| `example_glyphs.py` | Glyph (arrow) rendering with multi-field mapping |
| `example_points.py` | Point cloud with field-based sizing |
| `example_slice.py` | Volume slicing on irregular grids |
| `example_streamlines.py` | Streamline tracing through a velocity field |
| `example_volume.py` | Volume rendering of a scalar field |
| `example_nvdb_slice.py` | NanoVDB slicing with animation |

## CGNS — hex_timesteps.cgns

| Script | Description |
|--------|-------------|
| `example_temporal_interpolation.py` | Temporal interpolation on a time-varying hex mesh |

## NumPy — disk_out_ref.npz

| Script | Description |
|--------|-------------|
| `example_npz_flow.py` | Flow simulation with smoke injection |
| `example_npz_point_cloud.py` | Point cloud with Gaussian splatting |
| `example_npz_streamlines.py` | Streamlines from NumPy arrays |

## VTK — headsq.vti (requires VTK)

```bash
./repo.sh launch -n omni.cae_vtk.kit -- --exec scripts/example_headsq_vti.py
```

| Script | Description |
|--------|-------------|
| `example_headsq_vti.py` | VTK ImageData with volume rendering and ROI |

## Tutorial — Generated Data

| Script | Description |
|--------|-------------|
| `generate_scae_data.py` | Generates a synthetic `.scae` torus dataset, imports it, and visualizes with bounding box and volume rendering |

## Developer Notes

These scripts are tested as part of the
[`omni.cae.bundle`](../source/extensions/omni.cae.bundle/python/tests/test_examples.py)
extension. If adding a new script, ensure that a test has been added for that
script in `omni.cae.bundle`.
