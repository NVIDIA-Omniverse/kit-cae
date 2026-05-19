# Format Reference

Stage paths are *illustrative* — always inspect or use stage discovery for your data.

## CGNS (.cgns)

**Kit:** `omni.cae.kit` | **Import:** `from omni.cae.importer.cgns import import_to_stage`

```python
await import_to_stage(path, prim_path, time_scale=1.0, time_offset=0.0, time_source="TimeStep")
```

`time_source`: `"TimeStep"` (index) or `"TimeValue"` (simulation time).

**Stage:** `/World/<root>/Base/<Zone>/{GridCoordinates|<MeshBlock>|<FlowSolution>/<Field>}`

**Name sanitization:** dots/spaces → underscores (`B1.P3` → `B1_P3`).

**Velocity naming:** varies by solver — `VelocityX/Y/Z`, `Velocity_0/1/2`, `U/V/W`. Never hardcode.

**Inspect:** `CAE_INSPECT_FILE=<f> ./repo.sh launch -n omni.cae.kit -- --exec skills/cae-core/scripts/inspect_cgns.py --no-window`

---

## VTK (.vti .vtu .vts .vtp .vtk)

**Kit:** `omni.cae_vtk.kit` | **Import:** `from omni.cae.importer.vtk import import_to_stage`

```python
await import_to_stage(path, prim_path)
```

**Requires:** `./repo.sh pip_download`

**Stage:** `/World/<name>/VTK{ImageData|UnstructuredGrid|StructuredGrid|PolyData}`
Fields: `/World/<name>/{PointData|CellData}/<Field>`

**Inspect:** `CAE_INSPECT_FILE=<f> ./repo.sh launch -n omni.cae_vtk.kit -- --exec skills/cae-core/scripts/inspect_vtk.py --no-window`

---

## EnSight Gold (.case .encas)

**Kit:** `omni.cae.kit` | **Import:** `from omni.cae.importer.ensight import import_to_stage`

```python
await import_to_stage(path, prim_path, time_scale=1.0, time_offset=0.0)
```

**Stage:** `/World/<name>/VTK_Part/Variables/<Field>`. Multi-part: `VTK_Part_<N>`.

**⚠** `CreateCaeVizFaces` `external_only` unsupported for volumetric parts.

---

## OpenFOAM (.foam)

**Kit:** `omni.cae.kit` | No scripted `import_to_stage` — UI only (File > Import).

**Stage:** `/World/<case>/{Volume|boundaries/<patch>|internalFields/<f>}`

Create empty `.foam` file in case root if needed.

---

## NumPy (.npz .npy)

**Kit:** `omni.cae.kit` | **Import:** `from omni.cae.importer.npz import import_to_stage`

```python
await import_to_stage(path, prim_path, schema_type="SIDS Unstructured")
# schema_type: "SIDS Unstructured" (mesh) or "Point Cloud"
```

**Stage:** Dataset at `/World/<name>/NumPyDataSet`, fields at `/World/<name>/NumPyArrays/<Array>`.

**⚠ SIDS Unstructured requires field association fix** (see `kit-cae-api.md`).

**Volume type:** Use `type="irregular"` for NPZ with cell connectivity (`element_connectivity` array), `type="vdb"` for point clouds without topology.

**Field naming:** Array names from the NPZ file are used directly (e.g., `Temp`, `V`, `Pres`). Never assume field names — always discover via USDRT.

---

## EDEM (.dem)

**Kit:** `omni.cae.kit` + `--enable omni.cae.delegate.edem --enable omni.cae.importer.edem`

Not bundled by default. Uses asset importer API.

---

## Custom Formats

4-layer architecture: Schema → Delegate → Importer → API Schema.
See `extensibility.md` or `docs/FormatOnboarding.md`.
