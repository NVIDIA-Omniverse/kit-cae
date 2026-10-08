# Troubleshooting

Start with `report.diagnostics` and the `OperatorResult` for the affected
recipe. A committable report does not mean every operator succeeded. The table
below covers common problems.

| Symptom | Check or action |
| --- | --- |
| Import fails on `libGL.so.1` | Install `libgl1`; OVStage needs it even in headless processes |
| Missing CAE file reader | Install the `usd` extra for the CAE OpenUSD Plugins |
| Scientific payload does not compose | Import `ovcae.readers` (or call `ovcae.setup()`) before the first usd-core stage so the file formats register |
| Recipe properties absent in OVStage | Call `ovcae.setup()` before creating the Stage and populate with `ovcae.ovstage_population_desc(...)` |
| Dataset cannot be resolved | Check the selection target, identifier, payload paths, and that the Data Source composes the same layer stack |
| Multiple sources claim a key | Remove the overlapping claim; `CompositeDataSource` rejects ambiguity by design |
| `OrdinalError` | Commit the inputs first and choose an output ordinal above both the input and the write floor |
| `UpdateInFlightError` after `done()` returned true | Call `Update.wait()` to consume the completion before submitting again |
| `UpdateInFlightError` from `UpdateLoop.write` or `submit` | An update is running; call `pump()` or `flush()` to commit it first, or present the previous frame and retry next frame |
| `TypeError: cannot determine the USD type` from `UpdateLoop` | The attribute's type cannot be read or inferred from the value; pass `types={(path, name): "float"}` with the USD type. `request` raises it at once, or from the `submit` or `pump` that writes the edit when an update was in flight at request time |
| `StageSessionConflictError` | Close the existing Session on that Stage before creating another |
| `ActiveFrameDataError` on close | Exit every `at_time` context before closing the source |
| Hidden recipe accepts invalid settings | Hidden recipes defer validation; show it and run an update to get diagnostics |
| Geometry stale after showing a recipe | Run and commit an update after changing visibility, before rendering |
| Geometry stale after many presentation commits | Check the [history-retention requirement](integration-guide.md#history-retention) |
| Replaced file does not update | Close and reopen the USD source; there is no live reload |
| Streamlines evaluates on every update | Expected; scene seeds have no comparison identity |
| Materials render flat or with the wrong color | Register materials before creating the renderer and author a known `color_range` |
| Colormap renders as a grey ramp | Create the OVRTX renderer before populating the OVStage; textures given as search-path-relative `cae/colormaps/...` paths resolve only then |
| `SCHEMA_NOT_CAPTURED` warning and a custom operator's setting is out of scope | Register the schema folder with `ovcae.register_schema_paths(path)` before the process creates any stage |
| `OutOfScopeError` or `ExpiredViewError` in a custom operator | Read only planned properties and keep handle use inside `execute` |
| Custom operator fails on return | Match the product type to the recipe prim, synchronize arrays first, and return `op_utils.empty()` rather than `None` for an empty result |

If `report.committable` is false, do not advance the write floor to that
output ordinal. Inspect the publication and cleanup diagnostics; if a consistent
Stage cannot be recovered, rebuild the application Stage and Session from the
authored inputs. `UpdateError.report` may contain a partial outcome.

For CAE Viewer problems, see its [README](../examples/cae-viewer/README.md).
When reporting an issue, include the commit tested, the versions of OVStage,
Warp SimData, and the CAE OpenUSD Plugins, and the diagnostics from the report.
