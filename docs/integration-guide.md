# Integration guide

This guide covers stage ownership, data access, recipe evaluation, and result
reuse in an application that embeds OVCAE. Start with the
[getting-started tutorial](getting-started.md) for a complete example.

## Concepts

**Execution Stage.** The application-owned `ovstage.Stage`. It holds the
committed recipes, the application's other scene content, and the geometry
OVCAE produces. The application creates it, populates it, and commits every
ordinal, either directly or through `UpdateLoop`. Session reads committed
inputs and writes products without committing them.

**Operator Recipe.** A prim carrying `CaeVizOperatorAPI`, one operator schema
such as `CaeVizFacesAPI`, and selection schemas naming its datasets and fields.
The recipe prim is also the product target: a Faces recipe is a `Mesh` whose
geometry OVCAE fills in, a Streamlines recipe is a `BasisCurves`, and a Points
recipe is a `Points`. Glyphs uses a `PointInstancer` with an
application-authored prototype hierarchy.

**Visualization Product.** The generated geometry and primvars written onto a
recipe prim by evaluating it. Points are in the recipe prim's local coordinates.

**Data Source.** A provider of scientific datasets that is independent of the
Execution Stage. It yields time-bound **Frame Data**, an opaque lease that stays
stable for the lifetime of an update. Operators receive Warp SimData datasets
from it and never see files, readers, or usd-core objects.

**Session.** A long-lived evaluator that borrows one Execution Stage and is the
only publisher of products on it. It owns operator bindings, generated
attributes, and diagnostics.

**Update.** One evaluation of the known recipes, reading inputs committed
through at least the input ordinal and writing into one caller-selected output
ordinal. It completes with an immutable **report** of per-operator results and
**diagnostics**.

The application owns the Stage, Data Source, renderer, and presentation.
Session finds recipes, resolves their data, computes them, replaces generated
attributes, and reports what happened. The optional `UpdateLoop` manages
ordinals and commits on the application's behalf; `LiveView` runs that loop
and rendering on its own thread.

## Initialize the process

Call `ovcae.setup()` before creating any stage, OVStage or usd-core. It
imports every installed reader so usd-core knows their file formats, registers
the codeless schemas with usd-core, registers the schema search paths with
OVStage, and registers the bundled MDL material library with the MDL SDK
environment. When rendering with OVRTX, call `ovrtx.register_schema_paths()`
first, then `ovcae.setup()`, then create the Stage and the renderer before
populating the Stage. After `setup()` the order in which `ovcae.authoring` and
reader packages are imported no longer matters. See
[installation](installation.md#initialize-your-application) for the details.

## Populate the Stage

Populate recipe prims with the descriptors returned by
`ovcae.ovstage_population_desc(...)`:

```python
input_ordinal = 1
ovstage.population.open_usd_with_desc(
    stage,
    "scene.usda",
    ordinal=input_ordinal,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
stage.advance_write_floor(input_ordinal).wait()
```

Population's `time_code` uses USD time codes; scientific data requests use
seconds, as described [below](#scientific-time-and-lease-lifetime).

The descriptors carry the CAE schema properties directly, so OVStage is the one
authority for recipe settings, transforms, visibility, and material bindings.
OVStage has no CAE file-format plug-ins: a scientific payload does not compose
there, and a dataset selection inside a payload is a path string that OVStage
passes through untouched. The Data Source resolves that path against the same
layer stack, which is why population and the Data Source must see identical
layers.

## Choose a Data Source

| Source | Construction | Use |
| --- | --- | --- |
| Files in a USD scene | `ovcae.readers.UsdDataSource.open(path)` | Compose the scene privately with usd-core and the CAE plug-ins |
| Existing USD stage | `ovcae.readers.UsdDataSource(usd_stage)` | Reuse an application-owned `pxr.Usd.Stage` configured for held interpolation |
| Memory | `ovcae.MemoryDataSource(mapping)` | Serve Warp SimData datasets from memory; `publish` replaces them or adds time samples |
| Multiple sources | `ovcae.CompositeDataSource(*sources)` | Route each dataset key to exactly one child |

`ovcae` exposes the memory and composite sources; `ovcae.readers` provides
file-backed access to datasets in their native formats. The
[readers guide](readers.md) covers it and how to write another source.
`UsdDataSource.open` takes a file path and no other options. To
restrict composition on a large scene, construct a masked usd-core stage
yourself and pass it to `UsdDataSource`; that is an optimization, never a
requirement.
`MemoryDataSource()` with no datasets is valid for discovery or cleanup runs,
although a recipe that requests a missing dataset still reports a resolution
failure.

### How recipes name datasets

`CaeVizDatasetSelectionAPI:<role>` has one `target` relationship and an optional
`identifier` string. The target path is the dataset key. A payloaded source file
composes into `OmniSciDataset` prims beneath its holder prim, so a recipe can
target a dataset prim directly, or target the holder and name a dataset beneath
it in `identifier`. Custom sources use descriptor prims carrying their own API
schema and receive qualifiers through `identifier` as well. Neither holders nor
descriptors give the scientific arrays transform or visibility semantics.

A selected scene prim that is `UsdGeomPointBased` (`Points`, `Mesh`,
`BasisCurves`, ...) is not a scientific dataset. OVCAE reads it from the bounded
scene view and converts it to a point cloud; Streamlines uses this route for its
seeds. Field selection on such inputs is not supported.

### In-memory datasets

`MemoryDataSource` holds Warp SimData datasets and serves them under the keys
recipes select. Mapping keys are dataset paths; a `(key, identifier)` tuple
registers a qualified dataset for recipes that set an `identifier`. Build the
dataset with Warp SimData's own data models and attach fields with
`Dataset.add_field` under the names the recipes' field selections use:

```python
import numpy as np
import warp as wp
import warp_simdata as simdata
from warp_simdata.data_models.custom import point_cloud

import ovcae

points = wp.array(np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32), dtype=wp.vec3f, device="cuda:0")
dataset = point_cloud.create_dataset(points)
temperature = wp.array(np.array([290, 310], dtype=np.float32), device="cuda:0")
dataset.add_field("Temperature", simdata.Field.from_array(temperature, simdata.AssociationType.NODE))

source = ovcae.MemoryDataSource({"/Data": dataset})
source.close()
```

`ovcae.data.dataset_from_arrays` builds a point cloud or an unstructured grid
from NumPy arrays and VTK-style connectivity when that is more convenient.

At resolution time the source hands each operator a shallow copy of the
dataset carrying only the selected fields, renamed to the recipe's roles: a
selection naming one field attaches it as is, and one naming several scalar
components combines them into a vector field. The published dataset is never
modified. A selection that names a field the dataset does not have fails with
a diagnostic listing the fields it does have. Datasets are served on the
device they were built on; a Session evaluating on another device fails to
resolve them rather than copying.

`publish(key, dataset, *, identifier="", time=None)` replaces a dataset after
construction, and `remove(key, *, identifier="")` forgets one. Each publish
gives the dataset a new comparison identity; operators using an unchanged
sample can reuse their committed products. Frame Data taken before a publish
keeps serving the datasets it saw, so an Update already running never observes
the replacement. This is the path for
data produced while the application runs: publish the new arrays, run an
Update, commit. Keep published arrays stable while updates use them; in-place
edits are not detected without another publish.

Pass `time` in seconds to publish one sample of a time-varying dataset.
`at_time(t)` then serves the latest sample at or before `t`, or the first
sample when `t` precedes the series, the same held-sample rule the USD source
follows. Without `time` a dataset is static and serves every time. Publishing
a static dataset under a key discards its samples, and publishing a sample
discards the static dataset. `published()` lists the `(key, identifier)`
pairs that currently hold data.

### Scientific time and lease lifetime

`source.at_time(time_seconds)` returns a context manager yielding Frame Data.
Scientific time selects data samples; scene ordinals order publications.
Submit updates inside that context. A submitted update retains the source
resources it needs until it finishes, even if the application's lease ends
earlier. Do not construct Frame Data yourself or reuse a lease after its context
has exited.

`UsdDataSource.open()` configures held interpolation. When supplying your own
usd-core stage, set held interpolation before wrapping it, as shown in the
[readers guide](readers.md#read-at-evaluation-time). Requests are in seconds and
the source converts them to the stage's time-code scale; default-only data is
static. A USD source retains the
most recently selected sample per dataset prim and device, including loaded
arrays and acceleration structures, and replaces that load when a new sample is
selected. Composed layers and backing assets must remain unchanged while a USD
source is open. After replacing a file, close and reopen the source; there is no
file watching or live reload.

Close a source only after its application leases have exited. `close()` is
idempotent and defers releasing resources until submitted work finishes. It
raises `ActiveFrameDataError` while an application lease is active, and a closed
source raises `DataSourceClosedError` on new leases. Closing a source never
changes the Execution Stage.

### Combine or implement sources

`CompositeDataSource` leases every child for the requested time and routes each
dataset key to the one child whose `resolves(key)` claims it. Zero claims or
several claims are resolution errors by design. It requires at least one child,
and the composite closes its children.

To implement a source, subclass `ovcae.DataSource`, call `super().__init__()`,
and implement the following. The [readers guide](readers.md#write-a-reader)
describes how such a source is packaged alongside its schema.

- `resolves(key)`: claim keys without loading arrays; return false for unknown keys.
- `frame_at_time(time_seconds)`: return a provider fixed to that time selection.
- `release_resources()`: release source-owned resources at close.

The provider implements `resolve_dataset(key, *, identifier, device, fields)`
and returns a Warp SimData dataset; `fields` maps output field roles to backing
array names. To take part in change detection, the provider also implements
`describe_component(key, identifier, component)` and returns a comparable
identity together with a `DatasetSample`, without loading arrays. A provider
without that hook is valid and simply executes conservatively. `ovcae.data`
exports the helpers such a source needs: `resolve_asset_path`, `held_time`,
`dataset_from_arrays`, `add_fields`, and `DatasetSample`. Resolve relative
asset paths against the layer that authored them. The
[custom Data Source appendix](../tutorials/A2_custom_data_source.py) shows
descriptor prims, identifiers, held sampling, and identities over VTK.

## Create a Session

```python
import ovcae

# stage is the populated, application-owned ovstage.Stage.
with ovcae.Session(stage, device="cuda:0") as session:
    ...
```

`Session(stage, *, device="auto", operators=None)` borrows the Stage and
reserves it: a second live Session on the same Stage raises
`StageSessionConflictError`. Use `device="cuda:0"` to select the first GPU.
With the default `auto`, a recipe's own device setting is honored and otherwise
the first available GPU is selected. An explicit Session device applies to
every recipe. `operators=None`
selects the built-in operators; a supplied `OperatorRegistry` is snapshotted at
construction, so later registrations affect only new Sessions. See
[operators](operators.md#register-and-match).

A Session holds one submitted update at a time and is not thread-safe.
Separate Sessions may drive separate Stages. Closing a Session waits for
outstanding work, releases the reservation, and performs no Stage writes:
generated attributes stay on the recipe prims.

## Update and commit

When using Session directly, the application controls each ordinal and
write-floor advance:

1. Populate or edit recipe inputs, then commit the input ordinal.
2. Lease scientific data with `data_source.at_time(time_seconds)`.
3. Call `session.update(input_ordinal=..., output_ordinal=..., frame_data=...)`.
4. Keep recipe inputs and referenced scene prims stable until `update.wait()` returns.
5. Inspect the report and commit the output ordinal only if `report.committable`.
6. Let the renderer consume the committed output, then start the next cycle.

The input ordinal is a minimum committed floor, not a request for historical
values: a read may observe newer committed state, which is why inputs must stay
stable during the update. The output ordinal must exceed both the input ordinal
and the current write floor. Session never advances the write floor. Do not
step the renderer while an update is in flight or its output is uncommitted;
keep displaying the previous frame until the output is committed.

For frame-driven hosts, `update.done(timeout=0.0)` polls for completion and a
positive timeout waits at most that long. Polling neither consumes the report
nor frees the Session's update slot: call `wait()` once `done()` returns true,
or `UpdateInFlightError` follows on the next submission. `wait()` blocks to a
terminal outcome, is idempotent, and returns the same report on repeated calls.
There is no cancellation. Buffer edits made while an update runs and submit
them together in the next one; the [CAE Viewer](../examples/cae-viewer/README.md)
does this so a gizmo drag never queues more than one evaluation.

### The update loop

`ovcae.app.UpdateLoop` handles editing, evaluation, and commits for an
interactive application. It creates the Session, owns the ordinal from the
one you populated at, and takes edits as `{(prim path, property): value}`
batches. Recipe edits go
through `request`; each submit writes the batch at the next ordinal,
advances the floor, and starts an update. Presentation edits, a camera move
or a material binding, go through `write`, which advances the floor with no
update. The application decides which is which. Values are written with the
type the committed attribute or the prim's typed schema declares, with
explicit types or inference available when neither declares one.

Call the loop from one application thread; other threads can queue edits for
it to collect. This frame-loop fragment assumes a populated Stage committed
at ordinal 1, an open source, and an attached renderer. The host supplies `ui`,
`clock.now` in seconds, `running`, and `render_products`, a set of authored
render-product paths:

```python
from ovcae.app import UpdateLoop

loop = UpdateLoop(stage, source, ordinal=1, device="cuda:0", time_code=lambda: clock.now)
loop.update()  # evaluate the populated scene and commit

while running:
    loop.request(ui.take_recipe_edits())  # queued UI edits, coalesced latest-per-key
    report = loop.pump()  # commit a finished update, or submit the pending batch
    if report is not None:
        ui.show(report)
    if loop.evaluating:
        # Keep displaying the previous frame; do not step the renderer yet.
        loop.await_evaluation(0.005)  # cede the GIL to the update worker
        continue
    loop.write(ui.take_camera_edits())  # presentation: floor advances, no update
    renderer.step(render_products=render_products, delta_time=0.0, ordinal=loop.ordinal)
loop.close()
```

`pump` polls evaluation and commits a finished update before submitting the
next batch, so the renderer sees each result before the following one starts.
`update(edits)` is the synchronous form, `flush()` waits for the in-flight
update, `invalidate()` forces an update on the next pump after new data was
published, and `time_code` is the scientific time in seconds each update
evaluates at. The [package reference](packages.md#ovcaeapp) lists the whole
surface;
`ovcae.stream.LiveView` runs the loop on a render thread with mailboxes in
front and OVStream behind, and the CAE Viewer runs it inside omni.ui's frame
loop.

## Read the report

`UpdateReport` is immutable and carries `input_ordinal`, `output_ordinal`,
`operators`, `diagnostics`, and `committable`. Each `OperatorResult` has
`operator_path`, `status`, an optional `reason`, and `product_paths`.

| Status | Meaning |
| --- | --- |
| `succeeded` | The recipe was evaluated and its product published |
| `unchanged` | The committed product was reused without writes; `product_paths` is empty |
| `skipped` | Evaluation was suppressed, for example by visibility or the enabled flag |
| `failed` | The recipe could not be evaluated or published; see its diagnostics |
| `removed` | A previously known recipe disappeared and its cleanup was processed |

Each `Diagnostic` has `severity` (`info`, `warning`, `error`), `code`,
`message`, and an optional `operator_path`. `committable` answers one question:
is it safe to advance the write floor to the output ordinal? A committable
report can still contain failed operators when OVCAE safely cleared their stale
products. Use per-operator results to decide whether your application accepts
the outcome.

Invalid submissions raise synchronously: `OrdinalError`, `FrameDataError`,
`SessionClosedError`, and `UpdateInFlightError`. A failure that prevents a
reliable outcome raises `UpdateError` from `wait()`; its `report` attribute
carries any reliable partial result. If a report is not committable, do not
advance the write floor to that ordinal. If a consistent Stage cannot be
recovered from the diagnostics, rebuild the Stage and Session from authored
inputs.

## Product ownership and visibility

Session replaces only the geometry and primvars it generated. Recipe schemas,
relationships, transforms, material bindings, visibility, and every other
application-owned property survive replacement. Do not write Session-owned
output attributes or mutate published buffers in place. Double-precision output
primvars and widths are converted to float32 on the producing device so the
bundled materials can read them; float32 and integer primvars keep their types.

Hidden recipes keep their geometry and skip evaluation, scientific loading, and
validation, so hidden status says nothing about a recipe's validity. After
showing a recipe, run and commit an update before rendering it. Hiding an input
prim, such as a seed `Points`, does not disable a visible operator that uses it.

Disabled, removed, invalid, and failed recipes have their generated attributes
cleared. Glyphs retains owned empty instance arrays because deleting them can
leave stale rendered instances; its scientific primvars are removed. Other product
types delete their generated attributes. To clear products before shutting down,
disable or remove the recipes
and run one more update at a new output ordinal; closing the Session alone
writes nothing.

## Change detection

Session can reuse an unchanged committed product before loading any scientific
arrays. Faces, Points, Glyphs, Iso Surface, and Planar Slice qualify; Streamlines
executes on every update because scene-authored seeds have no comparison
identity. When an operator does run, it computes a complete current product.

| Condition | Behavior |
| --- | --- |
| Same held USD sample or published memory dataset, unchanged recipe, valid committed product | Reuse |
| A recipe input property written or deleted, including rewriting an identical value | Execute |
| Relevant scientific identity changed, including a field-only change | Execute the whole operator |
| Product attributes missing, modified, or otherwise invalid | Execute |
| Scientific comparison evidence unavailable, such as a custom source without identities | Execute |
| Previous result was an intentionally empty product | Execute again |

USD identities are coarse and dataset-wide: a temporal change to an unselected
field can invalidate a dataset, while changes to unrelated datasets do not.
Memory sources compare publish generations, so a replacement changes every
component's identity for that dataset sample.
Comparisons never hash buffer contents, so in-place mutation is not detected.

A completed evaluation becomes a reuse baseline only after Session observes
that its output ordinal was committed and validates the generated attributes.
Calling `wait()` alone does not establish a baseline, and a failed publication
or cleanup never does. Hidden operators keep any invalidation observed while
hidden, which is why showing one requires an update and commit before rendering.

### History retention

Recipe and product checks compare against OVStage's retained native history,
measured in commits, not wall-clock time. Camera and presentation commits
advance that history even though they do not touch recipes. Session rolls its
checkpoints forward on every unchanged check, including checks of hidden
recipes, without executing operators or rewriting products.

If the interval since the previous check extends beyond the retained history,
Session assumes the expired portion contained no changes and examines only the
retained portion. A relevant write confined to the expired portion goes
undetected and leaves stale geometry; expiry alone does not force
recomputation. Schedule updates often enough, in commits, that this cannot
happen. There is no wall-clock polling interval that guarantees correctness.
Previously observed changes remain invalidating, and missing attributes or
failed reads still force execution.

## Render the results

OVCAE writes standard geometry, so any OVStage consumer can render it. For
OVRTX, register schema paths and call `ovcae.setup()` before creating the
renderer, and create the renderer before populating the Stage, so the MDL SDK
finds the bundled `cae/mdl/basic.mdl` materials and colormap textures.
Recipes authored with the builders bind those materials and read the selected
field primvar through a scalar colormap; the shader needs an
authored `color_range` because Session does not evaluate automatic rescale
ranges. See the [authoring guide](authoring-guide.md#materials-and-colormaps).

The [CAE Viewer](../examples/cae-viewer/README.md) is the reference for a
rendering application: it runs the [update loop](#the-update-loop) inside
omni.ui's frame loop, pins OVRTX to the compute GPU, presents the previous
frame while an output is uncommitted, and reattaches the renderer when a
`Points` product grows from an empty buffer. For a browser instead of a
desktop window, `ovcae.stream.LiveView` streams the render product over
WebRTC; [tutorial 2](../tutorials/02_interactive_stream.py) shows it.
