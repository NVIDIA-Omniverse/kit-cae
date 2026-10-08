# Packages and public API

The `ovcae` wheel includes the runtime, schema bundle, material library, recipe
authoring, readers, application update loop, and browser streaming.
Imports are layered: importing `ovcae` loads only the standard library, and
authoring scenes does not load OVStage or Warp. The base install still includes
usd-core and the declared runtime dependencies. Extras add dependencies for
native reading and streaming. The table distinguishes installation requirements
from what each import loads; the linked guides show the APIs in use.

| Package | Purpose | Imports at load | Needs |
| --- | --- | --- | --- |
| [`ovcae`](#ovcae) | Evaluate recipes and publish products into an OVStage | Standard library only; OVStage, Warp, Warp SimData, and NumPy load on first use | Base wheel |
| [`ovcae.readers`](#ovcaereaders) | Read datasets in their native formats through the [CAE OpenUSD Plugins](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins) | usd-core and the CAE OpenUSD Plugins | `usd` extra |
| [`ovcae.authoring`](#ovcaeauthoring) | Create, edit, and validate recipe scenes | usd-core | Base wheel |
| [`ovcae.schemas`](#ovcaeschemas) | Schema bundle, typed schema classes, usd-core schema registry access | Standard library only | Base wheel |
| [`ovcae.materials`](#ovcaematerials) | Bundled MDL materials and colormap textures | Standard library only | Base wheel |
| [`ovcae.app`](#ovcaeapp) | The edit, update, commit loop an interactive application runs, for any renderer | Standard library and the runtime package | Base wheel |
| [`ovcae.stream`](#ovcaestream) | Render on a thread and stream OVRTX frames to a browser over WebRTC | Standard library and NumPy; OVStream, OVRTX, and Warp on first use | `stream` extra and an OVRTX wheel |

Nothing is registered as a side effect of importing `ovcae`, `ovcae.schemas`,
or `ovcae.materials`. Importing `ovcae.authoring` registers the schemas with
usd-core and importing `ovcae.readers` registers the CAE file formats, because
usd-core must know both before the first stage is created.

Readers define how scientific datasets are represented and resolved. Each
provides schemas, a Data Source, and any needed authoring helpers. OVCAE includes
`ovcae.readers`; custom readers can ship as separate packages, as the
[readers guide](readers.md) describes.

## ovcae

The runtime. An application populates an `ovstage.Stage`, opens a Data Source,
and asks a `Session` to evaluate the committed recipes into a new output
ordinal. See the [integration guide](integration-guide.md) for the full
lifecycle and the [getting-started tutorial](getting-started.md) for a minimal
program.

This fragment needs the `usd` extra, an RTX-capable NVIDIA GPU available as
`cuda:0`, and an `ovstage.Stage` named `stage`, populated from `scene.usda`
and committed at ordinal 1. Call `ovcae.setup()`
before creating the stage, following the [initialization order](installation.md#initialize-your-application).

```python
import ovcae
import ovcae.readers

with ovcae.readers.UsdDataSource.open("scene.usda") as source:
    with ovcae.Session(stage, device="cuda:0") as session:
        with source.at_time(0.0) as frame_data:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame_data).wait()
        if not report.committable:
            raise RuntimeError(report.diagnostics)
        stage.advance_write_floor(report.output_ordinal).wait()
```

**Process setup and population**

| Name | Purpose |
| --- | --- |
| `setup(*, strict=False)` | Import every installed reader, register the codeless schemas with usd-core and OVStage, and the material library with the MDL environment; idempotent |
| `load_readers(*, strict=False)` | The reader step of `setup()`: import each package in the `ovcae.readers` entry-point group and return the names that loaded |
| `register_schema_paths(*directories)` | The schema step of `setup()`: register codeless schema plug-in directories (the bundle by default) with usd-core and OVStage; returns the applied API schema identifiers they declare |
| `registered_schemas()` | Bundled schema identifiers plus custom identifiers registered through `register_schema_paths` |
| `register_material_library()` | The material step of `setup()`; returns the library path |
| `ovstage_population_desc(*, domains, schemas=None)` | Population descriptors that carry the registered schemas' properties into OVStage; `schemas` names them explicitly |
| `codeless_schema_path()` | Directory of the codeless schema plug-in; the same directory `ovcae.schemas.schema_resource_path()` returns |
| `__version__` | Distribution version string |

**Evaluation**

| Name | Purpose |
| --- | --- |
| `Session(stage, *, device="auto", operators=None)` | Borrows one Stage, evaluates recipes, writes products without committing; context manager |
| `Update` | Handle for one submitted evaluation: `done(timeout=0.0)` polls, `wait()` returns the report |
| `UpdateReport` | Immutable result: `input_ordinal`, `output_ordinal`, `operators`, `diagnostics`, `committable` |
| `OperatorResult` | Per-recipe outcome: `operator_path`, `status`, `reason`, `product_paths` |
| `Diagnostic` | `severity`, `code`, `message`, optional `operator_path` |

**Data Sources**

| Name | Purpose |
| --- | --- |
| `MemoryDataSource(mapping)` | Serve Warp SimData datasets from memory; `publish` replaces a dataset or adds a time sample, `remove` forgets one |
| `CompositeDataSource(*sources)` | Route each key to the one child that claims it; requires at least one `DataSource` |
| `DataSource` | Base class every reader implements; `at_time(seconds)` yields `FrameData` |
| `FrameData` | Opaque, time-bound lease passed to `Session.update`; never constructed directly |

File-backed sources live in reader packages; see [`ovcae.readers`](#ovcaereaders).

**Exceptions**

| Name | Raised when |
| --- | --- |
| `OvcaeError` | Base class for the runtime errors listed here; authoring and schema errors derive from `ValueError` |
| `OrdinalError` | An update violates the ordinal contract |
| `FrameDataError` | Frame Data is invalid for a new update |
| `UpdateInFlightError` | The Session already has an update awaiting `wait()` |
| `SessionClosedError` | An operation needs an open Session |
| `StageSessionConflictError` | A live Session already publishes to the Stage |
| `UpdateError` | A process-wide failure prevents a reliable outcome; `report` carries any partial result |
| `DataSourceOpenError`, `DataSourceClosedError` | A source cannot be opened, or is used after closing |
| `ActiveFrameDataError` | A source is closed while an application lease is active |
| `DatasetResolutionError` | Frame Data cannot resolve a requested dataset |

**Submodules**

| Module | Contents |
| --- | --- |
| `ovcae.operators` | The operator extension interface: the `@operator` decorator, `OperatorRegistry` with `register`, `register_module`, `snapshot`, and `with_builtins`, `OperatorContext`, `Changes`, `InvalidRecipe`, `OperatorRegistrationError` raised by invalid declarations, `AmbiguousOperatorError`, `OperatorDeclaration` |
| `ovcae.operators.utils` | Operator helpers: `get_input_dataset`, `get_input_samples`, `get_field_selections` returning `FieldSelection` records, `probe_fields`, `get_local_transform`, `resolve_widths`, `is_point_based`, and the product constructors `mesh`, `mesh_from_surface`, `mesh_from_surfaces`, `curves`, `points`, `instancer`, `primvar`, `empty` |
| `ovcae.scene` | The bounded scene view operators read through: `StageView`, `PrimView`, `AttributeView`, `RelationshipView`, and `SceneViewError` with `OutOfScopeError` and `ExpiredViewError` |
| `ovcae.data` | Building blocks for readers: `DataSource`, `CompositeDataSource`, `FrameData`, `DatasetSample`, `dataset_from_arrays`, `add_fields`, `held_time`, `resolve_asset_path` |

Custom operators are documented in [operators](operators.md); custom readers
in the [readers guide](readers.md#write-a-reader).

## ovcae.app

An optional helper for the edit, update, and commit cycle, independent of the renderer. An
application populates a Stage and opens a Data Source as before, then hands
both to an `UpdateLoop`, which creates the Session and owns the ordinal from
there. Edits are `{(prim path, property): value}` batches of two kinds, and
the application says which: `request` takes recipe edits, anything an
operator reads, written together at the next ordinal and followed by an
update; `write` takes presentation edits such as a camera move, written at
once with no update. Values are written with the type the committed
attribute or the prim's typed schema declares, or the type inferred from the
value. A renderer steps at `loop.ordinal`. The
[integration guide](integration-guide.md#the-update-loop) shows it in a
frame loop; the [CAE Viewer](../examples/cae-viewer/README.md) and
[tutorial 2](../tutorials/02_interactive_stream.py) show it in use.

This fragment starts with an open `source` and a Stage committed at ordinal 1,
with a camera at `/World/Camera` and an Iso Surface recipe at `/World/Iso`.
`matrix` is the camera's 4×4 transform, supplied by the application.

```python
from ovcae.app import UpdateLoop

with UpdateLoop(stage, source, ordinal=1, device="cuda:0") as loop:
    loop.update()  # the first evaluation, committed
    loop.write({("/World/Camera", "omni:xform"): matrix})  # presentation: no update
    loop.request({("/World/Iso", "cae:viz:isoSurface:isoValue"): 250.0})  # recipe
    loop.pump()  # submit the queued recipe edits
    report = loop.flush()  # wait and commit; a frame loop would keep calling pump
```

| Name | Purpose |
| --- | --- |
| `UpdateLoop(stage, data_source, *, ordinal, device="auto", operators=None, time_code=0.0)` | Create the Session and take over ordinals from the committed `ordinal`; `time_code` is a number or a callable returning the scientific time each update evaluates at; context manager |
| `loop.ordinal`, `loop.last_report`, `loop.session` | The committed ordinal a renderer steps at, the most recent committed report, and the Session the loop created |
| `loop.request(edits, *, types=None)` | Buffer recipe edits, latest value per key, for the next submit. `types` overrides the USD type an edit is written as |
| `loop.write(edits, *, types=None)` | Write presentation edits at the next ordinal and advance the floor; raises `UpdateInFlightError` while an update runs |
| `loop.pending` | The buffered recipe edits the next submit writes |
| `loop.property_type(path, name)` | The USD type an edit is written as: from the committed attribute or the prim's typed schema; None when nothing declares it |
| `loop.value(path, name)` | The pending edit for the key if there is one, else the committed value |
| `loop.pump()` | Once per frame: commit a finished update and return its report, or submit the pending batch without waiting for evaluation |
| `loop.submit(*, wait=False)` | Write the pending batch at the next ordinal and start an update, synchronously with `wait`; None when nothing is pending |
| `loop.update(edits=None, *, types=None)` | Synchronously request, submit, and commit; the report |
| `loop.flush()` | Wait for the in-flight update and commit it |
| `loop.evaluating`, `loop.await_evaluation(budget)` | Whether an update is in flight, and a bounded wait that cedes the GIL to the worker |
| `loop.invalidate()` | Make the next `pump` run an update even without recipe edits, after new data was published for instance |
| `loop.close()` | Drop pending edits and close the Session; the Stage and Data Source stay with the application |
| `write_attribute(stage, path, name, value, ordinal, *, type_name)` | The typed OVStage write the loop uses: tokens, strings, relationships, scalars, vectors, matrices, and arrays by USD type name |
| `committed_type(stage, path, name, ordinal)` | The USD type name of a committed attribute, recovered from its native layout |

An uncommittable update raises `UpdateError` from `submit`, `update`, `flush`,
or `pump`, with the report attached; the committed ordinal does not move. The
loop runs on the caller's thread; `ovcae.stream` shows one way to put it on a
render thread with mailboxes in front. Which edits are presentation is the
application's call: the Session's change detection would skip unchanged
operators on a needless update, but Streamlines re-integrate on every update
because scene seeds carry no identity, and an update in flight holds the
renderer for a round trip.

## ovcae.stream

Tutorial 2's live view as a package: render continuously on a background
thread, stream each frame to a browser over WebRTC through
[OVStream](https://github.com/NVIDIA-Omniverse/ovstream), orbit a camera
from the browser's mouse, and apply edits between frames through an
`UpdateLoop`. Install the `stream` extra for OVStream; OVRTX is installed
separately, as the [installation guide](installation.md) describes. The
[tutorials README](../tutorials/README.md#streaming-to-the-browser) covers
the network requirements WebRTC brings.

| Name | Purpose |
| --- | --- |
| `LiveView(loop, renderer, *, render_product, camera, eye, target, gpu_index="0", signal_port=49100, public_host=None, fps=30, up_axis="Z")` | A render thread over an attached OVRTX renderer and a loop that has committed its first update; `start()` and `stop()`, or use as a context manager |
| `view.request(path, name, value)`, `view.write(path, name, value)` | Drop one recipe or presentation edit in the mailbox from any thread; the render thread hands each frame's batches to `loop.update` and `loop.write` |
| `view.wait_for_update(timeout=60.0)` | Block until every request so far has been written and evaluated; the latest report |
| `view.ordinal`, `view.last_report`, `view.orbit`, `view.error` | The loop's committed ordinal and report, the `OrbitCamera` mouse input drives, and the exception that stopped the thread, if any |
| `view.snapshot()`, `view.html()`, `view.width`, `view.height` | The current frame as RGBA, the browser client for `IPython.display.HTML`, and the render product's resolution |
| `OrbitCamera(eye, target, *, up_axis="Z")` | Yaw, pitch, and zoom around a target as a USD camera transform; `matrix()` |
| `webrtc_client_html(signal_port, width, height, *, server=None)` | The notebook-sized browser client for an OVStream WebRTC server |

## ovcae.readers

The first-party reader. A source file is payloaded onto a holder
prim, the CAE OpenUSD Plugins compose it into `OmniSciDataset` prims, and
`UsdDataSource` resolves recipe selections against them on a usd-core stage.
The `usd` extra supplies the plug-ins; usd-core is part of the base install.
Missing dependencies raise an `ImportError` naming the extra. The reader is
declared in the `ovcae.readers` entry-point group, so `ovcae.setup()` imports it
before the first stage exists. See the [readers guide](readers.md).

With a supported `simulation.cgns` containing at least one dataset in the
working directory:

```python
import ovcae.readers  # registers the CAE file formats with usd-core
from pxr import Usd

stage = Usd.Stage.CreateNew("scene.usda")
holder = ovcae.readers.import_dataset(stage, "/Data", "./simulation.cgns")
dataset = ovcae.readers.dataset_prims(stage, holder)[0]
stage.Save()
```

**Reading**

| Name | Purpose |
| --- | --- |
| `UsdDataSource.open(path)` | Compose the scene privately on a held-interpolation usd-core stage and resolve dataset keys against it |
| `UsdDataSource(stage)` | Resolve against an application-owned `pxr.Usd.Stage`, composed with the plug-ins registered and configured for held interpolation; see the [reader example](readers.md#read-at-evaluation-time) |

**Authoring holders**

| Name | Purpose |
| --- | --- |
| `import_dataset(stage, path, file_name, *, format=None, edit_target=None, **args)` | Payload one source file onto a holder prim and apply its format-arguments schema; raises `ValueError` on bad arguments |
| `dataset_prims(stage, holder)` | The `OmniSciDataset` prims composed beneath a holder |

## ovcae.authoring

Recipe construction on usd-core stages. It imports usd-core and the
`ovcae.schemas` and `ovcae.materials` bundles without loading OVStage, Warp,
or a reader. See the [authoring guide](authoring-guide.md).

This independent fragment assumes `scene.usda` already contains a dataset
descriptor at `/Data/Zone`. For native-data payloads, import `ovcae.readers`
before opening the stage.

```python
import ovcae.authoring as author  # registers schemas with usd-core
from pxr import Usd

stage = Usd.Stage.Open("scene.usda")  # holds a dataset prim at /Data/Zone
author.create_faces(stage, "/Surface", source="/Data/Zone", colors="Temperature", color_range=(290.0, 310.0))
stage.Save()
```

**Recipes**

| Name | Purpose |
| --- | --- |
| `create_faces`, `create_points`, `create_iso_surface`, `create_planar_slice`, `create_streamlines` | Author one recipe each: prim type, operator schema, selections, placeholder geometry, and optional materials |
| `create_glyphs(stage, path, *, source, shape="Sphere", prototype=None, ...)` | PointInstancer recipe with a Sphere, Cone, Arrow, or Custom prototype |
| `validate_recipe(prim)` | Structural checks for built-in recipes; custom operator validation belongs to the operator |
| `AuthoringError` | Arguments or typed access that cannot produce a valid recipe |

**Controls**

| Name | Purpose |
| --- | --- |
| `create_seed_points(stage, path, points, ...)` | Explicit streamline seeds as an invisible `Points` prim |
| `create_seed_sphere(stage, path, *, count=64, bounds=None, center=None, radius=None, ...)` | Seeds on a sphere sized from `bounds`, or from an explicit `center` and `radius` |
| `sphere_points(count, *, center, radius)` | The sphere positions without authoring a prim |
| `create_bounding_box(stage, path, bounds, ...)` | A wireframe box; a reference, not an operator |
| `Bounds` | Axis-aligned bounds with `center`, `diagonal`, and `is_empty` |

**Materials**

| Name | Purpose |
| --- | --- |
| `create_mdl_material(stage, path, sub_identifier, *, colormap, color_range, enable_coloring)` | Define a material from the bundled MDL module |
| `bind_material(prim, material)` | Bind with `weakerThanDescendants` strength |
| `setup_scalar_materials(prim, *, sub_identifiers, bound, colormap, color_range, ...)` | The scalar-coloring material setup the builders perform |
| `setup_width_mapping(prim, *, width, mapped_range)` | The `widths` role scaffolding for Points and Streamlines |
| `wire_rescale_range(prim, role, *, includes, enable_includes)` | Apply `CaeVizRescaleRangeAPI:<role>` wiring |
| `colormap_asset(colormap)` | Search-path-relative identifier for a bundled colormap |

**Schemas and registration**

| Name | Purpose |
| --- | --- |
| `ovcae.authoring.schemas` | Typed schema classes bound to `Usd.Prim`: `FacesAPI`, `PointsAPI`, `FieldSelectionAPI`, and the rest, plus `UsdPrimBackend`, `coerce`, and `schema_from_registry` |
| `SchemaError` | Unknown schema, wrong prim type, or invalid value in typed access |
| `register_schemas()` | The registration performed on import, for callers that need to repeat it |

## ovcae.schemas

The codeless OmniCaeViz schema bundle as importable resources, typed schema
classes whose behavior is independent of any scene library, and access to
usd-core's schema registry for every other applied API schema. Importing it
loads nothing but the standard library; the registry functions import `pxr`
on call. See the
[authoring guide](authoring-guide.md#schema-resources-and-registry).

```python
from ovcae.schemas import typed as T

print(T.FacesAPI.external_only.default)
print(T.StreamlinesAPI.can_apply_to("BasisCurves"))
```

**Resources and registry**

| Name | Purpose |
| --- | --- |
| `schema_resource_path()` | Directory holding `schema.usda`, `generatedSchema.usda`, and `plugInfo.json` |
| `schema_source_path()` | The canonical `schema.usda` |
| `register_usd_plugin(*directories)` | Register codeless plug-in directories (the bundle by default) with usd-core's `Plug` registry, before the process's first schema read; returns the applied API schema identifiers they declare |
| `schema_from_registry(name)` | Typed class for any applied API schema usd-core knows, built once from the registry |
| `find_schema(name)` | The same, or `None` when usd-core does not know the schema |
| `is_usd_builtin_schema(name)` | Whether usd-core itself ships the schema |
| `plain_value(value, type_name)` | Reduce a pxr value to plain Python |
| `OPERATORS`, `OperatorInfo` | The operator schemas, their prim types, and the applied schemas each needs |
| `SchemaError` | Unknown names and invalid usage |

**Typed classes**

| Name | Purpose |
| --- | --- |
| `ovcae.schemas.typed` | One class per bundled schema, built from usd-core's registry on first access: `OperatorAPI`, `FacesAPI`, `PointsAPI`, `GlyphsAPI`, `IsoSurfaceAPI`, `PlanarSliceAPI`, `StreamlinesAPI`, `DatasetSelectionAPI`, `FieldSelectionAPI`, `RescaleRangeAPI`, and the rest; importing it registers the bundle with usd-core |
| `AppliedSchema`, `SingleApplyAPI`, `MultipleApplyAPI` | Base classes; instances wrap one prim and, for multiple-apply schemas, one instance name |
| `Attribute`, `Relationship` | Descriptors that read with schema defaults and validate on write |
| `PrimBackend`, `register_backend(adapter)`, `backend_for(prim)` | The protocol that binds the typed classes to a concrete prim type; `ovcae.authoring.schemas` registers the `Usd.Prim` backend |
| `make_schema_class(...)` | Build a typed class from a schema description, as `schema_from_registry` does |

The bundle declares every OmniCaeViz schema, including ones without a
runtime operator, such as the IndeX volumes. Whether a schema
executes is documented in [operators](operators.md#built-in-operators).

## ovcae.materials

The CAE MDL module and colormap textures, laid out so that scenes authored for
Kit-CAE resolve the same asset identifiers. Importing it has no side effects.
See the [authoring guide](authoring-guide.md#materials-and-colormaps).

```python
import ovcae.materials

ovcae.materials.register_material_library()  # before constructing the renderer
```

| Name | Purpose |
| --- | --- |
| `register_material_library(environ=None)` | Prepend the library to `MDL_USER_PATH` in `os.environ` or the given mapping; returns the path |
| `material_library_path()` | The directory MDL search paths must include |
| `colormap_paths()` | The bundled colormap PNG textures, sorted by name |
| `MDL_MODULE` | The module identifier scenes reference, `cae/mdl/basic.mdl` |
| `MDL_SEARCH_PATH_VARIABLE` | The environment variable name, `MDL_USER_PATH` |

The module defines the `ScalarColor`, `UnlitScalarColor`, and
`AnimatedStreaks` materials. The textures are `afmhot`, `cividis`, `coolwarm`,
`gist_gray`, `gist_rainbow`, and `turbo`, rasterized from Matplotlib's colormap tables;
see [NOTICE.md](../NOTICE.md).
