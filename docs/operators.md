# Operators

The first half of this guide is the reference for the six built-in operators:
what each one computes and where its limits are. The second half covers
custom operators: declaring inputs, registering the class, reading the scene,
asking what changed, and returning a product.

## Built-in operators

The [recipe builders](authoring-guide.md#create-recipes) author the prim type,
the operator schema, and the selection schemas together, which is the easiest
way to get a valid recipe. All six operators run through the same Session
pipeline, with GPU processing provided by Warp SimData.

| Operator | Schema | Recipe and product type | Inputs and behavior |
| --- | --- | --- | --- |
| Faces | `CaeVizFacesAPI` | `Mesh` | Extracts external or all element faces from `source`; publishes selected node fields as `vertex` primvars and element fields as `uniform` primvars |
| Points | `CaeVizPointsAPI` | `Points` | Publishes `source` nodes with selected fields converted to nodes; can exclude nodes unused by cells, enforce `maxCount` deterministically, and publish a widths field or constant width |
| Glyphs | `CaeVizGlyphsAPI` | `PointInstancer` | Instances one authored prototype at selected source nodes, with scalar/vector scales, directions or quaternions, and scientific primvars |
| Iso Surface | `CaeVizIsoSurfaceAPI` | `Mesh` | Contours `source` on the `contour` field at `isoValue`; requires `colors` and publishes selected fields |
| Planar Slice | `CaeVizPlanarSliceAPI` | `Mesh` | Slices `source` with free or axis-aligned planes positioned by the recipe transform; requires `colors` |
| Streamlines | `CaeVizStreamlinesAPI` | `BasisCurves` | Integrates the `velocities` field of `source` from scene-authored `seeds`; publishes optional colors, widths, and integration times |

Every recipe also carries `CaeVizOperatorAPI`, whose `enabled` attribute
controls evaluation and product cleanup. Visibility suspends evaluation while
preserving the product; see the
[integration guide](integration-guide.md#product-ownership-and-visibility).

**Fields.** `CaeVizFieldSelectionAPI:<role>` selects backing arrays by
`fieldNames` from the dataset role named in `input` (default `source`); `mode`
and `componentIndex` control interpretation. The role name becomes the primvar
name. Derived surfaces and curves publish fields sampled or gathered onto their
generated geometry. Double-precision primvars and widths are published as
float32 for material lookup.

**Coordinates.** Generated points are local to the recipe prim. Planar Slice
positions its planes with the recipe's captured local transform; in `free` mode
the rotated local +Y axis is the plane normal, and the axis modes select the
corresponding axis-aligned planes. Multiple slice surfaces are combined on the
producing device. Streamlines applies the seed prim's captured local transform
before integration. Transforms on dataset holders or descriptors are not
applied to scientific arrays, and general dataset transform preprocessing is
not supported.

**Integration.** Streamlines uses Warp SimData's default stopping conditions
together with the authored step and tolerance settings. The schema's
`threshold` attribute, displayed as Velocity Threshold, remains for scene
compatibility but is not used.

**Animated streamlines.** Streamlines publishes vertex integration `times` and
uniform `rnd` primvars for the AnimatedStreaks and AnimatedDots materials.
Each curve receives a deterministic random phase in [0, 1), staggering heads
without frame-to-frame jitter. Phases remain stable for the same output curve
ordering; topology changes can reassign them. A selected `rnd` field takes
precedence over the generated phases. The application must advance renderer
time for the animation to move.

**Repeated updates.** Faces, Points, Glyphs, Iso Surface, and Planar Slice can reuse an
unchanged committed product. A relevant field change re-executes the whole
operator. Streamlines executes on every update because scene seeds have no
comparison identity, and an `EmptyProduct` result executes again. By contrast,
Glyphs publishes empty instance arrays as a real, reusable product. See
[change detection](integration-guide.md#change-detection).

**Limits.** NVIDIA IndeX, XAC, and Flow are
not supported; a schema's presence in the bundle does not imply a runtime
operator. Automatic material range rescaling is not evaluated, so authors
supply a known `color_range`, as described in the
[authoring guide](authoring-guide.md#materials-and-colormaps).

## Glyphs

`create_glyphs` authors a Sphere, Cone, Arrow, or internally referenced Custom
prototype. Arrows extend from the origin to +X. All generated positions are in
the recipe's local coordinates. Use `use_cell_points=False` for point clouds;
otherwise only cell-connected nodes are selected. Cell fields are converted to
nodes before deterministic stride sampling by `max_count` (0 means all).

The optional `scales` field has one or three components and replaces the constant
`scale`. `CaeVizFieldMappingAPI:scales` applies a clamped, componentwise mapping
when its domain is valid; an invalid domain leaves values unchanged. A degenerate
domain maps to the lower output endpoint. Automatic domain rescaling remains
unsupported, so supply a known domain when using scale mapping.

The optional `orientations` field has three components for directions or four
for quaternions in `(x, y, z, w)` order. The compatibility token `eulerAngles`
means **directions from +X**, matching Kit-CAE; it does not mean Euler angles.
Zero directions use identity and exactly -X produces a valid 180-degree rotation.
Quaternion inputs must be unit quaternions. Publication uses standard `quath[]`
rotations; conversion and half-precision packing stay on the producing device.
Removing the field restores identity orientations. Other selected fields,
including colors and opacity, become per-instance `vertex` primvars.

Session publishes `positions`, `protoIndices` (all zero), `scales`, `orientations`,
and scientific primvars. Empty inputs publish consistently empty arrays. Disabling
a recipe or clearing a failed product retains owned empty transform arrays to
ensure renderers remove previous instances; scientific primvars are deleted.
Applications own `prototypes`, its hierarchy, transforms, and material bindings.
Populate prototype edits through USD; do not rewrite OVStage's flattened runtime
relationship as an authoring operation. This operator uses one source prototype,
which may contain multiple renderable children.

## Write a custom operator

An operator is an ordinary class decorated with `@ovcae.operators.operator`. It
reads a bounded view of the scene, loads scientific datasets, and returns
geometry for Session to publish. This Faces operator declares its inputs and
therefore participates in change detection:

```python
import warp as wp
from warp_simdata.operators import element_faces

from ovcae import operators
from ovcae.operators import utils as op_utils


@operators.operator(
    required_schemas={"CaeVizFacesAPI", "CaeVizDatasetSelectionAPI:source"},
    optional_schemas={"CaeVizFieldSelectionAPI:*"},
    dataset_components={"source": {"topology", "geometry"}},
    priority=100,
)
class CustomFaces:
    def execute(self, context):
        if context.prim.type_name != "Mesh":
            raise operators.InvalidRecipe("Faces requires a Mesh recipe")
        source = op_utils.get_input_dataset(context, role="source")
        external_only = context.prim.get_attribute("cae:viz:faces:externalOnly").get()
        surface = element_faces.compute(source, external_only=bool(external_only))
        wp.synchronize_device(context.device)
        return op_utils.mesh(
            points=surface.handle.points,
            face_vertex_counts=surface.handle.face_vertex_counts,
            face_vertex_indices=surface.handle.face_vertex_indices,
        )
```

This example returns geometry only; it declares selected fields as inputs but
does not publish them. The
[built-in Faces operator](../src/ovcae/operators/_builtins/faces.py) shows field
gathering and primvars, [tutorial 5](../tutorials/05_extend_it.py) writes a
complete operator with a schema of its own in Warp, and
[appendix A3](../tutorials/A3_vtk_operator.py) wraps a VTK filter, including
the copies between device and host.

Operators never import OVStage, never see the live Execution Stage, and never
write products themselves. Session creates one instance per bound recipe and
calls `execute` synchronously on its update worker.

## Give an operator its own settings

A built-in operator's settings, such as `cae:viz:faces:externalOnly`, live in
the bundled codeless schemas. A custom operator that needs a setting defines a
schema of its own: a codeless schema folder holding `plugInfo.json` and
`generatedSchema.usda`, as `usdGenSchema` writes them. Register the folder
once, before the process creates any stage:

```python
import ovcae

ovcae.setup()
ovcae.register_schema_paths("/path/to/warp_by_vector_schema")  # -> ("CaeVizWarpByVectorAPI",)
```

That one call registers the schema with usd-core, so the attribute has a type
and a fallback when authored, and with OVStage, so population copies it from
the scene. Nothing is declared to OVCAE separately: the Session asks usd-core's
schema registry what each applied schema declares when it builds an operator's
scene view, and `ovcae.ovstage_population_desc(...)` populates every schema
registered this way (pass `schemas=` to choose them explicitly).

The timing matters. usd-core and OVStage each build their schema registry
once, on the first schema read in the process, and a plug-in registered after
that is never seen. Creating a stage, applying a schema, or touching a typed
class all count as a schema read; importing `ovcae.authoring` does not.
Register custom schemas right after `ovcae.setup()`, before any of those. When
the Session meets an applied `CaeViz*` token usd-core has no definition for,
it reports `SCHEMA_NOT_CAPTURED` naming the token and the operator cannot
read the attribute, which fails as outside its input scope. Tutorial 5 walks
through the registration with a working operator.

Applied schemas usd-core itself ships, such as `MaterialBindingAPI`, are not
part of an operator's input scope: they describe presentation, and editing
them never re-executes a recipe.

## Register and match

Operators are never discovered implicitly. Build a registry, register your
class, and hand it to the Session. Here `stage` is the populated Execution
Stage from the [integration guide](integration-guide.md#populate-the-stage):

```python
import ovcae
from ovcae import operators

registry = operators.OperatorRegistry.with_builtins()
registry.register(CustomFaces)
with ovcae.Session(stage, device="cuda:0", operators=registry) as session:
    ...
```

Decoration validates the declaration; registration is a separate, explicit
step. `OperatorRegistry()` starts empty and `with_builtins()` starts from the
standard set. `register_module(module)` registers every decorated class in a
module; there is no unregister. An invalid declaration raises
`OperatorRegistrationError` at decoration or registration time. Session
snapshots the registry at construction, so later changes affect only new
Sessions.

Matching uses applied schemas. Every recipe needs `CaeVizOperatorAPI`; all
`required_schemas` selectors must match, and `optional_schemas` describe
supported additions. A selector is an exact applied token or a multiple-apply
family ending in `:*`. Among matching operators the highest `priority` wins,
and equal priorities produce an ambiguity diagnostic; the example's priority of
100 overrides the built-in Faces operator. Prim types are not part of matching,
so validate the type in `execute` and return a product of the matching kind.
Unknown or ambiguous visible recipes produce diagnostics; hidden recipes defer
matching and validation until shown.

## Declare and read inputs

`dataset_components` declares comparison dependencies for each dataset role:
`topology`, `geometry`, or both; an empty set declares only the schema-selected
fields. These are not independent loading controls: scientific dataset
resolution materializes topology and geometry, so both become effective
dependencies. A `"*"` entry supplies a default for all roles and exact entries
replace it. Every role the recipe actually matches must be covered, and
optional roles absent from the recipe make no request.

With declared inputs, `op_utils.get_input_dataset(context, role="source")`
returns the `warp_simdata.Dataset` for a role. Field requests derive from the
recipe's `CaeVizFieldSelectionAPI` instances; do not repeat them as eager
`fields`, `topology`, or `geometry` arguments. Every matched input counts as a
dependency even when an execution branch does not load it.

Operators without `dataset_components` use the eager helper arguments instead.
They execute on every update and cannot query `context.changes` or the current
samples.

`context` exposes `prim`, `stage`, `input_ordinal`, `output_ordinal`,
`device`, and, for declared inputs, `changes`. Inputs must stay stable until the
update finishes; the input ordinal is a committed floor, not a request for
historical values.

## Read the scene

`context.stage` and `context.prim` are read-through handles onto a bounded part
of the Execution Stage. OVCAE plans which properties an operator may read from
the recipe's applied schemas and its selected relationship targets, then reads
values on demand. It does not copy the recipe or retain a historical snapshot.

| Handle | Operations |
| --- | --- |
| `StageView` | `ordinal`, `get_prim_at_path(path)` |
| `PrimView` | `path`, `stage`, `type_name`, `applied_api_schemas`, `has_api(schema_name, *, instance=None)`, `api_instances(schema_name)` |
| `PrimView` | `attribute_names()`, `relationship_names()`, `get_attribute(name)`, `get_relationship(name)` |
| `AttributeView` | `name`, `type_name`, `get()` |
| `RelationshipView` | `name`, `targets` |

Type and applied-schema metadata are read during planning; attribute values and
relationship targets are read when requested. `attribute_names()` lists the
allowed properties, including ones that are not authored. For an allowed
attribute, `get()` returns the authored value, the schema default when absent,
or `None` when neither exists. An absent allowed relationship has empty
targets, and a planned target that does not exist returns `None` from
`get_prim_at_path()`.

Handles are valid only inside `execute`. Access outside the planned scope
raises `OutOfScopeError`, and use after execution raises `ExpiredViewError`. Do
not retain contexts, handles, Frame Data views, or scientific intermediates
between calls. Returned values own their storage: scalars and tokens are Python
values, multi-value properties are tuples, and scene points are owned NumPy
arrays of shape `(N, 3)`. You may keep an owned value after its handle expires;
mutating it does not write the Stage.

Scene points selected from a `UsdGeomPointBased` prim, such as streamline
seeds, arrive through this interface rather than through Frame Data.
`op_utils.get_local_transform(prim)` reads the populated `omni:xform` local
matrix, which OVStage population composes from the USD transform operations,
and converts USD's row-vector convention for computational use. Direct Stage
producers must author that property with MATRIX semantics; absence means
identity.

Scene access does not support arbitrary traversal, layers or composition,
authoring, or the raw native Stage. Properties from bundled and registered
custom schemas, plus explicitly planned relationship targets, are in scope.
An unknown schema may appear in metadata but does not extend property access.

## Detect changes

Operators with declared inputs can query the same baseline Session uses to
decide whether to run them at all. Inside `execute`, for a recipe with a
`source` dataset and a `colors` field selection:

```python
changes = context.changes
recipe_changed = changes.recipe_changed()
geometry_changed = changes.input_changed("source", components={"topology", "geometry"})
colors_changed = changes.input_changed("source", fields={"colors"})
```

`recipe_changed()` checks the declared recipe properties and selected scene
dependencies under the history policy in the
[integration guide](integration-guide.md#history-retention). `input_changed()`
is true when any requested component or field differs. Field names are output
roles, not backing array names. The calls load no arrays and write nothing.

| Query condition | Result |
| --- | --- |
| No valid committed baseline, or comparison evidence unknown | `True` |
| Comparable identities unchanged | `False` |
| Invalid role, unavailable component or field, or empty selection | `InvalidRecipe` |
| Bare string instead of a collection of names | `InvalidRecipe` |
| Operator without declared inputs asks for changes | `InvalidRecipe` |
| Saved context or `Changes` view used after execution | `ExpiredViewError` |

These queries let an operator skip parts of its own work, but there is no
intermediate-cache contract: every invocation still returns a complete current
product. `op_utils.get_input_samples(context, role="source")` reads the current
provider metadata without loading arrays and returns `topology`, `geometry`,
and `fields` sample descriptions, or `None` where evidence is unavailable. USD
sources describe `DatasetSample(time, granularity="dataset")` with time in
seconds, or `time=None` for default-only data. Sample descriptions are informational
and are not complete cache keys.

## Return products

The helpers in `ovcae.operators.utils` build the return value:

| Helper | Purpose |
| --- | --- |
| `mesh(points=..., face_vertex_counts=..., face_vertex_indices=..., primvars=...)` | Mesh product |
| `curves(points=..., curve_vertex_counts=..., widths=..., primvars=...)` | BasisCurves product; `widths` is a required `Primvar` |
| `instancer(positions=..., proto_indices=..., scales=..., orientations=..., primvars=...)` | PointInstancer arrays; application-owned prototypes; float16 xyzw orientations |
| `points(points=..., widths=..., primvars=...)` | Points product; `widths` is a required `Primvar` |
| `primvar(name, values, interpolation)` | One primvar description, also used for `widths` |
| `mesh_from_surface`, `mesh_from_surfaces` | Convert Warp SimData surfaces, including supported fields |
| `get_field_selections(context, *, input=None, exclude=())` | The recipe's field selections as `FieldSelection` records with `output_name`, `field_names`, `mode`, `component_index`, and `input` |
| `probe_fields(source, result, *, selections)` | Sample those selections onto a computed result as primvars |
| `get_local_transform(prim)`, `resolve_widths(context, *, result, role, default)` | Read a scene transform; resolve a widths field or constant into a `Primvar` |
| `empty(reason=...)` | A successful empty result that removes prior generated geometry |

Product arrays may stay on the producing device; synchronize computation before
returning, and Session retains the publication buffers until the writes
complete. Return points in the recipe's local coordinate system. Mesh, Curves,
Points, and Instancer products require `Mesh`, `BasisCurves`, `Points`, and `PointInstancer` recipes
respectively. Returning `None` is an error; return `empty()` for an intentional
empty result. There is no multi-product return, no operator close hook, and no
API for retaining intermediates between calls.

Raise `InvalidRecipe` for unusable authored input. Session reports the failure,
removes the recipe's stale output, and lets independent operators continue.

## Test an operator

Register the operator in a registry, author a recipe with the
[builders](authoring-guide.md#create-recipes), populate an OVStage as in the
[tutorial](getting-started.md), and run an update with a
`MemoryDataSource`. Assert on the operator's status and diagnostics in the
report and on the committed attributes of the recipe prim. The repository's
[authoring round-trip test](../tests/test_authoring_roundtrip.py) builds its
scene this way, and the [custom operator test](../tests/test_custom_schema.py)
shows the assertions against a registered custom operator.
