# Authoring guide

`ovcae.authoring` creates and edits recipe scenes on usd-core stages, so you
build scenes in Python instead of hand-editing `.usda`. It uses usd-core
without loading OVStage, Warp, or readers. Authoring tools
do not need an evaluator. Importing it without usd-core raises an
`ImportError` naming the missing package.

Importing `ovcae.authoring` registers the codeless schemas with usd-core, so
import it before the first usd-core stage exists in the process: usd-core fixes
its schema registry on first use. A recipe can select any prim a reader can
resolve. The reader imports the source file into the scene, which is why
`import_dataset` lives in `ovcae.readers` and needs the `usd` extra.

This guide also covers the two resource packages authoring builds on:
`ovcae.materials`, the bundled MDL materials and colormaps, and
`ovcae.schemas`, the schema bundle, typed schema classes, and access to
usd-core's schema registry.

## Import a scientific file

Install the `usd` extra and place a supported `simulation.cgns` in the working
directory. This example writes `surface.usda` beside it. Later fragments reuse
this stage; choose field names and bounds that match your data, and save the
stage again after edits.

```python
import ovcae.authoring as author
import ovcae.readers
from pxr import Usd

stage = Usd.Stage.CreateNew("surface.usda")
world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)
holder = ovcae.readers.import_dataset(stage, "/World/Data", "./simulation.cgns")
datasets = list(ovcae.readers.dataset_prims(stage, holder))
for dataset in datasets:
    print(dataset.GetPath())
if not datasets:
    raise RuntimeError("No scientific datasets were composed")
surface = author.create_faces(
    stage,
    "/World/Surface",
    source=datasets[0],
    external_only=True,
    materials=False,
)
problems = author.validate_recipe(surface)
if problems:
    raise ValueError(problems)
stage.Save()
```

`ovcae.readers.import_dataset` authors a payload on an untyped holder prim and
applies the file format's `OmniSciFileFormatArgs<Format>API`; the
[readers guide](readers.md#author-a-holder) describes its arguments.
`dataset_prims` lists the `OmniSciDataset` prims the plug-in composed beneath
the holder. Choose the dataset your application needs; this example takes the
first one. A recipe's `source` can equally be a descriptor prim for a custom
reader; authoring checks only that the prim exists, and whether a reader can
resolve it is decided at evaluation time.

## Create recipes

Every builder takes `(stage, path, ...)`, returns the created `Usd.Prim`, and
raises `AuthoringError` on invalid input. A builder authors the recipe prim
type, the operator schema, the selection schemas, and placeholder geometry.
Optional settings left as `None` keep their schema fallbacks; builders write the
values you supply rather than every fallback.

Recipe and seed/control builders require an unused destination path. If a prim
already exists there, including an untyped or inactive prim or one composed from
another layer, they raise `AuthoringError` without changing the scene. Use the
typed schema wrappers to edit an existing recipe; do not call its builder again.

| Builder | Required keyword inputs | Useful optional settings |
| --- | --- | --- |
| `create_faces` | `source` | `external_only`, `colors` |
| `create_points` | `source` | `colors`, `widths`, `width`, `max_count`, `use_cell_points` |
| `create_glyphs` | `source` | `shape`, `prototype`, `colors`, `scales`, `orientations`, `orientations_mode`, `scale`, `max_count`, `use_cell_points` |
| `create_iso_surface` | `source`, `contour`, `iso_value` | `colors` (defaults to the contour field) |
| `create_planar_slice` | `source`, `colors` | `mode`, `origin`, `rotation_xyz` |
| `create_streamlines` | `source`, `seeds`, `velocities` | `colors`, `direction`, `max_steps`, `initial_step_size`, `min_step_size`, `max_step_size`, `tolerance`, `width` |

All builders also accept `identifier`, `enabled`, `display_color`, `materials`,
`colormap`, `color_range`, and `edit_target`. `source` is a prim or a path that
resolves on the authoring stage; use `identifier=` to qualify a dataset beneath
a holder or descriptor. For a runtime memory source, define an `OmniSciDataset`
prim at the matching key, as the [tutorial](getting-started.md) does. `colors`
names one field or a sequence of field components. Streamline `velocities`
names a vector field or its scalar components in order, subject to the Data
Source's field support. Planar Slice authors translate, rotateXYZ, and scale
operations so the application can position the plane later.

## Seeds and controls

```python
seeds = author.create_seed_sphere(
    stage,
    "/World/Seeds",
    bounds=((0, 0, 0), (1, 1, 1)),
    count=64,
)
author.create_streamlines(
    stage,
    "/World/Flow",
    source=datasets[0],
    seeds=seeds,
    velocities=("VelocityX", "VelocityY", "VelocityZ"),
    materials=False,
)
```

`create_seed_points(stage, path, points, *, width=None)` authors explicit
seed coordinates. `create_seed_sphere(stage, path, *, count=64, bounds=None,
center=None, radius=None)` distributes seeds on a sphere. Given `bounds`, the
sphere is centered on them with a radius of five percent of the half diagonal;
`center` and `radius` override either value, and without `bounds` both are
required.
`sphere_points` computes the same positions without authoring a prim.
`create_bounding_box` authors a line box, and `Bounds` provides `center`,
`diagonal`, and `is_empty`. Seeds may be any `UsdGeomPointBased` prim, such as
`Points` or `Mesh`; choose bounds and field names from your dataset.

## How a recipe is represented

The builders produce ordinary USD that you can also author by hand or inspect
in other tools. A recipe prim has:

- the product prim type: `Mesh` for Faces, Iso Surface, and Planar Slice,
  `BasisCurves` for Streamlines, `Points` for Points, `PointInstancer` for Glyphs;
- `CaeVizOperatorAPI`, whose `enabled` attribute controls evaluation and
  product cleanup;
- one operator schema, such as `CaeVizFacesAPI`, carrying that operator's
  settings as `cae:viz:<operator>:<name>` attributes;
- one `CaeVizDatasetSelectionAPI:<role>` instance per dataset role, with a
  `target` relationship and an optional `identifier`;
- zero or more `CaeVizFieldSelectionAPI:<role>` instances, each with
  `fieldNames`, an `input` token naming the dataset role (default `source`),
  and `mode` and `componentIndex` controlling interpretation.

Output field role names become primvar names on the product, so a `colors`
selection produces a `colors` primvar. Visibility follows normal USD rules and
suspends evaluation while keeping the last product, as described in the
[integration guide](integration-guide.md#product-ownership-and-visibility).

## Materials and colormaps

`ovcae.materials` ships the CAE MDL module and colormap textures in the wheel.
Scenes reference them by stable asset identifiers such as `cae/mdl/basic.mdl`
and `cae/colormaps/gist_rainbow.png`, so scenes stay relocatable across
installations. The module defines `ScalarColor`, `UnlitScalarColor`, and
`AnimatedStreaks`. `MDL_MODULE` names the module and `colormap_paths()` lists
the bundled PNG textures.

By default a recipe builder creates an MDL material under `<recipe>/Materials`,
wires the selected field and rescale range into it, and binds it with
`weakerThanDescendants`. Session does not evaluate automatic rescale ranges, so
pass a known scalar range when selecting a color field; otherwise the shader
`domain` stays at its `(0, -1)` placeholder and renders flat:

```python
author.create_faces(
    stage,
    "/World/ColoredSurface",
    source=datasets[0],
    colors="Temperature",
    colormap="cividis",
    color_range=(290.0, 310.0),
)
```

`colormap` selects a bundled texture by name: `afmhot`, `cividis`, `coolwarm`,
`gist_gray`, `gist_rainbow`, or `turbo`. `colormap_asset` resolves the name;
an explicit asset path selects another. Pass `materials=False` when your
application supplies its own material system. For custom bindings, the public
helpers are `create_mdl_material`, `bind_material`, `colormap_asset`,
`setup_scalar_materials`, `setup_width_mapping`, and `wire_rescale_range`.
Material prims and bindings are application-owned and survive geometry updates.

For the renderer to find these assets, the material library must be on
`MDL_USER_PATH` before the renderer initializes. `ovcae.setup()` does this for
OVCAE applications. With OVRTX, create the renderer before populating the
Stage so texture assets resolve. Tools without OVCAE can call
`ovcae.materials.register_material_library()`, which prepends the library path
while preserving existing entries, or pass a mutable mapping through
`register_material_library(environ=...)` to prepare a child-process
environment. `material_library_path()` returns the directory for tools that
manage search paths themselves.

## Edit with typed schemas

`ovcae.authoring.schemas` exposes one Python class per schema, with properties
named after the schema `apiName` values in snake case:

```python
from ovcae.authoring import schemas as S

faces = S.FacesAPI.get(surface)
if faces is not None:
    faces.external_only = False
    del faces.external_only  # Clear the authored opinion; use the schema fallback.

colors = S.FieldSelectionAPI.apply(surface, "colors")
colors.field_names = ["Temperature"]
colors.mode = "vector_magnitude"
for role, selection in S.FieldSelectionAPI.instances(surface).items():
    print(role, list(selection.field_names or ()))
```

`Schema.get(prim, instance=None)` returns `None` when the schema is absent. `Schema.apply(prim,
instance=None)` applies and wraps it, enforcing the schema's allowed prim types
unless `check=False` is passed; raw codeless USD does not enforce those
restrictions. Reads return schema defaults for unauthored properties. Writes
validate allowed tokens and coerce ordinary Python values to pxr types. Use
`faces.attribute("external_only")` for the underlying `Usd.Attribute` when you
need time samples or metadata; relationships have typed access too. For schemas
outside the bundle, such as file-format argument APIs and custom operator
schemas, use `S.schema_from_registry(name)`; the bundled classes are built
the same way.

`validate_recipe(prim)` checks recipes against the bundled operator contracts
and reports structural problems: missing operator or role schemas, incompatible
prim types, empty selections, and disallowed tokens. It
does not load scientific arrays or guarantee that runtime field resolution and
computation succeed; those outcomes come back as update diagnostics.

## Layers and edit targets

Recipe and control builders accept `edit_target=` as an `Sdf.Layer` or `Usd.EditTarget`.
Use it to author into a session or application layer and to fit undo into your
application's layer policy. Builders never save; call `stage.Save()` or export
the layer yourself.

## Schema resources and registry

`ovcae.schemas` holds the codeless OmniCaeViz schema bundle. Importing it loads
neither pxr, OVStage, nor Warp, which makes it usable in inspectors and
validators that have no scene runtime. There are no compiled pxr schema
bindings; usd-core's own schema registry is the source of truth once the
bundle is registered with it.

| Function | Returns |
| --- | --- |
| `schema_resource_path()` | Directory with `schema.usda`, `generatedSchema.usda`, and `plugInfo.json` |
| `schema_source_path()` | The canonical `schema.usda` |
| `register_usd_plugin(*directories)` | Registers codeless plug-in directories with usd-core (the bundle by default) and returns the applied API schema identifiers they declare |
| `schema_from_registry(name)` | The typed class for any applied API schema usd-core knows |
| `find_schema(name)` | The same, or `None` when usd-core does not know the schema |
| `is_usd_builtin_schema(name)` | Whether usd-core itself ships the schema |

Applications get both the OVStage and the usd-core registration from
`ovcae.setup()`, and `ovcae.authoring` registers with usd-core on import. A
plain usd-core tool that imports neither can register the plug-in itself:

```python
from ovcae.schemas import register_usd_plugin

register_usd_plugin()
```

Metadata questions are answered by the typed classes. `ovcae.schemas.typed`
holds one per bundled schema and `schema_from_registry` covers the rest; both
are built from usd-core's registry on first use:

```python
from ovcae.schemas import schema_from_registry
from ovcae.schemas import typed as T

print(T.FacesAPI.external_only.default, T.FacesAPI.external_only.type_name)
print(T.FacesAPI.can_apply_to("Mesh"))
print(T.FieldSelectionAPI.field_names.resolve("colors"))
args = schema_from_registry("OmniSciFileFormatArgsCgnsAPI")
print(list(args.properties()))
```

Descriptors carry the type, default, allowed tokens, display metadata, and
relationship target kind, keyed by snake-case attribute (`external_only`)
with the schema spelling available as `api_name` (`externalOnly`). The
exported `OPERATORS` table lists operator metadata for tools; a schema's
presence there does not imply that Session executes it. See the
[operator reference](operators.md#built-in-operators).

The typed classes in `ovcae.schemas.typed` are backend-neutral. `PrimBackend`,
`register_backend`, `backend_for`, and `make_schema_class` let an application
adapt its own prim type; `ovcae.authoring.schemas` is the `Usd.Prim` backend.
There is no bundled backend for the operator scene view, so custom operators
read recipe properties through the [scene-access API](operators.md#read-the-scene).

## Glyph recipes

```python
glyphs = author.create_glyphs(
    stage,
    "/Glyphs",
    source=datasets[0],
    shape="Arrow",
    colors="Temperature",
    color_range=(280.0, 400.0),
    scales="Pressure",
    orientations=("VelocityX", "VelocityY", "VelocityZ"),
    max_count=100_000,
)
mapping = author.schemas.FieldMappingAPI.apply(glyphs, "scales")
mapping.domain = (-930.0, 1300.0)
mapping.range = (0.001, 0.1)
```

Use `shape="Custom", prototype="/Templates/MyShape"` to reference a prototype
hierarchy. The template must be outside the recipe hierarchy. Built-in shapes
are `Sphere` (the default, radius 0.5), `Cone` (height 1 along X), and `Arrow`
(base at the origin, tip at +X). By default, `use_cell_points=True` selects
cell-connected nodes and `max_count=10_000_000` limits the instance count.
Pass `use_cell_points=False` for datasets without cells and `max_count=0`
to include all selected nodes.

`orientations_mode="quaternion"` consumes four-component unit quaternions in
xyzw order. The default `eulerAngles` token retains Kit's meaning of direction
vectors from +X. Constant `scale` defaults to 1 and applies only without a
selected scale field. Color/opacity material setup follows the other builders,
including the need to supply material ranges explicitly. Prototype hierarchy changes must be populated
through USD before evaluation/rendering; Session only publishes instance arrays.
