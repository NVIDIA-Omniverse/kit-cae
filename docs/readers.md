# Readers

A reader defines how scientific datasets are represented and resolved in a
scene. It provides the schemas, registration, a `DataSource` that resolves
recipe selections, and any helpers needed to author its prims. The runtime uses the
Data Source contract, and recipe builders select dataset prims without needing
to know their representation.

`ovcae.readers` is the first-party one. It reads datasets in their native
formats, CGNS and VTK among them, composed into the scene by the
[CAE OpenUSD Plugins](https://github.com/NVIDIA-Omniverse/cae-openusd-plugins).
This page covers that reader first, then the shape your own would take.

## Read datasets with `ovcae.readers`

`ovcae.readers` needs the `usd` extra from the
[installation guide](installation.md), which adds the CAE OpenUSD Plugins;
usd-core is included in the base install. Importing the reader
registers the plug-ins' file formats with usd-core, and usd-core fixes that
registry when the first stage is created, so the reader must be imported before
any stage in the process. The simplest way to guarantee that is to call
`ovcae.setup()` first: it imports every installed reader before doing anything
else. Importing the reader after a stage exists raises a `RuntimeError` saying
so: source files could never compose in that process, and a stage created
earlier cannot compose them even when handed to `UsdDataSource`. The order of
`ovcae.readers` relative to `ovcae.authoring` does not matter.

### How a dataset appears in a scene

A source file is a payload on an untyped holder prim. The holder carries the
file format's `OmniSciFileFormatArgs<Format>API`, whose attributes become the
format's dynamic arguments when the payload composes. The plug-in composes the
file into typed `OmniSciDataset` prims beneath the holder, one per zone, block,
or piece, with the arrays exposed through `OmniSciArrayAPI` instances. A recipe
selects a dataset prim directly, or selects the holder and names a dataset
beneath it in the selection's `identifier`.

OVStage has no file-format plug-ins. It populates the holder as an ordinary
prim and passes dataset paths through untouched, which is why the usd-core
stage behind the reader must compose the same layers as the Execution Stage.

### Author a holder

This example requires `simulation.cgns` with a zone named `Zone1` in the
working directory and writes `surface.usda` beside it.

```python
import ovcae.authoring as author
import ovcae.readers
from pxr import Usd

stage = Usd.Stage.CreateNew("surface.usda")
world = stage.DefinePrim("/World", "Xform")
stage.SetDefaultPrim(world)
holder = ovcae.readers.import_dataset(stage, "/World/Data", "./simulation.cgns", zoneName="Zone1")
datasets = list(ovcae.readers.dataset_prims(stage, holder))
if not datasets:
    raise RuntimeError("No datasets were composed")
author.create_faces(stage, "/World/Surface", source=datasets[0], materials=False)
stage.Save()
```

`import_dataset(stage, path, file_name, *, format=None, edit_target=None,
**args)` defines the holder, applies the format-arguments schema, and adds the
payload. `file_name` is authored as given; pass a layer-relative string to keep
the scene relocatable. `format` names the plug-in's format by short id such as
`"cgns"` or `"vtk"` and defaults to the format registered for the file
extension. Keyword arguments set that schema's attributes by leaf name
(`zoneName="Zone1"`) or Python name (`zone_name="Zone1"`). Bad arguments
raise `ValueError`. The helper authors through usd-core; recipe creation around
it is `ovcae.authoring`'s job.

`dataset_prims(stage, holder)` loads the holder if necessary and returns the
composed `OmniSciDataset` prims in stage order. The list is empty when the
payload did not compose.

### Read at evaluation time

This fragment assumes `ovcae.setup()` has run before any stage was created,
and `execution_stage` is an application-owned `ovstage.Stage` populated from
`surface.usda` and committed at ordinal 1. It is separate from the usd-core
authoring stage above. An RTX-capable NVIDIA GPU must be available as `cuda:0`.
See [getting started](getting-started.md) for population and commit code.

```python
import ovcae
import ovcae.readers

with ovcae.readers.UsdDataSource.open("surface.usda") as source:
    with ovcae.Session(execution_stage, device="cuda:0") as session:
        with source.at_time(0.0) as frame_data:
            report = session.update(input_ordinal=1, output_ordinal=2, frame_data=frame_data).wait()
        if not report.committable:
            raise RuntimeError(report.diagnostics)
        execution_stage.advance_write_floor(report.output_ordinal).wait()
```

`UsdDataSource.open(path)` composes the scene on a private usd-core stage with
held interpolation and takes no other options. `UsdDataSource(stage)` resolves
against a `pxr.Usd.Stage` the application already owns and uses it as-is, for
example a masked stage built to limit composition on a large scene. Set held
interpolation on that stage before wrapping it:

```python
from pxr import Usd

# usd_stage is an existing usd-core stage composed with the reader registered.
usd_stage.SetInterpolationType(Usd.InterpolationTypeHeld)
with ovcae.readers.UsdDataSource(usd_stage) as source:
    ...
```

The source claims any key that is, or composes, an `OmniSciDataset`. For each
frame it selects one dataset-wide sample by held time, describes every
component of that dataset with the same identity so change detection can skip
unchanged work, and retains the most recently loaded sample per dataset prim
and device so acceleration structures survive across updates. Composed layers
and backing files must stay unchanged for the source's lifetime; open a new
source after replacing them.

## Write a reader

You can add a reader without modifying `ovcae` or `ovcae.authoring`. For
example, descriptor prims can point to files read by VTK. A reader has four parts.
[Appendix A2](../tutorials/A2_custom_data_source.py) of the tutorials writes a
complete one over VTK's Exodus II reader.

1. **A schema.** Define the API schema your descriptor prims carry, as a
   codeless plug-in like the tutorials' `vtk_external_dataset_schema`. Recipes select the descriptor
   prim and pass any qualifier through the selection's `identifier`. The
   schema stays out of the `ovcae.schemas` bundle; it belongs to the reader.

2. **Registration.** Register the schema with usd-core's `Plug` registry when
   your package is imported, as `ovcae.readers` does for its file formats. Then
   declare the package in your wheel's `ovcae.readers` entry-point group:

   ```toml
   [project.entry-points."ovcae.readers"]
   vtk = "my_vtk_reader"
   ```

   `ovcae.setup()` imports every package in that group before the process
   creates its first stage, so applications initialize your reader without
   knowing its name. Raise `ImportError` when an optional dependency is
   missing, which `setup()` skips with a warning, and any other error for a
   process state you cannot recover from, which it propagates.

3. **A Data Source.** Subclass `ovcae.DataSource` and implement `resolves`,
   `frame_at_time`, and `release_resources` as the
   [integration guide](integration-guide.md#combine-or-implement-sources)
   describes. `ovcae.data` provides the schema-free pieces every reader repeats:
   `resolve_asset_path` for layer-relative files, `held_time` and
   `DatasetSample` for held sampling and change detection, and
   `dataset_from_arrays` with `add_fields` to turn arrays into a Warp SimData
   dataset. Compose your source with `ovcae.readers` through
   `ovcae.CompositeDataSource` when one scene uses both.

4. **An authoring helper, optionally.** If applications should create your
   descriptor prims from Python, ship a function that defines the prim and
   applies your schema with usd-core, as `ovcae.readers.import_dataset` does.
   Recipe builders accept any existing prim as `source`, so no change to
   authoring is needed.

Tests keep the generic packages generic: none of them may import a reader or
name its schema vocabulary, so adding a reader never means editing them.
