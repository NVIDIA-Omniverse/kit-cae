# Getting started

This tutorial authors a Points recipe, evaluates three in-memory points on
the GPU, and commits the result into an OVStage. Use the base install from the
[installation guide](installation.md), with an RTX-capable NVIDIA GPU and
compatible driver. The example evaluates geometry without rendering. The final
section adapts it to read a native simulation file.

## Evaluate an in-memory dataset

Save the following as `first_scene.py` in the repository root. It writes
`points.usda` in the working directory.

```python
from pathlib import Path

import numpy as np
import ovcae
import ovcae.authoring as author
import ovstage
import warp as wp
from pxr import Usd
from warp_simdata.data_models.custom import point_cloud

ovcae.setup()

# 1. Author the recipe scene with usd-core.
scene_path = Path("points.usda").resolve()
recipe_stage = Usd.Stage.CreateNew(str(scene_path))
source = recipe_stage.DefinePrim("/Data", "OmniSciDataset")
author.create_points(
    recipe_stage,
    "/Points",
    source=source,
    width=0.1,
    use_cell_points=False,
    materials=False,
)
recipe_stage.Save()

# 2. Populate the application-owned OVStage and commit the inputs.
stage = ovstage.Stage("first-scene")
ovstage.population.open_usd_with_desc(
    stage,
    str(scene_path),
    ordinal=1,
    time_code=0.0,
    descs=ovcae.ovstage_population_desc(domains=ovstage.PopulationDomain.ALL),
)
stage.advance_write_floor(1).wait()

# 3. Evaluate the recipe on the GPU and commit the output.
points = wp.array(
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    dtype=wp.vec3f,
    device="cuda:0",
)
datasets = {"/Data": point_cloud.create_dataset(points)}
with ovcae.MemoryDataSource(datasets) as data_source:
    with ovcae.Session(stage, device="cuda:0") as session:
        with data_source.at_time(0.0) as frame_data:
            report = session.update(
                input_ordinal=1,
                output_ordinal=2,
                frame_data=frame_data,
            ).wait()
        for diagnostic in report.diagnostics:
            print(diagnostic.severity, diagnostic.code, diagnostic.message)
        if not report.committable:
            raise RuntimeError("The output ordinal is not safe to commit")
        stage.advance_write_floor(2).wait()
        for result in report.operators:
            print(result.operator_path, result.status, result.product_paths)
```

Run it from the repository root:

```console
.venv/bin/python first_scene.py
```

With the base install, a warning that the native reader is unavailable is
expected and can be ignored for this in-memory example.

The program has three parts, and every application that embeds OVCAE has the
same three.

1. **Author the recipe.** `create_points` defines a `Points` prim at `/Points`,
   applies the operator schema, and points its `source` selection at `/Data`.
   The `OmniSciDataset` prim at `/Data` is only a key here; the in-memory
   source supplies a Warp SimData dataset for it later, here a point cloud
   built from three points.
2. **Populate and commit.** The application creates the OVStage, populates it
   from the scene with the descriptors returned by `ovstage_population_desc`,
   and commits the inputs at ordinal 1. `Session` never advances the write floor.
3. **Evaluate and commit.** A `Session` borrows the Stage. Inside a
   `MemoryDataSource` lease at time zero, one update reads the committed recipe
   and writes the points at ordinal 2. The report says whether that ordinal is
   safe to commit and lists each operator's outcome.

The result for `/Points` should be `succeeded`, with `/Points` as its product
path. The committed Stage now holds the points and widths. `points.usda` still
contains only the authored recipe; evaluation never writes generated geometry
back to the file. Closing the Session leaves the products in place.

## Read a real dataset

To evaluate a CGNS, VTK, or other supported file instead of in-memory arrays,
install the `usd` extra and change two things.

First, add `import ovcae.readers` to the imports, before any stage is created.
In step 1, replace the `/Data` definition and `create_points` call with a
payload and Faces recipe, keeping `recipe_stage.Save()` afterward.
This fragment requires `simulation.cgns` beside
`points.usda`; the [readers guide](readers.md#author-a-holder) explains the
holder and dataset prims:

```python
holder = ovcae.readers.import_dataset(recipe_stage, "/Data", "./simulation.cgns")
datasets = list(ovcae.readers.dataset_prims(recipe_stage, holder))
if not datasets:
    raise RuntimeError("No datasets were composed")
author.create_faces(recipe_stage, "/Surface", source=datasets[0], materials=False)
```

Second, in step 3, replace the point array, dataset mapping, and
`MemoryDataSource` with a reader over the same saved scene. Keep the Session,
update, and commit code inside it:

```python
with ovcae.readers.UsdDataSource.open(str(scene_path)) as data_source:
    ...
```

Population and the Data Source have to compose the same layer stack. OVStage
populates the lightweight recipe prims and passes dataset paths through
untouched; the Data Source is what composes the payload, through the CAE
file-format plug-ins, and resolves those paths against it.

## Next steps

- [Integration guide](integration-guide.md): the update and commit cycle in
  an interactive application, Data Sources and scientific time, reports, and
  change detection.
- [Authoring guide](authoring-guide.md): the rest of the recipe builders,
  plus seeds, materials, and typed schema editing.
- [Package reference](packages.md): the public API of each subpackage.
- [CAE Viewer](../examples/cae-viewer/README.md): a complete rendering
  application that opens recipe scenes and manipulates seeds with viewport gizmos.
