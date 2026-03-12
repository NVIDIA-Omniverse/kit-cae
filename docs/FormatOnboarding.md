# Onboarding a Custom Data Format into Kit-CAE

This tutorial shows you how to add support for a new scientific data format
in Kit-CAE, from USD schema definition to File > Import integration.  The
example uses a simple teaching format called **"Simple CAE"** referred to here as **Scae** (`.scae`), a JSON
manifest plus a raw binary data file.  Every step applies to any format.

> **Audience**: Python developers who know basic USD concepts (prims, attributes,
> stages) but are new to Kit-CAE internals.

## How to Follow This Tutorial

All reference files live in [`docs/format_tutorial_reference/`](format_tutorial_reference/).
Choose your preferred track:

### Track 1: Autocomplete

Run the auto-complete script to place all tutorial files at once, then jump
straight to the smoke test:

```bash
python docs/format_tutorial_reference/autocomplete_tutorial.py
./repo.sh build -xr
./repo.sh launch -n omni.cae.kit -- --exec scripts/generate_scae_data.py
```

Then read through the steps below at your own pace to understand what was
done and why.

### Track 2: Guided Copy

Follow each step below.  Where a reference file is provided, copy it from
`docs/format_tutorial_reference/` to the indicated location.  Each step notes
exactly which files to copy.

### Track 3: Fully Manual

Build everything from scratch.  For new extensions, scaffold with:

```bash
./repo.sh template new
```

Select **Basic Python Extension** (`basic_python`), then implement the code
shown in each step.  Reference existing format extensions
(e.g. `omni.cae.delegate.npz`, `omni.cae.importer.npz`) for patterns.

---

## What You Can Use Independently

This tutorial walks through the full integration stack, but each tier stands
on its own.  Choose the depth that matches your workflow:

| Tier | Steps | What You Get |
|------|-------|--------------|
| **OpenUSD for Scientific Data** | 1–3 | A USD schema subtype and data delegate give you USD-native data descriptions with lazy loading — no Kit or Omniverse dependency required. |
| **Common Operators via API Schemas** | + 4 (partially) | Applying a data-model API schema (e.g. `CaePointCloudAPI`) lets Kit-CAE's shared visualization and processing operators interpret your data topology. |
| **Full Kit-CAE / Omniverse Integration** | 4–7 | An asset importer, bundle wiring, and File > Import support give you the complete Kit-CAE application experience. |

The steps below demonstrate all three tiers end-to-end using the example
`.scae` format, but you can stop at any tier boundary.

---

## Substitution Map

When adapting this tutorial for your own format, replace these placeholders.

| Placeholder | Worked Example | Meaning |
|---|---|---|
| `<FormatName>` | `Scae` | PascalCase identifier root |
| `<format_ext>` | `.scae` | File extension |
| `<SchemaLib>` | `omniCaeScae` | USD schema library name |
| `<SchemaClass>` | `CaeScaeFieldArray` | New `CaeFieldArray` subtype |
| `<DelegateExt>` | `omni.cae.delegate.scae` | Delegate extension ID |
| `<ImporterExt>` | `omni.cae.importer.scae` | Importer extension ID |

## Architecture Overview

Kit-CAE format onboarding has four layers.  Each layer has a distinct
responsibility, and separating them keeps the system modular.  You can swap out
a data reader without touching the importer, or change the USD layout without
rewriting your parser.

1. **USD Schema subtype** : Defines format-specific attributes on `CaeFieldArray`.
   This tells Kit-CAE's generic visualization pipeline *what metadata your format
   needs* without coupling it to how the data is actually read.
2. **Data Delegate** : A `DataDelegateBase` subclass that reads data at runtime.
   Delegates are called lazily (when the renderer or an operator needs array
   values), not at import time.  This means large datasets don't block the UI.
3. **Asset Importer** : Registers with `omni.kit.tool.asset_importer` so your
   format appears in File > Import.  The importer creates the USD stage
   hierarchy and wires up the prims, but does *not* read the data itself.
4. **Data Model API Schema** : Declares the topology of your dataset so the
   DAV visualization pipeline knows how to render it.  For example,
   `CaePointCloudAPI` declares which array holds coordinates;
   `CaeSidsUnstructuredAPI` adds element connectivity.  Applied by the
   importer at import time.

Reference: [Extensions.md](Extensions.md), [CaeSchemas.md](CaeSchemas.md)

---

## Step 1: Define the Example Format

The `.scae` format consists of two files:

| File | Content |
|------|---------|
| `<name>.scae` | JSON manifest describing arrays and their locations |
| Referenced `.bin` | Raw binary blob with concatenated array data |

**Manifest structure** (`sample.scae`):
```json
{
  "version": 1,
  "binary_file": "sample.bin",
  "arrays": {
    "Coordinates": {"dtype": "float32", "shape": [1000, 3], "offset_bytes": 0},
    "Temperature": {"dtype": "float32", "shape": [1000], "offset_bytes": 12000},
    "Pressure":    {"dtype": "float32", "shape": [1000], "offset_bytes": 16000}
  }
}
```

### Key Concepts

- **Manifest + binary split**: The manifest is a lightweight description of
  what's in the binary file.  The binary blob contains raw array data packed
  end-to-end.  This pattern separates *metadata* (which arrays exist, their
  types and shapes) from *payload* (the actual numbers), which means you can
  inspect the manifest without loading gigabytes of data.

- **`offset_bytes`**: Each array starts at a known byte position in the binary
  file.  Offset-based access lets the reader jump directly to any array without
  scanning the entire file, which is critical for large datasets.

- **`dtype` and `shape`**: These tell the reader how to interpret raw bytes.
  `float32` means 4 bytes per value; `shape: [1000, 3]` means 1000 vectors of
  3 components (3000 floats total = 12000 bytes).

This is the same idea used by real scientific formats where data blocks sit at
known offsets (CGNS, HDF5, EnSight).

---

## Step 2: Create a USD Schema Subtype

**Why schema inheritance?**  Kit-CAE has two layers that need to understand
your data: the *data delegate pipeline* and the *DAV (Data API Visualization)
pipeline*.  Inheriting from `CaeFieldArray` satisfies the first: delegates,
operators, and the data registry all work with the base type, so your
format-specific prim is automatically compatible without those systems knowing
about `.scae` or `.cgns` specifically.  The format-specific attributes you add
(like `arrayName`) give *your delegate* the metadata it needs to locate data,
while everything else treats the prim generically.

However, the DAV layer also needs to know the *topology* of your dataset:
is it a point cloud, an unstructured mesh, or a structured volume?  This is
declared by applying a **Data Model API schema** (e.g. `CaePointCloudAPI`,
`CaeSidsUnstructuredAPI`) to the dataset prim at import time.  DAV looks for
these API schemas to discover which arrays are coordinates, which represent
connectivity, and how to drive visualization.  Without an applied API schema,
DAV won't render your data.  Step 4 covers which API schema to choose and how
the importer applies it.

**File**: `source/schemas/formats/omniCaeScae/schema.usda`

```usda
#usda 1.0
(
    subLayers = [ @../../shared/omniCae/schema.usda@ ]
)

over "GLOBAL" (
    customData = {
        string libraryName = "omniCaeScae"
        string libraryPath = "omniCaeScae"
        string libraryPrefix = "OmniCaeScae"
        bool useLiteralIdentifier = true
    }
) {}

class CaeScaeFieldArray "CaeScaeFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Field array for SimpleCAE manifest + binary payload format."
    customData = { string className = "ScaeFieldArray" }
)
{
    string arrayName = "" (
        doc = "Name of the array entry inside the .scae manifest."
    )
    uniform string slice = "" (
        doc = "Optional NumPy-style slice expression. Supports {ts} substitution."
    )
    int ts = -1 (
        doc = "Time-sample value for slice format expressions."
    )
}
```

**Why `arrayName`?** This maps a USD prim to a specific entry in the manifest.
Without it, the delegate wouldn't know which array to read because the manifest
may contain dozens of arrays while each prim represents exactly one.

**Why `slice` and `ts`?** These are optional attributes for time-varying
datasets. The `slice` attribute holds a NumPy-style expression like `"{ts}:"`
and `ts` provides the time-sample value, allowing the delegate to extract a
sub-range of the data per frame.  Most simple formats won't need these.

### Decision Checklist : Schema Attributes

| Question | If YES | If NO |
|---|---|---|
| Does the format have named arrays? | Add `arrayName` attribute | Use `fileNames` path alone |
| Is path navigation needed within files? | Add a `fieldPath` attribute (like CGNS) | Omit |
| Is time-dependent slicing needed? | Add `slice` + `ts` attributes | Omit |
| Are there security-sensitive load modes? | Add safety flags (like `allowPickle`) | Omit |

### Wire the Schema into the Build

1. Add a `[repo_usd.plugin.omniCaeScae]` block to `repo_schemas.toml`
   (mirror the `omniCaeNumPy` block, replacing names).
2. Add `"omniCaeScae"` to the `schemas` table and `schema_base_deps` in
   `source/schemas/premake5.lua`.
3. Register in `omni.cae.schema`:
   - `config/extension.toml` : add `[[python.module]]` and `[[native.library]]`
   - `extension.py` : add `"OmniCaeScae"` to the schema list
   - `cae.py` : add `from OmniCaeScae import ScaeFieldArray  # noqa: F401`

> **Track 2**: Copy the following reference files to their destinations:
>
> | Reference file | Destination |
> |---|---|
> | `format_tutorial_reference/new_schemas/formats/omniCaeScae/` | `source/schemas/formats/omniCaeScae/` |
> | `format_tutorial_reference/repo_schemas.toml` | `repo_schemas.toml` |
> | `format_tutorial_reference/premake5.lua` | `source/schemas/premake5.lua` |
> | `format_tutorial_reference/updated_extensions/omni.cae.schema/` | overlay onto `source/extensions/omni.cae.schema/` |
>
> **Track 3**: Create the directory and files manually using the schema
> definition above, then edit the three build/registration files by hand.

**Verification**: After building (`./repo.sh build -xr`), confirm
`cae.ScaeFieldArray` is importable in the Kit Script Editor.

Reference: [UsdSchemas.md](UsdSchemas.md), [CaeSchemas.md](CaeSchemas.md)

---

## Step 3: Implement the Data Delegate

**Why a separate delegate?**  The data delegate decouples *reading* from
*stage layout*.  The importer (Step 4) creates prims and relationships, but
the actual array values are read lazily by the delegate, only when something
needs them (a color mapper, a bounding-box computation, etc.).  This keeps
import fast and avoids loading data that might never be displayed.

**Extension**: `source/extensions/omni.cae.delegate.scae/`

> **Track 2**: Copy `format_tutorial_reference/updated_extensions/omni.cae.delegate.scae/`
> to `source/extensions/omni.cae.delegate.scae/`.
>
> **Track 3**: Scaffold the extension:
> ```bash
> ./repo.sh template new
> # Select: Basic Python Extension (basic_python)
> # Name: omni.cae.delegate.scae
> ```
> Then implement the files shown below.

### 3.1  Extension Lifecycle (`extension.py`)

```python
import omni.ext

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        from omni.cae.data import get_data_delegate_registry
        from .scae import ScaeDataDelegate

        self._registry = get_data_delegate_registry()
        self._delegate = ScaeDataDelegate(ext_id)
        self._registry.register_data_delegate(self._delegate)

    def on_shutdown(self):
        self._registry.deregister_data_delegate(self._delegate)
        del self._delegate
```

The delegate registers itself with the global registry on startup and
deregisters on shutdown.  This lets Kit-CAE discover it automatically.

### 3.2  Delegate Logic (`scae.py`)

The delegate implements two methods on `DataDelegateBase`:

- **`can_provide(prim)`** - returns `True` if the prim is a `ScaeFieldArray`.
  The registry calls this to find which delegate can handle a given prim.
- **`get_field_array(prim, time)`** - reads the manifest, locates the named
  array by `offset_bytes`, reads that slice of the binary file, and returns a
  `numpy.ndarray`.

Core reading logic:

```python
import json, os, numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.client import get_local_file

class ScaeDataDelegate(DataDelegateBase):
    def can_provide(self, prim):
        return prim and prim.IsValid() and prim.IsA(cae.ScaeFieldArray)

    def get_field_array(self, prim, time):
        prim_t = cae.ScaeFieldArray(prim)
        array_name = prim_t.GetArrayNameAttr().Get(time)
        file_names = prim_t.GetFileNamesAttr().Get(time)
        if not array_name or not file_names:
            return None

        manifest_path = get_local_file(file_names[0].resolvedPath)[1]
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        spec = manifest["arrays"].get(array_name)
        if spec is None:
            return None

        binary_path = os.path.join(os.path.dirname(manifest_path), manifest["binary_file"])
        dtype, shape = np.dtype(spec["dtype"]), tuple(spec["shape"])
        count = int(np.prod(shape))
        return np.fromfile(binary_path, dtype=dtype, count=count,
                           offset=spec.get("offset_bytes", 0)).reshape(shape)
```

**How this works**: The delegate reads the prim's `arrayName` attribute to know
which entry to look up in the JSON manifest.  The manifest tells it the
`offset_bytes`, `dtype`, and `shape`, which is enough to jump to the right position in
the binary file and read exactly the right number of bytes.  `np.fromfile` with
an `offset` parameter makes this a single seek + read.

The full implementation (see the reference `scae.py`) adds error handling,
logging, multi-file concatenation, and optional slice expression support.

**Why `fileNames` is an array**: USD asset paths are stored as an array to
support multi-file datasets, which is common in time-varying simulations where each
timestep is in a separate file.  The delegate iterates `fileNames`, reads each
manifest, and concatenates the results.

**Verification**: Load the extension, create a `CaeScaeFieldArray` prim pointing
at `sample.scae`, call `registry.get_field_array(prim)` and expect a `(1000, 3)`
float32 array for Coordinates.

Reference: [DataDelegate.md](DataDelegate.md)

---

## Step 4: Implement the Asset Importer

**Why a separate importer?**  The importer's job is to create the USD stage
hierarchy (prims, relationships, and metadata) but not to read the actual
data.  This separation means importing is fast (just creating prims) while
data loading happens lazily through the delegate.

**Extension**: `source/extensions/omni.cae.importer.scae/`

> **Track 2**: Copy `format_tutorial_reference/updated_extensions/omni.cae.importer.scae/`
> to `source/extensions/omni.cae.importer.scae/`.
>
> **Track 3**: Scaffold the extension:
> ```bash
> ./repo.sh template new
> # Select: Basic Python Extension (basic_python)
> # Name: omni.cae.importer.scae
> ```
> Then implement the files shown below.

### 4.1  Importer Class (`importer.py`)

Subclass `AbstractImporterDelegate` from `omni.kit.tool.asset_importer`:

```python
from omni.kit.tool.asset_importer import AbstractImporterDelegate

class ScaeAssetImporter(AbstractImporterDelegate):
    @property
    def name(self):
        return "SimpleCAE Importer"

    @property
    def filter_regexes(self):
        return [r".*\.scae$"]

    @property
    def filter_descriptions(self):
        return ["SimpleCAE files (*.scae)"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    async def convert_assets(self, paths, **kwargs):
        # For each file: create a USD stage, populate it, return stage ID or path
        ...
```

### 4.2  Stage Population Logic (`_populate_stage`)

The importer creates this USD hierarchy:

```
/World
  /<filename>
    /ScaeDataSet          (CaeDataSet + CaePointCloudAPI)
    /ScaeFieldArrayClass  (class prim with shared fileNames)
    /ScaeArrays
      /Coordinates        (CaeScaeFieldArray, specialises class prim)
      /Temperature        (CaeScaeFieldArray)
      /Pressure           (CaeScaeFieldArray)
      ...
```

Key steps:
1. Create `/World` xform, set as default prim, Z-up axis.
2. Create `CaeDataSet`, apply `CaePointCloudAPI`.
3. Create a **class prim** with shared `fileNames` and `fieldAssociation`.
4. Parse the `.scae` manifest.
5. For each array: define a `CaeScaeFieldArray` that specialises the class prim.
6. Detect coordinate arrays and wire `CaePointCloudAPI.coordinates` relationship.

**Why class prims for shared attributes?**  Every array prim needs the same
`fileNames` (path to the `.scae` file) and `fieldAssociation` (vertex, element,
etc.).  Rather than repeating these on every prim, we define them once on a
class prim and have each
array prim *specialise* it.  This is USD's equivalent of the DRY principle.

**Why coordinate detection heuristics?**  The importer automatically wires
`CaePointCloudAPI.coordinates` by matching array names against common patterns
(`Coordinates`, `Coords`, `Points`, `XYZ`, `GridCoordinates`, and per-component
`X`/`Y`/`Z`).  This means users don't have to manually set up the coordinate
relationship, so it just works for most datasets.

### 4.3  Public API

```python
async def import_to_stage(path: str, prim_path: str) -> Usd.Prim:
    """Import a .scae file into the current stage at the given prim path."""
```

### Decision Checklist : API Schema

Each API schema represents a different data topology.  Choose based on what
your data actually contains:

| Your data has... | Apply this API schema | What it means |
|---|---|---|
| Point coordinates only | `CaePointCloudAPI` | Unconnected points in 3D space (particles, measurement probes) |
| Mesh connectivity (elements) | `CaeSidsUnstructuredAPI` | Elements with connectivity arrays linking vertices into faces/cells |
| Structured grid (IJK extents) | `CaeDenseVolumeAPI` | A regular 3D grid where position is implicit from IJK indices |
| Custom relationships | Define your own API schema | When none of the above fits your data model |

**Verification**: File > Import shows `.scae` filter.  Import creates expected
prim hierarchy.  `CaePointCloudAPI.coordinates` relationship resolves.

Reference: [Extensions.md](Extensions.md)

---

## Step 5: Write Tests (Optional)

Testing is important for production formats but not required to complete this
tutorial.  When you're ready to add tests, follow the patterns established by
existing delegates and importers:

- **Delegate tests**: See `source/extensions/omni.cae.delegate.npz/python/tests/`
  for examples of testing `can_provide()` and `get_field_array()`.
- **Importer tests**: See `source/extensions/omni.cae.importer.npz/python/tests/`
  for examples of testing `import_to_stage()` and verifying prim hierarchy.
- **Running tests**: `./repo.sh test --ext <extension_id>`

---

## Step 6: Wire into Bundle (Optional)

> **Tip**: During development you can skip this step and instead load your
> extensions at launch with `--enable` (shown in Step 7).  This avoids
> adding in-progress extensions to the build and is useful when iterating
> quickly.  Once your format is stable, add the bundle dependencies below
> so the extensions load automatically.  If you ran the autocomplete
> script (Track 1), this step was already done for you.

Add your new extensions to `source/extensions/omni.cae.bundle/config/extension.toml`
so they load automatically when Kit-CAE launches:

```toml
"omni.cae.delegate.scae" = {}
"omni.cae.importer.scae" = {}
```

> **Track 2**: Copy `format_tutorial_reference/updated_extensions/omni.cae.bundle/config/extension.toml`
> to `source/extensions/omni.cae.bundle/config/extension.toml`.
>
> **Track 3**: Edit the file by hand.  Add the two lines above to the
> `[dependencies]` section, alongside the existing delegate/importer entries.

---

## Step 7: Build and Smoke Test

1. **Build** (use `-x` for a clean rebuild so the new schema gets generated):
   ```bash
   ./repo.sh build -xr
   ```
2. **Launch with the smoke-test script** : this generates sample data into
   `data/`, imports it into the stage, and frames the viewport.
   If you completed Step 6 (bundle wiring):
   ```bash
   ./repo.sh launch -n omni.cae.kit -- --exec scripts/generate_scae_data.py
   ```
   If you skipped Step 6, enable the extensions at launch:
   ```bash
   ./repo.sh launch -n omni.cae.kit -- \
       --enable omni.cae.delegate.scae \
       --enable omni.cae.importer.scae \
       --exec scripts/generate_scae_data.py
   ```
3. **Verify** in the running application:
   - Stage window shows the expected prim hierarchy under `/World/scae_torus/`
   - A bounding box appears around the torus
   - Color-map Temperature on the torus : expect a smooth gradient from blue
     (300 K) to red (500 K)

---

## Porting to Your Format

| Layer | What to change |
|---|---|
| Schema | Replace `CaeScaeFieldArray` → `Cae<YourFormat>FieldArray`; adjust attributes for your format's metadata |
| Delegate | Replace JSON+binary parsing with your format's reader (e.g. h5py, custom C parser) |
| Importer | Adjust `filter_regexes`, array detection heuristics, API schema selection |
| Tests | Replace test data files; adjust expected values |

### Format Category Decision Tree

- **Binary with header**: Similar to Scae.  Delegate reads header + data blocks.
- **HDF5-based**: Inherit `CaeHdf5FieldArray`; use `h5py` in delegate.
- **Text/CSV**: Parse with stdlib; same delegate pattern, return ndarray.
- **Requires C++ performance**: See `omni.cae.delegate.cgns` for C++ delegate reference.

---

## Cross-References

- [DataDelegate.md](DataDelegate.md) : Data delegate API and lifecycle
- [CaeSchemas.md](CaeSchemas.md) : Schema design philosophy
- [UsdSchemas.md](UsdSchemas.md) : USD schema overview
- [Extensions.md](Extensions.md) : Extension architecture
- [Build.md](Build.md) : Build prerequisites and commands
