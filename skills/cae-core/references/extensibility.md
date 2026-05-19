# Format Extensibility

Kit-CAE onboards new formats via 4 layers:

| Layer | Role | Example |
|-------|------|---------|
| USD Schema | Format-specific `CaeFieldArray` attributes | `omniCaeScae` |
| Data Delegate | Lazy runtime data reading (`get_field_array`) | `omni.cae.delegate.scae` |
| Asset Importer | Creates USD stage hierarchy on import | `omni.cae.importer.scae` |
| API Schema | Declares topology | `CaePointCloudAPI`, `CaeSidsUnstructuredAPI`, `CaeDenseVolumeAPI` |

**API Schema selection:**

| Data type | API Schema | VTK equivalent |
|-----------|-----------|----------------|
| Point coordinates only | `CaePointCloudAPI` | — |
| Mesh connectivity | `CaeSidsUnstructuredAPI` | UnstructuredGrid |
| Structured IJK grid | `CaeDenseVolumeAPI` | ImageData |

`CaeDenseVolumeAPI` requires `minExtent`, `maxExtent`, `spacing` attributes.

**⚠ Dense-volume flat byte layout (Fortran order):** The voxelization kernel
expects the field-array payload to be in **Fortran order** (i-fastest), so that
`flat[i + j*ni + k*ni*nj] == data[i,j,k]`. A numpy array of shape `(ni, nj, nk)`
written with the default `.tobytes()` (C order) has `k` fastest and will render
as garbage or a spatially-constant blob that appears not to update with time.
Always write payloads via `np.asfortranarray(arr).tobytes(order="F")`.

**Full tutorial:** `docs/FormatOnboarding.md`
**Working reference:** `docs/format_tutorial_reference/` — complete SCAE format
implementation (schema, delegate, importer, data generator). Copy structure for
new formats.

## Build System Requirements

### Per-extension files

```
source/extensions/omni.cae.<type>.<name>/
    premake5.lua              # REQUIRED — without this, extension silently skipped
    config/extension.toml
    python/__init__.py        # MUST import Extension (see below)
    python/extension.py
    python/<implementation>.py
```

### __init__.py MUST re-export Extension (CRITICAL)

`python/__init__.py` must contain:
```python
from .extension import Extension  # noqa: F401
```
Without this, Kit logs `[ext: ...] startup` but **never calls `on_startup()`**.
The delegate/importer silently does nothing — no error, no registration.
This applies to both delegates and importers.

**Importers must also expose `import_to_stage` at package level** so callers can
`from omni.cae.importer.<name> import import_to_stage`:
```python
# python/__init__.py
from .extension import Extension          # noqa: F401
from .importer import import_to_stage     # noqa: F401
```

### premake5.lua (CRITICAL)

```lua
local ext = get_current_extension_info()
project_ext(ext)
repo_build.prebuild_link {
    { "python", ext.target_dir.."/<dotted>/<module>/<path>" },
}
```

Target path must match Python import path (e.g., `omni/cae/delegate/nvol/`).

### extension.toml rules

- `[[python.module]]`: `name` only, NOT `path` (premake handles it)
- Dependencies: `omni.cae.data`, `omni.cae.schema`, and `omni.cae.dav` (for delegates)
- Delegates: register format in `[settings.exts."omni.cae.data".formats]`
- Importers: use `ai.register_importer()` (NOT `get_importer().register_importer()`)

### Schema registration (3 places)

1. `repo_schemas.toml` — `[[schema_library]]` entry
2. `source/schemas/premake5.lua` — add to `install_usdgenschema` list
3. `source/extensions/omni.cae.schema/.../cae.py` — Python class wrapper

### Bundle wiring

Add new extensions as deps in `omni.cae.bundle/config/extension.toml`.

### DAV command naming

Delegates must define `{SchemaName}ConvertToDAVDataSet` command
(e.g., `CaeDenseVolumeConvertToDAVDataSet`). Dispatch looks up by applied API schema name.

**Before applying a shared API schema, check whether a default converter already ships.**
`omni.cae.dav.commands` bundles converters for `CaePointCloudAPI`, `CaeMeshAPI`,
`CaeSidsUnstructuredAPI`, and each `CaeVtk*API`. **`CaeDenseVolumeAPI` has no bundled
converter** — if your importer applies it, your delegate extension must also register
`CaeDenseVolumeConvertToDAVDataSet` (use `CaeVtkImageDataConvertToDAVDataSet` as a
~15-line reference). Without it, `CreateCaeVizVolume` fails with
`NotImplementedError: Failed to execute command 'ConvertToDAVDataSet'` at render time.

### Post-onboarding

- Update `kit-cae-api.md` Stage Discovery container list with new format's data container names
- Clean rebuild: `./repo.sh build -xr`
