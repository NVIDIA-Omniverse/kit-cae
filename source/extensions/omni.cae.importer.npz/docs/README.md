# CAE NPZ Importer [omni.cae.importer.npz]

This extension provides asset importer functionality for NumPy .npz files, integrating them into Omniverse through
the standard asset import framework.

## Usage

Once this extension is loaded, `.npz` files can be imported using the `File > Import` dialog in Omniverse applications.
The importer processes numpy archives and converts the data to USD format with appropriate CAE schema representations.

## Python Usage

The extension provides a convenient Python API for programmatically importing NPZ files into a USD stage.

### `import_to_stage` Function

```python
async def import_to_stage(
    path: str,
    prim_path: str,
    *,
    schema_type: str = "SIDS Unstructured",
    allow_pickle: bool = False
)
```

Import a NPZ file directly into the current USD stage at the specified prim path.

**Parameters:**
- `path` (str): Path to the NPZ file (can be a local path or URL)
- `prim_path` (str): USD prim path where the data should be imported
- `schema_type` (str, optional): Schema type to use for the dataset. Options:
  - `"SIDS Unstructured"` (default): Import as SIDS unstructured mesh
  - `"Point Cloud"`: Import as point cloud data
- `allow_pickle` (bool, optional): Whether to allow pickle for reading NPY files. Only enable for trusted files. Default is `False`.

**Example Usage:**

```python
import asyncio
from omni.cae.importer.npz import import_to_stage

# Import as SIDS Unstructured mesh
prim = await import_to_stage(
    "path/to/data.npz",
    "/World/MyData"
)

# Import as Point Cloud with custom settings
prim = await import_to_stage(
    "path/to/points.npz",
    "/World/PointCloud",
    schema_type="Point Cloud",
    allow_pickle=False
)
```

**Notes:**

- The function is asynchronous and must be called with `await` or run in an async context
- The NPZ file can contain multiple numpy arrays which will be converted to appropriate USD representations
- Array names are automatically mapped to CAE schema relationships when possible (e.g., coordinates, connectivity)
- The function creates a complete USD hierarchy including DataSet, FieldArrays, and appropriate schema APIs
