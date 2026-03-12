# CAE VTK Importer [omni.cae.importer.vtk]

This extension provides asset importer functionality for VTK files, integrating them into Omniverse through
the standard asset import framework.

## Usage

Once this extension is loaded, VTK files (`.vtk`, `.vti`, `.vtu`, `.vts`, `.vtp`) can be imported using the `File > Import` dialog in Omniverse applications.
The importer processes VTK files and converts the data to USD format with appropriate CAE schema representations.

## Python Usage

The extension provides a convenient Python API for programmatically importing VTK files into a USD stage.

### `import_to_stage` Function

```python
async def import_to_stage(
    path: str,
    prim_path: str
)
```

Import a VTK file directly into the current USD stage at the specified prim path.

**Parameters:**
- `path` (str): Path to the VTK file (can be a local path or URL)
- `prim_path` (str): USD prim path where the data should be imported

**Example Usage:**

```python
import asyncio
from omni.cae.importer.vtk import import_to_stage

# Import a VTK file
prim = await import_to_stage(
    "path/to/data.vtu",
    "/World/MyData"
)

# Import a structured grid
prim = await import_to_stage(
    "path/to/grid.vti",
    "/World/StructuredGrid"
)
```

**Notes:**

- The function is asynchronous and must be called with `await` or run in an async context
- Supports all VTK file formats: `.vtk`, `.vti`, `.vtu`, `.vts`, `.vtp`
- The VTK file data is converted to appropriate USD representations with CAE schema
- The function creates a complete USD hierarchy including geometry, fields, and appropriate attributes
