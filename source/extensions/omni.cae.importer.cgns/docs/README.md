# CAE CGNS Importer [omni.cae.importer.cgns]

This extension provides asset importer functionality for CGNS files, integrating them into Omniverse through the standard import framework.

## Usage

Once this extension is loaded, CGNS files can be imported using the standard `File > Import` dialog in Omniverse applications.
The importer converts CGNS data to USD format with appropriate CAE schema representations.

## Python Usage

The extension provides a convenient Python API for programmatically importing CGNS files into a USD stage.

### `import_to_stage` Function

```python
async def import_to_stage(
    path: str,
    prim_path: str,
    *,
    time_scale: float = 1.0,
    time_offset: float = 0.0,
    time_source: str = "TimeStep",
)
```

Import a CGNS file directly into the current USD stage at the specified prim path.

**Parameters:**
- `path` (str): Path to the CGNS file (can be a local path or URL)
- `prim_path` (str): USD prim path where the data should be imported
- `time_scale` (float, optional): Scale factor for time values. Default is `1.0`.
- `time_offset` (float, optional): Offset time values by this amount (applied after scaling). Default is `0.0`.
- `time_source` (str, optional): The time source to use. Options:
  - `"TimeStep"` (default): Use time step values
  - `"TimeValue"`: Use time values

**Returns:**
The imported USD prim.

**Example Usage:**

```python
import asyncio
from omni.cae.importer.cgns import import_to_stage

# Import with default settings
prim = await import_to_stage(
    "path/to/data.cgns",
    "/World/MyData"
)

# Import with custom time settings
prim = await import_to_stage(
    "path/to/simulation.cgns",
    "/World/Simulation",
    time_scale=0.001,
    time_offset=0.0,
    time_source="TimeValue"
)
```

**Notes:**

- The function is asynchronous and must be called with `await` or run in an async context
- CGNS files are converted to USD using the CGNS file format plugin
- The importer supports time-varying data with configurable time scaling and offset
- The function creates a complete USD hierarchy with appropriate CAE schema representations
