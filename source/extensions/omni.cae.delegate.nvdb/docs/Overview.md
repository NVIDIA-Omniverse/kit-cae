# Overview

The `omni.cae.delegate.nvdb` extension provides a data delegate for reading NanoVDB (`.nvdb`) files.

## Usage

Add a `CaeNvdbFieldArray` prim to your stage and set `fileNames` to the path of a `.nvdb` file.
The delegate will load the file using `warp.Volume.load_from_nvdb` and return the raw NanoVDB
buffer as a field array, which can then be consumed by the CAE rendering and analysis pipeline.

```usda
def CaeNvdbFieldArray "Pressure" (
    prepend apiSchemas = ["CaeNanoVDBFieldArrayAPI"]
)
{
    asset[] fileNames = [@./pressure.nvdb@]
    uniform token fieldAssociation = "vertex"
    int3 cae:nanovdb_field_array:dims = (64, 64, 64)
    int3 cae:nanovdb_field_array:origin = (0, 0, 0)
}
```
