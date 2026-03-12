# CAE Scae Importer [omni.cae.importer.scae]

This extension provides asset importer support for tutorial `.scae` files.

## Usage

Once loaded, `.scae` files can be imported from **File > Import**.

The importer creates:
- A `CaeDataSet` prim with `CaePointCloudAPI`
- A `ScaeArrays` scope containing `CaeScaeFieldArray` prims
- Field relationships for non-coordinate arrays
- A `coordinates` relationship for point-cloud coordinates

## Python Usage

```python
from omni.cae.importer.scae import import_to_stage

prim = await import_to_stage("path/to/sample.scae", "/World/ScaeCase")
```
