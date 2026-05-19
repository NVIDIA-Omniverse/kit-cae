# Overview

This extension is an internal extension designed that brings in OmniCae schemas into Omniverse. It also provides
Python bindings for the USD Schemas. To access the OmniCae schemas in Python, you can use `omni.cae.schema`
Python package in dependent extensions or Kit applications as follows. The USD schema plugins are registered at
startup from the extension's `usd/plugin/*/resources` tree by `omni/cae/schema/extension.py`.

The schema Python modules are published under the `pxr` namespace (matching Pixar's USD schema
convention), so they can also be imported directly, e.g.
`from pxr import OmniCae, OmniCaeSids, OmniCaeVtk`.

```py
from omni.cae.schema import cae, sids
from pxr import Usd

# to check if a prim is a CaeDataSet
prim: Usd.Prim = ...
if prim.IsA(cae.DataSet):
    ds = cae.DataSet(prim)
    # ....

# to check if a prim has
if prim.HasAPI(sids.UnstructuredAPI):
    sidsApi = sids.UnstructuredAPI(prim):
    ...
else:
   sids.UnstructuredAPI.Apply(prim)

```
