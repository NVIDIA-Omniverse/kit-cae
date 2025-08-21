# Overview

This extension is an internal extension designed that brings in OmniCae schemas into Omniverse. It also provides
Python bindings for the USD Schemas. To access the OmniCae schemas in Python, you can use `omni.cae.schema`
Python package in dependent extensions or Kit applications as follows:

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