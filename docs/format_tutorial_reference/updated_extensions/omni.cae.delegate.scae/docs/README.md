# CAE Scae [omni.cae.delegate.scae]

This extension provides data delegate support for the tutorial `.scae` format.

The `.scae` format uses:
- A JSON manifest that describes arrays (`dtype`, `shape`, `offset_bytes`)
- A contiguous binary payload file that stores raw array bytes

## Features

- Registers `ScaeDataDelegate` with the CAE data delegate registry
- Reads `CaeScaeFieldArray` prims via the `omni.cae.schema` bindings
- Supports optional slice expressions (`slice` and `ts` attributes)
