# Overview

`omni.cae.delegate.scae` is a reference Python delegate extension used by the
format-onboarding tutorial. It demonstrates the minimal `DataDelegateBase`
implementation needed to map a custom `CaeFieldArray` subtype to NumPy arrays.

The delegate:
1. Validates `CaeScaeFieldArray` prim type support (`can_provide`)
2. Resolves manifest asset paths from `fileNames`
3. Reads manifest metadata and slices binary payloads into typed arrays
4. Returns NumPy arrays for CAE operator and visualization consumers
