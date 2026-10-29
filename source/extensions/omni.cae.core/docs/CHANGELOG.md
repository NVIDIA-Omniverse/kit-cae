# Changelog

## 1.1.1

- Missing command diagnostics now list the attempted schema and type command
  names and identify the compatibility bundle for legacy `CaeDataSet` prims.
- Command dispatch no longer treats the internal `Tf.Type` root sentinel as a
  schema command candidate.

## 1.1.0

- Added `array_utils.get_magnitude`, a Warp-kernel helper returning per-element L2
  magnitudes of a vector array as a scalar array (used for component-wise
  statistics in the property panel).

## 1.0.0

- Added `omni.cae.core` as the active shared CAE utility extension.
- Kept legacy Data Delegate and `IFieldArray` APIs in `source/legacy_extensions/omni.cae.data`.
