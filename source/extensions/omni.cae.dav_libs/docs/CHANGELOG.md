# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.1.0]

- Added `Field.to_nanovdb()`: element-for-element conversion of a `Field` to a NanoVDB-backed `Field` without
  spatial resampling.  Supports `float32` scalar and `vec3f` vector fields; integral types are auto-narrowed.
- `probe` operator: added `output_mask_field_name` parameter to expose out-of-domain probe results as a
  named mask field alongside the probed values.
- VTK data model: improved polyhedron face handling in the VTK dataset utilities.

## [1.0.0]

- Initial version.
