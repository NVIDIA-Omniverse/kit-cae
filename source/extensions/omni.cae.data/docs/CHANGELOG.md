# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.2.0]

- `put_ex()` now automatically expands each `PrimWatch` prim to include all transitively related
  `CaeDataSet` and `CaeFieldArray` prims, mirroring the long-standing behavior of `put()`. This
  fixes stale intermediate caches when a `FieldArray` property (e.g. `fileNames`) changes after
  a result was cached against the parent `DataSet` prim only.
- When a `PrimWatch` carries a `schemas` filter, only relationships whose names appear in the
  resolved schema-property set are followed at the first traversal hop; subsequent transitive hops
  follow all relationships freely.
- `IUsdUtils::getRelatedDataPrims()` gains a new `relNames` parameter (default `{}`) for
  first-hop relationship filtering. `IDataDelegateInterface` bumped to v0.5.

## [2.1.0]

- `PrimWatch` now accepts `(class_or_str, instance_name)` tuples for filtering property-update triggers against
  multi-apply schema instances (e.g. `(cae_viz.DatasetSelectionAPI, "source")`).  The `__INSTANCE_NAME__`
  placeholder in schema property names is substituted with the actual instance name at filter-resolution time.
- Fixed incomplete and coarse caching across the `get_input_dataset` call chain: intermediate results are now
  stored with targeted `PrimWatch` guards rather than broad prim-level invalidation.

## [2.0.0]

- Added several new features and enhacements to support cae-viz operators.
- Changed settings. Several cae-algorithm specific settings have been flagged as legacy
  and introduced new settings for cache control.
- Added support for IFieldArrayUtils to copy data between devices or reinterpret data.

## [1.2.0]

- Changed array releted APIs to be more robust and support multi-component arrays.

## [1.1.0]

- Added optional `extra_fields` attribute to `GenerateStreamlines` command to request additional fields mapped
  on to the generated streamlines.

## [1.0.0]

- Initial version.
