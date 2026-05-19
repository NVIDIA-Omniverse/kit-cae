# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0]

- Added `CaeSidsUnstructuredGetField` to remap cell-centered CGNS fields from sibling NFACE_n volume cells
  onto NGON_n faces, so face-based visualization can color by volume cell data.

## [1.1.0]

- `ConvertToDAVDataSet.invoke` now stores results via `cache.put_ex` with a `PrimWatch`-based invalidation guard
  instead of a plain dict-key cache entry, ensuring precise invalidation when the source prim changes.

## [1.0.0]

- Initial version.
