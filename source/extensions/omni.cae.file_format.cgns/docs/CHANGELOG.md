# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0]

- The reader now creates `field:*` relationships for cell-centered fields on NGON_n sections,
  enabling face-based visualization to consume volume cell data (paired with
  `omni.cae.dav` 1.2.0's NFACE_n→NGON_n remapping).

## [1.1.0]

- Added support for file format arguments to provide time scale and offset when opening CGNS files.

## [1.0.0]

- Initial version.
