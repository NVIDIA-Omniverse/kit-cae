# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.2.0]

- Changed how color map domains are set/reset. New behavior is more consistent with expectation: ranges are reset
  if no valid range is present or if the field chosen was changed.
- Points, Glyphs, Surfaces now support coloring by multi-component arrays.

## [1.1.0]

- Added support to Streamlines algorithm to propagate additional fields as primvars for shaders.

## [1.0.0]

- Initial version