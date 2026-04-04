# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.4]

- [BUG] Support filenames with spaces in `.case`/`.encas` files by using shell-style quoted string parsing.

## [1.0.3]

- [FEATURE] Support quoted filenames in `.case`/`.encas` files (geometry and variable sections).

## [1.0.2]

- [BUG] Fix variable name parsing when multiple spaces separate the name from the filename in the VARIABLE section of a `.case` file.
- [BUG] Fix per-element variable reader failing when the var file lists element type sections in a different order than the geo file.

## [1.0.1]

- [BUG] Fix incorrect asset path when importing local files on remote stages.

## [1.0.0]

- Initial version.
