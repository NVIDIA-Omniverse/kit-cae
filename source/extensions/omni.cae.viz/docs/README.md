# CAE Visualization Extension

This extension provides visualization algorithms and utilities for CAE (Computer-Aided Engineering) data using the OmniCaeViz USD schemas.

## Overview

The `omni.cae.viz` extension serves as a foundation for CAE visualization capabilities in Omniverse. It works with the OmniCaeViz USD schemas to enable:

- Dataset selection and field visualization
- Glyph-based visualizations (vectors, points)
- Surface extraction and rendering
- Streamline visualization
- Volume rendering
- Voxelization and point cloud processing

## Dependencies

This extension depends on:
- `omni.cae.schema` - Provides the core USD schemas including OmniCaeViz
- `omni.cae.data` - Provides data delegate infrastructure for reading CAE field arrays
- `omni.usd` - USD core functionality

## Usage

Import the extension:

```python
import omni.cae.viz
```

The extension automatically initializes when enabled and provides access to visualization utilities and algorithms.

## API Overview

The extension provides utilities for working with OmniCaeViz schemas:

- **Dataset Selection API**: Apply and manage dataset selections on prims
- **Field Selection API**: Select and configure field arrays for visualization
- **Visualization APIs**: Apply glyph, surface, streamline, and volume APIs to prims

## Development

To extend this extension:

1. Add new modules to the `python/impl/` directory
2. Register new commands in a `commands.py` module
3. Update the extension initialization in `extension.py` as needed

## See Also

- [OmniCaeViz Schema Documentation](../../usdSchema/README.md)
- [CAE Data Extension](../omni.cae.data/docs/README.md)
- [CAE Schema Extension](../omni.cae.schema/docs/README.md)
