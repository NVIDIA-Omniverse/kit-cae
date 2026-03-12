# CAE Visualization Extension - Technical Overview

## Introduction

The `omni.cae.viz` extension provides a framework for visualizing Computer-Aided Engineering (CAE) data in Omniverse using the OmniCaeViz USD schemas.

## Architecture

### Schema Integration

This extension works closely with the OmniCaeViz USD schemas which define API schemas for:

1. **DatasetSelectionAPI** - Multiple-apply schema for selecting CaeDataSet prims
2. **FieldSelectionAPI** - Multiple-apply schema for selecting field arrays with various modes
3. **GlyphAPI** - Single-apply schema for glyph-based visualizations
4. **SurfaceAPI** - Single-apply schema for extracted surfaces
5. **StreamlinesAPI** - Single-apply schema for streamline visualizations
6. **VolumeAPI** - Single-apply schema for volume rendering
7. **VoxelizationAPI** - Single-apply schema for voxelization settings

### Data Flow

1. CAE datasets are represented using `CaeDataSet` prims (from omni.cae.schema)
2. Field arrays are stored as `CaeFieldArray` prims with various format-specific subtypes
3. Visualization APIs from OmniCaeViz are applied to prims (Mesh, BasisCurves, Points, Volume)
4. The omni.cae.data extension provides the infrastructure to read field data
5. This extension provides utilities and algorithms to process and visualize the data

## Extension Components

### Extension Lifecycle

The extension follows the standard Omniverse extension lifecycle:
- `on_startup()` - Initialize the extension, register commands
- `on_shutdown()` - Clean up resources, unregister commands

### Future Components

Future versions of this extension may include:

- **Commands Module**: Kit commands for applying visualization APIs
- **Algorithms Module**: Processing algorithms for CAE data
- **UI Module**: User interface widgets for visualization control
- **Utilities Module**: Helper functions for working with OmniCaeViz schemas

## Design Principles

1. **Schema-Driven**: All visualization parameters are stored in USD using OmniCaeViz schemas
2. **Modular**: Components can be used independently
3. **Extensible**: Easy to add new visualization types and algorithms
4. **Performance**: Leverage GPU acceleration where possible (via Warp, Flow, etc.)

## Integration Points

### With omni.cae.schema
- Uses OmniCaeViz USD schemas for visualization parameters
- Reads CaeDataSet and CaeFieldArray prims

### With omni.cae.data
- Uses data delegates to read field array data
- Leverages caching and intermediate result storage
- Uses voxelization and other data processing utilities

## Examples

See the individual algorithm and command documentation for usage examples.
