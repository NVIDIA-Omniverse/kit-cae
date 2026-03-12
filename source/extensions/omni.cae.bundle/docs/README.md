# CAE Extension Bundle

## Overview

The `omni.cae.bundle` extension is a convenience bundle that aggregates all CAE-related extensions, dependencies, and settings required for Kit-CAE applications. This extension simplifies application configuration by providing a single extension that can be referenced instead of listing all individual CAE components.

It's direct dependencies include extensions part of Kit-CAE itself as well some standard Kit SDK extensions beyond the default
Kit Editor that make sense for Kit-CAE use-cases.

## Settings

This extension configures several important settings:

### Material Library
Extends material options to include MDL file support.

### IndeX Settings
- Subdivision mode: KD-tree
- Subdivision part count: 1
- Composite rendering: Enabled

### Hydra/Fabric Settings
Enables transform reading from Fabric in render delegate for usdrt-based algorithms.

### Flow Settings
Configures voxelization parameters for velocity fields.

## Usage

Simply add this extension as a dependency in your `.kit` file:

```toml
[dependencies]
"omni.cae.bundle" = {}
```

This will automatically load all CAE extensions and apply the necessary settings.
