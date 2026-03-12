# Launch Instructions

This document describes how to launch Kit-CAE and run sample scripts.

## Launching the Application

### Basic Launch

After building (see [Build Instructions](./Build.md)), launch the application:

```sh
# On Windows
repo.bat launch -n omni.cae.kit

# On Linux
./repo.sh launch -n omni.cae.kit
```

### Launching with VTK Support

Kit-CAE can optionally use algorithms that require [VTK](https://vtk.org) for data processing:

```sh
# On Windows
repo.bat launch -n omni.cae_vtk.kit

# On Linux
./repo.sh launch -n omni.cae_vtk.kit
```

**Note:** VTK support requires the VTK pip package. See [Build Instructions - Installing Optional Dependencies](./Build.md#installing-optional-dependencies-using-pip) for details.

## Running Sample Scripts

Kit-CAE includes several sample scripts demonstrating various features. Scripts are located in the [scripts](../scripts/) directory.

### Basic Scripts

Run scripts that don't require VTK:

```sh
# On Linux
./repo.sh launch -n omni.cae.kit -- --exec scripts/example-bounding-box.py

# On Windows
repo.bat launch -n omni.cae.kit -- --exec scripts/example-bounding-box.py
```

### VTK-Dependent Scripts

For scripts requiring VTK, use the VTK-enabled application variant:

```sh
# On Linux
./repo.sh launch -n omni.cae_vtk.kit -- --exec scripts/example-streamlines.py

# On Windows
repo.bat launch -n omni.cae_vtk.kit -- --exec scripts/example-streamlines.py
```

### Available Sample Scripts

Browse the [scripts](../scripts/) directory to see all available examples, including:
- Bounding box calculations
- Streamline generation
- Data visualization workflows
- And more...

## User Guide

For step-by-step instructions on using Kit-CAE features and workflows, refer to the [Online User Guide](https://docs.omniverse.nvidia.com/guide-kit-cae/latest/index.html).
