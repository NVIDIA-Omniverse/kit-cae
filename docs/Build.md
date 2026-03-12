# Build Instructions

This document provides detailed instructions for building Kit-CAE on Windows and Linux platforms.

## Prerequisites

### Windows

Visual Studio 2019 or 2022, along with the corresponding Windows SDK and build tools for C++ applications, must be installed on your system.

### Linux

Standard development tools including GCC/Clang compiler toolchain.

## Building Kit-CAE

### Quick Start

**On Windows:**

```sh
repo.bat build -r
```

**On Linux:**

```sh
./repo.sh build -r
```

Schema source code is automatically generated during fetch and compiled during build.

### Configuration Options

On Windows, you can edit [repo.toml](../repo.toml) to set `vs_version = "vs2019"` or `vs_version = "vs2022"` to select a specific Visual Studio version, or pass it on the command line:

```sh
repo.bat --set-token vs_version=vs2022 build -r
```

Use `./repo.sh --help` or `./repo.sh [tool] --help` to view all available build options.

## Optional Dependencies

Some extensions require external Python packages that are not bundled with Kit-CAE.

### Which features need optional dependencies

| Package | Version | Extensions | Features |
|---------|---------|------------|----------|
| vtk | 9.4 | omni.cae.vtk_libs | VTK file format import and data delegate |
| h5py | 3.15.1 | omni.cae.delegate.edem | EDEM file format import |

Without these packages, the corresponding features are disabled with an error in the console. All other Kit-CAE functionality works normally.

### Installing optional dependencies

```sh
# On Windows
repo.bat pip_download

# On Linux
./repo.sh pip_download
```

This installs the optional packages into the build directory. After running this command, relaunch the application and the features will be available. No additional launch flags are needed.

## Selecting Kit SDK Version

Kit-CAE can be built against different versions of the Omniverse Kit SDK. Available versions are managed in `tools/kit-versions.json`.

### Interactive Selection

```sh
# On Linux
./repo.sh select_kit_version

# On Windows
repo.bat select_kit_version
```

This displays available Kit versions and prompts you to select one. Your selection is saved to `.kit_selection.json`.

### Non-Interactive Selection

```sh
# Select a specific version
./repo.sh select_kit_version --version 108.0.0

# Use default version
./repo.sh select_kit_version --default

# Use tracked version (ideal for CI/CD)
./repo.sh select_kit_version --auto
```

### Clean Builds After Version Changes

**After changing Kit SDK versions, you must perform clean builds:**

```sh
# On Linux
./repo.sh build -r -x # or use -rx, instead of -r -x

# On Windows
repo.bat build -r -x  # or use -rx, instead of -r -x
```

These commands:
- `build -x`: Perform a clean build of all extensions

For additional details see [Selecting Kit SDK Version](./SelectKitVersion.md).
