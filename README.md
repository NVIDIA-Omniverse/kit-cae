# Kit-CAE: Reference Application

![Kit-CAE Banner](./docs/kit-cae-gh-banner.png)

Kit-CAE is a reference application demonstrating a set of technologies for CAE and scientific data workflows. It shows how OpenUSD can serve as a composition and access layer over engineering data from solvers, AI surrogates, sensors, and more, all while keeping source files in their native format. No conversion or copying is required, so data provenance and fidelity are preserved.

The key technologies integrated in Kit-CAE include:

- **OpenUSD Schemas & Data Delegates**: An open-standard foundation for composing and accessing scientific datasets without conversion. Reference implementations are provided for CGNS, EnSight, VTK, OpenFOAM, and others, but the same approach extends to any format
- **NVIDIA Warp**: GPU-accelerated data processing and visualization algorithms for interactive exploration of large datasets
- **RTX & IndeX Rendering**: High-fidelity visualization of engineering data including surfaces, volumes, and particles
- **Kit Application Framework**: Extensible UI, pixel streaming, and integration with the broader set of Omniverse libraries

Each technology can be used independently. The repository is structured to make that separation clear. Developers are encouraged to use Kit-CAE as a testbed for integrating their own tools with any of the enablement technologies provided.

## Quick Start

### Build

```sh
# On Linux
./repo.sh build -r

# On Windows
repo.bat build -r
```

USD schema source code is generated during fetch and compiled automatically during the build process.

See [Build Instructions](./docs/Build.md) for details on prerequisites, optional dependencies, and Kit SDK version selection.

> **Note:** Some format support (such as VTK) requires optional pip dependencies that are not included in the default build. See [Optional Dependencies](./docs/Build.md#optional-dependencies) in the build docs for setup instructions.

### Launch

```sh
# Basic application
./repo.sh launch -n omni.cae.kit

# With VTK support (see note above about additional steps before launching)
./repo.sh launch -n omni.cae_vtk.kit
```

See [Launch Instructions](./docs/Launch.md) for running sample scripts and detailed launch options.

## Documentation

### Getting Started
- **[Build Instructions](./docs/Build.md)** - Prerequisites, building, and dependencies
- **[Launch Instructions](./docs/Launch.md)** - Launching the app and running samples
- **[User Guide](https://docs.omniverse.nvidia.com/guide-kit-cae/latest/index.html)** - Step-by-step feature walkthroughs
- **[Reference Guide](./docs/Misc.md)** - Assortment of notes on some topics for power users / developers.

### Architecture & Design
- **[USD Schemas](./docs/UsdSchemas.md)** - CAE-specific USD schema design
- **[CAE Viz Schemas](./docs/CaeVizSchemas.md)** - Visualization operator schemas (`omni.cae.viz`)
- **[Extensions Overview](./docs/Extensions.md)** - Omniverse extensions architecture
- **[Data Delegate API](./docs/DataDelegate.md)** - Data access abstraction layer
- **[Format Onboarding Tutorial](./docs/FormatOnboarding.md)** - End-to-end custom format integration walkthrough

### Advanced Topics
- **[Integration Guide](./docs/Integration.md)** - Combining with Kit Application Template apps
- **[Kit SDK Version Selection](./docs/SelectKitVersion.md)** - Managing Kit SDK versions

## License

Development using the Omniverse Kit SDK is subject to licensing terms detailed [here](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).

This project uses several open-source libraries:

1. [CGNS](./tpl_licenses/cgns-LICENSE.txt) - CFD Notation System
2. [h5py](./tpl_licenses/h5py-LICENSE.txt) - Python interface for HDF5
3. [HDF5](./tpl_licenses/hdf5-LICENSE.txt) - High performance data software library
4. [VTK](./tpl_licenses/vtk-LICENSE.txt) - Visualization Toolkit
5. [Zlib](./tpl_licenses/zlib-LICENSE.txt) - General purpose data compression library
6. [pyparsing](./tpl_licenses/pyparsing-LICENSE.txt) - Python Parsing
7. [nvtx](./tpl_licenses/nvtx-LICENSE.txt) - NVTX Code Annotation
8. [lz4](./tpl_licenses/lz4-LICENSE.txt) - LZ4 compression library
9. [trimesh](./tpl_licenses/trimesh-LICENSE.md) - Trimesh library

Review the license terms of these open source projects before use.

## Agent Skills

Kit-CAE ships with agent skills in `skills/` that let agents (Codex, Claude Code, OpenClaw, Cursor, etc.) drive Kit-CAE programmatically: importing data, setting up visualizations, and capturing renders without manual intervention.

### Onboarding

1. Point your agent's skill/tool search path at `<kit-cae-dir>/skills/`.
2. The agent reads the relevant `SKILL.md` on demand — no further configuration needed.

| Skill | Purpose |
|-------|--------|
| `cae-core` | Shared foundation: APIs, stage discovery, format references, utility scripts. Loaded as a dependency — not invoked directly. |
| `cae-visualization` | End-to-end visualization: inspect data → import → create viz operators → render. |
| `cae-capture` | Production capture: clean PNG/EXR screenshots and MP4 video with NVENC encoding. |
| `cae-streaming` | Run Kit-CAE as a WebRTC streamed app; remote clients view and control it via data-channel messages. |

Skills reference each other via relative paths (e.g., `cae-core/references/kit-cae-api.md`), so the `skills/` directory is self-contained and portable.

### Onboarding Prompt

Paste this into Codex, Claude Code, OpenClaw, Cursor, or any skills-aware agent after pointing it at this repo:

```text
This repository ships agent skills under `skills/` (cae-core, cae-visualization, cae-capture, cae-streaming). Read each SKILL.md, run any preflight or environment checks the skills prescribe, and confirm Kit-CAE is ready to use on this system. Then give me a short summary of what each skill enables so I know what to ask you to do next.
```

## Contributing

We provide this source code as-is and are currently not accepting outside contributions.
