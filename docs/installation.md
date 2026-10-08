# Installation

## Prerequisites

Use Linux x86-64 with glibc 2.35 or newer (Ubuntu 22.04+ or equivalent),
CPython 3.12, and an RTX-capable NVIDIA GPU with a compatible driver for the
documented workflows.
Rendering requires an application-provided renderer such as OVRTX, which is
not part of the base install.

On minimal Debian-based images, install `libgl1`, `libx11-6`, and `libgomp1`
before importing OVStage. Its [native runtime](https://github.com/NVIDIA-Omniverse/ovstage#packaging-and-dependencies)
requires these host libraries even in headless processes.

Install this experimental branch from a source checkout. All dependencies,
including Warp SimData and CAE OpenUSD Plugins, come from public PyPI.
No package-index credentials are needed.

## Install from source

Clone the repository and let `uv` build the environment with the project's
Python version and pinned dependencies:

```console
git clone https://github.com/NVIDIA-Omniverse/kit-cae.git
cd kit-cae
git checkout experimental-ovcae
uv sync --frozen --no-dev
```

This creates `.venv` with OVCAE and the extras you asked for. `--frozen` uses
the checked-in `uv.lock` without updating it. `--no-dev` leaves out the test
and lint tooling; omit `--no-dev` when working on OVCAE itself.

| Install | Adds |
| --- | --- |
| No extras | The `ovcae` package and every subpackage, OVStage, NumPy, Warp, Warp SimData, and usd-core |
| `--extra usd` | The CAE OpenUSD Plugins, for `ovcae.readers` and file-backed payloads |
| `--extra stream` | OVStream, for `ovcae.stream` to send frames to a browser over WebRTC |
| `--group render-test` | OVRTX, for rendering |

Evaluation and recipe authoring need no extra. Reading datasets in their
native formats, whether at authoring time or through `ovcae.readers`, needs
`usd`. Browser streaming also needs `stream` and OVRTX; add the
`render-test` group for the renderer version tested with this checkout.
Every subpackage is installed; extras supply optional dependencies.

## Install into another environment

Build a wheel in the checkout:

```console
uv build
```

Activate the target virtual environment, then install the wheel with the extras
your application needs:

```console
uv pip install "dist/ovcae-0.1.0.dev0-py3-none-any.whl[usd,stream]"
```

Omit `[usd,stream]` for the base install. Install OVRTX separately if your
application renders; its tested version is in the `render-test` group in
[`pyproject.toml`](../pyproject.toml).

## Dependency ranges

These ranges come from the distribution metadata. The repository's `uv.lock`
records the exact development environment; consumers are not required to use
those exact versions.

| Dependency | Declared range |
| --- | --- |
| Python | `>=3.12,<3.13` |
| NumPy | `>=2.2,<3` |
| OVStage | `>=0.2.0.377349,<0.3` |
| Warp | `>=1.17,<2` |
| Warp SimData | `>=0.2.0,<0.3` |
| usd-core | `>=25.11,<=26.8` |
| CAE OpenUSD Plugins (`usd`) | `>=0.1.2,<0.2` |
| OVStream (`stream`) | `>=0.4.5,<0.5` |

The CAE plug-in wheel bundles native payloads for usd-core 25.11, 26.3, 26.5,
and 26.8 and selects the payload matching the installed runtime when it
registers, which is why OVCAE declares a usd-core range rather than one exact
version.

## Initialize your application

Importing `ovcae` registers nothing. Before creating any stage, call
`ovcae.setup()`. It imports every installed reader package so usd-core knows
their file formats, registers the codeless schemas with usd-core and with
OVStage, and adds the bundled MDL material library to `MDL_USER_PATH`. Setup is idempotent. A reader whose
optional dependencies are missing is skipped with a warning; pass `strict=True`
to fail instead. Applications that need finer control can call
`ovcae.load_readers()`, `ovcae.register_schema_paths()`, and
`ovcae.register_material_library()` separately.

When rendering with OVRTX, call `ovrtx.register_schema_paths()` first, then
`ovcae.setup()`, before creating any stage or renderer. Create the renderer
before populating the OVStage so bundled colormap paths resolve. The MDL SDK
reads `MDL_USER_PATH` when the renderer initializes; registering materials
afterwards does not reach that renderer.

Importing `ovcae.authoring` registers the codeless schemas with usd-core, and
importing `ovcae.readers` registers the CAE file formats. usd-core fixes its
file-format registry the first time a stage is created and its schema registry
on the first schema read, so import `ovcae.readers`, or call `ovcae.setup()`,
and register any custom operator schemas with `ovcae.register_schema_paths`,
before any stage is created.

## Verify the installation

```console
.venv/bin/python -c "import ovcae, ovcae.schemas, ovcae.materials; print(ovcae.__version__)"
```

Then run the [getting-started tutorial](getting-started.md), which evaluates
your first recipe on the GPU using in-memory data.
