# Reference Guide

This guide covers features and utilities in Kit-CAE that don't belong to a single subsystem but are useful to know when extending or integrating with the application.

**Contents**

- [Settings](#settings) — persistent user settings and command-line extension settings
- [Colormap Textures](#colormap-textures) — automatic dynamic texture publishing for `Colormap` prims
- [Configuring Warp](#configuring-warp) — `warp.config` overrides and the Blackwell PTX workaround
- [Precompiling Warp Kernels](#precompiling-warp-kernels-experimental) — ahead-of-time kernel compilation for production deployments

---

## Settings

Kit-CAE provides two types of settings: persistent user settings and non-persistent extension settings.

### User Settings (Persistent)

User settings are preserved between application launches and can be accessed through the Preferences panel:

1. Open **Edit > Preferences** from the main menu
2. Navigate to the **CAE** page
3. Modify settings as needed

These settings are saved to your user configuration and will persist across sessions.

### Extension Settings (Non-Persistent)

Extensions expose developer-focused settings that are typically non-persistent. These settings are passed via command-line arguments when launching the application:

```sh
# On Linux
./repo.sh launch -n omni.cae.kit -- --/foo/bar=12

# On Windows
repo.bat launch -n omni.cae.kit -- --/foo/bar=12
```

**Format:** Settings follow the pattern `--/path/to/setting=value` where the path corresponds to the extension's settings hierarchy.

**Example:**

```sh
# Set pip archive directory
./repo.sh launch -n omni.cae_vtk.kit -- --/exts/omni.kit.pipapi/archiveDirs=[/tmp/pip_archives]
```

These command-line settings override default values but are not saved between sessions unless configured in application `.kit` files.

## Colormap Textures

CAE scenes use `Colormap` prims as the canonical description of a scientific color ramp. The NVIDIA IndeX renderer for volume visualization can consume these prims directly — it reads `rgbaPoints` and `xPoints` natively. However, MDL shaders used for surfaces, streamlines, and other geometry-based representations expect a **1-D texture asset** for their LUT input rather than a USD prim.

This creates a split: the same colormap definition must be accessible in two different forms depending on the renderer. Without automation, a developer would have to bake a texture file for every colormap, keep it synchronized whenever the ramp changes, and wire up each shader manually.

`ColormapTextureManager` eliminates this by automatically publishing every `Colormap` prim that has `CaeVizColormapTextureAPI` applied as a `dynamic://` texture at runtime. Any MDL shader that needs the LUT can bind it by URL — derived from a stable identifier stored on the prim — without any manual bake step. The texture is regenerated automatically whenever the prim's control points change, so IndeX and MDL shaders always see the same ramp.

### How it works

`ColormapTextureManager` (a process-wide singleton owned by the extension) subscribes to stage attach/detach events and USD object-change notices. It only tracks `Colormap` prims that have `CaeVizColormapTextureAPI` applied. Whenever such a prim is added, removed, or modified, it:

1. Reads the prim's `cae:viz:colormapTexture:identifier` attribute.
2. Reads `rgbaPoints` (Nx4 float) and `xPoints` (N float) and samples them into a uniform 256-sample 1-D RGBA LUT.
3. Uploads the result to an `omni.ui.DynamicTextureProvider` named `cae_colormap_<identifier>`.

### Texture URL convention

The dynamic texture URL is derived from the prim's `identifier` attribute (a UUID set once at creation), not from its path. This makes the URL stable under USD composition — the prim can be relocated via references or payloads without breaking any shaders that bind the URL.

| `identifier` value | Dynamic texture URL |
|---|---|
| `a1b2c3d4e5f6...` | `dynamic://cae_colormap_a1b2c3d4e5f6...` |

`CaeVizColormapTextureAPI` is applied automatically when a `Colormap` prim is created through Kit-CAE's create commands.

### Python API

Use `get_dynamic_url_for_identifier` to construct the texture URL from a known identifier:

```python
from omni.cae.viz.colormap_texture_manager import get_dynamic_url_for_identifier

identifier = colormap_prim.GetAttribute("cae:viz:colormapTexture:identifier").Get()
url = get_dynamic_url_for_identifier(identifier)   # "dynamic://cae_colormap_<identifier>"
```

To check whether a texture has been registered or retrieve its entry from the running manager:

```python
from omni.cae.viz.colormap_texture_manager import ColormapTextureManager

mgr = ColormapTextureManager.get_instance()   # None if extension not loaded

if mgr and mgr.has_colormap("/World/Foo/Material/Colormap"):
    url = mgr.get_dynamic_url("/World/Foo/Material/Colormap")
```

### Copying the URL from the stage

Right-clicking a `Colormap` prim in the stage panel shows a **Copy LUT Texture URL** option. This copies the
`dynamic://` URL for that prim directly to the clipboard, so it can be pasted into any MDL shader's LUT input
without constructing the path by hand.

### Binding in an MDL shader

Pass the URL returned by `get_dynamic_url_for_colormap_path` to any shader input that accepts a texture asset. In Python:

```python
shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Foo/Material/Shader"))
shader.GetInput("lut").Set(Sdf.AssetPath(get_dynamic_url_for_colormap_path(colormap_prim)))
```

The texture is available as soon as the `Colormap` prim exists on the stage — no explicit refresh call is needed.

---

## Configuring Warp

Kit-CAE uses [Warp](https://nvidia.github.io/warp/index.html) quite extensively for implementing most (if not all) of the data transformation operations needed to prepare data for rendering. Warp exposes several package-level options that control how the CUDA / CPU kernels are generated.

Kit-CAE initializes Warp on startup and applies a set of non-persistent settings (under `/exts/omni.cae.data/warp/`) that map directly to `warp.config` attributes. These can be set in a `.kit` file or via command-line arguments (see [Extension Settings](#extension-settings-non-persistent) above).

### Blackwell GPU workaround

On CUDA compute architectures ≥ 100 (Blackwell and later), the CUDA 12.x toolchain currently bundled with Kit compiles kernels too slowly. Kit-CAE automatically works around this by forcing PTX output targeting `sm_90`:

```toml
# Applied automatically on Blackwell — equivalent to:
[settings]
exts."omni.cae.data".warp.cudaOutput = "ptx"
exts."omni.cae.data".warp.ptxTargetArch = 90
```

To opt out of this automatic override (e.g. to test native Blackwell compilation):

```sh
./repo.sh launch -n omni.cae.kit -- --/exts/omni.cae.data/warp/skipBlackwellPtxOverride=true
```

### Available `warp.config` overrides

The following settings are supported. Each is optional — if not defined, the corresponding `warp.config` attribute is left at its default value.

| Setting path | `warp.config` attribute | Type | Description |
|---|---|---|---|
| `/exts/omni.cae.data/warp/mode` | `mode` | string | Warp execution mode (e.g. `"kernel"`) |
| `/exts/omni.cae.data/warp/verifyFp` | `verify_fp` | bool | Enable floating-point verification |
| `/exts/omni.cae.data/warp/verifyCuda` | `verify_cuda` | bool | Enable CUDA error verification |
| `/exts/omni.cae.data/warp/verbose` | `verbose` | bool | Enable verbose Warp output |
| `/exts/omni.cae.data/warp/verboseWarnings` | `verbose_warnings` | bool | Enable verbose Warp warnings |
| `/exts/omni.cae.data/warp/ptxTargetArch` | `ptx_target_arch` | int | Target SM architecture for PTX compilation |
| `/exts/omni.cae.data/warp/maxUnroll` | `max_unroll` | int | Maximum loop unroll factor |
| `/exts/omni.cae.data/warp/cudaOutput` | `cuda_output` | string | CUDA output format (`"ptx"` or `"cubin"`) |
| `/exts/omni.cae.data/warp/skipBlackwellPtxOverride` | *(guard)* | bool | Skip the automatic Blackwell PTX workaround |

**Example** — enable verbose output and force PTX compilation:

```sh
./repo.sh launch -n omni.cae.kit -- \
    --/exts/omni.cae.data/warp/verbose=true \
    --/exts/omni.cae.data/warp/cudaOutput=ptx
```

Or in a `.kit` file:

```toml
[settings]
exts."omni.cae.data".warp.verbose = true
exts."omni.cae.data".warp.cudaOutput = "ptx"
```

## Precompiling Warp kernels (Experimental)

Warp compiles kernels the first time they are executed (just-in-time). For production deployments or automated testing it may be preferable to front-load that compilation and ship pre-built kernel cache entries instead. Kit-CAE supports this via a two-step workflow:

### Step 1 — Record an AOT configuration

The kernel inputs (mesh topology, data types, device targets, etc.) that are needed to compile each kernel ahead-of-time are captured automatically while the application runs normally. Once you have exercised the data you care about, open the preferences panel and use the **AOT Configuration Recorder** section under **Edit > Preferences > CAE**:

1. Set the **Output Path** to the desired location for the JSON file (default: `aot_config.json`).
2. Click **Save AOT Config** — this writes the current recorder state to the file.

The file defaults to `aot_config.json` at the repository root, which is the path `repo_precompile_kernels` looks for by default (see `repo.toml`).

### Step 2 — Precompile using the repo tool

Run the `precompile_kernels` repo tool, pointing it at the JSON file produced in step 1:

```sh
# Linux — uses aot_config.json and compiles for cuda + cpu (as configured in repo.toml)
./repo.sh precompile_kernels

# Override the JSON path and/or device list on the command line
./repo.sh precompile_kernels --json /path/to/my_config.json --devices cuda cpu
```

On Windows replace `./repo.sh` with `repo.bat`.

The tool launches Kit internally, loads the DAV extensions, and runs `dav.aot_compile.compile()` against the recorded configuration. The resulting compiled kernels are written to Warp's kernel cache (controlled by `WARP_CACHE_PATH` or `repo.toml`'s `kernel_cache_dir`).

### `repo.toml` reference

The relevant defaults in `repo.toml`:

```toml
[repo_precompile_kernels]
enabled = true
devices = ["cuda", "cpu"]
json = "${root}/aot_config.json"
# kernel_cache_dir = "${root}/_build/kernel_cache"  # uncomment to fix the cache location
```

All three options can be overridden on the command line (`--devices`, `--json`, `--kernel-cache-dir`).
