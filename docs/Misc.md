# Miscellaneous

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
