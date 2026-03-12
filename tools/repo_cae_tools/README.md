# repo_cae_tools

Custom CAE-specific tools for the Kit-CAE repository management system.

## Overview

This package contains custom repoman tools specific to the Kit-CAE project. It follows the same structure as `repo_kit_tools` to provide a clean separation between upstream template tools and project-specific customizations.

## Tools Included

### Custom Build (`build_cae`)

### Kit Version Configuration (`select_kit_version`)
Manages Kit SDK version selection and generates files from templates based on the selected Kit SDK version.

**Features:**
- Interactive version selection with detailed version information
- Generates packman XML files from `.template` files with variable substitution
- Processes Jinja2 templates in app configuration files
- Creates symlinks for static files (development) or copies them (packaging)
- Validates generated files against expected configuration
- JSON output for automation and CI/CD integration

**Basic Usage:**
```bash
# Interactive selection (generates files immediately)
./repo.sh select_kit_version

# Select specific version (generates files)
./repo.sh select_kit_version --version 108.0.0

# Use default version (generates files)
./repo.sh select_kit_version --default

# Use tracked or default version (automation-friendly)
./repo.sh select_kit_version --auto

# List available versions
./repo.sh select_kit_version --list-versions

# List available versions in JSON format
./repo.sh select_kit_version --list-versions --json
```

**Generation and Validation:**
```bash
# Validate current files against tracked version
./repo.sh select_kit_version --dry-run

# Preview what selecting a version would do
./repo.sh select_kit_version --version 108.0.0 --dry-run

# Preview what auto mode would do
./repo.sh select_kit_version --auto --dry-run

# Generate with copies instead of symlinks (for packaging, sticky)
./repo.sh select_kit_version --auto --no-use-symlinks
```

**Configuration Display:**
```bash
# Show current tool configuration
./repo.sh select_kit_version --show-config
```

**Automation and CI/CD:**
```bash
# Quiet mode for automation (suppresses non-essential output)
./repo.sh select_kit_version --auto --quiet

# JSON output for parsing
./repo.sh select_kit_version --list-versions --json
```

**Configuration (in `repo_tools.toml`):**

The tool is highly configurable. All settings are under the `[repo_select_kit_version]` section:

```toml
[repo_select_kit_version]
enabled = false  # Set to true to enable the tool

# Path to kit-versions.json configuration file
# This file defines available Kit versions and their templates
versions_config_file = "${root}/tools/kit-versions.json"

# Path to state file (git-ignored)
# Stores the currently selected Kit version and file mode
tracking_file = "${root}/.kit_selection.json"
```

**Configuration Options Explained:**

- **`versions_config_file`**: Path to JSON file defining available Kit versions, their variables, and templates
- **`tracking_file`**: Git-ignored file that tracks the currently selected version

**Kit Versions Configuration (`kit-versions.json`):**

The `kit-versions.json` file defines available Kit versions and their associated templates:

```json
{
  "common": {
    "templates": [
      {
        "src": "${root}/tools/deps/kit-sdk.packman.xml.template",
        "dest": "${root}/tools/deps/kit-sdk.packman.xml"
      }
    ]
  },
  "versions": {
    "108.0.0": {
      "variables": {
        "KIT_VERSION_TAG": "108.0.0+release.123456.abcdef",
        "INDEX_HEADERS_TAG": "387500.2199"
      },
      "templates": [
        {
          "src": "${root}/templates/kit_cae/kit_cae_editor.kit",
          "dest": "${root}/apps/kit_cae_editor.kit"
        }
      ]
    }
  },
  "default": "108.0.0"
}
```

**Template Processing:**

The tool processes three types of templates:

1. **`.template` files**: Performs `@@VARIABLE@@` substitution using version variables
2. **Jinja2 files**: Processes `{{ version }}` tags and comments out other Jinja2 constructs
3. **Plain files**: Creates symlinks (default) or copies (with `--no-use-symlinks`)

**Exit Codes:**

- `0`: Success (operation completed successfully)
- `1`: Error (validation failed, file errors, or configuration issues)
- `2`: User cancelled operation (Ctrl+C in interactive mode)

**Workflow:**

1. Run `select_kit_version` to choose a Kit version (generates files immediately)
2. The tool saves your selection to `.kit_selection.json` (git-ignored)
3. Use `--dry-run` to validate files without making changes
4. Use `--no-use-symlinks` when preparing for packaging (sticky — persisted for future calls)
5. Use `--auto` in automation/CI to use tracked or default version

**Note:** You can override these settings in your project's `repo.toml` file under the `[repo_select_kit_version]` section.

### Schema Generation (`schema`)
Manages USD schema generation and building for CAE-specific schemas.

**Usage:**
```bash
# Generate and build all schemas
./repo.sh schema

# Only fetch dependencies
./repo.sh schema --fetch-only

# Only generate schemas (without building)
./repo.sh schema --generate-only

# Only build schemas (assumes already generated)
./repo.sh schema --build-only

# Clean before running
./repo.sh schema --clean

# Clean only (no generation or building)
./repo.sh schema --clean-only

# Windows: Use Visual Studio 2022 instead of 2019
./repo.sh schema --vs2022
```

**Configuration (in `repo_tools.toml`):**

The schema tool is highly configurable. All settings are under the `[repo_schema]` section:

```toml
[repo_schema]
# Schema generation and build directories
schema_generated_source_root = "${root}/_schemas/source"
schema_build_root = "${root}/_schemas/build"
schema_install_root = "${root}/_schemas/install"

# List of packman dependency files to pull (in order)
dependencies_packman_xml = [
  "${root}/tools/deps/kit-sdk.packman.xml",
  "${root}/tools/deps/kit-sdk-schema-deps.packman.xml",
  "${root}/tools/deps/kit-cae-deps.packman.xml",
]

# CMake root directory (executable will be located at cmake_root/bin/cmake[.exe])
cmake_root = "${root}/_build/host-deps/cmake"

# Repo USD template CMake directory (used for schema generation)
repo_usd_template_cmake_dir = "${root}/_repo/deps/repo_usd/templates/cmake"
```

**Configuration Options Explained:**

- **`schema_generated_source_root`**: Directory where USD schema code is generated
- **`schema_build_root`**: Directory where schemas are built
- **`schema_install_root`**: Directory where built schemas are installed
- **`dependencies_packman_xml`**: List of packman XML files to pull for dependencies
- **`cmake_root`**: Root directory of cmake installation (executable path is automatically determined as `cmake_root/bin/cmake` or `cmake_root/bin/cmake.exe` on Windows)
- **`repo_usd_template_cmake_dir`**: Directory containing USD CMake templates

**Note:** The schema tool includes its own CMake setup templates packaged in `omni/repo/cae_tools/templates/`. These templates are automatically located and used by the tool.

### Version Bump (`bump`)
Extends the standard bump tool to support bumping package versions in addition to extension versions.

**Usage:**
```bash
./repo.sh bump
```

### Custom Build (`build_cae`)
Custom build tool for Kit-CAE with CAE-specific build steps.

**Status:** Template/boilerplate - ready for custom implementation

**Usage:**
```bash
./repo.sh build_cae
```

**Configuration (in `repo_tools.toml`):**

```toml
[repo_build_cae]
enabled = false  # Set to true when ready to use

# Add build-specific configuration here
```

### Pip Download (`pip_download`)
Tool for downloading pip package archives.

**Usage:**
```bash
# Download a specific package
./repo.sh pip_download numpy

# Download with specific version
./repo.sh pip_download "numpy==1.21.0"

# Download from a requirements file
./repo.sh pip_download -r requirements.txt

# Download from configured requirements files
./repo.sh pip_download

# Clean archives directory before downloading
./repo.sh pip_download --clean

# Specify custom destination
./repo.sh pip_download --dest /path/to/archives numpy
```

**Configuration (in `repo_tools.toml`):**

```toml
[repo_pip_download]
# Directory where pip archives will be downloaded
archives_dir = "${root}/_pip_archives"

# List of requirements files to download (when no package is specified)
requirements_files = [
    "${root}/requirements.txt",
    "${root}/requirements-dev.txt",
]

# Python executable path (resolved automatically from target-deps)
python_root = "${root}/_build/target-deps/python"
```

**Configuration Options Explained:**

- **`archives_dir`**: Directory where downloaded pip archives (.whl, .tar.gz) will be saved
- **`requirements_files`**: List of requirements.txt files to process when no package is specified on command line
- **`python_root`**: Root directory of python installation (executable path is automatically determined)

## Package Structure

```
tools/repo_cae_tools/
├── omni/
│   └── repo/
│       └── cae_tools/
│           ├── __init__.py
│           ├── build.py            # Custom build tool (boilerplate)
│           ├── bump.py
│           ├── helpers.py
│           ├── pip_download.py
│           ├── schema.py
│           ├── select_kit_version.py
│           └── templates/          # Schema build templates
│               ├── setup.cmake
│               ├── CMakeLists.txt.in
│               └── cmake/
│                   └── gccdefaults.cmake
├── repo_tools.toml
├── VERSION
├── LICENSE.txt
└── README.md
```

The `templates/` directory contains CMake scripts used by the schema tool for setting up and building USD schemas. These are automatically located by the tool at runtime.

## Helper Utilities

The `omni.repo.cae_tools.helpers` module provides common utilities for CAE tools:

### `invoke_tool(tool, args, tokens, exit_on_error, silent)`
Invoke another repo tool as a subprocess.

**Example:**
```python
from omni.repo.cae_tools.helpers import invoke_tool

# Call packman
invoke_tool("packman",
    args=["pull", "${root}/tools/deps/kit-sdk.packman.xml", "-p", "${platform}"],
    tokens=["config:release"]
)

# Call build
invoke_tool("build", args=["--config", "release"])
```

### `run_command(command, exit_on_error, silent)`
Run an arbitrary command (not a repo tool).

**Example:**
```python
from omni.repo.cae_tools.helpers import run_command

# Run cmake
run_command([
    "cmake",
    "-S", "${root}/source",
    "-B", "${root}/_build"
])
```

### `quiet_error(message)`
Raise a QuietExpectedError for user-facing error messages.

**Example:**
```python
from omni.repo.cae_tools.helpers import quiet_error

if not config_file.exists():
    quiet_error(f"Configuration file not found: {config_file}")
```

### `unresolve_tokens(path)`
Convert resolved paths back to token format (reverse of `resolve_tokens`).

This is useful when you need to pass paths with token placeholders to other tools,
even though you've already resolved them locally.

**Example:**
```python
from omni.repo.cae_tools.helpers import unresolve_tokens
from omni.repo.man import resolve_tokens

# Resolve a path
schema_root = resolve_tokens("${root}/_schemas/source")
# schema_root = "/home/user/project/_schemas/source"

# Unresolve it back to token format
token_path = unresolve_tokens(schema_root)
# token_path = "${root}/_schemas/source"

# Useful for passing to other tools that expect tokens
tokens = [f"schema_root:{unresolve_tokens(schema_root)}"]
```
