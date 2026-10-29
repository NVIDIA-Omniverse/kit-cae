# Selecting Kit SDK Version

Kit-CAE can be built against different versions of the Omniverse Kit SDK. The available versions and their configurations
are managed in `tools/kit-versions.json`. Use the `select_kit_version` tool to manage Kit SDK version selection and
generate configuration files from templates.

Kit 110.4.1 is the default for new checkouts. Existing checkouts keep their saved
selection; run `./repo.sh select_kit_version --default` to switch to the new default.

## Quick Start

**Interactive Version Selection:**

```sh
# On Linux
./repo.sh select_kit_version

# On Windows
repo.bat select_kit_version
```

This displays a table of available Kit versions and prompts you to select one. Your selection is saved to
`.kit_selection.json` (git-ignored) and files are generated immediately from templates.

**Non-Interactive Selection:**

```sh
# Select a specific version and generate files
./repo.sh select_kit_version --version 110.4.1

# Use default version and generate files
./repo.sh select_kit_version --default

# Use tracked (or default if not tracked) version and generate files
# Ideal for automation and CI/CD
./repo.sh select_kit_version --auto
```

## Template Processing

The tool processes templates and generates configuration files:

- **Packman XML Templates** (`.template` files): Uses `@@VARIABLE@@` substitution with version-specific values
- **Jinja2 App Templates**: Processes `{{ version }}` tags and comments out other Jinja2 constructs
- **Static Files**: Creates symlinks by default (development) or copies with `--no-use-symlinks` (packaging)

**Note for Windows users:** Symlink creation requires Developer Mode (Windows 10/11) or Administrator rights. The tool automatically falls back to copying if symlinks fail.

## Validation and Dry-Run

Preview changes or validate existing files without modifying anything:

```sh
# Validate current files against tracked version
./repo.sh select_kit_version --dry-run

# Preview what selecting a version would do
./repo.sh select_kit_version --version 110.4.1 --dry-run

# Preview what auto mode would do
./repo.sh select_kit_version --auto --dry-run
```

The dry-run mode checks:

- Whether generated files exist
- If file types are correct (symlink vs. regular file)
- If content matches expected output
- If symlinks point to correct targets

## Listing Available Versions

```sh
# Display version information in a table
./repo.sh select_kit_version --list-versions

# Output in JSON format (for automation)
./repo.sh select_kit_version --list-versions --json
```

## Automation and CI/CD

For automated builds or CI/CD pipelines:

```sh
# Use tracked or default version, suppress non-essential output
./repo.sh select_kit_version --auto --quiet

# Explicitly use default version with minimal output
./repo.sh select_kit_version --default --quiet

# Get version info in JSON for parsing
./repo.sh select_kit_version --list-versions --json
```

The tool provides clear exit codes:

- `0`: Success
- `1`: Error (validation failed, missing files, etc.)
- `2`: User cancelled (interactive mode only)

## Clean Builds After Changing Kit Versions

**After changing Kit SDK versions, you must perform clean builds** to ensure all dependencies and generated files are updated
correctly:

```sh
# On Linux
./repo.sh schema --clean-only
./repo.sh build -r -x

# On Windows
repo.bat schema --clean-only
repo.bat build -r -x
```

These commands:

- `schema --clean-only`: Clean generated schema files (regenerated on next build)
- `build -x`: Perform a clean build of all extensions

This ensures compatibility with the newly selected Kit version and prevents runtime errors due to version mismatches or stale build artifacts.

## Symlinks vs. Copies for Packaging

The tool creates **absolute path symlinks** for static template files by default. This is ideal for development as files stay synchronized with templates during git operations. However, **symlinks cannot be used for packaging** because `repo_package` does not follow symlinks.

The file mode setting is **sticky** — once set, it is persisted in `.kit_selection.json` and reused on every subsequent invocation until explicitly changed.

**For Packaging:**

Switch to copy mode (sticky — stays active for future calls):

```sh
# Switch to copy mode
./repo.sh select_kit_version --auto --no-use-symlinks

# Now package as usual
./repo.sh package --thin -c release
```

**For Development:**

Switch back to symlink mode:

```sh
./repo.sh select_kit_version --auto --use-symlinks
```

Or simply omit the flag after the initial setup — the persisted mode will be used.

## Configuration

The tool can be configured in `repo.toml` under the `[repo_select_kit_version]` section:

```toml
[repo_select_kit_version]
# Path to kit-versions.json (defines available versions)
versions_config_file = "${root}/tools/kit-versions.json"

# Path to state file (stores selected version and file mode, git-ignored)
tracking_file = "${root}/.kit_selection.json"
```

To view the current configuration:

```sh
./repo.sh select_kit_version --show-config
```

## Adding New Kit Versions

To add support for a new Kit SDK version, edit `tools/kit-versions.json`:

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
    "110.4.1": {
      "variables": {
        "KIT_VERSION_TAG": "110.4.1+production",
        "INDEX_HEADERS_TAG": "380600.2218",
        "USD_VERSION_TAG": "0.25.11"
      },
      "templates": [
        {
          "src": "${root}/templates/110.4.1/kit_cae/kit_cae_editor.kit",
          "dest": "${root}/source/apps/kit_cae_editor.kit"
        }
      ]
    }
  },
  "default": "110.4.1"
}
```

The structure includes:

- **`common.templates`**: Templates applied to all versions
- **`versions.<label>.variables`**: Version-specific variable substitutions
- **`versions.<label>.templates`**: Version-specific template files
- **`default`**: Default version label to use

### Validate a new SDK's dependency locks

Resolve editor and streaming dependencies with the selected SDK's default
registries. Do not force the shared registry to the Kit major version: SDKs
such as 110.4.1 use a minor-version registry (`110.4/shared`). A lock generated
against a different registry can build locally but fail when a clean thin
package downloads its dependencies.

Regenerate the new SDK's locks with an empty extension cache, then validate the
thin package with `./repo.sh test --config release --from-package`. This checks
the package's dependency-download path and installs the optional test packages;
testing only the development build does not exercise that path.
