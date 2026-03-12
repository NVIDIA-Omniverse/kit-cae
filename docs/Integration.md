# Integrating Kit-CAE with KAT-based Applications

This document describes how to combine Kit-CAE with other
[Kit Application Template (KAT)](https://github.com/NVIDIA-Omniverse/kit-app-template)-based applications.
So long as the Kit Kernel versions match, Kit-CAE extensions can be brought into another Kit application either
at runtime or at build time. This document describes both options.

## Prerequisites

### Verify Kit Kernel Compatibility

Check that Kit Kernel versions match between Kit-CAE and your application. As of 2.0, Kit-CAE can be made to target
a specific version of Kit using the `select_kit_version` tool (see [Build Instructions](./Build.md)). Confirm that the
build you're using matches the Kit Kernel version of your application.

## Runtime-only Integration (Local Testing)

For local development and testing:

### 1. Build Both Applications

```sh
# Build Kit-CAE (see Build Instructions)
cd <path-to-kit-cae>
./repo.sh schema
./repo.sh build -r

# Build your KAT application
cd <path-to-your-kat-app>
./repo.sh build
```

### 2. Launch with Kit-CAE Extensions

From your KAT application directory:

```sh
# On Linux
./repo.sh launch -n <your-app> -- \
      --ext-folder <path-to-kit-cae>/_build/linux-x86_64/release/exts \
      --ext-folder <path-to-kit-cae>/_build/linux-x86_64/release/apps \
      --enable omni.cae

# On Windows
repo.bat launch -n <your-app> -- \
      --ext-folder <path-to-kit-cae>\_build\windows-x86_64\release\exts \
      --ext-folder <path-to-kit-cae>\_build\windows-x86_64\release\apps \
      --enable omni.cae
```

### Extension Variants

- **Basic CAE**: `--enable omni.cae`
- **With VTK**: `--enable omni.cae_vtk`
- **Individual Extensions**: `--enable <ext-name>` for specific extensions

## Full Integration (Packaging)

To generate packages that include Kit-CAE components:

### 1. Add Kit-CAE as dependency

```sh
# From your KAT application root
git clone <kit-cae-repo> kit-cae

# or use git submodule
git submodule add <kit-cae-repo> kit-cae
```

### 2. Modify `repo.toml`

Update `repo.toml` to include building Kit-CAE as a pre-build step. There are many
ways to do this; the simplest is using `before_pull_commands` as follows:

```toml
# in repo.toml
[repo_build]
fetch.before_pull_commands = [
  # tell Kit-CAE to use same Kit-version as we will use for your app
  # `--no-use-symlinks` ensure we can package Kit-CAE .kit files without additional steps.
  ["${root}/kit-cae/repo${shell_ext}", "select_kit_version", "--version", "109.0.3", "--no-use-symlinks"],

  # build kit-cae
  ["${root}/kit-cae/repo${shell_ext}", "--set-token", "vs_version=${vs_version}", "build", "-r"]
]
```

Kit apps often use the `repo_precache_exts` tool to precache and version-lock extensions. Ensure that
this tool can locate Kit-CAE extensions/apps by adding or modifying the following section in your `repo.toml`:

```toml
# in repo.toml

[repo_precache_exts]
# Extension search folders (extends base template to include Kit-CAE extensions/apps)
ext_folders = [
  # standard app-specific dirs
  "${root}/source/extensions",
  "${root}/source/apps",

  # Kit-CAE dirs
  "${root}/kit-cae/source/extensions",
  "${root}/kit-cae/source/legacy_extensions",
  "${root}/kit-cae/source/apps",
]
```

### 3. Update premake5.lua

Add to the end of your top-level `./premake5.lua` (in your KAT application, not Kit-CAE):

```lua
repo_build.prebuild_link {
   {
     "%{root}/kit-cae/_build/%{platform}/%{config}/apps",
     "%{root}/_build/%{platform}/%{config}/kit-cae/apps"
   },
   {
     "%{root}/kit-cae/_build/%{platform}/%{config}/exts",
     "%{root}/_build/%{platform}/%{config}/kit-cae/exts"
   }
}
```

Another way to accomplish this is to update one of your `*-packman.xml` files to
link to the Kit-CAE directories. For example:

```xml
<!-- in tools/deps/kit-sdk-deps.packman.xml -->
<project toolsVersion="5.0">

  <!-- ....  snip .... -->

  <!-- link to Kit-CAE build to make it easy for this app to locate all the apps and extensions -->
  <dependency name="kit-cae-apps" linkPath="../../_build/${platform_target}/${config}/kit-cae/apps">
    <source path="${root}/kit-cae/_build/${platform_target}/${config}/apps" />
  </dependency>

  <dependency name="kit-cae-exts" linkPath="../../_build/${platform_target}/${config}/kit-cae/exts">
    <source path="${root}/kit-cae/_build/${platform_target}/${config}/exts" />
  </dependency>
</project>
```

### 4. Update Application .kit Files

For each application in `source/apps/`, edit the `.kit` file to add Kit-CAE paths:

```toml
[settings.app.exts]
folders.'++' = [
  "${app}/../apps",
  "${app}/../exts",
  "${app}/../extscache/",

  # Kit-CAE paths
  "${app}/../kit-cae/apps",
  "${app}/../kit-cae/exts",
]
```

### 5. Build and Package

```sh
# Build your application
./repo.sh build

# Package your application
./repo.sh package
```

Kit-CAE extensions will now be included in the generated package.

## Selective Extension Loading

You don't need to enable all Kit-CAE extensions. Choose what you need:

```sh
# Enable specific extensions only
./repo.sh launch -n <your-app> -- \
      --ext-folder <kit-cae-exts> \
      --enable omni.cae.schema \
      --enable omni.cae.delegate.cgns \
      --enable omni.cae.importer.cgns
```

See [Extensions Overview](./Extensions.md) for a complete list of available extensions.

## Troubleshooting

### Version Mismatch Issues

If you encounter errors related to Kit SDK versions:

1. Verify both applications use compatible Kit SDK versions
2. Perform clean builds of both applications
3. Check that schema versions match

### Extension Load Failures

If extensions fail to load:

1. Verify extension paths are correct in launch commands
2. Check that dependencies are built
3. Review console output for specific error messages
4. Ensure required pip packages are installed (for VTK, etc.)

## Related Documentation

- [Build Instructions](./Build.md) - Building Kit-CAE
- [Extensions Overview](./Extensions.md) - Available extensions
