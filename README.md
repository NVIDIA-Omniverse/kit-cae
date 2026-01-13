# Kit-CAE: Omniverse Sample Application

Kit-CAE is an Omniverse sample that demonstrates CAE data processing and rendering workflows in Omniverse.
This repository contains the source code for the Omniverse extensions developed for this sample application.

![Kit-CAE Based Editor](./docs/kit-cae-based-editor.png)

## Getting Started

To try out the sample, first clone or check out the source code, then build it using the following steps.

**On Windows:**

Visual Studio 2019 or 2022, along with the corresponding Windows SDK and build tools for C++ applications, must be installed on your system.
The build instructions differ slightly depending on which version of Visual Studio you use.

**For Visual Studio 2019:**

```sh
# 1. Build the USD schemas.
repo.bat schema

# 2. Build the Omniverse extensions.
repo.bat --set-token vs_version:vs2019 build -r

# 3. Launch the sample application.
repo.bat launch -n omni.cae.kit
```

**For Visual Studio 2022:**

```sh
# 1. Build the USD schemas.
repo.bat schema --vs2022

# 2. Build the Omniverse extensions.
repo.bat --set-token vs_version:vs2022 build -r

# 3. Launch the sample application.
repo.bat launch -n omni.cae.kit
```

You can also edit [repo.toml](./repo.toml) to change the `vs_version = ""` line to `vs_version = "vs2019"` or `vs_version = "vs2022"`.

**On Linux:**

```sh
# 1. Build the USD schemas.
./repo.sh schema

# 2. Build the Omniverse extensions.
./repo.sh build -r

# 3. Launch the sample application.
./repo.sh launch -n omni.cae.kit
```

This will download all the necessary dependencies, including the KIT SDK, libraries, and headers.
Use `./repo.sh --help` or `./repo.sh [tool] --help` to view available options for customizing any step.

The sample can optionally use algorithms that require [VTK](https://vtk.org) for data processing.
To enable these components, launch the application as follows:

```sh
# On Windows
repo.bat launch -n omni.cae_vtk.kit

# On Linux
./repo.sh launch -n omni.cae_vtk.kit
```

This depends on the VTK pip package. Refer to the following section on installing pip packages for details.

## Installing optional dependencies using PIP

This sample includes extensions that use external Python packages to demonstrate how such extensions can be integrated.
When trying out these components, you must manually download the required pip package archives for the sample to use.
For example, when using the VTK-based application variant, ensure the VTK pip package is already downloaded.

To download these packages, use the following commands. You can replace the path to `pip_archives` to be a any
location where you want the archives to be downloaded on your system. Always use absolute paths for the download location
to avoid issues.

```sh
# On Windows
repo.bat pip_download --dest C:/temp/pip_archives -r ./tools/deps/requirements.txt

# On Linux
./repo.sh pip_download --dest /tmp/pip_archives -r ./tools/deps/requirements.txt
```

After downloading the required packages, launch the application and pass the download location on the command line as follows.

**NOTE: PLEASE NOTE THE `[` and `]` WHEN PASSING THE DOWNLOAD LOCATION TO `--/exts/omni.kit.pipapi/archiveDirs` COMMAND LINE ARGUMENT.**

```sh
# On Windows; PLEASE DON'T FORGET THE `[` AND `]` AROUND THE PATH. THEY ARE REQUIRED.
repo.bat launch -n omni.cae_vtk.kit -- --/exts/omni.kit.pipapi/archiveDirs=[C:/temp/pip_archives]

# On Linux; PLEASE DON'T FORGET THE `[` AND `]` AROUND THE PATH. THEY ARE REQUIRED.
./repo.sh launch -n omni.cae_vtk.kit -- --/exts/omni.kit.pipapi/archiveDirs=[/tmp/pip_archives]
```

This step is only needed when launching for the first time (or after a cache cleanup). The necessary packages are then
installed in the local cache using the package archives from the directory you provide. No online pip index is used, so
all required package archives must be available in the specified directory.

## Users Guide

[The User Guide](https://docs.omniverse.nvidia.com/guide-kit-cae/latest/index.html)
provides step-by-step instructions for trying out each of the example features
included in this sample.

## Sample scripts

This repository includes several sample scripts that one can use to try functionality in Kit-CAE. These scripts are
available under [scripts](./scripts/). These scripts can be run as follows:

```sh
# On Linux
> ./repo.sh launch -n omni.cae.kit -- --exec scripts/example-bounding-box.py

# On Windows
> repo.bat launch -n omni.cae.kit -- --exec scripts/example-bounding-box.py
```

For scripts that depend on VTK, launch the `omni.cae_vtk.kit` application instead:

```sh
# On Linux
> ./repo.sh launch -n omni.cae_vtk.kit -- --exec scripts/example-streamlines.py

# On Windows
> repo.bat launch -n omni.cae_vtk.kit -- --exec scripts/example-streamlines.py
```

## Omni CAE USD Schema

To support CAE use cases, we extend USD to add new prim types that help represent scientific datasets while preserving
native mechanisms like file formats. For a detailed description of this USD schema, see the
[Omni CAE USD Schema documentation](./usdSchema/README.md).

![USD Schema](./usdSchema/docs/OmniCae.schema.svg)

The major elements of this schema design are:

* `CaeDataSet` prim type is similar to `UsdVolVolume` and acts as a representative for any scientific dataset.
* `CaeFieldArray` prim and its subtypes are used to represent data arrays stored in different files or assets.
  For any new file type, you must define new subtypes, e.g., `CaeCgnsFieldArray`, `CaeNumPyFieldArray`.
* Information about how to interpret `CaeDataSet` and its arrays is provided using single-apply API schemas.
  Currently, we provide `CaePointCloudAPI` and `CaeSidsUnstructuredAPI` as examples of such data model API schemas.

## Omniverse Extensions Overview

Kit-CAE includes several Omniverse extensions. Here's a brief summary of some of these extensions:

* **USD Schemas**: These modules/extensions relate to the core schemas for representing CAE datasets in USD.

  * [`usdSchema`](./usdSchema/): Technically not an Omniverse extension, these are USD plugins that add support for
    USD schemas. We introduce new USD prims to support CAE datasets and files. These new prims enable referencing CAE
    datasets through external assets, allowing applications to use native formats and minimize data duplication.
    These schemas are extensible to support non-IO-centric use cases as well.

  * [`omni.cae.schema`](./source/extensions/omni.cae.schema/): An extension to load the CAE schemas into the Omniverse
    ecosystem. USD schemas must be explicitly imported into Omniverse using an extension. For schemas to work correctly,
    they must be imported early during application initialization. Having a separate extension makes this possible.

  * [`omni.cae.algorithms.schema`](./source/extensions/omni.cae.algorithms.schema/): Similar to `omni.cae.schema`,
    except this extension loads USD schemas for various example algorithms. All schemas in this mode are codeless (i.e.,
    they don't require any compiled code). This extension demonstrates how codeless schemas can be integrated into an Omniverse application.

* **Data Importers**: These extensions add support to the `File > Import` menu for various file formats. Importers
  typically open the files, read lightweight metadata, and then populate the USD stage with USD prims that use
  CAE-specific USD schemas to add the datasets to the stage.

  * [`omni.cae.asset_importer.cgns`](./source/extensions/omni.cae.asset_importer.cgns/): Adds support to import CGNS (.cgns) files.
    Relies on USD file format plugin defined in `omni.cae.file_format.cgns` extension.

  * [`omni.cae.asset_importer.npz`](./source/extensions/omni.cae.asset_importer.npz/): Adds support to import `.npz` and `.npy` files.
    These are typically used for arrays representing point clouds.

  * [`omni.cae.asset_importer.vtk`](./source/extensions/omni.cae.asset_importer.vtk/): Adds limited support for importing
    certain VTK files (`.vtk`, `.vti`, and `.vtu`).

  * [`omni.cae.asset_importer.ensight`](./source/extensions/omni.cae.asset_importer.ensight/): Adds limited support for importing
    EnSight Gold CASE (`.case`) files. Only surface meshes are currently supported.

* **Data Processing**: Kit-CAE includes prototype implementations for two core components that it relies on.
  These are not intended to be the recommended way of processing scientific data in Omniverse, but serve as illustrations.

  * [`omni.cae.data`](./source/extensions/omni.cae.data/): This extension introduces the concept of [**Data Delegate**](./docs/DataDelegate.md).
    The Data Delegate API provides an extensible mechanism to add support for handling `CaeFieldArray` prim and its subtypes. Data Delegate
    has two sets of APIs: APIs to access raw data referenced by a `CaeFieldArray` prim, and APIs to register delegates that can handle the
    *reading* of raw data referenced by a subtype of `CaeFieldArray`. This extension also defines `omni.kit.Command` types called
    [`Operator Commands`](./docs/OperatorCommands.md). These commands can be used by algorithms in Kit-CAE for basic data processing
    operations needed for supported algorithms. They provide an extensible mechanism that allows extensions to introduce new data models
    for handling different types of data in their native representation. In a typical CAE application, one would adopt a
    data model to represent data within the system; however, for Kit-CAE, we intentionally avoid making that choice.

  * [`omni.cae.algorithms.core`](./source/extensions/omni.cae.algorithms.core/): This extension defines a simple backbone
    to manage the execution of algorithms based on USD scene changes. It introduces an algorithm factory with which
    extensions can register new algorithms. The extension also includes implementations of several of the core algorithms
    exposed in Kit-CAE. These implementations demonstrate how to use `Operator Commands` to perform necessary data transformations.

* **Data Delegate Implementations**: These extensions add concrete implementations of data delegates to support different file formats.

  * [`omni.cae.file_format.cgns`](./source/extensions/omni.cae.file_format.cgns/): This extension add a USD File Format plugin for
    loading CGNS files. This serves as an example of adding USD file formats.

  * [`omni.cae.cgns`](./source/extensions/omni.cae.cgns/): Adds CGNS data delegate to support reading data referenced in a
    `CaeCgnsFieldArray` prim.

  * [`omni.cae.hdf5`](./source/extensions/omni.cae.hdf5/): Adds HDF5 data delegate to support reading data referenced in a
    `CaeHdf5FieldArray` prim.

  * [`omni.cae.npz`](./source/extensions/omni.cae.npz/): This extension adds support for NumPy file formats
    such as `.npy` and `.npz` by providing an `IDataDelegate` implementation for handling prims of type `CaeNumPyFieldArray`.
    This extension demonstrates how pure-Python extensions can be used to add support for file formats via data delegates.

  * [`omni.cae.ensight`](./source/extensions/omni.cae.ensight/): This extension adds support for the EnSight
    file format. It also includes operator commands for handling EnSight-specific API schemas for processing the EnSight data model.

  * [`omni.cae.vtk`](./source/extensions/omni.cae.vtk/): This extension adds support for VTK file formats.
    It also includes operator commands for handling VTK-specific API schemas for processing the VTK data model.

  * [`omni.cae.sids`](./source/extensions/omni.cae.sids/): This extension adds operator commands for
    handling data that adheres to the SIDS data model—the data model typically used by datasets in CGNS files.

* **Miscellaneous**:

  * [`omni.cae.algorithms.vtk`](./source/extensions/omni.cae.algorithms.vtk/): This extension adds operator command
    implementations that use [VTK](https://www.vtk.org) for data processing. It demonstrates how external libraries
    can be integrated for data processing.

  * [`omni.cae.algorithms.warp`](./source/extensions/omni.cae.algorithms.warp/): This extension adds operator command
    implementations using [NVIDIA Warp](https://developer.nvidia.com/warp-python).

  * [`omni.cae.index`](./source/extensions/omni.cae.index/): This extension adds algorithms that use IndeX for rendering
    volumes and slices. It also defines new operator command types to enable extensions to introduce new data models that work with these algorithms.

  * [`omni.cae.flow`](./source/extensions/omni.cae.flow/): Similar to IndeX, this extension adds support for new algorithms that use `Flow`.

* **UI**: These are UI-specific extensions.

  * [`omni.cae.context_menu`](./source/extensions/omni.cae.context_menu/): Adds support for context menus in the Stage
    widget for CAE algorithms and advanced rendering techniques. Demonstrates how to use `omni.kit.commands` to build such menus.

  * [`omni.cae.property.bundle`](./source/extensions/omni.cae.property.bundle/): Adds property widget customizations
    for the schemas defined in this project.

  * [`omni.cae.widget.stage_icons`](./source/extensions/omni.cae.widget.stage_icons/): Extension to add new icons
    for certain USD prim types defined in this project.

## Design

For a detailed look at the Data Delegate API, refer to the [Data Delegate documentation](./docs/DataDelegate.md).
To enable extensions that introduce support for new data models, Kit-CAE relies on a collection of operator commands.
Refer to the [Operator Commands documentation](./docs/OperatorCommands.md) for design details and supported operators.

## Combining with Kit Application Template-based Applications

While Kit-CAE is a standalone application, all the Omniverse extensions it comprises can be imported into an existing
Omniverse application based on the Kit Application Template (KAT), as long as the Kit Kernel versions match. To verify
the Kit Kernel version, inspect the [kit-sdk-packman.xml](./tools/deps/kit-sdk.packman.xml) in both the Kit-CAE source
and your Kit application. Once you have confirmed compatibility, follow these steps:

1. Build Kit-CAE using steps described in the **Getting Started** section.
2. Build your KAT-based application using build steps specific to your application.
3. Launch your application as follows:

   ```sh
   # Assuming current working dir is your KAT-based application's dir.

   # On Linux
   ./repo.sh launch -n <your application> -- \
         --ext-folder <path to kit-cae>/_build/linux-x86_64/release/exts \
         --ext-folder <path to kit-cae>/_build/linux-x86_64/release/apps \
         --enable omni.cae

   # On Windows
   repo.bat launch -n <your application> -- \
         --ext-folder <path to kit-cae>\_build\windows-x86_64\release\exts \
         --ext-folder <path to kit-cae>\_build\windows-x86_64\release\apps \
         --enable omni.cae
   ```

This will enable all extensions used by the `omni.cae.kit` application. To enable the VTK-based variable,
use `--enable omni.cae_vtk` instead. Alternatively, you can explicitly enable individual extensions that you want to enable
as well using the same `--enable <ext-name>` flag.

The above is sufficient for running and testing locally. If you want to generate a package that combines Kit-CAE components
with your existing application, then use the following steps:

1. Add the Kit-CAE source directory under your KAT application directory, e.g. `vendor/kit-cae`.
2. Build `Kit-CAE` under this location using steps described in **Getting Started** section.
3. Add the following lines at the end of the the top-level `./premake5.lua` file in your KAT application (not Kit-CAE).

   ```lua
   repo_build.prebuild_link {
      {
        "%{root}/vendor/kit-cae/_build/%{platform}/%{config}/apps",
        "%{root}/_build/%{platform}/%{config}/kit-cae/apps"
      },
      {
        "%{root}/vendor/kit-cae/_build/%{platform}/%{config}/exts",
        "%{root}/_build/%{platform}/%{config}/kit-cae/exts"
      }
   }
   ```

4. Finally, for each KAT application you have created, locate its `.kit` file under `source/apps/` and edit it to modify
   the value for `settings.apps.exts` key by adding `kit-cae` specific folders as follows:

   ```toml
   [settings.app.exts]
   folders.'++' = [
     "${app}/../apps",
     "${app}/../exts",
     "${app}/../extscache/",

     # Kit-CAE paths
     "${app}/../kit-cae/apps",
     "${app}/../kit-cae/exts",

       # Extra paths for repo_precache_exts step
     "./vendor/kit-cae/_build/${platform}/${config}/apps",
     "./vendor/kit-cae/_build/${platform}/${config}/exts"
   ]
   ```

Now use build and package your KAT-based app as usual i.e. `repo.* build` followed by `repo.* package` and Kit-CAE extensions
will be included in the generated package.

## License

Development using the Omniverse Kit SDK is subject to the licensing terms detailed [here](https://docs.omniverse.nvidia.com/install-guide/latest/common/NVIDIA_Omniverse_License_Agreement.html).

This project also uses several open-source libraries. Review the license terms of these open source projects before use.

1. [CGNS](./tpl_licenses/cgns-LICENSE.txt): CFD Notation System
2. [h5py](./tpl_licenses/h5py-LICENSE.txt): Python interface for HDF5.
3. [HDF5](./tpl_licenses/hdf5-LICENSE.txt): High performance data software library and file format
4. [VTK](./tpl_licenses/vtk-LICENSE.txt): Visualization Toolkit
5. [Zlib](./tpl_licenses/zlib-LICENSE.txt): General purpose data compression library

## Contributing

We provide this source code as-is and are currently not accepting outside contributions.
