#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set -e
CWD="$( cd "$( dirname "$0" )" && pwd )"

# default config is release
CLEAN=false
BUILD=false
GENERATE=false
CONFIGURE=false
HELP=false
CONFIG=release
HELP_EXIT_CODE=0
CMAKE_CMD="$CWD/_build/target-deps/cmake/bin/cmake"

DIRECTORIES_TO_CLEAN=(
    _install
    _build
    _repo
)

while [ $# -gt 0 ]
do
    if [[ "$1" == "--clean" ]]
    then
        CLEAN=true
    fi
    if [[ "$1" == "--generate" ]]
    then
        GENERATE=true
    fi
    if [[ "$1" == "--build" ]]
    then
        BUILD=true
    fi
    if [[ "$1" == "--configure" ]]
    then
        CONFIGURE=true
    fi
    if [[ "$1" == "--debug" ]]
    then
        CONFIG=debug
    fi
    if [[ "$1" == "--help" ]]
    then
        HELP=true
    fi
    shift
done

if [[
        "$CLEAN" != "true"
        && "$GENERATE" != "true"
        && "$BUILD" != "true"
        && "$CONFIGURE" != "true"
        && "$HELP" != "true"
    ]]
then
    # default action when no arguments are passed is to do everything
    GENERATE=true
    BUILD=true
    CONFIGURE=true
fi

# requesting how to run the script
if [[ "$HELP" == "true" ]]
then
    echo "build.sh [--clean] [--generate] [--build] [--configure] [--debug] [--help]"
    echo "--clean: Removes the following directories (customize as needed):"
    for dir_to_clean in "${DIRECTORIES_TO_CLEAN[@]}" ; do
        echo "      $dir_to_clean"
    done
    echo "--generate: Perform code generation of schema libraries"
    echo "--build: Perform compilation and installation of USD schema libraries"
    echo "--prep-ov-install: Preps the kit-extension by copying it to the _install directory and stages the"
    echo "      built USD schema libraries in the appropriate sub-structure"
    echo "--configure: Performs a configuration step after you have built and"
    echo "      staged the schema libraries to ensure the plugInfo.json has the right information"
    echo "--debug: Performs the steps with a debug configuration instead of release"
    echo "      (default = release)"
    echo "--help: Display this help message"
    exit $HELP_EXIT_CODE
fi

cd "$CWD"

# do we need to clean?
if [[ "$CLEAN" == "true" ]]
then
    for dir_to_clean in "${DIRECTORIES_TO_CLEAN[@]}" ; do
        rm -rf "$CWD/$dir_to_clean"
    done
fi

# do we need to generate?
if [[ "$GENERATE" == "true" ]]
then
    mkdir -p _build/generated
    # pull down NVIDIA USD libraries
    # NOTE: If you have your own local build, you can comment out this step
    $CWD/../tools/packman/packman pull deps/usd-deps.packman.xml -p manylinux_2_35_$(arch) -t config=$CONFIG -t platform_host=linux-$(arch)

    # generate the schema code and plug-in information
    # NOTE: this will pull the NVIDIA repo_usd package to do this work
    export CONFIG=$CONFIG
    $CWD/../tools/packman/python.sh bootstrap.py usd --configuration $CONFIG
fi

# do we need to build?

# NOTE: Modify this build step if using a build system other than cmake (ie, premake)
if [[ "$BUILD" == "true" ]]
then
    # pull down target-deps to build dynamic payload which relies on CURL
    $CWD/../tools/packman/packman pull deps/target-deps.packman.xml -p manylinux_2_35_$(arch) -t config=$CONFIG -t platform_host=linux-$(arch)

    # Below is an example of using CMake to build the generated files
    $CMAKE_CMD -B ./_build/cmake -DCMAKE_BUILD_TYPE=$CONFIG
    $CMAKE_CMD --build ./_build/cmake --config $CONFIG --target install -j
fi

# do we need to configure? This will configure the plugInfo.json files
if [[ "$CONFIGURE" == "true" ]]
then
    $CWD/../tools/packman/python.sh bootstrap.py usd --configure-pluginfo --configuration $CONFIG
fi