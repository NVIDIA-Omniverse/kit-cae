rem SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
rem SPDX-License-Identifier: LicenseRef-NvidiaProprietary
rem
rem NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
rem property and proprietary rights in and to this material, related
rem documentation and any modifications thereto. Any use, reproduction,
rem disclosure or distribution of this material and related documentation
rem without an express license agreement from NVIDIA CORPORATION or
rem  its affiliates is strictly prohibited.

@echo off
setlocal enabledelayedexpansion

pushd %~dp0

REM options defining what the script runs
set GENERATE=false
set BUILD=false
set CLEAN=false
set CONFIGURE=false
set HELP=false
set CONFIG=release
set HELP_EXIT_CODE=0
set CMAKE_GENERATOR_TOOLSET=v143
set CMAKE_GENERATOR=Visual Studio 17 2022
set CMAKE_CMD="_build/target-deps/cmake/bin/cmake.exe"

set DIRECTORIES_TO_CLEAN=_install _build _repo

REM default arguments for script - note this script does not actually perform
REM any build step so that you can integrate the generated code into your build
REM system - an option can be added here to run your build step (e.g. cmake)
REM on the generated files
:parseargs
if not "%1"=="" (
    if "%1" == "--clean" (
        set CLEAN=true
    )
    if "%1" == "--generate" (
        set GENERATE=true
    )
    if "%1" == "--build" (
        set BUILD=true
    )
    if "%1" == "--configure" (
        set CONFIGURE=true
    )
    if "%1" == "--debug" (
        set CONFIG=debug
    )
    if "%1" == "--help" (
        set HELP=true
    )
    if "%1" == "--vs2019" (
        set CMAKE_GENERATOR_TOOLSET=v142
        set CMAKE_GENERATOR=Visual Studio 16 2019
    )
    shift
    goto :parseargs
)

if not "%CLEAN%" == "true" (
    if not "%GENERATE%" == "true" (
        if not "%BUILD%" == "true" (
            if not "%CONFIGURE%" == "true" (
                if not "%HELP%" == "true" (
                    REM default action when no arguments are passed is to do everything
                    set GENERATE=true
                    set BUILD=true
                    set CONFIGURE=true
                )
            )
        )
    )
)

REM requesting how to run the script
if "%HELP%" == "true" (
    echo build.bat [--clean] [--generate] [--build] [--configure] [--debug] [--help] [--vs2019]
    echo --clean: Removes the following directories ^(customize as needed^):
    for %%a in (%DIRECTORIES_TO_CLEAN%) DO (
        echo       %%a
    )
    echo --generate: Perform code generation of schema libraries
    echo --build: Perform compilation and installation of USD schema libraries
    echo --prep-ov-install: Preps the kit-extension by copying it to the _install directory and stages the
    echo       built USD schema libraries in the appropriate sub-structure
    echo --configure: Performs a configuration step after you have built and
    echo       staged the schema libraries to ensure the plugInfo.json has the right information
    echo --debug: Performs the steps with a debug configuration instead of release
    echo       ^(default = release^)
    echo --vs2019: Use VS 2019 toolkit ^(v142^) ^(default VS 2022 ^(v143^)^)
    echo --help: Display this help message
    exit %HELP_EXIT_CODE%
)

REM should we clean the target directory?
if "%CLEAN%" == "true" (
    for %%a in (%DIRECTORIES_TO_CLEAN%) DO (
        if exist "%~dp0%%a/" (
            rmdir /s /q "%~dp0%%a"
        )
    )

    if !errorlevel! neq 0 (goto Error)

    goto Success
)

REM should we generate the schema code?
if "%GENERATE%" == "true" (
    mkdir "%~dp0_build/generated"

    REM pull down NVIDIA USD libraries
    REM NOTE: If you have your own local build, you can comment out this step
    call "%~dp0..\tools\packman\packman.cmd" pull deps/usd-deps.packman.xml -p windows-x86_64 -t config=%CONFIG% -t platform_host=windows-x86_64

    if !errorlevel! neq 0 ( goto Error )

    REM generate the schema code and plug-in information
    REM NOTE: this will pull the NVIDIA repo_usd package to do this work
    call "%~dp0..\tools\packman\python.bat" bootstrap.py usd --configuration %CONFIG%

    if !errorlevel! neq 0 ( goto Error )
)

REM should we build the USD schema?

REM NOTE: Modify this build step if using a build system other than cmake (ie, premake)
if "%BUILD%" == "true" (
    REM pull down target-deps to build dynamic payload which relies on CURL
    call "%~dp0..\tools\packman\packman.cmd" pull deps/target-deps.packman.xml -p windows-x86_64 -t config=%CONFIG% -t platform_host=windows-x86_64

    REM Below is an example of using CMake to build the generated files
    REM You may also want to explicitly specify the toolset depending on which
    REM version of Visual Studio you are using (e.g. -T v141)
    REM NVIDIA USD 22.11 was built with the v142 toolset, so we set that here
    REM Note that NVIDIA USD 20.08 was build with the v141 toolset
    %CMAKE_CMD% -B ./_build/cmake/%CONFIG% -DCMAKE_CXX_FLAGS_INIT="/DBOOST_LIB_TOOLSET=\\""vc142\\"""
    %CMAKE_CMD% --build ./_build/cmake/%CONFIG% --config=%CONFIG% --target install -j
)

REM is this a configure only?  This will configure the plugInfo.json files
if "%CONFIGURE%" == "true" (
    call "%~dp0..\tools\packman\python.bat" bootstrap.py usd --configure-pluginfo --configuration %CONFIG%

    if !errorlevel! neq 0 (goto Error)
)

:Success
exit /b 0

:Error
exit /b !errorlevel!
