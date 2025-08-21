# SPDX-FileCopyrightText: Copyright 2016 Pixar
# SPDX-FileCopyrightText: Modifications copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# This work is derivative of the existing cmake definitions in the Pixar
# USD build here:
# https://github.com/PixarAnimationStudios/USD/tree/release/cmake/defaults

# Enable all warnings.
set(_PXR_GCC_CLANG_SHARED_CXX_FLAGS "${_PXR_GCC_CLANG_SHARED_CXX_FLAGS} -Wall -Wformat-security")

# If using pthreads then tell the compiler.  This should automatically cause
# the linker to pull in the pthread library if necessary so we also clear
# PXR_THREAD_LIBS.
if(CMAKE_USE_PTHREADS_INIT)
    set(_PXR_GCC_CLANG_SHARED_CXX_FLAGS "${_PXR_GCC_CLANG_SHARED_CXX_FLAGS} -pthread")
    set(PXR_THREAD_LIBS "")
endif()

set(_PXR_PLUGIN_CXX_FLAGS "${_PXR_GCC_CLANG_SHARED_CXX_FLAGS}")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
