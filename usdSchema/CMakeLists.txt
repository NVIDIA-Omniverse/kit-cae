# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# 3.20 is required to access target_link_directories and cmake_path
cmake_minimum_required(VERSION 3.20)

project(usd-plugins)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/../_repo/deps/repo_usd/templates/cmake)
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

include(PxrPlugin)

add_subdirectory(_build/generated/omniCae)
add_subdirectory(_build/generated/omniCaeSids)
add_subdirectory(_build/generated/omniCaeVtk)
add_subdirectory(_build/generated/omniCaeEnSight)
