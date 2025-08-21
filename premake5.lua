-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
--  its affiliates is strictly prohibited.

repo_build = require("omni/repo/build")

-- Repo root
root = repo_build.get_abs_path(".")

-- KIT CAE Extras
dofile("source/premake/macros.lua")


-- Kit USD is built with boost and v142 toolset. Let's make sure we correctly forward the information
-- so boost picks the right libraries to autolink with.
filter "system:windows"
defines { "BOOST_LIB_TOOLSET=\"vc142\"" }
filter {}

-- Run repo_kit_tools premake5-kit that includes a bunch of Kit-friendly tooling configuration.
kit = require("_repo/deps/repo_kit_tools/kit-template/premake5-kit")
kit.setup_all({ cppdialect = "C++17" })


-- Registries config for testing
repo_build.prebuild_copy {
    { "%{root}/tools/deps/user.toml", "%{root}/_build/deps/user.toml" },
}

-- copy license files to the package
repo_build.prebuild_copy {
    {"%{root}/LICENSE", "%{root}/_build/PACKAGE-LICENSES/kit-cae/LICENSE"},
    {"%{root}/tpl_licenses/**", "%{root}/_build/PACKAGE-LICENSES/kit-cae/"}
}

-- Apps: for each app generate batch files and a project based on kit files (e.g. my_name.my_app.kit)
define_app("omni.cae.kit")
define_app("omni.cae_vtk.kit")
define_app("omni.cae_streaming.kit")
