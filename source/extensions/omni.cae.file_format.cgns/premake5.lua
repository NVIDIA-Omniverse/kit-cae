-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
--  its affiliates is strictly prohibited.

local ext = get_current_extension_info()

project_ext (ext)
    repo_build.prebuild_link {
        { "data", ext.target_dir.."/data" },
        { "docs", ext.target_dir.."/docs" },
        { "python", ext.target_dir.."/omni/cae/file_format/cgns"}
    }
    repo_build.prebuild_copy {
        { "plugins/fileformat/plugInfo_%{platform}.json", ext.target_dir.."/plugin/omni.cae.file_format.cgns.plugin/resources/plugInfo.json" },
    }

-- Usd File Format Plugin
project("omni.cae.file_format.cgns.plugin")
    kind "SharedLib"
    targetdir(ext.target_dir.."/bin") -- placing this under bin so all CGNS dependecies are easily found
    location(workspace_dir.."/%{prj.name}")
    files("plugins/fileformat/*")

    -- Override default compilation settings
    exceptionhandling "On"
    rtti "On"
    staticruntime "Off"

    add_usd({ "sdf", "vt", "usd", "usdGeom" })
    add_cae_usd_schemas({"omniCae", "omniCaeSids"})
    add_cgns()

    filter { "system:windows" }
        --disable some double-float errors
        disablewarnings { "4244", "4305" }
        -- The v142 toolset (and later) strips out the "arch_ctor_<n>"
        -- symbols with /Zc:inline, so all USD based tools need to build with
        -- /Zc:inline- if they deal with arch or tf
        removeunreferencedcodedata("off")
    filter { "system:linux" }
        disablewarnings { "error=deprecated-declarations", "deprecated-declarations", "deprecated"}
    filter {}