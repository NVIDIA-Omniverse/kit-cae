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
    }

-- C++ Carbonite plugin
project_ext_plugin(ext, "omni.cae.cgns.plugin")
    local plugin_name = "omni.cae.cgns"

    add_files("impl", "plugins/"..plugin_name)

    includedirs {
        "plugins/"..plugin_name,
    }

    -- TODO: need to copy the libs for runtime dependencies.
    filter { "configurations:debug" }
        runtime "Debug"
    filter  { "configurations:release" }
        runtime "Release"
    filter {}

    add_usd({ "sdf", "vt", "usd", "usdGeom" })
    add_cae_usd_schemas({"omniCae", "omniCaeSids"})
    add_hdf5()
    add_cgns()