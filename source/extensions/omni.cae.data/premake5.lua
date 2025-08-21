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

-- Link only those files and folders into the extension target directory
repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
}

-- Build Carbonite plugin to be loaded by the extension. This plugin implements
-- omni::ext::IExt interface to be automatically started by extension system.
project_ext_plugin(ext, "omni.cae.data.plugin")
    local plugin_name = "omni.cae.data"
    add_files("iface", "%{root}/include/omni/ext", "IExt.h")
    add_files("impl", "plugins/"..plugin_name)
    includedirs { "plugins/"..plugin_name }

    add_usd({"usd", "sdf", "vt"})
    add_cae_usd_schemas({"omniCae"})
    add_cuda_build_support()
    add_omni_client_library()

-- Python Bindings for Carobnite Plugin
project_ext_bindings {
    ext = ext,
    project_name = "omni.cae.data.python",
    module = "_omni_cae_data",
    src = "bindings/python/omni.cae.data",
    target_subdir = "omni/cae/data"
}

    filter { "configurations:debug" }
        defines { "TBB_USE_DEBUG=1" }
    filter  { "configurations:release" }
        defines { "TBB_USE_DEBUG=0" }
    filter {}

    add_usd({"usd", "sdf", "vt", "usdUtils"})
    add_cuda_build_support()


repo_build.prebuild_link {
    { "python/tests", ext.target_dir.."/omni/cae/data/tests" },
    { "python/impl", ext.target_dir.."/omni/cae/data/impl" },
}