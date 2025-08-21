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

project_ext(ext)

repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
    { "python/impl", ext.target_dir.."/omni/cae/index/impl" },
}

project_ext_plugin(ext, "omni.cae.index.plugin")
    local plugin_name = "omni.cae.index"
    local targetDeps="%{target_deps}"
    add_files("iface", "%{root}/include/omni/ext", "IExt.h")
    add_files("impl", "plugins/"..plugin_name)
    includedirs {
        "plugins/"..plugin_name,
        targetDeps.."/nvindex/include",
        targetDeps.."/nvindex/src/integration/application_layer/public/",
        targetDeps.."/pybind11/include",

        -- needed for UsdContext
        "%{root}/_build/%{platform}/%{config}/kit/dev/gsl/include",
    }

    libdirs { "%{root}/_build/%{platform}/%{config}/extsbuild/omni.usd.core/bin" }
    links { "omni.usd", "carb" }

    add_usd()
    add_cae_usd_schemas({"omniCae", "omniCaeSids"})
    add_cuda_build_support()

project_ext_bindings {
    ext = ext,
    project_name = "omni.cae.index.python",
    module = "_omni_cae_index",
    src = "bindings/python",
    target_subdir = "omni/cae/index"
}
    includedirs {
        targetDeps.."/nvindex/include",
    }

    filter { "configurations:debug" }
        defines { "TBB_USE_DEBUG=1" }
    filter  { "configurations:release" }
        defines { "TBB_USE_DEBUG=0" }
    filter {}
