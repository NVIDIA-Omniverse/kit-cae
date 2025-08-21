-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
--  its affiliates is strictly prohibited.

function add_cae_usd_schemas(libs)
    local schemaRoot = "%{root}/_build/target-deps/omni-cae/%{config}"
    includedirs { schemaRoot.."/include" }
    libdirs { schemaRoot.."/lib" }
    for _, lib in ipairs(libs) do
        links { lib }
    end
end

function copy_cae_schema(schema, ext)
    local schemaRoot = "%{root}/_build/target-deps/omni-cae/%{config}"
    local libname = schema:sub(1, 1):lower()..schema:sub(2)

    repo_build.prebuild_copy {
        { schemaRoot.."/plugins/"..schema, ext.target_dir.."/plugins/"..schema},
        { schemaRoot.."/"..schema.."/**", ext.target_dir.."/"..schema},
    }

    repo_build.prebuild_copy (
        { schemaRoot.."/lib/${lib_prefix}"..libname.."${lib_ext}", ext.target_dir.."/lib" }, "linux"
    )

    repo_build.prebuild_copy (
        { schemaRoot.."/bin/${lib_prefix}"..libname.."${lib_ext}", ext.target_dir.."/bin" }, "windows"
    )

    repo_build.prebuild_copy (
        { schemaRoot.."/lib/${lib_prefix}"..libname.."${staticlib_ext}", ext.target_dir.."/lib" }, "windows"
    )
end

function add_omni_client_library()
    local targetDeps = "%{target_deps}"
    includedirs { targetDeps.."/omni_client_library/include" }
    libdirs { targetDeps.."/omni_client_library/%{config}" }
    links { "omniclient" }
end

function add_hdf5()
    local targetDeps = "%{target_deps}"
    includedirs {
        targetDeps.."/hdf5/include/hdf5",
    }

    libdirs {
        targetDeps.."/hdf5/lib",
    }

    filter { "system:windows" }
        defines { "H5_BUILT_AS_DYNAMIC_LIB" }
        links { "hdf5", "hdf5_cpp" }
    filter { "system:linux" }
        links { "hdf5", "hdf5_cpp" }
    filter {}
end

function add_cgns()
    local targetDeps = "%{target_deps}"
    includedirs {
        targetDeps.."/cgns/include",
    }

    libdirs {
        targetDeps.."/cgns/lib",
    }

    filter { "system:windows" }
        links { "cgnsdll" }
    filter { "system:linux" }
        links { "cgns" }
        links { "dl" }
        -- linkoptions{"-Wl,--no-undefined"}
        -- -- necessary to use rpath to ensure libraries are correctly found transitively.
        -- -- linkoptions{"-Wl,--disable-new-dtags"}
        -- removeflags { "FatalCompileWarnings", "UndefinedIdentifiers" }
        -- buildoptions { "-Wno-error=undef", "-Wno-deprecated", "-Wno-deprecated-declarations" }
    filter {}
end