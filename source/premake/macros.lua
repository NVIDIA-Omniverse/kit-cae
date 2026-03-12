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
    -- Schemas are built via premake into the standard build output directory
    local schemaRoot = "%{root}/_build/%{platform}/%{config}/schemas"
    includedirs { schemaRoot.."/include" }
    libdirs { schemaRoot.."/lib" }
    for _, lib in ipairs(libs) do
        links { lib }
    end
end

function copy_cae_schema(schema, ext)
    -- Schemas are built via premake into the standard build output directory
    local schemaRoot = "%{root}/_build/%{platform}/%{config}/schemas"
    local libname = schema:sub(1, 1):lower()..schema:sub(2)

    -- Use postbuildcommands since schemas are built in the same premake pass
    -- and prebuild_copy won't work (files don't exist yet at prebuild time)
    postbuildcommands {
        "{MKDIR} "..ext.target_dir.."/plugins",
    }

    -- {COPYDIR} has different semantics per platform (premake issue #1232):
    --   Linux (cp -rf):  creates basename(src)/ inside dst/ when dst exists
    --   Windows (xcopy): copies CONTENTS of src/ flat into dst/
    -- So on Windows we must include the schema name in the destination path
    filter { "system:linux" }
        postbuildcommands {
            "{COPYDIR} "..schemaRoot.."/plugins/"..schema.." "..ext.target_dir.."/plugins",
            "{COPYDIR} "..schemaRoot.."/"..schema.." "..ext.target_dir,
        }
    filter { "system:windows" }
        postbuildcommands {
            "{COPYDIR} "..schemaRoot.."/plugins/"..schema.." "..ext.target_dir.."/plugins/"..schema,
            "{COPYDIR} "..schemaRoot.."/"..schema.." "..ext.target_dir.."/"..schema,
        }
    filter {}

    -- Copy native libraries ({COPYFILE} has consistent behavior cross-platform)
    filter { "system:linux" }
        postbuildcommands {
            "{MKDIR} "..ext.target_dir.."/lib",
            "{COPYFILE} "..schemaRoot.."/lib/lib"..libname..".so "..ext.target_dir.."/lib/lib"..libname..".so",
        }
    filter { "system:windows" }
        postbuildcommands {
            "{MKDIR} "..ext.target_dir.."/bin",
            "{MKDIR} "..ext.target_dir.."/lib",
            "{COPYFILE} "..schemaRoot.."/bin/"..libname..".dll "..ext.target_dir.."/bin/"..libname..".dll",
            "{COPYFILE} "..schemaRoot.."/lib/"..libname..".lib "..ext.target_dir.."/lib/"..libname..".lib",
        }
    filter {}
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