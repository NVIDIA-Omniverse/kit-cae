-- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: LicenseRef-NvidiaProprietary
--
-- NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
-- property and proprietary rights in and to this material, related
-- documentation and any modifications thereto. Any use, reproduction,
-- disclosure or distribution of this material and related documentation
-- without an express license agreement from NVIDIA CORPORATION or
--  its affiliates is strictly prohibited.

-- USD Schema Projects for CAE
-- These projects build the USD schema libraries (omniCae, omniCaeSids, etc.)
-- The generated premake5.lua files are created by repo_usd during the schema generation phase.

local schemas = {
    "omniCae",
    "omniCaeSids",
    "omniCaeNumPy",
    "omniCaeHdf5",
    "omniCaeCgns",
    "omniCaeVtk",
    "omniCaeEnSight",
    "omniCaeViz",
    "omniCaeOpenFoam",
    "omniCaeTrimesh",
    "omniCaeScae",
}

-- Schema dependencies - Python binding projects need to link against base schema libraries
-- on Windows for proper symbol resolution (Linux resolves through shared object dependencies)
local schema_base_deps = {
    omniCaeNumPy = { "omniCae" },
    omniCaeHdf5 = { "omniCae" },
    omniCaeCgns = { "omniCae" },
    omniCaeVtk = { "omniCae" },
    omniCaeEnSight = { "omniCae" },
    omniCaeOpenFoam = { "omniCae" },
    omniCaeTrimesh = { "omniCae" },
    omniCaeScae = { "omniCae" },
}

-- Track which schemas were successfully loaded
local loaded_schemas = {}

for _, schema in ipairs(schemas) do
    local generated_file = root.."/_build/generated/schemas/"..schema.."/premake5.lua"

    -- Check if the generated file exists before trying to load it
    if os.isfile(generated_file) then
        local ok, err = pcall(function()
            dofile(generated_file)
        end)

        if ok then
            table.insert(loaded_schemas, schema)

            -- Add cross-schema include paths and visibility for the C++ plugin project
            -- -fvisibility=default is required to export RTTI symbols (typeinfo, vtable)
            -- which are needed for USD schema inheritance to work correctly
            -- Use buildoptions to ensure this comes after Kit's -fvisibility=hidden
            project(schema)
                -- USD schema DLLs must use dynamic CRT (/MD) to match USD/Kit SDK.
                -- Kit workspace sets staticruntime "On" (/MT), but project_ext_plugin()
                -- overrides to "Off" for regular extensions. usd_plugin() doesn't, so
                -- we must override here to avoid CRT heap mismatch and heap corruption.
                staticruntime "Off"
                includedirs {
                    "%{root}/_build/generated/schemas",
                    "%{root}/_build/%{platform}/%{config}/schemas/include",
                }
                filter { "system:linux" }
                    buildoptions { "-fvisibility=default" }
                filter { "system:windows" }
                    -- Suppress conversion warnings from USD headers (treated as errors by Kit's build)
                    -- C4244: conversion from 'double' to 'float', possible loss of data
                    -- C4305: truncation from 'double' to 'float'
                    disablewarnings { "4244", "4305" }
                filter {}

            -- Add cross-schema include paths for the Python binding project
            project("_"..schema)
                staticruntime "Off"
                includedirs {
                    "%{root}/_build/generated/schemas",
                    "%{root}/_build/%{platform}/%{config}/schemas/include",
                }
                filter { "system:windows" }
                    -- Suppress conversion warnings from USD headers (treated as errors by Kit's build)
                    disablewarnings { "4244", "4305" }
                    -- Python bindings need to link against base schema libraries for symbol resolution
                    -- (Linux resolves these through shared object dependencies, Windows needs explicit links)
                    if schema_base_deps[schema] then
                        libdirs { "%{root}/_build/%{platform}/%{config}/schemas/lib" }
                        links(schema_base_deps[schema])
                    end
                filter {}
        else
            print("Warning: Failed to load schema "..schema..": "..tostring(err))
        end
    end
end

if #loaded_schemas == 0 then
    print("Note: No schema projects loaded. Run './repo.sh schema --generate-only' first.")
elseif #loaded_schemas < #schemas then
    print("Note: Loaded "..#loaded_schemas.." of "..#schemas.." schema projects.")
end

