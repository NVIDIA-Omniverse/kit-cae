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
        { "docs", ext.target_dir.."/docs" }
    }

    -- Copy HDF5 runtime dependencies
    local targetDeps="%{target_deps}"

    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/bin/hdf5.dll", ext.target_dir.."/bin" }, "windows"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/bin/hdf5_cpp.dll", ext.target_dir.."/bin" }, "windows"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/bin/hdf5_hl.dll", ext.target_dir.."/bin" }, "windows"
    )

    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/bin/hdf5_hl_cpp.dll", ext.target_dir.."/bin" }, "windows"
    )

    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5.so", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5.so.310", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5.so.310.5.1", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_cpp.so", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_cpp.so.310", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_cpp.so.310.0.6", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl.so", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl.so.310", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl.so.310.0.6", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl_cpp.so", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl_cpp.so.310", ext.target_dir.."/lib" }, "linux"
    )
    repo_build.prebuild_copy (
        { targetDeps.."/hdf5/lib/libhdf5_hl_cpp.so.310.0.6", ext.target_dir.."/lib" }, "linux"
    )