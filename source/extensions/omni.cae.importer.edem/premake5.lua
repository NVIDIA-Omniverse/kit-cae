local ext = get_current_extension_info()

project_ext (ext)

repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
    { "python", ext.target_dir.."/omni/cae/importer/edem" },
}
