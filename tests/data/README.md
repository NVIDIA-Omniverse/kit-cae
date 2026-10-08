# Test data

`StaticMixer.cgns` is copied from
`warp-simdata/tests/data/StaticMixer.cgns` and stored through Git LFS.

- Size: 741,544 bytes
- SHA-256: `8340df61c455ca1807030c748910c789de460640a9d31356851a02d4c237eb2d`

`scenes/assets.usda` payloads the CGNS onto an untyped holder so the CAE file-format
plug-ins compose it into `OmniSciDataset` prims. `scenes/payload_faces.usda` targets one
of those prims directly and, in a second recipe, targets the holder with a
relative `identifier`; the other stages target the holder and select datasets
with `identifier` values such as `/Base/StaticMixer/B1_P3`. Streamlines selects
its seeds from an authored `UsdGeomPoints` prim; only its flow source resolves
through the CGNS payload.
`scenes/streamlines_mesh_seeds.usda` also covers seeds from a translated `Mesh`.

The Points scenario resolves `/Base/StaticMixer/B1_P3`, converts selected
fields to node association, filters unused nodes, and applies deterministic
`maxCount` decimation before publishing standard `UsdGeomPoints` attributes.

`scaled_points_schema/` is a codeless USD schema plug-in (`plugInfo.json` and
`generatedSchema.usda`) declaring `CaeVizScaledPointsAPI` for the custom
operator tests; it is written by hand and carries no external data.
