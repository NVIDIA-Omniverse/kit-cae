# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os.path
import warnings
from logging import getLogger
from pathlib import Path

# Suppress h5py warning: runtime HDF5 (2.1.0) differs from build-time (2.0.0).
# Safe because both share soversion 320 (ABI-compatible).
warnings.filterwarnings("ignore", message="h5py is running against HDF5")
import h5py as h5
import omni.client.utils as clientutils
from omni.cae.data import progress
from omni.cae.schema import cae
from omni.client import get_local_file_async
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, Vt

logger = getLogger(__name__)


async def populate_stage(uri: str, local_path: str, stage: Usd.Stage):
    # Implementation of the function
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Scope.Define(stage, world.GetPath().AppendChild(Tf.MakeValidIdentifier(os.path.basename(uri))))
    rootPath = root.GetPath()

    with h5.File(local_path, "r") as deck:
        nb_timesteps = deck.attrs.get("num timesteps", 0)
        if nb_timesteps < 1:
            logger.warning(f"No timesteps found in {uri}")
            return

        # read geometry groups
        geometry_groups = await read_geometry_groups(uri, stage, rootPath)

        # read particle types; only reads from the 1st timestep.
        particle_types = await read_particle_types(uri, stage, rootPath)

        # now read the particles as a PointCloud.
        configs = []
        for t in range(nb_timesteps):
            p_uri = Path(uri)
            h_uri = str(p_uri.parent / f"{p_uri.stem}_data" / f"{t}.h5")
            with progress.ProgressContext(f"Loading timestep {t}"):
                _, h_path = await get_local_file_async(h_uri)

                with h5.File(h_path, "r") as h:
                    if "TimestepData" not in h:
                        continue
                    for _, node in h["TimestepData"].items():
                        data_time = node.attrs.get("time", 0.0)
                        configs.append(
                            {
                                "time": data_time,
                                "path": h_uri,
                                "h5_path_prefix": node.name,
                            }
                        )

                        # read xform for geometry groups if available
                        if geometry := node.get("GeometryGroups"):
                            for group_name, (mesh, op) in geometry_groups.items():
                                xform_path = f"{group_name}/Kinematics/0/global transform"
                                if xform := geometry.get(xform_path):
                                    assert isinstance(xform, h5.Dataset), "xform should be a dataset"
                                    transform = xform[:].reshape(4, 4)
                                    op.Set(Gf.Matrix4d(transform).GetTranspose(), Usd.TimeCode(len(configs) - 1))

        if len(configs) != nb_timesteps:
            logger.error(f"Expected {nb_timesteps} timesteps, but found {len(configs)} in {uri}!")
            raise RuntimeError(f"Expected {nb_timesteps} timesteps, but found {len(configs)} in {uri}!")

        await read_particle_datasets(uri, stage, rootPath, particle_types, configs)


async def read_particle_datasets(
    uri: str, stage: Usd.Stage, rootPath: Sdf.Path, particle_types: list[str], configs: list[dict]
):
    for p_type in particle_types:
        # Not entirely sure how to name the dataset, the name of the particle type is not a good choice, but
        # we'll go with that for now.
        dataset = cae.DataSet.Define(stage, rootPath.AppendChild(Tf.MakeValidIdentifier(p_type["name"])))
        cae.PointCloudAPI.Apply(dataset.GetPrim())
        cls = cae.FieldArray(stage.CreateClassPrim(dataset.GetPath().AppendChild("TemporalFieldArrayClass")))
        cls.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
        attr = cls.CreateFileNamesAttr()
        for idx, config in enumerate(configs):
            attr.Set(
                [clientutils.make_file_url_if_possible(config["path"])], Usd.TimeCode(idx)
            )  # FIXME: use index or time or shift or scale?

        # now add each field array.
        for field_name in ["position", "ids", "scale", "orientation", "velocity"]:
            field_array = cae.Hdf5FieldArray.Define(
                stage, dataset.GetPath().AppendChild(Tf.MakeValidIdentifier(field_name))
            )
            dataset.GetPrim().CreateRelationship(f"field:{field_name}").SetTargets({field_array.GetPath()})
            field_array.GetPrim().GetSpecializes().SetSpecializes([cls.GetPath()])
            for idx, config in enumerate(configs):
                field_array.CreateHdf5PathAttr().Set(
                    f"{config['h5_path_prefix']}/ParticleTypes/{p_type['node_name']}/{field_name}", Usd.TimeCode(idx)
                )  # FIXME
            if field_name == "position":
                cae.PointCloudAPI(dataset).CreateCoordinatesRel().SetTargets({field_array.GetPath()})


async def read_particle_types(uri: str, stage: Usd.Stage, rootPath: Sdf.Path) -> list[str]:
    # read the first timestep to particle shapes.
    # dirname is simply the file name without extension
    p_types = []
    p_uri = Path(uri)
    h0_uri = str(p_uri.parent / f"{p_uri.stem}_data" / "0.h5")
    with progress.ProgressContext(f"Loading timestep 0"):
        _, h0_path = await get_local_file_async(h0_uri)

    with h5.File(h0_path, "r") as h0:
        if "/CreatorData/0/ParticleTypes" in h0:
            scope = UsdGeom.Scope.Define(stage, rootPath.AppendChild("ParticleTypes"))
            # hide scope from view, so prototypes are not shown in viewport
            scope.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            scope_path = scope.GetPath()

            for node_name, node in h0["/CreatorData/0/ParticleTypes"].items():
                if not isinstance(node, h5.Group):
                    continue

                name = node.attrs.get("name", "Unknown")
                p_types.append({"node_name": node_name, "name": name})
                if spheres_node := node.get("spheres"):
                    assert isinstance(spheres_node, h5.Dataset), "spheres should be a dataset"

                    xform = UsdGeom.Xform.Define(stage, scope_path.AppendChild(Tf.MakeValidIdentifier(name)))
                    dset = spheres_node[:]
                    pos = dset["pos"]
                    physicalRadius = dset["physicalRadius"]
                    contactRadius = dset["contactRadius"]
                    names = [x.decode("utf-8") for x in dset["name"]]
                    for idx, name in enumerate(names):
                        sphere = UsdGeom.Sphere.Define(stage, xform.GetPath().AppendChild(Tf.MakeValidIdentifier(name)))
                        sphere.CreateRadiusAttr(physicalRadius[idx])
                        sphere.AddTranslateOp().Set((pos[idx][0], pos[idx][1], pos[idx][2]))

                        # just some metadata
                        sphere.GetPrim().CreateAttribute(
                            "edem:physicalRadius", Sdf.ValueTypeNames.Float, custom=True
                        ).Set(physicalRadius[idx])
                        sphere.GetPrim().CreateAttribute("edem:name", Sdf.ValueTypeNames.String, custom=True).Set(name)
                        sphere.GetPrim().CreateAttribute("edem:position", Sdf.ValueTypeNames.Float3, custom=True).Set(
                            (pos[idx][0], pos[idx][1], pos[idx][2])
                        )
                        sphere.GetPrim().CreateAttribute(
                            "edem:contactRadius", Sdf.ValueTypeNames.Float, custom=True
                        ).Set(contactRadius[idx])
                elif "triangle nodes" in node and "coords" in node:
                    # polyhedral particles
                    tri_node = node["triangle nodes"]
                    coords_node = node["coords"]

                    mesh = UsdGeom.Mesh.Define(stage, scope_path.AppendChild(Tf.MakeValidIdentifier(name)))
                    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords_node[:].reshape(-1, 3)))
                    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tri_node[:].ravel()))
                    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(len(tri_node) * [3]))

                    # just some metadata
                    mesh.GetPrim().CreateAttribute("edem:name", Sdf.ValueTypeNames.String, custom=True).Set(name)
                    mesh.GetPrim().CreateAttribute("edem:rawMass", Sdf.ValueTypeNames.Float, custom=True).Set(
                        node.attrs.get("raw mass", 0.0)
                    )
                    mesh.GetPrim().CreateAttribute("edem:rawVolume", Sdf.ValueTypeNames.Float, custom=True).Set(
                        node.attrs.get("raw volume", 0.0)
                    )
                    mesh.GetPrim().CreateAttribute("edem:rawSurfaceArea", Sdf.ValueTypeNames.Float, custom=True).Set(
                        node.attrs.get("raw surface area", 0.0)
                    )
                    mesh.GetPrim().CreateAttribute("edem:sphericity", Sdf.ValueTypeNames.Float, custom=True).Set(
                        node.attrs.get("sphericity", 0.0)
                    )
    return p_types


async def read_geometry_groups(uri: str, stage: Usd.Stage, rootPath: Sdf.Path):
    geometry_groups = {}
    p_uri = Path(uri)
    h_uri = str(p_uri.parent / f"{p_uri.stem}_data" / "0.h5")
    with progress.ProgressContext(f"Loading geometry groups from timestep 0"):
        _, h_path = await get_local_file_async(h_uri)

    with h5.File(h_path, "r") as h:
        if "/CreatorData/0/GeometryGroups" in h:
            scope = UsdGeom.Scope.Define(stage, rootPath.AppendChild("GeometryGroups"))
            # hide scope from view, so prototypes are not shown in viewport
            scope.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            scope_path = scope.GetPath()

            for group_name, group in h["/CreatorData/0/GeometryGroups"].items():
                if not isinstance(group, h5.Group):
                    continue

                name = group.attrs.get("name", "Unknown")
                if "coords" in group and "triangle nodes" in group:
                    # For now, directly reading the mesh and populating USDGeomMesh during import itself.
                    # Eventually, we might want to create cae.DataSet instead to lazy load the data.
                    mesh = UsdGeom.Mesh.Define(stage, scope_path.AppendChild(Tf.MakeValidIdentifier(name)))
                    coords_node = group["coords"]
                    tri_node = group["triangle nodes"]
                    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(coords_node[:].reshape(-1, 3)))
                    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(tri_node[:].ravel()))
                    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(len(tri_node) * [3]))

                    # add xform op
                    op = mesh.AddTransformOp()
                    op.Set(Gf.Matrix4d().SetIdentity())
                    geometry_groups[group_name] = (mesh, op)
    return geometry_groups
