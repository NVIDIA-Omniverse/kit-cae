# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
__all__ = ["OpenFoamImporter"]


import os.path
from logging import getLogger

import omni.client.utils as clientutils
from omni.cae.schema import cae, openfoam
from omni.kit.tool.asset_importer import AbstractImporterDelegate
from pxr import Sdf, Tf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class OpenFoamImporter(AbstractImporterDelegate):
    @property
    def name(self):
        return "OpenFOAM Importer"

    @property
    def filter_regexes(self) -> list[str]:
        return [r".*\.foam$"]

    @property
    def filter_descriptions(self) -> list[str]:
        return ["OpenFOAM files (*.foam)"]

    def show_destination_frame(self):
        return True

    def supports_usd_stage_cache(self):
        return True

    def build_options(self, paths: list[str]) -> None:
        pass

    async def convert_assets(self, paths: list[str], **kwargs):
        result = {}

        for path in filter(lambda uri: clientutils.is_local_url(uri), paths):
            normalized_path = clientutils.normalize_url(path)
            result[path] = await self._convert_asset(
                normalized_path, kwargs.get("import_as_reference"), kwargs.get("export_folder")
            )
        return result

    async def _convert_asset(self, path: str, import_as_reference: bool, export_folder: str):

        def populate_stage(stage: Usd.Stage):
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageUpAxis(stage, "Z")

            root = UsdGeom.Scope.Define(
                stage, world.GetPath().AppendChild(Tf.MakeValidIdentifier(os.path.basename(path)))
            )
            process_foam(stage, path, root.GetPath())
            return root

        if import_as_reference:
            # when importing as reference, create a new stage file and then return that.
            output_dir = export_folder if export_folder else os.path.dirname(path)
            name, _ = os.path.splitext(os.path.basename(path))
            usd_path = os.path.join(output_dir, f"{name}.usda")
            # TODO: if file exists, warn!!!

            stage = Usd.Stage.CreateNew(usd_path)
            populate_stage(stage)
            stage.Save()
            return usd_path
        else:
            # when adding directly to stage, just create an in memory stage
            # and return its id
            stage = Usd.Stage.CreateInMemory()
            populate_stage(stage)
            stage_id = UsdUtils.StageCache.Get().Insert(stage)
            return stage_id.ToString()


def process_foam(stage: Usd.Stage, path: str, root_path: Sdf.Path):
    logger.warning(f"Processing OpenFOAM case at {path}")
    dirname = os.path.dirname(path)

    # check if this is parallel case
    if os.path.exists(os.path.join(dirname, "processor0")):
        logger.warning("Parallel OpenFOAM case detected")
        # locate all directories named "processor[number]"
        mesh_dirs = []
        for entry in os.listdir(dirname):
            if entry.startswith("processor") and os.path.isdir(os.path.join(dirname, entry)):
                mesh_dirs.append(os.path.join(dirname, entry))

        if not mesh_dirs:
            raise RuntimeError(
                f"No processor directories found in {dirname}. Expected at least one directory named 'processor[number]'."
            )

        if len(mesh_dirs) > 1:
            # sort mesh_dirs to ensure consistent ordering
            mesh_dirs = sorted(mesh_dirs, key=lambda x: int(x.split("processor")[-1]))
    else:
        logger.warning("Single processor OpenFOAM case detected")
        mesh_dirs = [dirname]

    # check if "constant/polyMesh/points" exists
    points_path = os.path.join(mesh_dirs[0], "constant", "polyMesh", "points")
    if not os.path.exists(points_path):
        raise RuntimeError(f"Points file not found in {points_path}")

    faces_path = os.path.join(mesh_dirs[0], "constant", "polyMesh", "faces")
    if not os.path.exists(faces_path):
        raise RuntimeError(f"Faces file not found in {faces_path}")

    neighbours_path = os.path.join(mesh_dirs[0], "constant", "polyMesh", "neighbour")
    if not os.path.exists(neighbours_path):
        raise RuntimeError(f"neighbour file not found in {neighbours_path}")

    owner_path = os.path.join(mesh_dirs[0], "constant", "polyMesh", "owner")
    if not os.path.exists(owner_path):
        raise RuntimeError(f"Owner file not found in {owner_path}")

    dataset = cae.DataSet.Define(stage, root_path.AppendChild("Volume"))
    openfoam.PolyMeshAPI.Apply(dataset.GetPrim())
    polyMeshAPI = openfoam.PolyMeshAPI(dataset.GetPrim())

    # we keep mesh fields in a separate scope so they can be shared with the boundary meshes.
    meshFieldsPath = root_path.AppendChild("meshFields")
    UsdGeom.Scope.Define(stage, meshFieldsPath)
    for type_ in ["points", "neighbour", "owner", "faces"]:
        array = openfoam.FieldArray.Define(stage, meshFieldsPath.AppendChild(type_))
        array.CreateFileNamesAttr().Set([os.path.join(x, "constant", "polyMesh", type_) for x in mesh_dirs])
        array.GetFileNamesAttr().SetCustomDataByKey("omni:kit:locked", True)
        array.CreateTypeAttr().Set(type_)
        array.GetTypeAttr().SetCustomDataByKey("omni:kit:locked", True)
        dataset.GetPrim().CreateRelationship(f"cae:foam:{type_}").SetTargets([array.GetPath()])

    # create facesOffsets
    arrayOffsets = openfoam.FieldArray.Define(stage, meshFieldsPath.AppendChild("facesOffsets"))
    arrayOffsets.CreateFileNamesAttr().Set([os.path.join(x, "constant", "polyMesh", "faces") for x in mesh_dirs])
    arrayOffsets.GetFileNamesAttr().SetCustomDataByKey("omni:kit:locked", True)
    arrayOffsets.CreateTypeAttr().Set(openfoam.Tokens.facesOffsets)
    arrayOffsets.GetTypeAttr().SetCustomDataByKey("omni:kit:locked", True)
    polyMeshAPI.CreateFacesOffsetsRel().SetTargets([arrayOffsets.GetPath()])
    polyMeshAPI.GetFacesOffsetsRel().SetCustomDataByKey("omni:kit:locked", True)

    # populate boundary meshes.
    UsdGeom.Scope.Define(stage, root_path.AppendChild("boundaries"))
    boundary_meshes = process_foam_boundary(stage, mesh_dirs, root_path.AppendChild("boundaries"))
    for boundary_mesh in boundary_meshes:
        polyBoundaryMeshAPI = openfoam.PolyBoundaryMeshAPI(boundary_mesh)
        polyBoundaryMeshAPI.CreatePointsRel().SetTargets([meshFieldsPath.AppendChild("points")])
        polyBoundaryMeshAPI.CreateFacesOffsetsRel().SetTargets([meshFieldsPath.AppendChild("facesOffsets")])
        polyBoundaryMeshAPI.CreateFacesRel().SetTargets([meshFieldsPath.AppendChild("faces")])
        polyBoundaryMeshAPI.CreateOwnerRel().SetTargets([meshFieldsPath.AppendChild("owner")])

    # scan mesh_dirs[0] for "<float>" directories which correspond to time steps
    # number can be a float, e.g., "9.50001811981" or scientific notation like "1.0e-5"
    time_dirs = []
    for entry in os.listdir(mesh_dirs[0]):
        if os.path.isdir(os.path.join(mesh_dirs[0], entry)):
            try:
                float(entry)  # check if the directory name can be converted to float
                time_dirs.append(entry)
            except ValueError:
                continue
    time_dirs = sorted(time_dirs, key=lambda x: float(x))  # sort time directories numerically

    # Now locate all internal fields
    internalFieldsPath = root_path.AppendChild("internalFields")
    UsdGeom.Scope.Define(stage, internalFieldsPath)
    # for internalField arrays, 0 is initial condition and generally should not be included in the time steps
    f_time_dirs = time_dirs[1:] if "0" in time_dirs else time_dirs
    fields_dir = os.path.join(mesh_dirs[0], f_time_dirs[0])
    if os.path.exists(fields_dir):
        # get all files in the fields directory
        fields = [
            os.path.basename(f)
            for f in os.listdir(fields_dir)
            if f[0] != "." and os.path.isfile(os.path.join(fields_dir, f))
        ]

        for field in fields:
            valid_field = Tf.MakeValidIdentifier(field)
            array = openfoam.FieldArray.Define(stage, internalFieldsPath.AppendChild(valid_field))
            array.CreateNameAttr().Set(field)
            array.GetNameAttr().SetCustomDataByKey("omni:kit:locked", True)
            array.CreateTypeAttr().Set(openfoam.Tokens.internalField)
            array.GetTypeAttr().SetCustomDataByKey("omni:kit:locked", True)
            array.CreateFieldAssociationAttr().Set(cae.Tokens.cell)
            array.GetFieldAssociationAttr().SetCustomDataByKey("omni:kit:locked", True)
            fnameAttr = array.CreateFileNamesAttr()

            # create file names for each time step
            for idx, time_dir in enumerate(f_time_dirs):
                paths = [os.path.join(x, time_dir, field) for x in mesh_dirs]
                fnameAttr.Set(paths, Usd.TimeCode(idx) if len(f_time_dirs) > 1 else Usd.TimeCode.Default())
            fnameAttr.SetCustomDataByKey("omni:kit:locked", True)
            dataset.GetPrim().CreateRelationship(f"field:{valid_field}").SetTargets([array.GetPath()])
    logger.warning(
        f"Processed OpenFOAM case at {path} with {len(mesh_dirs)} mesh directories and {len(time_dirs)} time steps."
    )


def process_foam_boundary(stage: Usd.Stage, mesh_dirs: list[str], root_path: Sdf.Path):
    boundary_path = os.path.join(mesh_dirs[0], "constant", "polyMesh", "boundary")
    if not os.path.exists(boundary_path):
        logger.warning(f"Boundary file not found in {boundary_path}. No boundary meshes will be created.")
        return []

    patches = read_patches(os.path.join(mesh_dirs[0], "constant", "polyMesh", "boundary"))
    # print(f"Found {len(patches)} patches in boundary file.", patches)

    boundary_meshes = []
    for name, data in patches.items():
        # create a boundary mesh for each patch
        dataset = cae.DataSet.Define(stage, root_path.AppendChild(Tf.MakeValidIdentifier(name)))
        openfoam.PolyBoundaryMeshAPI.Apply(dataset.GetPrim())
        polyBoundaryMeshAPI = openfoam.PolyBoundaryMeshAPI(dataset.GetPrim())
        polyBoundaryMeshAPI.CreateNameAttr().Set(name)
        polyBoundaryMeshAPI.GetNameAttr().SetCustomDataByKey("omni:kit:locked", True)
        polyBoundaryMeshAPI.CreateStartFaceAttr().Set(data["startFace"])
        polyBoundaryMeshAPI.GetStartFaceAttr().SetCustomDataByKey("omni:kit:locked", True)
        polyBoundaryMeshAPI.CreateNFacesAttr().Set(data["nFaces"])
        polyBoundaryMeshAPI.GetNFacesAttr().SetCustomDataByKey("omni:kit:locked", True)
        boundary_meshes.append(dataset.GetPrim())
    return boundary_meshes


def read_patches(boundary_file: str) -> dict[str, dict[str, int]]:
    """
    Reads the OpenFOAM boundary file and returns a dictionary of patches.
    Each patch is represented as a dictionary with keys 'startFace' and 'nFaces'.
    """

    import pyparsing as pp

    LBRACE, RBRACE, SEMI = map(pp.Suppress, "{};")
    identifier = pp.Word(pp.alphas, pp.alphanums + "_-")
    number = pp.Word(pp.nums).setParseAction(lambda t: int(t[0]))
    value = pp.QuotedString('"') | pp.QuotedString("'") | pp.Word(pp.alphanums + "-_./ ()")

    # Key-value pair: key value;
    key_value = pp.Group(identifier("key") + value("value") + SEMI)

    # FoamFile { ... }
    foamfile_header = pp.Literal("FoamFile").suppress() + LBRACE + pp.Dict(pp.OneOrMore(key_value))("entries") + RBRACE

    patch = identifier("name") + LBRACE + pp.Dict(pp.OneOrMore(key_value))("properties") + RBRACE

    # Patch entries: name { properties }
    patch = pp.Group(identifier("name") + LBRACE + pp.Dict(pp.OneOrMore(key_value))("properties") + RBRACE)("patch")

    # FoamFile block: FoamFile { entries } count (patches)
    foamfile_block = (
        foamfile_header + number("count") + pp.Suppress("(") + pp.OneOrMore(patch)("patches") + pp.Suppress(")")
    )

    parser = foamfile_block.ignore(pp.cppStyleComment)

    try:
        parsed = parser.parseFile(boundary_file, parseAll=True)
    except pp.ParseException as e:
        raise RuntimeError(f"Failed to parse boundary file {boundary_file}: {e}")

    patches = {}
    for patch in parsed.patches:
        name = patch.name
        properties = patch.properties
        start_face = int(properties.get("startFace", 0))
        n_faces = int(properties.get("nFaces", 0))

        patches[name] = {"startFace": start_face, "nFaces": n_faces}

    return patches
