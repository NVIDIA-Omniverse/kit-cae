# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import os.path
from enum import Enum, auto
from logging import getLogger

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae, ensight
from pxr import Sdf, Tf, Usd, UsdGeom, UsdUtils

logger = getLogger(__name__)


class TimeOffset:
    offset: float = 0
    scale: float = 5.0

    @classmethod
    def get(cls, t):
        return t * cls.scale + cls.offset

    # @classmethod
    # def get(cls, t):
    #     t = t * cls.scale + cls.offset
    #     return cls.offset + (445 - t)


def get_resolved_path(path: Sdf.AssetPath) -> str:
    return path.resolvedPath


def get_geometry_filename(f):
    geo_filename = None
    ts = 1
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith("model:"):
            components = line.split(":")[-1].strip().split(" ")
            if len(components) >= 2:
                ts = int(components[0])
            geo_filename = components[-1]
    return geo_filename, ts


def get_timeset(f):
    nb_steps = 1
    start_number = 1
    increment = 1
    while True:
        line = f.readline().strip()
        if not line:
            break
        if line.startswith("number of steps:"):
            nb_steps = int(line.split(":")[-1])
        elif line.startswith("filename start number:"):
            start_number = int(line.split(":")[-1])
        elif line.startswith("filename increment:"):
            increment = int(line.split(":")[-1])
    return nb_steps, start_number, increment


def get_filenames(fname, start, increment, nb_steps):
    geo_filenames = []
    timesteps = []
    count = fname.count("*")
    if count == 0:
        return [fname], [start]

    for idx in range(nb_steps):
        # count '*' in geo_filename
        file_num = start + idx * increment

        # replace '*' with idx padded with count zeros
        path = fname.replace("*" * count, str(file_num).zfill(count))
        geo_filenames.append(path)
        timesteps.append(file_num)

    return geo_filenames, timesteps


def process_gold_case(stage: Usd.Stage, case_filename: str, rootPath: Sdf.Path):

    with open(case_filename, "r") as f:
        if f.readline().strip() != "FORMAT":
            raise RuntimeError("Invalid EnSight Gold file")
        if f.readline().strip().lower() != "type: ensight gold":
            raise RuntimeError("Invalid EnSight Gold file")
        processing_variables = False
        variables = {}
        while True:
            line = f.readline()
            if not line:
                break
            elif line.strip() == "GEOMETRY":
                processing_variables = False
                geo_filename, _ = get_geometry_filename(f)
            elif line.strip() == "VARIABLE":
                processing_variables = True
            elif line.strip() == "TIME":
                processing_variables = False
                nb_steps, start_number, increment = get_timeset(f)
                # time_values = get_timesteps()
            elif processing_variables and ":" in line:
                type, data = line.strip().split(":")
                type = type.strip()
                if type.startswith("complex") or type.startswith("constant"):
                    # for now, we skip these
                    continue
                elif type.startswith("scalar"):
                    var_type = ensight.Tokens.scalar
                elif type.startswith("vector"):
                    var_type = ensight.Tokens.vector
                elif type.startswith("tensor symm"):
                    var_type = ensight.Tokens.tensor
                elif type.startswith("tensor asymm"):
                    var_type = ensight.Tokens.tensor9

                if type.endswith(" per node"):
                    parts = data.strip().split(" ")
                    variables[parts[-2]] = (parts[-1], cae.Tokens.vertex, var_type)
                elif type.endswith(" per element"):
                    parts = data.strip().split(" ")
                    variables[parts[-2]] = (parts[-1], cae.Tokens.cell, var_type)
                else:
                    continue

    case_dir = os.path.dirname(case_filename)
    geo_filenames, geo_ts = get_filenames(geo_filename, start_number, increment, nb_steps)

    geo_file = EnsightGoldGeo(os.path.join(case_dir, geo_filenames[0]))
    for part, pieces in geo_file.parts_with_pieces():

        partPrim = stage.DefinePrim(rootPath.AppendChild(Tf.MakeValidIdentifier(part.description)), "EnsightGoldPart")

        caeFieldArrayClass = cae.FieldArray(stage.CreateClassPrim(partPrim.GetPath().AppendChild("GeoFieldArrayClass")))
        attr = caeFieldArrayClass.CreateFileNamesAttr()
        for ts, fname in enumerate(geo_filenames):  # zip(geo_ts, geo_filenames):
            attr.Set([os.path.join(case_dir, fname)], TimeOffset.get(ts))

        all_datasets = []
        # Create a singular dataset for the coordinates
        dataset = cae.DataSet.Define(stage, partPrim.GetPath().AppendChild("Coordinates"))
        all_datasets.append(dataset)
        cae.PointCloudAPI.Apply(dataset.GetPrim())
        pcAPI = cae.PointCloudAPI(dataset.GetPrim())

        coords = []
        for name, type in zip(
            ["X", "Y", "Z"], [ensight.Tokens.coordinateX, ensight.Tokens.coordinateY, ensight.Tokens.coordinateZ]
        ):
            field = ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild(f"Coordinate{name}"))
            field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
            field.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
            field.CreateTypeAttr().Set(type)
            field.CreatePartIdAttr().Set(part.id)
            coords.append(field.GetPath())

        pcAPI.CreateCoordinatesRel().SetTargets(coords)

        # Create a dataset for each piece
        for idx, piece in enumerate(pieces):
            dataset = cae.DataSet.Define(
                stage, partPrim.GetPath().AppendChild(Tf.MakeValidIdentifier(f"{part.description}_{idx}"))
            )
            all_datasets.append(dataset)
            ensight.UnstructuredPieceAPI.Apply(dataset.GetPrim())
            pieceAPI = ensight.UnstructuredPieceAPI(dataset.GetPrim())

            field = ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("Connectivity"))
            field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
            field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
            field.CreateTypeAttr().Set(ensight.Tokens.connectivity)
            field.CreatePartIdAttr().Set(part.id)
            field.CreatePieceIdAttr().Set(idx)

            pieceAPI.CreateConnectivityRel().SetTargets({field.GetPath()})
            pieceAPI.CreateCoordinatesRel().SetTargets(coords)
            pieceAPI.CreateElementTypeAttr().Set(piece.element_type.name)
            if piece.element_type == ElementType.nsided:
                field = ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("ElementNodeCounts"))
                field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
                field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
                field.CreateTypeAttr().Set(ensight.Tokens.elementNodeCounts)
                field.CreatePartIdAttr().Set(part.id)
                field.CreatePieceIdAttr().Set(idx)
                pieceAPI.CreateElementNodeCountsRel().SetTargets({field.GetPath()})
            elif piece.element_type == ElementType.nfaced:
                raise ValueError("nfaced elements not supported")

        if variables:
            scope = UsdGeom.Scope.Define(stage, partPrim.GetPath().AppendChild("Variables"))
            for name, (fname, assoc, var_type) in variables.items():
                clean_name = Tf.MakeValidIdentifier(name)
                field = ensight.GoldVarFieldArray.Define(stage, scope.GetPath().AppendChild(clean_name))
                field.CreateFieldAssociationAttr().Set(assoc)
                field.CreatePartIdAttr().Set(part.id)
                field.CreateTypeAttr().Set(var_type)

                attr = field.CreateFileNamesAttr()
                var_fnames, var_ts = get_filenames(fname, start_number, increment, nb_steps)
                for ts, var_fname in enumerate(var_fnames):  # zip(var_ts, var_fnames):
                    attr.Set([os.path.join(case_dir, var_fname)], TimeOffset.get(ts))

                attr = field.CreateGeoFileNamesAttr()
                for ts, gname in enumerate(geo_filenames):  # zip(geo_ts, geo_filenames):
                    attr.Set([os.path.join(case_dir, gname)], TimeOffset.get(ts))

                for dataset in all_datasets:
                    dataset.GetPrim().CreateRelationship(f"field:{clean_name}").SetTargets({field.GetPath()})


class ElementType(Enum):
    point = auto()
    bar2 = auto()
    bar3 = auto()
    tria3 = auto()
    tria6 = auto()
    quad4 = auto()
    quad8 = auto()
    tetra4 = auto()
    tetra10 = auto()
    pyramid5 = auto()
    pyramid13 = auto()
    penta6 = auto()
    penta15 = auto()
    hexa8 = auto()
    hexa20 = auto()
    nsided = auto()
    nfaced = auto()

    @classmethod
    def get_enum(cls, text: str):
        for member in cls:
            if member.name.lower() == text.lower():
                return member
        raise ValueError(f"No matching enum member for text: {text}")

    def get_num_nodes(self) -> int:
        if self == ElementType.point:
            return 1
        elif self == ElementType.bar2:
            return 2
        elif self == ElementType.bar3:
            return 3
        elif self == ElementType.tria3:
            return 3
        elif self == ElementType.tria6:
            return 6
        elif self == ElementType.quad4:
            return 4
        elif self == ElementType.quad8:
            return 8
        elif self == ElementType.tetra4:
            return 4
        elif self == ElementType.tetra10:
            return 10
        elif self == ElementType.pyramid5:
            return 5
        elif self == ElementType.pyramid13:
            return 13
        elif self == ElementType.penta6:
            return 6
        elif self == ElementType.penta15:
            return 15
        elif self == ElementType.hexa8:
            return 8
        elif self == ElementType.hexa20:
            return 20
        elif self == ElementType.nsided:
            return 0
        elif self == ElementType.nfaced:
            return 0
        else:
            raise ValueError(f"Unknown element type: {self}")


class EnsightGoldBase:
    class Part:
        description: str = None
        id: int = None
        num_nodes: int = None

    class Piece:
        element_type: ElementType = None
        num_elems: int = None
        id: int = None

    def readline(self, f, size):
        data = f.read(size)
        if not data:
            return ""
        return data.decode("utf-8").strip()


class EnsightGoldGeo(EnsightGoldBase):

    def __init__(self, filename):
        self.filename = filename

    def read_has_ids(self, f, what):
        header = self.readline(f, 80).lower().split(" ")
        assert header[0] == what
        assert header[1] == "id"
        return header[2] in ["given", "ignore"]

    def read_elem_type(self, f) -> ElementType:
        header = self.readline(f, 80).lower()
        return ElementType.get_enum(header)

    def parts(self, part_id: int = None):
        for type, part in self._read({"parts"}, part_id):
            if type == "parts":
                yield part

    def parts_with_pieces(self, part_id: int = None, piece_id: int = None):
        part = None
        pieces = []
        for type, data in self._read({"parts", "pieces"}, part_id, piece_id):
            if type == "parts":
                if part:
                    yield part, pieces
                part = data
                pieces = []
            elif type == "pieces":
                pieces.append(data)
        if part:
            yield part, pieces

    def coordinateX(self, part_id: int = None):
        for type, data in self._read({"coordinateX"}, part_id):
            if type == "coordinateX":
                yield data

    def coordinateY(self, part_id: int = None):
        for type, data in self._read({"coordinateY"}, part_id):
            if type == "coordinateY":
                yield data

    def coordinateZ(self, part_id: int = None):
        for type, data in self._read({"coordinateZ"}, part_id):
            if type == "coordinateZ":
                yield data

    def elems(self, part_id: int = None, piece_id: int = None):
        for type, data in self._read({"elems"}, part_id, piece_id):
            if type == "elems":
                yield data

    def elemNodeCounts(self, part_id: int = None, piece_id: int = None):
        for type, data in self._read({"elem_node_counts"}, part_id, piece_id):
            if type == "elem_node_counts":
                yield data

    def _read(self, components: set[str], part_id: int = None, piece_id: int = None):
        with open(self.filename, "rb") as f:
            if self.readline(f, 80) != "C Binary":
                raise ValueError("Only binary files are supported")
            self.readline(f, 80)  # description #1
            self.readline(f, 80)  # description #2
            has_node_ids = self.read_has_ids(f, "node")
            has_elem_ids = self.read_has_ids(f, "element")
            logger.info("has_node_ids=%s, has_elem_ids=%s", has_node_ids, has_elem_ids)

            header = self.readline(f, 80).lower()
            if header == "extents":
                extents = np.fromfile(f, dtype=np.float64, count=6)
                if "extents" in components:
                    yield "extents", extents
                header = self.readline(f, 80).lower()

            while header:
                logger.debug("header=%s", header)
                if header == "part":
                    part = EnsightGoldGeo.Part()
                    part.id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    part.description = self.readline(f, 80)
                    logger.info("part.id=%s, part_desc=%s", part.id, part.description)
                    assert (
                        self.readline(f, 80).lower() == "coordinates"
                    ), "Only unstructured parts are supported currently"
                    part.num_nodes = np.fromfile(f, dtype=np.int32, count=1)[0]

                    if "parts" in components and (part_id is None or part_id == part.id):
                        yield "parts", part

                    logger.info("part.desc=%s, part.id=%s, num_nodes=%s", part.description, part.id, part.num_nodes)

                    if has_node_ids:
                        f.seek(part.num_nodes * np.int32().itemsize, os.SEEK_CUR)
                    if "coordinateX" in components and (part_id is None or part_id == part.id):
                        yield "coordinateX", np.fromfile(f, dtype=np.float32, count=part.num_nodes)
                    else:
                        f.seek(part.num_nodes * np.float32().itemsize, os.SEEK_CUR)
                    if "coordinateY" in components and (part_id is None or part_id == part.id):
                        yield "coordinateY", np.fromfile(f, dtype=np.float32, count=part.num_nodes)
                    else:
                        f.seek(part.num_nodes * np.float32().itemsize, os.SEEK_CUR)
                    if "coordinateZ" in components and (part_id is None or part_id == part.id):
                        yield "coordinateZ", np.fromfile(f, dtype=np.float32, count=part.num_nodes)
                    else:
                        f.seek(part.num_nodes * np.float32().itemsize, os.SEEK_CUR)

                    header = self.readline(f, 80).lower()
                    piece_idx = 0
                    while header and header != "part":
                        piece = EnsightGoldGeo.Piece()
                        piece.element_type = ElementType.get_enum(header)
                        piece.num_elems = np.fromfile(f, dtype=np.int32, count=1)[0]
                        piece.id = piece_idx
                        piece_idx += 1

                        if (
                            "pieces" in components
                            and (part_id is None or part_id == part.id)
                            and (piece_id is None or piece_id == piece.id)
                        ):
                            yield "pieces", piece

                        if has_elem_ids:
                            f.seek(piece.num_elems * np.int32().itemsize, os.SEEK_CUR)

                        if piece.element_type == ElementType.nsided:
                            e_np = np.fromfile(f, dtype=np.int32, count=piece.num_elems)
                            if (
                                "elem_node_counts" in components
                                and (part_id is None or part_id == part.id)
                                and (piece_id is None or piece_id == piece.id)
                            ):
                                yield "elem_node_counts", e_np

                            if (
                                "elems" in components
                                and (part_id is None or part_id == part.id)
                                and (piece_id is None or piece_id == piece.id)
                            ):
                                yield "elems", np.fromfile(f, dtype=np.int32, count=e_np.sum())
                            else:
                                f.seek(e_np.sum() * np.int32().itemsize, os.SEEK_CUR)

                        elif piece.element_type == ElementType.nfaced:
                            raise ValueError("nfaced elements not supported")

                        else:
                            if (
                                "elems" in components
                                and (part_id is None or part_id == part.id)
                                and (piece_id is None or piece_id == piece.id)
                            ):
                                yield "elems", np.fromfile(
                                    f, dtype=np.int32, count=piece.num_elems * piece.element_type.get_num_nodes()
                                ).reshape((-1, piece.element_type.get_num_nodes()), order="C")
                            else:
                                f.seek(
                                    piece.num_elems * piece.element_type.get_num_nodes() * np.int32().itemsize,
                                    os.SEEK_CUR,
                                )
                        header = self.readline(f, 80).lower().strip()
                else:
                    raise ValueError(f"Unknown header: {header}")


class EnSightGoldVar(EnsightGoldBase):

    def __init__(self, var_filename: str, geo_filename, nb_comps: int, assoc: str):
        self.var_filename = var_filename
        self.nb_comps = nb_comps
        self.assoc = assoc
        self.geo = EnsightGoldGeo(geo_filename)

    def data(self, part_id: int = None, piece_id: int = None):
        if self.assoc == cae.Tokens.vertex:
            return self._read_node_var(part_id)
        elif self.assoc == cae.Tokens.cell:
            raise NotImplementedError("Cell data not supported")

    def _read_node_var(self, part_id: int):
        assert self.assoc == cae.Tokens.vertex
        part_generator = self.geo.parts()

        with open(self.var_filename, "rb") as f:
            self.readline(f, 80)  # description #1
            title = self.readline(f, 80)
            while title:
                if title == "part":
                    cur_id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    assert (
                        self.readline(f, 80).lower() == "coordinates"
                    ), "Only unstructured parts are supported currently"
                    part = next(part_generator)
                    while part.id < cur_id:
                        part = next(part_generator)
                    assert part.id == cur_id
                    if part_id == part.id:
                        array = np.fromfile(f, dtype=np.float32, count=part.num_nodes * self.nb_comps)
                        if self.nb_comps == 1:
                            yield array
                        else:
                            yield array.reshape((-1, self.nb_comps), order="F")

                        break  # we've read the required part
                    else:
                        f.seek(part.num_nodes * self.nb_comps * np.float32().itemsize, os.SEEK_CUR)
                title = self.readline(f, 80)


class EnsightGoldGeoDelegate(DataDelegateBase):

    def __init__(self, extensionId):
        super().__init__(extensionId)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode):
        try:
            return self.get_field_array_impl(prim, time)
        except FileNotFoundError as e:
            logger.error("Failed : %s", e)
            return None

    def get_field_array_impl(self, prim: Usd.Prim, time: Usd.TimeCode):
        primT = ensight.GoldGeoFieldArray(prim)
        fileNames = primT.GetFileNamesAttr().Get(time)
        ensight_type = primT.GetTypeAttr().Get(time)
        if len(fileNames) > 1:
            raise ValueError("Multiple files not supported")

        if ensight_type == ensight.Tokens.coordinateX:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateX(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == ensight.Tokens.coordinateY:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateY(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == ensight.Tokens.coordinateZ:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateZ(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == ensight.Tokens.connectivity:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.elems(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time))))
        elif ensight_type == ensight.Tokens.elementNodeCounts:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(
                list(geo.elemNodeCounts(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time)))
            )
        else:
            raise ValueError(f"Unknown ensight type: {ensight_type}")

    def can_provide(self, prim):
        if prim and prim.IsValid() and prim.IsA(ensight.GoldGeoFieldArray):
            return True
        return False


class EnsightGoldVarDelegate(DataDelegateBase):

    def __init__(self, extensionId):
        super().__init__(extensionId)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode):
        try:
            return self.get_field_array_impl(prim, time)
        except FileNotFoundError as e:
            logger.error("Failed : %s", e)
            return None

    def get_field_array_impl(self, prim: Usd.Prim, time: Usd.TimeCode):
        logger.info("get_field_array %s (%s)", prim, time)
        primT = ensight.GoldVarFieldArray(prim)
        fileNames = primT.GetFileNamesAttr().Get(time)
        geoFileNames = primT.GetGeoFileNamesAttr().Get(time)
        ensight_type = primT.GetTypeAttr().Get(time)
        assoc = primT.GetFieldAssociationAttr().Get(time)

        if len(fileNames) > 1:
            raise ValueError("Multiple files not supported")
        if len(geoFileNames) > 1:
            raise ValueError("Multiple geo files not supported")

        match ensight_type:
            case ensight.Tokens.scalar:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 1, assoc)
            case ensight.Tokens.vector:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 3, assoc)
            case ensight.Tokens.tensor:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 6, assoc)
            case ensight.Tokens.tensor9:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 9, assoc)
            case _:
                raise ValueError(f"Unknown ensight type: {ensight_type}")
        return np.concatenate(list(var.data(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time))))

    def can_provide(self, prim):
        if prim and prim.IsValid() and prim.IsA(ensight.GoldVarFieldArray):
            return True
        return False
