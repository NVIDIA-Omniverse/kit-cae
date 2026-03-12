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
import threading
from enum import Enum, auto
from logging import getLogger

import numpy as np
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.cae.schema import ensight as cae_ensight
from omni.client.utils import make_file_url_if_possible
from pxr import Sdf, Tf, Usd, UsdGeom

logger = getLogger(__name__)


class TimeOffset:
    offset: float = 0
    scale: float = 5.0

    def get(self, t):
        return t * self.scale + self.offset


def get_resolved_path(path: Sdf.AssetPath) -> str:
    return path.resolvedPath


def get_geometry_filename(f):
    geo_filename = None
    ts = 1
    while True:
        pos = f.tell()
        line = f.readline().strip()
        if not line:
            break
        # Skip comments
        if line.startswith("#"):
            continue
        if line in ["VARIABLE", "TIME", "FILE"]:
            # GEOMETRY section is done.
            f.seek(pos)
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
        # Skip comments
        if line.startswith("#"):
            continue
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


def _read_case_line(f):
    """Read the next non-comment, non-empty line from a case file."""
    while True:
        line = f.readline()
        if not line:  # EOF
            return None
        stripped = line.strip()
        # Skip comments and empty lines
        if stripped.startswith("#") or not stripped:
            continue
        return line


def process_gold_case(stage: Usd.Stage, case_filename: str, rootPath: Sdf.Path, timeOffset: TimeOffset):

    with open(case_filename, "r") as f:
        first_line = _read_case_line(f)
        if first_line is None or first_line.strip() != "FORMAT":
            raise RuntimeError("Invalid EnSight Gold file")
        # Normalize whitespace for flexible matching (some files have extra spaces)
        type_line_raw = _read_case_line(f)
        if type_line_raw is None:
            raise RuntimeError("Invalid EnSight Gold file")
        type_line = " ".join(type_line_raw.strip().lower().split())
        if type_line != "type: ensight gold":
            raise RuntimeError("Invalid EnSight Gold file")
        processing_variables = False
        variables = {}
        # Default values for files without a TIME section
        nb_steps = 1
        start_number = 1
        increment = 1
        while True:
            line = _read_case_line(f)
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
                    var_type = cae_ensight.Tokens.scalar
                elif type.startswith("vector"):
                    var_type = cae_ensight.Tokens.vector
                elif type.startswith("tensor symm"):
                    var_type = cae_ensight.Tokens.tensor
                elif type.startswith("tensor asymm"):
                    var_type = cae_ensight.Tokens.tensor9

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
        partDataset = cae.DataSet.Define(stage, rootPath.AppendChild(Tf.MakeValidIdentifier(part.description)))
        partPrim = partDataset.GetPrim()

        cae_ensight.UnstructuredPartAPI.Apply(partPrim)
        partAPI = cae_ensight.UnstructuredPartAPI(partPrim)
        partAPI.CreateIdAttr().Set(part.id)

        caeFieldArrayClass = cae.FieldArray(stage.CreateClassPrim(partPrim.GetPath().AppendChild("GeoFieldArrayClass")))
        attr = caeFieldArrayClass.CreateFileNamesAttr()
        for ts, fname in enumerate(geo_filenames):  # zip(geo_ts, geo_filenames):
            attr.Set([make_file_url_if_possible(os.path.join(case_dir, fname))], timeOffset.get(ts))

        # Create a singular dataset for the coordinates
        coordsDataset = cae.DataSet.Define(stage, partPrim.GetPath().AppendChild("Coordinates"))
        cae.PointCloudAPI.Apply(coordsDataset.GetPrim())
        pcAPI = cae.PointCloudAPI(coordsDataset.GetPrim())

        coords = []
        for name, type in zip(
            ["X", "Y", "Z"],
            [cae_ensight.Tokens.coordinateX, cae_ensight.Tokens.coordinateY, cae_ensight.Tokens.coordinateZ],
        ):
            field = cae_ensight.GoldGeoFieldArray.Define(
                stage, coordsDataset.GetPath().AppendChild(f"Coordinate{name}")
            )
            field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
            field.CreateFieldAssociationAttr().Set(cae.Tokens.vertex)
            field.CreateTypeAttr().Set(type)
            field.CreatePartIdAttr().Set(part.id)
            coords.append(field.GetPath())

        pcAPI.CreateCoordinatesRel().SetTargets(coords)
        partAPI.CreateCoordinatesRel().SetTargets(coords)

        # Create a dataset for each piece
        for idx, piece in enumerate(pieces):
            dataset = stage.DefinePrim(
                partPrim.GetPath().AppendChild(Tf.MakeValidIdentifier(f"Piece_{idx}")), "CaeEnSightPiece"
            )
            partAPI.CreatePiecesRel().AddTarget(dataset.GetPath())
            cae_ensight.UnstructuredPieceAPI.Apply(dataset.GetPrim())
            pieceAPI = cae_ensight.UnstructuredPieceAPI(dataset.GetPrim())

            field = cae_ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("Connectivity"))
            field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
            field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
            field.CreateTypeAttr().Set(cae_ensight.Tokens.connectivity)
            field.CreatePartIdAttr().Set(part.id)
            field.CreatePieceIdAttr().Set(idx)

            pieceAPI.CreateConnectivityRel().SetTargets({field.GetPath()})
            pieceAPI.CreateElementTypeAttr().Set(piece.element_type.name)

            if piece.element_type == ElementType.nsided:
                cae_ensight.NSidedPieceAPI.Apply(dataset.GetPrim())
                nsided_api = cae_ensight.NSidedPieceAPI(dataset.GetPrim())

                field = cae_ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("ElementNodeCounts"))
                field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
                field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
                field.CreateTypeAttr().Set(cae_ensight.Tokens.elementNodeCounts)
                field.CreatePartIdAttr().Set(part.id)
                field.CreatePieceIdAttr().Set(idx)
                nsided_api.CreateElementNodeCountsRel().SetTargets({field.GetPath()})

            elif piece.element_type == ElementType.nfaced:
                # raise ValueError("nfaced elements not supported")
                cae_ensight.NFacedPieceAPI.Apply(dataset.GetPrim())
                nface_api = cae_ensight.NFacedPieceAPI(dataset.GetPrim())

                field = cae_ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("ElementFaceCounts"))
                field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
                field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
                field.CreateTypeAttr().Set(cae_ensight.Tokens.elementFaceCounts)
                field.CreatePartIdAttr().Set(part.id)
                field.CreatePieceIdAttr().Set(idx)
                nface_api.CreateElementFaceCountsRel().SetTargets({field.GetPath()})

                field = cae_ensight.GoldGeoFieldArray.Define(stage, dataset.GetPath().AppendChild("FaceNodeCounts"))
                field.GetPrim().GetSpecializes().SetSpecializes([caeFieldArrayClass.GetPath()])
                field.CreateFieldAssociationAttr().Set(cae.Tokens.none)
                field.CreateTypeAttr().Set(cae_ensight.Tokens.faceNodeCounts)
                field.CreatePartIdAttr().Set(part.id)
                field.CreatePieceIdAttr().Set(idx)
                nface_api.CreateFaceNodeCountsRel().SetTargets({field.GetPath()})

        if variables:
            scope = UsdGeom.Scope.Define(stage, partPrim.GetPath().AppendChild("Variables"))
            for name, (fname, assoc, var_type) in variables.items():
                clean_name = Tf.MakeValidIdentifier(name)
                field = cae_ensight.GoldVarFieldArray.Define(stage, scope.GetPath().AppendChild(clean_name))
                field.CreateFieldAssociationAttr().Set(assoc)
                field.CreatePartIdAttr().Set(part.id)
                field.CreateTypeAttr().Set(var_type)

                attr = field.CreateFileNamesAttr()
                var_fnames, var_ts = get_filenames(fname, start_number, increment, nb_steps)
                for ts, var_fname in enumerate(var_fnames):  # zip(var_ts, var_fnames):
                    attr.Set([make_file_url_if_possible(os.path.join(case_dir, var_fname))], timeOffset.get(ts))

                attr = field.CreateGeoFileNamesAttr()
                for ts, gname in enumerate(geo_filenames):  # zip(geo_ts, geo_filenames):
                    attr.Set([make_file_url_if_possible(os.path.join(case_dir, gname))], timeOffset.get(ts))

                # Associate field with dataset(s) based on association type
                partDataset.GetPrim().CreateRelationship(f"field:{clean_name}").SetTargets({field.GetPath()})
                if assoc == cae.Tokens.vertex:
                    coordsDataset.GetPrim().CreateRelationship(f"field:{clean_name}").SetTargets({field.GetPath()})


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
        # Strip both whitespace and null bytes (some files use null padding)
        return data.decode("utf-8").strip("\x00").strip()


class _PieceIndex:
    """Byte-offset index entry for one piece inside a geo file part."""

    __slots__ = (
        "id",
        "element_type",
        "num_elems",
        "node_counts_offset",  # nsided: file offset to per-element node-count array
        "face_counts_offset",  # nfaced: file offset to per-element face-count array
        "face_node_counts_offset",  # nfaced: file offset to per-face node-count array
        "nb_faces",  # nfaced: total number of faces
        "conn_offset",  # file offset to connectivity array
        "total_conn_count",  # number of int32 values in connectivity array
    )

    def __init__(
        self,
        id,
        element_type,
        num_elems,
        node_counts_offset,
        face_counts_offset,
        face_node_counts_offset,
        nb_faces,
        conn_offset,
        total_conn_count,
    ):
        self.id = id
        self.element_type = element_type
        self.num_elems = num_elems
        self.node_counts_offset = node_counts_offset
        self.face_counts_offset = face_counts_offset
        self.face_node_counts_offset = face_node_counts_offset
        self.nb_faces = nb_faces
        self.conn_offset = conn_offset
        self.total_conn_count = total_conn_count


class _PartIndex:
    """Byte-offset index entry for one part inside a geo file."""

    __slots__ = ("id", "description", "num_nodes", "coord_x_offset", "coord_y_offset", "coord_z_offset", "pieces")

    def __init__(self, id, description, num_nodes, coord_x_offset, coord_y_offset, coord_z_offset):
        self.id = id
        self.description = description
        self.num_nodes = num_nodes
        self.coord_x_offset = coord_x_offset
        self.coord_y_offset = coord_y_offset
        self.coord_z_offset = coord_z_offset
        self.pieces: list[_PieceIndex] = []


class _GeoIndex:
    """Parsed file-offset index for an EnsightGold geo file."""

    def __init__(self):
        self.parts: list[_PartIndex] = []
        self.parts_by_id: dict[int, _PartIndex] = {}


# Module-level cache: geo filename -> _GeoIndex (built once per file, reused for all subsequent reads)
_geo_index_cache: dict[str, _GeoIndex] = {}
_geo_index_cache_lock = threading.Lock()


def _build_geo_index(filename: str) -> _GeoIndex:
    """Scan a geo file once and record byte offsets for every data array."""
    index = _GeoIndex()
    base = EnsightGoldBase()
    with open(filename, "rb") as f:
        if base.readline(f, 80) != "C Binary":
            raise ValueError("Only binary files are supported")
        base.readline(f, 80)  # description #1
        base.readline(f, 80)  # description #2

        node_id_hdr = base.readline(f, 80).lower().split()
        assert node_id_hdr[0] == "node" and node_id_hdr[1] == "id"
        has_node_ids = node_id_hdr[2] in ("given", "ignore")

        elem_id_hdr = base.readline(f, 80).lower().split()
        assert elem_id_hdr[0] == "element" and elem_id_hdr[1] == "id"
        has_elem_ids = elem_id_hdr[2] in ("given", "ignore")

        header = base.readline(f, 80).lower()
        if header == "extents":
            extents_pos = f.tell()
            np.fromfile(f, dtype=np.float32, count=6)
            header = base.readline(f, 80).lower()
            if header != "part" and header:
                f.seek(extents_pos)
                np.fromfile(f, dtype=np.float64, count=6)
                header = base.readline(f, 80).lower()

        while header:
            if header == "part":
                part_id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                description = base.readline(f, 80)
                assert base.readline(f, 80).lower() == "coordinates"
                num_nodes = int(np.fromfile(f, dtype=np.int32, count=1)[0])

                if has_node_ids:
                    f.seek(num_nodes * 4, os.SEEK_CUR)
                coord_x_offset = f.tell()
                f.seek(num_nodes * 4, os.SEEK_CUR)
                coord_y_offset = f.tell()
                f.seek(num_nodes * 4, os.SEEK_CUR)
                coord_z_offset = f.tell()
                f.seek(num_nodes * 4, os.SEEK_CUR)

                part_idx = _PartIndex(part_id, description, num_nodes, coord_x_offset, coord_y_offset, coord_z_offset)
                index.parts.append(part_idx)
                index.parts_by_id[part_id] = part_idx

                header = base.readline(f, 80).lower()
                piece_id = 0
                while header and header != "part":
                    elem_type = ElementType.get_enum(header)
                    num_elems = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    if has_elem_ids:
                        f.seek(num_elems * 4, os.SEEK_CUR)

                    node_counts_offset = face_counts_offset = face_node_counts_offset = conn_offset = None
                    nb_faces = total_conn_count = 0

                    if elem_type == ElementType.nsided:
                        node_counts_offset = f.tell()
                        e_np = np.fromfile(f, dtype=np.int32, count=num_elems)
                        total_conn_count = int(e_np.sum())
                        conn_offset = f.tell()
                        f.seek(total_conn_count * 4, os.SEEK_CUR)
                    elif elem_type == ElementType.nfaced:
                        face_counts_offset = f.tell()
                        nf_e = np.fromfile(f, dtype=np.int32, count=num_elems)
                        nb_faces = int(nf_e.sum())
                        face_node_counts_offset = f.tell()
                        np_f_e = np.fromfile(f, dtype=np.int32, count=nb_faces)
                        total_conn_count = int(np_f_e.sum())
                        conn_offset = f.tell()
                        f.seek(total_conn_count * 4, os.SEEK_CUR)
                    else:
                        total_conn_count = num_elems * elem_type.get_num_nodes()
                        conn_offset = f.tell()
                        f.seek(total_conn_count * 4, os.SEEK_CUR)

                    part_idx.pieces.append(
                        _PieceIndex(
                            piece_id,
                            elem_type,
                            num_elems,
                            node_counts_offset,
                            face_counts_offset,
                            face_node_counts_offset,
                            nb_faces,
                            conn_offset,
                            total_conn_count,
                        )
                    )
                    piece_id += 1
                    header = base.readline(f, 80).lower().strip()
            else:
                raise ValueError(f"Unknown header: {header}")
    return index


def _get_geo_index(filename: str) -> _GeoIndex:
    index = _geo_index_cache.get(filename)
    if index is None:
        with _geo_index_cache_lock:
            index = _geo_index_cache.get(filename)
            if index is None:
                index = _build_geo_index(filename)
                _geo_index_cache[filename] = index
    return index


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

    def elemFaceCounts(self, part_id: int = None, piece_id: int = None):
        for type, data in self._read({"elem_face_counts"}, part_id, piece_id):
            if type == "elem_face_counts":
                yield data

    def faceNodeCounts(self, part_id: int = None, piece_id: int = None):
        for type, data in self._read({"face_node_counts"}, part_id, piece_id):
            if type == "face_node_counts":
                yield data

    def _read(self, components: set[str], part_id: int = None, piece_id: int = None):
        index = _get_geo_index(self.filename)
        parts = (
            index.parts if part_id is None else ([index.parts_by_id[part_id]] if part_id in index.parts_by_id else [])
        )
        with open(self.filename, "rb") as f:
            for pi in parts:
                part = EnsightGoldGeo.Part()
                part.id = pi.id
                part.description = pi.description
                part.num_nodes = pi.num_nodes
                if "parts" in components:
                    yield "parts", part
                if "coordinateX" in components:
                    f.seek(pi.coord_x_offset)
                    yield "coordinateX", np.fromfile(f, dtype=np.float32, count=pi.num_nodes)
                if "coordinateY" in components:
                    f.seek(pi.coord_y_offset)
                    yield "coordinateY", np.fromfile(f, dtype=np.float32, count=pi.num_nodes)
                if "coordinateZ" in components:
                    f.seek(pi.coord_z_offset)
                    yield "coordinateZ", np.fromfile(f, dtype=np.float32, count=pi.num_nodes)
                pieces = pi.pieces if piece_id is None else [q for q in pi.pieces if q.id == piece_id]
                for qi in pieces:
                    piece = EnsightGoldGeo.Piece()
                    piece.id = qi.id
                    piece.element_type = qi.element_type
                    piece.num_elems = qi.num_elems
                    if "pieces" in components:
                        yield "pieces", piece
                    if qi.element_type == ElementType.nsided:
                        if "elem_node_counts" in components:
                            f.seek(qi.node_counts_offset)
                            yield "elem_node_counts", np.fromfile(f, dtype=np.int32, count=qi.num_elems)
                        if "elems" in components:
                            f.seek(qi.conn_offset)
                            yield "elems", np.fromfile(f, dtype=np.int32, count=qi.total_conn_count)
                    elif qi.element_type == ElementType.nfaced:
                        if "elem_face_counts" in components:
                            f.seek(qi.face_counts_offset)
                            yield "elem_face_counts", np.fromfile(f, dtype=np.int32, count=qi.num_elems)
                        if "face_node_counts" in components:
                            f.seek(qi.face_node_counts_offset)
                            yield "face_node_counts", np.fromfile(f, dtype=np.int32, count=qi.nb_faces)
                        if "elems" in components:
                            f.seek(qi.conn_offset)
                            yield "elems", np.fromfile(f, dtype=np.int32, count=qi.total_conn_count)
                    else:
                        if "elems" in components:
                            f.seek(qi.conn_offset)
                            yield "elems", np.fromfile(f, dtype=np.int32, count=qi.total_conn_count)


class EnSightGoldVar(EnsightGoldBase):

    def __init__(self, var_filename: str, geo_filename, nb_comps: int, assoc: str):
        self.var_filename = var_filename
        self.nb_comps = nb_comps
        self.assoc = assoc
        self.geo = EnsightGoldGeo(geo_filename)

    def data(self, part_id: int = None, piece_id: int = None):
        if part_id == -1:
            part_id = None
        if piece_id == -1:
            piece_id = None

        if self.assoc == cae.Tokens.vertex:
            return self._read_node_var(part_id)
        elif self.assoc == cae.Tokens.cell:
            return self._read_elem_var(part_id, piece_id)
        else:
            raise NotImplementedError(f"Association type {self.assoc} not supported")

    def _read_node_var(self, part_id: int):
        assert self.assoc == cae.Tokens.vertex, "Association must be vertex for node variables"
        assert part_id is not None, "part_id must be specified for cell variables"
        geo_index = _get_geo_index(self.geo.filename)
        with open(self.var_filename, "rb") as f:
            self.readline(f, 80)  # description #1
            title = self.readline(f, 80)
            while title:
                if title == "part":
                    cur_id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    assert self.readline(f, 80).lower() == "coordinates"
                    pi = geo_index.parts_by_id[cur_id]
                    if part_id == cur_id:
                        yield self._reshape(np.fromfile(f, dtype=np.float32, count=pi.num_nodes * self.nb_comps))
                        break  # we've read the required part
                    else:
                        f.seek(pi.num_nodes * self.nb_comps * 4, os.SEEK_CUR)
                title = self.readline(f, 80)

    def _read_elem_var(self, part_id: int, piece_id: int):
        assert self.assoc == cae.Tokens.cell, "Association must be cell for element variables"
        assert part_id is not None, "part_id must be specified for cell variables"
        geo_index = _get_geo_index(self.geo.filename)
        with open(self.var_filename, "rb") as f:
            self.readline(f, 80)  # description #1
            header = self.readline(f, 80)
            while header:
                if header == "part":
                    cur_part_id = int(np.fromfile(f, dtype=np.int32, count=1)[0])
                    pi = geo_index.parts_by_id[cur_part_id]
                    for qi in pi.pieces:
                        element_type = self.readline(f, 80).lower()
                        assert (
                            element_type == qi.element_type.name.lower()
                        ), "Element type mismatch between geo and var files"
                        nb_values = qi.num_elems * self.nb_comps
                        if part_id == cur_part_id and (piece_id is None or piece_id == qi.id):
                            yield self._reshape(np.fromfile(f, dtype=np.float32, count=nb_values))
                            if piece_id is not None:
                                break  # we've read the required piece
                        else:
                            f.seek(nb_values * 4, os.SEEK_CUR)
                    if part_id == cur_part_id:
                        break  # we've read the required part, no need to continue
                header = self.readline(f, 80)

    def _reshape(self, array):
        if self.nb_comps == 1:
            return array
        else:
            return array.reshape((-1, self.nb_comps), order="F")


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
        primT = cae_ensight.GoldGeoFieldArray(prim)
        fileNames = primT.GetFileNamesAttr().Get(time)
        ensight_type = primT.GetTypeAttr().Get(time)
        if len(fileNames) > 1:
            raise ValueError("Multiple files not supported")

        if ensight_type == cae_ensight.Tokens.coordinateX:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateX(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == cae_ensight.Tokens.coordinateY:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateY(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == cae_ensight.Tokens.coordinateZ:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.coordinateZ(primT.GetPartIdAttr().Get(time))))
        elif ensight_type == cae_ensight.Tokens.connectivity:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(list(geo.elems(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time))))

        elif ensight_type == cae_ensight.Tokens.elementNodeCounts:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(
                list(geo.elemNodeCounts(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time)))
            )
        elif ensight_type == cae_ensight.Tokens.elementFaceCounts:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(
                list(geo.elemFaceCounts(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time)))
            )
        elif ensight_type == cae_ensight.Tokens.faceNodeCounts:
            geo = EnsightGoldGeo(get_resolved_path(fileNames[0]))
            return np.concatenate(
                list(geo.faceNodeCounts(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time)))
            )
        else:
            raise ValueError(f"Unknown ensight type: {ensight_type}")

    def can_provide(self, prim):
        if prim and prim.IsValid() and prim.IsA(cae_ensight.GoldGeoFieldArray):
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
        primT = cae_ensight.GoldVarFieldArray(prim)
        fileNames = primT.GetFileNamesAttr().Get(time)
        geoFileNames = primT.GetGeoFileNamesAttr().Get(time)
        ensight_type = primT.GetTypeAttr().Get(time)
        assoc = primT.GetFieldAssociationAttr().Get(time)

        if len(fileNames) > 1:
            raise ValueError("Multiple files not supported")
        if len(geoFileNames) > 1:
            raise ValueError("Multiple geo files not supported")

        match ensight_type:
            case cae_ensight.Tokens.scalar:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 1, assoc)
            case cae_ensight.Tokens.vector:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 3, assoc)
            case cae_ensight.Tokens.tensor:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 6, assoc)
            case cae_ensight.Tokens.tensor9:
                var = EnSightGoldVar(get_resolved_path(fileNames[0]), get_resolved_path(geoFileNames[0]), 9, assoc)
            case _:
                raise ValueError(f"Unknown cae_ensight type: {ensight_type}")
        try:
            return np.concatenate(list(var.data(primT.GetPartIdAttr().Get(time), primT.GetPieceIdAttr().Get(time))))
        except Exception as e:
            logger.exception("Error reading variable data: %s", e, exc_info=True)
            return None

    def can_provide(self, prim):
        if prim and prim.IsValid() and prim.IsA(cae_ensight.GoldVarFieldArray):
            return True
        return False
