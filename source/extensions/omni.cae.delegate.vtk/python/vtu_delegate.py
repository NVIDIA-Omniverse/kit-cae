# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

__all__ = ["VTUDataDelegate", "VTUMetadata"]

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from logging import getLogger

import numpy as np
import pyparsing as pp
from omni.cae.data.delegates import DataDelegateBase
from omni.cae.schema import cae
from omni.cae.schema import vtk as cae_vtk
from omni.client import get_local_file
from omni.stageupdate import get_stage_update_interface
from pxr import Usd

logger = getLogger(__name__)

# Map VTK XML type strings to numpy dtypes
_VTK_DTYPE_MAP: dict[str, type] = {
    "Float32": np.float32,
    "Float64": np.float64,
    "Int8": np.int8,
    "Int16": np.int16,
    "Int32": np.int32,
    "Int64": np.int64,
    "UInt8": np.uint8,
    "UInt16": np.uint16,
    "UInt32": np.uint32,
    "UInt64": np.uint64,
}

# Grammar for a single XML attribute:  name="value"  or  name='value'
_xml_attr = pp.Group(
    pp.Word(pp.alphanums + "_-:") + pp.Suppress(pp.Literal("=")) + (pp.QuotedString('"') | pp.QuotedString("'"))
)


def _xml_open_tag(name: str):
    """Return a grammar that matches  <name attr1="v1" ...>  and produces a Dict of attributes."""
    return pp.Suppress(pp.Literal(f"<{name}")) + pp.Dict(pp.ZeroOrMore(_xml_attr)) + pp.Suppress(pp.Literal(">"))


_vtk_file_tag = _xml_open_tag("VTKFile")
_piece_tag = _xml_open_tag("Piece")
_appended_data_tag = _xml_open_tag("AppendedData")


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataArrayInfo:
    """Metadata for a single <DataArray> element in the VTU XML header."""

    name: str
    dtype: np.dtype
    num_components: int  # components per element (1 for scalars, 3 for 3-D vectors, …)
    offset: int  # byte offset within the appended binary payload (relative to appended_data_offset)
    section: str  # owning section: "PointData", "CellData", "Points", or "Cells"


@dataclass
class PieceMetadata:
    """Parsed metadata for a single <Piece> element."""

    num_points: int
    num_cells: int
    # Field arrays at nodes/vertices, keyed by array Name
    point_arrays: dict[str, DataArrayInfo] = field(default_factory=dict)
    # Field arrays at cells, keyed by array Name
    cell_arrays: dict[str, DataArrayInfo] = field(default_factory=dict)
    # Coordinate array from <Points> (Name is typically "Points")
    points: DataArrayInfo | None = None
    # Topology arrays from <Cells>, keyed by Name
    # Standard names: "connectivity", "offsets", "types"
    # Extended names (polyhedra): "face_connectivity", "face_offsets",
    #   "polyhedron_to_faces", "polyhedron_offsets"
    cells: dict[str, DataArrayInfo] = field(default_factory=dict)


def _parse_data_array(elem: ET.Element, section: str) -> DataArrayInfo:
    attrs = elem.attrib
    type_str = attrs.get("type", "Float32")
    dtype = np.dtype(_VTK_DTYPE_MAP.get(type_str, np.float32))
    return DataArrayInfo(
        name=attrs.get("Name", ""),
        dtype=dtype,
        num_components=int(attrs.get("NumberOfComponents", "1")),
        offset=int(attrs.get("offset", "0")),
        section=section,
    )


def _parse_piece(piece_elem: ET.Element) -> PieceMetadata:
    meta = PieceMetadata(
        num_points=int(piece_elem.attrib.get("NumberOfPoints", "0")),
        num_cells=int(piece_elem.attrib.get("NumberOfCells", "0")),
    )

    pd = piece_elem.find("PointData")
    if pd is not None:
        for da in pd.findall("DataArray"):
            info = _parse_data_array(da, "PointData")
            if info.name:
                meta.point_arrays[info.name] = info

    cd = piece_elem.find("CellData")
    if cd is not None:
        for da in cd.findall("DataArray"):
            info = _parse_data_array(da, "CellData")
            if info.name:
                meta.cell_arrays[info.name] = info

    pts = piece_elem.find("Points")
    if pts is not None:
        da = pts.find("DataArray")
        if da is not None:
            meta.points = _parse_data_array(da, "Points")

    cells = piece_elem.find("Cells")
    if cells is not None:
        for da in cells.findall("DataArray"):
            info = _parse_data_array(da, "Cells")
            if info.name:
                meta.cells[info.name] = info

    return meta


@dataclass
class VTUMetadata:
    """
    Header metadata parsed from a VTK XML UnstructuredGrid (.vtu) file.

    Binary layout of a raw-appended VTU file::

        <AppendedData encoding="raw">\\n   _<block0><block1>...

    Each uncompressed block:  [N : header_dtype][N bytes of element data]
    Each compressed block:    [num_blocks, uncompressed_size, partial_size,
                               comp_size_0, …, comp_size_n : header_dtype]
                              [compressed_data_0][…][compressed_data_n]

    ``appended_data_offset`` is the byte offset in the *file* of the first
    byte of ``<block0>`` (the byte immediately after the ``_`` marker).

    Each ``DataArrayInfo.offset`` value is relative to ``appended_data_offset``.
    """

    byte_order: str  # "LittleEndian" or "BigEndian"
    header_dtype: np.dtype  # dtype of per-block length prefix (UInt32 or UInt64)
    compressor: str | None  # e.g. "vtkLZ4DataCompressor", or None if uncompressed
    pieces: list[PieceMetadata]
    appended_data_offset: int  # file byte offset of the first binary byte (after '_')

    @classmethod
    def parse(cls, filename: str) -> "VTUMetadata":
        """
        Parse the XML header of a raw-appended VTU file.

        Parameters
        ----------
        filename : str
            Local filesystem path to the .vtu file.

        Raises
        ------
        ValueError
            If no ``<AppendedData>`` section is found.
        OSError
            On file read errors.
        """
        _CHUNK = 65536  # 64 KB; enough for any realistic VTU header

        with open(filename, "rb") as f:
            raw = b""
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    raise ValueError(f"No <AppendedData> tag found in {filename!r}")
                raw += chunk
                if b"<AppendedData" in raw:
                    # Read a small extra buffer to guarantee the '_' marker is captured
                    raw += f.read(256)
                    break

        # Locate the binary payload.
        # Structure: <AppendedData encoding="raw">\n   _<binary...>
        appended_tag_start = raw.index(b"<AppendedData")
        tag_close = raw.index(b">", appended_tag_start)  # closing '>' of opening tag
        underscore = raw.index(b"_", tag_close)  # '_' data marker
        appended_data_offset = underscore + 1  # first byte of binary payload

        # Parse the XML header: everything before <AppendedData is well-formed
        # XML except that it is missing the closing </VTKFile> tag.
        header_text = raw[:appended_tag_start].decode("ascii", errors="ignore")
        root = ET.fromstring(header_text + "</VTKFile>")

        byte_order = root.attrib.get("byte_order", "LittleEndian")
        header_type_str = root.attrib.get("header_type", "UInt32")
        header_dtype = np.dtype(_VTK_DTYPE_MAP.get(header_type_str, np.uint32))
        compressor = root.attrib.get("compressor") or None

        pieces: list[PieceMetadata] = []
        ug = root.find("UnstructuredGrid")
        if ug is not None:
            for piece_elem in ug.findall("Piece"):
                pieces.append(_parse_piece(piece_elem))

        return cls(
            byte_order=byte_order,
            header_dtype=header_dtype,
            compressor=compressor,
            pieces=pieces,
            appended_data_offset=appended_data_offset,
        )


# ---------------------------------------------------------------------------
# Binary reading helpers
# ---------------------------------------------------------------------------


def _locate_array(piece: PieceMetadata, special, assoc, array_name: str) -> DataArrayInfo | None:
    """
    Map a (special, assoc, array_name) triple to the DataArrayInfo to read.

    Mirrors the dispatch logic in VTKDataDelegate.get_field_array but uses
    the VTU array names written by VTK 2.3.
    """
    if special == cae_vtk.Tokens.points:
        return piece.points
    if special == cae_vtk.Tokens.connectivity_array:
        return piece.cells.get("connectivity")
    if special == cae_vtk.Tokens.connectivity_offsets:
        return piece.cells.get("offsets")
    if special == cae_vtk.Tokens.cell_types:
        return piece.cells.get("types")
    if special == cae_vtk.Tokens.polyhedron_faces_offsets:
        return piece.cells.get("face_offsets")
    if special == cae_vtk.Tokens.polyhedron_faces_connectivity_array:
        return piece.cells.get("face_connectivity")
    if special == cae_vtk.Tokens.polyhedron_face_locations_offsets:
        return piece.cells.get("polyhedron_offsets")
    if special == cae_vtk.Tokens.polyhedron_face_locations_connectivity_array:
        return piece.cells.get("polyhedron_to_faces")
    if special == cae_vtk.Tokens.none:
        if assoc == cae.Tokens.vertex:
            # Primary lookup; fall back to cell data for dual-mesh arrays
            return piece.point_arrays.get(array_name) or piece.cell_arrays.get(array_name)
        if assoc == cae.Tokens.cell:
            return piece.cell_arrays.get(array_name)
    return None


def _decompress_lz4_blocks(f, header_dtype: np.dtype, elem_dtype: np.dtype) -> np.ndarray:
    """
    Read and decompress one VTK LZ4-compressed DataArray block from *f*.

    Block header layout (all values in *header_dtype*):
        num_blocks | uncompressed_block_size | last_partial_size |
        comp_size_0 … comp_size_{num_blocks-1}

    The total uncompressed size is computed from the header so the output
    array can be allocated once.  Each decompressed block is then written
    directly into a ``memoryview`` slice of that array, avoiding the
    intermediate joined-bytes allocation that ``b"".join`` would require.

    Returns an array of shape ``(count + 1,)`` with ``[0] = 0`` and the
    element data in ``[1:]``.
    """
    import lz4.block as lz4_block

    h = header_dtype.itemsize
    num_blocks = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    uncomp_block_size = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    partial_size = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    comp_sizes = np.frombuffer(f.read(h * num_blocks), dtype=header_dtype)

    last_usize = partial_size if partial_size > 0 else uncomp_block_size
    total_uncomp = (num_blocks - 1) * uncomp_block_size + last_usize
    out = np.empty(total_uncomp // elem_dtype.itemsize + 1, dtype=elem_dtype)
    out[0] = 0
    out_bytes = memoryview(out[1:].view(np.uint8))

    ustart = 0
    for i, csize in enumerate(comp_sizes):
        usize = int(partial_size) if (i == num_blocks - 1 and partial_size > 0) else uncomp_block_size
        out_bytes[ustart : ustart + usize] = lz4_block.decompress(f.read(int(csize)), uncompressed_size=usize)
        ustart += usize

    return out


def _decompress_zlib_blocks(f, header_dtype: np.dtype, elem_dtype: np.dtype) -> np.ndarray:
    """
    Read and decompress one VTK zlib-compressed DataArray block from *f*.

    Same allocation strategy as :func:`_decompress_lz4_blocks`: the header
    is parsed first so the output array can be pre-allocated, and each block
    is decompressed directly into a ``memoryview`` slice.
    """
    import zlib

    h = header_dtype.itemsize
    num_blocks = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    uncomp_block_size = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    partial_size = int(np.frombuffer(f.read(h), dtype=header_dtype)[0])
    comp_sizes = np.frombuffer(f.read(h * num_blocks), dtype=header_dtype)

    last_usize = partial_size if partial_size > 0 else uncomp_block_size
    total_uncomp = (num_blocks - 1) * uncomp_block_size + last_usize
    out = np.empty(total_uncomp // elem_dtype.itemsize + 1, dtype=elem_dtype)
    out[0] = 0
    out_bytes = memoryview(out[1:].view(np.uint8))

    ustart = 0
    for i, csize in enumerate(comp_sizes):
        usize = int(partial_size) if (i == num_blocks - 1 and partial_size > 0) else uncomp_block_size
        out_bytes[ustart : ustart + usize] = zlib.decompress(f.read(int(csize)))
        ustart += usize

    return out


def _read_array_from_file(fname: str, meta: VTUMetadata, info: DataArrayInfo, prepend_zero: bool = False) -> np.ndarray:
    """
    Seek to and read the binary block for *info* in *fname*, returning a numpy array.

    Handles uncompressed, LZ4, and zlib block formats.  The returned array
    dtype and byte order are derived from *info* and *meta* respectively.

    The internal allocation is always ``count + 1`` elements with ``[0] = 0``
    and element data in ``[1:]``.  When *prepend_zero* is ``True`` the full
    ``count + 1`` array is returned (CSR offset format); otherwise the
    ``[1:]`` view is returned (no extra copy).
    """
    byte_order = "<" if meta.byte_order == "LittleEndian" else ">"
    header_dtype = meta.header_dtype.newbyteorder(byte_order)
    elem_dtype = info.dtype.newbyteorder(byte_order)

    with open(fname, "rb") as f:
        f.seek(meta.appended_data_offset + info.offset)

        if meta.compressor is None:
            block_size = int(np.frombuffer(f.read(header_dtype.itemsize), dtype=header_dtype)[0])
            count = block_size // elem_dtype.itemsize
            out = np.empty(count + 1, dtype=elem_dtype)
            out[0] = 0
            f.readinto(out[1:].view(np.uint8))
        elif meta.compressor == "vtkLZ4DataCompressor":
            out = _decompress_lz4_blocks(f, header_dtype, elem_dtype)
        elif meta.compressor == "vtkZLibDataCompressor":
            out = _decompress_zlib_blocks(f, header_dtype, elem_dtype)
        else:
            raise NotImplementedError(f"Unsupported VTK compressor: {meta.compressor!r}")

    if info.num_components > 1:
        return out[1:].reshape(-1, info.num_components)
    return out if prepend_zero else out[1:]


class VTUReader:

    @classmethod
    def can_read(cls, filename: str) -> bool:
        """Return True if the file is a VTU with raw-appended binary data and VTK XML version 2.3."""
        fname = get_local_file(filename)[1]
        try:
            with open(fname, "rb") as f:
                # Only the XML header is needed; the binary payload begins after <AppendedData>.
                # 16 KB covers even headers with many DataArray entries.
                header = f.read(16384).decode("ascii", errors="ignore")
        except OSError:
            return False

        vtk_matches = _vtk_file_tag.searchString(header)
        if not vtk_matches:
            return False
        attrs = vtk_matches[0]
        if attrs.get("type") != "UnstructuredGrid" or attrs.get("version") != "2.3":
            return False

        if len(_piece_tag.searchString(header)) != 1:
            return False

        appended_matches = _appended_data_tag.searchString(header)
        return any(m.get("encoding") == "raw" for m in appended_matches)


class VTUDataDelegate(DataDelegateBase):

    def __init__(self, ext_id):
        super().__init__(ext_id)
        # filename -> VTUMetadata; cleared on stage detach
        self._meta_cache: dict[str, VTUMetadata] = {}
        self._stage_subscription = get_stage_update_interface().create_stage_update_node(
            "cae.vtu.delegate",
            on_detach_fn=self._meta_cache.clear,
        )

    def __del__(self):
        if self._stage_subscription:
            del self._stage_subscription
            self._stage_subscription = None

    def _get_meta(self, filename: str) -> VTUMetadata:
        if filename not in self._meta_cache:
            self._meta_cache[filename] = VTUMetadata.parse(filename)
        return self._meta_cache[filename]

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        primT = cae_vtk.FieldArray(prim)
        array_name = primT.GetArrayNameAttr().Get(time)
        file_names = primT.GetFileNamesAttr().Get(time)
        assoc = primT.GetFieldAssociationAttr().Get(time)
        special = primT.GetSpecialAttr().Get(time)
        logger.info("start reading %s (%s)", special, prim.GetPath())

        arrays = []
        for f_ref in file_names:
            fname = get_local_file(f_ref.resolvedPath)[1]
            meta = self._get_meta(fname)
            piece = meta.pieces[0]

            info = _locate_array(piece, special, assoc, array_name)
            if info is None:
                continue

            # VTK XML stores offset arrays with num_cells entries (end-only,
            # no leading 0). VTK's internal GetOffsetsArray() and DAV both use
            # CSR format: num_cells+1 entries with a leading 0.
            needs_csr = special in (
                cae_vtk.Tokens.connectivity_offsets,
                cae_vtk.Tokens.polyhedron_faces_offsets,
                cae_vtk.Tokens.polyhedron_face_locations_offsets,
            )
            array = _read_array_from_file(fname, meta, info, prepend_zero=needs_csr)

            # Type promotion: IFieldArray does not support narrow int/float types
            if np.issubdtype(array.dtype, np.integer) and array.itemsize < 4:
                array = array.astype(np.int32, copy=False)
            elif np.issubdtype(array.dtype, np.unsignedinteger) and array.itemsize < 4:
                array = array.astype(np.uint32, copy=False)
            elif np.issubdtype(array.dtype, np.floating) and array.itemsize < 4:
                array = array.astype(np.float32, copy=False)
            arrays.append(array)

        if not arrays:
            return None
        if len(arrays) > 1:
            logger.info("  concatenating ...")
            return np.concatenate(arrays)
        logger.info("  done.")
        return arrays[0]

    def can_provide(self, prim: Usd.Prim) -> bool:
        if prim and prim.IsValid() and prim.IsA(cae_vtk.FieldArray):
            primT = cae_vtk.FieldArray(prim)
            fileNames = primT.GetFileNamesAttr().Get(Usd.TimeCode.EarliestTime())
            if not all(f.resolvedPath.split(".")[-1].lower() == "vtu" for f in fileNames):
                return False

            # check if this is form we can read
            if not all(VTUReader.can_read(f.resolvedPath) for f in fileNames):
                return False

            return True

        return False
