# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from logging import getLogger

import dav
import numpy as np
import warp as wp
from dav.data_models.custom import surface_mesh
from dav.data_models.openfoam import boundary_mesh as openfoam_boundary_mesh
from dav.data_models.openfoam import polymesh as openfoam_polymesh
from dav.data_models.sids import nface_n as dav_sids_nface_n
from dav.data_models.sids import sids_shapes
from dav.data_models.sids import unstructured as dav_sids_unstructured
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.schema import cae
from omni.cae.schema import ensight as cae_ensight
from omni.cae.schema import openfoam as cae_openfoam
from omni.cae.schema import sids as cae_sids
from omni.cae.schema import vtk as cae_vtk
from pxr import Gf, Usd, UsdGeom

from .command_types import ConvertToDAVDataSet, GetField

logger = getLogger(__name__)

# dav.config.enable_timing = True
# dav.config.enable_nvtx = True


@wp.kernel
def _map_nface_to_ngon_cell_field_indices_kernel(
    connectivity: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    ngon_start: wp.int32,
    ngon_count: wp.int32,
    field_offset: wp.int32,
):
    cell_idx = wp.tid()
    start = offsets[cell_idx]
    end = offsets[cell_idx + 1]
    cell_field_idx = field_offset + cell_idx

    for offset in range(start, end):
        face_id = connectivity[offset]
        if face_id < 0:
            face_id = -face_id

        face_idx = face_id - ngon_start
        if face_idx >= 0 and face_idx < ngon_count:
            wp.atomic_min(indices, face_idx, cell_field_idx)


@wp.kernel
def _validate_ngon_cell_field_indices_kernel(
    indices: wp.array(dtype=wp.int32), missing_sentinel: wp.int32, missing_info: wp.array(dtype=wp.int32)
):
    idx = wp.tid()
    if indices[idx] == missing_sentinel:
        wp.atomic_add(missing_info, 0, 1)
        wp.atomic_min(missing_info, 1, idx)


class UsdGeomMeshConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a UsdGeomMesh CAE dataset to a DAV DataSet.

    This builds a DAV dataset using the surface_mesh data model, which follows
    the UsdGeomMesh specification directly.
    """

    def _apply_xform(self, mesh: UsdGeom.Mesh, coords: np.ndarray, timeCode: Usd.TimeCode) -> np.ndarray:
        _xform_cache = UsdGeom.XformCache(Usd.TimeCode.EarliestTime())
        matrix: Gf.Matrix4d = _xform_cache.GetLocalTransformation(mesh.GetPrim())[0]
        if matrix:
            coords_h = np.hstack([coords, np.ones((coords.shape[0], 1))])
            coords_h = coords_h @ matrix
            coords = coords_h[:, :3]  # / coords_h[:, 3]
        return coords

    async def do(self) -> dav.Dataset:
        logger.info("executing %s.do()", self.__class__.__name__)

        mesh = UsdGeom.Mesh(self.dataset)

        # Surface mesh requires both points and topology
        # Get points (required for surface_mesh)
        np_coords = np.asarray(mesh.GetPointsAttr().Get(self.timeCode)).astype(np.float32, copy=False).reshape(-1, 3)
        # FIXME: how to correctly deal with xform.
        np_coords = self._apply_xform(mesh, np_coords, self.timeCode)
        points = wp.array(np_coords, dtype=wp.vec3f, device=self.device, copy=False)
        del np_coords

        # Get topology (required for surface_mesh)
        np_face_vertex_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(self.timeCode)).astype(
            np.int32, copy=False
        )
        np_face_vertex_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(self.timeCode)).astype(
            np.int32, copy=False
        )
        face_vertex_counts = wp.array(np_face_vertex_counts, dtype=wp.int32, device=self.device, copy=False)
        face_vertex_indices = wp.array(np_face_vertex_indices, dtype=wp.int32, device=self.device, copy=False)

        # Create surface mesh handle
        return surface_mesh.create_dataset(
            points=points,
            face_vertex_indices=face_vertex_indices,
            face_vertex_counts=face_vertex_counts,
        )


class CaeMeshConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a prim with CaeMeshAPI to a DAV DataSet.

    This builds a DAV dataset using the surface_mesh data model, reading
    points and topology from the CAE mesh relationships.
    """

    async def do(self) -> dav.Dataset:
        logger.info("executing %s.do()", self.__class__.__name__)
        assert self.dataset.HasAPI(cae.MeshAPI)

        timeCode = self.timeCode

        points = await usd_utils.get_vecN_from_relationship(self.dataset, cae.Tokens.caeMeshPoints, 3, timeCode)
        points = wp.array(points, dtype=wp.vec3f, device=self.device, copy=False)

        face_vertex_indices = await usd_utils.get_array_from_relationship(
            self.dataset, cae.Tokens.caeMeshFaceVertexIndices, timeCode
        )
        face_vertex_counts = await usd_utils.get_array_from_relationship(
            self.dataset, cae.Tokens.caeMeshFaceVertexCounts, timeCode
        )

        return surface_mesh.create_dataset(
            points=points,
            face_vertex_indices=wp.array(face_vertex_indices, dtype=wp.int32, device=self.device, copy=False),
            face_vertex_counts=wp.array(face_vertex_counts, dtype=wp.int32, device=self.device, copy=False),
        )


class CaePointCloudConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE point cloud dataset to a DAV DataSet.

    This builds a zero-radius point cloud DAV dataset with dav.data_models.custom.point_cloud.DataModel.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.custom import point_cloud

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae.PointCloudAPI)

        points = await usd_utils.get_vecN_from_relationship(
            self.dataset, cae.Tokens.caePointCloudCoordinates, 3, timeCode
        )
        points = wp.array(points, dtype=wp.vec3f, device=self.device, copy=False)
        return point_cloud.create_dataset(points=points)


class CaeVtkImageDataConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE VTK Image Data dataset to a DAV DataSet.

    This builds a dense volume DAV dataset with dav.data_models.vtk.dense_volume.DataModel.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.vtk import image_data

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_vtk.ImageDataAPI)

        caeImageDataAPI = cae_vtk.ImageDataAPI(self.dataset)
        spacing = caeImageDataAPI.GetSpacingAttr().Get(timeCode)
        origin = caeImageDataAPI.GetOriginAttr().Get(timeCode)
        minExtent = caeImageDataAPI.GetMinExtentAttr().Get(timeCode)
        maxExtent = caeImageDataAPI.GetMaxExtentAttr().Get(timeCode)

        return image_data.create_dataset(
            origin=wp.vec3f(*origin),
            spacing=wp.vec3f(*spacing),
            extent_min=wp.vec3i(*minExtent),
            extent_max=wp.vec3i(*maxExtent),
            device=self.device,
        )


class CaeVtkStructuredGridConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE VTK Structured Grid dataset to a DAV DataSet.

    This builds a dense volume DAV dataset with dav.data_models.vtk.dense_volume.DataModel.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.vtk import structured_grid

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_vtk.StructuredGridAPI)

        points = await usd_utils.get_vecN_from_relationship(self.dataset, cae_vtk.Tokens.caeVtkPoints, 3, timeCode)

        caeStructuredGridAPI = cae_vtk.StructuredGridAPI(self.dataset)
        minExtent = caeStructuredGridAPI.GetMinExtentAttr().Get(timeCode)
        maxExtent = caeStructuredGridAPI.GetMaxExtentAttr().Get(timeCode)

        return structured_grid.create_dataset(
            points=wp.array(points, dtype=wp.vec3f, device=self.device, copy=False),
            extent_min=wp.vec3i(*minExtent),
            extent_max=wp.vec3i(*maxExtent),
        )


class CaeVtkPolyDataConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE VTK Poly Data dataset to a DAV DataSet.

    This builds a VTK PolyData DAV dataset with dav.data_models.vtk.polydata.DataModel.
    Supports verts (0D), lines (1D), and polys (2D) cell types.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.vtk import polydata

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_vtk.PolyDataAPI)

        caePolyDataAPI = cae_vtk.PolyDataAPI(self.dataset)

        points = await usd_utils.get_vecN_from_relationship(self.dataset, cae_vtk.Tokens.caeVtkPoints, 3, timeCode)

        # Read verts arrays (0D cells)
        verts_offsets = None
        verts_connectivity = None
        if caePolyDataAPI.GetVertsConnectivityOffsetsRel().GetTargets():
            verts_offsets = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkVertsConnectivityOffsets, timeCode
            )
            verts_connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkVertsConnectivityArray, timeCode
            )

        # Read lines arrays (1D cells)
        lines_offsets = None
        lines_connectivity = None
        if caePolyDataAPI.GetLinesConnectivityOffsetsRel().GetTargets():
            lines_offsets = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkLinesConnectivityOffsets, timeCode
            )
            lines_connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkLinesConnectivityArray, timeCode
            )

        # Read polys arrays (2D cells)
        polys_offsets = None
        polys_connectivity = None
        if caePolyDataAPI.GetPolysConnectivityOffsetsRel().GetTargets():
            polys_offsets = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkPolysConnectivityOffsets, timeCode
            )
            polys_connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkPolysConnectivityArray, timeCode
            )

        # Create polydata handle with available cell types
        return polydata.create_dataset(
            points=wp.array(points, dtype=wp.vec3f, device=self.device, copy=False),
            verts_offsets=(
                wp.array(verts_offsets, dtype=wp.int32, device=self.device, copy=False)
                if verts_offsets is not None
                else None
            ),
            verts_connectivity=(
                wp.array(verts_connectivity, dtype=wp.int32, device=self.device, copy=False)
                if verts_connectivity is not None
                else None
            ),
            lines_offsets=(
                wp.array(lines_offsets, dtype=wp.int32, device=self.device, copy=False)
                if lines_offsets is not None
                else None
            ),
            lines_connectivity=(
                wp.array(lines_connectivity, dtype=wp.int32, device=self.device, copy=False)
                if lines_connectivity is not None
                else None
            ),
            polys_offsets=(
                wp.array(polys_offsets, dtype=wp.int32, device=self.device, copy=False)
                if polys_offsets is not None
                else None
            ),
            polys_connectivity=(
                wp.array(polys_connectivity, dtype=wp.int32, device=self.device, copy=False)
                if polys_connectivity is not None
                else None
            ),
        )


class CaeVtkUnstructuredGridConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE VTK Unstructured Grid dataset to a DAV DataSet.

    This builds a VTK unstructured DAV dataset with dav.data_models.vtk.unstructured_grid.DataModel.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.vtk import unstructured_grid

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_vtk.UnstructuredGridAPI), "Dataset is not a CAE VTK Unstructured Grid."

        caeUnstructuredGridAPI = cae_vtk.UnstructuredGridAPI(self.dataset)

        points = await usd_utils.get_vecN_from_relationship(self.dataset, cae_vtk.Tokens.caeVtkPoints, 3, timeCode)
        cell_types = await usd_utils.get_array_from_relationship(self.dataset, cae_vtk.Tokens.caeVtkCellTypes, timeCode)
        cell_offsets = await usd_utils.get_array_from_relationship(
            self.dataset, cae_vtk.Tokens.caeVtkConnectivityOffsets, timeCode
        )
        cell_connectivity = await usd_utils.get_array_from_relationship(
            self.dataset, cae_vtk.Tokens.caeVtkConnectivityArray, timeCode
        )

        faces_offsets = await usd_utils.get_array_from_relationship(
            self.dataset, cae_vtk.Tokens.caeVtkPolyhedronFacesOffsets, timeCode, quiet=True
        )
        if faces_offsets is not None:
            faces_connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkPolyhedronFacesConnectivityArray, timeCode
            )
            face_locations_offsets = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkPolyhedronFaceLocationsOffsets, timeCode
            )
            face_locations_connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_vtk.Tokens.caeVtkPolyhedronFaceLocationsConnectivityArray, timeCode
            )
        else:
            faces_connectivity = None
            face_locations_offsets = None
            face_locations_connectivity = None

        return unstructured_grid.create_dataset(
            points=wp.array(points, dtype=wp.vec3f, device=self.device, copy=False),
            cell_types=wp.array(cell_types, dtype=wp.int32, device=self.device, copy=False),
            cell_offsets=wp.array(cell_offsets, dtype=wp.int32, device=self.device, copy=False),
            cell_connectivity=wp.array(cell_connectivity, dtype=wp.int32, device=self.device, copy=False),
            faces_offsets=(
                wp.array(faces_offsets, dtype=wp.int32, device=self.device, copy=False)
                if faces_offsets is not None
                else None
            ),
            faces_connectivity=(
                wp.array(faces_connectivity, dtype=wp.int32, device=self.device, copy=False)
                if faces_connectivity is not None
                else None
            ),
            face_locations_offsets=(
                wp.array(face_locations_offsets, dtype=wp.int32, device=self.device, copy=False)
                if face_locations_offsets is not None
                else None
            ),
            face_locations_connectivity=(
                wp.array(face_locations_connectivity, dtype=wp.int32, device=self.device, copy=False)
                if face_locations_connectivity is not None
                else None
            ),
        )


class CaeEnSightUnstructuredPartConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE EnSight Unstructured Part dataset to a DAV DataSet.

    This builds a EnSight unstructured DAV dataset with dav.data_models.ensight.unstructured.DataModel.
    """

    async def do(self) -> dav.Dataset:
        from dav.data_models.ensight_gold import unstructured_part

        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_ensight.UnstructuredPartAPI), "Dataset is not a CAE EnSight Unstructured Part."

        points = await usd_utils.get_vecN_from_relationship(
            self.dataset, cae_ensight.Tokens.caeEnsightPartCoordinates, 3, timeCode
        )
        points = wp.array(points, dtype=wp.vec3f, device=self.device, copy=False)

        piece_prims = usd_utils.get_target_prims(self.dataset, cae_ensight.Tokens.caeEnsightPartPieces)
        if len(piece_prims) == 0:
            raise ValueError("Missing EnSight unstructured part pieces.")

        handles = [await self._create_piece_handle(piece_prim, timeCode) for piece_prim in piece_prims]
        return unstructured_part.create_dataset(points=points, pieces=handles)

    async def _create_piece_handle(self, piece_prim, timeCode):
        from dav.data_models.ensight_gold import unstructured_part

        assert piece_prim.HasAPI(
            cae_ensight.UnstructuredPieceAPI
        ), "EnSight unstructured part piece is missing PieceAPI."

        piece_api = cae_ensight.UnstructuredPieceAPI(piece_prim)

        element_type = piece_api.GetElementTypeAttr().Get(timeCode)
        connectivity = await usd_utils.get_array_from_relationship(
            piece_prim, cae_ensight.Tokens.caeEnsightPieceConnectivity, timeCode
        )
        # element_node_counts is only required for nsided elements
        if element_type == cae_ensight.Tokens.nsided:
            assert piece_prim.HasAPI(cae_ensight.NSidedPieceAPI), "Nsided elements must have NSidedPieceAPI."
            element_node_counts = await usd_utils.get_array_from_relationship(
                piece_prim,
                cae_ensight.Tokens.caeEnsightPieceElementNodeCounts,
                timeCode,
            )
        else:
            element_node_counts = None

        if element_type == cae_ensight.Tokens.nfaced:
            assert piece_prim.HasAPI(cae_ensight.NFacedPieceAPI), "NFaced elements must have NFacedPieceAPI."
            element_face_counts = await usd_utils.get_array_from_relationship(
                piece_prim,
                cae_ensight.Tokens.caeEnsightPieceElementFaceCounts,
                timeCode,
            )
            face_node_counts = await usd_utils.get_array_from_relationship(
                piece_prim,
                cae_ensight.Tokens.caeEnsightPieceFaceNodeCounts,
                timeCode,
            )
        else:
            element_face_counts = None
            face_node_counts = None

        return unstructured_part.create_piece_handle(
            element_type=self._get_dav_element_type(element_type),
            connectivity=wp.array(connectivity, dtype=wp.int32, device=self.device, copy=False),
            element_node_counts=(
                wp.array(element_node_counts, dtype=wp.int32, device=self.device, copy=False)
                if element_node_counts is not None
                else None
            ),
            element_face_counts=(
                wp.array(element_face_counts, dtype=wp.int32, device=self.device, copy=False)
                if element_face_counts is not None
                else None
            ),
            face_node_counts=(
                wp.array(face_node_counts, dtype=wp.int32, device=self.device, copy=False)
                if face_node_counts is not None
                else None
            ),
        )

    def _get_dav_element_type(self, ensight_element_type):
        from dav.data_models.ensight_gold import ensight_shapes

        # Map EnSight element types to DAV
        _ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP = {
            cae_ensight.Tokens.point: ensight_shapes.EN_point,
            cae_ensight.Tokens.bar2: ensight_shapes.EN_bar2,
            cae_ensight.Tokens.bar3: ensight_shapes.EN_bar3,
            cae_ensight.Tokens.tria3: ensight_shapes.EN_tria3,
            cae_ensight.Tokens.tria6: ensight_shapes.EN_tria6,
            cae_ensight.Tokens.quad4: ensight_shapes.EN_quad4,
            cae_ensight.Tokens.quad8: ensight_shapes.EN_quad8,
            cae_ensight.Tokens.tetra4: ensight_shapes.EN_tetra4,
            cae_ensight.Tokens.tetra10: ensight_shapes.EN_tetra10,
            cae_ensight.Tokens.pyramid5: ensight_shapes.EN_pyramid5,
            cae_ensight.Tokens.pyramid13: ensight_shapes.EN_pyramid13,
            cae_ensight.Tokens.penta6: ensight_shapes.EN_penta6,
            cae_ensight.Tokens.penta15: ensight_shapes.EN_penta15,
            cae_ensight.Tokens.hexa8: ensight_shapes.EN_hexa8,
            cae_ensight.Tokens.hexa20: ensight_shapes.EN_hexa20,
            cae_ensight.Tokens.nsided: ensight_shapes.EN_nsided,
            cae_ensight.Tokens.nfaced: ensight_shapes.EN_nfaced,
        }
        if ensight_element_type not in _ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP:
            raise ValueError(f"Unsupported EnSight element type: {ensight_element_type}")
        return _ENSIGHT_TO_DAV_ELEMENT_TYPE_MAP[ensight_element_type]


class CaeOpenFoamPolyMeshConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE OpenFOAM PolyMesh dataset to a DAV DataSet.

    This builds an OpenFOAM polyMesh DAV dataset with dav.data_models.openfoam.polymesh.DataModel.
    """

    async def do(self) -> dav.Dataset:
        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_openfoam.PolyMeshAPI), "Dataset is not a CAE OpenFOAM PolyMesh."

        polyMeshAPI = cae_openfoam.PolyMeshAPI(self.dataset)

        # Get points from the mesh
        points = await usd_utils.get_vecN_from_relationship(
            self.dataset, cae_openfoam.Tokens.caeFoamPoints, 3, timeCode
        )

        # Get faces array (flat array of vertex indices)
        faces = await usd_utils.get_array_from_relationship(self.dataset, cae_openfoam.Tokens.caeFoamFaces, timeCode)

        # Get face offsets (size = num_faces + 1)
        face_offsets = await usd_utils.get_array_from_relationship(
            self.dataset, cae_openfoam.Tokens.caeFoamFacesOffsets, timeCode
        )

        # Get owner array (cell ID that owns each face)
        owner = await usd_utils.get_array_from_relationship(self.dataset, cae_openfoam.Tokens.caeFoamOwner, timeCode)

        # Get neighbour array (neighboring cell ID, -1 for boundary faces)
        neighbour = await usd_utils.get_array_from_relationship(
            self.dataset, cae_openfoam.Tokens.caeFoamNeighbour, timeCode
        )

        # Create polymesh handle
        return openfoam_polymesh.create_dataset(
            points=wp.array(points, dtype=wp.vec3f, device=self.device, copy=False),
            faces=wp.array(faces, dtype=wp.int32, device=self.device, copy=False),
            owner=wp.array(owner, dtype=wp.int32, device=self.device, copy=False),
            neighbour=wp.array(neighbour, dtype=wp.int32, device=self.device, copy=False),
            face_offsets=wp.array(face_offsets, dtype=wp.int32, device=self.device, copy=False),
        )


class CaeOpenFoamPolyBoundaryMeshConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a CAE OpenFOAM Boundary Mesh dataset to a DAV DataSet.

    This builds an OpenFOAM boundary mesh DAV dataset with dav.data_models.openfoam.boundary_mesh.DataModel.
    """

    async def do(self) -> dav.Dataset:
        logger.info("executing %s.do()", self.__class__.__name__)
        timeCode = self.timeCode
        assert self.dataset.HasAPI(cae_openfoam.PolyBoundaryMeshAPI), "Dataset is not a CAE OpenFOAM Boundary Mesh."

        boundaryMeshAPI = cae_openfoam.PolyBoundaryMeshAPI(self.dataset)

        # Get boundary-specific attributes
        start_face = boundaryMeshAPI.GetStartFaceAttr().Get(timeCode)
        n_faces = boundaryMeshAPI.GetNFacesAttr().Get(timeCode)

        # Get points from the parent mesh (boundary meshes inherit from parent dataset)
        # The boundary prim should have relationships to the parent mesh's field arrays
        points = await usd_utils.get_vecN_from_relationship(
            self.dataset, cae_openfoam.Tokens.caeFoamPoints, 3, timeCode
        )

        # Get faces array (flat array of vertex indices)
        faces = await usd_utils.get_array_from_relationship(self.dataset, cae_openfoam.Tokens.caeFoamFaces, timeCode)

        # Get face offsets (size = num_faces + 1)
        face_offsets = await usd_utils.get_array_from_relationship(
            self.dataset, cae_openfoam.Tokens.caeFoamFacesOffsets, timeCode
        )

        # Create boundary mesh handle
        return openfoam_boundary_mesh.create_dataset(
            points=wp.array(points, dtype=wp.vec3f, device=self.device, copy=False),
            faces=wp.array(faces, dtype=wp.int32, device=self.device, copy=False),
            face_offsets=wp.array(face_offsets, dtype=wp.int32, device=self.device, copy=False),
            start_face=start_face,
            n_faces=n_faces,
        )


class CaeSidsUnstructuredConvertToDAVDataSet(ConvertToDAVDataSet):
    """
    Convert a SIDS unstructured dataset to a DAV DataSet.
    """

    async def do(self) -> dav.Dataset:
        """
        Execute the command to convert a SIDS unstructured dataset to a DAV DataSet.

        Returns:
            A dav.Dataset object with SIDS unstructured data model
        """
        logger.info("executing %s.do()", self.__class__.__name__)

        if not self.dataset.HasAPI(cae_sids.UnstructuredAPI):
            raise usd_utils.QuietableException("Dataset (%s) does not support cae_sids.UnstructuredAPI!" % self.dataset)

        device = self.device
        timeCode = self.timeCode

        # Get mesh vertices and elements
        grid_coordinates = await usd_utils.get_vecN_from_relationship(
            self.dataset, cae_sids.Tokens.caeSidsGridCoordinates, 3, timeCode
        )
        grid_coordinates = wp.array(grid_coordinates, dtype=wp.vec3f, device=device, copy=False)

        element_type = sids_shapes.get_element_type_from_string(
            usd_utils.get_attribute(self.dataset, cae_sids.Tokens.caeSidsElementType).lower()
        )

        if element_type == sids_shapes.ET_MIXED:
            logger.error(
                "This CGNS dataset uses Mixed element types. Support for Mixed element types is not implemented at this point "
                "Please convert the dataset to use a single element type and try again. "
                "dataset: [%s]" % self.dataset.GetPath()
            )
            raise usd_utils.QuietableException("Mixed element types are not supported for dataset (%s)!" % self.dataset)

        if element_type == sids_shapes.ET_NFACE_n:
            dataset = await self.nface_n_to_dav_dataset(self.dataset, grid_coordinates)
        else:
            dataset = await self.standard_to_dav_dataset(self.dataset, grid_coordinates)
        return dataset

    async def standard_to_dav_dataset(self, prim, grid_coordinates) -> dav.DatasetLike:
        device = self.device
        timeCode = self.timeCode

        e_type = sids_shapes.get_element_type_from_string(
            usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementType).lower()
        )
        e_start = usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementRangeStart)
        e_end = usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementRangeEnd)

        # ensure both e_start and e_end are <= int32 max
        int32_max = np.iinfo(np.int32).max
        if e_start > int32_max or e_end > int32_max:
            raise ValueError(
                f"Element range start and end must be less than or equal to {int32_max}. "
                f"Got start: {e_start}, end: {e_end} for dataset: {prim.GetPath()}"
            )

        connectivity = await usd_utils.get_array_from_relationship(
            prim, cae_sids.Tokens.caeSidsElementConnectivity, timeCode
        )
        if connectivity.shape[0] > int32_max:
            raise ValueError(
                f"Element connectivity array size must be less than or equal to {int32_max}. "
                f"Got size: {connectivity.shape[0]} for dataset: {prim.GetPath()}"
            )
        connectivity = wp.array(connectivity, dtype=wp.int32, device=device, copy=False)

        e_start_offsets = await usd_utils.get_array_from_relationship(
            prim, cae_sids.Tokens.caeSidsElementStartOffset, timeCode, quiet=True
        )
        if e_start_offsets is not None:
            if e_start_offsets.shape[0] > int32_max:
                raise ValueError(
                    f"Element start offset array size must be less than or equal to {int32_max}. "
                    f"Got size: {e_start_offsets.shape[0]} for dataset: {prim.GetPath()}"
                )
            e_start_offsets = wp.array(e_start_offsets, dtype=wp.int32, device=device, copy=False)

        # Create DAV dataset
        return dav_sids_unstructured.create_dataset(
            grid_coords=grid_coordinates,
            element_type=e_type,
            element_range=wp.vec2i(e_start, e_end),
            element_connectivity=connectivity,
            element_start_offset=e_start_offsets,
        )

    async def nface_n_to_dav_dataset(self, nface_prim, grid_coordinates) -> dav.DatasetLike:
        # process all ngon_n blocks
        ngon_prims = usd_utils.get_target_prims(nface_prim, cae_sids.Tokens.caeSidsNgons)
        ngon_dav_handles = []
        assert len(ngon_prims) > 0, "No ngon blocks found for nface_n dataset."
        for ngon_prim in ngon_prims:
            ngon_dav_dataset = await self.standard_to_dav_dataset(ngon_prim, grid_coordinates)
            ngon_dav_handles.append(ngon_dav_dataset.handle)

        nface_n_element_dataset = await self.standard_to_dav_dataset(nface_prim, grid_coordinates)
        return dav_sids_nface_n.create_dataset(nface_n_element_dataset.handle, ngon_dav_handles)


class OmniCaeDataSetGetField(GetField):
    """
    Retrieve a named field from a CaeDataSet prim as a DAV field.
    """

    async def do(self) -> dav.Field:
        logger.info("executing %s.do()", self.__class__.__name__)

        field_prims = [
            usd_utils.get_target_prim(self.dataset, f"field:{field_name}") for field_name in self.field_names
        ]
        assert len(field_prims) > 0, "No field names were specified"
        timeCode = usd_utils.snap_time_code_to_prims(field_prims, self.timeCode)

        if any(prim.HasAPI(cae.NanoVDBFieldArrayAPI) for prim in field_prims):
            if len(field_prims) > 1:
                raise ValueError(
                    "CaeNanoVDBFieldArrayAPI only supports a single field prim; " f"{len(field_prims)} were provided."
                )
            prim = field_prims[0]
            nanovdb_api = cae.NanoVDBFieldArrayAPI(prim)

            origin_val = nanovdb_api.GetOriginAttr().Get(timeCode)
            origin = wp.vec3i(*origin_val) if origin_val is not None else wp.vec3i(0, 0, 0)

            dims_val = prim.GetAttribute("cae:nanovdb_field_array:dims").Get(timeCode)
            if dims_val is None:
                raise ValueError("CaeNanoVDBFieldArrayAPI: 'dims' attribute is not set.")
            dims = wp.vec3i(*dims_val)

            association_token = cae.FieldArray(prim).GetFieldAssociationAttr().Get(timeCode)
            dav_association = self._get_dav_association(association_token, prim)

            array = (await usd_utils.get_arrays(field_prims, timeCode))[0]
            np_array = array.numpy().view(np.byte)
            wp_raw = wp.array(np_array, device=self.device)
            volume = wp.Volume(wp_raw)
            return dav.Field.from_volume(volume, dims=dims, association=dav_association, origin=origin)

        associations = [cae.FieldArray(fa).GetFieldAssociationAttr().Get(timeCode) for fa in field_prims]
        if not all(assoc == associations[0] for assoc in associations):
            raise ValueError("Multiple different associations found; only one is supported currently.")

        dav_association = self._get_dav_association(associations[0], field_prims[0])
        arrays = await usd_utils.get_arrays(field_prims, timeCode)
        if len(arrays) > 1 and any(self._get_num_components(array) != 1 for array in arrays):
            raise ValueError(
                "FieldSelectionAPI with multiple target prims only supports arrays with a single component each."
            )

        wp_arrays = [array_utils.as_warp_array(array).to(self.device) for array in arrays]
        return dav.Field.from_arrays(wp_arrays, dav_association)

    @staticmethod
    def _get_num_components(array) -> int:
        if array.ndim == 1:
            return 1
        if array.ndim == 2:
            return array.shape[1]
        raise ValueError(f"Array with ndim {array.ndim} is not supported for component selection.")

    @staticmethod
    def _get_dav_association(association_token, field_prim: Usd.Prim) -> dav.AssociationType:
        if association_token == cae.Tokens.vertex:
            return dav.AssociationType.VERTEX
        if association_token == cae.Tokens.cell:
            return dav.AssociationType.CELL
        raise ValueError(f"Unsupported field association '{association_token}' on {field_prim.GetPath()}")


class CaeSidsUnstructuredGetField(OmniCaeDataSetGetField):
    """
    Retrieve a field from a SIDS unstructured dataset.

    CGNS polyhedral zones store volume cell data on NFACE_n cells, while NGON_n
    sections expose those faces as unstructured cells.  For NGON_n datasets,
    remap cell-centered fields through the sibling NFACE_n block so each NGON
    face receives the value from a referencing volume cell.
    """

    async def do(self) -> dav.Field:
        field = await super().do()
        if field.association != dav.AssociationType.CELL:
            return field

        element_type = self._get_element_type(self.dataset)
        if element_type != sids_shapes.ET_NGON_n:
            return field

        nface_infos = self._get_nface_infos_for_ngon()
        if not nface_infos:
            return field

        ngon_start, ngon_end = self._get_element_range(self.dataset)
        ngon_count = ngon_end - ngon_start + 1
        total_nface_cells = sum(info["count"] for info in nface_infos)
        if field.size != total_nface_cells:
            if field.size == ngon_count:
                return field
            raise ValueError(
                f"Cell field size {field.size} on NGON_n dataset {self.dataset.GetPath()} does not match "
                f"the referenced NFACE_n cell count {total_nface_cells} or NGON_n face count {ngon_count}."
            )

        indices = await self._compute_ngon_cell_field_indices(nface_infos, ngon_start, ngon_end)
        return field.subset(indices)

    @staticmethod
    def _get_element_type(prim: Usd.Prim) -> int:
        return sids_shapes.get_element_type_from_string(
            usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementType).lower()
        )

    @staticmethod
    def _get_element_range(prim: Usd.Prim) -> tuple[int, int]:
        start = usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementRangeStart)
        end = usd_utils.get_attribute(prim, cae_sids.Tokens.caeSidsElementRangeEnd)
        return int(start), int(end)

    def _get_nface_infos_for_ngon(self) -> list[dict[str, object]]:
        parent = self.dataset.GetParent()
        if not parent:
            return []

        nface_prims = []
        for sibling in parent.GetChildren():
            if not sibling.HasAPI(cae_sids.UnstructuredAPI):
                continue
            if self._get_element_type(sibling) != sids_shapes.ET_NFACE_n:
                continue
            ngon_prims = usd_utils.get_target_prims(sibling, cae_sids.Tokens.caeSidsNgons, quiet=True)
            if any(ngon_prim.GetPath() == self.dataset.GetPath() for ngon_prim in ngon_prims):
                nface_prims.append(sibling)

        nface_prims.sort(key=lambda prim: self._get_element_range(prim)[0])
        infos = []
        field_offset = 0
        for prim in nface_prims:
            start, end = self._get_element_range(prim)
            count = end - start + 1
            infos.append({"prim": prim, "start": start, "end": end, "count": count, "field_offset": field_offset})
            field_offset += count
        return infos

    async def _compute_ngon_cell_field_indices(
        self, nface_infos: list[dict[str, object]], ngon_start: int, ngon_end: int
    ) -> wp.array:
        cache_key = self._get_ngon_cell_field_indices_cache_key(nface_infos, ngon_start, ngon_end)
        cached_indices = cache.get(cache_key, timeCode=self.timeCode)
        if cached_indices is not None:
            return cached_indices

        ngon_count = ngon_end - ngon_start + 1
        total_nface_cells = sum(info["count"] for info in nface_infos)
        int32_max = np.iinfo(np.int32).max
        if ngon_count > int32_max or total_nface_cells > int32_max:
            raise ValueError(
                f"NGON_n cell field subset for {self.dataset.GetPath()} exceeds the supported int32 index range."
            )

        missing_sentinel = int(total_nface_cells)
        indices = wp.full(shape=ngon_count, value=missing_sentinel, dtype=wp.int32, device=self.device)

        for info in nface_infos:
            nface_prim = info["prim"]
            nface_count = info["count"]
            field_offset = info["field_offset"]

            connectivity = await usd_utils.get_array_from_relationship(
                nface_prim, cae_sids.Tokens.caeSidsElementConnectivity, self.timeCode
            )
            connectivity = self._as_int32_warp_array(connectivity, "ElementConnectivity", nface_prim)

            offsets = await usd_utils.get_array_from_relationship(
                nface_prim, cae_sids.Tokens.caeSidsElementStartOffset, self.timeCode
            )
            offsets = self._as_int32_warp_array(offsets, "ElementStartOffset", nface_prim)

            wp.launch(
                _map_nface_to_ngon_cell_field_indices_kernel,
                dim=nface_count,
                inputs=[
                    connectivity,
                    offsets,
                    indices,
                    ngon_start,
                    ngon_count,
                    field_offset,
                ],
                device=self.device,
            )

        missing_info = wp.array(np.array([0, ngon_count], dtype=np.int32), dtype=wp.int32, device=self.device)
        wp.launch(
            _validate_ngon_cell_field_indices_kernel,
            dim=ngon_count,
            inputs=[indices, missing_sentinel, missing_info],
            device=self.device,
        )

        missing_count, first_missing_idx = missing_info.numpy().tolist()
        if missing_count:
            first_missing_id = ngon_start + int(first_missing_idx)
            raise ValueError(
                f"Could not find NFACE_n cell data indices for {missing_count} NGON_n faces on "
                f"{self.dataset.GetPath()}; first missing face id is {first_missing_id}."
            )
        cache.put_ex(
            cache_key,
            indices,
            prims=self._get_ngon_cell_field_indices_cache_watches(nface_infos),
            timeCode=self.timeCode,
        )
        return indices

    def _get_ngon_cell_field_indices_cache_key(
        self, nface_infos: list[dict[str, object]], ngon_start: int, ngon_end: int
    ) -> tuple:
        return (
            "dav:CaeSidsUnstructuredGetField:ngon_cell_field_indices",
            str(self.dataset.GetPath()),
            self._get_device_cache_key(),
            ngon_start,
            ngon_end,
            tuple(
                (
                    str(info["prim"].GetPath()),
                    info["start"],
                    info["end"],
                    info["field_offset"],
                )
                for info in nface_infos
            ),
        )

    def _get_device_cache_key(self) -> str:
        return self.device.alias if hasattr(self.device, "alias") else str(self.device)

    def _get_ngon_cell_field_indices_cache_watches(self, nface_infos: list[dict[str, object]]) -> list[cache.PrimWatch]:
        watches = [cache.PrimWatch(self.dataset, schemas=[cae_sids.UnstructuredAPI])]
        watches.extend(cache.PrimWatch(info["prim"], schemas=[cae_sids.UnstructuredAPI]) for info in nface_infos)
        parent = self.dataset.GetParent()
        if parent:
            watches.append(cache.PrimWatch(parent, on="resync"))
        return watches

    def _as_int32_warp_array(self, array, array_name: str, prim: Usd.Prim) -> wp.array:
        result = array_utils.as_warp_array(array).to(self.device)
        if result.ndim != 1:
            raise ValueError(f"NFACE_n {array_name} on {prim.GetPath()} must be one-dimensional.")
        if result.shape[0] > np.iinfo(np.int32).max:
            raise ValueError(
                f"NFACE_n {array_name} on {prim.GetPath()} has {result.shape[0]} entries; "
                "the CGNS NGON field subset path supports at most int32-sized arrays."
            )
        if result.dtype != wp.int32:
            result = wp.array(result, dtype=wp.int32, device=result.device)
        return result
