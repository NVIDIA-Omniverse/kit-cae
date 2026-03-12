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
from omni.cae.data import array_utils, progress, usd_utils
from omni.cae.schema import cae
from omni.cae.schema import ensight as cae_ensight
from omni.cae.schema import openfoam as cae_openfoam
from omni.cae.schema import sids as cae_sids
from omni.cae.schema import vtk as cae_vtk
from pxr import Gf, Usd, UsdGeom

from .command_types import ConvertToDAVDataSet

logger = getLogger(__name__)

# dav.config.enable_timing = True
# dav.config.enable_nvtx = True


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
