# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

from logging import getLogger

import numpy as np
from omni.cae.data import array_utils, cache, progress, usd_utils
from omni.cae.data.commands import ConvertToMesh, Mesh
from omni.cae.schema import cae
from omni.cae.schema import ensight as cae_ensight
from pxr import Gf

from . import ensight

logger = getLogger(__name__)


class CaeEnSightUnstructuredPieceConvertToMesh(ConvertToMesh):

    async def do(self):
        logger.info("executing %s.do()", self.__class__.__name__)

        nb_fields = len(self.fields)
        total_work_untils = nb_fields + 1
        work_fraction = 1.0 / total_work_untils

        key = {"type": "ensight-mesh", "dataset": str(self.dataset.GetPath())}

        with progress.ProgressContext("Reading mesh", scale=work_fraction):
            element_type = ensight.ElementType.get_enum(
                usd_utils.get_attribute(self.dataset, cae_ensight.Tokens.caeEnsightPieceElementType, self.timeCode)
            )
            coords = await usd_utils.get_vecN_from_relationship(
                self.dataset, cae_ensight.Tokens.caeEnsightPieceCoordinates, 3, self.timeCode
            )
            coords = array_utils.as_numpy_array(coords)

            connectivity = await usd_utils.get_array_from_relationship(
                self.dataset, cae_ensight.Tokens.caeEnsightPieceConnectivity, self.timeCode
            )
            connectivity = array_utils.as_numpy_array(connectivity) - 1  # Ensight indices are 1-based

            if element_type not in [ensight.ElementType.tria3, ensight.ElementType.quad4, ensight.ElementType.nsided]:
                raise usd_utils.QuietableException("Only linear 2D elements are currently supported")

            if element_type == ensight.ElementType.nsided:
                faceVertexCounts = await usd_utils.get_array_from_relationship(
                    self.dataset, cae_ensight.Tokens.caeEnsightPieceElementNodeCounts, self.timeCode
                )
            else:
                faceVertexCounts = None

        logger.debug("coords.shape: %s", coords.shape)
        logger.debug("connectivity.shape: %s", connectivity.shape)
        if faceVertexCounts is not None:
            logger.debug(
                "faceVertexCounts.shape: %s, min=%s, max=%s",
                faceVertexCounts.shape,
                np.min(faceVertexCounts),
                np.max(faceVertexCounts),
            )
        else:
            logger.debug("faceVertexCounts: None")

        fields = {}
        for idx, fname in enumerate(self.fields):
            with progress.ProgressContext(
                "Reading field %s" % fname, shift=(1 + idx) * work_fraction, scale=work_fraction
            ):
                f_prim = usd_utils.get_target_prim(self.dataset, f"field:{fname}")
                assoc = usd_utils.get_attribute(f_prim, cae.Tokens.fieldAssociation, self.timeCode)
                f_array = await usd_utils.get_array_from_relationship(self.dataset, f"field:{fname}", self.timeCode)
                f_array = array_utils.as_numpy_array(f_array)
                if assoc == cae.Tokens.vertex:
                    fields[fname] = f_array
                elif assoc == cae.Tokens.cell:
                    raise NotImplementedError("Cell field association not supported")
                else:
                    raise usd_utils.QuietableException(f"Unsupported field association: {assoc}")

        mesh = Mesh()
        mesh.points = coords
        mesh.faceVertexIndices = connectivity.ravel()
        mesh.fields = fields
        mesh.extents = Gf.Range3d(np.min(mesh.points, axis=0).tolist(), np.max(mesh.points, axis=0).tolist())
        mesh.faceVertexCounts = (
            faceVertexCounts if faceVertexCounts is not None else np.full(connectivity.shape[0], connectivity.shape[1])
        )
        return mesh
