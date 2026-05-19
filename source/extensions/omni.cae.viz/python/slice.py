# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Planar Slice operator for extracting and rendering 2D slices from datasets.

Each mesh prim handled by this operator is a UsdGeomMesh quad with
CaeVizPlanarSliceAPI applied.  On every execution the operator:

  1. Derives the slice plane from the prim's local transform: the +Y axis gives
     the plane normal and the origin gives a point on the plane.
  2. Intersects that infinite plane with the dataset's bounding box to find the
     tightest-fitting quad that covers the intersection polygon (see
     _compute_tight_fit_quad).
  3. Writes the four world-space corner points directly onto the mesh and sets
     the renderer's local matrix to identity (via UsdRt) so no additional
     transform is applied.
  4. Probes the 'colors' field on a regular pixel-centre grid spanning the quad
     using dav.operators.probe.
  5. Uploads the probed scalar values to an omni.ui.DynamicTextureProvider
     (R32_SFLOAT, one float per texel) and wires the resulting dynamic:// URL
     to the SliceTexture MDL shader's slice_texture input.
"""

import ctypes
import hashlib
from logging import getLogger

import numpy as np
import omni.ui as ui
import warp as wp
from dav.data_models.custom import point_cloud as dav_point_cloud
from dav.operators import probe as dav_probe
from omni.cae.data import cache, progress, usd_utils
from omni.cae.schema import viz as cae_viz
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from usdrt import Gf as GfRt
from usdrt import Rt
from usdrt import Sdf as SdfRt
from usdrt import UsdGeom as UsdGeomRt
from usdrt import Vt as VtRt

from . import utils as viz_utils
from .execution_context import ExecutionContext
from .operator import operator

# Configure ctypes signature for PyCapsule_New once at import time
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object
ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

logger = getLogger(__name__)


def _compute_plane(xform: "Gf.Matrix4d") -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the plane position and normal from an xform matrix.

    Parameters
    ----------
    xform : Gf.Matrix4d
        Local-to-parent transform of the plane prim (row-vector convention).

    Returns
    -------
    position : np.ndarray, shape (3,)
        World-space origin of the plane (row 3 of the matrix).
    normal : np.ndarray, shape (3,)
        Unit normal of the plane — the +Y axis transformed by the matrix's
        rotation (row 1, normalised to remove any scale).
    """
    mat = np.array(xform)  # (4,4) row-major
    position = mat[3, :3]
    normal = mat[1, :3]
    normal = normal / np.linalg.norm(normal)
    return position, normal


def _compute_tight_fit_quad(
    center: np.ndarray,
    normal: np.ndarray,
    bounds: "Gf.Range3d",
) -> np.ndarray:
    """
    Intersect an infinite plane with an AABB and return the corners of the
    tight-fitting quad that covers the intersection polygon.

    The plane is infinite; the AABB may yield a 3–6 vertex convex polygon.
    The quad is axis-aligned in the plane's own 2-D (u, v) frame and is the
    smallest rectangle in that frame that contains all intersection points.

    Parameters
    ----------
    center : np.ndarray, shape (3,)
        A point on the plane.
    normal : np.ndarray, shape (3,)
        Unit normal of the plane.
    bounds : Gf.Range3d
        Axis-aligned bounding box to intersect.

    Returns
    -------
    np.ndarray, shape (4, 3)
        World-space corners in UV order: P00, P10, P11, P01.
    """
    # Build a stable orthonormal basis in the plane from the normal.
    ref = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(ref, normal)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    # 8 corners of the AABB  (np.ndindex order: last index varies fastest)
    mn, mx = np.array(bounds.GetMin(), dtype=np.float64), np.array(bounds.GetMax(), dtype=np.float64)
    idx = np.array(list(np.ndindex(2, 2, 2)), dtype=np.float64)  # (8, 3)
    corners_8 = mn + idx * (mx - mn)

    # Signed distance from each corner to the plane
    d = (corners_8 - center) @ normal  # (8,)

    # 12 AABB edges (pairs of corner indices that differ in exactly one bit)
    edges = [
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # along axis-0 (X)
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),  # along axis-1 (Y)
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),  # along axis-2 (Z)
    ]

    pts = []
    for i, j in edges:
        di, dj = d[i], d[j]
        if di * dj < 0:  # edge straddles the plane
            t = di / (di - dj)
            pts.append(corners_8[i] + t * (corners_8[j] - corners_8[i]))
    for i, di in enumerate(d):
        if abs(di) < 1e-9:  # corner lies exactly on the plane
            pts.append(corners_8[i])

    if not pts:
        # Plane does not intersect the box — return a degenerate quad at center.
        return np.array([center, center, center, center], dtype=np.float32)

    pts = np.array(pts)
    rel = pts - center
    u_coords = rel @ u_axis
    v_coords = rel @ v_axis

    u_min, u_max = u_coords.min(), u_coords.max()
    v_min, v_max = v_coords.min(), v_coords.max()

    return np.array(
        [
            center + u_min * u_axis + v_min * v_axis,  # P00
            center + u_max * u_axis + v_min * v_axis,  # P10
            center + u_max * u_axis + v_max * v_axis,  # P11
            center + u_min * u_axis + v_max * v_axis,  # P01
        ],
        dtype=np.float32,
    )


def _create_probe_grid(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Build a regular grid of world-space probe positions over a planar quad.

    Parameters
    ----------
    corners : np.ndarray
        shape (4, 3) — P00 (UV=0,0), P10 (UV=1,0), P11 (UV=1,1), P01 (UV=0,1).
    width, height : int
        Texture resolution; probe positions are at pixel centres.

    Returns
    -------
    np.ndarray
        shape (height * width, 3) — row-major (row 0 = v near 0).
    """
    P00, P10, P11, P01 = corners[0], corners[1], corners[2], corners[3]

    u = (np.arange(width, dtype=np.float32) + 0.5) / width  # (W,)
    v = (np.arange(height, dtype=np.float32) + 0.5) / height  # (H,)
    uu, vv = np.meshgrid(u, v)  # (H, W)
    uu = uu.ravel()[:, None]  # (N, 1)
    vv = vv.ravel()[:, None]  # (N, 1)

    positions = (1 - vv) * ((1 - uu) * P00 + uu * P10) + vv * ((1 - uu) * P01 + uu * P11)
    return positions.astype(np.float32)


def _save_debug_png(values: np.ndarray, width: int, height: int, path: str):
    """Save probed scalar values as a grayscale PNG for debugging."""
    try:
        from PIL import Image

        img_data = values.reshape(height, width).astype(np.float32)
        v_min, v_max = img_data.min(), img_data.max()
        if v_max > v_min:
            img_data = (img_data - v_min) / (v_max - v_min)
        img_data = (img_data * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_data, mode="L").save(path)
        logger.info("Saved debug slice PNG to %s (range %.4f – %.4f)", path, v_min, v_max)
    except Exception as e:
        logger.warning("Could not save debug PNG to %s: %s", path, e)


# Maximum number of simultaneous planes (matches the 3 texture slots in the MDL shader).
_NUM_PLANES = 3

# Maps mode token → list of (slot_index, axis) pairs for the active planes.
# slot_index selects the texture slot (tex_idx primvar: 0/1/2).
# axis "" means free plane — normal from the prim's +Y axis; "x"/"y"/"z" are cardinal normals.
# y always uses slot 1 and z always uses slot 2 for consistency across multi-plane modes.
_MODE_PLANES: dict[str, list[tuple[int, str]]] = {
    "free": [(0, "")],
    "x": [(0, "x")],
    "y": [(1, "y")],
    "z": [(2, "z")],
    "xy": [(0, "x"), (1, "y")],
    "xz": [(0, "x"), (2, "z")],
    "yz": [(1, "y"), (2, "z")],
    "xyz": [(0, "x"), (1, "y"), (2, "z")],
}

_AXIS_NORMALS: dict[str, np.ndarray] = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _plane_center_and_normal(axis: str, local_mat) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, normal) for the given axis label and local transform.

    For the free plane (axis="") the normal comes from the prim's +Y axis.
    For axis-aligned planes the normal is a unit cardinal vector and the center
    is the prim's translation, so all planes in a multi-plane set share one point.
    """
    if not axis:
        return _compute_plane(local_mat)
    center = np.array(local_mat)[3, :3]
    return center, _AXIS_NORMALS[axis]


def _plane_tex_name(prim_path: str, slot: int) -> str:
    """Unique DynamicTextureProvider base name for a texture slot (prim + slot index)."""
    h = hashlib.sha512(prim_path.encode()).hexdigest()[:8]
    return f"cae_slice_{h}_{slot}"


def _prepare_quad_rt_prim(prim: Usd.Prim, prim_rt, slot: int, active: bool):
    """Get or create the RT-only mesh prim for texture slot *slot* (0/1/2).

    Prims live at /CaePlanarSlice/{hash}_{slot} — a flat stage root with no
    inherited transform, so corners can be written directly in world space.
    All three prims share the single Materials/SliceTexture material; the constant
    primvar tex_idx tells the shader which texture slot to sample.  Visibility is
    toggled each call so only planes relevant to the current mode are rendered.
    """
    rt_stage = prim_rt.GetStage()

    root_path = SdfRt.Path("/CaePlanarSlice")
    if not rt_stage.GetPrimAtPath(root_path):
        rt_stage.DefinePrim(root_path, "Scope")

    h = hashlib.sha512(prim.GetPath().pathString.encode()).hexdigest()[:8]
    quad_path = SdfRt.Path(f"/CaePlanarSlice/slice_{h}_{slot}")
    quad_prim = rt_stage.GetPrimAtPath(quad_path)

    if not quad_prim:
        quad_prim = rt_stage.DefinePrim(quad_path, "Mesh")

        # Workaround for a USDRT bug: hierarchy matrix attrs must be explicitly created.
        rtboundable = Rt.Boundable(quad_prim)
        rtboundable.CreateFabricHierarchyLocalMatrixAttr()
        rtboundable.CreateFabricHierarchyWorldMatrixAttr()

        # RT-only prims need explicit visibility and purpose attributes.
        quad_prim.CreateAttribute("purpose", SdfRt.ValueTypeNames.Token, False).Set(UsdGeomRt.Tokens.default_)
        mesh_rt = UsdGeomRt.Mesh(quad_prim)
        mesh_rt.CreateVisibilityAttr().Set(UsdGeomRt.Tokens.inherited)

        # add default points
        mesh_rt.CreatePointsAttr().Set([(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)])

        # Single quad face — topology never changes.
        mesh_rt.CreateFaceVertexCountsAttr().Set(VtRt.IntArray(np.array([[4]], dtype=np.intc)))
        mesh_rt.CreateFaceVertexIndicesAttr().Set(VtRt.IntArray(np.array([[0], [1], [2], [3]], dtype=np.intc)))

        # Texture coordinates (static — UV layout matches corner order).
        uv_coords = VtRt.Vec2fArray([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        UsdGeomRt.PrimvarsAPI(quad_prim).CreatePrimvar(
            "st", SdfRt.ValueTypeNames.TexCoord2fArray, UsdGeomRt.Tokens.vertex
        ).Set(uv_coords)

        # Constant tex_idx primvar — tells the shader which texture slot to read.
        UsdGeomRt.PrimvarsAPI(quad_prim).CreatePrimvar(
            "tex_idx", SdfRt.ValueTypeNames.Int, UsdGeomRt.Tokens.constant
        ).Set(slot)

        # All planes share the single SliceTexture material.
        mat_usd_path = prim.GetPath().AppendChild("Materials").AppendChild("SliceTexture")
        quad_prim.CreateRelationship("material:binding").SetTargets([SdfRt.Path(str(mat_usd_path))])

    # Toggle visibility every call — only active planes should render.
    quad_prim.CreateAttribute("_worldVisibility", SdfRt.ValueTypeNames.Bool).Set(active)
    quad_prim.GetAttribute("visibility").Set(UsdGeomRt.Tokens.inherited if active else UsdGeomRt.Tokens.invisible)
    return quad_prim


def _update_quad_rt_prim(quad_prim, corners: np.ndarray) -> None:
    """Write world-space corner points and bounding box to the RT mesh.

    corners is shape (4, 3) float32 in world space.
    """
    viz_utils.set_array_attribute(UsdGeomRt.Mesh(quad_prim).CreatePointsAttr(), corners)
    min_pt = corners.min(axis=0).astype(np.float64)
    max_pt = corners.max(axis=0).astype(np.float64)
    Rt.Boundable(quad_prim).CreateWorldExtentAttr().Set(GfRt.Range3d(GfRt.Vec3d(*min_pt), GfRt.Vec3d(*max_pt)))


@operator()
class PlanarSlice:
    """
    Operator for generating a texture-mapped planar slice from a dataset.

    Each prim handled by this operator is a UsdGeomMesh quad with
    CaeVizPlanarSliceAPI applied. The operator:
      1. Validates the mesh is a planar quad.
      2. Fetches the source dataset (with the 'colors' field) via DAV.
      3. Builds a regular grid of probe positions over the quad.
      4. Probes the 'colors' field at those positions using dav.operators.probe.
      5. Saves a debug PNG (to be replaced by a dynamic texture later).
    """

    prim_type: str = "Mesh"
    api_schemas: set[str] = {
        "CaeVizOperatorAPI",
        "CaeVizPlanarSliceAPI",
        "CaeVizDatasetSelectionAPI:source",
        "CaeVizDatasetTransformingAPI:self",
        "CaeVizFieldSelectionAPI:colors",
    }

    optional_api_schemas: set[str] = {
        "CaeVizRescaleRangeAPI",
        "CaeVizDatasetVoxelizationAPI:source",
    }

    async def exec(self, prim: Usd.Prim, device: str, context: ExecutionContext):
        # --- 1. Read texture resolution and mode ---
        planar_slice_api = cae_viz.PlanarSliceAPI(prim)
        tex_res = planar_slice_api.GetTextureResolutionAttr().Get()
        width, height = int(tex_res[0]), int(tex_res[1])
        mode = planar_slice_api.GetModeAttr().Get() or "free"
        active_planes = _MODE_PLANES.get(mode, _MODE_PLANES["free"])  # [(slot, axis), ...]
        active_slots = {slot for slot, _ in active_planes}
        logger.debug(
            "PlanarSlice.exec() prim=%s resolution=(%d,%d) mode=%s active_slots=%s",
            prim.GetPath(),
            width,
            height,
            mode,
            active_slots,
        )

        # Pre-create all RT quad prims (invisible) before the data fetch so that the
        # renderer discovers them in Fabric early.  Newly created RT prims are not
        # rendered until the renderer has seen them for at least one cycle; pre-creating
        # them here ensures that by the time a successful exec sets slot 0 visible, the
        # renderer already knows about the prims and displays them immediately.
        prim_rt = usd_utils.get_prim_rt(prim)
        prim_path_str = prim.GetPath().pathString
        for slot in range(_NUM_PLANES):
            _prepare_quad_rt_prim(prim, prim_rt, slot, False)

        # --- 2. Fetch source dataset ---
        source_dataset = await viz_utils.get_input_dataset(
            prim,
            "source",
            timeCode=context.timecode,
            device=device,
            required_fields={"colors"},
        )
        if not source_dataset.has_field("colors"):
            logger.warning("PlanarSlice: 'colors' field not found in dataset for %s — skipping", prim.GetPath())
            return

        # --- 3. Process rescale range APIs ---
        viz_utils.process_rescale_range_apis(prim, source_dataset)

        # --- 4. Shared setup ---
        raw = source_dataset.get_bounds()
        bounds = Gf.Range3d(Gf.Vec3d(*raw[0]), Gf.Vec3d(*raw[1]))
        xform_cache = UsdGeom.XformCache(context.timecode)
        local_mat, _ = xform_cache.GetLocalTransformation(prim)

        # Hide the operator prim's own geometry in the RT layer.
        mesh_rt = UsdGeomRt.Mesh(prim_rt)
        mesh_rt.CreatePointsAttr().Set([])
        mesh_rt.CreateFaceVertexCountsAttr().Set([])
        mesh_rt.CreateFaceVertexIndicesAttr().Set([])
        mesh_rt.CreateVisibilityAttr().Set(UsdGeomRt.Tokens.invisible)

        # --- 5. Compute corners for active planes ---
        slot_corners: dict[int, np.ndarray] = {}
        for slot, axis in active_planes:
            center, normal = _plane_center_and_normal(axis, local_mat)
            slot_corners[slot] = _compute_tight_fit_quad(center, normal, bounds).astype(np.float32, copy=False)

        # --- 6. Create/update all _NUM_PLANES RT quad prims; toggle visibility per mode ---
        # If the operator prim has a parent with a world transform, propagate that to the
        # quad prims' FabricHierarchyWorldMatrixAttr so the renderer applies it to the
        # (parent-local) corner points.
        parent = prim.GetParent()
        parent_world_gfrt = None
        if parent and parent.IsValid() and not parent.IsPseudoRoot():
            m = np.array(xform_cache.GetLocalToWorldTransform(parent))
            parent_world_gfrt = GfRt.Matrix4d(*m.flatten().tolist())

        for slot in range(_NUM_PLANES):
            active = slot in active_slots
            quad_prim = _prepare_quad_rt_prim(prim, prim_rt, slot, active)
            if parent_world_gfrt is not None:
                Rt.Boundable(quad_prim).CreateFabricHierarchyWorldMatrixAttr().Set(parent_world_gfrt)
            if active:
                _update_quad_rt_prim(quad_prim, slot_corners[slot])

        # --- 6b. Register guard (deletion cleanup) and visibility listener (once per prim) ---
        h = hashlib.sha512(prim_path_str.encode()).hexdigest()[:8]
        quad_paths = [SdfRt.Path(f"/CaePlanarSlice/slice_{h}_{slot}") for slot in range(_NUM_PLANES)]
        viz_utils.RtSubPrimGuard.register(prim, prim_rt.GetStage(), quad_paths)

        # --- 7. Ensure all texture providers are registered (all slots, both textures) ---
        prim_watch = cache.PrimWatch(prim, on="delete")
        tex_providers: dict[int, ui.DynamicTextureProvider] = {}
        mask_providers: dict[int, ui.DynamicTextureProvider] = {}
        for slot in range(_NUM_PLANES):
            tex_name = _plane_tex_name(prim_path_str, slot)

            key = f"[viz:slice_texture]::{prim_path_str}:{slot}"
            p = cache.get(key)
            if p is None:
                p = ui.DynamicTextureProvider(tex_name)
                cache.put_ex(key, p, prims=[prim_watch], force=True)
            tex_providers[slot] = p

            mask_key = f"[viz:slice_mask_texture]::{prim_path_str}:{slot}"
            mp = cache.get(mask_key)
            if mp is None:
                mp = ui.DynamicTextureProvider(tex_name + "_mask")
                cache.put_ex(mask_key, mp, prims=[prim_watch], force=True)
            mask_providers[slot] = mp

        # --- 8. Wire all texture slots into the single shared shader ---
        material_prim = prim.GetChild("Materials").GetChild("SliceTexture")
        if material_prim:
            shader = UsdShade.Material(material_prim).ComputeSurfaceSource("mdl")[0]
            if shader:
                for slot in range(_NUM_PLANES):
                    tex_name = _plane_tex_name(prim_path_str, slot)
                    shader.CreateInput(f"slice_texture_{slot}", Sdf.ValueTypeNames.Asset).Set(f"dynamic://{tex_name}")
                    shader.CreateInput(f"mask_texture_{slot}", Sdf.ValueTypeNames.Asset).Set(
                        f"dynamic://{tex_name}_mask"
                    )
        else:
            logger.warning("PlanarSlice: no SliceTexture material on %s", prim.GetPath())

        # --- 9. Probe and upload textures for active slots only ---
        for slot, axis in active_planes:
            corners = slot_corners[slot]
            label = f"planar_slice[{axis or 'free'}]"

            with progress.ProgressContext(f"Executing DAV [{label}: build probe grid]"):
                probe_positions = _create_probe_grid(corners, width, height)
                probe_positions_wp = wp.array(probe_positions, dtype=wp.vec3f, device=device)
                positions_dataset = dav_point_cloud.create_dataset(probe_positions_wp)

            with progress.ProgressContext(f"Executing DAV [{label}: probe]"):
                probed_dataset = dav_probe.compute(
                    source_dataset, "colors", positions_dataset, output_mask_field_name="cae_mask"
                )

            values_np = probed_dataset.get_field("probed_values").to_array().numpy()
            if values_np.ndim == 2:  # vector field → magnitude
                values_np = np.linalg.norm(values_np, axis=-1)

            # mask is wp.uint32 (0=out-of-domain, 1=valid); cast to float32 for R32_SFLOAT upload.
            mask_np = probed_dataset.get_field("cae_mask").to_array().numpy().astype(np.float32)
            if source_dataset.has_field("cae_mask"):
                with progress.ProgressContext(f"Executing DAV [{label}: probe input mask]"):
                    input_mask_dataset = dav_probe.compute(source_dataset, "cae_mask", positions_dataset)
                input_mask_np = input_mask_dataset.get_field("probed_values").to_array().numpy()
                mask_np = np.where((mask_np > 0) & (input_mask_np > 0), 1.0, 0.0).astype(np.float32)

            # Upload via CPU path — avoids GPU device mismatch between warp/CUDA and the renderer.
            # set_raw_bytes_data takes a PyCapsule (pybind11 const void*), not a bare integer.
            tex_data = np.ascontiguousarray(values_np.reshape(height, width)[::-1], dtype=np.float32)
            tex_providers[slot].set_raw_bytes_data(
                ctypes.pythonapi.PyCapsule_New(tex_data.ctypes.data, None, None),
                [width, height],
                ui.TextureFormat.R32_SFLOAT,
            )

            mask_tex_data = np.ascontiguousarray(mask_np.reshape(height, width)[::-1])
            mask_providers[slot].set_raw_bytes_data(
                ctypes.pythonapi.PyCapsule_New(mask_tex_data.ctypes.data, None, None),
                [width, height],
                ui.TextureFormat.R32_SFLOAT,
            )
