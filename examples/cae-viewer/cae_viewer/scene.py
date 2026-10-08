# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Native scene queries and a transient, non-destructive viewer render rig."""

from __future__ import annotations

import numpy as np


def read_attribute(stage, path, name, ordinal):
    """Read owned values through public OVStage APIs, releasing native groups."""
    import ovstage

    with ovstage.PathDictionary(stage) as paths:
        plist = paths.create_path_list_from_strings([path])
        query = stage.query_from_path_list(plist)
        try:
            with stage.read_attributes(query, [paths.intern_token(name)], ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete or not group.tensor_count:
                            continue
                        values = group.array(0).copy()
                        if group.raw.semantic == ovstage.AttributeSemantic.TOKEN_ID or name in {
                            "usd-schemas",
                            "usd-prim-type",
                        }:
                            tokens = tuple(paths.token_to_string(int(v)) for v in values.reshape(-1))
                            return tokens if group.is_array else (tokens[0] if tokens else None)
                        return values if group.is_array or values.size != 1 else values.item()
                    finally:
                        stage.release_group(group)
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(plist)
    return None


def prim_paths(stage, ordinal, *, schema=None):
    """List composed prim paths, optionally selected by an applied API schema."""
    import ovstage

    predicates = [] if schema is None else [ovstage.Predicate("usd-schemas", ovstage.FilterOp.CONTAINS, [schema])]
    result = set()
    with ovstage.PathDictionary(stage) as paths, stage.query(filter=ovstage.Filter(predicates)) as query:
        query.wait()
        attribute = "usd-schemas" if schema else "usd-prim-type"
        with stage.read_attributes(
            query, [paths.intern_token(attribute)], ovstage.OrdinalRange.latest(ordinal)
        ) as read:
            read.wait()
            for group in read.groups():
                try:
                    if not group.is_delete:
                        ids = paths.get_paths(group.prim_list)
                        result.update(paths.path_to_string(ids[group.prim_index(i)]) for i in range(group.prim_count))
                finally:
                    stage.release_group(group)
    return sorted(result)


def hidden_ancestor(stage, path, ordinal):
    """Identify an invisible parent without changing inherited presentation."""
    parent = path.rsplit("/", 1)[0]
    while parent:
        if read_attribute(stage, parent, "visibility", ordinal) == "invisible":
            return parent
        parent = parent.rsplit("/", 1)[0]
    return None


def compose_viewer_scene(filename, width, height):
    """Build an anonymous layer over the file; return text, rig path, up axis, and bound camera.

    OpenUSD is used only for initial composition/metadata, never as a live edit
    authority. Absolute sublayer identifiers preserve the source's relative
    asset resolution. No source layer is modified or saved. The bound camera is
    the pose Kit saved in the root layer's ``cameraSettings`` metadata, or None.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom

    import ovcae.readers  # noqa: F401  # registers the CAE file formats before the first stage
    from ovcae.schemas import register_usd_plugin

    # Even an empty startup scene performs the first pxr schema read. Register
    # now so opening a CAE file later in the same process still sees its APIs.
    register_usd_plugin()

    layer = Sdf.Layer.CreateAnonymous("viewer.usda")
    if filename is not None:
        layer.subLayerPaths = [str(filename)]
    source = Usd.Stage.Open(layer)
    # Stage metadata (unlike prim opinions) does not compose from sublayers.
    bound_camera = None
    if filename is not None:
        original = Usd.Stage.Open(str(filename))
        UsdGeom.SetStageUpAxis(source, UsdGeom.GetStageUpAxis(original))
        UsdGeom.SetStageMetersPerUnit(source, UsdGeom.GetStageMetersPerUnit(original))
        bound_camera = read_bound_camera(original, width, height)
    root = "/_CaeViewer"
    while source.GetPrimAtPath(root):
        root += "_"
    up_axis = str(UsdGeom.GetStageUpAxis(source))
    has_lights = any(p.GetTypeName().endswith("Light") for p in source.Traverse())
    source.DefinePrim(root, "Scope")
    camera = UsdGeom.Camera.Define(source, root + "/Camera")
    camera.CreateClippingRangeAttr((0.01, 1000000))
    camera.AddTranslateOp().Set((0, 0, 12))
    product = source.DefinePrim(root + "/Render", "RenderProduct")
    product.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(width, height))
    product.CreateRelationship("camera").SetTargets([camera.GetPath()])
    product.CreateRelationship("orderedVars").SetTargets([root + "/Render/LdrColor"])
    var = source.DefinePrim(root + "/Render/LdrColor", "RenderVar")
    var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
    var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque)
    if not has_lights:
        light = source.DefinePrim(root + "/Light", "DomeLight")
        light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000)
    return layer.ExportToString(), root, up_axis, bound_camera


# Kit's implicit viewport cameras live only in its session layer; their saved
# poses are keyed by name under ``cameraSettings``. Only the perspective one can
# drive this viewer, which has no orthographic projection.
KIT_PERSPECTIVE_CAMERA = "/OmniverseKit_Persp"
# Kit's implicit perspective camera: focalLength 18.147562, horizontalAperture 20.955.
KIT_PERSPECTIVE_HORIZONTAL_FOV = 60.0


def read_bound_camera(stage, width, height):
    """Return the pose Kit saved for the stage's bound camera, or None.

    Kit writes ``customLayerData.cameraSettings`` on the root layer with the
    bound camera path and, per implicit camera, its position/target. A bound
    camera that is a real ``UsdGeom.Camera`` prim is read from its composed
    transform instead, so the file renders from the view it was saved with.
    """
    import math

    from pxr import Gf, Usd, UsdGeom

    settings = (stage.GetRootLayer().customLayerData or {}).get("cameraSettings")
    if not isinstance(settings, dict):
        return None
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    bound = settings.get("boundCamera")
    if not isinstance(bound, str) or not bound:
        bound = KIT_PERSPECTIVE_CAMERA
    prim = stage.GetPrimAtPath(bound) if bound.startswith("/") else None
    camera = UsdGeom.Camera(prim) if prim and prim.IsValid() else None
    if camera:
        time = Usd.TimeCode(0.0)
        matrix = camera.ComputeLocalToWorldTransform(time)
        eye = Gf.Vec3d(matrix.ExtractTranslation())
        forward = matrix.TransformDir(Gf.Vec3d(0, 0, -1))
        if forward.GetLength() <= 0:
            return None
        forward = forward.GetNormalized()
        distance = camera.GetFocusDistanceAttr().Get(time) or 0.0
        if not distance > 0:
            saved = settings.get(bound.rsplit("/", 1)[-1])
            distance = saved.get("radius", 0.0) if isinstance(saved, dict) else 0.0
        if not distance > 0:
            distance = 10.0
        target = eye + forward * float(distance)
        focal = camera.GetFocalLengthAttr().Get(time) or 50.0
        aperture = camera.GetVerticalApertureAttr().Get(time) or 15.2908
        fov = math.degrees(2 * math.atan(aperture / (2 * focal))) if focal > 0 else 50.0
        pose = dict(eye=tuple(eye), target=tuple(target), fov_degrees=fov, prim_path=bound)
    else:
        if bound != KIT_PERSPECTIVE_CAMERA:
            return None
        saved = settings.get("Perspective")
        if not isinstance(saved, dict) or "position" not in saved or "target" not in saved:
            return None
        half = math.tan(math.radians(KIT_PERSPECTIVE_HORIZONTAL_FOV / 2)) * height / max(width, 1)
        pose = dict(
            eye=tuple(float(v) for v in saved["position"]),
            target=tuple(float(v) for v in saved["target"]),
            fov_degrees=math.degrees(2 * math.atan(half)),
            prim_path="",
        )
    values = (*pose["eye"], *pose["target"], pose["fov_degrees"])
    if len(values) != 7 or not all(math.isfinite(v) for v in values):
        return None
    if math.dist(pose["eye"], pose["target"]) <= 0 or not 0 < pose["fov_degrees"] < 180:
        return None
    return dict(pose, up_axis=up_axis)


def scene_bounds(stage, ordinal):
    """Frame visible point-based geometry, including freshly generated products."""
    bounds = world_aabb(stage, ordinal)
    if bounds is None:
        return np.zeros(3), 5.0
    low, high = bounds
    return (low + high) / 2, max(float(np.linalg.norm(high - low) / 2), 0.01)


def world_aabb(stage, ordinal, paths=None):
    """World bounds of visible point-based prims, or None when nothing contributes.

    ``paths`` restricts the result to those prims and their descendants, so a
    selected group frames its subtree. Invisible prims never contribute.
    """
    roots = None if paths is None else tuple(paths)
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    for path in prim_paths(stage, ordinal):
        if roots is not None and not any(path == root or path.startswith(root + "/") for root in roots):
            continue
        if read_attribute(stage, path, "visibility", ordinal) == "invisible" or hidden_ancestor(stage, path, ordinal):
            continue
        points = read_attribute(stage, path, "points", ordinal)
        if points is None or np.asarray(points).size == 0:
            continue
        points = np.asarray(points).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        # Transform the eight local bounding-box corners instead of all points.
        lo, hi = points.min(axis=0), points.max(axis=0)
        corners = np.array([[x, y, z, 1] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        parent = path
        while parent:
            matrix = read_attribute(stage, parent, "omni:xform", ordinal)
            if matrix is not None:
                corners = corners @ np.asarray(matrix).reshape(4, 4)
            parent = parent.rsplit("/", 1)[0]
        low = np.minimum(low, corners[:, :3].min(axis=0))
        high = np.maximum(high, corners[:, :3].max(axis=0))
    if not np.isfinite(low).all():
        return None
    return low, high
