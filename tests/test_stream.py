# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import math

import numpy as np
import pytest

from ovcae.stream import WEBRTC_CLIENT_LIBRARY_URL, OrbitCamera, webrtc_client_html


def _looks_at(matrix, eye, target):
    np.testing.assert_allclose(matrix[3, :3], eye, atol=1e-9)
    forward = -matrix[2, :3]
    direction = np.asarray(target, dtype=float) - eye
    np.testing.assert_allclose(forward, direction / np.linalg.norm(direction), atol=1e-9)
    # Rows are an orthonormal basis; row vectors, camera looks down its local -Z.
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("up_axis", ["Z", "Y"])
def test_orbit_camera_starts_at_the_given_pose(up_axis):
    eye, target = (4.0, -3.0, 2.0), (1.0, 0.5, 0.25)
    camera = OrbitCamera(eye, target, up_axis=up_axis)
    matrix = camera.matrix()
    _looks_at(matrix, np.asarray(eye), target)
    up = {"Z": (0.0, 0.0, 1.0), "Y": (0.0, 1.0, 0.0)}[up_axis]
    assert np.dot(matrix[1, :3], up) > 0  # the camera's up leans along the stage's up


def test_orbit_camera_yaw_keeps_distance_and_zoom_changes_it():
    camera = OrbitCamera((4.0, 0.0, 1.0), (0.0, 0.0, 1.0))
    start = camera.matrix()
    camera.yaw = math.pi / 2
    turned = camera.matrix()
    assert np.linalg.norm(turned[3, :3] - camera.target) == pytest.approx(np.linalg.norm(start[3, :3] - camera.target))
    np.testing.assert_allclose(turned[3, :3], (0.0, 4.0, 1.0), atol=1e-9)
    _looks_at(turned, turned[3, :3], camera.target)
    camera.zoom = -2.0
    assert np.linalg.norm(camera.matrix()[3, :3] - camera.target) == pytest.approx(2.0)
    camera.zoom = -100.0  # never passes through the target
    assert np.linalg.norm(camera.matrix()[3, :3] - camera.target) == pytest.approx(0.5)


def test_orbit_camera_rejects_other_up_axes():
    with pytest.raises(ValueError, match="up_axis"):
        OrbitCamera((1, 0, 0), (0, 0, 0), up_axis="X")


def test_webrtc_client_html_targets_the_server():
    html = webrtc_client_html(49123, 1280, 720)
    assert "signalingPort: 49123" in html and "aspect-ratio:1280/720" in html
    assert "window.location.hostname" in html and WEBRTC_CLIENT_LIBRARY_URL in html
    assert "server: 'gpu-box'" in webrtc_client_html(49123, 640, 480, server="gpu-box")


def test_live_view_names_the_missing_packages(monkeypatch):
    import sys

    from ovcae.stream import LiveView

    monkeypatch.setitem(sys.modules, "ovstream", None)
    with pytest.raises(ImportError, match=r"ovcae\[stream\]"):
        LiveView(object(), object(), render_product="/R", camera="/C", eye=(1, 0, 0), target=(0, 0, 0))
