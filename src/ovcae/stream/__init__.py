# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Stream an OVRTX render product to a browser and edit the scene while it plays.

:class:`LiveView` runs the :class:`~ovcae.app.UpdateLoop` on a render thread,
streams each frame over WebRTC through OVStream, and orbits a camera from the
browser's mouse; :func:`webrtc_client_html` is the page that shows the video,
sized for a notebook cell. Importing this package loads only the standard
library and NumPy; OVStream, OVRTX, and Warp load when a view is created.
Install the ``stream`` extra for OVStream; OVRTX is installed separately.
"""

from ._camera import OrbitCamera
from ._client import WEBRTC_CLIENT_LIBRARY_URL, webrtc_client_html
from ._view import LiveView

__all__ = ["WEBRTC_CLIENT_LIBRARY_URL", "LiveView", "OrbitCamera", "webrtc_client_html"]
