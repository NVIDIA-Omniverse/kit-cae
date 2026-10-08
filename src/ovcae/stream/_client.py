# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The browser side of a stream: an HTML fragment that connects to an OVStream WebRTC server."""

from __future__ import annotations

# The stock ovstream browser client library, served from the tagged public
# repository. Keep the tag in step with the ovstream version in pyproject.toml.
WEBRTC_CLIENT_LIBRARY_URL = "https://cdn.jsdelivr.net/gh/NVIDIA-Omniverse/ovstream@v0.4.5/examples/webrtc_client/omniverse-webrtc-streaming-library.js"


def webrtc_client_html(signal_port: int, width: int, height: int, *, server: str | None = None) -> str:
    """Return an HTML fragment that connects to an OVStream WebRTC server and shows its video.

    This is the ovstream repository's ``examples/webrtc_client/index.html``
    reduced to what a notebook cell needs: a video element, the client
    library, and the same ``AppStreamer.connect`` call. ``server`` is the host
    the browser connects to; None means the host the page was loaded from,
    which is right when the notebook server runs on the GPU machine. The
    fragment takes keyboard focus while the pointer is over the video, so the
    notebook's own shortcuts keep working elsewhere, and reconnects when
    frames stop arriving.
    """
    server_js = repr(server) if server else "window.location.hostname || '127.0.0.1'"
    element = f"ovcae-stream-{signal_port}"
    return f"""
<div style="max-width:{width}px">
  <div id="{element}-status" style="font:12px sans-serif;color:#666;margin-bottom:4px">connecting...</div>
  <video id="{element}" autoplay playsinline tabindex="-1"
         style="width:100%;aspect-ratio:{width}/{height};background:#111;outline:none"></video>
  <audio id="{element}-audio" autoplay></audio>
</div>
<script>
(function () {{
  const status = document.getElementById("{element}-status");
  const video = document.getElementById("{element}");
  let connected = false;
  let lastFrames = -1, stalledChecks = 0, reconnecting = false;
  function connect() {{
    const {{ AppStreamer, StreamType }} = window.OVWebRTC;
    connected = false;
    status.textContent = "connecting...";
    AppStreamer.connect({{
      streamSource: StreamType.DIRECT,
      streamConfig: {{
        videoElementId: "{element}",
        audioElementId: "{element}-audio",
        server: {server_js},
        signalingPort: {signal_port},
        fps: 60,
        maxReconnects: 3,
        onStart: (msg) => {{
          if (msg.action !== "start") return;
          if (msg.status === "success") {{
            connected = true;
            reconnecting = false;
            video.muted = false;
            if (video.srcObject) video.play().catch(() => {{}});
          }} else if (msg.status === "error") {{
            status.textContent = "error: " + (msg.info || "unknown");
          }}
        }},
        onStop: (msg) => {{
          if (msg.action !== "terminate") return;
          connected = false;
          if (!reconnecting) status.textContent = "disconnected";
        }},
      }},
    }}).catch((err) => {{ status.textContent = "connect failed: " + err.message; }});
  }}
  // The library forwards mouse and keyboard input only while the video has
  // focus, and a focused video swallows the notebook's own shortcuts such as
  // Shift+Enter. Take focus on click and give it back when the pointer leaves.
  video.addEventListener("pointerdown", () => video.focus());
  video.addEventListener("mouseleave", () => video.blur());
  // JupyterLab's notebook windowing detaches and re-attaches cell outputs as
  // you move between cells, and a media element removed from the document
  // pauses. Resume whenever that happens so the picture does not go black.
  const resume = () => {{
    if (video.isConnected && video.srcObject && video.paused) video.play().catch(() => {{}});
  }};
  video.addEventListener("pause", () => setTimeout(resume, 50));
  // Once a second: resume if paused, report what the element is doing, and if
  // decoded frames stop arriving for several seconds, reconnect the stream.
  setInterval(() => {{
    resume();
    if (!connected) return;
    const track = video.srcObject && video.srcObject.getVideoTracks()[0];
    const frames = video.getVideoPlaybackQuality ? video.getVideoPlaybackQuality().totalVideoFrames : -1;
    const flowing = frames !== lastFrames;
    stalledChecks = flowing ? 0 : stalledChecks + 1;
    lastFrames = frames;
    const detail = [
      video.videoWidth + "x" + video.videoHeight,
      frames + " frames",
      video.paused ? "paused" : "playing",
      track ? "track " + track.readyState + (track.muted ? " muted" : "") : "no track",
      video.isConnected ? "" : "detached",
    ].filter(Boolean).join(" · ");
    if (stalledChecks >= 4 && !reconnecting) {{
      reconnecting = true;
      status.textContent = "stream stalled (" + detail + "), reconnecting...";
      const {{ AppStreamer }} = window.OVWebRTC;
      Promise.resolve(AppStreamer.terminate()).catch(() => {{}}).then(() => setTimeout(connect, 500));
      return;
    }}
    const hint = " · click the video to take control, left-drag orbits, wheel zooms";
    status.textContent = (flowing ? "connected · " : "stalled · ") + detail + hint;
  }}, 1000);
  if (window.OVWebRTC) {{
    connect();
  }} else {{
    const script = document.createElement("script");
    script.src = "{WEBRTC_CLIENT_LIBRARY_URL}";
    script.onload = connect;
    script.onerror = () => {{ status.textContent = "could not load the ovstream browser client library"; }};
    document.head.appendChild(script);
  }}
}})();
</script>
"""


__all__ = ["WEBRTC_CLIENT_LIBRARY_URL", "webrtc_client_html"]
