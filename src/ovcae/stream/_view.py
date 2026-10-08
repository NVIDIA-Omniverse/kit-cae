# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""A render thread that streams an OVRTX render product to a browser and takes edits through a mailbox."""

from __future__ import annotations

import contextlib
import threading
import time
from functools import cache
from typing import Any

from ..app import UpdateLoop
from ._camera import OrbitCamera

YAW_GAIN = PITCH_GAIN = 0.005  # radians per pixel of drag
WHEEL_GAIN = 0.25  # scene units per wheel notch
_MISSING = "ovcae.stream needs the OVStream and OVRTX Python packages; install ovcae[stream] and an OVRTX wheel"


class LiveView:
    """Render continuously on a background thread, stream the frames over WebRTC, and apply edits between frames.

    One thread owns every stage write. Other threads, a notebook's widget
    callbacks or OVStream's input callback, drop ``(prim path, property)``
    edits into a mailbox: :meth:`request` for recipe edits, :meth:`write` for
    presentation edits. Once per frame the render thread empties both, hands
    the presentation batch to the :class:`~ovcae.app.UpdateLoop` to write and
    the recipe batch to run an update on. Mouse input from the browser orbits
    :attr:`orbit` and writes the camera prim's ``omni:xform``. Frames are
    rendered at the loop's committed ordinal, converted to BGRA on the GPU,
    and encoded by an OVStream WebRTC server on ``signal_port``.

    The loop must have committed its first update, and the renderer must be
    attached to the loop's stage, before the view is created; the caller
    releases both after :meth:`stop`. Requires the ``stream`` extra and OVRTX.
    """

    def __init__(
        self,
        loop: UpdateLoop,
        renderer,
        *,
        render_product: str,
        camera: str,
        eye,
        target,
        gpu_index: str = "0",
        signal_port: int = 49100,
        public_host: str | None = None,
        fps: int = 30,
        up_axis: str = "Z",
    ):
        try:
            import ovrtx
            import ovstream  # noqa: F401
        except ImportError as exc:
            raise ImportError(_MISSING) from exc
        import warp as wp

        self.loop = loop
        self.renderer = renderer
        self.render_product = render_product
        self.camera = camera
        self.signal_port = signal_port
        self.public_host = public_host
        self.fps = fps
        self.orbit = OrbitCamera(eye, target, up_axis=up_axis)
        self.error: BaseException | None = None
        self._color_output = f"{render_product}/LdrColor"
        self._gpu_index = str(gpu_index)
        self._cuda_device = f"cuda:{self._gpu_index}"
        self._device_enum = ovrtx.Device.CUDA
        self._mailbox: dict[tuple[str, str], Any] = {}
        self._presentation: dict[tuple[str, str], Any] = {}
        self._requested = 0  # requests made so far; _applied trails it as the render thread writes them
        self._applied = 0
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._drag = {"active": False, "last": None}
        self._thread: threading.Thread | None = None
        self._server = None
        self.height, self.width = self._render_size()
        self._buffer = wp.zeros((self.height, self.width, 4), dtype=wp.uint8, device=self._cuda_device)
        self._stream = wp.get_stream(self._cuda_device)
        self._event = wp.Event(device=self._cuda_device)

    # -- state -----------------------------------------------------------------------

    @property
    def ordinal(self) -> int:
        """The committed ordinal the thread renders at."""
        return self.loop.ordinal

    @property
    def last_report(self):
        """The report of the most recent committed update."""
        return self.loop.last_report

    # -- requests --------------------------------------------------------------------

    def request(self, path: str, name: str, value) -> None:
        """Ask the render thread to write recipe edit ``value`` to ``name`` on ``path`` and run an update.

        The latest value per key wins when several arrive within one frame.
        """
        with self._lock:
            self._mailbox[(path, name)] = value
            self._requested += 1

    def write(self, path: str, name: str, value) -> None:
        """Ask the render thread to write presentation edit ``value`` to ``name`` on ``path``; no update follows."""
        with self._lock:
            self._presentation[(path, name)] = value
            self._requested += 1

    def wait_for_update(self, timeout: float = 60.0):
        """Block until every request made so far has been written and evaluated; return the latest report."""
        with self._lock:
            target = self._requested
        started = time.perf_counter()
        while self._applied < target:
            time.sleep(0.05)
            if self.error is not None:
                raise RuntimeError("the render thread stopped") from self.error
            if time.perf_counter() - started > timeout:
                raise TimeoutError("the render thread did not apply the requests in time")
        return self.loop.last_report

    def snapshot(self):
        """The frame the stream is showing, as an RGBA NumPy array."""
        import numpy as np

        return np.ascontiguousarray(self._buffer.numpy()[..., [2, 1, 0, 3]])

    def html(self) -> str:
        """The browser client for this stream, for ``IPython.display.HTML``."""
        from ._client import webrtc_client_html

        return webrtc_client_html(self.signal_port, self.width, self.height, server=self.public_host)

    # -- lifecycle -------------------------------------------------------------------

    def start(self) -> None:
        """Start the OVStream server and the render thread."""
        import ovstream
        import warp as wp

        ovstream.initialize(log_min_severity=ovstream.LogLevel.WARNING)
        self._server = ovstream.Server(ovstream.ServerType.WEBRTC)
        self._server.on_connection = self._on_connection
        self._server.on_input = self._on_input
        config = ovstream.ServerConfig(
            width=self.width,
            height=self.height,
            target_fps=self.fps,
            video_input=ovstream.VideoInput.CUDA,
            # Pin the encoder to the GPU and CUDA context the frames are produced in.
            cuda_device=int(self._gpu_index),
            cuda_context=int(wp.get_device(self._cuda_device).context),
            webrtc_signal_port=self.signal_port,
            webrtc_public_ip=self.public_host,
        )
        for attempt in range(5):  # a server that just shut down in another process can hold the ports briefly
            try:
                self._server.start(config)
                break
            except ovstream.OvstreamError:
                if attempt == 4:
                    raise
                time.sleep(2.0)
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="ovcae-stream-render", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the render thread, then the server. The caller releases the loop, stage, and renderer."""
        import ovstream

        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                raise RuntimeError("the render thread is still busy; wait for its update to finish and stop again")
            self._thread = None
        if self._server is not None:
            with contextlib.suppress(ovstream.OvstreamError):
                self._server.stop()
            self._server.close()
            ovstream.shutdown()
            self._server = None

    def __enter__(self) -> LiveView:
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.stop()

    # -- the render thread -----------------------------------------------------------

    def _step(self):
        return self.renderer.step(
            render_products={self.render_product}, delta_time=1.0 / self.fps, ordinal=self.ordinal
        )

    def _render_size(self) -> tuple[int, int]:
        products = self._step()
        with products[self.render_product].frames[0].render_vars[self._color_output].map(device=self._device_enum) as m:
            return int(m.shape[0]), int(m.shape[1])

    def _render(self) -> None:
        import warp as wp

        products = self._step()
        with products[self.render_product].frames[0].render_vars[self._color_output].map(device=self._device_enum) as m:
            wp.copy(self._buffer, wp.from_dlpack(m))

    def _stream_frame(self) -> None:
        import ovstream
        import warp as wp

        self._render()
        wp.launch(
            _swap_red_blue_kernel(), dim=(self.width, self.height), inputs=[self._buffer], device=self._cuda_device
        )
        self._stream.record_event(self._event)
        frame = ovstream.VideoFrame.from_cuda_array(
            self._buffer,
            sync=ovstream.CudaSync(stream=self._stream.cuda_stream, wait_event=self._event.cuda_event),
        )
        with contextlib.suppress(ovstream.OvstreamError):  # raises while no client is connected
            self._server.stream_video(frame)

    def _loop(self) -> None:
        import ovstream_utils

        try:
            with ovstream_utils.Loop(ovstream_utils.LoopConfig(fps_target=self.fps)) as pacer:
                while self._running.is_set():
                    pacer.tick()
                    with self._lock:
                        presentation, edits = dict(self._presentation), dict(self._mailbox)
                        self._presentation.clear()
                        self._mailbox.clear()
                        drained = self._requested
                    if presentation:
                        self.loop.write(presentation)
                    if edits:
                        self.loop.update(edits)
                    self._applied = drained
                    self._stream_frame()
        except BaseException as exc:  # keep the failure for the caller instead of losing it in the thread
            self.error = exc
            self._running.clear()

    def _on_connection(self, connected: bool) -> None:
        print("Browser", "connected" if connected else "disconnected")

    def _on_input(self, event) -> None:
        import ovstream

        if event.type != ovstream.InputEventType.MOUSE:
            return
        mouse = event.mouse
        if mouse.type == ovstream.MouseEventType.BUTTON and mouse.data == ovstream.MouseButton.LEFT:
            self._drag["active"] = mouse.button_state == ovstream.KeyState.DOWN
            self._drag["last"] = (mouse.x, mouse.y)
        elif mouse.type == ovstream.MouseEventType.MOVE and self._drag["active"] and self._drag["last"] is not None:
            dx, dy = mouse.x - self._drag["last"][0], mouse.y - self._drag["last"][1]
            self._drag["last"] = (mouse.x, mouse.y)
            self.orbit.yaw -= dx * YAW_GAIN
            self.orbit.pitch = max(-1.3, min(1.3, self.orbit.pitch + dy * PITCH_GAIN))
            self.write(self.camera, "omni:xform", self.orbit.matrix())
        elif mouse.type == ovstream.MouseEventType.WHEEL:
            self.orbit.zoom -= mouse.scroll_y * WHEEL_GAIN
            self.write(self.camera, "omni:xform", self.orbit.matrix())


@cache
def _swap_red_blue_kernel():
    """OVRTX writes RGBA8; OVStream wants BGRA8. Swap in place on the GPU, built on first use."""
    import warp as wp

    @wp.kernel
    def swap_red_blue(pixels: wp.array3d(dtype=wp.uint8)):
        x, y = wp.tid()
        red = pixels[y, x, 0]
        blue = pixels[y, x, 2]
        pixels[y, x, 0] = blue
        pixels[y, x, 2] = red

    return swap_red_blue


__all__ = ["LiveView"]
