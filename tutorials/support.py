# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Small helpers shared by the tutorials.

Sections: environment (device and asset locations), images, OVStage reads
and writes, and the stream host setting. Evaluation, session, and renderer
calls stay in the tutorials themselves; nothing here hides an OVCAE concept.
"""

import os
import struct
import zlib
from pathlib import Path

import numpy as np

# --- Environment ------------------------------------------------------------

DEVICE_VARIABLE = "OVCAE_TUTORIAL_DEVICE"
ASSETS_VARIABLE = "OVCAE_TUTORIAL_ASSETS"

REPO_ROOT = Path(__file__).resolve().parents[1]
# Where tutorials write the files they generate; never next to their sources.
OUTPUT_DIR = REPO_ROOT / "build" / "tutorials"
# Scenes and datasets the tutorials read, checked in (large ones through Git LFS);
# see tutorials/data/README.md for provenance. OVCAE_TUTORIAL_ASSETS relocates them.
DATA_DIR = REPO_ROOT / "tutorials" / "data"
DEFAULT_ASSETS_DIR = DATA_DIR


def assets_dir() -> Path:
    """Return the directory holding the tutorial scenes and datasets.

    ``OVCAE_TUTORIAL_ASSETS`` wins when set; ``tutorials/run.py --assets``
    sets it for the kernels it launches. Otherwise ``tutorials/data``.
    """
    return Path(os.environ.get(ASSETS_VARIABLE) or DEFAULT_ASSETS_DIR).expanduser().resolve()


def asset(name: str) -> Path:
    """Resolve one file in the assets directory, failing with instructions if it is absent."""
    path = assets_dir() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} not found in {assets_dir()}. Run `git lfs pull` in the checkout, or point "
            f"{ASSETS_VARIABLE} (or `tutorials/run.py --assets`) at a directory that holds the tutorial data."
        )
    return path


def cuda_gpu_index(device: str) -> str | None:
    """Map a Warp device name to OVRTX's ``active_cuda_gpus`` string.

    ``cuda:N`` pins the renderer to GPU N and bare ``cuda`` to GPU 0. For any
    other device (``cpu``) return None, which leaves OVRTX's own GPU selection
    alone: rendering always happens on a GPU even when compute runs on CPU.
    """
    name, _, index = device.partition(":")
    if name != "cuda":
        return None
    return index or "0"


def tutorial_device() -> str:
    """Return the Warp device name the tutorials evaluate on.

    ``OVCAE_TUTORIAL_DEVICE`` wins when set (``cpu`` or ``cuda:N``). Otherwise
    the first CUDA device is used when Warp can see one, and CPU is the
    fallback so every tutorial still runs on a machine without a GPU.
    """
    requested = os.environ.get(DEVICE_VARIABLE)
    if requested:
        return requested
    import warp as wp

    # Probing devices initializes Warp; keep its banner out of the caller's output.
    wp.config.log_level = wp.LOG_WARNING
    return "cuda:0" if wp.is_cuda_available() else "cpu"


# --- Images -----------------------------------------------------------------


def save_png(pixels, path) -> Path:
    """Write an RGBA (or RGB) uint8 image as a PNG without any imaging dependency."""
    pixels = np.ascontiguousarray(pixels)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or pixels.dtype != np.uint8:
        raise ValueError("expected a height x width x 3|4 uint8 array")
    height, width, channels = pixels.shape
    color_type = 6 if channels == 4 else 2
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- OVStage reads and writes -----------------------------------------------
#
# Every value an application changes after population is written at a new
# ordinal through an OVStage query, and committed values are read back the
# same way. The helpers below wrap the query and path-dictionary boilerplate
# so tutorials can show the read or write itself.


def read_attribute(stage, path: str, name: str, ordinal: int):
    """Return the committed value of ``name`` on ``path`` at ``ordinal``, or None if unset.

    Arrays come back as NumPy arrays (copied out of the stage), scalars as
    Python values, and token attributes as strings.
    """
    import ovstage

    with ovstage.PathDictionary(stage) as paths:
        path_list = paths.create_path_list_from_strings([path])
        query = stage.query_from_path_list(path_list)
        try:
            with stage.read_attributes(query, [paths.intern_token(name)], ovstage.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete or not group.tensor_count:
                            continue
                        values = group.array(0).copy()
                        if group.raw.semantic == ovstage.AttributeSemantic.TOKEN_ID:
                            tokens = tuple(paths.token_to_string(int(v)) for v in values.reshape(-1))
                            return tokens if group.is_array else (tokens[0] if tokens else None)
                        return values if group.is_array or values.size != 1 else values.item()
                    finally:
                        stage.release_group(group)
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)
    return None


def write_scalar(stage, path: str, name: str, value, ordinal: int) -> None:
    """Write one scalar attribute (a recipe setting, say) at ``ordinal``."""
    import ovstage

    with ovstage.PathDictionary(stage) as paths:
        path_list = paths.create_path_list_from_strings([path])
        query = stage.query_from_path_list(path_list)
        try:
            stage.write_attribute(
                query,
                name,
                ordinal=ordinal,
                tensors=np.asarray([value]),
                is_array=False,
                prim_mode=ovstage.PrimMode.UPSERT,
            ).wait()
        finally:
            stage.release_query(query).wait()
            paths.destroy_path_list(path_list)


# --- Browser stream host ----------------------------------------------------

STREAM_HOST_VARIABLE = "OVCAE_TUTORIAL_STREAM_HOST"


def stream_host() -> str | None:
    """Host name the browser should connect to, or None to use the notebook page's own host."""
    return os.environ.get(STREAM_HOST_VARIABLE) or None
