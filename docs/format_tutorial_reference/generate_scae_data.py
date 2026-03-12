# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import asyncio
import json
import math
from array import array
from pathlib import Path

from omni.cae.data.commands import execute_command
from omni.cae.importer.scae import import_to_stage
from omni.cae.schema import viz as cae_viz
from omni.cae.testing import frame_prims, wait_for_update
from omni.usd import get_context

# Usage:
# Copy paste this script into the Script Editor (Developer > Script Editor) or execute it on launch w/
# ./repo.sh launch -n omni.cae.kit -- --exec scripts/generate_scae_data.py


# ---------------------------------------------------------------------------
# Data generation (stdlib only — no numpy required)
# ---------------------------------------------------------------------------


def _f32(values: list[float]) -> array:
    values_f32 = array("f", values)
    if values_f32.itemsize != 4:
        raise RuntimeError("Expected 32-bit float array")
    return values_f32


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def _generate_scae_torus(
    output_dir: Path,
    stem: str = "sample",
    num_u: int = 40,
    num_v: int = 25,
    major_radius: float = 5.0,
    minor_radius: float = 1.5,
) -> tuple[Path, Path]:
    """Generate a synthetic torus dataset in .scae format (JSON manifest + binary blob)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_values: list[float] = []
    temperature_values: list[float] = []
    pressure_values: list[float] = []
    velocity_values: list[float] = []
    connectivity_values: list[float] = []
    connectivity_offset_values: list[float] = [0]

    for i_u in range(num_u):
        u = (2.0 * math.pi * i_u) / num_u
        sin_u = math.sin(u)
        cos_u = math.cos(u)
        for i_v in range(num_v):
            v = (2.0 * math.pi * i_v) / num_v
            sin_v = math.sin(v)
            cos_v = math.cos(v)

            ring = major_radius + minor_radius * cos_v
            x = ring * cos_u
            y = ring * sin_u
            z = minor_radius * sin_v
            coords_values.extend((x, y, z))

            temperature = 300.0 + 200.0 * (0.5 + 0.5 * sin_u)
            pressure = 101325.0 + 2600.0 * math.cos(u - 0.5 * v) + 900.0 * math.sin(3.0 * u)

            speed = 1.0 + 0.2 * cos_v
            vx = -sin_u * speed
            vy = cos_u * speed
            vz = 0.35 * sin_v

            temperature_values.append(temperature)
            pressure_values.append(pressure)
            velocity_values.extend((vx, vy, vz))

    for i_u in range(num_u):
        i_u_next = (i_u + 1) % num_u
        for i_v in range(num_v):
            i_v_next = (i_v + 1) % num_v
            a = i_u * num_v + i_v + 1
            b = i_u_next * num_v + i_v + 1
            c = i_u_next * num_v + i_v_next + 1
            d = i_u * num_v + i_v_next + 1
            connectivity_values.extend((a, b, c))
            connectivity_offset_values.append(connectivity_offset_values[-1] + 3)
            connectivity_values.extend((a, c, d))
            connectivity_offset_values.append(connectivity_offset_values[-1] + 3)

    num_points = num_u * num_v
    num_triangles = num_u * num_v * 2
    coords = _f32(coords_values)
    temperature = _f32(temperature_values)
    pressure = _f32(pressure_values)
    velocity = _f32(velocity_values)
    connectivity = array("i", (int(value) for value in connectivity_values))
    connectivity_offsets = array("i", (int(value) for value in connectivity_offset_values))
    if connectivity.itemsize != 4 or connectivity_offsets.itemsize != 4:
        raise RuntimeError("Expected 32-bit integer arrays for connectivity")

    fields = [
        ("Coordinates", coords, (num_points, 3)),
        ("Temperature", temperature, (num_points,)),
        ("Pressure", pressure, (num_points,)),
        ("Velocity", velocity, (num_points, 3)),
        ("ElementConnectivity", connectivity, (num_triangles * 3,)),
        ("ElementStartOffset", connectivity_offsets, (num_triangles + 1,)),
    ]

    binary_path = output_dir / f"{stem}.bin"
    offset = 0
    arrays_spec: dict[str, dict[str, int | str | list[int]]] = {}
    with binary_path.open("wb") as stream:
        for name, values, shape in fields:
            values.tofile(stream)
            arrays_spec[name] = {
                "dtype": "float32" if values.typecode == "f" else "int32",
                "shape": list(shape),
                "offset_bytes": offset,
                "byte_length": _shape_size(shape) * values.itemsize,
            }
            offset += _shape_size(shape) * values.itemsize

    manifest = {
        "version": 1,
        "description": "Synthetic torus flow dataset for Scae onboarding tutorial.",
        "binary_file": binary_path.name,
        "point_count": num_points,
        "topology_hint": "tri_surface",
        "cell_count": num_triangles,
        "arrays": arrays_spec,
    }

    manifest_path = output_dir / f"{stem}.scae"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, binary_path


# ---------------------------------------------------------------------------
# Kit launch entry point
# ---------------------------------------------------------------------------


async def main():
    # 1. Generate the .scae + .bin data files into data/
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    manifest_path, binary_path = _generate_scae_torus(output_dir=data_dir)
    print(f"Generated {manifest_path}")
    print(f"Generated {binary_path}")

    # 2. Import the generated dataset into the Kit stage
    await import_to_stage(str(manifest_path), "/World/scae_torus")
    await wait_for_update()

    # 3. Create a bounding box for the dataset
    dataset_path = "/World/scae_torus/ScaeDataSet"
    bbox_path = "/World/CAE/BoundingBox_ScaeTorus"
    await execute_command("CreateCaeVizBoundingBox", dataset_paths=[dataset_path], prim_path=bbox_path)
    await wait_for_update()

    # 4. Create a NanoVDB volume with Gaussian splatting
    stage = get_context().get_stage()
    volume_path = "/World/CAE/Volume_ScaeTorus"
    await execute_command("CreateCaeVizVolume", dataset_path=dataset_path, prim_path=volume_path, type="vdb")
    await wait_for_update()

    volume_prim = stage.GetPrimAtPath(volume_path)

    # Colour by Temperature
    colors_fs_api = cae_viz.FieldSelectionAPI(volume_prim, "colors")
    colors_fs_api.CreateTargetRel().SetTargets(["/World/scae_torus/ScaeArrays/Temperature"])

    # Set Gaussian splatting radius
    gaussian_splatting_api = cae_viz.DatasetGaussianSplattingAPI(volume_prim, "source")
    gaussian_splatting_api.CreateRadiusFactorAttr().Set(5.0)
    await wait_for_update()

    # 5. Frame the camera on the volume
    await frame_prims([volume_path], zoom=0.9)


if __name__ == "__main__":
    asyncio.ensure_future(main())
