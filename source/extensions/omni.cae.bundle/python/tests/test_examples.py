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

import omni.kit.test
from omni.cae.testing import new_stage

logger = getLogger(__name__)


class Test(omni.kit.test.AsyncTestCase):

    def is_windows(self):
        import platform

        return platform.system() == "Windows"

    async def test_example_bounding_box(self):
        from scripts import example_bounding_box

        if self.is_windows():
            logger.warning("Skipping bounding box example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running bounding box example")
            await example_bounding_box.main()
            logger.info("Bounding box example completed")

    async def test_example_faces(self):
        from scripts import example_faces

        if self.is_windows():
            logger.warning("Skipping faces example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running faces example")
            await example_faces.main()
            logger.info("Faces example completed")

    async def test_example_glyphs(self):
        from scripts import example_glyphs

        if self.is_windows():
            logger.warning("Skipping glyphs example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running glyphs example")
            await example_glyphs.main()
            logger.info("Glyphs example completed")

    async def test_example_npz_flow(self):
        from scripts import example_npz_flow

        if self.is_windows():
            logger.warning("Skipping npz flow example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running npz flow example")
            await example_npz_flow.main(skip_play=True)
            logger.info("NPZ flow example completed")

    async def test_example_npz_point_cloud(self):
        from scripts import example_npz_point_cloud

        if self.is_windows():
            logger.warning("Skipping npz point cloud example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running npz point cloud example")
            await example_npz_point_cloud.main()
            logger.info("NPZ point cloud example completed")

    async def test_example_npz_streamlines(self):
        from scripts import example_npz_streamlines

        if self.is_windows():
            logger.warning("Skipping npz streamlines example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running npz streamlines example")
            await example_npz_streamlines.main()
            logger.info("NPZ streamlines example completed")

    async def test_example_nvdb_slice(self):
        from scripts import example_nvdb_slice

        if self.is_windows():
            logger.warning("Skipping nvdb slice example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running nvdb slice example")
            await example_nvdb_slice.main(skip_play=True)
            logger.info("NVDB slice example completed")

    async def test_example_points(self):
        from scripts import example_points

        if self.is_windows():
            logger.warning("Skipping points example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running points example")
            await example_points.main()
            logger.info("Points example completed")

    async def test_example_slice(self):
        from scripts import example_slice

        if self.is_windows():
            logger.warning("Skipping slice example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running slice example")
            await example_slice.main()
            logger.info("Slice example completed")

    async def test_example_streamlines(self):
        from scripts import example_streamlines

        if self.is_windows():
            logger.warning("Skipping streamlines example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running streamlines example")
            await example_streamlines.main()
            logger.info("Streamlines example completed")

    async def test_example_temporal_interpolation(self):
        from scripts import example_temporal_interpolation

        if self.is_windows():
            logger.warning("Skipping temporal interpolation example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running temporal interpolation example")
            await example_temporal_interpolation.main(skip_play=True)
            logger.info("Temporal interpolation example completed")

    async def test_example_volume(self):
        from scripts import example_volume

        if self.is_windows():
            logger.warning("Skipping volume example on Windows due to known issues with the example.")
            return

        async with new_stage():
            logger.info("Running volume example")
            await example_volume.main()
            logger.info("Volume example completed")
