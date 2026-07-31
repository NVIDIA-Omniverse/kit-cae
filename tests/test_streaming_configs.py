# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import unittest

STREAMING_CONFIGS_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'templates', '110.1.3', 'apps', 'streaming_configs'
)


class TestStreamingConfigs(unittest.TestCase):
    def test_app_extensions_exclude_key(self):
        """Streaming configs must use 'exclude =' not 'excluded =' under [settings.app.extensions]."""
        for filename in sorted(os.listdir(STREAMING_CONFIGS_DIR)):
            if not filename.endswith('.kit'):
                continue
            path = os.path.join(STREAMING_CONFIGS_DIR, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            section_start = content.find('[settings.app.extensions]')
            self.assertNotEqual(section_start, -1,
