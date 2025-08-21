# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
#  its affiliates is strictly prohibited.

import logging


class TestSchemas(object):
    """
    Encapsulates sanity tests for the built schemas.
    """

    config = ""

    def test_cae_schema(self):
        import OmniCae

        logging.info("Importing OmniCae succeeded!")

    def test_sids_schema(self):
        import OmniCaeSids

        logging.info("Importing OmniCaeSids succeeded!")

    def test_cgns_schema(self):
        import OmniCaeCgns

        logging.info("Importing OmniCgns succeeded!")

    def test_npz_schema(self):
        import OmniCaeNumPy

        logging.info("Importing OmniCaeNumPy succeeded!")
