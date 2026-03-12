# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import dav
from omni.cae.data import cache, commands
from omni.cae.schema import cae
from omni.kit.commands import Command
from pxr import Usd


class ConvertToDAVDataSet(Command):
    """
    Command to convert a USD Prim representing a CAE DataSet into a dav.DataSet.
    When introducing any new data model, one must provide a way to register the
    data model with DAV so that it can be processed by the algorithms provided by DAV.
    """

    def __init__(
        self, dataset: Usd.Prim, device: str, timeCode: Usd.TimeCode, needs_topology: bool, needs_geometry: bool
    ) -> None:
        self._dataset = dataset
        self._device = device
        self._timeCode = timeCode
        self._needs_topology = needs_topology
        self._needs_geometry = needs_geometry

    @property
    def dataset(self) -> Usd.Prim:
        """The CAE dataset prim to convert."""
        return self._dataset

    @property
    def device(self) -> str:
        """Device to use for DAV processing (e.g., 'cpu', 'cuda', 'gpu')."""
        return self._device

    @property
    def timeCode(self) -> Usd.TimeCode:
        """Time code for data retrieval."""
        return self._timeCode

    @property
    def needs_topology(self) -> bool:
        """Whether the dataset needs topology information."""
        return self._needs_topology

    @property
    def needs_geometry(self) -> bool:
        """Whether the dataset needs geometry information."""
        return self._needs_geometry

    @classmethod
    async def invoke(
        cls, dataset: Usd.Prim, device: str, timeCode: Usd.TimeCode, needs_topology: bool, needs_geometry: bool
    ) -> dav.Dataset:
        """
        Convert a CAE dataset to a DAV DataSet.

        Args:
            dataset: The prim to convert
            device: Device to use for DAV processing (e.g., 'cpu', 'cuda', 'gpu')
            timeCode: Time code for data retrieval
            needs_topology: Whether the dataset needs topology information
            needs_geometry: Whether the dataset needs geometry information
        Returns:
            A dav.DataSet object
        """
        cache_key = {
            "label": "ConvertToDAVDataSet",
            "dataset": str(dataset.GetPath()),
            "device": device,
            "needs_topology": needs_topology,
            "needs_geometry": needs_geometry,
        }

        cache_state = {}
        dav_dataset = cache.get(str(cache_key), cache_state, timeCode=timeCode)
        if dav_dataset is None:
            dav_dataset = await commands.execute(
                cls.__name__,
                dataset,
                dataset=dataset,
                device=device,
                timeCode=timeCode,
                needs_topology=needs_topology,
                needs_geometry=needs_geometry,
            )
            if dav_dataset:
                cache.put(
                    str(cache_key),
                    dav_dataset,
                    state=cache_state,
                    sourcePrims=[dataset],
                    timeCode=timeCode,
                )
        return dav_dataset

    async def do(self) -> dav.Dataset:
        """
        Execute the command to convert a CAE dataset to a DAV DataSet.

        This is a base implementation that should be overridden by subclasses
        for specific dataset types.

        Returns:
            A dav.DataSet object
        """
        raise NotImplementedError(
            f"Conversion to DAV DataSet not implemented for dataset type: {self.dataset.GetTypeName()}. "
            f"Please implement a subclass of ConvertToDAVDataSet for this dataset type."
        )
