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
        cache_key = f"[dav:ConvertToDAVDataSet]::{dataset.GetPath()}::{device}::{needs_topology}::{needs_geometry}"
        if dav_dataset := cache.get(cache_key, timeCode=timeCode):
            return dav_dataset

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
            cache.put_ex(cache_key, dav_dataset, prims=[cache.PrimWatch(dataset)], timeCode=timeCode)
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


class GetField(Command):
    """
    Command to retrieve a named field from a USD dataset prim as a DAV field.

    When introducing a new dataset schema that needs custom field loading behavior,
    register a subclass named <SchemaTypeName>GetField via omni.kit.commands.
    """

    def __init__(self, dataset: Usd.Prim, field_names: list[str], device: str, timeCode: Usd.TimeCode) -> None:
        self._dataset = dataset
        self._field_names = field_names
        self._device = device
        self._timeCode = timeCode

    @property
    def dataset(self) -> Usd.Prim:
        """The dataset prim to load the field from."""
        return self._dataset

    @property
    def field_names(self) -> list[str]:
        """One or more dataset field names to combine into a DAV field."""
        return self._field_names

    @property
    def device(self) -> str:
        """Device to use for DAV field data."""
        return self._device

    @property
    def timeCode(self) -> Usd.TimeCode:
        """Time code for field data retrieval."""
        return self._timeCode

    @classmethod
    async def invoke(
        cls, dataset: Usd.Prim, field_name_or_names: str | list[str], device: str, timeCode: Usd.TimeCode
    ) -> dav.Field:
        """
        Return the DAV field for one or more field names on *dataset*.
        """
        return await commands.execute(
            cls.__name__,
            dataset,
            dataset=dataset,
            field_names=field_name_or_names if isinstance(field_name_or_names, list) else [field_name_or_names],
            device=device,
            timeCode=timeCode,
        )

    async def do(self) -> dav.Field:
        """
        Execute the command to retrieve a field from a dataset prim.
        """
        raise NotImplementedError(
            f"GetField not implemented for dataset type: {self.dataset.GetTypeName()}. "
            f"Please implement a subclass of GetField for this dataset type."
        )
