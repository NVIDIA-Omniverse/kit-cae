# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import importlib

import warp as wp

# While this model is in `dav.core` this will depend on other modules in `dav` as needed.

configuration = {
    # defines which devices we want AOT compile for.
    "devices": ["cpu"],
    # identify field models to compile for.
    # Each entry is a full type spec:
    #   array:         [layout, dtype, length]       e.g. ["SoA", "float32", 3]
    #   nanovdb:       [dtype]                        e.g. ["float32"]
    #   collection:    [inner_field_type, *inner_args] e.g. ["array", "AoS", "float32", 3]
    #   vector_reduced: [inner_spec, reduction, *reduction_args]
    #                  and reduction is "component" (with int arg) or "magnitude"
    #   selection:     [inner_spec, mode]
    #                  where mode is "subrange" or "indexed_subset"
    #   inner_spec can be a storage model spec such as ["array", ...] or a
    #   nested view model spec such as ["vector_reduced", inner_spec, ...].
    "field_models": {
        "array": [["SoA", "float32", 3], ["AoS", "float32", 1], ["AoS", "int32", 1], ["AoS", "uint32", 1]],
        "nanovdb": [["float32"], ["vec3f"]],
        "collection": [],  # disabling for now as collection support is still in infancy
        "vector_reduced": [
            [["array", "SoA", "float32", 3], "component", 0],
            [["array", "SoA", "float32", 3], "component", 1],
            [["array", "SoA", "float32", 3], "component", 2],
            [["array", "SoA", "float32", 3], "magnitude"],
        ],
        "selection": [],
    },
    "data_models": {
        "custom": {
            "curves": None,  # disabled
            "gaussian_point_cloud": {},
            "point_cloud": {},
            "surface_mesh": None,  # disabled
        },
        "ensight_gold": {
            "unstructured_part": [
                ["tetra4", "hexa8", "pyramid5", "penta6", "nfaced"],  # 3D elements
                ["tria3", "quad4", "nsided"],  # 2.5D elements
            ]
        },
        "openfoam": {
            "boundary_mesh": None,  # disabled since only used in simple operators
            "polymesh": {},  # enabled
        },
        "sids": {
            "nface_n": {},  # polyhedral element blocks
            "unstructured": [
                ["ngon_n"],  # polyhedral elements with explicit face connectivity
                ["tetra_4", "hexa_8", "pyra_5", "penta_6"],  # 3D elements
                ["tri_3", "quad_4"],  # 2.5D elements
            ],
        },
        "vtk": {
            "image_data": {},  # enabled
            "polydata": None,  # not need to pre-compile since used in simple operators
            "structured_grid": {},  # enabled
            "unstructured_grid": [
                ["TETRA", "HEXAHEDRON", "WEDGE", "PYRAMID", "PENTAGONAL_PRISM"],  # 3D elements
                ["TRIANGLE", "QUAD", "POLYGON"],  # 2.5D elements
                ["POLYHEDRON"],  # polyhedral elements with explicit face connectivity
            ],
        },
    },
    "operators": {
        "advection": {
            "seeds": {
                # specify data models for "seeds" input
                "data_models": {
                    "custom": {
                        "point_cloud": {},  # enabled
                        "surface_mesh": {},
                    }
                    # not enabling any other data models for seeds since that's rare
                }
            },
            # specify data models for "dataset" (default) input
            "data_models": {
                "ensight_gold": {
                    "unstructured_part": [
                        ["tetra4", "hexa8", "pyramid5", "penta6", "nfaced"]  # 3D elements
                    ]
                },
                "openfoam": {
                    "boundary_mesh": None,  # disabled since only used in simple operators
                    "polymesh": {},  # enabled
                },
                "sids": {
                    "nface_n": {},  # polyhedral element blocks
                    "unstructured": [
                        ["ngon_n"],  # polyhedral elements with explicit face connectivity
                        ["tetra_4", "hexa_8", "pyra_5", "penta_6"],  # 3D elements
                    ],
                },
                "vtk": {
                    "image_data": {},  # enabled
                    "structured_grid": {},  # enabled
                    "unstructured_grid": [
                        ["TETRA", "HEXAHEDRON", "WEDGE", "PYRAMID", "PENTAGONAL_PRISM"],  # 3D elements
                        ["POLYHEDRON"],  # polyhedral elements with explicit face connectivity
                    ],
                },
            },
        },
        "probe": {},  # enabled with no overrides
        "point_field": {},
        "cell_field": {},
        "centroid": {},
        "cell_sizes": {},
        "voxelization": {},  # enabled with no overrides
    },
}


def get_devices():
    """Get a list of devices to compile AOT for."""
    return configuration.get("devices", [])


def get_scalar_types():
    """Get the unique scalar types used in length-1 array field models.

    Used by other modules (e.g. ``dav.data_models.utils``) to generate AOT
    overloads for scalar kernels.  Derived from ``field_models.array`` entries
    with ``length == 1`` so there is no separate ``scalar_types`` key to keep
    in sync.
    """
    types = []
    seen = set()
    for spec in configuration.get("field_models", {}).get("array", []):
        _, dtype_str, length = spec
        if length == 1 and dtype_str not in seen:
            seen.add(dtype_str)
            types.append(getattr(wp, dtype_str))
    return types


def _resolve_inner_spec(spec, array_mod, nanovdb_mod, collection_mod=None, vector_reduced_mod=None, selection_mod=None):
    """Resolve an inner field model spec to a model class.

    Inner specs have the form ``[field_type, *args]``:
      - ``["array", layout, dtype_str, length]``
      - ``["nanovdb", dtype_str]``
      - ``["collection", inner_spec]``
      - ``["vector_reduced", inner_spec, "component", component]``
      - ``["vector_reduced", inner_spec, "magnitude"]``
      - ``["selection", inner_spec, "subrange"|"indexed_subset"]``
    """
    field_type, *args = spec
    if field_type == "array":
        layout, dtype_str, length = args
        dtype = getattr(wp, dtype_str)
        if layout == "SoA":
            return array_mod.get_field_model_SoA(dtype, length)
        else:
            return array_mod.get_field_model_AoS(dtype, length)
    elif field_type == "nanovdb":
        return nanovdb_mod.get_field_model(getattr(wp, args[0]))
    elif field_type == "collection":
        if collection_mod is None:
            from dav.fields import collection as collection_mod

        return collection_mod.get_field_model(_resolve_inner_spec(args[0], array_mod, nanovdb_mod, collection_mod, vector_reduced_mod, selection_mod))
    elif field_type == "vector_reduced":
        if vector_reduced_mod is None:
            from dav.fields import vector_reduced as vector_reduced_mod

        inner_spec, reduction, *reduction_args = args
        inner_model = _resolve_inner_spec(inner_spec, array_mod, nanovdb_mod, collection_mod, vector_reduced_mod, selection_mod)
        if reduction == "component":
            return vector_reduced_mod.get_field_model_vector_reduced(inner_model, component=reduction_args[0])
        elif reduction == "magnitude":
            return vector_reduced_mod.get_field_model_vector_reduced(inner_model, magnitude=True)
    elif field_type == "selection":
        if selection_mod is None:
            from dav.fields import selection as selection_mod

        inner_spec, mode = args
        inner_model = _resolve_inner_spec(inner_spec, array_mod, nanovdb_mod, collection_mod, vector_reduced_mod, selection_mod)
        if mode == "subrange":
            return selection_mod.get_field_model_subrange(inner_model)
        elif mode == "indexed_subset":
            return selection_mod.get_field_model_indexed_subset(inner_model)
    else:
        raise ValueError(f"Unknown field type in inner spec: {field_type!r}")

    raise ValueError(f"Unknown field model spec: {spec!r}")


def get_field_models(*, selected_length: int = None, selected_element_type=None, specialization: str = None):
    """Get a list of all field models we want to compile AOT.

    Args:
        selected_length: Optional filter for specific vector lengths (e.g. ``1`` or ``3``).
        selected_element_type: Optional filter for a specific scalar element type
            (e.g. ``wp.float32``).  For array specs this is the dtype stored in the
            spec directly; for nanovdb it is the scalar component of the volume type.
        specialization: Optional dot-notation path into the configuration to select a specific
            set of field models. For example, ``"operators.point_field"`` navigates to
            ``configuration["operators"]["point_field"]["field_models"]``.
            If None, uses the top-level ``configuration["field_models"]``.
    """
    from dav import utils as dav_utils
    from dav.fields import array, collection, nanovdb, selection, vector_reduced

    field_models = []

    if specialization is None:
        field_models_config = configuration.get("field_models", {})
    else:
        keys = specialization.split(".")
        # Walk down the path, collecting each node along the way
        nodes = [configuration]
        node = configuration
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
            nodes.append(node)
        # Search from most specific to least specific for a "field_models" key
        field_models_config = {}
        for candidate in reversed(nodes):
            if isinstance(candidate, dict) and "field_models" in candidate:
                field_models_config = candidate["field_models"]
                break

    array_models = []
    for spec in field_models_config.get("array", []):
        layout, dtype_str, length = spec
        if selected_length is not None and length != selected_length:
            continue
        dtype = getattr(wp, dtype_str)
        if selected_element_type is not None and dtype != selected_element_type:
            continue
        if layout == "SoA":
            array_models.append(array.get_field_model_SoA(dtype, length))
        elif layout == "AoS":
            array_models.append(array.get_field_model_AoS(dtype, length))
    field_models.extend(array_models)

    nanovdb_models = []
    for spec in field_models_config.get("nanovdb", []):
        dtype = getattr(wp, spec[0])
        if dav_utils.is_vector_dtype(dtype):
            elem_type = dav_utils.get_scalar_dtype(dtype)
            length = dav_utils.get_vector_length(dtype)
        else:
            elem_type = dtype
            length = 1
        if selected_length is not None and length != selected_length:
            continue
        if selected_element_type is not None and elem_type != selected_element_type:
            continue
        nanovdb_models.append(nanovdb.get_field_model(dtype))
    field_models.extend(nanovdb_models)

    for spec in field_models_config.get("collection", []):
        inner_model = _resolve_inner_spec(spec, array, nanovdb, collection, vector_reduced, selection)
        field_models.append(collection.get_field_model(inner_model))

    # vector_reduced always outputs scalars (length=1); skip when caller filters for length>1.
    if selected_length is None or selected_length == 1:
        for entry in field_models_config.get("vector_reduced", []):
            inner_spec, reduction, *reduction_args = entry
            inner_model = _resolve_inner_spec(inner_spec, array, nanovdb, collection, vector_reduced, selection)
            inner_scalar = dav_utils.get_scalar_dtype(type(inner_model.FieldAPI.zero()))
            if selected_element_type is not None and inner_scalar != selected_element_type:
                continue
            if reduction == "component":
                field_models.append(vector_reduced.get_field_model_vector_reduced(inner_model, component=reduction_args[0]))
            elif reduction == "magnitude":
                field_models.append(vector_reduced.get_field_model_vector_reduced(inner_model, magnitude=True))

    for entry in field_models_config.get("selection", []):
        inner_spec, mode = entry
        inner_model = _resolve_inner_spec(inner_spec, array, nanovdb, collection, vector_reduced, selection)
        inner_dtype = type(inner_model.FieldAPI.zero())
        inner_length = dav_utils.get_vector_length(inner_dtype)
        inner_scalar = dav_utils.get_scalar_dtype(inner_dtype)
        if selected_length is not None and inner_length != selected_length:
            continue
        if selected_element_type is not None and inner_scalar != selected_element_type:
            continue
        if mode == "subrange":
            field_models.append(selection.get_field_model_subrange(inner_model))
        elif mode == "indexed_subset":
            field_models.append(selection.get_field_model_indexed_subset(inner_model))

    return field_models


def get_vec3_field_models(*, specialization: str = None):
    """Get a list of all field models with vec3 element type."""
    return get_field_models(selected_length=3, specialization=specialization)


def get_data_models(*, specialization: str = None):
    """Get a list of all data models we want to compile AOT.

    Args:
        specialization: Optional dot-notation path into the configuration to select a specific
            set of data models. For example, ``"operators.advection.seeds"`` navigates to
            ``configuration["operators"]["advection"]["seeds"]["data_models"]``.
            If None, uses the top-level ``configuration["data_models"]``.
    """

    data_models = []

    if specialization is None:
        data_models_config = configuration.get("data_models", {})
    else:
        keys = specialization.split(".")
        # Walk down the path, collecting each node along the way
        nodes = [configuration]
        node = configuration
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                break
            node = node[key]
            nodes.append(node)
        # Search from most specific to least specific for a "data_models" key
        data_models_config = {}
        for candidate in reversed(nodes):
            if isinstance(candidate, dict) and "data_models" in candidate:
                data_models_config = candidate["data_models"]
                break

    for package_name, package_config in data_models_config.items():
        if package_config is None:
            continue
        for module_name, module_config in package_config.items():
            if module_config is None:
                continue
            module = importlib.import_module(f"dav.data_models.{package_name}.{module_name}")
            if isinstance(module_config, list):
                for config in module_config:
                    data_models.append(module.get_data_model(config))
            else:
                data_models.append(module.get_data_model())

    return data_models


def get_operators():
    """Returns a list of operator names to compile AOT."""
    return list(configuration.get("operators", {}).keys())
