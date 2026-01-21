# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CGNS-based I/O utilities using pyCGNS."""

import logging
from pathlib import Path
from typing import Union

import warp as wp
from dav.dataset import Dataset

logger = logging.getLogger(__name__)

# Type alias for CGNS zone data - dict of element blocks
CGNSZone = dict[str, Dataset]


def _compute_start_offsets_for_mixed(connectivity, num_cells, device):
    """Compute start offsets for MIXED element connectivity when missing.

    CGNS SIDS requires start offsets for MIXED elements, but some datasets
    (like StaticMixer.cgns) may have them missing. This computes them on-the-fly.

    Args:
        connectivity: Element connectivity array (includes element types)
        num_cells: Number of cells in the section
        device: Device to create the array on

    Returns:
        wp.array: Start offsets array of size (num_cells + 1)
    """
    import numpy as np

    # Element type to vertex count mapping
    vertex_counts = {
        3: 2,  # BAR_2
        5: 3,  # TRI_3
        7: 4,  # QUAD_4
        10: 4,  # TETRA_4
        12: 5,  # PYRA_5
        14: 6,  # PENTA_6
        17: 8,  # HEXA_8
    }

    # Compute offsets on CPU (this is a sequential operation)
    conn_np = connectivity if isinstance(connectivity, np.ndarray) else connectivity.numpy()
    offsets = np.zeros(num_cells + 1, dtype=np.int32)

    pos = 0
    for i in range(num_cells):
        offsets[i] = pos

        # Read element type from connectivity
        elem_type = int(conn_np[pos])

        # Get number of vertices for this element type
        num_verts = vertex_counts.get(elem_type, 0)
        if num_verts == 0:
            raise ValueError(f"Unsupported element type {elem_type} in MIXED connectivity at position {pos}")

        # Advance position: 1 (for element type) + num_verts
        pos += 1 + num_verts

    offsets[num_cells] = pos

    return wp.array(offsets, dtype=wp.int32, device=device)


def _extract_flow_solutions(zone_node, zone_name: str, num_points: int, device):
    """Extract flow solution fields from a CGNS zone.

    Args:
        zone_node: CGNS zone node
        zone_name: Name of the zone (for error messages)
        num_points: Number of points in the grid (for validation)
        device: Device to create fields on

    Returns:
        dict: Dictionary mapping field names to Field objects

    Note:
        Only supports Vertex and CellCenter grid locations.
        Warns and skips FaceCenter and other grid locations.

        Automatically detects vector components (e.g., VelocityX, VelocityY, VelocityZ)
        and combines them into a single Structure-of-Arrays (SoA) field.
    """
    import re

    import numpy as np
    from dav.field import Field
    from dav.fields import AssociationType

    fields = {}

    # Find all FlowSolution nodes
    flow_solution_nodes = [node for node in zone_node[2] if node[3] == "FlowSolution_t"]

    if not flow_solution_nodes:
        return fields  # No flow solutions found

    for flow_sol_node in flow_solution_nodes:
        flow_sol_name = flow_sol_node[0]

        # Find GridLocation to determine association (Vertex or CellCenter)
        grid_location = None
        for node in flow_sol_node[2]:
            if node[0] == "GridLocation":
                # GridLocation value is stored as a string (can be bytes or array of bytes)
                if node[1] is not None and len(node[1]) > 0:
                    # Handle both string and byte array formats
                    if isinstance(node[1], bytes):
                        grid_location = node[1].decode()
                    elif isinstance(node[1], str):
                        grid_location = node[1]
                    elif hasattr(node[1], "__iter__") and len(node[1]) > 0:
                        # Array of bytes - join them
                        try:
                            grid_location = "".join(chr(b) if isinstance(b, int) else b.decode() for b in node[1])
                        except Exception as _:
                            grid_location = str(node[1])
                    else:
                        grid_location = str(node[1])
                break

        # Default to Vertex if no GridLocation specified (CGNS convention)
        if grid_location is None:
            grid_location = "Vertex"

        # Determine association type
        if grid_location == "Vertex":
            association = AssociationType.VERTEX
        elif grid_location == "CellCenter":
            association = AssociationType.CELL
        else:
            logger.warning(
                "Skipping FlowSolution '%s' with unsupported GridLocation '%s'", flow_sol_name, grid_location
            )
            logger.warning(
                "         DAV currently only supports Vertex and CellCenter. Face-centered and other locations are not yet implemented."
            )
            continue

        # First pass: collect all scalar fields
        scalar_fields = {}
        for node in flow_sol_node[2]:
            if node[3] == "DataArray_t":
                field_name = node[0]
                field_data = node[1]

                if field_data is None or len(field_data) == 0:
                    logger.warning("Skipping empty field '%s' in FlowSolution '%s'", field_name, flow_sol_name)
                    continue

                # Only process scalar (1D) and vector (Nx3) fields
                if len(field_data.shape) == 1:
                    scalar_fields[field_name] = field_data
                elif len(field_data.shape) == 2 and field_data.shape[1] == 3:
                    # Already a vector field, add directly
                    data_array = wp.array(field_data.astype(np.float32), dtype=wp.vec3f, device=device)
                    field = Field.from_array(data_array, association)

                    # Add field with unique name
                    if len(flow_solution_nodes) > 1:
                        # full_field_name = f"{flow_sol_name}_{field_name}"
                        full_field_name = field_name
                    else:
                        full_field_name = field_name

                    fields[full_field_name] = field
                    logger.info(
                        "  Loaded field '%s': %s, shape=%s (AoS vector)",
                        full_field_name,
                        association.name,
                        field_data.shape,
                    )
                else:
                    logger.warning("Skipping field '%s' with unsupported shape %s", field_name, field_data.shape)
                    logger.warning(
                        "         DAV currently only supports scalar (1D) and 3-component vector (Nx3) fields."
                    )

        # Second pass: detect and combine vector components
        # Patterns to detect: X/Y/Z, x/y/z, 0/1/2, _x/_y/_z, _X/_Y/_Z, _0/_1/_2
        vector_component_patterns = [
            (r"(.+)[Xx]$", r"(.+)[Yy]$", r"(.+)[Zz]$"),  # VelocityX, VelocityY, VelocityZ
            (r"(.+)_[Xx]$", r"(.+)_[Yy]$", r"(.+)_[Zz]$"),  # Velocity_X, Velocity_Y, Velocity_Z
            (r"(.+)_0$", r"(.+)_1$", r"(.+)_2$"),  # Velocity_0, Velocity_1, Velocity_2
            (r"(.+)0$", r"(.+)1$", r"(.+)2$"),  # Velocity0, Velocity1, Velocity2
            (r"(.+)\[0\]$", r"(.+)\[1\]$", r"(.+)\[2\]$"),  # Velocity[0], Velocity[1], Velocity[2]
        ]

        processed_fields = set()

        for x_pattern, y_pattern, z_pattern in vector_component_patterns:
            # Find all potential X components
            for field_name in list(scalar_fields.keys()):
                if field_name in processed_fields:
                    continue

                x_match = re.match(x_pattern, field_name)
                if x_match:
                    base_name = x_match.group(1)

                    # Look for corresponding Y and Z components
                    y_name = None
                    z_name = None

                    for other_name in scalar_fields.keys():
                        if other_name in processed_fields:
                            continue
                        y_match = re.match(y_pattern, other_name)
                        if y_match and y_match.group(1) == base_name:
                            y_name = other_name
                        z_match = re.match(z_pattern, other_name)
                        if z_match and z_match.group(1) == base_name:
                            z_name = other_name

                    # If we found all three components, create a SoA vector field
                    if y_name and z_name:
                        x_data = scalar_fields[field_name].astype(np.float32)
                        y_data = scalar_fields[y_name].astype(np.float32)
                        z_data = scalar_fields[z_name].astype(np.float32)

                        # Verify all components have the same length
                        if len(x_data) == len(y_data) == len(z_data):
                            # Create SoA field
                            x_array = wp.array(x_data, dtype=wp.float32, device=device)
                            y_array = wp.array(y_data, dtype=wp.float32, device=device)
                            z_array = wp.array(z_data, dtype=wp.float32, device=device)

                            field = Field.from_arrays([x_array, y_array, z_array], association)

                            # Add field with base name
                            if len(flow_solution_nodes) > 1:
                                # full_field_name = f"{flow_sol_name}_{base_name}"
                                full_field_name = base_name
                            else:
                                full_field_name = base_name

                            fields[full_field_name] = field
                            logger.info(
                                "  Loaded field '%s': %s, shape=(%d, 3) (SoA vector from %s, %s, %s)",
                                full_field_name,
                                association.name,
                                len(x_data),
                                field_name,
                                y_name,
                                z_name,
                            )

                            # Mark these fields as processed
                            processed_fields.add(field_name)
                            processed_fields.add(y_name)
                            processed_fields.add(z_name)
                        else:
                            logger.warning(
                                "Vector components %s, %s, %s have mismatched lengths", field_name, y_name, z_name
                            )

        # Third pass: add remaining scalar fields that weren't part of vectors
        for field_name, field_data in scalar_fields.items():
            if field_name in processed_fields:
                continue

            # Create scalar field
            data_array = wp.array(field_data.astype(np.float32), dtype=wp.float32, device=device)
            field = Field.from_array(data_array, association)

            # Add field with unique name
            if len(flow_solution_nodes) > 1:
                # full_field_name = f"{flow_sol_name}_{field_name}"
                full_field_name = field_name
            else:
                full_field_name = field_name

            fields[full_field_name] = field
            logger.info(
                "  Loaded field '%s': %s, shape=%s (scalar)", full_field_name, association.name, field_data.shape
            )

    return fields


def _create_nface_n_dataset(
    section_name, nface_connectivity, nface_elem_range, nface_start_offset, ngon_sections, coords_array, device
):
    """Create an NFACE_n polyhedral dataset by combining NFACE_n and NGON_n sections.

    Args:
        section_name: Name of the NFACE_n section
        nface_connectivity: NFACE_n connectivity (cell -> face IDs)
        nface_elem_range: NFACE_n element range [start, end] (1-based, inclusive)
        nface_start_offset: NFACE_n start offsets
        ngon_sections: Dict of NGON_n sections {name: {connectivity, element_range, element_start_offset}}
        coords_array: Grid coordinates (wp.array of wp.vec3f)
        device: Device to create arrays on

    Returns:
        Dataset: Dataset using nface_n data model

    Raises:
        ValueError: If no NGON_n sections found or incompatible data
    """
    import numpy as np
    from dav.data_models.sids import nface_n, unstructured
    from dav.dataset import Dataset

    if not ngon_sections:
        raise ValueError(f"NFACE_n section '{section_name}' requires NGON_n sections, but none found")

    nface_handle = nface_n.DatasetHandle()

    # Create NFACE_n block (stores cell -> face connectivity)
    nface_handle.nface_n_block.grid_coords = coords_array
    nface_handle.nface_n_block.element_type = wp.int32(23)  # NFACE_n
    nface_handle.nface_n_block.hex_is_axis_aligned = False
    nface_handle.nface_n_block.element_range = wp.vec2i(int(nface_elem_range[0]), int(nface_elem_range[1]))
    nface_handle.nface_n_block.element_connectivity = wp.array(nface_connectivity, dtype=wp.int32, device=device)

    if not isinstance(nface_start_offset, wp.array):
        nface_start_offset = wp.array(nface_start_offset, dtype=wp.int32, device=device)
    nface_handle.nface_n_block.element_start_offset = nface_start_offset
    nface_handle.nface_n_block.cell_bvh_id = wp.uint64(0)

    # Create NGON_n blocks (store face -> vertex connectivity)
    ngon_datasets = []
    ngon_element_range_starts = []

    # Sort NGON sections by element range start to build binary search array
    sorted_ngon_sections = sorted(ngon_sections.items(), key=lambda item: item[1]["element_range"][0])

    for ngon_name, ngon_data in sorted_ngon_sections:
        ngon_handle = unstructured.DatasetHandle()
        ngon_handle.grid_coords = coords_array
        ngon_handle.element_type = wp.int32(22)  # NGON_n
        ngon_handle.hex_is_axis_aligned = False

        elem_range = ngon_data["element_range"]
        ngon_handle.element_range = wp.vec2i(int(elem_range[0]), int(elem_range[1]))
        ngon_handle.element_connectivity = wp.array(ngon_data["connectivity"], dtype=wp.int32, device=device)

        start_offset = ngon_data["element_start_offset"]
        if not isinstance(start_offset, wp.array):
            start_offset = wp.array(start_offset, dtype=wp.int32, device=device)
        ngon_handle.element_start_offset = start_offset
        ngon_handle.cell_bvh_id = wp.uint64(0)

        ngon_element_range_starts.append(int(elem_range[0]))
        ngon_datasets.append(Dataset(data_model=unstructured.DataModel, handle=ngon_handle, device=device))

    # Create NFACE_n dataset handle
    nface_handle.ngon_n_element_range_starts = wp.array(
        np.array(ngon_element_range_starts, dtype=np.int32), dtype=wp.int32, device=device
    )
    nface_handle.ngon_n_blocks = wp.array(
        [ngon.handle for ngon in ngon_datasets], dtype=unstructured.DatasetHandle, device=device
    )

    print(
        f"nface_n.element_range: {nface_handle.nface_n_block.element_range[0]}, {nface_handle.nface_n_block.element_range[1]}"
    )

    # Create Dataset with nface_n data model
    # pass ngon_datasets as kwargs so that they don't get garbage collected
    dataset = Dataset(
        data_model=nface_n.DataModel, handle=nface_handle, device=device, ngon_datasets=tuple(ngon_datasets)
    )

    return dataset


def _find_zone(tree, zone_name: Union[str, None] = None):
    """Find and return a zone node from the CGNS tree.

    Args:
        tree: CGNS tree structure
        zone_name: Optional zone path (e.g., "Base/Zone1" or "Zone1").
                   If None, returns the first zone in the first base.

    Returns:
        tuple: (zone_node, zone_name, base_name)

    Raises:
        ValueError: If no bases/zones found or requested zone not found
    """
    # Get the first CGNSBase_t if exists
    bases = [node for node in tree[2] if node[3] == "CGNSBase_t"]
    if not bases:
        raise ValueError("No CGNSBase_t node found in CGNS file")

    base_node = bases[0]
    base_name = base_node[0]

    # Get zones from the base
    zones = [node for node in base_node[2] if node[3] == "Zone_t"]
    if not zones:
        raise ValueError(f"No Zone_t nodes found in base '{base_name}'")

    # Select zone
    if zone_name is None:
        # Use first zone
        zone_node = zones[0]
        actual_zone_name = zone_node[0]
        logger.info("No zone specified, using first zone: %s/%s", base_name, actual_zone_name)
        return (zone_node, actual_zone_name, base_name)

    # Parse zone path (can be "ZoneName" or "Base/ZoneName")
    if "/" in zone_name:
        parts = zone_name.split("/")
        requested_base = parts[0]
        requested_zone = parts[-1]

        # Check if base matches
        if requested_base != base_name:
            available_bases = [b[0] for b in bases]
            raise ValueError(f"Requested base '{requested_base}' not found. Available bases: {available_bases}")
    else:
        requested_zone = zone_name

    # Find the zone
    zone_node = None
    for z in zones:
        if z[0] == requested_zone:
            zone_node = z
            break

    if zone_node is None:
        available_zones = [f"{base_name}/{z[0]}" for z in zones]
        raise ValueError(f"Zone '{requested_zone}' not found in base '{base_name}'. Available zones: {available_zones}")

    actual_zone_name = zone_node[0]
    return (zone_node, actual_zone_name, base_name)


def read(
    filename: Union[str, Path], zone: Union[str, None] = None, device: Union[str, wp.context.Device, None] = None
) -> CGNSZone:
    """Load a CGNS zone using CGNS MLL and convert to DAV datasets.

    Reads a CGNS file and extracts the specified zone, returning a dictionary
    of datasets for each element block in that zone. Uses the SIDS data model
    to preserve the native CGNS structure without conversion.

    Args:
        filename: Path to the CGNS file to load (.cgns extension).
        zone: Name of the zone to read (e.g., "Base/Zone1"). If None, reads the
              first zone found in the first base.
        device: Device to create the datasets on. If None, uses Warp's current device context.
                Can be a string like "cuda:0" or a warp.context.Device.

    Returns:
        Dictionary mapping element block names to Dataset objects. Each dataset
        uses the SIDS data model. For example:

        .. code-block:: python

            {
                "B1_P3": Dataset(...),  # First element block
                "B2_P3": Dataset(...),  # Second element block
                ...
            }

    Raises:
        FileNotFoundError: If the CGNS file doesn't exist.
        ValueError: If the zone is not found or the file format is invalid.
        ImportError: If pyCGNS is not installed.

    Examples:
        >>> import dav
        >>> from dav.io.cgns import read
        >>>
        >>> # Load the first zone automatically
        >>> cgns_zone = read("StaticMixer.cgns")
        >>>
        >>> # Load a specific zone
        >>> cgns_zone = read("StaticMixer.cgns", zone="Base/Zone1")
        >>>
        >>> # Or just specify zone name if base name is known
        >>> cgns_zone = read("simulation.cgns", zone="Zone_1")
        >>>
        >>> # Access a specific element block
        >>> volume_block = cgns_zone.get("B1_P3")
        >>> print(f"Block has {volume_block.get_num_cells()} cells")
        >>>
        >>> # Load on GPU
        >>> cgns_zone = read("simulation.cgns", device="cuda:0")
        >>>
        >>> # Process all blocks in a zone
        >>> for block_name, block_dataset in cgns_zone.items():
        ...     print(f"Processing block: {block_name}")
        ...     # Apply operators to each block
        ...     result = dav.operators.bounds.compute(block_dataset)

    Note:
        This function preserves the CGNS SIDS data model. If you need to work
        with CGNS data in VTK format, use VTK's vtkCGNSReader with
        ``dav.data_models.vtk.utils.vtk_to_dataset`` instead.
    """
    try:
        import CGNS.MAP as cgm  # type: ignore
    except ImportError as e:
        raise ImportError("pyCGNS is required to use dav.io.cgns. Install it with: pip install pyCGNS") from e

    import numpy as np

    # Use Warp's current device if not specified
    if device is None:
        device = wp.get_device()

    # Convert to Path object
    filepath = Path(filename)

    # Check if file exists
    if not filepath.exists():
        raise FileNotFoundError(f"CGNS file not found: {filepath}")

    # Check file extension
    if filepath.suffix.lower() != ".cgns":
        raise ValueError(
            f"Expected .cgns file extension, got: {filepath.suffix}. "
            "If this is a valid CGNS file with a different extension, "
            "rename it to use the .cgns extension."
        )

    # Load the CGNS file using MLL
    try:
        (tree, links, paths) = cgm.load(str(filepath))
    except Exception as e:
        raise ValueError(
            f"Failed to load CGNS file '{filepath}'. The file may be corrupted or in an unsupported format. Error: {e}"
        ) from e

    # Find the requested zone (or first zone if not specified)
    try:
        zone_node, zone_name, base_name = _find_zone(tree, zone)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Error accessing CGNS structure: {e}") from e

    # Import SIDS data models
    from dav.data_models.sids import unstructured
    from dav.dataset import Dataset

    # Parse the zone and extract element blocks
    result = {}

    # Track NGON_n sections separately to associate with NFACE_n later
    ngon_sections = {}  # dict mapping section_name -> (connectivity, element_range, start_offset)

    try:
        # Get grid coordinates
        grid_coords_node = None
        for node in zone_node[2]:
            if node[3] == "GridCoordinates_t":
                grid_coords_node = node
                break

        if grid_coords_node is None:
            raise ValueError(f"No GridCoordinates_t found in zone '{zone_name}'")

        # Extract X, Y, Z coordinates
        coords_dict = {}
        for coord_node in grid_coords_node[2]:
            if coord_node[3] == "DataArray_t":
                coord_name = coord_node[0]
                coord_data = coord_node[1]
                # Support both CGNS naming conventions: CoordinateX/Y/Z and CoordinatesX/Y/Z
                if coord_name in ["CoordinateX", "CoordinatesX", "Coordinate X", "X"]:
                    coords_dict["X"] = coord_data
                elif coord_name in ["CoordinateY", "CoordinatesY", "Coordinate Y", "Y"]:
                    coords_dict["Y"] = coord_data
                elif coord_name in ["CoordinateZ", "CoordinatesZ", "Coordinate Z", "Z"]:
                    coords_dict["Z"] = coord_data

        if len(coords_dict) != 3:
            raise ValueError(f"Expected 3 coordinate arrays, found {len(coords_dict)}")

        # Combine into vec3f array
        num_points = len(coords_dict["X"])
        coords = np.zeros((num_points, 3), dtype=np.float32)
        coords[:, 0] = coords_dict["X"].astype(np.float32)
        coords[:, 1] = coords_dict["Y"].astype(np.float32)
        coords[:, 2] = coords_dict["Z"].astype(np.float32)

        coords_array = wp.array(coords, dtype=wp.vec3f, device=device)

        # Get all element sections
        element_sections = [node for node in zone_node[2] if node[3] == "Elements_t"]

        if not element_sections:
            raise ValueError(f"No element sections found in zone '{zone_name}'")

        # Process each element section
        for elem_section in element_sections:
            section_name = elem_section[0]

            # Get element range
            elem_range_node = None
            for node in elem_section[2]:
                if node[0] == "ElementRange":
                    elem_range_node = node
                    break

            if elem_range_node is None:
                print(f"Warning: Skipping section '{section_name}' - no ElementRange found")
                continue

            elem_range = elem_range_node[1]  # [start, end] inclusive, 1-based

            # Get element type
            elem_type = elem_section[1][0] if len(elem_section[1]) > 0 else None
            if elem_type is None:
                print(f"Warning: Skipping section '{section_name}' - no element type")
                continue

            # Skip boundary/surface elements (we only want volume elements)
            # CGNS element types: TRI_3=5, QUAD_4=7, TETRA_4=10, PYRA_5=12, PENTA_6=14, HEXA_8=17, MIXED=20, NGON_n=22, NFACE_n=23
            volume_element_types = [10, 12, 14, 17, 20, 22, 23]  # TETRA, PYRA, PENTA, HEXA, MIXED, NGON_n, NFACE_n
            if elem_type not in volume_element_types:
                logger.info("Skipping section '%s' - not a volume element (type=%d)", section_name, elem_type)
                continue

            # Get connectivity data
            connectivity_node = None
            for node in elem_section[2]:
                if node[0] == "ElementConnectivity":
                    connectivity_node = node
                    break

            if connectivity_node is None:
                logger.warning("Skipping section '%s' - no ElementConnectivity", section_name)
                continue

            connectivity = connectivity_node[1].astype(np.int32)

            # Handle start offsets for MIXED, NGON_n, and NFACE_n element types
            element_start_offset = None
            if elem_type in [20, 22, 23]:  # MIXED, NGON_n, NFACE_n
                # Look for ElementStartOffset
                for node in elem_section[2]:
                    if node[0] == "ElementStartOffset":
                        element_start_offset = node[1].astype(np.int32)
                        break

                if element_start_offset is None:
                    if elem_type == 20:  # MIXED
                        # MIXED without start offsets - compute them
                        logger.warning("Section '%s' is MIXED but missing ElementStartOffset", section_name)
                        logger.warning(
                            "         Computing start offsets. Consider adding ElementStartOffset to the CGNS file."
                        )
                        num_cells = elem_range[1] - elem_range[0] + 1
                        element_start_offset = _compute_start_offsets_for_mixed(connectivity, num_cells, device)
                    else:
                        # NGON_n and NFACE_n require start offsets
                        elem_type_name = "NGON_n" if elem_type == 22 else "NFACE_n"
                        logger.warning(
                            "Skipping section '%s' - %s requires ElementStartOffset", section_name, elem_type_name
                        )
                        continue

            # Handle NGON_n sections - store for later association with NFACE_n
            if elem_type == 22:  # NGON_n
                ngon_sections[section_name] = {
                    "connectivity": connectivity,
                    "element_range": elem_range,
                    "element_start_offset": element_start_offset,
                }
                logger.info("Collected NGON_n section '%s': faces=%d", section_name, elem_range[1] - elem_range[0] + 1)
                continue  # Don't create dataset yet - wait for NFACE_n

            # Handle NFACE_n sections - create polyhedral dataset
            if elem_type == 23:  # NFACE_n
                # Create NFACE_n dataset by combining with NGON_n sections
                try:
                    nface_dataset = _create_nface_n_dataset(
                        section_name,
                        connectivity,
                        elem_range,
                        element_start_offset,
                        ngon_sections,
                        coords_array,
                        device,
                    )
                    result[section_name] = nface_dataset
                    logger.info(
                        "Loaded NFACE_n section '%s': cells=%d, points=%d, using %d NGON_n section(s)",
                        section_name,
                        elem_range[1] - elem_range[0] + 1,
                        num_points,
                        len(ngon_sections),
                    )
                    continue
                except Exception as e:
                    logger.error("Failed to create NFACE_n dataset for '%s': %s", section_name, e)
                    continue

            # Create SIDS unstructured dataset for standard element types (TETRA, HEXA, PYRA, PENTA, MIXED)
            ds_handle = unstructured.DatasetHandle()
            ds_handle.grid_coords = coords_array
            ds_handle.element_type = wp.int32(elem_type)
            ds_handle.hex_is_axis_aligned = False  # Conservative assumption
            ds_handle.element_range = wp.vec2i(int(elem_range[0]), int(elem_range[1]))
            ds_handle.element_connectivity = wp.array(connectivity, dtype=wp.int32, device=device)

            # Set element_start_offset (required for MIXED, NGON_n, NFACE_n)
            if element_start_offset is not None:
                if not isinstance(element_start_offset, wp.array):
                    element_start_offset = wp.array(element_start_offset, dtype=wp.int32, device=device)
                ds_handle.element_start_offset = element_start_offset
            else:
                ds_handle.element_start_offset = wp.empty(0, dtype=wp.int32, device=device)

            ds_handle.cell_bvh_id = wp.uint64(0)

            # Create Dataset
            dataset = Dataset(data_model=unstructured.DataModel, handle=ds_handle, device=device)

            # Add to result dictionary
            result[section_name] = dataset

            logger.info(
                "Loaded section '%s': type=%d, cells=%d, points=%d",
                section_name,
                elem_type,
                elem_range[1] - elem_range[0] + 1,
                num_points,
            )

        if not result:
            raise ValueError(f"No valid volume element sections found in zone '{zone_name}'")

        # Extract and add flow solution fields to all datasets
        # (All element sections share the same grid coordinates, so fields apply to all)
        flow_fields = _extract_flow_solutions(zone_node, zone_name, num_points, device)
        if flow_fields:
            logger.info("  Adding %d field(s) to all datasets in zone", len(flow_fields))
            for dataset in result.values():
                for field_name, field in flow_fields.items():
                    dataset.fields[field_name] = field

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse CGNS zone '{zone_name}'. Error: {e}") from e

    return result
