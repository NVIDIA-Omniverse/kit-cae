# DataSet Operator Commands

Kit CAE enables developers to add new schemas for file formats and data models. The built-in schemas beyond `CaeFieldArray` and `CaeDataSet`
serve as examples of what such schemas may look like. For existing Kit CAE functionality to work with custom data models, we need a plugin/callback
mechanism for data model-aware processing. For example, when voxelization of a dataset is needed, we require a mechanism that allows code requesting
voxelized data to do so in a data-model-agnostic fashion from a `CaeDataSet`.

## Overview

Kit CAE uses Omniverse's Command system (`omni.kit.commands`) to achieve this functionality. Commands provide a mechanism
for extensions to define new subclasses of `omni.kit.commands.Command` that can be executed using a central `omni.kit.commands.execute()` call.

Kit CAE defines a set of `Command` subclasses called **Operator Commands**. These cover a range of data transformation
operations needed to support the data processing and visualization functionality exposed in Kit CAE. Extensions can register
subclasses based on these Operator Commands with the Command system to provide data model-specific implementations.

### Command Resolution

For example, the operator command `ComputeBounds` computes bounding boxes for datasets. To provide specialization for a
`CaeDataSet` primitive that has the `CaeSidsUnstructuredAPI` applied, you simply need to register a command named `CaeSidsUnstructuredComputeBounds`.

When a request is made to execute an operator command on any primitive, the implementation uses the following resolution approach:

1. **API Schema Resolution**: For each applied API schema, a command name is constructed using the API schema's name (without the `API` suffix).
   This forms the prefix, which is then added to the operator command class name. For example, if the operator command is `ComputeBounds`
   being invoked on a `CaeDataSet` primitive with `CaeSidsUnstructuredAPI` applied, the command name tested will be `CaeSidsUnstructuredComputeBounds`.
   If a command with that name has not been registered with the Command system, the implementation moves on to the next applied API schema until
   all applied API schemas are tried.

2. **Type Hierarchy Resolution**: If none of the applied schema checks work, the same process is performed using the
   primitive type name itself, moving up the USD primitive type hierarchy. For a `CaeDataSet` primitive, the implementation
   will try `CaeDataSetComputeBounds`, `TypedComputeBounds`, and finally `SchemaBaseComputeBounds` as the last resort.

### Command Interface

Each operator command class has an `invoke` class method that documents the arguments and return type of that command.
The `invoke()` method throws exceptions on errors. When the return value is an array, we return a `FieldArrayLike`, which
can represent GPU or CPU arrays. This enables development of operator implementations that are not required to explicitly
perform data transfers between device and host before returning results. Device transfer can be deferred until absolutely
necessary when the result is consumed.

## Core Operator Commands (`omni.cae.data`)

The `omni.cae.data.commands` module defines several core operator command types:

### ComputeBounds

Computes the physical data bounds of a dataset primitive. The bounds are returned as a `Gf.Range3d`.

**Usage:**
```python
from omni.cae.data.commands import ComputeBounds
from pxr import Gf

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    bds: Gf.Range3d = await ComputeBounds.invoke(datasetPrim, timeCode=timeCode)
    if bds.IsEmpty():
        logger.info("empty bounds!")
```

### ComputeIJKExtents

Similar to `ComputeBounds`, but instead of world bounds, this computes structured `IJK` extents for the dataset primitive.
The command arguments allow the invoker to specify maximum dimensions along each axis and/or a region of interest
that can be used to further qualify the extents.

**Usage:**
```python
from omni.cae.data import IJKExtents
from omni.cae.data.commands import ComputeIJKExtents
from pxr import Gf

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    exts: IJKExtents = await ComputeIJKExtents.invoke(
        datasetPrim,
        max_dims=(128, 128, 128),
        roi=Gf.Range3d((0.0, 0.0, 0.0), (50.5, 10.0, 10.0)),  # or None
        timeCode=timeCode
    )
    if exts.IsEmpty():
        logger.info("empty extents!")
```

### ConvertToPointCloud

Converts a dataset to a standard point cloud representation. The command arguments allow optional selection of field arrays
to load, specified by `field:<name>` relationships on the dataset primitive. Since field arrays can be associated with
cell centers instead of vertices in the dataset, implementations of this command are expected to perform necessary
transformations to convert cell-centered fields to vertex-centered fields.

The returned result (`PointCloud`) intentionally mirrors a subset of attributes on `UsdGeom.Points` primitives. Data arrays
in the result are provided as `FieldArrayLike` and can represent both GPU or CPU hosted arrays. Use
`omni.cae.data.array_utils.as_numpy_array()` to reliably access a CPU hosted array if needed.
`PointCloud` provides a convenience method `PointCloud.numpy()` to convert all arrays to NumPy (with device-to-host copying if necessary).

**Usage:**
```python
from omni.cae.data.commands import ConvertToPointCloud, PointCloud

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    result: PointCloud = await ConvertToPointCloud.invoke(
        datasetPrim,
        fields=["Temp", "Pressure"],  # or []
        timeCode=timeCode
    )

    logger.info("Points shape: %s", result.coords.shape)
    logger.info("Num fields: %d", len(result.fields))
    logger.info("Temp shape: %s", result.fields["Temp"].shape)
    logger.info("Pressure shape: %s", result.fields["Pressure"].shape)
```

### ConvertToMesh

Converts a dataset to a surface mesh representation. This is similar to `ConvertToPointCloud`,
except the returned value is a `Mesh` suitable for setting up attributes on a `UsdGeom.Mesh` primitive.

**Usage:**
```python
from omni.cae.data.commands import ConvertToMesh, Mesh

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    result: Mesh = await ConvertToMesh.invoke(
        datasetPrim,
        fields=["Temp"],
        timeCode=timeCode
    )
```

### Voxelize

Voxelizes a dataset into a volumetric representation.

**Usage:**
```python
from omni.cae.data.commands import Voxelize
from omni.cae.data.types import Range3i
import warp as wp

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    result: wp.Volume = await Voxelize.invoke(
        datasetPrim,
        fields=["Temp"],
        bbox=Range3i((0, 0, 0), (128, 128, 128)),  # or None
        voxel_size=0.5,
        device_ordinal=0,
        timeCode=timeCode
    )
```

### GenerateStreamlines

Advects particles over a vector field and returns the streamlines. The attributes on the returned `Streamlines` object are aligned with `UsdGeom.BasisCurves`.

**Usage:**
```python
from omni.cae.data.commands import Streamlines, GenerateStreamlines

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    result: Streamlines = await GenerateStreamlines.invoke(
        datasetPrim,
        seeds,
        velocity_fields=["VelocityX", "VelocityY", "VelocityZ"],
        color_field="Temp",
        dX=0.1,
        maxLength=100,
        timeCode=timeCode
    )
```

## VTK Operator Commands (`omni.cae.vtk`)

The `omni.cae.vtk.commands` module defines VTK-specific operator command types used by algorithms that require VTK for data processing.

### ConvertToVTKDataSet

Converts any dataset to a VTK dataset. If you want to introduce a new data model and want it to work with VTK algorithms,
you must implement a specialization for this operator. The signature is similar to `ConvertToMesh` with one additional
parameter, `forcePointData`, which provides a convenience capability to internally convert all cell-centered data to point-centered data.

**Usage:**
```python
from omni.cae.vtk.commands import ConvertToVTKDataSet
from vtkmodules.numpy_interface import dataset_adapter as dsa

async def coroutine():
    # ...
    datasetPrim: Usd.Prim = ...
    timeCode: Usd.TimeCode = ...
    result: dsa.DataSet = await ConvertToVTKDataSet.invoke(
        datasetPrim,
        fields=["Temp"],
        forcePointData=False,
        timeCode=timeCode
    )
```

## IndeX Operator Commands (`omni.cae.index`)

The `omni.cae.index.commands` module defines NVIDIA IndeX-specific operator command types used by IndeX importers or
compute tasks to perform necessary data transformation for consumption by IndeX.

### CreateIrregularVolumeSubset

Converts any dataset to an `IIrregular_volume_subset` used for unstructured grid volume rendering. This command is
called by IndeX code, so applications will rarely need to call `CreateIrregularVolumeSubset.invoke(...)` directly.
Instead, here's how to implement and populate an `IIrregular_volume_subset`.

**Implementation Example:**
```python
from omni.cae.index.commands import CreateIrregularVolumeSubset
from omni.cae.index.bindings import (
    IIrregular_volume_subset, Mesh_parameters, Mesh_storage,
    Attribute_parameters, Attribute_storage, Attribute_affiliation,
    Attribute_type, BBox_float32
)
import numpy as np

class ExampleSchemaTypeNameCreateIrregularVolumeSubset(CreateIrregularVolumeSubset):

    async def do(self) -> None:
        dataset: Usd.Prim = self.dataset  # the dataset primitive
        fields: list[str] = self.field  # list of field names to volume render
        bbox: BBox_float32 = self.bbox  # bounding box to limit the import conversion
        subset: IIrregular_volume_subset = self.subset  # the instance to populate

        # 1. Populate Mesh_parameters; this tells IndeX how much memory it needs to allocate.
        # With IIrregular_volume_subset, cells are defined as a collection of triangular
        # or quadrangular faces.
        params = Mesh_parameters()
        params.nb_vertices = ...  # number of vertices
        params.nb_cells = ...     # number of cells
        params.nb_faces = ...     # number of faces
        params.nb_face_vtx_indices = ...    # number of face vertex indices
        params.nb_cell_face_indices = ...   # number of cell face indices

        # 2. Allocate storage
        storage: Mesh_storage = subset.generate_mesh_storage(params)

        # 3. Once storage is allocated, populate the actual data in the allocated memory.

        # Vertex coordinates: x-y-z coordinates for the vertices
        s_verts: np.ndarray = storage.get_vertices(params)
        assert (s_verts.dtype == np.float32 and s_verts.ndim == 2 and
                s_verts.shape[1] == 3 and s_verts.shape[0] == params.nb_vertices)
        # ... populate vertex data ...

        # Cells: specified as (nb_faces, start_face_index) for each cell where:
        # - nb_faces: number of faces forming this cell
        # - start_face_index: starting index into the cell-face index array
        s_cells: np.ndarray = storage.get_cells(params)
        assert (s_cells.dtype == np.uint32 and s_cells.ndim == 2 and
                s_cells.shape[1] == 2 and s_cells.shape[0] == params.nb_cells)
        # ... populate cell data ...

        # Cell face indices array: 0-based indices for `faces` array that provides face definitions
        s_cell_face_indices: np.ndarray = storage.get_cell_face_indices(params)
        assert (s_cell_face_indices.dtype == np.uint32 and s_cell_face_indices.ndim == 1 and
                s_cell_face_indices.shape[0] == params.nb_cell_face_indices)
        # ... populate cell face indices ...

        # Faces: specified as (nb_vertices, start_vertex_index) for each face where:
        # - nb_vertices: number of vertices in the face
        # - start_vertex_index: starting index into the face-vertex index array
        s_faces: np.ndarray = storage.get_faces(params)
        assert (s_faces.dtype == np.uint32 and s_faces.ndim == 2 and
                s_faces.shape[1] == 2 and s_faces.shape[0] == params.nb_faces)
        # ... populate face data ...

        # Face vertex indices array: 0-based indices for `vertices` array that comprise the faces
        s_face_vtx_indices: np.ndarray = storage.get_face_vtx_indices(params)
        assert (s_face_vtx_indices.dtype == np.uint32 and s_face_vtx_indices.ndim == 1 and
                s_face_vtx_indices.shape[0] == params.nb_face_vtx_indices)
        # ... populate face vertex indices ...
```

## Best Practices

When implementing operator commands for your custom data models:

1. **Follow Naming Conventions**: Use the pattern `{SchemaName}{OperatorCommand}` for your command classes
2. **Handle Errors Gracefully**: Use appropriate exception handling and logging
3. **Optimize for Performance**: Consider GPU/CPU memory management and avoid unnecessary data transfers
4. **Document Your Implementation**: Provide clear documentation for your custom operator commands
5. **Test Thoroughly**: Ensure your implementations work correctly with various data sizes and edge cases