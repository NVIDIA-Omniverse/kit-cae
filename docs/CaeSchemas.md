# Omni CAE USD Schema

## TLDR;

This extension proposes a USD schema for scientific datasets in Computer-Aided Engineering and Simulation workflows. A
short summary of our design is as follows:

* `OmniCaeDataSet` is new prim type to represent scientific datasets. `OmniCaeDataSet` is similar to `UsdVolVolume` and
  exposes API for adding field relationships.
* `OmniCaeFieldArray` is a new prim type used to represent individual field arrays in datasets. This is similar to
  `UsdVolFieldBase`. Format specific subtypes can be defined for support different file formats or in-memory storage.
* Single-apply API schemas are used to add data model specific properties to `OmniCaeDataSet` prims. These API schemas
  help the USD applications to correctly interpret the data described by the `OmniCaeDataSet`.

`OmniCaeDataSet` and `OmniCaeFieldArray` would easily have been subtypes of `UsdVolVolume` and `UsdVolFieldBase`
respectively. However, that leads to ambiguity since strictly speaking a CAE dataset is not necessarily a 3D volume and could
easily be representing a 2D mesh, defined in USD using `UsdGeomMesh`. To avoid this confusion, we add these as separate types.

## Introduction

### Data Model

At its core, any scientific dataset is merely a collection of data arrays. Agreed upon conventions on how to interpret
those arrays helps us describe complex data structures using simple data arrays. **Data model** is simply a collection
of such conventions for a specific application or domain.

To understand this concept, consider USD's schema for representing 3D surface meshes using `UsdGeomMesh`.
The specification requires three arrays named `points`, `faceVertexIndices`, and `faceVertexCounts` of
defined type and form. If we have triangle mesh, for example, given this `UsdGeomMesh` specification, we can describe the
triangle mesh using the required arrays. Thus, we can say `UsdGeomMesh` schema defines a data model for representing
surface meshes. There are of course, arbitrarily many ways of defining surface meshes. For example, if our application
only works triangle meshes, we can simply the data model by only requiring the `points` array and implicitly treating
them as sequence of 3 points, each defining a triangle in counter-clockwise order. For our application, this then
becomes the data model. Applications adopt different data models based on their specific needs and trade offs.
Here, for example, this triangle-mesh data model trades flexibility for memory efficiency and simplicity when compared with
the `UsdGeomMesh` data model.

Scientific codes have adopted a wide variety of data models. We want to ensure that any and all of them can be expressed
in USD.

### File Formats

Data models help us interpret scientific data in memory. Similarly, file formats help us
save / restore scientific data to files. In some cases, data model and file format go hand in hand,
for example `CGNS` includes both a file format and a data model. More often than not, when one says
CGNS, they are referring to `.cgns` files. However, CGNS also includes a
`Standard Interface Data Structures (SIDS)` specification which forms the **data model**.

In our design, we explicitly decouple the file format from the data model. This decoupling gives us the flexibility to mix
and match the two with ease.

### Example

Here's a simple example representing an unstructured dataset stored in a CGNS file.

```usda
def Xform "World"
{
    def Xform "DataSetHex"
    {
        def CaeDataSet "ElementSet" (
            # application of the API tells application how to interpret the field arrays.
            # here, we're indicating this dataset is using CGNS's SIDS data model for
            # unstructured datasets.
            prepend apiSchemas = ["CaeSidsUnstructuredAPI"]
        )
        {
            # these are attributes introduced by the CaeSidsUnstructuredMeshAPI schema.
            token cae:sids:elementType = "HEXA_8"
            uint64 cae:sids:elementRangeStart = 1
            uint64 cae:sids:elementRangeEnd = 1025
            rel cae:sids:gridCoordinates = [<Arrays/CoordinatesX>, <Arrays/CoordinatesY>, <Arrays/CoordinatesZ>]
            rel cae:sids:elementConnectivity = <Arrays/Connectivity>

            # fields using to represent data arrays
            rel field:pressure = <Arrays/Pressure>
            rel field:temperature = <Arrays/Temperature>

            def Scope "Arrays"
            {
                # here `class` together with `specializes=` is used to avoid repeating the
                # common properties for all field arrays.
                class "StaticMixerFieldArray"
                {
                    asset[] fileNames = [@/tmp/staticMixer.cgns@]
                }
                def CaeCgnsFieldArray "Pressure" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "vertex"
                    string fieldPath = "/Base1/Zone1/FlowSolution/Pressure"
                }
                def CaeCgnsFieldArray "Temperature" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "vertex"
                    string fieldPath = "/Base1/Zone1/FlowSolution/Temperature"
                }
                def CaeCgnsFieldArray "CoordinatesX" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "vertex"
                    string fieldPath = "/Base1/Zone1/GridCoordinates/CoordinatesX"
                }
                def CaeCgnsFieldArray "CoordinatesY" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "vertex"
                    string fieldPath = "/Base1/Zone1/GridCoordinates/CoordinatesY"
                }
                def CaeCgnsFieldArray "CoordinatesZ" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "vertex"
                    string fieldPath = "/Base1/Zone1/GridCoordinates/CoordinatesZ"
                }
                def CaeCgnsFieldArray "Connectivity" ( specializes=<../StaticMixerFieldArray>)
                {
                    uniform token fieldAssociation = "none"
                    string fieldPath = "/Base1/Zone1/ElementSet1/ElementConnectivity"
                }
            }
        }
    }
}
```

## CAE Schemas

To support CAE datasets, we introduce two new base prim types: `CaeDataSet` and `CaeFieldArray`. `CaeDataSet` represents
a singular dataset or mesh. Scientific datasets often comprise of elemental datasets that are organized into hierarchies
or classified under groups to represents deferent components in the application domain.
`CaeDataSet` does not represent such composites, instead is intended for a single atomic dataset.
In USD, `CaeDataSet` is simply a container for `CaeFieldArray` prims specified as relationships named with `field:`
namespace. This notion of specifying fields is intentionally similar to `UsdVolVolume`. `CaeFieldArray`, is similar to
`UsdVolFieldBase` and is intended to represent individual data arrays that comprise the dataset.
Concrete subtypes can be added for `CaeFieldArray` for specific file formats and external data stores. Our initial
implementation includes two subtypes: `CaeCgnsFieldArray` and `CaeNumPyFieldArray` for representing data arrays stored
in CGNS (`*.cgns`) and NumPy (`*.npy`, `*.npz`) files.

Next, we need a means to specify the data model. In other words, we need a mechanism to indicate how to interpret
these arrays. For that, we use single-apply API schemas. To add support for a data model, one simply needs to introduce
an API schema for that data model and apply it to `CaeDataSet` prim. The attributes and relationships on the API schema
can then help applications understand how to interpret the field-arrays.
`CaeSidsUnstructuredAPI`, for example, is an API schema that can be used for any datasets that follows
the CGNS `Standard Interface Data Structures (SIDS)` data model for unstructured element sets aka `Element_t` nodes.

The schemas are organized into ten packages: one shared core package (`OmniCae`), one shared data-model package
(`OmniCaeSids`), and eight format-specific packages (`OmniCaeCgns`, `OmniCaeHdf5`, `OmniCaeNumPy`, `OmniCaeVtk`,
`OmniCaeEnSight`, `OmniCaeOpenFoam`, `OmniCaeTrimesh`, `OmniCaeNvdb`). Let's take a closer look at each.

### OmniCae Schemas

The `OmniCae` package defines the core schemas: `CaeDataSet`, `CaeFieldArray`, and the shared data-model API schemas
`CaePointCloudAPI`, `CaeMeshAPI`, `CaeDenseVolumeAPI`, and `CaeNanoVDBFieldArrayAPI`. Format-specific `CaeFieldArray`
subtypes (`CaeCgnsFieldArray`, `CaeHdf5FieldArray`, `CaeNumPyFieldArray`, etc.) each live in their own package rather
than in `OmniCae`, but are documented here for convenience alongside the core types they extend.

#### CaeDataSet

`CaeDataSet` schema describes a dataset composed of multiple fields. A field is a relationship pointing to a prim of
type `CaeFieldArray` or one of its subtypes. One can apply API schemas to a `CaeDataSet` prim to indicate
to applications the data model to use to interpret the field arrays.

```usda
# CaeDataSet schema definition
class CaeDataSet "CaeDataSet" (
    inherits = </Typed>
    customData = {
        string className = "DataSet"
    }
    doc = """A scientific dataset. A dataset is made up of any number of CaeFieldArray
             primitives bound together in this dataset. Each CaeFieldArray primitive is specified as a
             relationship with namespace prefix of "field"."""
)
{
}
```

#### CaeFieldArray

`CaeFieldArray` is the base type for all field arrays and is a schema
representing a single dataset field. This schema should be inherited to add
support for reading data arrays from various file formats. The schema definition is as follows:

```usda
# CaeFieldArray schema definition
class CaeFieldArray "CaeFieldArray" (
    inherits = </Typed>
    customData = {
        string className = "FieldArray"
    }
)
{
   asset[] fileNames = [] (
        doc = """Specifies the assets for the files. With multiple assets
            are specified then they are treated as spatial partitions of the same
            dataset. Temporal partitions may be specified by animating this attribute
            i.e. using time codes.
            """
        displayName = "File Names"
    )

    uniform token fieldAssociation = "none" (
        allowedTokens = [ "none", "vertex", "cell" ]
        doc = "Specifies the dataset element this field array is associated with."
        displayName = "Field Association"
    )
}
```

`fileNames` attribute is used to specify asset paths to data stored outside of USD.
It can be used to represent temporally and spatially partitioned files as illustrated by the following examples:

```usda
def CaeFieldArray "SingleFile"
{
  asset[] fileNames = [@data/test.cgns@]
}

def CaeFieldArray "PartitionedFiles"
{
  asset[] fileNames = [@data/test_0_2.pvtk@,
                       @data/test_1_2.pvtk@]
}

def CaeFieldArray "TemporalFileSequence"
{
  asset[] fileNames.timeSamples = {
    0:  [@data/test_0.cgns@],
    10: [@data/test_1.cgns@],
    20: [@data/test_2.cgns@]
  }
}

def CaeFieldArray "TemporalPartitionedFiles"
{
  asset[] fileNames.timeSamples = {
    0:  [@data/test_0_0_2.pvtk@, @data/test_0_1_2.pvtk@],
    10: [@data/test_10_0_2.pvtk@, @data/test_10_1_2.pvtk@],
    20: [@data/test_20_0_2.pvtk@, @data/test_20_1_2.pvtk@]
  }
}
```

`fieldAssociation` attribute can optionally specify which element type the field array is associated with.

#### CaeHdf5FieldArray

`CaeHdf5FieldArray` inherits from `CaeFieldArray` and is a concrete schema for field arrays
stored in a HDF5 dataset. Attribute `hdf5Path` can be used to specify the HDF5 dataset path.

```usda
# CaeHdf5FieldArray schema definition
class CaeHdf5FieldArray "CaeHdf5FieldArray" (
    inherits = </CaeFieldArray>
    customData = {
        string className = "Hdf5FieldArray"
    }
)
{
    string hdf5Path = "" (
        doc = "Specifies the path to the HDF5 dataset."
        displayName = "Hdf5 Path"
    )
}
```

#### CaeCgnsFieldArray

`CaeCgnsFieldArray` inherits from `CaeFieldArray` and is a concrete schema for field arrays
stored in a CGNS asset. Attribute `fieldPath` can be used to specify the CGNS node path to
the `DataArray_t` node containing the data.

```usda
# CaeCgnsFieldArray schema definition
class CaeCgnsFieldArray "CaeCgnsFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Defines a CGNS data array"
    customData = {
        string className = "CgnsFieldArray"
    }
)
{
    string fieldPath = "" (
        doc = "Specifies the path to the node in the CGNS database."
    )
}
```

#### CaeNumPyFieldArray

`CaeNumPyFieldArray` inherits from `CaeFieldArray` and is a concrete schema for field arrays
stored in a NumPy asset. `arrayName` is a string that identifies a specific array from the file. `allowPickle`
controls how should the binary data be processed.

```usda
# CaeNumPyFieldArray schema definition
class CaeNumPyFieldArray "CaeNumPyFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Defines a NumPy data array"
    customData = {
        string className = "NumPyFieldArray"
    }
)
{
    bool allowPickle = false (
        doc = """Specifies whether to allow pickle when reading NumPy files.
        Only enable for files from trusted sources.  Refer to NumPy documentation
        for `numpy.load` function for details and security implications."""
    )

    string arrayName = "" (
        doc = "Specifies the name for the array in the NumPy file."
    )

    uniform string slice = "" (
        doc = """Specifies the slicing/index expression to use to extract data from the numpy array.
        Python format string is accepted. The `ts` identifier can be used to replace with the
        value of attribute `ts`."""
    )

    int ts = -1 (
        doc = "Specifies a time-sampled attribute value that can be used in the `slice` format expressions."
    )
}
```


#### CaePointCloudAPI

`CaePointCloudAPI` is a single-apply API schema for describing datasets that are to be treated as point clouds.
`cae:pointCloud:coordinates` relationship can be used to specify the field array(s) that are to be treated as coordinates.

```usda
class "CaePointCloudAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents a point cloud."
    customData = {
        string className = "PointCloudAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:pointCloud:coordinates (
        doc = """Specifies the CaeFieldArray that must be treated as the coordinates. Multiple targets
                 may be specified when individual components are split among multiple field arrays."""
        displayName = "Coordinates"
        displayGroup = "Relationships"
        customData = {
            string apiName = "coordinates"
        }
    )
}
```

#### CaeMeshAPI

`CaeMeshAPI` is a single-apply API schema for describing datasets that are to be treated as surface meshes. The relationships
on this schema specify field array(s) that have similar roles and structure as `UsdGeomMesh` prim.

```usda
class "CaeMeshAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents a surface mesh."
    customData = {
        string className = "MeshAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:mesh:points (
        doc = "Specifies the CaeFieldArray that must be treated as the points."
        displayName = "Points"
        displayGroup = "Relationships"
        customData = {
            string apiName = "points"
        }
    )

    rel cae:mesh:faceVertexIndices (
        doc = "Specifies the CaeFieldArray that must be treated as the face vertex indices."
        displayName = "Face Vertex Indices"
        displayGroup = "Relationships"
        customData = {
            string apiName = "faceVertexIndices"
        }
    )

    rel cae:mesh:faceVertexCounts (
        doc = "Specifies the CaeFieldArray that must be treated as the face vertex counts."
        displayName = "Face Vertex Counts"
        displayGroup = "Relationships"
        customData = {
            string apiName = "faceVertexCounts"
        }
    )
}
```

#### CaeDenseVolumeAPI

`CaeDenseVolumeAPI` is intended for describing 3D volumetric datasets.

```usda
class "CaeDenseVolumeAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents a dense volume."
    customData = {
        string className = "DenseVolumeAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    uniform int3 cae:denseVolume:minExtent (
        doc = "Specifies the minimum structured (IJK) extent for the volume."
        displayName = "Min Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "minExtent"
        }
    )

    uniform int3 cae:denseVolume:maxExtent (
        doc = "Specifies the maximum structured (IJK) extent for the volume."
        displayName = "Max Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "maxExtent"
        }
    )

    uniform float3 cae:denseVolume:spacing = (1.0, 1.0, 1.0) (
        doc = "Specifies the spacing along each axis."
        displayName = "Spacing"
        displayGroup = "Attributes"
        customData = {
            string apiName = "spacing"
        }
    )
}
```

#### CaeNanoVDBFieldArrayAPI

`CaeNanoVDBFieldArrayAPI` is a single-apply API schema that can be applied to any `CaeFieldArray` to declare that it
represents a NanoVDB volume, specifying the IJK origin and dimensions of the volume grid.

```usda
class "CaeNanoVDBFieldArrayAPI" (
    inherits = </APISchemaBase>
    doc = """Applies to any CaeFieldArray that represents a NanoVDB volume."""
    customData = {
        string className = "NanoVDBFieldArrayAPI"
        token apiSchemaType = "singleApply"
        token[] apiSchemaCanOnlyApplyTo = ["CaeFieldArray"]
    }
)
{
    int3 cae:nanovdb_field_array:origin = (0, 0, 0) (
        doc = "Specifies the IJK origin of the NanoVDB volume."
        displayName = "Origin"
        customData = {
            string apiName = "origin"
        }
    )

    int3 cae:nanovdb_field_array:dims = (1, 1, 1) (
        doc = "Specifies the IJK dimensions (Ni, Nj, Nk) of the NanoVDB volume."
        displayName = "Dims"
        customData = {
            string apiName = "dims"
        }
    )
}
```

### OmniCaeSids Schemas

`OmniCaeSids` package is intended to includes single-apply API schemas for describing datasets that follow [CGNS
Standard Interface Data Structures (SIDS)](https://cgns.github.io/standard/SIDS/CGNS_SIDS.html)
data model. Currently, it includes `CaeSidsUnstructuredAPI` for unstructured
grids. We intend to add more schemas for structured grids and others types supported by SIDS as needed.

![OmniCaeSids Schemas](./OmniCaeSids.schema.svg)

#### CaeSidsUnstructuredAPI

`CaeSidsUnstructuredAPI` is an API schema for representing CGNS SIDS unstructured element set i.e. unstructured `Element_t`.
Named relationships referring to `CaeFieldArray` nodes help indicate which field arrays are to be treated as
grid coordinates, element connectivity, etc. It also indicates to the applications specific information about the shape,
type of those field arrays. For example, element connectivity field arrays are required to 1D with vertex indices specified
starting 1.

```usda
# CaeSidsUnstructuredAPI schema definition

class "CaeSidsUnstructuredAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a mesh that follows the CGNS SIDS data model for unstructured data."
    customData = {
        string className = "UnstructuredAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    uniform token cae:sids:elementType = "ElementTypeNull" (
        allowedTokens = [
            "ElementTypeNull", "ElementTypeUserDefined", "NODE", "BAR_2", "BAR_3",
            "TRI_3", "TRI_6", "QUAD_4", "QUAD_8", "QUAD_9",
            "TETRA_4", "TETRA_10", "PYRA_5", "PYRA_14",
            "PENTA_6", "PENTA_15", "PENTA_18", "HEXA_8", "HEXA_20", "HEXA_27",
            "MIXED", "PYRA_13", "NGON_n", "NFACE_n",
            "BAR_4", "TRI_9", "TRI_10", "QUAD_12", "QUAD_16",
            "TETRA_16", "TETRA_20", "PYRA_21", "PYRA_29", "PYRA_30",
            "PENTA_24", "PENTA_38", "PENTA_40", "HEXA_32", "HEXA_56", "HEXA_64",
            "BAR_5", "TRI_12", "TRI_15", "QUAD_P4_16", "QUAD_25", "TETRA_22",
            "TETRA_34", "TETRA_35", "PYRA_P4_29", "PYRA_50", "PYRA_55", "PENTA_33", "PENTA_66",
            "PENTA_75", "HEXA_44", "HEXA_98", "HEXA_125"
        ]
        doc = "Specifies required element type."
        displayName = "Element Type"
        displayGroup = "Attributes"
        customData = {
            string apiName = "elementType"
        }
    )

    uniform uint64 cae:sids:elementRangeStart (
        doc = "Specifies required element range start which is the index of the first element in this section."
        displayName = "Element Range Start"
        displayGroup = "Attributes"
        customData = {
            string apiName = "elementRangeStart"
        }
    )

    uniform uint64 cae:sids:elementRangeEnd (
        doc = "Specifies required element range end which is the index of the last eleemnt in this section."
        displayName = "Element Range End"
        displayGroup = "Attributes"
        customData = {
            string apiName = "elementRangeEnd"
        }
    )

    rel cae:sids:elementConnectivity (
        doc = "Specifies required CaeFieldArray that must be treated as element connectivity specification."
        displayName = "Element Connectivity"
        displayGroup = "Relationships"
        customData = {
            string apiName = "elementConnectivity"
        }

    )

    rel cae:sids:elementStartOffset (
        doc = "Specifies optional CaeFieldArray that must be treated as element start offsets specification."
        displayName = "Element Start Offset"
        displayGroup = "Relationships"
        customData = {
            string apiName = "elementStartOffset"
        }
    )

    rel cae:sids:gridCoordinates (
        doc = """Specifies optional CaeFieldArray that must be treated as the grid coordinate. Multiple targets
        may be specified when individual components are split among multiple CaeFieldArrays."""
        displayName = "Grid Coordinates"
        displayGroup = "Relationships"
        customData = {
            string apiName = "gridCoordinates"
        }
    )

    rel cae:sids:ngons (
        doc ="""For datasets of 'NFACE_n' element type, this relationship is used to specify dataset prims that
        describe the faces (N-gons) that form the polyhedral elements. Multiple targets may be specified
        if the face definitions are split among multiple element blocks."""
        displayName = "N_GONs"
        displayGroup = "Relationships"
        customData = {
            string apiName = "ngons"
        }
    )
}
```

Looking at the [`Element_t`](https://cgns.github.io/CGNS_docs_current/sids/gridflow.html#Elements) definition,
it's clear that this definition is a subset of `Element_t`. In addition, we add `gridCoordinates`
properties to make this definition self contained. In SIDS, `Element_t` goes together with external
`GridCoordinates_t` node. It's arguable if the core schema should instead introduce a separate type for the
coordinate system. That will be well aligned with data models like CGNS and VTK-m. For now, we defer that decision.

### OmniCaeVtk Schemas

Similarly to `OmniCaeSids` schemas, `OmniCaeVtk` is intended to add schemas for supporting the [VTK](https://www.vtk.org) data model.
This is meant as just an illustration on how VTK data model can be represented as a USD schema. VTK data model is quite extensive
and hence we don't make any attempt to support it all in this initial pass.

![OmniCaeVtk Schemas](./OmniCaeVtk.schema.svg)

#### CaeVtkFieldArray

`CaeVtkFieldArray` adds support for data arrays stored in VTK formats, specifically `.vtk`, `.vtu`, and `.vti`. `special` token
helps us reference non-named arrays that play special roles in VTK data models.

```usda
class CaeVtkFieldArray "CaeVtkFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Defines a VTK data array"
    customData = {
        string className = "FieldArray"
    }
)
{
    string arrayName = "" (
        doc = "Specifies the name for the array in the Vtk file."
        displayName = "Array Name"
    )

    uniform token special = "none" (
        allowedTokens = [
            "none",
            "points",
            "connectivity_offsets",
            "connectivity_array",
            "cell_types",
            "verts_connectivity_offsets",
            "verts_connectivity_array",
            "lines_connectivity_offsets",
            "lines_connectivity_array",
            "polys_connectivity_offsets",
            "polys_connectivity_array",
            "strips_connectivity_offsets",
            "strips_connectivity_array",
            "polyhedron_faces_offsets",
            "polyhedron_faces_connectivity_array",
            "polyhedron_face_locations_offsets",
            "polyhedron_face_locations_connectivity_array",
        ]
        doc = "Specifies if the array is special array."
    )
}
```

#### CaeVtkUnstructuredGridAPI

`CaeVtkUnstructuredGridAPI` can be used for datasets following the `vtkUnstructuredGrid` data model.

```usda
class "CaeVtkUnstructuredGridAPI" (
    inherits = </APISchemaBase>
    doc = """Defines a dataset that follows vtkUnstructuredGrid data model."""
    customData = {
        string className = "UnstructuredGridAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:vtk:points (
        doc = "Specifies the CaeFieldArray that must be treated as points."
        displayName = "Points"
        displayGroup = "Relationships"
        customData = {
            string apiName = "points"
        }
    )

    rel cae:vtk:connectivityOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as connectivity offsets."
        displayName = "Connectivity Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "connectivityOffsets"
        }
    )

    rel cae:vtk:connectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as connectivity array."
        displayName = "Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "connectivityArray"
        }
    )

    rel cae:vtk:cellTypes (
        doc = "Specifies the CaeFieldArray that must be treated as cell types."
        displayName = "Cell Types"
        displayGroup = "Relationships"
        customData = {
            string apiName = "cellTypes"
        }
    )

    rel cae:vtk:polyhedronFacesOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as polyhedron faces offsets specification."
        displayName = "Polyhedron Faces Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polyhedronFacesOffsets"
        }
    )

    rel cae:vtk:polyhedronFacesConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as polyhedron faces connectivity specification."
        displayName = "Polyhedron Faces Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polyhedronFacesConnectivityArray"
        }
    )

    rel cae:vtk:polyhedronFaceLocationsOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as polyhedron face locations offsets specification."
        displayName = "Polyhedron Face Locations Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polyhedronFaceLocationsOffsets"
        }
    )

    rel cae:vtk:polyhedronFaceLocationsConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as polyhedron face locations connectivity specification."
        displayName = "Polyhedron Face Locations Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polyhedronFaceLocationsConnectivityArray"
        }
    )
}
```

#### CaeVtkStructuredGridAPI

`CaeVtkStructuredGridAPI` can be used for datasets following the `vtkStructuredGrid` data model.

```usda
class "CaeVtkStructuredGridAPI" (
    inherits = </APISchemaBase>
    doc = """Defines a dataset that follows vtkStructuredGrid data model."""
    customData = {
        string className = "StructuredGridAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:vtk:points (
        doc = "Specifies the CaeFieldArray that must be treated as points."
        displayName = "Points"
        displayGroup = "Relationships"
        customData = {
            string apiName = "points"
        }
    )

    uniform int3 cae:vtk:minExtent = (0, 0, 0) (
        doc = "Specifies the minimum extent of the structured grid."
        displayName = "Minimum Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "minExtent"
        }
    )

    uniform int3 cae:vtk:maxExtent = (0, 0, 0) (
        doc = "Specifies the maximum extent of the structured grid."
        displayName = "Maximum Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "maxExtent"
        }
    )
}
```

#### CaeVtkImageDataAPI

`CaeVtkImageDataAPI` can be used for datasets following the `vtkImageData` data model (uniform rectilinear grids).

```usda
class "CaeVtkImageDataAPI" (
    inherits = </APISchemaBase>
    doc = """Defines a dataset that follows vtkImageData data model."""
    customData = {
        string className = "ImageDataAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    uniform float3 cae:vtk:origin = (0.0, 0.0, 0.0) (
        doc = "Specifies the origin of the image data."
        displayName = "Origin"
        displayGroup = "Attributes"
        customData = {
            string apiName = "origin"
        }
    )

    uniform float3 cae:vtk:spacing = (1.0, 1.0, 1.0) (
        doc = "Specifies the spacing of the image data."
        displayName = "Spacing"
        displayGroup = "Attributes"
        customData = {
            string apiName = "spacing"
        }
    )

    uniform int3 cae:vtk:minExtent = (0, 0, 0) (
        doc = "Specifies the minimum extent of the image data."
        displayName = "Min Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "minExtent"
        }
    )

    uniform int3 cae:vtk:maxExtent = (0, 0, 0) (
        doc = "Specifies the maximum extent of the image data."
        displayName = "Max Extent"
        displayGroup = "Attributes"
        customData = {
            string apiName = "maxExtent"
        }
    )
}
```

#### CaeVtkPolyDataAPI

`CaeVtkPolyDataAPI` can be used for datasets following the `vtkPolyData` data model.

```usda
class "CaeVtkPolyDataAPI" (
    inherits = </APISchemaBase>
    doc = """Defines a dataset that follows vtkPolyData data model."""
    customData = {
        string className = "PolyDataAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:vtk:points (
        doc = "Specifies the CaeFieldArray that must be treated as points."
        displayName = "Points"
        displayGroup = "Relationships"
        customData = {
            string apiName = "points"
        }
    )

    rel cae:vtk:vertsConnectivityOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as verts connectivity offsets."
        displayName = "Verts Connectivity Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "vertsConnectivityOffsets"
        }
    )

    rel cae:vtk:vertsConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as verts connectivity array."
        displayName = "Verts Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "vertsConnectivityArray"
        }
    )

    rel cae:vtk:linesConnectivityOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as lines connectivity offsets."
        displayName = "Lines Connectivity Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "linesConnectivityOffsets"
        }
    )

    rel cae:vtk:linesConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as lines connectivity array."
        displayName = "Lines Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "linesConnectivityArray"
        }
    )

    rel cae:vtk:polysConnectivityOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as polys connectivity offsets."
        displayName = "Polys Connectivity Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polysConnectivityOffsets"
        }
    )

    rel cae:vtk:polysConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as polys connectivity array."
        displayName = "Polys Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "polysConnectivityArray"
        }
    )

    rel cae:vtk:stripsConnectivityOffsets (
        doc = "Specifies the CaeFieldArray that must be treated as strips connectivity offsets."
        displayName = "Strips Connectivity Offsets"
        displayGroup = "Relationships"
        customData = {
            string apiName = "stripsConnectivityOffsets"
        }
    )

    rel cae:vtk:stripsConnectivityArray (
        doc = "Specifies the CaeFieldArray that must be treated as strips connectivity array."
        displayName = "Strips Connectivity Array"
        displayGroup = "Relationships"
        customData = {
            string apiName = "stripsConnectivityArray"
        }
    )
}
```

### OmniCaeEnSight

`OmniCaeEnSight` adds schemas for some EnSight field arrays and data types.

![OmniCaeEnSight Schemas](./OmniCaeEnSight.schema.svg)

#### CaeEnSightGoldGeoFieldArray

```usda
class CaeEnSightGoldGeoFieldArray "CaeEnSightGoldGeoFieldArray" (
    inherits = </CaeFieldArray>
    customData = {
        string className = "GoldGeoFieldArray"
    }
)
{
    uniform token type (
        allowedTokens = [ "coordinateX", "coordinateY", "coordinateZ", "connectivity", "elementNodeCounts",
                          "elementFaceCounts", "faceNodeCounts", "faceConnectivity" ]
        doc = "Specifies the type of data this field array represents."
        displayName = "Type"
    )

    uniform int partId (
        doc = "Specifies the part id."
        displayName = "Part Id"
    )

    uniform int pieceId  = -1 (
        doc = "Specifies the piece id."
        displayName = "Piece Id"
    )
}
```

#### CaeEnSightGoldVarFieldArray

```usda
class CaeEnSightGoldVarFieldArray "CaeEnSightGoldVarFieldArray" (
    inherits = </CaeFieldArray>
    customData = {
        string className = "GoldVarFieldArray"
    }
)
{
    uniform token type (
        allowedTokens = [ "scalar", "vector", "tensor", "tensor9" ]
        doc = "Specifies the type of data this field array represents."
        displayName = "Type"
    )

    uniform int partId (
        doc = "Specifies the part id."
        displayName = "Part Id"
    )

    uniform int pieceId  = -1 (
        doc = "Specifies the piece id."
        displayName = "Piece Id"
    )

    asset[] geoFileNames = [] (
        doc = """Specifies the assets for the geometry files. With multiple assets
            are specified then they are treated as spatial partitions of the same
            dataset. Temporal partitions may be specified by animating this attribute
            i.e. using time codes.
            """
        displayName = "Geometry File Names"
    )
}
```

#### CaeEnSightUnstructuredPieceAPI

```usda
class "CaeEnSightUnstructuredPieceAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents a piece of an EnSight dataset."
    customData = {
        string className = "UnstructuredPieceAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:ensight:piece:connectivity (
        doc = "Specifies the CaeFieldArray that must be treated as the connectivity."
        displayName = "Connectivity"
        displayGroup = "Relationships"
        customData = {
            string apiName = "connectivity"
        }
    )

    uniform token cae:ensight:piece:elementType (
        allowedTokens = [ "point", "bar2", "bar3", "tria3", "tria6", "quad4", "quad8",
                          "tetra4", "tetra10", "pyramid5", "pyramid13", "penta6", "penta15", "hexa8", "hexa20",
                          "nsided", "nfaced"]
        doc = "Specifies the element type."
        displayName = "Element Type"
        displayGroup = "Attributes"
        customData = {
            string apiName = "elementType"
        }
    )
}
```

#### CaeEnSightNSidedPieceAPI

`CaeEnSightNSidedPieceAPI` extends an `nsided` piece with the per-element node count array.

```usda
class "CaeEnSightNSidedPieceAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents an EnSight nsided piece."
    customData = {
        string className = "NSidedPieceAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:ensight:piece:elementNodeCounts (
        doc = "Specifies the CaeFieldArray that must be treated as the element node counts."
        displayName = "Element Node Counts"
        displayGroup = "Relationships"
        customData = {
            string apiName = "elementNodeCounts"
        }
    )
}
```

#### CaeEnSightNFacedPieceAPI

`CaeEnSightNFacedPieceAPI` extends an `nfaced` piece with per-element face counts and per-face node counts.

```usda
class "CaeEnSightNFacedPieceAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents an EnSight nfaced piece."
    customData = {
        string className = "NFacedPieceAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    rel cae:ensight:piece:elementFaceCounts (
        doc = "Specifies the CaeFieldArray that must be treated as the element face counts."
        displayName = "Element Face Counts"
        displayGroup = "Relationships"
        customData = {
            string apiName = "elementFaceCounts"
        }
    )

    rel cae:ensight:piece:faceNodeCounts (
        doc = "Specifies the CaeFieldArray that must be treated as the face node counts."
        displayName = "Face Node Counts"
        displayGroup = "Relationships"
        customData = {
            string apiName = "faceNodeCounts"
        }
    )
}
```

#### CaeEnSightUnstructuredPartAPI

`CaeEnSightUnstructuredPartAPI` describes a whole EnSight Gold unstructured part — its ID, shared coordinates, and the
set of element-type pieces that make it up.

```usda
class "CaeEnSightUnstructuredPartAPI" (
    inherits = </APISchemaBase>
    doc = "Defines a dataset that represents an EnSight Gold unstructured part."
    customData = {
        string className = "UnstructuredPartAPI"
        token apiSchemaType = "singleApply"
    }
)
{
    uniform int cae:ensight:part:id (
        doc = "Specifies the part id."
        displayName = "Part Id"
        displayGroup = "Attributes"
        customData = {
            string apiName = "id"
        }
    )

    rel cae:ensight:part:coordinates (
        doc = "Specifies the coordinates for this part."
        displayName = "Coordinates"
        displayGroup = "Relationships"
        customData = {
            string apiName = "coordinates"
        }
    )

    rel cae:ensight:part:pieces (
        doc = "Specifies the pieces that make up this part."
        displayName = "Pieces"
        displayGroup = "Relationships"
        customData = {
            string apiName = "pieces"
        }
    )
}
```

### OmniCaeOpenFoam

`OmniCaeOpenFoam` adds schemas for OpenFOAM case data.

#### CaeOpenFoamFieldArray

`CaeOpenFoamFieldArray` inherits from `CaeFieldArray` and represents a single OpenFOAM field array. The `type` token
identifies the role of the array within the mesh, and `name` specifies the field name for `internalField` arrays.

```usda
class CaeOpenFoamFieldArray "CaeOpenFoamFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Defines an OpenFOAM field array."
    customData = {
        string className = "FieldArray"
    }
)
{
    uniform token type (
        allowedTokens = [ "internalField", "points", "facesOffsets", "faces", "owner", "neighbour" ]
    )

    uniform string name (
        doc = "Specifies the name of the field array."
        displayName = "Field Name"
        customData = {
            string apiName = "name"
        }
    )
}
```

#### CaeOpenFoamPolyMeshAPI

`CaeOpenFoamPolyMeshAPI` describes the polyMesh structure of an OpenFOAM case.

```usda
class "CaeOpenFoamPolyMeshAPI" (
    inherits = </APISchemaBase>
    doc = "An OpenFOAM mesh API defining the polyMesh and internal field arrays."
    customData = {
        string className = "PolyMeshAPI"
    }
)
{
    rel cae:foam:points (
        doc = "Specifies the FieldArray that must be treated as the points of the mesh."
        displayName = "Points"
        customData = { string apiName = "points" }
    )

    rel cae:foam:facesOffsets (
        doc = "Specifies the FieldArray that must be treated as the faces offsets of the mesh."
        displayName = "Faces Offsets"
        customData = { string apiName = "facesOffsets" }
    )

    rel cae:foam:faces (
        doc = "Specifies the FieldArray that must be treated as the faces indices of the mesh."
        displayName = "Faces"
        customData = { string apiName = "faces" }
    )

    rel cae:foam:owner (
        doc = "Specifies the FieldArray that must be treated as the owner of the mesh."
        displayName = "Owner"
        customData = { string apiName = "owner" }
    )

    rel cae:foam:neighbour (
        doc = "Specifies the FieldArray that must be treated as the neighbour of the mesh."
        displayName = "Neighbour"
        customData = { string apiName = "neighbour" }
    )
}
```

#### CaeOpenFoamPolyBoundaryMeshAPI

`CaeOpenFoamPolyBoundaryMeshAPI` describes a single boundary patch within an OpenFOAM mesh.

```usda
class "CaeOpenFoamPolyBoundaryMeshAPI" (
    inherits = </APISchemaBase>
    doc = "An OpenFOAM poly boundary mesh API defining the boundaries of a mesh."
    customData = {
        string className = "PolyBoundaryMeshAPI"
    }
)
{
    uniform string cae:foam:name (
        doc = "Specifies the name of the boundary mesh."
        displayName = "Boundary Name"
        customData = { string apiName = "name" }
    )

    uniform int cae:foam:nFaces (
        doc = "Specifies the number of faces in this boundary mesh."
        displayName = "Number of Faces"
        customData = { string apiName = "nFaces" }
    )

    uniform int cae:foam:startFace (
        doc = "Specifies the starting face index for the boundary mesh."
        displayName = "Start Face"
        customData = { string apiName = "startFace" }
    )

    rel cae:foam:points (
        doc = "Specifies the FieldArray that must be treated as the points of the mesh."
        displayName = "Points"
        customData = { string apiName = "points" }
    )

    rel cae:foam:facesOffsets (
        doc = "Specifies the FieldArray that must be treated as the faces offsets of the mesh."
        displayName = "Faces Offsets"
        customData = { string apiName = "facesOffsets" }
    )

    rel cae:foam:faces (
        doc = "Specifies the FieldArray that must be treated as the faces indices of the mesh."
        displayName = "Faces"
        customData = { string apiName = "faces" }
    )

    rel cae:foam:owner (
        doc = "Specifies the FieldArray that must be treated as the owner of the mesh."
        displayName = "Owner"
        customData = { string apiName = "owner" }
    )
}
```

### OmniCaeTrimesh

`OmniCaeTrimesh` adds a `CaeFieldArray` subtype for surface mesh formats read via the `trimesh` Python library
(STL, OBJ, PLY, OFF, GLTF/GLB, and others).

#### CaeTrimeshFieldArray

```usda
class CaeTrimeshFieldArray "CaeTrimeshFieldArray" (
    inherits = </CaeFieldArray>
    doc = """Defines a surface mesh data array read via the trimesh library."""
    customData = {
        string className = "TrimeshFieldArray"
    }
)
{
    uniform token special = "none" (
        allowedTokens = [
            "none",
            "vertices",
            "faces",
            "face_offsets",
            "face_counts",
            "vertex_normals",
            "face_normals",
            "vertex_colors",
        ]
        doc = """Specifies which component of the surface mesh to extract.
                 - none          : use arrayName to look up a named vertex/face attribute
                 - vertices      : Nx3 vertex positions (float32)
                 - faces         : flattened Mx3 triangle indices (int32)
                 - face_offsets  : per-face connectivity offsets [0, 3, 6, ... 3M] (int32)
                 - face_counts   : per-face vertex counts, always 3 for a trimesh (int32)
                 - vertex_normals: Nx3 per-vertex normals (float32)
                 - face_normals  : Mx3 per-face normals (float32)
                 - vertex_colors : Nx4 RGBA vertex colours (uint8)"""
        displayName = "Special"
    )

    string arrayName = "" (
        doc = """Name of a vertex or face attribute to return when special is 'none'.
                 Looked up in mesh.vertex_attributes or mesh.face_attributes."""
        displayName = "Array Name"
    )
}
```

### OmniCaeNvdb

`OmniCaeNvdb` adds a `CaeFieldArray` subtype for NanoVDB volumes stored in `.nvdb` files.

#### CaeNvdbFieldArray

```usda
class CaeNvdbFieldArray "CaeNvdbFieldArray" (
    inherits = </CaeFieldArray>
    doc = "Defines a NanoVDB field array stored in a .nvdb file."
    customData = {
        string className = "FieldArray"
    }
)
{
}
```

## Examples

The follow table lists several examples of CGNS and NumPy datasets represented in USD.

| DataSet | USD |
| --------|-----|
| [yf17_hdf5.cgns](https://cgns.github.io/CGNSFiles.html) | [yf17_hdf5.usda](./examples/usda/yf17_hdf5.usda) |
| [StaticMixer.cgns](https://cgns.github.io/CGNSFiles.html) | [StaticMixer.usda](./examples/usda/StaticMixer.usda) |
| [tut21.cgns](https://cgns.github.io/CGNSFiles.html) | [tut21_cgns.usda](./examples/usda/tut21_cgns.usda) |
| [StaticMixer_tet.npz](./examples/data/StaticMixer_tet.npz) | [StaticMixer_tet.usda](./examples/usda/StaticMixer_tet.usda) |

## Discussion

* It is arguable that `CaeDataSet` and `CaeFieldArray` should either simply be replaced by or made sub types of `UsdVolVolume`
  and `UsdVolFieldBase`. The reason this design does that do that is to avoid conceptual conflicts in interpretations. For example,
  `UsdVolFieldBase` is a subtype of `UsdGeomXformable`, which represents an renderable, and transformable entity in the Usd stage.
  `CaeFieldArray`, however, is simply neither. An array representing element connectivity, for example, is neither renderable or
  transformable.

* `CaeFieldArray` and sub types intentionally avoid adding attributes that explicitly depend on the data contained in the files
  such as  data ranges etc. It's generally not trivial to compute these and adding these attributes may require importers
  to process the whole dataset and not just metadata. If needed, applications can introduce new API schemas for adding any extra
  application specific metadata like ranges and then simply apply those to the field array prims instead.

* Attributes like `CaeCgnsFieldArray.fieldPath` are not `uniform`. A CGNS file with multiple timesteps stores flow solution
  arrays under different `FlowSolution_t` instances, one for each timestep. These can be expressed nicely using
  `CaeCgnsFieldArray.fieldPath.timeSamples`. Same is true for `CaeFieldArrayNumPy`. If we simply stored each timestep
  as a separate array with unique name, we can correctly express that in USD because `CaeFieldArrayNumPy.fieldName` is not
  `uniform`. Note, if each timestep is stored in a separate file, then `OmniUsdFile.fileNames.timeSamples` takes care of that
  and one doesn't have to use `timeSamples` on `fieldPath` or `fieldName` attributes.

* While adding support for new "file formats" is easy and does not affect any code in the extensions for rendering
  and processing, adding a new data model API schema will require changes to such extensions. Initially,
  however, all CFD applications can internally do conversions, if needed, to transform data arrays to CGNS SIDS e.g.
  if connectivity is not specified the same way as CGNS, some conversion will be needed on the data loader side.
