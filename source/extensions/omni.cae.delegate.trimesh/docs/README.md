# CAE Trimesh Delegate [omni.cae.delegate.trimesh]

This extension provides data delegate functionality for reading surface mesh formats using the [trimesh](https://trimesh.org/) Python library.

## Features

- Data delegate for loading surface mesh geometry from trimesh-supported formats
- Supported file formats: STL, OBJ, PLY, OFF, GLTF/GLB, 3MF
- Extracts vertices, face indices, normals, vertex colours, and named attributes
- Reader caching: a mesh file is loaded only once per stage, even when multiple field arrays reference it
- Integrates with the CAE data delegate registry
