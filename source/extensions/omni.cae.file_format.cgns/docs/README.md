# CGNS File Format Extension [omni.cae.file_format.cgns]

This extension provides USD file format support for CGNS files, allowing them to be opened directly in USD-based applications.

## Features

- USD file format plugin for .cgns files
- Integration with CAE schemas for field array references
- Support for CGNS bases, zones, grid coordinates, and flow solutions

## Usage

When this extension is loaded, .cgns files can be opened directly in USD applications. The CGNS file structure is
automatically converted to a USD scene graph with appropriate CAE schema prims representing the data hierarchy.
