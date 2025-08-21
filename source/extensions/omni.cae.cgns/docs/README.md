# CGNS Data Delegate Extension [omni.cae.cgns]

This extension provides data delegate for reading CGNS field arrays from CGNS files.

## Features

- Data delegate for loading field arrays from CGNS files
- Configurable data type conversion (double to float, int64 to int32)
- Thread-safe access to CGNS files

## Settings

- `exts."omni.cae.cgns".convertDoubleToFloats`: Enable downsampling of double precision arrays to float (default: true)
- `exts."omni.cae.cgns"convertElementConnectivityToInt32`: Enable downsampling of element connectivity arrays to int32 (default: true)
