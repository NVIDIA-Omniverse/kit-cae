# HDF5 Data Delegate Extension [omni.cae.hdf5]

This extension provides data delegate for reading HDF5 field arrays from HDF5 files.

## Features

- Data delegate for loading field arrays from HDF5 files
- Configurable double-to-float conversion
- Direct HDF5 file access using HDF5 libraries
- Integration with the CAE data delegate registry

## Settings

- `exts."omni.cae.hdf5".convertDoubleToFloats`: Enable downsampling of double precision arrays to float (default: true).
