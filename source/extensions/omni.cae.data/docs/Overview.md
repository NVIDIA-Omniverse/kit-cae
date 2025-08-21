# Overview

Extension that adds support for the CAE Data Delegate API. This extension exposes a light-weight and extensible
framework to add support for scientific file formats.

The extension introduces following main classes:

* `omni::cae::data::IDataDelegateInterface`: the Carbonite plugin interface for accessing the data delegate infrastructure.
* `omni::cae::data::IDataDelegateRegistry`: acts as the central repository for all registered data delegates. API lets
  dependent extensions (and applications) register `IDataDelegate` subclasses. Also exposes `getFieldArray` API call that
  will check with all registered data delegates and return the array from the first delegate that can read the requested
  prim.
* `omni::cae::data::IDataDelegate`: abstract base class for all data delegates. Developers can subclass this interface in
  C++ and Python to add support for new file formats.
* `omni::cae::data::IFieldArray`: abstract base class for an array. `IFieldArray` exposes a NumPy ndarray like array
  container.
* `omni::cae::data::IMutableFieldArray`: interface for a mutable `IFieldArray`.
* `omni::cae::IFieldArrayUtils`: collection of utility functions for working with IFieldArray.
