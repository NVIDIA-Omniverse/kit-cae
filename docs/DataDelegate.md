# Data Delegate

The Data Delegate infrastructure provides a collection of classes that centralize data loading and sharing across extensions in Kit CAE.
The Data Delegate system works in conjunction with the [CAE USD Schema](../usdSchema/README.md) to enable seamless data access.
When an extension needs to access data represented in a USD stage through a `CaeFieldArray` or its subtypes, it can utilize
the Data Delegate Registry ([`IDataDelegateRegistry`](../include/omni/cae/data/IDataDelegateRegistry.h)) singleton.

## Overview

The `IDataDelegateRegistry` provides two primary sets of APIs:
- **Registration APIs**: Register and deregister concrete data delegate implementations (`IDataDelegate` subclasses)
- **Data Access APIs**: Retrieve data arrays associated with `CaeFieldArray` primitives

## IDataDelegateRegistry Interface

```c++
/// IDataDelegateRegistry Interface in namespace omni::cae::data
class IDataDelegateRegistry
{
public:
    /**
     * Register a data delegate. Delegates with higher priority are checked first
     * when using APIs like `getFieldArray`.
     *
     * @param dataDelegate Data delegate to register.
     * @param priority Priority for the data delegate.
     */
    virtual void registerDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate, DelegatePriority priority = 0) = 0;

    /**
     * Deregister a data delegate.
     *
     * @param dataDelegate Data delegate to deregister.
     */
    virtual void deregisterDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate) = 0;

    /**
     * Deregister all data delegates that were registered by the specified
     * extension.
     *
     * @param extensionId The ID of the source extension that registered the data
     * delegates.
     */
    virtual void deregisterAllDataDelegatesForExtension(const char* extensionId) = 0;

    /**
     * Iterates through all registered data delegates and returns a field array
     * if any delegate supports it.
     *
     * @param fieldArrayPrim Field array primitive of type pxr::OmniCaeFieldArray or a
     * subtype.
     * @param time TimeCode to use when looking up primitive attributes.
     *
     * @return Field array or null pointer.
     */
    virtual carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim,
                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;

    /**
     * Asynchronous version of `getFieldArray`.
     */
    virtual carb::tasking::Future<carb::ObjectPtr<IFieldArray>> getFieldArrayAsync(
        pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;
};
```

## IDataDelegate Interface

The `IDataDelegate` interface is responsible for the actual data reading operations referenced in a `CaeFieldArray`
prim. Typically, you would implement a new `IDataDelegate` subclass for each new `CaeFieldArray` schema subtype.

```c++
/// IDataDelegate interface in namespace omni::cae::data
class IDataDelegate : public carb::IObject
{
public:
    /**
     * Returns `true` if this delegate can provide the field array.
     *
     * @param fieldArrayPrim Field array primitive of type pxr::OmniCaeFieldArray or a
     * subtype.
     * @return `true` if the field array is supported, `false` otherwise.
     */
    virtual bool canProvide(pxr::UsdPrim fieldArrayPrim) const = 0;

    /**
     * Returns the data array described by the primitive.
     *
     * @param fieldArrayPrim Field array primitive of type pxr::OmniCaeFieldArray or a
     * subtype.
     * @param timeCode Time code for the time to read.
     * @return Smart pointer to the field array on success. Returns empty smart
     * pointer if the read operation failed.
     */
    virtual carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim,
                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;

    /**
     * Get the ID of the source extension that registered this delegate.
     *
     * @return ID of the source extension that registered this delegate.
     */
    virtual const char* getExtensionId() const = 0;

}; // IDataDelegate
```

## Implementation Examples

### C++ Extension Implementation

Extensions that need to support reading new types of `CaeFieldArray` primitives can implement the `IDataDelegate`
interface and register the implementation with the delegate registry. Here's an example from the
CGNS extension ([`omni.cae.cgns`](../source/extensions/omni.cae.cgns/)):

```c++
class Extension : public omni::ext::IExt
{
    std::string m_extensionId;

public:
    void onStartup(const char* extId) override
    {
        this->m_extensionId = extId;
        auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        auto* registry = iface->getDataDelegateRegistry();
        auto* utils = iface->getFieldArrayUtils();

        // Register CGNS data delegate
        omni::cae::data::IDataDelegatePtr cgnsDelegate =
            carb::stealObject<omni::cae::data::IDataDelegate>(new omni::cae::data::cgns::DataDelegate(extId, utils));
        registry->registerDataDelegate(cgnsDelegate);
    }

    void onShutdown() override
    {
        auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        auto* registry = iface->getDataDelegateRegistry();
        registry->deregisterAllDataDelegatesForExtension(m_extensionId.c_str());
    }
};
```

### Python Extension Implementation

For Python extensions, [`omni.cae.npz`](../source/extensions/omni.cae.npz/) provides a good reference implementation:

```python
# Snippet from source/extensions/omni.cae.npz/python/extension.py

class Extension(omni.ext.IExt):
    def on_startup(self, ext_id):
        from .npz import NPZDataDelegate
        from omni.cae.data import get_data_delegate_registry

        self._registry = get_data_delegate_registry()

        # NOTE: You must maintain a reference to the delegate in Python
        # to prevent it from being garbage collected.
        self._delegate = NPZDataDelegate(ext_id)
        self._registry.register_data_delegate(self._delegate)

    def on_shutdown(self):
        self._registry.deregister_data_delegate(self._delegate)
        del self._delegate
```

## IFieldArray Interface

[`IFieldArray`](../include/omni/cae/data/IFieldArray.h) serves as an N-dimensional data array container that supports
reference counting for easy sharing and caching. `IDataDelegate` subclasses read data referenced by a `CaeFieldArray`
prim into an `IFieldArray` instance.

The field array primitive passed to `IDataDelegate::getFieldArray` can provide attributes such as filename and
data array name, which the delegate uses to determine which data array to read from the file or memory. By design,
`IFieldArray` is read-only. Data delegates can use [`IFieldArrayUtils`](../include/omni/cae/data/IFieldArrayUtils.h)
to allocate memory for a mutable `IFieldArray` (i.e., [`IMutableFieldArray`](../include/omni/cae/data/IFieldArray.h)).
Alternatively, data delegates can implement their own `IFieldArray` subclasses and return pointers to existing memory buffers.

### IFieldArray API

```c++
/**
 * IFieldArray is an interface for an N-dimensional data array. The ndims,
 * shape, and strides intentionally align with the definitions in NumPy `ndarray`.
 */
class IFieldArray : public carb::IObject
{
public:
    /**
     * Return a pointer to the data held by this instance.
     *
     * @return nullptr if isValid() == false, otherwise a read-only pointer to the data.
     */
    virtual const void* getData() const = 0;

    template <typename T>
    const T* getData() const
    {
        return reinterpret_cast<const T*>(this->getData());
    }

    /**
     * Returns the number of array dimensions.
     */
    virtual uint32_t getNDims() const = 0;

    /**
     * Returns the shape of the array.
     */
    virtual std::vector<uint64_t> getShape() const = 0;

    /**
     * Returns the strides of the array. Note: strides are specified in bytes.
     */
    virtual std::vector<uint64_t> getStrides() const = 0;

    /**
     * Returns the elemental type held by this array.
     */
    virtual ElementType getElementType() const = 0;

    /**
     * Returns the device ID for the device on which the data held by this field array
     * is hosted.
     *
     * When a non-negative number (>= 0) is returned, it corresponds to the CUDA device ID
     * returned by functions such as `cudaGetDevice()`.
     *
     * When -1 is returned, it indicates host (CPU) memory.
     */
    virtual int32_t getDeviceId() const = 0;

}; // IFieldArray
```

The interface intentionally mimics NumPy's array interface, with the addition of a device ID to enable referencing
CPU/GPU arrays. In Python, the wrapped `IFieldArray` exposes either the **NumPy Array Interface** (NAI) or
**CUDA Array Interface** (CAI) based on whether the device ID is -1 (CPU) or >= 0 (GPU). This allows seamless passing of
`IFieldArray` instances to NumPy, CuPy, or Warp for data processing without creating copies. Conversely, in Python,
`IFieldArray` exposes a `from_array` static method that can create a zero-copy `IFieldArray` instance from NumPy or CuPy
arrays, or any object that supports NAI or CAI.

### Data Delegate Implementation Examples

#### C++ Implementation

[`omni::cae::data::cgns::DataDelegate`](../source/extensions/omni.cae.cgns/plugins/omni.cae.cgns/DataDelegate.h)
 demonstrates how to access CGNS files to read requested data arrays:

```c++
// Snippet from source/extensions/omni.cae.cgns/plugins/omni.cae.cgns/DataDelegate.cpp

carb::ObjectPtr<IFieldArray> DataDelegate::getFieldArray(pxr::UsdPrim prim, pxr::UsdTimeCode time)
{
    // ... implementation details ...

    auto farray = m_fieldArrayUtils->createMutableFieldArray(etype, shape, -1, Order::fortran);
    CARB_LOG_INFO("Field array size: %" PRIu64 " bytes", farray->getMutableDataSizeInBytes());

    if (cgio_read_all_data_type(
            cgioFile, currentNodeId, getDataType(farray->getElementType()), farray->getMutableData()) != CG_OK)
    {
        CARB_LOG_ERROR("CGNS error: %s", cg_get_error());
        cgio_close_file(cgioFile);
        return {};
    }

    cgio_close_file(cgioFile);
    return carb::borrowObject<IFieldArray>(farray.get());
}
```

#### Python Implementation

[`omni.cae.npz`](../source/extensions/omni.cae.npz/) demonstrates the same functionality in a Python-only extension:

```python
# Snippet from source/extensions/omni.cae.npz/python/npz.py
from omni.cae.data.delegates import DataDelegateBase
import numpy as np

class NPZDataDelegate(DataDelegateBase):

    def __init__(self, extId: str):
        super().__init__(extId)

    def get_field_array(self, prim: Usd.Prim, time: Usd.TimeCode) -> np.ndarray:
        primT = npz.FieldArray(prim)
        arrayName = primT.GetArrayNameAttr().Get(time)
        fileNames = primT.GetFileNamesAttr().Get(time)
        allowPickle = primT.GetAllowPickleAttr().Get(time)

        arrays = []
        for f in fileNames:
            dataset = np.load(f.resolvedPath, allow_pickle=allowPickle)
            if isinstance(dataset, np.ndarray) and dataset.dtype == object:
                dataset = dataset.item(0)
            if arrayName in dataset.keys():
                array = dataset[arrayName]
                # ... additional processing ...
                arrays.append(array)

        return np.concatenate(arrays) if arrays else None

    def can_provide(self, prim: Usd.Prim) -> bool:
        return prim and prim.IsValid() and prim.IsA(npz.FieldArray)
```

When subclassing `DataDelegateBase`, the `get_field_array` method can return a NumPy ndarray, CuPy ndarray, or an `IFieldArray`.
Any object that supports either the **NumPy Array Interface** or **CUDA Array Interface** protocol is acceptable.
The Python wrapping code handles encapsulating the returned object into an `IFieldArray` appropriately.

## Using Data Delegate

The Data Delegate API provides access to scientific data referenced in USD files. Extensions can use
`IDataDelegateRegistry::getFieldArray` to access data for any `CaeFieldArray` primitive subtype.

### Example USD File

Here's a simple USD file containing just a field array primitive. Note that even without a `CaeDataSet`, this is a completely valid USD file:

```usd
#usda 1.0

def Xform "World"
{
    def CaeCgnsFieldArray "Temperature"
    {
        asset[] fileNames = [@StaticMixer.cgns@]
        uniform token fieldAssociation = "vertex"
        string fieldPath = "/Base/StaticMixer/Flow Solution/Temperature"
    }
}
```

### C++ Usage Example

```c++
auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
if (!iface)
{
    CARB_LOG_ERROR("Failed to get IDataDelegateInterface!");
    return /* error handling */;
}

auto* registry = iface->getDataDelegateRegistry();
if (!registry)
{
    CARB_LOG_ERROR("Missing DataDelegateRegistry!");
    return /* error handling */;
}

auto* usdContext = omni::usd::UsdContext::getContext();
auto stage = usdContext->getStage();
if (!stage)
{
    CARB_LOG_ERROR("Failed to locate stage!");
    return /* error handling */;
}

pxr::SdfPath path("/World/Temperature");
pxr::UsdPrim prim = stage->GetPrimAtPath(path);
if (!prim)
{
    CARB_LOG_ERROR("Failed to locate primitive at path '%s'.", path.GetText());
    return /* error handling */;
}

pxr::UsdTimeCode timeCode = /* specify time code */;
auto array = registry->getFieldArray(prim, timeCode);
if (!array)
{
    CARB_LOG_ERROR("Failed to load array for primitive at path '%s'.", path.GetText());
    return /* error handling */;
}

// Process the array
// ...
```

### Python Usage Example

```python
from omni.cae.data import get_data_delegate_registry
import warp as wp

stage: pxr.Usd.Stage = # ... get stage reference
timeCode: pxr.Usd.TimeCode = # ... specify time code
prim: pxr.Usd.Prim = stage.GetPrimAtPath("/World/Temperature")

# ... error checking ...

registry = get_data_delegate_registry()
temperature = registry.get_field_array(prim, timeCode)

# Since field array supports NAI or CAI, we can directly pass it to Warp
# without duplicating data (unless required because the active device differs
# from the array's device)
wp_temperature = wp.array(temperature, copy=False)
```

### Asynchronous Python Usage

Python `asyncio` is also supported for non-blocking data access:

```python
async def coroutine():
    array = await registry.get_field_array_async(prim, timeCode)
    # Process array...
```

In this case, the actual data reading occurs on a concurrent thread without blocking the main thread.