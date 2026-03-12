// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#define PYBIND11_DETAILED_ERROR_MESSAGES

// .clang-format off
#include <omni/cae/data/IDataDelegateIncludes.h>
#include <pxr/usd/usdUtils/stageCache.h>
// .clang-format on

#include <carb/BindingsPythonUtils.h>
#include <carb/tasking/ITasking.h>

#include <omni/cae/data/IDataDelegate.h>
#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/cae/data/IDataDelegateRegistry.h>
#include <omni/cae/data/IFieldArray.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omni/cae/data/IUsdUtils.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <cuda_runtime.h>
#include <iomanip>
#include <sstream>
#include <utility>

CARB_BINDINGS("omni.cae.data.python")

// FIXME: not entirely sure about this; but following the pattern in
// ActionBindingsPython.cpp
// DISABLE_PYBIND11_DYNAMIC_CAST(omni::cae::data::IDataDelegate);
// DISABLE_PYBIND11_DYNAMIC_CAST(omni::cae::data::IDataDelegateRegistry);
// DISABLE_PYBIND11_DYNAMIC_CAST(omni::cae::data::IFieldArray);
DISABLE_PYBIND11_DYNAMIC_CAST(pxr::UsdPrim);
DISABLE_PYBIND11_DYNAMIC_CAST(pxr::UsdTimeCode);

namespace
{
using namespace omni::cae::data;

std::string time_code_to_string(pxr::UsdTimeCode time)
{
    if (time.IsDefault())
    {
        return "Default";
    }
    else if (time.IsEarliestTime())
    {
        return "EarliestTime";
    }
    else
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << time.GetValue();
        return oss.str();
    }
}

std::string time_code_to_string(double time)
{
    return time_code_to_string(pxr::UsdTimeCode(time));
}

pxr::UsdStageCache::Id GetStageId(pxr::UsdStageRefPtr stage)
{
    return pxr::UsdUtilsStageCache::Get().GetId(stage);
}

pxr::UsdStageRefPtr GetStage(pxr::UsdStageCache::Id id)
{
    return pxr::UsdUtilsStageCache::Get().Find(id);
}

py::dtype getDType(const omni::cae::data::IFieldArray* self)
{
    auto np = py::module::import("numpy");
    switch (self->getElementType())
    {
    case ElementType::int32:
        // Force np.intc specifically to avoid "long" on Windows
        return py::dtype::from_args(np.attr("intc"));
    case ElementType::uint32:
        // Force np.uintc specifically to avoid "long" on Windows
        return py::dtype::from_args(np.attr("uintc"));
    case ElementType::int64:
        return py::dtype("i8");
    case ElementType::uint64:
        return py::dtype("u8");
    case ElementType::float32:
        return py::dtype("f4");
    case ElementType::float64:
        return py::dtype("f8");
    default:
        return py::dtype("V0");
    }
}

auto getElementType(py::dtype dtype)
{
    const char kind = dtype.attr("kind").cast<char>(); // "i", "u", "f"
    size_t itemsize = dtype.attr("itemsize").cast<size_t>(); // 4 bytes for int32, 8 bytes for int64, etc.

    if (kind == 'i')
    {
        if (itemsize == 4)
        {
            return omni::cae::data::ElementType::int32;
        }
        else if (itemsize == 8)
        {
            return omni::cae::data::ElementType::int64;
        }
    }
    else if (kind == 'u')
    {
        if (itemsize == 4)
        {
            return omni::cae::data::ElementType::uint32;
        }
        else if (itemsize == 8)
        {
            return omni::cae::data::ElementType::uint64;
        }
    }
    else if (kind == 'f')
    {
        if (itemsize == 4)
        {
            return omni::cae::data::ElementType::float32;
        }
        else if (itemsize == 8)
        {
            return omni::cae::data::ElementType::float64;
        }
    }
    CARB_LOG_ERROR("Unsupported dtype: '%c' (itemsize: %d), '%s'", kind, static_cast<int>(itemsize),
                   std::string(py::str(dtype)).c_str());
    throw py::value_error(std::string("Unsupported dtype: '") + kind + "', '" + std::string(py::str(dtype)).c_str() +
                          "', itemsize: " + std::to_string(itemsize));
}


py::dict getArrayInterface(const IFieldArray* self)
{
    if (self->getDeviceId() != -1)
    {
        throw py::attribute_error(std::string("__array__interface__ not supported by array hosted on device ") +
                                  std::to_string(self->getDeviceId()));
    }

    auto dtype = getDType(self);
    const auto shape = self->getShape();
    const auto strides = self->getStrides();

    py::dict iface;
    iface["data"] = py::make_tuple(reinterpret_cast<const uintptr_t>(self->getData()), /*readOnly*/ true);
    iface["shape"] = py::tuple(py::cast(shape));
    iface["strides"] = py::tuple(py::cast(strides));
    iface["typestr"] = dtype.attr("str"); // "<i4", "<f8", etc.
    iface["version"] = 3;
    return iface;
}

py::dict getCudaArrayInterface(const IFieldArray* self)
{
    if (self->getDeviceId() == -1)
    {
        throw py::attribute_error(std::string("__cuda_array__interface__ not supported by array hosted on device ") +
                                  std::to_string(self->getDeviceId()));
    }

    const auto shape = self->getShape();
    const auto strides = self->getStrides();
    auto dtype = getDType(self);

    py::dict iface;
    iface["data"] = py::make_tuple(reinterpret_cast<const uintptr_t>(self->getData()), /*readOnly*/ true);
    iface["shape"] = py::tuple(py::cast(shape));
    iface["strides"] = py::tuple(py::cast(strides));
    iface["typestr"] = dtype.attr("str"); // "<i4", "<f8", etc.
    iface["version"] = 2;
    return iface;
}


py::buffer_info getBufferInfo(const omni::cae::data::IFieldArray* self)
{
    // get the dtype from the IFieldArray
    auto dtype = getDType(self);
    const std::string format = dtype.attr("char").cast<std::string>(); // Extract "i", for example.
    const size_t itemsize = dtype.attr("itemsize").cast<size_t>(); // 4 bytes for int32, 8 bytes for int64, etc.

    return py::buffer_info(const_cast<void*>(self->getData()), itemsize, format.c_str(), self->getNDims(),
                           self->getShape(), self->getStrides(), true);
}

py::object getNumpyArray(carb::ObjectPtr<omni::cae::data::IFieldArray> array)
{
    if (!array)
    {
        return py::none();
    }
    if (array->getDeviceId() != -1)
    {
        throw std::runtime_error("Implicit copying from device to CPU/numpy arrays is not supported.");
    }
    return py::array(getBufferInfo(array.get()), py::cast(array) /* for memory management */);
}

int32_t getCudaDeviceId(const void* ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    if (err != cudaSuccess)
    {
        CARB_LOG_ERROR("Error getting pointer attributes: %s", cudaGetErrorString(err));
        throw std::runtime_error("Could not determine CUDA device!");
    }
    return attributes.device;
}

using IFieldArrayFuture = carb::tasking::Future<carb::ObjectPtr<omni::cae::data::IFieldArray>>;

py::object wrap_future(IFieldArrayFuture&& future)
{
    py::object asyncio = py::module::import("asyncio");
    py::object loop = asyncio.attr("get_event_loop")();
    py::object py_future = loop.attr("create_future")();

    // we use these raw pointers so we can clean them up properly while the GIL is held
    // in various lambdas.
    auto* py_future_ptr = new py::object(py_future);
    auto* py_loop_ptr = new py::object(loop);

    future.then(carb::tasking::Priority::eDefault, {},
                [py_future_ptr, py_loop_ptr](carb::ObjectPtr<omni::cae::data::IFieldArray> array)
                {
                    // remember: this is called on arbitrary thread.
                    CARB_LOG_INFO("carb::tasking::Future completed.");
                    py::gil_scoped_acquire acquire;

                    // Schedule the callback in the event loop thread.
                    py_loop_ptr->attr("call_soon_threadsafe")(py::cpp_function(
                        [py_future_ptr, array]()
                        {
                            py_future_ptr->attr("set_result")(array);
                            delete py_future_ptr;
                        }));
                    delete py_loop_ptr;
                });

    return py_future;
}

class PythonFieldArray final : public IFieldArray
{
    CARB_IOBJECT_IMPL

    const void* m_data;
    std::vector<uint64_t> m_shape;
    std::vector<uint64_t> m_strides;
    ElementType m_etype;
    int32_t m_deviceId;
    py::object m_object;

public:
    PythonFieldArray(const void* data,
                     const std::vector<uint64_t>& shape,
                     const std::vector<uint64_t>& strides,
                     ElementType etype,
                     int32_t deviceId,
                     py::object obj)
        : m_data(data), m_shape(shape), m_strides(strides), m_etype(etype), m_deviceId(deviceId), m_object(std::move(obj))
    {
    }

    ~PythonFieldArray()
    {
        py::gil_scoped_acquire acquire;
        m_data = nullptr;
        // do I need to set the active CUDA device? Shouldn't be needed
        // since the Python CUDA array should perhaps handle that.
        m_object = py::object{};
    }

    const void* getData() const override
    {
        return m_data;
    }

    std::vector<uint64_t> getShape() const override
    {
        return m_shape;
    }

    std::vector<uint64_t> getStrides() const override
    {
        return m_strides;
    }

    omni::cae::data::ElementType getElementType() const override
    {
        return m_etype;
    }

    uint32_t getNDims() const override
    {
        return static_cast<uint32_t>(m_shape.size());
    }

    int32_t getDeviceId() const override
    {
        return m_deviceId;
    }

    /// create from numpy.ndarray
    static PythonFieldArray* fromNDArray(py::array array)
    {
        std::vector<uint64_t> shape;
        std::vector<uint64_t> strides;
        ElementType etype = ::getElementType(array.dtype());

        auto ndim = array.ndim();
        std::copy_n(array.shape(), ndim, std::back_inserter(shape));
        std::copy_n(array.strides(), ndim, std::back_inserter(strides));

        return new PythonFieldArray(array.data(), shape, strides, etype, -1, array);
    }

    static PythonFieldArray* fromNumpyArrayInterface(py::object obj)
    {
        py::dict iface = obj.attr("__array_interface__").cast<py::dict>();
        // int version = iface["version"].cast<int>();
        uintptr_t data_ptr = iface["data"].cast<py::tuple>()[0].cast<uintptr_t>();
        auto shape = iface["shape"].cast<std::vector<uint64_t>>();
        ElementType etype = ::getElementType(py::dtype(iface["typestr"].cast<std::string>()));
        auto strides = PythonFieldArray::getStrides(iface, etype, shape);
        return new PythonFieldArray(reinterpret_cast<const void*>(data_ptr), shape, strides, etype, -1, obj);
    }

    static PythonFieldArray* fromCudaArrayInterface(py::object obj)
    {
        py::dict iface = obj.attr("__cuda_array_interface__").cast<py::dict>();
        // int version = iface["version"].cast<int>();
        uintptr_t data_ptr = iface["data"].cast<py::tuple>()[0].cast<uintptr_t>();
        auto shape = iface["shape"].cast<std::vector<uint64_t>>();
        ElementType etype = ::getElementType(py::dtype(iface["typestr"].cast<std::string>()));
        auto strides = PythonFieldArray::getStrides(iface, etype, shape);
        const auto* data = reinterpret_cast<const void*>(data_ptr);
        return new PythonFieldArray(data, shape, strides, etype, getCudaDeviceId(data), obj);
    }

private:
    static std::vector<uint64_t> getStrides(py::dict& iface, ElementType etype, const std::vector<uint64_t>& shape)
    {
        if (!iface.contains("strides") || iface["strides"].is_none())
        {
            // to interpret as C-style contiguous array.
            std::vector<uint64_t> strides(shape.size(), 1);
            const int ndims = static_cast<int>(shape.size());
            for (int i = ndims - 2; i >= 0; --i)
            {
                strides.at(i) = strides.at(i + 1) * shape.at(i + 1);
            }
            return strides;
        }
        else
        {
            return iface["strides"].cast<std::vector<uint64_t>>();
        }
    }
};


class DataDelegate : public omni::cae::data::IDataDelegate
{
    std::string m_extensionId;

public:
    DataDelegate(const std::string& extensionId) : m_extensionId(extensionId)
    {
    }
    ~DataDelegate() override
    {
    }

    const char* getExtensionId() const override
    {
        return m_extensionId.c_str();
    }

    carb::ObjectPtr<omni::cae::data::IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) override
    {
        CARB_LOG_INFO("getFieldArray from Python");

        // FIXME: need to find the stage id using fieldArrayPrim.GetPrim().
        auto stageId = GetStageId(fieldArrayPrim.GetStage());
        std::string primPath = fieldArrayPrim.GetPath().GetString();

        // acquire GIL since we're making Python call.
        py::gil_scoped_acquire acquire;
        py::object obj = this->_get_field_array(stageId.ToLongInt(), primPath, time.GetValue());

        if (obj.is_none())
        {
            return {};
        }

        if (py::isinstance<py::array>(obj))
        {
            auto arr = py::cast<py::array>(obj);
            return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNDArray(arr));
        }

        if (py::hasattr(obj, "__array_interface__"))
        {
            return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNumpyArrayInterface(obj));
        }

        if (py::hasattr(obj, "__cuda_array_interface__"))
        {
            return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromCudaArrayInterface(obj));
        }

        if (py::isinstance<carb::ObjectPtr<omni::cae::data::IFieldArray>>(obj))
        {
            auto arr = py::cast<carb::ObjectPtr<omni::cae::data::IFieldArray>>(obj);
            return arr;
        }

        return {};
    }

    bool canProvide(pxr::UsdPrim fieldArrayPrim) const override
    {
        // FIXME: need to find the stage id using fieldArrayPrim.GetPrim().
        auto stageId = GetStageId(fieldArrayPrim.GetStage());
        std::string primPath = fieldArrayPrim.GetPath().GetString();

        // acquire GIL since we're making Python call.
        py::gil_scoped_acquire acquire;
        return this->_can_provide(stageId.ToLongInt(), primPath);
    }

    virtual py::object _get_field_array(long int stageId, std::string primPath, double time) = 0;
    virtual bool _can_provide(long int stageId, std::string primPath) const = 0;
};

/// Trampoline class for pybind.
/// ref:
/// https://pybind11.readthedocs.io/en/latest/advanced/classes.html#overriding-virtual-functions-in-python
class PyDataDelegate : public DataDelegate
{
    CARB_IOBJECT_IMPL
public:
    using DataDelegate::DataDelegate;

    py::object _get_field_array(long int stageId, std::string primPath, double time) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, DataDelegate, _get_field_array, stageId, primPath, time);
    }

    bool _can_provide(long int stageId, std::string primPath) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, DataDelegate, _can_provide, stageId, primPath);
    }
};

PYBIND11_MODULE(_omni_cae_data, m)
{
    using namespace omni::cae::data;
    m.doc() = "pybind11 omni.cae.data bindings";
    py::enum_<ElementType>(m, "ElementType", R"(
        Enumeration of supported array element types.

        This enum defines the data types that can be stored in IFieldArray objects,
        covering common integer and floating-point types used in scientific computing.
    )")
        .value("int32", ElementType::int32, "32-bit signed integer")
        .value("int64", ElementType::int64, "64-bit signed integer")
        .value("uint32", ElementType::uint32, "32-bit unsigned integer")
        .value("uint64", ElementType::uint64, "64-bit unsigned integer")
        .value("float32", ElementType::float32, "32-bit floating point (single precision)")
        .value("float64", ElementType::float64, "64-bit floating point (double precision)");

    carb::defineInterfaceClass<IDataDelegateInterface>(
        m, "IDataDelegateInterface", "acquire_data_delegate_interface", "release_data_delegate_interface", R"(
        Core interface for accessing data delegation services.

        This interface provides access to the data delegate registry and USD utilities,
        which are used to manage field array data loading from various sources.
    )")
        .def(
            "get_data_delegate_registry",
            [](IDataDelegateInterface* iface)
            {
                auto* registry = iface->getDataDelegateRegistry();
                return py::cast(registry, py::return_value_policy::reference);
            },
            R"(
             Get the data delegate registry.

             Returns:
                 IDataDelegateRegistry: The registry for managing data delegates
             )")
        .def(
            "get_usd_utils",
            [](IDataDelegateInterface* iface)
            {
                auto* usdUtils = iface->getUsdUtils();
                return py::cast(usdUtils, py::return_value_policy::reference);
            },
            R"(
             Get USD utilities for time sampling and prim queries.

             Returns:
                 IUsdUtils: Utilities for working with USD time samples and relationships
             )");
    /**/;

    py::class_<DataDelegate, carb::ObjectPtr<DataDelegate>, PyDataDelegate>(m, "IDataDelegate", R"(
        Base class for custom data delegates that provide field array data to USD prims.

        Data delegates are responsible for loading and providing field array data from various
        sources (files, databases, procedural generation, etc.). Subclass this to implement
        custom data loading logic for your application.

        To implement a custom delegate:
        1. Subclass IDataDelegate in Python
        2. Implement _get_field_array() to return array data for a given prim and time
        3. Implement _can_provide() to indicate if this delegate can handle a prim
        4. Register the delegate with the IDataDelegateRegistry
    )")
        .def(py::init<const std::string&>(),
             R"(
             Create a new data delegate.

             Args:
                 extensionId: Identifier for the extension registering this delegate
             )",
             py::arg("extensionId"))
        .def("get_extension_id", &DataDelegate::getExtensionId, R"(
             Get the extension identifier for this delegate.

             Returns:
                 str: The extension ID provided during construction
             )")
        .def("_get_field_array", &DataDelegate::_get_field_array, R"(
             Get field array data for a specific prim and time (internal method).

             This method must be implemented by subclasses to provide the actual data loading logic.

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The path to the field array prim (as a string)
                 time: The time code to query (as a double)

             Returns:
                 object: A numpy array, CUDA array, or IFieldArray, or None if unavailable
             )",
             py::arg("stageId"), py::arg("primPath"), py::arg("time"))
        .def("_can_provide", &DataDelegate::_can_provide, R"(
             Check if this delegate can provide data for a specific prim (internal method).

             This method must be implemented by subclasses to indicate whether they can handle
             data requests for the given prim.

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The path to the field array prim (as a string)

             Returns:
                 bool: True if this delegate can provide data for the prim, False otherwise
             )",
             py::arg("stageId"), py::arg("primPath"))
        /**/;

    py::class_<IFieldArray, carb::ObjectPtr<IFieldArray>>(m, "IFieldArray", R"(
        N-dimensional array interface for scientific computing data.

        IFieldArray provides a unified interface for multi-dimensional arrays that can reside
        on CPU or GPU (CUDA) devices. It supports NumPy array interface and CUDA array interface
        for seamless integration with existing array libraries.

        The array properties (shape, strides, dtype, device_id) match NumPy conventions,
        making it easy to work with both CPU and GPU arrays in a consistent manner.
    )")
        // expose read-only properties for IFieldArray const API; the names match NumPy arrays.
        .def_property_readonly("shape", &IFieldArray::getShape,
                               "tuple[int, ...]: Shape of the array (number of elements along each dimension)")
        .def_property_readonly("strides", &IFieldArray::getStrides,
                               "tuple[int, ...]: Strides in bytes to move from one element to the next in each dimension")
        .def_property_readonly("ndim", &IFieldArray::getNDims, "int: Number of dimensions in the array")
        .def_property_readonly("dtype", &getDType, "numpy.dtype: Data type of the elements in the array")
        .def_property_readonly("device_id", &IFieldArray::getDeviceId, "int: Device ID (-1 for CPU, >= 0 for CUDA device)")

        // to make `len(farray)` work as it does with numpy arrays
        .def(
            "__len__", [](IFieldArray* self) { return self->getNDims() >= 1 ? self->getShape().at(0) : 0; },
            "Return the size of the first dimension (for len() operator)")

        // support numpy and cuda array interfaces
        .def_property_readonly("__array_interface__", &getArrayInterface,
                               "dict: NumPy array interface for zero-copy interoperability (CPU arrays only)")
        .def_property_readonly("__cuda_array_interface__", &getCudaArrayInterface,
                               "dict: CUDA array interface for zero-copy interoperability (GPU arrays only)")

        // explicit conversions
        .def("numpy", &getNumpyArray, R"(
            Convert to a NumPy array (CPU arrays only).

            Returns:
                numpy.ndarray: A NumPy array view of the data (zero-copy)

            Raises:
                RuntimeError: If the array is on a GPU device
            )")
        .def_static(
            "from_numpy",
            [](py::array array)
            { return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNDArray(array)); },
            R"(
            Create an IFieldArray from a NumPy array.

            Args:
                array: A NumPy array

            Returns:
                IFieldArray: An IFieldArray wrapping the NumPy array (zero-copy)
            )",
            py::arg("array"))
        .def_static(
            "from_array",
            [](py::array array)
            { return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNDArray(array)); },
            R"(
            Create an IFieldArray from a NumPy array.

            Args:
                array: A NumPy array

            Returns:
                IFieldArray: An IFieldArray wrapping the NumPy array (zero-copy)
            )",
            py::arg("array"))
        .def_static(
            "from_array",
            [](py::object obj)
            {
                if (py::hasattr(obj, "__cuda_array_interface__"))
                {
                    CARB_LOG_INFO("using CAI");
                    return carb::stealObject<IFieldArray>(PythonFieldArray::fromCudaArrayInterface(obj));
                }
                else if (py::hasattr(obj, "__array_interface__"))
                {
                    CARB_LOG_INFO("using NAI");
                    return carb::stealObject<IFieldArray>(PythonFieldArray::fromNumpyArrayInterface(obj));
                }
                throw py::value_error("Only objects with NumPy Array Interface or CUDA Array Interface are supported.");
            },
            R"(
            Create an IFieldArray from an object supporting array interfaces.

            This method accepts any object that implements either the NumPy array interface
            (__array_interface__) or CUDA array interface (__cuda_array_interface__).

            Args:
                obj: An object implementing __array_interface__ or __cuda_array_interface__

            Returns:
                IFieldArray: An IFieldArray wrapping the array data (zero-copy)

            Raises:
                ValueError: If the object doesn't support either array interface
            )",
            py::arg("obj"))
        .def(
            "to_device",
            [](carb::ObjectPtr<IFieldArray> self, int32_t targetDeviceId) -> carb::ObjectPtr<IFieldArray>
            {
                // Get the interface and utils
                auto* iface = carb::getCachedInterface<IDataDelegateInterface>();
                if (!iface)
                {
                    throw std::runtime_error("Failed to get IDataDelegateInterface");
                }
                auto* utils = iface->getFieldArrayUtils();
                if (!utils)
                {
                    throw std::runtime_error("Failed to get IFieldArrayUtils");
                }
                return utils->copyToDevice(self, targetDeviceId);
            },
            R"(
            Copy this array to a target device.

            If the array is already on the target device, returns the original array without copying.
            Handles CPU-to-CPU, CPU-to-GPU, GPU-to-CPU, and GPU-to-GPU transfers.

            Args:
                targetDeviceId: Target device ID (-1 for CPU, >= 0 for CUDA device)

            Returns:
                IFieldArray: The array on the target device (either a copy or the original)

            Example:
                # Copy CPU array to GPU 0
                gpu_array = cpu_array.to_device(0)
                # Copy GPU array back to CPU
                cpu_array = gpu_array.to_device(-1)
                # Same device - returns original (no copy)
                same_array = cpu_array.to_device(-1)
                assert same_array is cpu_array
            )",
            py::arg("targetDeviceId"))
        .def(
            "reinterpret",
            [](carb::ObjectPtr<IFieldArray> self, py::object dtype_obj) -> carb::ObjectPtr<IFieldArray>
            {
                // Convert numpy dtype to ElementType using existing helper
                py::dtype dtype = py::dtype::from_args(dtype_obj);
                ElementType targetType = ::getElementType(dtype);

                // Get the interface and utils
                auto* iface = carb::getCachedInterface<IDataDelegateInterface>();
                if (!iface)
                {
                    throw std::runtime_error("Failed to get IDataDelegateInterface");
                }
                auto* utils = iface->getFieldArrayUtils();
                if (!utils)
                {
                    throw std::runtime_error("Failed to get IFieldArrayUtils");
                }
                auto result = utils->reinterpretArray(self, targetType);
                if (!result)
                {
                    throw std::runtime_error("Failed to reinterpret array");
                }
                return result;
            },
            R"(
            Reinterpret array memory as a different type (zero-copy).

            This is a zero-copy operation that reinterprets the raw bytes as a different type.
            The total byte size must be compatible: sizeof(src_type) * src_count must equal
            sizeof(dst_type) * dst_count.

            For multi-dimensional arrays, only the last dimension is adjusted to maintain
            the same total byte size.

            Args:
                dtype: Target numpy dtype (np.float32, np.int32, np.int64, np.float64, etc.)

            Returns:
                IFieldArray: A new view with memory reinterpreted as target type (zero-copy)

            Raises:
                RuntimeError: If the size constraints are violated
                ValueError: If the dtype is not supported

            Example:
                # Reinterpret float32 as int32 (same size)
                int_view = float_array.reinterpret(np.int32)
                # Reinterpret int64 array as two int32 values
                int32_array = int64_array.reinterpret(np.int32)
                # Shape changes: [N] int64 -> [N*2] int32
            )",
            py::arg("dtype"))
        /**/;

    py::class_<IDataDelegateRegistry>(m, "IDataDelegateRegistry", R"(
        Registry for managing data delegates that provide field array data.

        The registry maintains a prioritized list of data delegates and dispatches data
        requests to the appropriate delegate based on their can_provide() responses.
        Delegates with higher priority values are queried first.
    )")
        .def(
            "register_data_delegate",
            [](IDataDelegateRegistry* self, carb::ObjectPtr<DataDelegate> delegate, DelegatePriority priority)
            {
                if (self && delegate)
                {
                    carb::ObjectPtr<IDataDelegate> borrowed = carb::borrowObject<IDataDelegate>(delegate.get());
                    self->registerDataDelegate(borrowed, priority);
                }
            },
            R"(
            Register a data delegate with the registry.

            Higher priority delegates are queried first when looking for data providers.
            Use this to register custom data loaders that can provide field arrays from
            various sources (files, databases, procedural generation, etc.).

            Args:
                delegate: The data delegate to register
                priority: Priority value (higher values = higher priority, default 0)
            )",
            py::arg("delegate"), py::arg("priority") = 0)
        .def(
            "deregister_data_delegate",
            [](IDataDelegateRegistry* self, carb::ObjectPtr<DataDelegate> delegate)
            {
                if (self && delegate)
                {
                    carb::ObjectPtr<IDataDelegate> borrowed = carb::borrowObject<IDataDelegate>(delegate.get());
                    self->deregisterDataDelegate(borrowed);
                }
            },
            R"(
            Deregister a data delegate from the registry.

            Args:
                delegate: The data delegate to deregister
            )",
            py::arg("delegate"))
        .def("deregister_all_data_delegates_for_extension",
             &IDataDelegateRegistry::deregisterAllDataDelegatesForExtension,
             R"(
             Deregister all data delegates registered by a specific extension.

             This is typically called during extension shutdown to clean up all delegates
             that were registered by that extension.

             Args:
                 extensionId: The extension identifier whose delegates should be removed
             )",
             py::arg("extensionId"))
        .def(
            "_get_field_array",
            [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> py::object
            {
                CARB_LOG_INFO(
                    "_get_field_array(%lu, %s, %s)", stageId, primPath.c_str(), time_code_to_string(time).c_str());
                carb::ObjectPtr<IFieldArray> array;
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (stage)
                {

                    // release GIL while we're doing I/O.
                    py::gil_scoped_release release;
                    pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                    pxr::UsdTimeCode timeCode(time);
                    array = registry->getFieldArray(prim, timeCode);
                }
                return array ? py::cast(array) : py::none();
            },
            R"(
            Get field array data for a specific prim and time (internal method).

            This method queries registered delegates to find one that can provide the data,
            then retrieves and returns the field array. The method releases the GIL during
            I/O operations for better concurrency.

            Args:
                stageId: The USD stage ID (as a long int)
                primPath: The path to the field array prim (as a string)
                timeDouble: The time code to query (as a double)

            Returns:
                IFieldArray or None: The field array if available, None otherwise
            )",
            py::arg("stageId"), py::arg("primPath"), py::arg("timeDouble"))
        .def(
            "_is_field_array_cached",
            [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> bool
            {
                CARB_LOG_INFO("_is_field_array_cached(%lu, %s, %s)", stageId, primPath.c_str(),
                              time_code_to_string(time).c_str());
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (stage)
                {
                    pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                    pxr::UsdTimeCode timeCode(time);
                    return registry->isFieldArrayCached(prim, timeCode);
                }
                return false;
            },
            R"(
             Check if a field array is currently cached (internal method).

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The path to the field array prim (as a string)
                 timeDouble: The time code to query (as a double)

             Returns:
                 bool: True if the field array is cached, False otherwise
             )",
            py::arg("stageId"), py::arg("primPath"), py::arg("timeDouble"))
        .def(
            "_get_field_array_async",
            [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> py::object
            {
                CARB_LOG_INFO(
                    "_get_field_array(%lu, %s, %s)", stageId, primPath.c_str(), time_code_to_string(time).c_str());
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (stage)
                {
                    pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                    pxr::UsdTimeCode timeCode(time);
                    IFieldArrayFuture future;
                    {
                        py::gil_scoped_release release; // release GIL while we're doing I/O.
                        future = registry->getFieldArrayAsync(prim, timeCode);
                    }
                    return wrap_future(std::move(future));
                }
                return py::none();
            },
            R"(
            Get field array data asynchronously for a specific prim and time (internal method).

            This method initiates an asynchronous data loading operation and returns a Python
            asyncio Future that will be resolved when the data is available. The GIL is released
            during I/O operations.

            Args:
                stageId: The USD stage ID (as a long int)
                primPath: The path to the field array prim (as a string)
                timeDouble: The time code to query (as a double)

            Returns:
                asyncio.Future[IFieldArray] or None: An asyncio Future that resolves to the field array,
                                                      or None if the stage is invalid
            )",
            py::arg("stageId"), py::arg("primPath"), py::arg("timeDouble"))

        /**/;

    py::class_<IUsdUtils>(m, "IUsdUtils", R"(
        USD utilities for time sampling and relationship queries.

        This class provides helper methods for working with USD time samples and traversing
        relationships between prims, which is essential for properly handling time-varying
        data and tracking dependencies for cache invalidation.
    )")
        .def(
            "get_bracketing_time_samples_for_prim",
            [](IUsdUtils* self, long int stageId, std::string primPath, double time) -> py::tuple
            {
                double lower = 0.0;
                double upper = 0.0;
                bool hasTimeSamples = false;
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (stage)
                {
                    pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                    bool success = self->getBracketingTimeSamplesForPrim(prim, time, &lower, &upper, &hasTimeSamples);
                    if (!success)
                    {
                        const double earliestTime = pxr::UsdTimeCode::EarliestTime().GetValue();
                        return py::make_tuple(earliestTime, earliestTime, false);
                    }
                    return py::make_tuple(lower, upper, hasTimeSamples);
                }
                const double earliestTime = pxr::UsdTimeCode::EarliestTime().GetValue();
                return py::make_tuple(earliestTime, earliestTime, false);
            },
            R"(
             Get bracketing time samples for a given prim.

             This method finds the closest time samples before and after the specified time,
             similar to UsdAttribute::GetBracketingTimeSamples. This is useful for determining
             when to interpolate between keyframes.

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The prim path as a string
                 time: The time code to query

             Returns:
                 tuple[float, float, bool]: A tuple containing:
                     - lower: The time sample at or before the query time
                     - upper: The time sample at or after the query time
                     - had_time_samples: True if time samples exist, False otherwise

                     When no time samples exist or the query fails, returns
                     (EarliestTime, EarliestTime, False).
             )",
            py::arg("stageId"), py::arg("primPath"), py::arg("time"))
        .def(
            "get_bracketing_time_samples_for_data_set_prim",
            [](IUsdUtils* self, long int stageId, std::string primPath, double time,
               bool traverseFieldRelationships) -> py::tuple
            {
                double lower = 0.0;
                double upper = 0.0;
                bool hasTimeSamples = false;
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (stage)
                {
                    pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                    bool success = self->getBracketingTimeSamplesForDataSetPrim(
                        prim, time, traverseFieldRelationships, &lower, &upper, &hasTimeSamples);
                    if (!success)
                    {
                        const double earliestTime = pxr::UsdTimeCode::EarliestTime().GetValue();
                        return py::make_tuple(earliestTime, earliestTime, false);
                    }
                    return py::make_tuple(lower, upper, hasTimeSamples);
                }
                const double earliestTime = pxr::UsdTimeCode::EarliestTime().GetValue();
                return py::make_tuple(earliestTime, earliestTime, false);
            },
            R"(
             Get bracketing time samples for a DataSet prim with optional field relationship traversal.

             This specialized method handles DataSet prims and can optionally traverse field:*
             relationships to determine time samples across the entire dataset structure. This is
             essential for determining when a dataset or its fields have keyframes.

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The prim path as a string
                 time: The time code to query
                 traverseFieldRelationships: If True, traverse field:* relationships to find
                                             time samples across related field prims; if False,
                                             only check the dataset prim itself

             Returns:
                 tuple[float, float, bool]: A tuple containing:
                     - lower: The time sample at or before the query time
                     - upper: The time sample at or after the query time
                     - had_time_samples: True if time samples exist, False otherwise

                     When no time samples exist or the query fails, returns
                     (EarliestTime, EarliestTime, False).
             )",
            py::arg("stageId"), py::arg("primPath"), py::arg("time"), py::arg("traverseFieldRelationships"))
        .def(
            "get_related_data_prims",
            [](IUsdUtils* self, long int stageId, std::string primPath, bool transitive, bool includeSelf) -> py::list
            {
                auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                if (!stage)
                {
                    return py::list();
                }

                pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                std::vector<pxr::UsdPrim> result = self->getRelatedDataPrims(prim, transitive, includeSelf);

                // Convert UsdPrim vector to string vector
                py::list resultPaths;
                for (const auto& p : result)
                {
                    resultPaths.append(p.GetPath().GetString());
                }
                return resultPaths;
            },
            R"(
             Get all related DataSet and FieldArray prims for a given prim.

             This method traverses relationships from the input prim and collects all related prims
             that are either OmniCaeDataSet or OmniCaeFieldArray types. This is essential for:
             - Cache invalidation: Changes to related prims should invalidate cached results
             - Dependency tracking: Understanding which prims depend on others
             - Data provenance: Tracing the source of field data

             Args:
                 stageId: The USD stage ID (as a long int)
                 primPath: The prim path as a string
                 transitive: If True, recursively traverse relationship targets (follow relationships
                            of relationship targets); if False, only return immediate relationship targets
                 includeSelf: If True, include the input prim in the result set; if False, only return
                             related prims (excluding the input prim itself)

             Returns:
                 list[str]: List of prim paths (as strings) for all related DataSet and FieldArray prims.
                           Returns an empty list if the stage is invalid.
             )",
            py::arg("stageId"), py::arg("primPath"), py::arg("transitive") = true, py::arg("includeSelf") = true)
        /**/;
}
} // namespace
