// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

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
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <cuda_runtime.h>
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

pxr::UsdStageCache::Id GetStageId(pxr::UsdStageRefPtr stage)
{
    return pxr::UsdUtilsStageCache::Get().GetId(stage);
}

pxr::UsdStageRefPtr GetStage(pxr::UsdStageCache::Id id)
{
    return pxr::UsdUtilsStageCache::Get().Find(id);
}

auto getDType(omni::cae::data::IFieldArray* self)
{
    switch (self->getElementType())
    {
    case ElementType::int32:
        return py::dtype::of<int32_t>();
    case ElementType::uint32:
        return py::dtype::of<uint32_t>();
    case ElementType::int64:
        return py::dtype::of<int64_t>();
    case ElementType::uint64:
        return py::dtype::of<uint64_t>();
    case ElementType::float32:
        return py::dtype::of<float>();
    case ElementType::float64:
        return py::dtype::of<double>();
    default:
        return py::dtype("V0");
    }
}

auto getElementType(py::dtype dtype)
{
    ElementType etype;
    if (dtype.is(py::dtype::of<int32_t>()))
    {
        etype = omni::cae::data::ElementType::int32;
    }
    else if (dtype.is(py::dtype::of<int64_t>()))
    {
        etype = omni::cae::data::ElementType::int64;
    }
    else if (dtype.is(py::dtype::of<uint32_t>()))
    {
        etype = omni::cae::data::ElementType::uint32;
    }
    else if (dtype.is(py::dtype::of<uint64_t>()))
    {
        etype = omni::cae::data::ElementType::uint64;
    }
    else if (dtype.is(py::dtype::of<float>()))
    {
        etype = omni::cae::data::ElementType::float32;
    }
    else if (dtype.is(py::dtype::of<double>()))
    {
        etype = omni::cae::data::ElementType::float64;
    }
    else
    {
        CARB_LOG_ERROR("Unsupported dtype: '%c', '%s'", dtype.kind(), std::string(py::str(dtype)).c_str());
        throw py::value_error("Unsupported array type");
    }
    return etype;
}

auto getFormatDescriptor(omni::cae::data::ElementType etype)
{
    using namespace omni::cae::data;
    switch (etype)
    {
    case ElementType::int32:
        return py::format_descriptor<int32_t>::format();
    case ElementType::uint32:
        return py::format_descriptor<uint32_t>::format();
    case ElementType::int64:
        return py::format_descriptor<int64_t>::format();
    case ElementType::uint64:
        return py::format_descriptor<uint64_t>::format();
    case ElementType::float32:
        return py::format_descriptor<float>::format();
    case ElementType::float64:
        return py::format_descriptor<double>::format();
    default:
        throw std::invalid_argument("Unsupported element type.");
    }
}

std::string getTypeStr(ElementType etype)
{
    switch (etype)
    {
    case ElementType::int32:
        return "<i4";
    case ElementType::uint32:
        return "<u4";
    case ElementType::int64:
        return "<i8";
    case ElementType::uint64:
        return "<u8";
    case ElementType::float32:
        return "<f4";
    case ElementType::float64:
        return "<f8";
    default:
        throw std::invalid_argument("Unsupported element type.");
    }
}

py::dict getArrayInterface(const IFieldArray* self)
{
    if (self->getDeviceId() != -1)
    {
        throw py::attribute_error(std::string("__array__interface__ not supported by array hosted on device ") +
                                  std::to_string(self->getDeviceId()));
    }

    const auto shape = self->getShape();
    const auto strides = self->getStrides();

    py::dict iface;
    iface["data"] = py::make_tuple(reinterpret_cast<const uintptr_t>(self->getData()), /*readOnly*/ true);
    iface["shape"] = py::tuple(py::cast(shape));
    iface["strides"] = py::tuple(py::cast(strides));
    iface["typestr"] = getTypeStr(self->getElementType());
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

    py::dict iface;
    iface["data"] = py::make_tuple(reinterpret_cast<const uintptr_t>(self->getData()), /*readOnly*/ true);
    iface["shape"] = py::tuple(py::cast(shape));
    iface["strides"] = py::tuple(py::cast(strides));
    iface["typestr"] = getTypeStr(self->getElementType());
    iface["version"] = 2;
    return iface;
}


py::buffer_info getBufferInfo(const omni::cae::data::IFieldArray* self)
{
    return py::buffer_info(const_cast<void*>(self->getData()), getElementSize(self->getElementType()),
                           getFormatDescriptor(self->getElementType()), self->getNDims(), self->getShape(),
                           self->getStrides(), true);
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
    py::enum_<ElementType>(m, "ElementType", "Element Types")
        .value("int32", ElementType::int32)
        .value("int64", ElementType::int64)
        .value("uint32", ElementType::uint32)
        .value("uint64", ElementType::uint64)
        .value("float32", ElementType::float32)
        .value("float64", ElementType::float64);

    carb::defineInterfaceClass<IDataDelegateInterface>(m, "IDataDelegateInterface", "acquire_data_delegate_interface",
                                                       "release_data_delegate_interface", "Data Delegate Interface")
        .def("get_data_delegate_registry",
             [](IDataDelegateInterface* iface)
             {
                 auto* registry = iface->getDataDelegateRegistry();
                 return py::cast(registry, py::return_value_policy::reference);
             });
    /**/;

    py::class_<DataDelegate, carb::ObjectPtr<DataDelegate>, PyDataDelegate>(m, "IDataDelegate")
        .def(py::init<const std::string&>())
        .def("get_extension_id", &DataDelegate::getExtensionId)
        .def("_get_field_array", &DataDelegate::_get_field_array)
        .def("_can_provide", &DataDelegate::_can_provide)
        /**/;

    py::class_<IFieldArray, carb::ObjectPtr<IFieldArray>>(m, "IFieldArray")
        // expose read-only properties for IFieldArray const API; the names match NumPy arrays.
        .def_property_readonly("shape", &IFieldArray::getShape, "Shape of the array")
        .def_property_readonly("strides", &IFieldArray::getStrides,
                               "The step-size required to move from one element to the next in memory.")
        .def_property_readonly("ndim", &IFieldArray::getNDims, "The arrays number of dimensions.")
        .def_property_readonly("dtype", &getDType, "Describes the format of the elements in the array.")
        .def_property_readonly("device_id", &IFieldArray::getDeviceId, "Device id, -1 for CPU and >= 0 for CUDA device.")

        // to make `size(farray)` work as it does with numpy arrays
        .def("__len__", [](IFieldArray* self) { return self->getNDims() >= 1 ? self->getShape().at(0) : 0; })

        // support numpy and cuda array interfaces
        .def_property_readonly("__array_interface__", &getArrayInterface)
        .def_property_readonly("__cuda_array_interface__", &getCudaArrayInterface)

        // explicit converions
        .def("numpy", &getNumpyArray) // TODO: remove?
        .def_static("from_numpy", [](py::array array)
                    { return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNDArray(array)); })
        .def_static("from_array", [](py::array array)
                    { return carb::stealObject<omni::cae::data::IFieldArray>(PythonFieldArray::fromNDArray(array)); })
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
            })
        /**/;

    py::class_<IDataDelegateRegistry>(m, "IDataDelegateRegistry", "Data Delegate Registry")
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
            Create and register a data delegate.

            Args:
                delegate: The data delegate to register.
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
            Create and register a data delegate.

            Args:
                delegate: The data delegate to register.
            )",
            py::arg("delegate"))
        .def("deregister_all_data_delegates_for_extension",
             &IDataDelegateRegistry::deregisterAllDataDelegatesForExtension,
             "Deregister all data loaders that were registered by the specified "
             "extension.",
             py::arg("extensionId"))
        .def(
            "_get_field_array",
            [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> py::object
            {
                CARB_LOG_INFO("_get_field_array(%lu, %s, %f)", stageId, primPath.c_str(), time);
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
            Get field array given the prim and time.

            Return:
                numpy.NDArray or None.
            )",
            py::arg("stageId"), py::arg("primPath"), py::arg("timeDouble"))
        .def("_is_field_array_cached",
             [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> bool
             {
                 CARB_LOG_INFO("_is_field_array_cached(%lu, %s, %f)", stageId, primPath.c_str(), time);
                 auto stage = GetStage(pxr::UsdStageCache::Id::FromLongInt(stageId));
                 if (stage)
                 {
                     pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(primPath));
                     pxr::UsdTimeCode timeCode(time);
                     return registry->isFieldArrayCached(prim, timeCode);
                 }
                 return false;
             })
        .def(
            "_get_field_array_async",
            [](IDataDelegateRegistry* registry, long int stageId, std::string primPath, double time) -> py::object
            {
                CARB_LOG_INFO("_get_field_array(%lu, %s, %f)", stageId, primPath.c_str(), time);
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
            Get field array given the prim and time.

            Return:
                numpy.NDArray or None.
            )",
            py::arg("stageId"), py::arg("primPath"), py::arg("timeDouble"))

        /**/;
}
} // namespace
