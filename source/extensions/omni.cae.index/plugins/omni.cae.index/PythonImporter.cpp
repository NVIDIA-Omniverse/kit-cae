// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.


#include "PythonImporter.h"

#include "DistributedDataProperties.h"

#include <carb/logging/Log.h>

#include <pybind11/attr.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace omni
{
namespace cae
{
namespace index
{

using omni::cae::data::IFieldArray;

class PythonImporter::Impl
{
    bool m_populated = false;
    std::unique_ptr<py::object> m_importer;

public:
    bool populate(const Importer_parameters& params);
    mi::math::Bbox<mi::Float32, 3> get_bounds();
    nv::index::IDistributed_data_subset* create_subset(const mi::math::Bbox<mi::Float32, 3>& bbox,
                                                       nv::index::IData_subset_factory* factory);

    ~Impl()
    {
        if (m_importer)
        {
            py::gil_scoped_acquire acquire;
            m_importer.reset();
        }
    }
};

bool PythonImporter::Impl::populate(const Importer_parameters& params)
{
    if (!m_populated)
    {
        m_populated = true;
        try
        {
            py::gil_scoped_acquire acquire;
            py::module mdl = py::module::import(params.python_module.c_str());
            py::dict params_dict;
            for (const auto& param : params.params)
            {
                params_dict[param.first.c_str()] = param.second.c_str();
            }
            m_importer.reset(new py::object(mdl.attr(params.python_class.c_str())(params_dict)));
        }
        catch (const py::import_error& e)
        {
            CARB_LOG_ERROR("Failed to import required Python module. Re-run with --info for details.");
            CARB_LOG_INFO("Python Exception:\n%s", e.what());
        }
        catch (const py::error_already_set& e)
        {
            CARB_LOG_ERROR("Failed to populate. Re-run with --info for details.");
            CARB_LOG_INFO("Python Exception:\n%s", e.what());
        }
        catch (const std::exception& e)
        {
            CARB_LOG_ERROR("C++ exception in populate: %s", e.what());
        }
        catch (...)
        {
            CARB_LOG_ERROR("Unknown exception in populate");
        }
    }

    return m_importer && !m_importer->is_none();
}

mi::math::Bbox<mi::Float32, 3> PythonImporter::Impl::get_bounds()
{
    try
    {
        py::gil_scoped_acquire acquire;
        return m_importer->attr("get_bounds")().cast<mi::math::Bbox<mi::Float32, 3>>();
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("Failed to get_bounds. Re-run with --info for details.");
        CARB_LOG_ERROR("Python Exception:\n%s", e.what());
        return {};
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("C++ exception in get_bounds: %s", e.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("Unknown exception in get_bounds");
    }
    return {};
}

nv::index::IDistributed_data_subset* PythonImporter::Impl::create_subset(const mi::math::Bbox<mi::Float32, 3>& bbox,
                                                                         nv::index::IData_subset_factory* factory)
{
    try
    {
        py::gil_scoped_acquire acquire;

        // Call Python create_subset method
        auto result = m_importer->attr("create_subset")(bbox, factory);

        if (result.is_none())
        {
            CARB_LOG_ERROR("Python create_subset returned None");
            return nullptr;
        }

        // Cast to raw pointer (pybind11 uses raw pointers with inheritance)
        auto* raw_ptr = result.cast<nv::index::IIrregular_volume_subset*>();
        if (!raw_ptr)
        {
            CARB_LOG_ERROR("Python create_subset returned null pointer");
            return nullptr;
        }

        // Convert to base class pointer and retain for caller
        nv::index::IDistributed_data_subset* base_ptr = raw_ptr;
        base_ptr->retain();
        return base_ptr;
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("Python exception in create_subset:");
        CARB_LOG_ERROR("%s", e.what());
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("C++ exception in create_subset: %s", e.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("Unknown exception in create_subset");
    }
    return nullptr;
}

PythonImporter::PythonImporter() : PythonImporter(Importer_parameters())
{
}

PythonImporter::PythonImporter(const Importer_parameters& params)
    : m_importer_params(params), m_impl(new PythonImporter::Impl())
{
}

PythonImporter::~PythonImporter()
{
    py::gil_scoped_acquire acquire;
    m_impl.reset();
}

void PythonImporter::set_verbose(bool enable)
{
    m_verbose = enable;
}

bool PythonImporter::get_verbose() const
{
    return m_verbose;
}

void PythonImporter::serialize(mi::neuraylib::ISerializer* serializer) const
{
    mi::Size len = m_importer_params.python_module.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_importer_params.python_module.c_str()), len);

    len = m_importer_params.python_class.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_importer_params.python_class.c_str()), len);

    len = m_importer_params.params.size();
    serializer->write(&len);
    for (const auto& param : m_importer_params.params)
    {
        mi::Size key_len = param.first.size() + 1;
        serializer->write(&key_len);
        serializer->write(reinterpret_cast<const mi::Uint8*>(param.first.c_str()), key_len);
        mi::Size val_len = param.second.size() + 1;
        serializer->write(&val_len);
        serializer->write(reinterpret_cast<const mi::Uint8*>(param.second.c_str()), val_len);
    }
}

void PythonImporter::deserialize(mi::neuraylib::IDeserializer* deserializer)
{
    mi::Size len;
    std::vector<char> buffer;

    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_importer_params.python_module = &buffer[0];

    deserializer->read(&len);
    buffer.resize(0);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_importer_params.python_class = &buffer[0];

    deserializer->read(&len);
    for (mi::Size i = 0; i < len; ++i)
    {
        mi::Size key_len;
        deserializer->read(&key_len);
        buffer.resize(0);
        buffer.resize(key_len, '\0');
        deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), key_len);
        std::string key = &buffer[0];
        mi::Size val_len;
        deserializer->read(&val_len);
        buffer.resize(0);
        buffer.resize(val_len, '\0');
        deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), val_len);
        std::string val = &buffer[0];
        m_importer_params.params[key] = val;
    }
}

const nv::index::IDistributed_data_properties* PythonImporter::get_dataset_properties() const
{
    if (m_impl->populate(m_importer_params))
    {
        return new DistributedDataProperties(m_impl->get_bounds());
    }
    return nullptr;
}

mi::Size PythonImporter::estimate(const mi::math::Bbox_struct<mi::Float32, 3>& bounding_box,
                                  mi::neuraylib::IDice_transaction* dice_transaction) const
{
    return 0ull;
}

nv::index::IDistributed_data_subset* PythonImporter::create(const mi::math::Bbox_struct<mi::Float32, 3>& bbox,
                                                            nv::index::IData_subset_factory* factory,
                                                            mi::neuraylib::IDice_transaction* dice_transaction) const
{
    if (m_impl->populate(m_importer_params))
    {
        nv::index::IDistributed_data_subset* subset = m_impl->create_subset(bbox, factory);
        if (subset)
        {
            return subset;
        }
    }

    CARB_LOG_ERROR("PythonImporter::create failed!");
    return nullptr;
}


} // namespace index
} // namespace cae
} // namespace omni
