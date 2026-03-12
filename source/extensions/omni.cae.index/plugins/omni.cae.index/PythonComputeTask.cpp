// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "PythonComputeTask.h"

#include <carb/logging/Log.h>

#include <nv/index/idistributed_compute_destination_buffer.h>
#include <pybind11/attr.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <sstream>
#include <thread>

namespace
{

std::string getThreadId()
{
    std::thread::id this_id = std::this_thread::get_id();
    std::ostringstream ss;
    ss << this_id;
    return ss.str();
}

} // namespace

namespace py = pybind11;

namespace omni
{
namespace cae
{
namespace index
{

class PythonComputeTask::Impl
{
    bool m_populated = false;
    std::unique_ptr<py::object> m_compute_task;

public:
    bool populate(const Compute_parameters& params);
    void launch_compute(mi::neuraylib::IDice_transaction* dice_transaction,
                        nv::index::IDistributed_compute_destination_buffer* dst_buffer);

    ~Impl()
    {
        if (m_compute_task)
        {
            py::gil_scoped_acquire acquire;
            m_compute_task.reset();
        }
    }
};

bool PythonComputeTask::Impl::populate(const Compute_parameters& params)
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
            m_compute_task.reset(new py::object(mdl.attr(params.python_class.c_str())(params_dict)));
        }
        catch (const py::import_error& e)
        {
            CARB_LOG_ERROR("PythonComputeTask: Failed to import required Python module. Re-run with --info for details.");
            CARB_LOG_INFO("Python Exception:\n%s", e.what());
        }
        catch (const py::error_already_set& e)
        {
            CARB_LOG_ERROR("PythonComputeTask: Failed to populate. Re-run with --info for details.");
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

    return m_compute_task && !m_compute_task->is_none();
}

void PythonComputeTask::Impl::launch_compute(mi::neuraylib::IDice_transaction* dice_transaction,
                                             nv::index::IDistributed_compute_destination_buffer* dst_buffer)
{
    try
    {
        py::gil_scoped_acquire acquire;
        if (auto vdb_dst_buffer = mi::base::make_handle(
                dst_buffer->get_interface<nv::index::IDistributed_compute_destination_buffer_VDB>()))
        {
            m_compute_task->attr("launch_compute")(vdb_dst_buffer.get());
        }
        else if (auto irregular_volume_dst_buffer = mi::base::make_handle(
                     dst_buffer->get_interface<nv::index::IDistributed_compute_destination_buffer_irregular_volume>()))
        {
            m_compute_task->attr("launch_compute")(irregular_volume_dst_buffer.get());
        }
        else
        {
            CARB_LOG_ERROR("PythonComputeTask: Failed to launch_compute. Unsupported destination buffer type.");
            return;
        }
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("PythonComputeTask: Failed to launch_compute. %s", e.what());
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("C++ exception in launch_compute: %s", e.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("Unknown exception in launch_compute");
    }
}

PythonComputeTask::PythonComputeTask() : m_impl(new Impl())
{
}

PythonComputeTask::PythonComputeTask(const Compute_parameters& params) : m_impl(new Impl()), m_params(params)
{
}

PythonComputeTask::~PythonComputeTask()
{
}

const char* PythonComputeTask::get_class_name() const
{
    return "PythonComputeTask";
}

bool PythonComputeTask::is_gpu_operation() const
{
    return m_params.is_gpu_operation;
}

nv::index::IDistributed_compute_technique::Invocation_mode PythonComputeTask::get_invocation_mode() const
{
    // Do not use INDIVIDUAL, this would launch this technique in parallel for all subsets.
    // Use GROUPED_PER_DEVICE to synchronize device access across subsets.
    return nv::index::IDistributed_compute_technique::GROUPED_PER_DEVICE;
}

mi::neuraylib::IElement* PythonComputeTask::copy() const
{
    PythonComputeTask* other = new PythonComputeTask();
    other->m_params = m_params;
    return other;
}

void PythonComputeTask::serialize(mi::neuraylib::ISerializer* serializer) const
{
    // Serialize python_module
    mi::Size len = m_params.python_module.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_params.python_module.c_str()), len);

    // Serialize python_class
    len = m_params.python_class.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_params.python_class.c_str()), len);

    // Serialize is_gpu_operation
    serializer->write(&m_params.is_gpu_operation);

    // Serialize params map
    mi::Size map_size = m_params.params.size();
    serializer->write(&map_size);
    for (const auto& param : m_params.params)
    {
        // Serialize key
        len = param.first.size() + 1;
        serializer->write(&len);
        serializer->write(reinterpret_cast<const mi::Uint8*>(param.first.c_str()), len);

        // Serialize value
        len = param.second.size() + 1;
        serializer->write(&len);
        serializer->write(reinterpret_cast<const mi::Uint8*>(param.second.c_str()), len);
    }
}

void PythonComputeTask::deserialize(mi::neuraylib::IDeserializer* deserializer)
{
    mi::Size len;
    std::vector<char> buffer;

    // Deserialize python_module
    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_params.python_module = &buffer[0];

    // Deserialize python_class
    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_params.python_class = &buffer[0];

    // Deserialize is_gpu_operation
    deserializer->read(&m_params.is_gpu_operation);

    // Deserialize params map
    mi::Size map_size;
    deserializer->read(&map_size);
    m_params.params.clear();
    for (mi::Size i = 0; i < map_size; ++i)
    {
        // Deserialize key
        deserializer->read(&len);
        buffer.resize(len, '\0');
        deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
        std::string key = &buffer[0];

        // Deserialize value
        deserializer->read(&len);
        buffer.resize(len, '\0');
        deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
        std::string value = &buffer[0];

        m_params.params[key] = value;
    }
}

void PythonComputeTask::launch_compute(mi::neuraylib::IDice_transaction* dice_transaction,
                                       nv::index::IDistributed_compute_destination_buffer* dst_buffer) const
{
    CARB_LOG_INFO("PythonComputeTask::launch_compute (self=%p, tid=%s)", this, getThreadId().c_str());

    // Populate the Python object if not already done
    if (!m_impl->populate(m_params))
    {
        CARB_LOG_ERROR("PythonComputeTask: Failed to populate Python compute task");
        return;
    }

    CARB_LOG_INFO("PythonComputeTask::launch_compute - Python object populated successfully");
    CARB_LOG_INFO("  Module: %s", m_params.python_module.c_str());
    CARB_LOG_INFO("  Class: %s", m_params.python_class.c_str());
    CARB_LOG_INFO("  Is GPU operation: %s", m_params.is_gpu_operation ? "true" : "false");
    CARB_LOG_INFO("  Parameters:");
    for (const auto& param : m_params.params)
    {
        CARB_LOG_INFO("    %s = %s", param.first.c_str(), param.second.c_str());
    }

    // Call the Python launch_compute method
    m_impl->launch_compute(dice_transaction, dst_buffer);

    CARB_LOG_INFO("PythonComputeTask::launch_compute - Complete");
}

} // namespace index
} // namespace cae
} // namespace omni
