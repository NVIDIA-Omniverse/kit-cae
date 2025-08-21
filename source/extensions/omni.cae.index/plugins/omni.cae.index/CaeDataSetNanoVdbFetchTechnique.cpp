// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

// clang-format off
#include <omni/cae/data/IDataDelegateIncludes.h>
#include <omni/cae/data/DataDelegateUtilsIncludes.h>
// clang-format on

#include "CaeDataSetNanoVdbFetchTechnique.h"

#include <carb/BindingsPythonUtils.h>
#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>

#include <nv/index/ivdb_subset.h>
#include <omni/cae/data/DataDelegateUtils.h>
#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/cae/data/IFieldArray.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <pybind11/attr.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <cuda_runtime.h>
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

namespace omni
{
namespace cae
{
namespace index
{
using omni::cae::data::ElementType;
using omni::cae::data::IFieldArray;
using omni::cae::data::IFieldArrayUtils;

size_t getSizeInBytes(const IFieldArray* array)
{
    if (array)
    {
        auto shape = array->getShape();
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{}) *
               omni::cae::data::getElementSize(array->getElementType());
    }
    return 0u;
}

struct VoxelizationData
{
    carb::ObjectPtr<IFieldArray> array;
    py::object scope;

    ~VoxelizationData()
    {
        py::gil_scoped_acquire acquire;
        array = nullptr;
        scope = py::object{};
    }
};

class CaeDataSetNanoVdbFetchTechnique::Impl
{
public:
    Impl()
    {
        auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        if (!iface)
        {
            CARB_LOG_ERROR("Failed to get IDataDelegateInterface!");
        }
        else
        {
            m_utils = iface->getFieldArrayUtils();
        }
    }
    IFieldArrayUtils* m_utils = nullptr;
    std::shared_ptr<VoxelizationData> m_last_voxels[2];
    std::shared_ptr<VoxelizationData> fetch_nanovdb(mi::Sint32 deviceId,
                                                    const Compute_parameters& params,
                                                    mi::Uint32 attribute_index) const;
};


std::shared_ptr<VoxelizationData> CaeDataSetNanoVdbFetchTechnique::Impl::fetch_nanovdb(mi::Sint32 deviceId,
                                                                                       const Compute_parameters& params,
                                                                                       mi::Uint32 attribute_index) const
{
    namespace py = pybind11;
    using namespace py::literals;

    try
    {
        py::gil_scoped_acquire acquire;
        py::module module = py::module::import("omni.cae.index.impl.helpers");

        auto impl = module.attr("CaeDataSetVoxelizedNanoVdbFetchTechnique")(params.prim_path, params.cache_key);
        py::tuple tuple = impl.attr("fetch_nanovdb")(deviceId, attribute_index);
        if (tuple[0].is_none())
        {
            return nullptr;
        }

        auto voxels = py::cast<carb::ObjectPtr<omni::cae::data::IFieldArray>>(tuple[0]);
        if (!voxels)
        {
            CARB_LOG_ERROR("Failed to cast to IFieldArray!");
            return nullptr;
        }
        auto result = std::make_shared<VoxelizationData>();
        if (voxels->getDeviceId() != deviceId)
        {
            if (!m_utils)
            {
                CARB_LOG_ERROR("Missing FieldArrayUtils!");
                return nullptr;
            }

            CARB_LOG_WARN("`fetch_nanovdb` callback returns data on device=%d. Copying :/", voxels->getDeviceId());
            auto clone = m_utils->createMutableFieldArray(voxels->getElementType(), voxels->getShape(), deviceId);
            if (cudaMemcpy(clone->getMutableData(), voxels->getData(), getSizeInBytes(voxels.get()),
                           cudaMemcpyDefault) != cudaSuccess)
            {
                CARB_LOG_ERROR("Failed cudaMemcpy");
                return nullptr;
            }

            result->array = carb::stealObject<IFieldArray>(clone.detach());
            result->scope = py::object{};
        }
        else
        {
            CARB_LOG_INFO("`fetch_nanovdb` result on target device! Yay!");
            result->array = voxels;
            result->scope = tuple[1];
        }
        return result;
    }
    catch (const py::import_error& e)
    {
        CARB_LOG_ERROR("Failed to import required Python module. Re-run with --info for details.");
        CARB_LOG_INFO("Python Exception:\n%s", e.what());
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("Failed to `fetch_nanovdb`. Re-run with --info for details.");
        CARB_LOG_INFO("Python Exception:\n%s", e.what());
    }
    return nullptr;
}

CaeDataSetNanoVdbFetchTechnique::CaeDataSetNanoVdbFetchTechnique() : m_impl(new Impl())
{
}

CaeDataSetNanoVdbFetchTechnique::CaeDataSetNanoVdbFetchTechnique(const Compute_parameters& params)
    : m_impl(new Impl()), m_params(params)
{
}

CaeDataSetNanoVdbFetchTechnique::~CaeDataSetNanoVdbFetchTechnique()
{
}

const char* CaeDataSetNanoVdbFetchTechnique::get_class_name() const
{
    return "CaeDataSetNanoVdbFetchTechnique";
}

bool CaeDataSetNanoVdbFetchTechnique::is_gpu_operation() const
{
    // since're producing device data.
    return true;
}

nv::index::IDistributed_compute_technique::Invocation_mode CaeDataSetNanoVdbFetchTechnique::get_invocation_mode() const
{
    // Do not use INDIVIDUAL, this would launch this technique in parallel for all subsets.
    // But device access needs to be synchronized, i.e. multiple subsets per device would not work.
    return nv::index::IDistributed_compute_technique::GROUPED_PER_DEVICE;
}

mi::neuraylib::IElement* CaeDataSetNanoVdbFetchTechnique::copy() const
{
    CaeDataSetNanoVdbFetchTechnique* other = new CaeDataSetNanoVdbFetchTechnique();
    other->m_params = m_params;
    return other;
}

void CaeDataSetNanoVdbFetchTechnique::serialize(mi::neuraylib::ISerializer* serializer) const
{
    mi::Size len = m_params.prim_path.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_params.prim_path.c_str()), len);

    len = m_params.cache_key.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_params.cache_key.c_str()), len);

    serializer->write(&m_params.execution_tag);
    serializer->write(&m_params.enable_interpolation);
}

void CaeDataSetNanoVdbFetchTechnique::deserialize(mi::neuraylib::IDeserializer* deserializer)
{
    mi::Size len;
    std::vector<char> buffer;
    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_params.prim_path = &buffer[0];

    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_params.cache_key = &buffer[0];

    deserializer->read(&m_params.execution_tag);
    deserializer->read(&m_params.enable_interpolation);
}

void CaeDataSetNanoVdbFetchTechnique::launch_compute(mi::neuraylib::IDice_transaction* dice_transaction,
                                                     nv::index::IDistributed_compute_destination_buffer* dst_buffer) const
{
    CARB_LOG_INFO("launch_compute (self=%p, tid=%s)", this, getThreadId().c_str());

    auto nvdb_dst_buffer =
        mi::base::make_handle(dst_buffer->get_interface<nv::index::IDistributed_compute_destination_buffer_VDB>());
    if (!nvdb_dst_buffer)
    {
        CARB_LOG_ERROR("CaeDataSetNanoVdbFetchTechnique: Unable to retrieve valid destination buffer interface.");
        return;
    }

    auto nvdb_subset = mi::base::make_handle<nv::index::IVDB_subset>(nvdb_dst_buffer->get_distributed_data_subset());
    auto nvdb_subset_device = mi::base::make_handle<nv::index::IVDB_subset_device>(nvdb_subset->get_device_subset());
    if (!nvdb_subset_device.is_valid_interface())
    {
        CARB_LOG_ERROR("CaeDataSetNanoVdbFetchTechnique: Unable to access IVDB_subset_device interface for subset");
        return;
    }

    CARB_LOG_INFO("begin launch_compute");
    const std::vector<mi::Uint32> attribute_indices =
        m_params.enable_interpolation ? std::vector<mi::Uint32>{ 0, 1 } : std::vector<mi::Uint32>{ 0 };

    for (auto& attribute_index : attribute_indices)
    {
        if (auto data = m_impl->fetch_nanovdb(nvdb_subset_device->get_device_id(), m_params, attribute_index))
        {
            if (!nvdb_subset_device->adopt_grid_buffer(
                    attribute_index, const_cast<void*>(data->array->getData()), getSizeInBytes(data->array.get())))
            {
                CARB_LOG_ERROR("CaeDataSetNanoVdbFetchTechnique: Failed to adopt device buffer");
            }
            else
            {
                // need to preserve until not needed.
                m_impl->m_last_voxels[attribute_index] = data;
            }
        }
        else
        {
            CARB_LOG_ERROR("fetch_nanovdb failed!");
        }
    }

    CARB_LOG_INFO("end launch_compute");
}

} // namespace index
} // namespace cae
} // namespace omni
