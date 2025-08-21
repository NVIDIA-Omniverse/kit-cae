// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#include "FieldArrayUtils.h"

#include <carb/ObjectUtils.h>
#include <carb/logging/Log.h>
#include <carb/tasking/ITasking.h>

#include <omni/cae/data/FieldArrayDispatch.h>

#include <algorithm>
#include <cassert>
#include <cuda_runtime.h>
#include <inttypes.h>
#include <numeric>

namespace omni
{
namespace cae
{
namespace data
{

class CpuMutableFieldArray final : public IMutableFieldArray
{
    CARB_IOBJECT_IMPL
public:
    // FIXME: make shape / strides 64 bit
    CpuMutableFieldArray(ElementType type, const std::vector<uint64_t>& shape, const std::vector<uint64_t>& strides)
        : m_elementType(type),
          m_shape(shape),
          m_strides(strides),
          m_buffer(std::accumulate(shape.begin(), shape.end(), 1llu, std::multiplies<uint64_t>()) * getElementSize(type))
    {
        if (shape.size() != strides.size())
        {
            throw std::runtime_error("Invalid shape and strides sizes!");
        }
    }

    uint32_t getNDims() const override
    {
        return static_cast<uint32_t>(m_shape.size());
    }

    const void* getData() const override
    {
        return m_buffer.data();
    }
    std::vector<uint64_t> getShape() const override
    {
        return m_shape;
    }
    std::vector<uint64_t> getStrides() const override
    {
        return m_strides;
    }
    ElementType getElementType() const override
    {
        return m_elementType;
    }
    void* getMutableData() override
    {
        return m_buffer.data();
    }

    uint64_t getMutableDataSizeInBytes() const override
    {
        return static_cast<uint64_t>(m_buffer.size());
    }

    int32_t getDeviceId() const override
    {
        return -1;
    }

private:
    ElementType m_elementType;
    std::vector<uint64_t> m_shape;
    std::vector<uint64_t> m_strides;
    std::vector<uint8_t> m_buffer;
};

class ScopedCudaDevice
{
public:
    ScopedCudaDevice(int32_t id)
    {
        cudaGetDevice(&m_prev_dev);
        if (id != m_prev_dev)
        {
            cudaSetDevice(id);
        }
        else
        {
            m_prev_dev = -1;
        }
    }

    ~ScopedCudaDevice()
    {
        if (m_prev_dev >= 0)
        {
            cudaSetDevice(m_prev_dev);
        }
    }

private:
    int32_t m_prev_dev = -1;
};

class CudaMutableFieldArray final : public IMutableFieldArray
{
    CARB_IOBJECT_IMPL

public:
    CudaMutableFieldArray(ElementType type,
                          const std::vector<uint64_t>& shape,
                          const std::vector<uint64_t>& strides,
                          int32_t deviceId)
        : m_elementType(type), m_shape(shape), m_strides(strides), m_buffer(nullptr), m_bufferSize(0u), m_deviceId(deviceId)
    {
        const uint64_t bufferSize =
            std::accumulate(shape.begin(), shape.end(), static_cast<uint64_t>(1), std::multiplies<uint64_t>{}) *
            static_cast<uint64_t>(getElementSize(m_elementType));
        if (bufferSize > 0)
        {
            ScopedCudaDevice nvdb_device_scope(m_deviceId);
            if (cudaMalloc(&m_buffer, bufferSize) != cudaSuccess)
            {
                throw std::runtime_error("Failed cudaMalloc");
            }
            m_bufferSize = bufferSize;
        }
    }

    ~CudaMutableFieldArray()
    {
        if (m_buffer != nullptr)
        {
            ScopedCudaDevice nvdb_device_scope(m_deviceId);
            cudaFree(m_buffer);
            m_buffer = nullptr;
        }
    }

    int32_t getDeviceId() const override
    {
        return m_deviceId;
    }

    uint32_t getNDims() const override
    {
        return static_cast<uint32_t>(m_shape.size());
    }

    const void* getData() const override
    {
        return m_buffer;
    }

    void* getMutableData() override
    {
        return m_buffer;
    }

    std::vector<uint64_t> getShape() const override
    {
        return m_shape;
    }

    std::vector<uint64_t> getStrides() const override
    {
        return m_strides;
    }

    ElementType getElementType() const override
    {
        return m_elementType;
    }

    uint64_t getMutableDataSizeInBytes() const override
    {
        return m_bufferSize;
    }

private:
    ElementType m_elementType;
    std::vector<uint64_t> m_shape;
    std::vector<uint64_t> m_strides;
    void* m_buffer = nullptr;
    uint64_t m_bufferSize = 0;
    int32_t m_deviceId;
};


carb::ObjectPtr<IMutableFieldArray> FieldArrayUtils::createMutableFieldArray(ElementType type,
                                                                             const std::vector<uint64_t>& shape,
                                                                             int32_t deviceId,
                                                                             Order order)
{
    const int ndims = static_cast<int>(shape.size());
    std::vector<uint64_t> strides(ndims, 1);
    const auto elementSize = getElementSize(type);

    if (order == Order::c)
    {
        // C-order (row-major)
        for (int i = ndims - 2; i >= 0; --i)
        {
            strides.at(i) = strides.at(i + 1) * shape.at(i + 1);
        }
    }
    else
    {
        // Fortran-order (column-major)
        for (int i = 1; i < ndims; ++i)
        {
            strides.at(i) = strides.at(i - 1) * shape.at(i - 1);
        }
    }

    // convert to bytes.
    std::transform(
        strides.begin(), strides.end(), strides.begin(), [&elementSize](uint64_t s) { return s * elementSize; });
    if (deviceId == -1)
    {
        return carb::stealObject<IMutableFieldArray>(new CpuMutableFieldArray(type, shape, strides));
    }
    else if (deviceId >= 0)
    {
        return carb::stealObject<IMutableFieldArray>(new CudaMutableFieldArray(type, shape, strides, deviceId));
    }
    else
    {
        CARB_LOG_ERROR("Invalid device id '%d'", deviceId);
        return nullptr;
    }
}

} // namespace data
} // namespace cae
} // namespace omni
