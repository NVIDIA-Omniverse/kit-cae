// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "FieldArrayUtils.h"

#include <carb/ObjectUtils.h>
#include <carb/logging/Log.h>
#include <carb/tasking/ITasking.h>

#include <omni/cae/data/FieldArrayDispatch.h>

#include <algorithm>
#include <cassert>
#include <cstring>
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

carb::ObjectPtr<IFieldArray> FieldArrayUtils::copyToDevice(carb::ObjectPtr<IFieldArray> array, int32_t targetDeviceId)
{
    if (!array)
    {
        CARB_LOG_ERROR("Cannot copy null array");
        return nullptr;
    }

    // If already on target device, return the original array
    if (array->getDeviceId() == targetDeviceId)
    {
        return array;
    }

    // Create a new mutable array on the target device
    auto targetArray = createMutableFieldArray(array->getElementType(), array->getShape(), targetDeviceId, Order::c);
    if (!targetArray)
    {
        CARB_LOG_ERROR("Failed to create target array on device %d", targetDeviceId);
        return nullptr;
    }

    const int32_t srcDeviceId = array->getDeviceId();
    const void* srcData = array->getData();
    void* dstData = targetArray->getMutableData();
    const uint64_t sizeInBytes = targetArray->getMutableDataSizeInBytes();

    // Perform the appropriate memory copy based on source and destination devices
    cudaError_t result = cudaSuccess;

    if (srcDeviceId == -1 && targetDeviceId == -1)
    {
        // CPU to CPU
        std::memcpy(dstData, srcData, sizeInBytes);
    }
    else if (srcDeviceId == -1 && targetDeviceId >= 0)
    {
        // CPU to GPU
        ScopedCudaDevice device_scope(targetDeviceId);
        result = cudaMemcpy(dstData, srcData, sizeInBytes, cudaMemcpyHostToDevice);
    }
    else if (srcDeviceId >= 0 && targetDeviceId == -1)
    {
        // GPU to CPU
        ScopedCudaDevice device_scope(srcDeviceId);
        result = cudaMemcpy(dstData, srcData, sizeInBytes, cudaMemcpyDeviceToHost);
    }
    else
    {
        // GPU to GPU (might be same or different device)
        if (srcDeviceId == targetDeviceId)
        {
            // Same GPU device
            ScopedCudaDevice device_scope(srcDeviceId);
            result = cudaMemcpy(dstData, srcData, sizeInBytes, cudaMemcpyDeviceToDevice);
        }
        else
        {
            // Different GPU devices - use peer-to-peer if available
            ScopedCudaDevice device_scope(targetDeviceId);
            result = cudaMemcpyPeer(dstData, targetDeviceId, srcData, srcDeviceId, sizeInBytes);
        }
    }

    if (result != cudaSuccess)
    {
        CARB_LOG_ERROR("Failed to copy array from device %d to device %d: %s", srcDeviceId, targetDeviceId,
                       cudaGetErrorString(result));
        return nullptr;
    }

    return carb::stealObject<IFieldArray>(targetArray.detach());
}

// Zero-copy wrapper that reinterprets an existing array as a different type
class ReinterpretedFieldArray final : public IFieldArray
{
    CARB_IOBJECT_IMPL

public:
    ReinterpretedFieldArray(carb::ObjectPtr<IFieldArray> sourceArray,
                            ElementType targetType,
                            const std::vector<uint64_t>& newShape)
        : m_sourceArray(sourceArray), m_targetType(targetType), m_shape(newShape)
    {
        // Calculate new strides based on target type
        const size_t targetElementSize = getElementSize(targetType);
        m_strides.resize(newShape.size());
        if (!newShape.empty())
        {
            m_strides[newShape.size() - 1] = targetElementSize;
            for (int i = static_cast<int>(newShape.size()) - 2; i >= 0; --i)
            {
                m_strides[i] = m_strides[i + 1] * newShape[i + 1];
            }
        }
    }

    const void* getData() const override
    {
        return m_sourceArray->getData();
    }

    uint32_t getNDims() const override
    {
        return static_cast<uint32_t>(m_shape.size());
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
        return m_targetType;
    }

    int32_t getDeviceId() const override
    {
        return m_sourceArray->getDeviceId();
    }

private:
    carb::ObjectPtr<IFieldArray> m_sourceArray; // Keep source alive
    ElementType m_targetType;
    std::vector<uint64_t> m_shape;
    std::vector<uint64_t> m_strides;
};

carb::ObjectPtr<IFieldArray> FieldArrayUtils::reinterpretArray(carb::ObjectPtr<IFieldArray> array, ElementType targetType)
{
    if (!array)
    {
        CARB_LOG_ERROR("Cannot reinterpret null array");
        return nullptr;
    }

    // If already the target type, return original
    if (array->getElementType() == targetType)
    {
        return array;
    }

    const ElementType srcType = array->getElementType();
    const size_t srcElementSize = getElementSize(srcType);
    const size_t dstElementSize = getElementSize(targetType);

    if (srcElementSize == 0 || dstElementSize == 0)
    {
        CARB_LOG_ERROR("Invalid element type for reinterpret");
        return nullptr;
    }

    // Calculate total size in bytes
    const auto srcShape = array->getShape();
    const uint64_t numSrcElements = std::accumulate(srcShape.begin(), srcShape.end(), 1llu, std::multiplies<uint64_t>());
    const uint64_t totalBytes = numSrcElements * srcElementSize;

    // Check if reinterpretation is valid
    if (totalBytes % dstElementSize != 0)
    {
        CARB_LOG_WARN("Cannot reinterpret array: total byte size (%" PRIu64
                      ") is not divisible by target element size (%zu)",
                      totalBytes, dstElementSize);
        return nullptr;
    }

    // Calculate new shape
    std::vector<uint64_t> newShape;
    const uint64_t numDstElements = totalBytes / dstElementSize;

    if (srcShape.size() == 1)
    {
        // 1D array stays 1D
        newShape = { numDstElements };
    }
    else if (srcShape.size() > 1)
    {
        // For multi-dimensional arrays, keep all dimensions except the last,
        // and adjust the last dimension
        newShape.assign(srcShape.begin(), srcShape.end() - 1);
        const uint64_t lastDimSize = (srcShape.back() * srcElementSize) / dstElementSize;
        if ((srcShape.back() * srcElementSize) % dstElementSize != 0)
        {
            CARB_LOG_ERROR("Cannot reinterpret array: last dimension size is not compatible with target type");
            return nullptr;
        }
        newShape.push_back(lastDimSize);
    }
    else
    {
        CARB_LOG_ERROR("Cannot reinterpret array with ndim = 0");
        return nullptr;
    }

    // Create zero-copy wrapper
    return carb::stealObject<IFieldArray>(new ReinterpretedFieldArray(array, targetType, newShape));
}

// Functor for casting and copying array elements
struct CastAndCopyFunctor
{
    template <typename SrcType, typename DstType>
    void operator()(IFieldArray* src, IMutableFieldArray* dst, uint64_t count)
    {
        const SrcType* srcData = static_cast<const SrcType*>(src->getData());
        DstType* dstData = static_cast<DstType*>(dst->getMutableData());
#ifdef _MSC_VER
#    pragma warning(push)
#    pragma warning(disable : 4244) // Suppress conversion warning - intentional cast operation
#endif
        std::copy(srcData, srcData + count, dstData);
#ifdef _MSC_VER
#    pragma warning(pop)
#endif
    }
};

carb::ObjectPtr<IFieldArray> FieldArrayUtils::castAndCopy(carb::ObjectPtr<IFieldArray> array, ElementType targetType)
{
    if (!array)
    {
        CARB_LOG_ERROR("Cannot cast null array");
        return nullptr;
    }

    // Check if array is on GPU
    if (array->getDeviceId() >= 0)
    {
        throw std::runtime_error("castAndCopy is not supported for GPU arrays");
    }

    // If already the target type, return the original array without copying
    const ElementType srcType = array->getElementType();
    if (srcType == targetType)
    {
        return array;
    }

    const auto shape = array->getShape();

    // Create a new CPU array with the target type
    auto targetArray = createMutableFieldArray(targetType, shape, -1, Order::c);
    if (!targetArray)
    {
        CARB_LOG_ERROR("Failed to create target array for casting");
        return nullptr;
    }

    // Calculate total number of elements
    const uint64_t numElements = std::accumulate(shape.begin(), shape.end(), 1llu, std::multiplies<uint64_t>());

    // Dispatch based on source and target types
    CastAndCopyFunctor functor;
    bool dispatched = FieldArrayDispatcher2<FieldArrayTypes, FieldArrayTypes>::dispatch(
        functor, array.get(), targetArray.get(), numElements);

    if (!dispatched)
    {
        CARB_LOG_ERROR("Unsupported type combination for castAndCopy: src=%d, dst=%d", static_cast<int>(srcType),
                       static_cast<int>(targetType));
        return nullptr;
    }

    return carb::stealObject<IFieldArray>(targetArray.detach());
}

} // namespace data
} // namespace cae
} // namespace omni
