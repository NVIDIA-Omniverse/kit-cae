// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#pragma once

#include <carb/IObject.h>

#include <vector>

namespace omni
{
namespace cae
{
namespace data
{

enum class ElementType : uint32_t
{
    unspecified,
    int32,
    int64,
    uint32,
    uint64,
    float32,
    float64
};

inline size_t getElementSize(ElementType type)
{
    switch (type)
    {
    case ElementType::int32:
        return sizeof(int32_t);
    case ElementType::int64:
        return sizeof(int64_t);
    case ElementType::uint32:
        return sizeof(uint32_t);
    case ElementType::uint64:
        return sizeof(uint64_t);
    case ElementType::float32:
        return sizeof(float);
    case ElementType::float64:
        return sizeof(double);
    case ElementType::unspecified:
    default:
        return 0u;
    }
}

template <typename>
struct ElementTypeTraits
{
};

template <>
struct ElementTypeTraits<int32_t>
{
    static ElementType get()
    {
        return ElementType::int32;
    }
};
template <>
struct ElementTypeTraits<uint32_t>
{
    static ElementType get()
    {
        return ElementType::uint32;
    }
};

template <>
struct ElementTypeTraits<int64_t>
{
    static ElementType get()
    {
        return ElementType::int64;
    }
};
template <>
struct ElementTypeTraits<uint64_t>
{
    static ElementType get()
    {
        return ElementType::uint64;
    }
};

template <>
struct ElementTypeTraits<float>
{
    static ElementType get()
    {
        return ElementType::float32;
    }
};
template <>
struct ElementTypeTraits<double>
{
    static ElementType get()
    {
        return ElementType::float64;
    }
};

/**
 * IFieldArray is an interface for an n-dimensional data array. The ndims,
 * shape, strides intentionally line up with the definitions in NumPy `ndarray`.
 */
class IFieldArray : public carb::IObject
{
public:
    /**
     * Return a pointer to the data held by this instance.
     *
     * @return nullptr, if isValid() == false, else read-only pointer to the data.
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
     * Returns the strides of the array. Note, strides are specified in bytes.
     */
    virtual std::vector<uint64_t> getStrides() const = 0;

    /**
     * Returns the elemental type held by this array.
     */
    virtual ElementType getElementType() const = 0;

    /**
     * Returns the device id for the device on which the data held by this field array
     * is hosted.
     *
     * When a non-negative number (i.e >= 0) is returned, it is same as the CUDA device id
     * returned by functions such as `cudaGetDevice()`.
     *
     * When -1 is returned, it indicate host (or CPU) memory.
     */
    virtual int32_t getDeviceId() const = 0;

}; // IFieldArray

/**
 * IMutableFieldArray is subclass of IFieldArray which provides mutable access
 * to the internal data. Use `IFieldArrayUtils::createMutableFieldArray` to
 * create an instance of this array.
 */
class IMutableFieldArray : public IFieldArray
{
public:
    /**
     * Returns pointer to internally allocated data buffer for writing.
     * Use `getMutableDataSizeInBytes()` to limit modifying beyond allocated
     * memory.
     */
    virtual void* getMutableData() = 0;

    template <typename T>
    T* getMutableData()
    {
        return reinterpret_cast<T*>(this->getMutableData());
    }

    /**
     * A convenience method to return the size of the internally allocated buffer
     * in bytes.
     */
    virtual uint64_t getMutableDataSizeInBytes() const = 0;
};

} // namespace data
} // namespace cae
} // namespace omni
