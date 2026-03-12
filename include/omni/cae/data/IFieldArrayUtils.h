// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#pragma once

#include <carb/Interface.h>
#include <carb/Types.h>

#include <omni/cae/data/IFieldArray.h>

namespace omni
{
namespace cae
{
namespace data
{

enum class Order : uint32_t
{
    c,
    fortran,
};

/**
 * A helper class for working with IFieldArray.
 */
class IFieldArrayUtils
{
public:
    /**
     * Creates a mutable field array of given type, shape, and device id (refer to `IFieldArray` documentation for
     * details). Order determines how strides are internally computed for multidimensional arrays.
     */
    virtual carb::ObjectPtr<IMutableFieldArray> createMutableFieldArray(ElementType type,
                                                                        const std::vector<uint64_t>& shape,
                                                                        int32_t deviceId = -1,
                                                                        Order order = Order::c) = 0;

    /**
     * Creates a copy of the given field array on the target device.
     * If the array is already on the target device, returns the input array without copying.
     *
     * @param array The source array to copy.
     * @param targetDeviceId The target device id (-1 for CPU, >= 0 for CUDA device).
     * @return A copy of the array on the target device, or the original array if already on target device.
     */
    virtual carb::ObjectPtr<IFieldArray> copyToDevice(carb::ObjectPtr<IFieldArray> array, int32_t targetDeviceId) = 0;

    /**
     * Reinterpret array memory as a different element type (zero-copy).
     * This is a zero-copy operation that reinterprets the raw bytes as a different type.
     * The total byte size must match: sizeof(src_type) * src_count == sizeof(dst_type) * dst_count
     *
     * @param array The source array to reinterpret.
     * @param targetType The target element type.
     * @return A new array view with memory reinterpreted as the target type.
     * @throws std::runtime_error if the size constraints are violated.
     */
    virtual carb::ObjectPtr<IFieldArray> reinterpretArray(carb::ObjectPtr<IFieldArray> array, ElementType targetType) = 0;

    /**
     * Casts and copies array data to a different element type.
     * This operation creates a new array with the target type and copies data with appropriate casting.
     *
     * @param array The source array to cast and copy.
     * @param targetType The target element type.
     * @return A new array with data casted to the target type.
     */
    virtual carb::ObjectPtr<IFieldArray> castAndCopy(carb::ObjectPtr<IFieldArray> array, ElementType targetType) = 0;
};

} // namespace data
} // namespace cae
} // namespace omni
