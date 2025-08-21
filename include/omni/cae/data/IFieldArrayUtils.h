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
};

} // namespace data
} // namespace cae
} // namespace omni
