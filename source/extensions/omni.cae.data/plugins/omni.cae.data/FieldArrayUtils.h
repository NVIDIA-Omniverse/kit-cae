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

#include <omni/cae/data/IFieldArrayUtils.h>

namespace omni
{
namespace cae
{
namespace data
{

class FieldArrayUtils final : public IFieldArrayUtils
{
public:
    carb::ObjectPtr<IMutableFieldArray> createMutableFieldArray(ElementType type,
                                                                const std::vector<uint64_t>& shape,
                                                                int32_t deviceId,
                                                                Order order) override;

    carb::ObjectPtr<IFieldArray> copyToDevice(carb::ObjectPtr<IFieldArray> array, int32_t targetDeviceId) override;

    carb::ObjectPtr<IFieldArray> reinterpretArray(carb::ObjectPtr<IFieldArray> array, ElementType targetType) override;

    carb::ObjectPtr<IFieldArray> castAndCopy(carb::ObjectPtr<IFieldArray> array, ElementType targetType) override;
};

} // namespace data
} // namespace cae
} // namespace omni
