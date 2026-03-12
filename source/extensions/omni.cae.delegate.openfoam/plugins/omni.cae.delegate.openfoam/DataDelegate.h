// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

// .clang-format off
// must include first.
#include <omni/cae/data/IDataDelegateIncludes.h>
// .clang-format on

#include <carb/ObjectUtils.h>
#include <carb/tasking/ITasking.h>
#include <carb/tasking/TaskingUtils.h>

#include <omni/cae/data/IDataDelegate.h>
#include <omni/cae/data/IFieldArrayUtils.h>

#include <string>

namespace omni
{
namespace cae
{
namespace data
{
namespace openfoam
{
/**
 * IDataDelegate subclass to support 'OmniCaeOpenFoamFieldArray' prims.
 */
class DataDelegate : public omni::cae::data::IDataDelegate
{
    CARB_IOBJECT_IMPL
public:
    DataDelegate(const std::string& extId, omni::cae::data::IFieldArrayUtils* utils)
        : m_extensionId(extId), m_fieldArrayUtils(utils)
    {
        m_tasking = carb::getFramework()->acquireInterface<carb::tasking::ITasking>();
    }

    bool canProvide(pxr::UsdPrim fieldArrayPrim) const override;

    carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) override;

    const char* getExtensionId() const override
    {
        return m_extensionId.c_str();
    }


private:
    std::string m_extensionId;
    omni::cae::data::IFieldArrayUtils* m_fieldArrayUtils;
    carb::tasking::ITasking* m_tasking;
    carb::tasking::MutexWrapper m_mutex;
};


} // namespace openfoam
} // namespace data
} // namespace cae
} // namespace omni
