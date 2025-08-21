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

// .clang-format off
// must include first.
#include <omni/cae/data/IDataDelegateIncludes.h>
// .clang-format on

#include <carb/ObjectUtils.h>
#include <carb/tasking/ITasking.h>
#include <carb/tasking/TaskingUtils.h>

#include <omni/cae/data/IDataDelegate.h>
#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omni/cae/data/IFileUtils.h>

#include <string>

namespace omni
{
namespace cae
{
namespace data
{
namespace cgns
{
/**
 * IDataDelegate subclass to support 'OmniCaeCgnsFieldArray' prims.
 */
class DataDelegate : public omni::cae::data::IDataDelegate
{
    CARB_IOBJECT_IMPL
public:
    DataDelegate(const std::string& extId, omni::cae::data::IFieldArrayUtils* utils)
        : m_extensionId(extId), m_fieldArrayUtils(utils)
    {
        auto iface = carb::getCachedInterface<omni::cae::data::IDataDelegateInterface>();
        m_fileUtils = iface->getFileUtils();
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
    omni::cae::data::IFileUtils* m_fileUtils;
    carb::tasking::MutexWrapper m_mutex;
};


} // namespace cgns
} // namespace data
} // namespace cae
} // namespace omni
