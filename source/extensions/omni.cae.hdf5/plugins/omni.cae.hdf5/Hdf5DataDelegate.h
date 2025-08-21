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

#include <carb/InterfaceUtils.h>
#include <carb/ObjectUtils.h>

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
namespace hdf5
{

// ideally, this class is moved to omni.cae.data, however I am leaving this hear
// to avoid having to deal with hdf5 dependencies etc.

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

    const char* getExtensionId() const override
    {
        return m_extensionId.c_str();
    }

    bool canProvide(pxr::UsdPrim fieldArrayPrim) const override;
    carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) override;


private:
    std::string m_extensionId;
    omni::cae::data::IFieldArrayUtils* m_fieldArrayUtils;
    omni::cae::data::IFileUtils* m_fileUtils;
};

} // namespace hdf5
} // namespace data
} // namespace cae
} // namespace omni
