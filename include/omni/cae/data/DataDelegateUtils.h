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

#ifndef DATA_DELEGATE_UTILS_INCLUDES
#    error "Please include DataDelegateUtilsIncludes.h before including this header or in pre-compiled header."
#endif

#include <carb/logging/Log.h>

#include <omni/cae/data/DataDelegateUtils.h>
#include <omni/cae/data/IDataDelegateRegistry.h>
#include <omni/cae/data/IFieldArray.h>
#include <omniCae/fieldArray.h>

namespace omni
{
namespace cae
{
namespace data
{

class IDataDelegateRegistry;
class IFieldArray;
/**
 * @class DataDelegateUtils
 *
 * DataDeleageUtils is a utility class intended to make it easier to work with
 * USD and Data Delegate.
 */
class DataDelegateUtils
{
public:
    /**
     * Gets the attribute array value.
     *
     * @tparam T The data type of the attribute array.
     * @param attribute The attribute to get value from.
     * @param out the value of the attribute.
     * @return True if the out value is valid.
     */
    template <typename T>
    static bool getAttributeArray(pxr::UsdAttribute&& attribute, pxr::VtArray<T>& out, pxr::UsdTimeCode time)
    {
        pxr::VtValue arrayDataValue;
        attribute.Get(&arrayDataValue, time);
        if (arrayDataValue.GetArraySize())
        {
            out = arrayDataValue.Get<pxr::VtArray<T>>();
            return true;
        }
        return false;
    }

    /**
     * Given the prim to a `OmniCaeFieldArray` prim, this can lookup the
     * filenames.
     */
    static bool getFileNames(pxr::VtArray<pxr::SdfAssetPath>& out, pxr::UsdPrim prim, pxr::UsdTimeCode timeCode)
    {
        if (!prim.IsValid())
        {
            CARB_LOG_ERROR("Invalid prim!");
            return false;
        }

        if (!prim.IsA<pxr::OmniCaeFieldArray>())
        {
            CARB_LOG_ERROR("Invalid prim type: %s. Expected OmniCaeFieldArray or sub-type.",
                           prim.GetPrimTypeInfo().GetSchemaTypeName().GetString().c_str());
            return false;
        }

        pxr::OmniCaeFieldArray fieldArray(prim);
        return getAttributeArray(fieldArray.GetFileNamesAttr(), out, timeCode);
    }

    /**
     * Gets the attribute value.
     *
     * @tparam T the data type of the attribute.
     * @param attribute The attribute to get value from.
     * @param time Current timecode.
     * @return the value of the attribute.
     */
    template <class T>
    static T getAttribute(const pxr::UsdAttribute& attribute, pxr::UsdTimeCode time)
    {
        T val;
        attribute.Get(&val, time);
        return val;
    }

    /**
     * Returns the field array pointed by the relationship.
     */
    static carb::ObjectPtr<omni::cae::data::IFieldArray> getFieldArray(omni::cae::data::IDataDelegateRegistry* registry,
                                                                       const pxr::UsdRelationship& rel,
                                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default());
    static carb::ObjectPtr<omni::cae::data::IFieldArray> getFieldArray(omni::cae::data::IDataDelegateRegistry* registry,
                                                                       pxr::UsdStageWeakPtr stage,
                                                                       const pxr::SdfPath& primPath,
                                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default());

    static std::vector<carb::ObjectPtr<omni::cae::data::IFieldArray>> getFieldArrays(
        omni::cae::data::IDataDelegateRegistry* registry,
        const pxr::UsdRelationship& rel,
        pxr::UsdTimeCode time = pxr::UsdTimeCode::Default());
    static std::vector<carb::ObjectPtr<omni::cae::data::IFieldArray>> getFieldArrays(
        omni::cae::data::IDataDelegateRegistry* registry,
        pxr::UsdStageWeakPtr stage,
        const pxr::SdfPathVector& primPaths,
        pxr::UsdTimeCode time = pxr::UsdTimeCode::Default());
};

/**
 * Gets the string attribute value.
 *
 * @param attribute The attribute to get value from.
 * @param time Current timecode.
 * @return the value of the attribute.
 */
template <> // Define it out of class body to avoid "Explicit specialization in
            // non-namespace scope" error.
inline std::string DataDelegateUtils::getAttribute(const pxr::UsdAttribute& attribute, pxr::UsdTimeCode time)
{
    pxr::VtValue val;
    attribute.Get(&val, time);
    if (attribute.GetTypeName() == pxr::SdfValueTypeNames->String)
    {
        return val.Get<std::string>();
    }
    else if (attribute.GetTypeName() == pxr::SdfValueTypeNames->Token)
    {
        return val.Get<pxr::TfToken>().GetString();
    }
    else if (attribute.GetTypeName() == pxr::SdfValueTypeNames->Asset)
    {
        auto path = val.Get<pxr::SdfAssetPath>();
        return path.GetAssetPath();
    }
    return "";
}

inline carb::ObjectPtr<omni::cae::data::IFieldArray> DataDelegateUtils::getFieldArray(
    omni::cae::data::IDataDelegateRegistry* registry, const pxr::UsdRelationship& rel, pxr::UsdTimeCode time)
{
    auto vec = DataDelegateUtils::getFieldArrays(registry, rel, time);
    if (vec.size() > 1)
    {
        CARB_LOG_WARN("Only first target for '%s' is considered.", rel.GetPath().GetText());
    }
    return vec.empty() ? nullptr : vec.at(0);
}

inline carb::ObjectPtr<omni::cae::data::IFieldArray> DataDelegateUtils::getFieldArray(
    omni::cae::data::IDataDelegateRegistry* registry,
    pxr::UsdStageWeakPtr stage,
    const pxr::SdfPath& primPath,
    pxr::UsdTimeCode time)
{
    auto vec = DataDelegateUtils::getFieldArrays(registry, stage, pxr::SdfPathVector{ primPath }, time);
    if (vec.size() > 1)
    {
        CARB_LOG_WARN("Only first target for '%s' is considered.", primPath.GetText());
    }
    return vec.empty() ? nullptr : vec.at(0);
}

inline std::vector<carb::ObjectPtr<omni::cae::data::IFieldArray>> DataDelegateUtils::getFieldArrays(
    omni::cae::data::IDataDelegateRegistry* registry, const pxr::UsdRelationship& rel, pxr::UsdTimeCode time)
{
    pxr::SdfPathVector paths;
    if (rel.GetForwardedTargets(&paths))
    {
        return DataDelegateUtils::getFieldArrays(registry, rel.GetStage(), paths, time);
    }
    return {};
}

inline std::vector<carb::ObjectPtr<omni::cae::data::IFieldArray>> DataDelegateUtils::getFieldArrays(
    omni::cae::data::IDataDelegateRegistry* registry,
    pxr::UsdStageWeakPtr stage,
    const pxr::SdfPathVector& primPaths,
    pxr::UsdTimeCode time)
{
    if (!registry)
    {
        CARB_LOG_WARN("Invalid registry!");
        return {};
    }
    if (!stage)
    {
        CARB_LOG_WARN("Invalid stage!");
        return {};
    }

    std::vector<carb::ObjectPtr<omni::cae::data::IFieldArray>> result;
    std::transform(primPaths.begin(), primPaths.end(), std::back_inserter(result),
                   [&](const pxr::SdfPath& path)
                   {
                       auto prim = stage->GetPrimAtPath(path);
                       return registry->getFieldArray(prim, time);
                   });
    return result;
}

} // namespace data
} // namespace cae
} // namespace omni
