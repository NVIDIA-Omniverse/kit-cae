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

#ifndef DATA_DELEGATE_INCLUDES
#    error "Please include IDataDelegateIncludes.h before including this header or in pre-compiled header."
#endif

#include <carb/IObject.h>

#include <omni/cae/data/IFieldArray.h>

namespace omni
{
namespace cae
{
namespace data
{

/// IDataDelegate defines the data delegate interface.
class IDataDelegate : public carb::IObject
{
public:
    /**
     * Returns `true` if this delegate can provide the field array.
     *
     * @param fieldArrayPrim Field array prim of type pxr::OmniCaeFieldArray or a
     * sub-type.
     * @return `true` if field array is supported else `false`.
     */
    virtual bool canProvide(pxr::UsdPrim fieldArrayPrim) const = 0;

    /**
     * Returns data array described by the prim.
     *
     * @param fieldArrayPrim Field array prim of type pxr::OmniCaeFieldArray or a
     * sub-type.
     * @param timeCode Time code for the time to read.
     * @return Smart point to the field array on success. Returns empty smart
     * pointer if the read failed.
     */
    virtual carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim,
                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;

    /**
     * Get the id of the source extension which registered this action.
     *
     * @return Id of the source extension which registered this action.
     */
    virtual const char* getExtensionId() const = 0;

}; // IDataDelegate

using IDataDelegatePtr = carb::ObjectPtr<IDataDelegate>;

inline bool operator==(const IDataDelegatePtr& left, const IDataDelegatePtr& right) noexcept
{
    return (left.get() == right.get());
}

} // namespace data
} // namespace cae
} // namespace omni
