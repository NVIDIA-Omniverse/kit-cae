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

#include <carb/Interface.h>
#include <carb/tasking/TaskingTypes.h>

#include <omni/cae/data/IDataDelegate.h>
#include <omni/cae/data/Types.h>

namespace omni
{
namespace cae
{
namespace data
{

/**
 * Defines the interface for DataDelegateRegistry.
 *
 * Extensions can register implementations of IDataDelegate to enable
 * the framework to understand loading different types of data.
 */
class IDataDelegateRegistry
{
public:
    /**
     * Register a data delegate. Delegates with higher priority are checked first
     * when in APIs like `getFieldArray`.
     *
     * @param dataDelegate Data delegate to register.
     * @param priority Priority for the data delegate.
     */
    virtual void registerDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate, DelegatePriority priority = 0) = 0;

    /**
     * Deregister a data delegate.
     *
     * @param dataDelegate Data delegate to deregister.
     */
    virtual void deregisterDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate) = 0;

    /**
     * Deregister all data delegates that were registered by the specified
     * extension.
     *
     * @param extensionId The id of the source extension that registered the data
     * delegates.
     */
    virtual void deregisterAllDataDelegatesForExtension(const char* extensionId) = 0;

    /**
     * Walks all registered data delegates to returns field array if any delegate
     * supports it.
     *
     * @param fieldArrayPrim Field array prim of type pxr::OmniCaeFieldArray or a
     * sub-type.
     * @param time TimeCode to use to lookup prim attributes.
     *
     * @return Field array or Null.
     */
    virtual carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim,
                                                       pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;

    /**
     * Return whether a field array is cached for the given prim, for faster access.
     *
     * @param fieldArrayPrim Field array prim of type pxr::OmniCaeFieldArray or a
     * sub-type.
     * @param timeCode Time code for the time to read.
     * @return true if the field array is cached.
     */
    virtual bool isFieldArrayCached(pxr::UsdPrim fieldArrayPrim,
                                    pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) const = 0;

    /**
     * Async version of `getFieldArray`.
     */
    virtual carb::tasking::Future<carb::ObjectPtr<IFieldArray>> getFieldArrayAsync(
        pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) = 0;
};

} // namespace data
} // namespace cae
} // namespace omni
