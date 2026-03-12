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

#ifndef DATA_DELEGATE_INCLUDES
#    error "Please include IDataDelegateIncludes.h before including this header or in pre-compiled header."
#endif


#include <carb/logging/Log.h>

namespace omni
{
namespace cae
{
namespace data
{

class IUsdUtils
{
public:
    /**
     * Get bracketing time samples for a given prim, similar to UsdAttribute::GetBracketingTimeSamples.
     *
     * This method recursively traverses all relationships on the prim and its targets to find
     * time samples. Only relationships targeting OmniCaeDataSet or OmniCaeFieldArray prims
     * are traversed.
     *
     * @param prim The UsdPrim to query
     * @param time The time to query
     * @param lower Output parameter for the lower bracketing time sample
     * @param upper Output parameter for the upper bracketing time sample
     * @param hasTimeSamples Output parameter indicating if time samples exist
     * @return true if the operation succeeded, false otherwise
     */
    virtual bool getBracketingTimeSamplesForPrim(
        const pxr::UsdPrim& prim, double time, double* lower, double* upper, bool* hasTimeSamples) const = 0;

    /**
     * Get bracketing time samples for a DataSet prim with optional control over field relationship traversal.
     *
     * This method is similar to getBracketingTimeSamplesForPrim but specifically designed for
     * DataSet prims. It allows controlling whether `field:*` relationships (e.g., `field:velocity`,
     * `field:pressure`) are traversed when determining time samples.
     *
     * When traverseFieldRelationships is false, only the DataSet prim's own attributes and
     * non-field relationships are considered. When true, field relationships are also traversed
     * recursively (matching the behavior of getBracketingTimeSamplesForPrim).
     *
     * @param prim The DataSet UsdPrim to query (should be an OmniCaeDataSet prim)
     * @param time The time to query
     * @param traverseFieldRelationships If true, traverse `field:*` relationships; if false, skip them
     * @param lower Output parameter for the lower bracketing time sample
     * @param upper Output parameter for the upper bracketing time sample
     * @param hasTimeSamples Output parameter indicating if time samples exist
     * @return true if the operation succeeded, false otherwise
     */
    virtual bool getBracketingTimeSamplesForDataSetPrim(const pxr::UsdPrim& prim,
                                                        double time,
                                                        bool traverseFieldRelationships,
                                                        double* lower,
                                                        double* upper,
                                                        bool* hasTimeSamples) const = 0;

    /**
     * Get all related DataSet and FieldArray prims for a given prim.
     *
     * This method traverses relationships from the input prim and collects all related prims
     * that are either OmniCaeDataSet or OmniCaeFieldArray types. This is useful for cache
     * invalidation tracking where changes to related data prims should invalidate cached results.
     *
     * @param prim The starting UsdPrim (typically a DataSet or FieldArray)
     * @param transitive If true, recursively traverse relationship targets; if false, only return
     *                   immediate relationship targets
     * @param includeSelf If true, include the input prim in the result set; if false, only return
     *                    related prims
     * @return Vector of related DataSet and FieldArray prims
     */
    virtual std::vector<pxr::UsdPrim> getRelatedDataPrims(const pxr::UsdPrim& prim,
                                                          bool transitive = true,
                                                          bool includeSelf = true) const = 0;
};

} // namespace data
} // namespace cae
} // namespace omni
