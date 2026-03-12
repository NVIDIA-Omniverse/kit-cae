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

#include <omni/cae/data/IUsdUtils.h>

namespace omni
{
namespace cae
{
namespace data
{

class UsdUtils final : public IUsdUtils
{
public:
    bool getBracketingTimeSamplesForPrim(
        const pxr::UsdPrim& prim, double time, double* lower, double* upper, bool* hasTimeSamples) const override;

    bool getBracketingTimeSamplesForDataSetPrim(const pxr::UsdPrim& prim,
                                                double time,
                                                bool traverseFieldRelationships,
                                                double* lower,
                                                double* upper,
                                                bool* hasTimeSamples) const override;

    std::vector<pxr::UsdPrim> getRelatedDataPrims(const pxr::UsdPrim& prim,
                                                  bool transitive = true,
                                                  bool includeSelf = true) const override;

private:
    void populateTimeSamplesForDataSet(const pxr::UsdPrim& prim,
                                       double time,
                                       bool traverseFieldRelationships,
                                       std::set<double>& times,
                                       std::set<pxr::SdfPath>& processedPrims) const;

    void collectRelatedDataPrims(const pxr::UsdPrim& prim,
                                 bool transitive,
                                 std::set<pxr::SdfPath>& processedPrims,
                                 std::vector<pxr::UsdPrim>& result) const;
};

} // namespace data
} // namespace cae
} // namespace omni
