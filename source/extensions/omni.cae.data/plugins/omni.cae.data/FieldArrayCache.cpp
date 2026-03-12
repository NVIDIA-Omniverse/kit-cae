// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.


// .clang-format off
#include <omni/cae/data/IDataDelegateIncludes.h>
// .clang-format on


#include "FieldArrayCache.h"

#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <iomanip>
#include <sstream>
#include <string>

namespace omni
{
namespace cae
{
namespace data
{

bool FieldArrayCache::CacheEntry::contains(pxr::UsdTimeCode time) const
{
    return m_fieldArrayMap.find(time) != m_fieldArrayMap.end();
}

void FieldArrayCache::CacheEntry::add(pxr::UsdTimeCode time, carb::ObjectPtr<IFieldArray> fieldArray)
{
    m_fieldArrayMap[time] = fieldArray;
}

carb::ObjectPtr<IFieldArray> FieldArrayCache::CacheEntry::get(pxr::UsdTimeCode time) const
{
    auto iter = m_fieldArrayMap.find(time);
    if (iter != m_fieldArrayMap.end())
    {
        return iter->second;
    }
    return {};
}

FieldArrayCache::FieldArrayCache(IUsdUtils* usdUtils)
    : m_listener(new UsdNoticeListener<FieldArrayCache>(this)), m_usdUtils(usdUtils)
{
}

bool FieldArrayCache::shouldCache(pxr::UsdPrim prim, pxr::UsdTimeCode time, pxr::UsdTimeCode* cacheKey) const
{
    carb::settings::ISettings* settings = carb::getCachedInterface<carb::settings::ISettings>();
    if (!settings)
    {
        return false;
    }

    const char* cacheModeCStr = settings->get<const char*>("/persistent/exts/omni.cae.data/cacheMode");
    const std::string cacheMode = cacheModeCStr ? cacheModeCStr : "always";

    if (cacheMode == std::string("disabled"))
    {
        return false;
    }
    else if (cacheMode == std::string("always"))
    {
        // Compute cache key if requested
        if (cacheKey)
        {
            *cacheKey = snapTime(prim, time);
        }
        return true;
    }
    else if (cacheMode == std::string("static-fields"))
    {
        // Only cache if there are no time samples (static data)
        if (!m_usdUtils)
        {
            return false;
        }

        double lower = 0.0;
        double upper = 0.0;
        bool hasTimeSamples = false;
        if (m_usdUtils->getBracketingTimeSamplesForPrim(prim, time.GetValue(), &lower, &upper, &hasTimeSamples))
        {
            if (!hasTimeSamples)
            {
                // Compute cache key if requested (use EarliestTime for static data)
                if (cacheKey)
                {
                    *cacheKey = pxr::UsdTimeCode::EarliestTime();
                }
                return true;
            }
        }
        return false;
    }

    return false;
}

pxr::UsdTimeCode FieldArrayCache::snapTime(pxr::UsdPrim prim, pxr::UsdTimeCode time) const
{
    if (!m_usdUtils)
    {
        return pxr::UsdTimeCode::EarliestTime();
    }

    // Get the bracketing time sample key for this prim and time
    double lower = 0.0;
    double upper = 0.0;
    bool hasTimeSamples = false;
    if (m_usdUtils->getBracketingTimeSamplesForPrim(prim, time.GetValue(), &lower, &upper, &hasTimeSamples))
    {
        // If no time samples exist, use EarliestTime since data doesn't change with time
        return hasTimeSamples ? pxr::UsdTimeCode(lower) : pxr::UsdTimeCode::EarliestTime();
    }
    else
    {
        // Fallback to using EarliestTime if getBracketingTimeSamplesForPrim fails
        return pxr::UsdTimeCode::EarliestTime();
    }
}

std::string FieldArrayCache::formatTimeCode(pxr::UsdTimeCode time)
{
    if (time.IsDefault())
    {
        return "Default";
    }
    else if (time.IsEarliestTime())
    {
        return "EarliestTime";
    }
    else
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << time.GetValue();
        return oss.str();
    }
}

bool FieldArrayCache::contains(pxr::UsdPrim prim, pxr::UsdTimeCode time) const
{
    pxr::UsdTimeCode cacheKey;
    if (!shouldCache(prim, time, &cacheKey))
    {
        return false;
    }

    std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
    auto iter = m_cache.find(prim.GetPath());
    if (iter != m_cache.end())
    {
        return iter->second.contains(cacheKey);
    }
    return false;
}

void FieldArrayCache::add(pxr::UsdPrim prim, pxr::UsdTimeCode time, carb::ObjectPtr<IFieldArray> fieldArray)
{
    pxr::UsdTimeCode cacheKey;
    if (!shouldCache(prim, time, &cacheKey))
    {
        CARB_LOG_INFO("[array-cache]:skip(add) %s, time=%s (snapped=%s)", prim.GetPath().GetText(),
                      formatTimeCode(time).c_str(), formatTimeCode(cacheKey).c_str());
        return;
    }

    std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
    auto& entry = m_cache[prim.GetPath()];
    entry.add(cacheKey, fieldArray);

    CARB_LOG_INFO("[array-cache]:update %s, time=%s (snapped=%s)", prim.GetPath().GetText(),
                  formatTimeCode(time).c_str(), formatTimeCode(cacheKey).c_str());
}

carb::ObjectPtr<IFieldArray> FieldArrayCache::get(pxr::UsdPrim prim, pxr::UsdTimeCode time) const
{
    pxr::UsdTimeCode cacheKey;
    if (!shouldCache(prim, time, &cacheKey))
    {
        CARB_LOG_INFO("[array-cache]:skip(get) %s, time=%s (snapped=%s)", prim.GetPath().GetText(),
                      formatTimeCode(time).c_str(), formatTimeCode(cacheKey).c_str());
        return {};
    }

    std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
    auto iter = m_cache.find(prim.GetPath());
    if (iter != m_cache.end())
    {
        auto array = iter->second.get(cacheKey);
        if (array)
        {
            CARB_LOG_INFO("[array-cache]:hit %s, time=%s (snapped=%s)", prim.GetPath().GetText(),
                          formatTimeCode(time).c_str(), formatTimeCode(cacheKey).c_str());
            return array;
        }
    }

    CARB_LOG_INFO("[array-cache]:miss %s, time=%s (snapped=%s)", prim.GetPath().GetText(), formatTimeCode(time).c_str(),
                  formatTimeCode(cacheKey).c_str());
    return {};
}

void FieldArrayCache::clear()
{
    std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
    if (!m_cache.empty())
    {
        CARB_LOG_INFO("[array-cache]:clear clearing %zu entries", m_cache.size());
        m_cache.clear();
    }
}

void FieldArrayCache::handleNotice(const pxr::UsdNotice::ObjectsChanged& notice)
{
    std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
    if (m_cache.empty())
    {
        return;
    }

    // Check if an attribute of a cached prim has been explicitly changed.
    for (const auto& path : notice.GetChangedInfoOnlyPaths())
    {
        if (!path.IsPropertyPath())
            continue;

        const auto& changedPrim = path.GetPrimPath();
        auto it = m_cache.find(changedPrim);
        if (it != end(m_cache))
        {
            CARB_LOG_INFO("[array-cache]:drop %s", changedPrim.GetText());
            m_cache.erase(it);
        }
    }

    // Check if the prim itself has been resynced or is part of a resynced tree
    for (const auto& path : notice.GetResyncedPaths())
    {
        for (auto it = cbegin(m_cache); it != cend(m_cache);)
        {
            const auto& cachedPath = it->first;
            if (cachedPath.HasPrefix(path))
            {
                CARB_LOG_INFO("[array-cache]:drop %s", cachedPath.GetText());
                it = m_cache.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
}

} // namespace data
} // namespace cae
} // namespace omni
