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

#include <carb/thread/Mutex.h>

#include <omni/cae/data/IFieldArray.h>
#include <omni/cae/data/IUsdUtils.h>
#include <pxr/base/tf/notice.h>
#include <pxr/base/tf/weakBase.h>
#include <pxr/base/tf/weakPtr.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/notice.h>
#include <pxr/usd/usd/prim.h>

#include <map>
#include <memory>

namespace omni
{
namespace cae
{
namespace data
{

template <typename T>
class UsdNoticeListener final : public pxr::TfWeakBase
{
    pxr::TfNotice::Key m_key;
    T* m_target = nullptr;

public:
    UsdNoticeListener(T* target) : m_target(target)
    {
        m_key = pxr::TfNotice::Register(pxr::TfCreateWeakPtr(this), &UsdNoticeListener::handleNotice);
    }

    ~UsdNoticeListener()
    {
        if (m_key)
        {
            pxr::TfNotice::Revoke(m_key);
        }
    }

    void handleNotice(const pxr::UsdNotice::ObjectsChanged& notice)
    {
        if (m_target)
        {
            m_target->handleNotice(notice);
        }
    }
};

/**
 * @class FieldArrayCache
 * @brief Thread-safe cache for IFieldArray objects keyed by USD prim path and time code.
 *
 * This cache stores field arrays retrieved from data delegates to avoid redundant data loading.
 * The cache behavior is controlled by the "cacheMode" setting which can be:
 * - "disabled": No caching is performed
 * - "always": All field arrays are cached regardless of time-varying behavior
 * - "static-fields": Only non-time-varying fields are cached (fields without time samples)
 *
 * Time Snapping:
 * When caching, the cache uses "snapped" time codes as keys. The snapped time is determined by:
 * - If the prim has time samples: Uses the lower bracketing time sample (via getBracketingTimeSamplesForPrim)
 * - If the prim has no time samples: Uses EarliestTime as the cache key (since data doesn't change with time)
 *
 * Cache Invalidation:
 * The cache automatically invalidates entries when:
 * - USD prim attributes are modified (via UsdNotice::ObjectsChanged)
 * - USD prims are resynced (via UsdNotice::ObjectsChanged)
 * - A stage is attached or detached (via IStageUpdate callbacks)
 *
 * Thread Safety:
 * All cache operations are protected by a mutex and are thread-safe.
 *
 * Logging:
 * All cache operations log with the "[array-cache]:" prefix:
 * - "[array-cache]:hit" - Cache hit (entry found)
 * - "[array-cache]:miss" - Cache miss (entry not found)
 * - "[array-cache]:update" - Entry added/updated
 * - "[array-cache]:skip(add)" - Skipped adding (caching disabled for this prim/time)
 * - "[array-cache]:skip(get)" - Skipped lookup (caching disabled for this prim/time)
 * - "[array-cache]:drop" - Entry dropped due to USD changes
 * - "[array-cache]:clear" - Cache cleared (stage attach/detach)
 *
 * Log messages include both the original time code and the snapped cache key for debugging.
 */
class FieldArrayCache
{
    struct CacheEntry
    {
        std::map<pxr::UsdTimeCode, carb::ObjectPtr<IFieldArray>> m_fieldArrayMap;

        bool contains(pxr::UsdTimeCode time) const;
        void add(pxr::UsdTimeCode time, carb::ObjectPtr<IFieldArray> fieldArray);
        carb::ObjectPtr<IFieldArray> get(pxr::UsdTimeCode time) const;
    };

    std::unique_ptr<UsdNoticeListener<FieldArrayCache>> m_listener;
    std::map<pxr::SdfPath, CacheEntry> m_cache;
    mutable carb::thread::mutex m_cacheMutex;
    IUsdUtils* m_usdUtils;

    /**
     * @brief Compute the snapped time code to use as a cache key for the given prim and time.
     *
     * The snapped time is determined by:
     * - If the prim has time samples: Returns the lower bracketing time sample
     * - If the prim has no time samples: Returns EarliestTime (since data doesn't change with time)
     *
     * @param prim The USD prim to query for time samples
     * @param time The original time code
     * @return The snapped time code to use as the cache key
     */
    pxr::UsdTimeCode snapTime(pxr::UsdPrim prim, pxr::UsdTimeCode time) const;

    /**
     * @brief Determine if caching should be enabled for the given prim and time based on cache mode setting.
     *
     * Cache modes:
     * - "disabled": Returns false (no caching)
     * - "always": Returns true, computes snapped cache key
     * - "static-fields": Returns true only if prim has no time samples, uses EarliestTime as cache key
     *
     * @param prim The USD prim to check
     * @param time The time code to check
     * @param cacheKey Optional output parameter to receive the snapped cache key (if caching is enabled)
     * @return true if caching should be enabled, false otherwise
     */
    bool shouldCache(pxr::UsdPrim prim, pxr::UsdTimeCode time, pxr::UsdTimeCode* cacheKey = nullptr) const;

    /**
     * @brief Format a time code as a human-readable string.
     *
     * Handles special values:
     * - "Default" for UsdTimeCode::Default()
     * - "EarliestTime" for UsdTimeCode::EarliestTime()
     * - Numeric values formatted with 3 decimal places (e.g., "1.234")
     *
     * @param time The time code to format
     * @return Formatted string representation
     */
    static std::string formatTimeCode(pxr::UsdTimeCode time);

public:
    /**
     * @brief Construct a FieldArrayCache.
     *
     * @param usdUtils Pointer to IUsdUtils interface for querying time samples. Must remain valid
     *                 for the lifetime of this cache.
     */
    explicit FieldArrayCache(IUsdUtils* usdUtils);
    ~FieldArrayCache() = default;

    /**
     * @brief Check if a field array is cached for the given prim and time.
     *
     * @note This method is primarily intended for testing. Use get() for normal cache lookups.
     *
     * @param prim The USD prim to check
     * @param time The time code to check
     * @return true if an entry exists in the cache, false otherwise
     */
    bool contains(pxr::UsdPrim prim, pxr::UsdTimeCode time) const;

    /**
     * @brief Add a field array to the cache.
     *
     * The field array is only cached if caching is enabled for this prim/time combination
     * (based on the cacheMode setting). If caching is disabled, this method
     * logs "[array-cache]:skip(add)" and returns without storing the entry.
     *
     * The cache key used is the snapped time code (see snapTime() for details).
     *
     * @param prim The USD prim associated with the field array
     * @param time The time code associated with the field array
     * @param fieldArray The field array to cache (may be null, in which case nothing is cached)
     */
    void add(pxr::UsdPrim prim, pxr::UsdTimeCode time, carb::ObjectPtr<IFieldArray> fieldArray);

    /**
     * @brief Get a field array from the cache.
     *
     * Returns a cached field array if one exists for the given prim and time. The lookup
     * uses the snapped time code as the cache key (see snapTime() for details).
     *
     * If caching is disabled for this prim/time, logs "[array-cache]:skip(get)" and returns null.
     * If no entry is found, logs "[array-cache]:miss" and returns null.
     * If an entry is found, logs "[array-cache]:hit" and returns the cached field array.
     *
     * @param prim The USD prim to look up
     * @param time The time code to look up
     * @return The cached field array, or null if not found or caching is disabled
     */
    carb::ObjectPtr<IFieldArray> get(pxr::UsdPrim prim, pxr::UsdTimeCode time) const;

    /**
     * @brief Clear all entries from the cache.
     *
     * This is automatically called when a stage is attached or detached (via IStageUpdate callbacks).
     * Logs "[array-cache]:clear" with the number of entries cleared.
     */
    void clear();

    /**
     * @brief Handle USD notice for cache invalidation.
     *
     * This method is called automatically via UsdNoticeListener when USD prims or attributes change.
     * It invalidates cache entries for:
     * - Prims whose attributes have been modified (GetChangedInfoOnlyPaths)
     * - Prims that have been resynced or are part of a resynced tree (GetResyncedPaths)
     *
     * Logs "[array-cache]:drop" for each entry that is invalidated.
     *
     * @param notice The USD objects changed notice
     */
    void handleNotice(const pxr::UsdNotice::ObjectsChanged& notice);
};

} // namespace data
} // namespace cae
} // namespace omni
