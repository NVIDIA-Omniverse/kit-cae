// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#define CARB_EXPORTS

// .clang-format off
#include <omni/cae/data/IDataDelegateIncludes.h>
// .clang-format on

#include "FieldArrayUtils.h"

#include <carb/PluginUtils.h>
#include <carb/events/EventsUtils.h>
#include <carb/events/IEvents.h>
#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>
#include <carb/tasking/ITasking.h>
#include <carb/tasking/TaskingUtils.h>
#include <carb/thread/Mutex.h>

#include <omni/cae/data/IDataDelegate.h>
#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/cae/data/IDataDelegateRegistry.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omni/cae/data/IFileUtils.h>
#include <omni/kit/IApp.h>
#include <omniCae/fieldArray.h>

#include <OmniClient.h>
#include <string>
#include <vector>

#define EXTENSION_NAME "omni.cae.data.plugin"

CARB_PLUGIN_IMPL_DEPS(carb::logging::ILogging, carb::tasking::ITasking, carb::settings::ISettings, omni::kit::IApp);

namespace omni
{
namespace cae
{
namespace data
{
class DataDelegateRegistry;
class FileUtils;
} // namespace data
} // namespace cae
} // namespace omni

static omni::cae::data::DataDelegateRegistry* s_registry;
static omni::cae::data::FieldArrayUtils* s_utils;
static omni::cae::data::FileUtils* s_fileUtils;

namespace omni
{
namespace cae
{
namespace data
{

constexpr carb::events::EventType kActNameEventType = CARB_EVENTS_TYPE_FROM_STR("omni.kit.window.status_bar@activity");

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

class FieldArrayCache
{
    struct CacheEntry
    {
        std::map<double, carb::ObjectPtr<IFieldArray>> m_fieldArrayMap;
        pxr::UsdAttributeVector m_timeVaryingAttrs;
        pxr::UsdPrim m_prim;

        bool contains(double time) const
        {
            auto key = get_key(time);
            return m_fieldArrayMap.find(key) != m_fieldArrayMap.end();
        }

        void add(pxr::UsdPrim prim, double time, carb::ObjectPtr<IFieldArray> fieldArray)
        {
            if (m_prim != prim)
            {
                init(prim);
            }
            m_fieldArrayMap[get_key(time)] = fieldArray;
        }

        carb::ObjectPtr<IFieldArray> get(double time) const
        {
            auto key = get_key(time);
            auto iter = m_fieldArrayMap.find(key);
            if (iter != m_fieldArrayMap.end())
            {
                return iter->second;
            }
            return {};
        }

    private:
        /// over all time varying attributes, returns the largest time that is
        /// less than or equal to the given time. if there are no time varying
        /// attributes, returns the earliest time.
        double get_key(double time) const
        {
            double key = pxr::UsdTimeCode::EarliestTime().GetValue();
            for (const auto& attr : m_timeVaryingAttrs)
            {
                double lower, upper;
                bool hasTimeSamples;
                if (attr.GetBracketingTimeSamples(time, &lower, &upper, &hasTimeSamples) && hasTimeSamples)
                {
                    key = std::max(key, lower);
                }
            }
            return key;
        }

        void init(pxr::UsdPrim prim)
        {
            m_prim = prim;
            m_timeVaryingAttrs.clear();
            m_fieldArrayMap.clear();

            const auto primAttrs = prim.GetAttributes();
            std::copy_if(primAttrs.begin(), primAttrs.end(), std::back_inserter(m_timeVaryingAttrs),
                         [](const auto& attr) { return attr.ValueMightBeTimeVarying(); });
        }
    };

    std::unique_ptr<UsdNoticeListener<FieldArrayCache>> m_listener;
    std::map<pxr::SdfPath, CacheEntry> m_cache;
    mutable carb::thread::mutex m_cacheMutex;

public:
    FieldArrayCache() : m_listener(new UsdNoticeListener<FieldArrayCache>(this))
    {
    }
    ~FieldArrayCache() = default;

    /// NOT INTENDED FOR USE; IT'S PRIMARILY FOR TESTING.
    bool contains(pxr::UsdPrim prim, pxr::UsdTimeCode time) const
    {
        std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
        auto iter = m_cache.find(prim.GetPath());
        return iter != m_cache.end() && iter->second.contains(time.GetValue());
    }

    /// @brief Add a field array to the cache
    void add(pxr::UsdPrim prim, pxr::UsdTimeCode time, carb::ObjectPtr<IFieldArray> fieldArray)
    {
        std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
        auto& entry = m_cache[prim.GetPath()];
        entry.add(prim, time.GetValue(), fieldArray);
    }

    /// @brief Get a field array from the cache
    carb::ObjectPtr<IFieldArray> get(pxr::UsdPrim prim, pxr::UsdTimeCode time) const
    {
        std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
        auto iter = m_cache.find(prim.GetPath());
        if (iter != m_cache.end())
        {
            return iter->second.get(time.GetValue());
        }
        return {};
    }

    /// @brief Clear the cache
    void clear()
    {
        std::lock_guard<carb::thread::mutex> g(m_cacheMutex);
        m_cache.clear();
    }

    void handleNotice(const pxr::UsdNotice::ObjectsChanged& notice)
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
                CARB_LOG_INFO("[cache-drop] %s", changedPrim.GetText());
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
                    CARB_LOG_INFO("[cache-drop] %s", cachedPath.GetText());
                    it = m_cache.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }
    }
};

class DataDelegateRegistry final : public IDataDelegateRegistry
{
    struct Item
    {
        carb::ObjectPtr<IDataDelegate> m_dataDelegate;
        DelegatePriority m_priority;

        Item(carb::ObjectPtr<IDataDelegate> ptr, DelegatePriority priority) : m_dataDelegate(ptr), m_priority(priority)
        {
        }

        bool operator<(const Item& other) const
        {
            return m_priority < other.m_priority;
        }
    };

public:
    DataDelegateRegistry()
    {
        m_tasking = carb::getFramework()->acquireInterface<carb::tasking::ITasking>();
    }

    ~DataDelegateRegistry() = default;

    void registerDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate, DelegatePriority priority = 0) override
    {
        if (!dataDelegate)
        {
            return;
        }

        std::lock_guard<carb::thread::mutex> g(m_registeredDelegatesMutex);
        CARB_LOG_INFO("register delegate (%p) for extension '%s'", dataDelegate.get(), dataDelegate->getExtensionId());
        m_registeredDelegatesByExtensionId.emplace_back(dataDelegate, priority);
        std::push_heap(
            m_registeredDelegatesByExtensionId.begin(), m_registeredDelegatesByExtensionId.end(), std::less<Item>{});
    }

    void deregisterDataDelegate(carb::ObjectPtr<IDataDelegate>& dataDelegate) override
    {
        if (!dataDelegate)
        {
            return;
        }

        std::lock_guard<carb::thread::mutex> g(m_registeredDelegatesMutex);
        auto iter = m_registeredDelegatesByExtensionId.begin();
        while (iter != m_registeredDelegatesByExtensionId.end())
        {
            if (iter->m_dataDelegate == dataDelegate)
            {
                CARB_LOG_INFO("deregistering delegate (%p) for extension '%s'", dataDelegate.get(),
                              dataDelegate->getExtensionId());
                iter = m_registeredDelegatesByExtensionId.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }

    void deregisterAllDataDelegatesForExtension(const char* extensionId) override
    {
        if (extensionId != nullptr)
        {
            std::lock_guard<carb::thread::mutex> g(m_registeredDelegatesMutex);
            CARB_LOG_INFO("deregistering all delegates for extension %s'", extensionId);
            auto iter = m_registeredDelegatesByExtensionId.begin();
            while (iter != m_registeredDelegatesByExtensionId.end())
            {
                auto* id = iter->m_dataDelegate->getExtensionId();
                if (id && strcmp(id, extensionId) == 0)
                {
                    CARB_LOG_INFO("deregistering delegate (%p)'", iter->m_dataDelegate.get());
                    iter = m_registeredDelegatesByExtensionId.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
        }
    }

    carb::tasking::Future<carb::ObjectPtr<IFieldArray>> getFieldArrayAsync(
        pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time = pxr::UsdTimeCode::Default()) override
    {
        if (!m_tasking)
        {
            CARB_LOG_ERROR("Missing tasking interface!!!");
            return {};
        }
        return m_tasking->addTask(carb::tasking::Priority::eDefault, {},
                                  [this, fieldArrayPrim, time]() { return this->getFieldArray(fieldArrayPrim, time); });
    }

    carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) override
    {
        carb::settings::ISettings* settings = carb::getCachedInterface<carb::settings::ISettings>();
        const bool useCache = (settings->getAsBool("/persistent/exts/omni.cae.data/enableCache"));

        if (useCache)
        {
            if (auto array = m_fieldArrayCache.get(fieldArrayPrim, time))
            {
                CARB_LOG_INFO("[cache-hit] %s, time=%.3f", fieldArrayPrim.GetPath().GetText(), time.GetValue());
                return array;
            }
            CARB_LOG_INFO("[cache-miss] %s, time=%.3f", fieldArrayPrim.GetPath().GetText(), time.GetValue());
        }
        else
        {
            CARB_LOG_INFO("[cache-skip] %s, time=%.3f", fieldArrayPrim.GetPath().GetText(), time.GetValue());
        }

        carb::ObjectPtr<IDataDelegate> delegate = getDelegate(fieldArrayPrim);
        if (!delegate)
        {
            CARB_LOG_WARN("Could not find delegate for prim '%s' at path '%s'",
                          fieldArrayPrim.GetTypeName().GetString().c_str(), fieldArrayPrim.GetPath().GetString().c_str());
            return {};
        }
        CARB_LOG_INFO("reading %s", fieldArrayPrim.GetPath().GetText());
        auto farray = delegate->getFieldArray(fieldArrayPrim, time);
        if (farray.get() != nullptr && useCache)
        {
            CARB_LOG_INFO("[cache-update] %s, time=%.3f", fieldArrayPrim.GetPath().GetText(), time.GetValue());
            m_fieldArrayCache.add(fieldArrayPrim, time, farray);
        }
        return farray;
    }

    carb::ObjectPtr<IDataDelegate> getDelegate(pxr::UsdPrim fieldArrayPrim) const
    {
        if (!fieldArrayPrim.IsValid() || !fieldArrayPrim.IsA<pxr::OmniCaeFieldArray>())
        {
            CARB_LOG_WARN("OmniCaeFieldArray or subtype expected!");
            return {};
        }

        std::lock_guard<carb::thread::mutex> g(m_registeredDelegatesMutex);
        for (const auto& item : m_registeredDelegatesByExtensionId)
        {
            if (item.m_dataDelegate->canProvide(fieldArrayPrim))
            {
                return item.m_dataDelegate;
            }
        }
        return {};
    }

    bool isFieldArrayCached(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) const override
    {
        return m_fieldArrayCache.contains(fieldArrayPrim, time.GetValue());
    }

private:
    mutable carb::thread::mutex m_registeredDelegatesMutex;
    std::vector<Item> m_registeredDelegatesByExtensionId;

    std::map<pxr::SdfPath, std::pair<carb::ObjectPtr<IFieldArray>, pxr::UsdPrim>> m_cache;
    FieldArrayCache m_fieldArrayCache;

    carb::tasking::ITasking* m_tasking;
};

class FileUtils final : public IFileUtils
{
public:
    struct Context
    {
        carb::tasking::TaskGroup task;
        std::string localFileName;
        OmniClientResult result;
    };

    static const char* hackFixWindowsLocalPath(const char* localFilePath)
    {
#if defined(_WIN32)
        if (localFilePath && (localFilePath[0] == '/') && (localFilePath[1] != 0) && (localFilePath[2] == ':'))
        {
            // skip the erroneous backslash in windows
            localFilePath++;
        }
#endif // #if defined(_WIN32)
        return localFilePath;
    }

    std::string getLocalFilePath(const std::string& filePath) override
    {
        Context context;
        context.task.enter();
        auto requestId = omniClientGetLocalFile(
            filePath.c_str(), /*download*/ true, &context,
            [](void* userData, OmniClientResult res, const char* localFileName) noexcept
            {
                Context* context = reinterpret_cast<Context*>(userData);
                context->result = res;
                context->localFileName = localFileName ? FileUtils::hackFixWindowsLocalPath(localFileName) : "";
                context->task.leave();
            });
        context.task.wait();
        (void)requestId;
        if (context.result < eOmniClientResult_Error)
        {
            return std::string(context.localFileName);
        }
        return context.localFileName;
    }
};

static IDataDelegateRegistry* getDataDelegateRegistry()
{
    return s_registry;
}

static IFieldArrayUtils* getFieldArrayUtils()
{
    return s_utils;
}

static IFileUtils* getFileUtils()
{
    return s_fileUtils;
}

} // namespace data
} // namespace cae
} // namespace omni

const struct carb::PluginImplDesc kPluginImpl = { EXTENSION_NAME, "Omni CAE Data plugin", "NVIDIA",
                                                  carb::PluginHotReload::eDisabled, "dev" };
CARB_PLUGIN_IMPL(kPluginImpl, omni::cae::data::IDataDelegateInterface)

void fillInterface(omni::cae::data::IDataDelegateInterface& iface)
{
    using namespace omni::cae::data;
    iface = { getDataDelegateRegistry, getFieldArrayUtils, getFileUtils };
}

CARB_EXPORT void carbOnPluginStartup()
{
    s_registry = new omni::cae::data::DataDelegateRegistry();
    s_utils = new omni::cae::data::FieldArrayUtils();
    s_fileUtils = new omni::cae::data::FileUtils();
}

CARB_EXPORT void carbOnPluginShutdown()
{
    delete s_registry;
    s_registry = nullptr;
    delete s_utils;
    s_utils = nullptr;
    delete s_fileUtils;
    s_fileUtils = nullptr;
}
