// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#define CARB_EXPORTS

// .clang-format off
#include <omni/cae/data/IDataDelegateIncludes.h>
// .clang-format on

#include "FieldArrayCache.h"
#include "FieldArrayUtils.h"
#include "UsdUtils.h"

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
#include <omni/cae/data/IUsdUtils.h>
#include <omni/kit/IApp.h>
#include <omni/kit/IStageUpdate.h>
#include <omniCae/fieldArray.h>
#include <pxr/usd/usdUtils/stageCache.h>

#include <OmniClient.h>
#include <string>
#include <vector>

#define EXTENSION_NAME "omni.cae.data.plugin"

CARB_PLUGIN_IMPL_DEPS(carb::logging::ILogging,
                      carb::tasking::ITasking,
                      carb::settings::ISettings,
                      omni::kit::IApp,
                      omni::kit::IStageUpdate);

namespace omni
{
namespace cae
{
namespace data
{
class DataDelegateRegistry;
class FileUtils;
class UsdUtils;
} // namespace data
} // namespace cae
} // namespace omni

static omni::cae::data::DataDelegateRegistry* s_registry;
static omni::cae::data::FieldArrayUtils* s_utils;
static omni::cae::data::FileUtils* s_fileUtils;
static omni::cae::data::UsdUtils* s_usdUtils;

namespace omni
{
namespace cae
{
namespace data
{

constexpr carb::events::EventType kActNameEventType = CARB_EVENTS_TYPE_FROM_STR("omni.kit.window.status_bar@activity");

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

    class StageUpdateListener
    {
        DataDelegateRegistry* m_registry;
        omni::kit::StageUpdateNode* m_updateNode;

    public:
        StageUpdateListener(DataDelegateRegistry* registry) : m_registry(registry), m_updateNode(nullptr)
        {
            auto* stageUpdate = carb::getFramework()->acquireInterface<omni::kit::IStageUpdate>();
            if (stageUpdate)
            {
                omni::kit::StageUpdateNodeDesc desc = { 0 };
                desc.displayName = "CAE Data Delegate Cache";
                desc.userData = this;
                desc.onAttach = [](long int stageId, double metersPerUnit, void* userData)
                {
                    auto* listener = reinterpret_cast<StageUpdateListener*>(userData);
                    if (listener && listener->m_registry)
                    {
                        listener->m_registry->m_fieldArrayCache.clear();
                    }
                };
                desc.onDetach = [](void* userData)
                {
                    auto* listener = reinterpret_cast<StageUpdateListener*>(userData);
                    if (listener && listener->m_registry)
                    {
                        listener->m_registry->m_fieldArrayCache.clear();
                    }
                };

                auto stageUpdateInterface = stageUpdate->getStageUpdate();
                if (stageUpdateInterface)
                {
                    m_updateNode = stageUpdateInterface->createStageUpdateNode(desc);
                }
            }
        }

        ~StageUpdateListener()
        {
            if (m_updateNode)
            {
                auto* stageUpdate = carb::getFramework()->acquireInterface<omni::kit::IStageUpdate>();
                if (stageUpdate)
                {
                    auto stageUpdateInterface = stageUpdate->getStageUpdate();
                    if (stageUpdateInterface)
                    {
                        stageUpdateInterface->destroyStageUpdateNode(m_updateNode);
                    }
                }
                m_updateNode = nullptr;
            }
        }
    };

public:
    DataDelegateRegistry(IUsdUtils* usdUtils) : m_fieldArrayCache(usdUtils)
    {
        m_tasking = carb::getFramework()->acquireInterface<carb::tasking::ITasking>();
        m_stageUpdateListener = std::make_unique<StageUpdateListener>(this);
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
        // If the field array is cached, return the cached value without creating a new task.
        // this ensures that cached fields arrays are returned immediately, without waiting for the task to complete.
        if (auto array = m_fieldArrayCache.get(fieldArrayPrim, time))
        {
            carb::tasking::Promise<carb::ObjectPtr<IFieldArray>> promise;
            promise.set_value(array);
            return promise.get_future();
        }
        return m_tasking->addTask(carb::tasking::Priority::eDefault, {},
                                  [this, fieldArrayPrim, time]() { return this->getFieldArray(fieldArrayPrim, time); });
    }

    carb::ObjectPtr<IFieldArray> getFieldArray(pxr::UsdPrim fieldArrayPrim, pxr::UsdTimeCode time) override
    {
        if (auto array = m_fieldArrayCache.get(fieldArrayPrim, time))
        {
            return array;
        }

        carb::ObjectPtr<IDataDelegate> delegate = getDelegate(fieldArrayPrim);
        if (!delegate)
        {
            // Use warning instead of error during test runs to avoid polluting test output
            auto settings = carb::getCachedInterface<carb::settings::ISettings>();
            bool isTestRun = settings && settings->getAsInt("/app/isTestRun") == 1;

            if (isTestRun)
            {
                CARB_LOG_WARN(
                    "Could not find delegate for prim '%s' at path '%s'. Are all necessary extensions installed and enabled?",
                    fieldArrayPrim.GetTypeName().GetString().c_str(), fieldArrayPrim.GetPath().GetString().c_str());
            }
            else
            {
                CARB_LOG_ERROR(
                    "Could not find delegate for prim '%s' at path '%s'. Are all necessary extensions installed and enabled?",
                    fieldArrayPrim.GetTypeName().GetString().c_str(), fieldArrayPrim.GetPath().GetString().c_str());
            }
            return {};
        }
        CARB_LOG_INFO("reading %s", fieldArrayPrim.GetPath().GetText());
        auto farray = delegate->getFieldArray(fieldArrayPrim, time);
        if (farray.get() != nullptr)
        {
            // Check if down-conversion from 64-bit to 32-bit is enabled
            auto settings = carb::getCachedInterface<carb::settings::ISettings>();
            bool downConvert64Bit = settings && settings->getAsBool("/persistent/exts/omni.cae.data/downConvert64Bit");

            if (downConvert64Bit)
            {
                ElementType currentType = farray->getElementType();
                ElementType targetType = ElementType::unspecified;

                // Map 64-bit types to their 32-bit counterparts
                switch (currentType)
                {
                case ElementType::int64:
                    targetType = ElementType::int32;
                    break;
                case ElementType::uint64:
                    targetType = ElementType::uint32;
                    break;
                case ElementType::float64:
                    targetType = ElementType::float32;
                    break;
                default:
                    // Not a 64-bit type, no conversion needed
                    break;
                }

                // Perform conversion if needed
                if (targetType != ElementType::unspecified && s_utils)
                {
                    CARB_LOG_INFO(
                        "Converting field array from 64-bit to 32-bit for %s", fieldArrayPrim.GetPath().GetText());
                    farray = s_utils->castAndCopy(farray, targetType);
                }
            }

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
    std::unique_ptr<StageUpdateListener> m_stageUpdateListener;

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

static IUsdUtils* getUsdUtils()
{
    return s_usdUtils;
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
    iface = { getDataDelegateRegistry, getFieldArrayUtils, getFileUtils, getUsdUtils };
}

CARB_EXPORT void carbOnPluginStartup()
{
    s_usdUtils = new omni::cae::data::UsdUtils();
    s_registry = new omni::cae::data::DataDelegateRegistry(s_usdUtils);
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
    delete s_usdUtils;
    s_usdUtils = nullptr;
}
