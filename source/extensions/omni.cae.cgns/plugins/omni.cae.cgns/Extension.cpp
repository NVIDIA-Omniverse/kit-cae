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

#include "DataDelegate.h"

#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>

#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/cae/data/IDataDelegateRegistry.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omni/ext/IExt.h>

#define EXTENSION_NAME "omni.cae.cgns.plugin"

using namespace carb;
CARB_PLUGIN_IMPL_DEPS(omni::cae::data::IDataDelegateInterface, carb::logging::ILogging)

class Extension : public omni::ext::IExt
{
    std::string m_extensionId;

public:
    void onStartup(const char* extId) override
    {
        this->m_extensionId = extId;
        auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        auto* registry = iface->getDataDelegateRegistry();
        auto* utils = iface->getFieldArrayUtils();

        // register CGNS data delegate
        omni::cae::data::IDataDelegatePtr cgnsDelegate =
            carb::stealObject<omni::cae::data::IDataDelegate>(new omni::cae::data::cgns::DataDelegate(extId, utils));
        registry->registerDataDelegate(cgnsDelegate);
    }

    void onShutdown() override
    {
        auto* iface = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        auto* registry = iface->getDataDelegateRegistry();
        registry->deregisterAllDataDelegatesForExtension(m_extensionId.c_str());
    }
};

const struct carb::PluginImplDesc kPluginImpl = { EXTENSION_NAME, "CAE CGNS Data Plugin.", "NVIDIA",
                                                  carb::PluginHotReload::eDisabled, "dev" };
CARB_PLUGIN_IMPL(kPluginImpl, Extension);

void fillInterface(Extension& iface)
{
}
