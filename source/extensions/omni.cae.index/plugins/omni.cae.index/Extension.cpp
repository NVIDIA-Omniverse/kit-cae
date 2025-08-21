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

#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>

#include <omni/cae/data/IDataDelegateInterface.h>
#include <omni/ext/IExt.h>
#include <rtx/index/IndexInstance.h>

// nvindex includes
#include "CaeDataSetImporter.h"
#include "CaeDataSetNanoVdbFetchTechnique.h"
#include "Importers.h"

#include <nv/index/app/application_layer/iapplication_layer.h>
#include <nv/index/app/application_layer/iimporter_manager.h>
#include <nv/index/app/application_layer/iinterface_factory_manager.h>
#include <nv/index/iindex.h>

#include <cassert>

#define EXTENSION_NAME "omni.cae.index.plugin"

using namespace carb;
CARB_PLUGIN_IMPL_DEPS(omni::cae::data::IDataDelegateInterface, rtx::index::IndexInstance, carb::logging::ILogging);

class Extension : public omni::ext::IExt
{
    std::string m_extensionId;
    omni::cae::data::IDataDelegateInterface* m_idataDelegate = nullptr;
    nv::index::IIndex* m_iindex = nullptr;
    nv::index::app::IApplication_layer* m_application_layer = nullptr;
    mi::base::Handle<nv::index::app::IImporter_manager> m_importer_manager;
    mi::base::Handle<nv::index::app::IInterface_factory_manager> m_interface_manager;
    mi::base::Handle<omni::cae::index::ImporterFactory> m_importer_factory;
    mi::base::Handle<omni::cae::index::InterfaceFactory> m_interface_factory;

public:
    void onStartup(const char* extId) override
    {
        this->m_extensionId = extId;
        m_idataDelegate = carb::getFramework()->acquireInterface<omni::cae::data::IDataDelegateInterface>();
        auto* ifaceIndex = carb::getFramework()->acquireInterface<rtx::index::IndexInstance>();
        m_iindex = ifaceIndex->getInterface<nv::index::IIndex>();
        m_application_layer = ifaceIndex->getInterface<nv::index::app::IApplication_layer>();

        // // The IndexUsd interface may not be available when running tests, e.g. omni.hydra.index_remote
        // if (carb::getFramework()->tryAcquireInterface<omni::indexusd::IndexUsd>())
        // {
        // m_iindex->register_serializable_class<omni::cae::index::CaeDataSetImporter>();
        m_iindex->register_serializable_class<omni::cae::index::CaeDataSetNanoVdbFetchTechnique>();
        // }
        // else
        // {
        //     CARB_LOG_INFO(
        //         "Could not acquire omni::indexusd::IndexUsd interface, will skip registration of USD importers.");
        // }


        // Retrieve and store the factory for creating importers
        m_importer_manager = m_application_layer->get_api_component<nv::index::app::IImporter_manager>();
        if (m_importer_manager)
        {
            m_importer_factory = new omni::cae::index::ImporterFactory(m_iindex, m_application_layer);
            if (m_importer_manager->register_importer_factory(m_importer_factory.get()) == 0)
            {
                CARB_LOG_INFO("Index importers for namespace '%s' registered successfully",
                              m_importer_factory->get_namespace_string());
            }
            else
            {
                CARB_LOG_ERROR("Failed to register omni::cae::index::ImporterFactory.");
            }
        }
        else
        {
            CARB_LOG_ERROR("IImporter_manager is not available.");
        }


        m_interface_manager = m_application_layer->get_api_component<nv::index::app::IInterface_factory_manager>();
        assert(m_interface_manager.is_valid_interface());
        m_interface_factory = new omni::cae::index::InterfaceFactory();
        if (m_interface_manager->register_iinterface_factory(m_interface_factory.get()) == 0)
        {
            CARB_LOG_WARN("Index interfaces for namespace '%s' registered successfully",
                          m_interface_factory->get_namespace_string());
        }
        else
        {
            CARB_LOG_ERROR("Failed to register omni::cae::index::InterfaceFactory.");
        }
    }

    void onShutdown() override
    {
        if (m_importer_manager && m_importer_factory)
        {
            m_importer_manager->unregister_importer_factory(m_importer_factory->get_namespace_string());
        }
    }
};

const struct carb::PluginImplDesc kPluginImpl = { EXTENSION_NAME, "Omni CAE IndeX Plugin.", "NVIDIA",
                                                  carb::PluginHotReload::eDisabled, "dev" };
CARB_PLUGIN_IMPL(kPluginImpl, Extension);
void fillInterface(Extension& iface)
{
}
