// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "Importers.h"

#include "PythonComputeTask.h"
#include "PythonImporter.h"

#include <carb/logging/Log.h>

#include <nv/index/app/application_layer/property_reader.h>

#include <map>
namespace omni
{
namespace cae
{
namespace index
{


ImporterFactory::ImporterFactory(nv::index::IIndex* index, nv::index::app::IApplication_layer* application_layer)
    : m_index(index), m_application_layer(application_layer)
{
    if (!m_index || !m_application_layer)
    {
        return;
    }

    m_index->register_serializable_class<PythonImporter>();
    m_index->register_serializable_class<PythonComputeTask>();
}

ImporterFactory::~ImporterFactory()
{
}

nv::index::IDistributed_data_import_callback* ImporterFactory::create_importer(
    const char* importer_name, const nv::index::app::IProperty_dict* in_dict, nv::index::app::IProperty_dict* out_dict) const
{
    if (strcmp(importer_name, "PythonImporter") == 0)
    {
        const nv::index::app::Property_reader reader(in_dict);

        PythonImporter::Importer_parameters importer_params;
        importer_params.python_module = reader.get_property("module_name");
        if (importer_params.python_module.empty())
        {
            CARB_LOG_ERROR("Invalid 'module_name' specified.");
            return nullptr;
        }

        importer_params.python_class = reader.get_property("class_name");
        if (importer_params.python_class.empty())
        {
            CARB_LOG_ERROR("Invalid 'class_name' specified.");
            return nullptr;
        }

        // progress all params_* properties
        size_t num_props = in_dict->size();
        for (size_t i = 0; i < num_props; ++i)
        {
            mi::base::Handle<mi::IString> key(in_dict->get_key(i));
            std::string key_str = key->get_c_str();
            if (key_str.find("params_") == 0)
            {
                mi::base::Handle<mi::IString> val(in_dict->get_value(key_str.c_str(), ""));
                importer_params.params[key_str.substr(strlen("params_"))] = val->get_c_str();
            }
        }

        auto* importer = new PythonImporter(importer_params);
        const std::string verbose = reader.get_property("is_verbose", "false");
        if (verbose == "true" || verbose == "yes" || verbose == "1")
        {
            importer->set_verbose(true);
        }
        return importer;
    }

    return nullptr;
}


mi::base::IInterface* InterfaceFactory::create_iinterface(const char* iinterface_name,
                                                          const nv::index::app::IProperty_dict* in_dict,
                                                          nv::index::app::IProperty_dict* out_dict) const
{
    const std::string name_str(iinterface_name);
    if (name_str == "PythonComputeTask")
    {
        auto size = in_dict->size();
        for (mi::Size cc = 0; cc < size; ++cc)
        {
            mi::base::Handle<mi::IString> key(in_dict->get_key(cc));
            mi::base::Handle<mi::IString> val(in_dict->get_value(key->get_c_str(), "(none)"));
            CARB_LOG_INFO("%s = %s", key->get_c_str(), val->get_c_str());
        }

        const nv::index::app::Property_reader reader(in_dict);

        PythonComputeTask::Compute_parameters params;
        params.python_module = reader.get_property("module_name");
        if (params.python_module.empty())
        {
            CARB_LOG_ERROR("Invalid 'module_name' specified.");
            return nullptr;
        }

        params.python_class = reader.get_property("class_name");
        if (params.python_class.empty())
        {
            CARB_LOG_ERROR("Invalid 'class_name' specified.");
            return nullptr;
        }

        params.is_gpu_operation = reader.get_property<bool>("is_gpu_operation", false);

        // Process all params_* properties
        size_t num_props = in_dict->size();
        for (size_t i = 0; i < num_props; ++i)
        {
            mi::base::Handle<mi::IString> key(in_dict->get_key(i));
            std::string key_str = key->get_c_str();
            if (key_str.find("params_") == 0)
            {
                mi::base::Handle<mi::IString> val(in_dict->get_value(key_str.c_str(), ""));
                params.params[key_str.substr(strlen("params_"))] = val->get_c_str();
            }
        }

        return new PythonComputeTask(params);
    }
    return nullptr;
}


} // namespace index
} // namespace cae
} // namespace omni
