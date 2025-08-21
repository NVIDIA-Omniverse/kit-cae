// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#include "Importers.h"

#include "CaeDataSetImporter.h"
#include "CaeDataSetNanoVdbFetchTechnique.h"

#include <carb/logging/Log.h>

#include <nv/index/app/application_layer/property_reader.h>

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

    m_index->register_serializable_class<CaeDataSetImporter>();
}

ImporterFactory::~ImporterFactory()
{
}

nv::index::IDistributed_data_import_callback* ImporterFactory::create_importer(
    const char* importer_name, const nv::index::app::IProperty_dict* in_dict, nv::index::app::IProperty_dict* out_dict) const
{
    if (strcmp(importer_name, "CaeDataSetImporter") == 0)
    {
        const nv::index::app::Property_reader reader(in_dict);
        const mi::Sint32 nb_fields = reader.get_property<mi::Sint32>("nb_fields", 0);

        CaeDataSetImporter::Importer_parameters params;
        for (mi::Sint32 cc = 0; cc < nb_fields; ++cc)
        {
            const std::string field_name_str = reader.get_property(std::string("field:") + std::to_string(cc));
            if (field_name_str.empty())
            {
                CARB_LOG_ERROR("Invalid 'field_name' specified for field:%d.", cc);
                return nullptr;
            }

            params.field_names.push_back(field_name_str);
        }

        params.mesh_prim_path = reader.get_property("mesh");
        if (params.mesh_prim_path.empty())
        {
            CARB_LOG_ERROR("Invalid 'mesh' prim path specified.");
            return nullptr;
        }

        params.time_code = reader.get_property<mi::Float64>("timeCode", 0.0);

        auto* importer = new CaeDataSetImporter(params);
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
    if (name_str == "CaeDataSetNanoVdbFetchTechnique")
    {
        auto size = in_dict->size();
        for (mi::Size cc = 0; cc < size; ++cc)
        {
            mi::base::Handle<mi::IString> key(in_dict->get_key(cc));
            mi::base::Handle<mi::IString> val(in_dict->get_value(key->get_c_str(), "(none)"));
            CARB_LOG_INFO("%s = %s", key->get_c_str(), val->get_c_str());
        }

        const nv::index::app::Property_reader reader(in_dict);

        CaeDataSetNanoVdbFetchTechnique::Compute_parameters params;
        params.prim_path = reader.get_property("prim");
        params.cache_key = reader.get_property("cache_key");
        params.execution_tag = reader.get_property<mi::Sint32>("execution_tag");
        params.enable_interpolation = reader.get_property<bool>("enable_interpolation", false);
        if (params.prim_path.empty())
        {
            CARB_LOG_ERROR("Invalid 'prim' path specified.");
            return nullptr;
        }
        if (params.cache_key.empty())
        {
            CARB_LOG_ERROR("Invalid 'cache_key' specified.");
            return nullptr;
        }
        if (params.execution_tag <= 0)
        {
            CARB_LOG_ERROR("Invalid execution tag specified.");
            return nullptr;
        }
        return new CaeDataSetNanoVdbFetchTechnique(params);
    }
    return nullptr;
}


} // namespace index
} // namespace cae
} // namespace omni
