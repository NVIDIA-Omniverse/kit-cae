// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#pragma once

#include <nv/index/app/application_layer/iapplication_layer.h>
#include <nv/index/app/application_layer/iimporter_factory.h>
#include <nv/index/app/application_layer/iinterface_factory.h>
#include <nv/index/iindex.h>

#include <string>
#include <vector>

namespace omni
{
namespace cae
{
namespace index
{

/**
 * Importer factory to let IndeX know of our importers.
 */
class ImporterFactory : public mi::base::Interface_implement<nv::index::app::IImporter_factory>
{
public:
    ImporterFactory(nv::index::IIndex* index, nv::index::app::IApplication_layer* application_layer);
    ~ImporterFactory();

    const char* get_namespace_string() const override
    {
        return "nv::omni::cae::index";
    }

    mi::Size get_nb_importers() const override
    {
        return m_importer_names.size();
    }

    const char* get_importer_name(mi::Size idx) const override
    {
        if (idx >= m_importer_names.size())
        {
            return nullptr;
        }

        return m_importer_names[idx].c_str();
    }

    const char* get_importer_description(mi::Size /*idx*/) const override
    {
        return "";
    }

    const nv::index::app::IDataset_property_provider* get_dataset_property_provider(const char* importer_name) const override
    {
        return nullptr;
    }

    nv::index::IDistributed_data_import_callback* create_importer(const char* importer_name,
                                                                  const nv::index::app::IProperty_dict* in_dict,
                                                                  nv::index::app::IProperty_dict* out_dict) const override;

private:
    const std::vector<std::string> m_importer_names = { "CaeDataSetImporter" };
    nv::index::IIndex* m_index;
    nv::index::app::IApplication_layer* m_application_layer;
};

/**
 * Needed to make Index aware of our compute technique.
 */
class InterfaceFactory : public mi::base::Interface_implement<nv::index::app::IInterface_factory>
{
public:
    /// Get namespace of this IInterface_factory
    ///
    /// \return namespace string
    const char* get_namespace_string() const override
    {
        return "nv::omni::cae::index";
    }

    /// Get number of create-able IInterface in this factory
    ///
    /// \return number of available IInterfaces
    mi::Size get_nb_iinterfaces() const override
    {
        return m_interface_names.size();
    }

    /// Get idx-th IInterface name
    ///
    /// \param[in] idx available IInterface index
    /// \return IInterface name
    const char* get_iinterface_name(mi::Size idx) const override
    {
        return idx < m_interface_names.size() ? m_interface_names.at(idx).c_str() : nullptr;
    }

    /// Get idx-th IInterface description
    ///
    /// \param[in] idx available IInterface index
    /// \return IInterface description
    const char* get_iinterface_description(mi::Size) const override
    {
        return "";
    }


    /// Create an IInterface with IProperty_dict.
    ///
    /// \param[in]  iinterface_name IInterface name
    /// \param[in]  in_dict         iinterface argument for creation
    /// \param[out] out_dict        (out) result information of iinterface creation
    /// \return an IInterface, nullptr when failed.
    ///
    mi::base::IInterface* create_iinterface(const char* iinterface_name,
                                            const nv::index::app::IProperty_dict* in_dict,
                                            nv::index::app::IProperty_dict* out_dict) const override;

private:
    const std::vector<std::string> m_interface_names = { "CaeDataSetNanoVdbFetchTechnique" };
};

} // namespace index
} // namespace cae
} // namespace omni
