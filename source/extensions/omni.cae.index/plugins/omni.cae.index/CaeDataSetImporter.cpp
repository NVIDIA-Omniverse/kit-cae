// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.


#include "CaeDataSetImporter.h"

#include "DistributedDataProperties.h"

#include <carb/logging/Log.h>

#include <pybind11/attr.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace omni
{
namespace cae
{
namespace index
{

using omni::cae::data::IFieldArray;

class CaeDataSetImporter::Impl
{
    bool m_populated = false;
    std::unique_ptr<py::object> m_importer;

public:
    bool populate(const Importer_parameters& params);
    mi::math::Bbox<mi::Float32, 3> get_bounds();
    bool create(const mi::math::Bbox<mi::Float32, 3>& bbox,
                nv::index::IIrregular_volume_subset* subset,
                mi::Float64 time_code);

    ~Impl()
    {
        if (m_importer)
        {
            py::gil_scoped_acquire acquire;
            m_importer.reset();
        }
    }
};

bool CaeDataSetImporter::Impl::populate(const Importer_parameters& params)
{
    if (!m_populated)
    {
        m_populated = true;
        try
        {
            py::gil_scoped_acquire acquire;
            py::module mdl = py::module::import("omni.cae.index.impl.helpers");
            py::list field_names;
            for (const auto& field : params.field_names)
            {
                field_names.append(field);
            }
            m_importer.reset(new py::object(mdl.attr("CaeDataSetImporter")(params.mesh_prim_path, field_names)));
        }
        catch (const py::import_error& e)
        {
            CARB_LOG_ERROR("Failed to import required Python module. Re-run with --info for details.");
            CARB_LOG_INFO("Python Exception:\n%s", e.what());
        }
        catch (const py::error_already_set& e)
        {
            CARB_LOG_ERROR("Failed to populate. Re-run with --info for details.");
            CARB_LOG_INFO("Python Exception:\n%s", e.what());
        }
    }

    return m_importer && !m_importer->is_none();
}

mi::math::Bbox<mi::Float32, 3> CaeDataSetImporter::Impl::get_bounds()
{
    try
    {
        py::gil_scoped_acquire acquire;
        return m_importer->attr("get_bounds")().cast<mi::math::Bbox<mi::Float32, 3>>();
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("Failed to get_bounds. Re-run with --info for details.");
        CARB_LOG_INFO("Python Exception:\n%s", e.what());
        return {};
    }
}

bool CaeDataSetImporter::Impl::create(const mi::math::Bbox<mi::Float32, 3>& bbox,
                                      nv::index::IIrregular_volume_subset* subset,
                                      mi::Float64 time_code)
{
    try
    {
        py::gil_scoped_acquire acquire;
        m_importer->attr("create")(bbox, subset, time_code);
        return true;
    }
    catch (const py::error_already_set& e)
    {
        CARB_LOG_ERROR("Failed to get_bounds. Re-run with --info for details.");
        CARB_LOG_INFO("Python Exception:\n%s", e.what());
        return false;
    }
}

CaeDataSetImporter::CaeDataSetImporter() : CaeDataSetImporter(Importer_parameters())
{
}

CaeDataSetImporter::CaeDataSetImporter(const Importer_parameters& params)
    : m_importer_params(params), m_impl(new CaeDataSetImporter::Impl())
{
}

CaeDataSetImporter::~CaeDataSetImporter()
{
    py::gil_scoped_acquire acquire;
    m_impl.reset();
}

void CaeDataSetImporter::set_verbose(bool enable)
{
    m_verbose = enable;
}

bool CaeDataSetImporter::get_verbose() const
{
    return m_verbose;
}

void CaeDataSetImporter::serialize(mi::neuraylib::ISerializer* serializer) const
{
    mi::Size len = m_importer_params.mesh_prim_path.size() + 1;
    serializer->write(&len);
    serializer->write(reinterpret_cast<const mi::Uint8*>(m_importer_params.mesh_prim_path.c_str()), len);

    mi::Size nb_fields = m_importer_params.field_names.size();
    serializer->write(&nb_fields);
    for (const auto& field : m_importer_params.field_names)
    {
        len = field.size() + 1;
        serializer->write(&len);
        serializer->write(reinterpret_cast<const mi::Uint8*>(field.c_str()), len);
    }

    serializer->write(&m_importer_params.time_code);
}

void CaeDataSetImporter::deserialize(mi::neuraylib::IDeserializer* deserializer)
{
    mi::Size len;
    std::vector<char> buffer;
    deserializer->read(&len);
    buffer.resize(len, '\0');
    deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
    m_importer_params.mesh_prim_path = &buffer[0];

    mi::Size nb_fields;
    deserializer->read(&nb_fields);
    m_importer_params.field_names.resize(nb_fields);

    for (auto& field : m_importer_params.field_names)
    {
        deserializer->read(&len);
        buffer.resize(0);
        buffer.resize(len, '\0');
        deserializer->read(reinterpret_cast<mi::Uint8*>(&buffer[0]), len);
        field = &buffer[0];
    }

    deserializer->read(&m_importer_params.time_code);
}

const nv::index::IDistributed_data_properties* CaeDataSetImporter::get_dataset_properties() const
{
    if (m_impl->populate(m_importer_params))
    {
        return new DistributedDataProperties(m_impl->get_bounds());
    }
    return nullptr;
}

mi::Size CaeDataSetImporter::estimate(const mi::math::Bbox_struct<mi::Float32, 3>& bounding_box,
                                      mi::neuraylib::IDice_transaction* dice_transaction) const
{
    return 0ull;
}

nv::index::IDistributed_data_subset* CaeDataSetImporter::create(const mi::math::Bbox_struct<mi::Float32, 3>& bbox,
                                                                nv::index::IData_subset_factory* factory,
                                                                mi::neuraylib::IDice_transaction* dice_transaction) const
{
    if (!m_impl->populate(m_importer_params))
    {
        return nullptr;
    }

    mi::base::Handle<nv::index::IIrregular_volume_subset> irregular_volume_subset(
        factory->create_data_subset<nv::index::IIrregular_volume_subset>());
    if (!irregular_volume_subset.is_valid_interface())
    {
        CARB_LOG_ERROR("Cannot create an irregular volume subset.");
        return nullptr;
    }

    if (m_impl->create(bbox, irregular_volume_subset.get(), m_importer_params.time_code))
    {
        irregular_volume_subset->retain();
        return irregular_volume_subset.get();
    }

    CARB_LOG_ERROR("CaeDataSetImporter::create failed!");
    return nullptr;
}


} // namespace index
} // namespace cae
} // namespace omni
