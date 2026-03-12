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

#include <carb/IObject.h>

#include <nv/index/idistributed_data_import_callback.h>
#include <nv/index/iirregular_volume_subset.h>
#include <omni/cae/data/IFieldArray.h>

#include <map>
#include <memory>

namespace omni
{
namespace cae
{
namespace index
{

class DistributedDataProperties;

/// Generic importer for Python-based data sources. Calls the `create_subset` method on an
/// instance of the caller-supplied `python_module.python_class` to populate a nv::index::IIrregular_volume_subset.
class PythonImporter
    : public nv::index::
          Distributed_continuous_data_import_callback<0x44f6c07c, 0xd11a, 0x5387, 0xa4, 0xd4, 0xf9, 0xa7, 0xd3, 0xdf, 0x9e, 0xbf>
{
public:
    struct Importer_parameters
    {
        std::string python_module; // Python module path
        std::string python_class; // Python class name
        std::map<std::string, std::string> params; // Additional parameters to pass to the Python class
    };

    PythonImporter();
    explicit PythonImporter(const Importer_parameters& params);
    ~PythonImporter() override;

    void set_verbose(bool enable);
    bool get_verbose() const;

    mi::Size estimate(const mi::math::Bbox_struct<mi::Float32, 3>& bounding_box,
                      mi::neuraylib::IDice_transaction* dice_transaction) const override;

    nv::index::IDistributed_data_subset* create(const mi::math::Bbox_struct<mi::Float32, 3>& bbox,
                                                nv::index::IData_subset_factory* factory,
                                                mi::neuraylib::IDice_transaction* dice_transaction) const override;

    mi::base::Uuid subset_id() const override
    {
        return nv::index::IIrregular_volume_subset::IID();
    }

    /// note: caller will call "delete" on the returned object
    const nv::index::IDistributed_data_properties* get_dataset_properties() const override;

    void serialize(mi::neuraylib::ISerializer* serializer) const override;
    void deserialize(mi::neuraylib::IDeserializer* deserializer) override;

private:
    bool m_verbose = false;
    Importer_parameters m_importer_params;

    class Impl;
    mutable std::unique_ptr<Impl> m_impl;
};

} // namespace index
} // namespace cae
} // namespace omni
