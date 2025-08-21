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

#include <carb/IObject.h>

#include <nv/index/idistributed_data_import_callback.h>
#include <nv/index/iirregular_volume_subset.h>
#include <omni/cae/data/IFieldArray.h>

#include <memory>

namespace omni
{
namespace cae
{
namespace index
{

class DistributedDataProperties;

/// Importer for CaeDataSet. This calls `omni.cae.index.impl.helpers.CaeDataSetImporter` to do the work of populate a
/// nv::index::IIrregular_volume_subset using a CaeDataSetPrim.
class CaeDataSetImporter
    : public nv::index::
          Distributed_continuous_data_import_callback<0x33e5b96b, 0xc005, 0x4276, 0x93, 0xcf, 0xe6, 0x96, 0xc2, 0xce, 0x8c, 0xac>
{
public:
    struct Importer_parameters
    {
        std::string mesh_prim_path; // path to the mesh
        std::vector<std::string> field_names; // name for the field array to read in.
        mi::Float64 time_code = 0.0; // time code for the data set
    };

    CaeDataSetImporter();
    explicit CaeDataSetImporter(const Importer_parameters& params);
    ~CaeDataSetImporter() override;

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
