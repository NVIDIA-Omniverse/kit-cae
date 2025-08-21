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

#include <nv/index/idistributed_compute_technique.h>

#include <memory>

namespace omni
{
namespace cae
{
namespace index
{

class CaeDataSetNanoVdbFetchTechnique
    : public nv::index::Distributed_compute_technique<0xbf2f7a23, 0x4904, 0x4bed, 0xd9, 0xd7, 0xbc, 0x10, 0x71, 0xf8, 0xcc, 0xf1>
{
public:
    struct Compute_parameters
    {
        std::string prim_path; // path to the volume prim
        std::string cache_key; // key to use for cache
        mi::Sint32 execution_tag; // execution tag
        bool enable_interpolation; // enable interpolation
    };

    CaeDataSetNanoVdbFetchTechnique();
    CaeDataSetNanoVdbFetchTechnique(const Compute_parameters& params);
    ~CaeDataSetNanoVdbFetchTechnique() override;

    const char* get_class_name() const override;
    bool is_gpu_operation() const override;
    nv::index::IDistributed_compute_technique::Invocation_mode get_invocation_mode() const override;

    mi::neuraylib::IElement* copy() const override;
    void serialize(mi::neuraylib::ISerializer* serializer) const override;
    void deserialize(mi::neuraylib::IDeserializer* deserializer) override;

    void launch_compute(mi::neuraylib::IDice_transaction* dice_transaction,
                        nv::index::IDistributed_compute_destination_buffer* dst_buffer) const override;

private:
    class Impl;
    mutable std::unique_ptr<Impl> m_impl;
    Compute_parameters m_params;
};


} // namespace index
} // namespace cae
} // namespace omni
