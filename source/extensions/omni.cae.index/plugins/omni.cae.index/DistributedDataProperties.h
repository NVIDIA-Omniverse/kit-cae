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

#include <nv/index/idistributed_data_import_callback.h>

namespace omni
{
namespace cae
{
namespace index
{

using Bbox3f = mi::math::Bbox<mi::Float32, 3>;
using Mat4f = mi::math::Matrix<mi::Float32, 4, 4>;

/// Dataset properties
class DistributedDataProperties final
    : public mi::neuraylib::Base<0x647b2cd6, 0x83cf, 0x4bb5, 0xaf, 0xfe, 0xa1, 0x1a, 0x35, 0xc1, 0xa9, 0x7f, nv::index::IDistributed_data_properties>
{
public:
    DistributedDataProperties() = default;
    DistributedDataProperties(const Bbox3f& bbox, const Mat4f& transform = Mat4f(1.f))
        : m_bbox(bbox), m_transform(transform)
    {
    }
    ~DistributedDataProperties() override = default;

    // from nv::index::IDistributed_data_properties
    mi::math::Bbox_struct<mi::Float32, 3> get_bounding_box() const override
    {
        return m_bbox;
    }
    mi::math::Matrix_struct<mi::Float32, 4, 4> get_transform() const override
    {
        return m_transform;
    };

    // from mi::neuraylib::ISerializable
    void serialize(mi::neuraylib::ISerializer* serializer) const override
    {
        serializer->write(m_bbox.min.begin(), 3);
        serializer->write(m_bbox.max.begin(), 3);

        serializer->write(m_transform.begin(), 16);
    }
    void deserialize(mi::neuraylib::IDeserializer* deserializer) override
    {
        deserializer->read(m_bbox.min.begin(), 3);
        deserializer->read(m_bbox.max.begin(), 3);

        deserializer->read(m_transform.begin(), 16);
    }

protected:
    Bbox3f m_bbox = Bbox3f();
    Mat4f m_transform = Mat4f(1.f);
};

} // namespace index
} // namespace cae
} // namespace omni
