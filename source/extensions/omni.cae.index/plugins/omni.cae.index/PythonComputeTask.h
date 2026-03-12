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

#include <nv/index/idistributed_compute_technique.h>

#include <map>
#include <memory>
#include <string>

namespace omni
{
namespace cae
{
namespace index
{

/// Generic compute task for Python-based IndeX compute programs.
/// This is analogous to PythonImporter but for distributed compute techniques.
/// Calls Python code to populate IndeX compute destination buffers.
class PythonComputeTask
    : public nv::index::Distributed_compute_technique<0x3a8c9f21, 0x5b12, 0x4e3d, 0xa1, 0x2f, 0x9c, 0x45, 0x67, 0x8e, 0xab, 0xcd>
{
public:
    struct Compute_parameters
    {
        std::string python_module; // Python module path
        std::string python_class; // Python class name
        bool is_gpu_operation; // Whether the compute task is a GPU operation
        std::map<std::string, std::string> params; // Additional parameters to pass to the Python class
    };

    PythonComputeTask();
    PythonComputeTask(const Compute_parameters& params);
    ~PythonComputeTask() override;

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
