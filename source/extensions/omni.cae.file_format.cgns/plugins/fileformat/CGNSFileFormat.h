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

#include <pxr/base/tf/token.h>
#include <pxr/pxr.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/layer.h>

#include <iosfwd>
#include <string>

PXR_NAMESPACE_OPEN_SCOPE

/**
 * @class CGNSFileFormat
 *
 * Represents a file format for CGNS data. This presents the CGNS database
 * hierarchy in USD using `omni.cae.schema` for holding actual data
 * references.
 */
class CGNSFileFormat : public SdfFileFormat
{
public:
    // SdfFileFormat overrides
    bool CanRead(const std::string& filePath) const override;
    bool Read(SdfLayer* layer, const std::string& resolvedPath, bool metadataOnly) const override;
    bool WriteToString(const SdfLayer& layer, std::string* str, const std::string& comment = std::string()) const override;
    bool WriteToStream(const SdfSpecHandle& spec, std::ostream& out, size_t indent) const override;

protected:
    SDF_FILE_FORMAT_FACTORY_ACCESS;

    CGNSFileFormat();
    ~CGNSFileFormat() override;
};

// clang-format off
TF_DECLARE_PUBLIC_TOKENS(
	CGNSFileFormatTokens,
	((Id, "CGNSFileFormat"))
	((Version, "1.0"))
	((Target, "usd"))
	((Extension, "cgns"))
	((Base_t, "CGNSBase"))
	((Zone_t, "CGNSZone"))
	((FlowSolution_t, "CGNSFlowSolution"))

);
// clang-format on

TF_DECLARE_WEAK_AND_REF_PTRS(CGNSFileFormat);

PXR_NAMESPACE_CLOSE_SCOPE
