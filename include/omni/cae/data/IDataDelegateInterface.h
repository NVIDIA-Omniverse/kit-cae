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

#include <carb/Interface.h>

namespace omni
{
namespace cae
{
namespace data
{

class IDataDelegateRegistry;
class IFieldArrayUtils;
class IFileUtils;

/**
 * Carbonite plugin interface for accessing the Data Delegate infrastructure.
 *
 * Changelog
 * - v0.1: Initial version.
 * - v0.2:
 *      - Added IDataDelegateRegistry::getFieldArrayAsync().
 *      - Added IDataDelegateRegistry::isFieldArrayCached().
 * - v0.3:
 *     - Added getFileUtils() to access IFileUtils singleton.
 */
struct IDataDelegateInterface
{
    CARB_PLUGIN_INTERFACE("omni::cae::data::IDataDelegateInterface", 0, 3)

    /**
     * Gets access the IDataDelegateRegistry singleton.
     */
    IDataDelegateRegistry*(CARB_ABI* getDataDelegateRegistry)() = 0;

    /**
     * Gets access the IFieldArrayUtils singleton.
     */
    IFieldArrayUtils*(CARB_ABI* getFieldArrayUtils)() = 0;

    /**
     * Gets access to the IFileUtils singleton.
     */
    IFileUtils*(CARB_ABI* getFileUtils)() = 0;
};

} // namespace data
} // namespace cae
} // namespace omni
