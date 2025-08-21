// SPDX-FileCopyrightText: Copyright (c) 2020-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
#pragma once

#include <carb/Interface.h>

namespace nv
{
namespace index
{
class IIndex;

namespace app
{
class IApplication_layer;
} // namespace app

} // namespace index
} // namespace nv

namespace mi
{
namespace base
{

class IInterface;
struct Uuid;

} // namespace base
} // namespace mi

namespace rtx
{
namespace index
{

enum class IndexRunState : uint32_t
{
    eUninitialized = 0, //< IndeX has not yet been loaded
    eLoaded = 1, //< IndeX has been loaded but not started
    eStarted = 2, //< IndeX is running
    eShutdown = 3 //< IndeX was shut down after running
};

struct IndexInstance
{
    CARB_PLUGIN_INTERFACE("rtx::index::IndexInstance", 0, 1)

    mi::base::IInterface*(CARB_ABI* getInterfaceInternal)(const mi::base::Uuid& interface_id);

    template <class T>
    T* getInterface() const
    {
        mi::base::IInterface* ptr_iinterface = getInterfaceInternal(typename T::IID());
        if (!ptr_iinterface)
        {
            return 0;
        }
        T* ptr_T = static_cast<T*>(ptr_iinterface);
        return ptr_T;
    }

    const char*(CARB_ABI* getIndexLibraryDirectory)();
    IndexRunState(CARB_ABI* getIndexRunState)();

    bool(CARB_ABI* startIndex)();
    bool(CARB_ABI* shutdownIndex)(bool force);
};

} // namespace index
} // namespace rtx
