// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "DataDelegate.h"

#include "OpenFoamUtils.h"

// .clang-format off
#include <omni/cae/data/DataDelegateUtilsIncludes.h>
// #include <omni/usd/UtilsIncludes.h>
// .clang-format on

#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <omni/cae/data/DataDelegateUtils.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omniCaeOpenFoam/fieldArray.h>
#include <omniCaeOpenFoam/tokens.h>

#include <cinttypes>
#include <fstream>
#include <functional>
#include <numeric>


namespace omni
{
namespace cae
{
namespace data
{
namespace openfoam
{

size_t getSizeInBytes(const IFieldArray* array)
{
    if (array)
    {
        auto shape = array->getShape();
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>{}) *
               omni::cae::data::getElementSize(array->getElementType());
    }
    return 0u;
}

carb::ObjectPtr<IFieldArray> readData(const std::string& path,
                                      const pxr::TfToken& type,
                                      omni::cae::data::IFieldArrayUtils* utils)
{
    if (type == pxr::OmniCaeOpenFoamTokens->internalField)
    {
        return readInternalField(path, utils);
    }
    else if (type == pxr::OmniCaeOpenFoamTokens->points)
    {
        return readPoints(path, utils);
    }
    else if (type == pxr::OmniCaeOpenFoamTokens->owner || type == pxr::OmniCaeOpenFoamTokens->neighbour)
    {
        return readOwnerOrNeighbour(path, utils);
    }
    else if (type == pxr::OmniCaeOpenFoamTokens->facesOffsets)
    {
        return readFaces(path, utils, /*offsets*/ true);
    }
    else if (type == pxr::OmniCaeOpenFoamTokens->faces)
    {
        return readFaces(path, utils, /*offset*/ false);
    }
    else
    {
        CARB_LOG_ERROR("Unsupported field array type: %s", type.GetText());
        return {};
    }
    return {};
}

bool DataDelegate::canProvide(pxr::UsdPrim prim) const
{
    return (prim.IsValid() && prim.IsA<pxr::OmniCaeOpenFoamFieldArray>());
}

carb::ObjectPtr<IFieldArray> DataDelegate::getFieldArray(pxr::UsdPrim prim, pxr::UsdTimeCode time)
{
    pxr::VtArray<pxr::SdfAssetPath> paths;
    if (!DataDelegateUtils::getFileNames(paths, prim, time))
    {
        return {};
    }

    if (paths.empty())
    {
        CARB_LOG_ERROR("No file paths found for prim '%s'", prim.GetPath().GetText());
        return {};
    }
    else if (paths.size() > 1)
    {
        // If we have multiple arrays, we need to merge them; however all label arrays need to be converted to
        // global labels.
        CARB_LOG_ERROR_ONCE(
            "Partitioned OpenFOAM field arrays are not correctly supported yet. "
            "This will lead to incorrect results when reading data split across multiple processor-files. "
            "Please use a non-partitioned case or merge the data manually before reading.");
    }

    std::vector<carb::ObjectPtr<IFieldArray>> arrays;
    arrays.resize(paths.size());

    pxr::OmniCaeOpenFoamFieldArray fa(prim);
    std::string type;
    pxr::TfToken typeToken;
    if (!fa.GetTypeAttr().Get(&typeToken, time))
    {
        CARB_LOG_ERROR("Failed to get 'type' from '%s'", prim.GetPath().GetText());
        return {};
    }

    m_tasking->parallelFor(static_cast<size_t>(0), paths.size(),
                           [this, &paths, &typeToken, &arrays](size_t i)
                           {
                               auto data = readData(paths[i].GetResolvedPath(), typeToken, m_fieldArrayUtils);
                               arrays[i] = data;
                           });

    if (arrays.size() == 1)
    {
        return arrays[0];
    }
    else if (arrays.size() > 1)
    {
        // Merge the arrays
        std::vector<uint64_t> shape = arrays[0]->getShape();
        for (size_t i = 1; i < arrays.size(); ++i)
        {
            const auto& arrShape = arrays[i]->getShape();
            if (arrShape[1] != shape[1])
            {
                CARB_LOG_ERROR(
                    "Field arrays have different shapes, cannot merge: %" PRIu64 " vs %" PRIu64, shape[1], arrShape[1]);
                return {};
            }
            shape[0] += arrShape[0]; // Assuming the first dimension is the one to merge
        }
        // Create a new field array with the merged shape
        auto mergedArray = m_fieldArrayUtils->createMutableFieldArray(arrays[0]->getElementType(), shape, -1, Order::c);
        if (!mergedArray)
        {
            CARB_LOG_ERROR("Failed to create merged field array for '%s'", prim.GetPath().GetText());
            return {};
        }
        // Copy data from each array into the merged array
        std::uint8_t* mergedData = mergedArray->getMutableData<std::uint8_t>();
        std::uint64_t offset = 0;
        for (const auto& array : arrays)
        {
            const std::uint8_t* data = array->getData<std::uint8_t>();
            std::size_t size = getSizeInBytes(array.get());
            std::memcpy(mergedData + offset, data, size);
            offset += size;
        }
        return carb::borrowObject<IFieldArray>(mergedArray.get());
    }
    return {};
}

} // namespace openfoam
} // namespace data
} // namespace cae
} // namespace omni
