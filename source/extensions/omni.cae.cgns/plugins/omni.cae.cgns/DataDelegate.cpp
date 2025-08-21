// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#include "DataDelegate.h"

// .clang-format off
#include <omni/cae/data/DataDelegateUtilsIncludes.h>
// #include <omni/usd/UtilsIncludes.h>
// .clang-format on

#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <omni/cae/data/DataDelegateUtils.h>
#include <omniCae/cgnsFieldArray.h>
// #include <omni/usd/UsdUtils.h>

// CGNS dependencies
#include <cgns_io.h>
#include <cgnslib.h>
#include <functional>
#include <numeric>


namespace omni
{
namespace cae
{
namespace data
{
namespace cgns
{

static std::vector<std::string> splitPath(const std::string& path)
{
    if (path.size() <= 1)
    {
        return {};
    }
    const char* c_str = path.c_str();
    const char* first = c_str + 1; // skip first '/'
    const char* last = first;

    std::vector<std::string> components;
    for (; *last; ++last)
    {
        if (*last == '/')
        {
            // End of a component.  Save it.
            components.emplace_back(first, last);
            first = last + 1;
        }
    }

    // Save the last component unless there were no components.
    if (last != (c_str + 1) && last > first)
    {
        components.emplace_back(first, last);
    }
    return components;
}

std::string get_cgio_error()
{
    char error_message[CGIO_MAX_ERROR_LENGTH];
    cgio_error_message(error_message);
    return std::string(error_message);
}

static int get_cgns_node_id(int cgioFile, const std::string& nodePath, double* nodeId)
{
    double rootId;
    if (cgio_get_root_id(cgioFile, &rootId) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("CGNS error: %s", get_cgio_error().c_str());
        return CG_ERROR;
    }

    const std::vector<std::string> parts = splitPath(nodePath);
    double currentNodeId = rootId;
    for (const std::string& part : parts)
    {
        double nextNodeId;
        if (cgio_get_node_id(cgioFile, currentNodeId, part.c_str(), &nextNodeId) != CGIO_ERR_NONE)
        {
            CARB_LOG_ERROR("CGNS error reading node '%s': %s", part.c_str(), get_cgio_error().c_str());
            return CG_ERROR;
        }
        currentNodeId = nextNodeId;
    }
    *nodeId = currentNodeId;
    return CGIO_ERR_NONE;
}

ElementType getElementType(const char* dataType, bool convert_double_to_float)
{
    if (strcmp(dataType, "I4") == 0)
    {
        return ElementType::int32;
    }
    else if (strcmp(dataType, "I8") == 0)
    {
        return ElementType::int64;
    }

    else if (strcmp(dataType, "U4") == 0)
    {
        return ElementType::uint32;
    }

    else if (strcmp(dataType, "U8") == 0)
    {
        return ElementType::uint64;
    }

    else if (strcmp(dataType, "R4") == 0)
    {
        return ElementType::float32;
    }
    else if (strcmp(dataType, "R8") == 0)
    {
        return convert_double_to_float ? ElementType::float32 : ElementType::float64;
    }
    return ElementType::unspecified;
}

const char* getDataType(ElementType eType)
{
    switch (eType)
    {
    case ElementType::int32:
        return "I4";
    case ElementType::int64:
        return "I8";
    case ElementType::uint32:
        return "U4";
    case ElementType::uint64:
        return "U8";
    case ElementType::float32:
        return "R4";
    case ElementType::float64:
        return "R8";
    default:
        return "";
    }
}

bool DataDelegate::canProvide(pxr::UsdPrim prim) const
{
    return (prim.IsValid() && prim.IsA<pxr::OmniCaeCgnsFieldArray>());
}

carb::ObjectPtr<IFieldArray> DataDelegate::getFieldArray(pxr::UsdPrim prim, pxr::UsdTimeCode time)
{
    // this is necessary since CGNS/HDF5 are not thread safe
    std::unique_lock<carb::tasking::MutexWrapper> lock(m_mutex);

    pxr::VtArray<pxr::SdfAssetPath> paths;
    if (!DataDelegateUtils::getFileNames(paths, prim, time))
    {
        return {};
    }

    if (paths.size() > 1)
    {
        CARB_LOG_WARN("Spatially split CGNS files are not supported yet. Only 1st file will be read.");
    }

    const pxr::OmniCaeCgnsFieldArray cgnsArray(prim);
    std::string fieldPath;
    if (!cgnsArray.GetFieldPathAttr().Get(&fieldPath, time))
    {
        CARB_LOG_ERROR("Failed to read 'fieldPath' attribute on prim '%s'", prim.GetPath().GetString().c_str());
        return {};
    }

    const std::string fname = m_fileUtils->getLocalFilePath(paths[0].GetResolvedPath());
    CARB_LOG_INFO("Reading field '%s' from '%s'", fieldPath.c_str(), fname.c_str());

    int cgioFile;
    if (cgio_open_file(fname.c_str(), CGIO_MODE_READ, CGIO_FILE_NONE, &cgioFile) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("Failed to open file (%s). CGNS error: %s", fname.c_str(), get_cgio_error().c_str());
        return {};
    }

    int file_type = CGIO_FILE_NONE;
    if (cgio_get_file_type(cgioFile, &file_type) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("Failed to get file type: %s", get_cgio_error().c_str());
        cgio_close_file(cgioFile);
        return {};
    }

    double currentNodeId;
    if (get_cgns_node_id(cgioFile, fieldPath, &currentNodeId) != CGIO_ERR_NONE)
    {
        cgio_close_file(cgioFile);
        return {};
    }

    char dataType[CGIO_MAX_DATATYPE_LENGTH + 1];
    if (cgio_get_data_type(cgioFile, currentNodeId, dataType) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("CGNS error: %s", get_cgio_error().c_str());
        cgio_close_file(cgioFile);
        return {};
    }

    cgsize_t dimensions[CGIO_MAX_DIMENSIONS];
    int numDimensions;
    if (cgio_get_dimensions(cgioFile, currentNodeId, &numDimensions, dimensions) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("CGNS error: %s", get_cgio_error().c_str());
        cgio_close_file(cgioFile);
        return {};
    }

    bool convert_double_to_float = false;
    carb::settings::ISettings* settings = carb::getCachedInterface<carb::settings::ISettings>();
    if (settings->getAsBool("/exts/omni.cae.cgns/convertDoubleToFloats"))
    {
        // I am not sure if CGIO_FILE_ADF2 supports conversions, but CGIO_FILE_ADF doesn't seem to.
        // To be safe, let's only support HDF5.
        if (file_type == CGIO_FILE_HDF5)
        {
            convert_double_to_float = true;
        }
        else
        {
            CARB_LOG_WARN_ONCE("CGNS: 'convertDoubleToFloats' is enabled, but file type is not HDF5. Hence ignored.");
        }
    }

    ElementType etype = getElementType(dataType, convert_double_to_float);
    if (etype == ElementType::int64)
    {
        auto idx = fieldPath.rfind("/");
        if (idx != std::string::npos && fieldPath.substr(idx + 1) == "ElementConnectivity" &&
            settings->getAsBool("/exts/omni.cae.cgns/convertElementConnectivityToInt32"))
        {
            CARB_LOG_INFO("... Downcasting int64 to int32");
            etype = ElementType::int32;
        }
    }

    if (etype == ElementType::unspecified)
    {
        CARB_LOG_ERROR("CGNS Error: unhandled data type: '%s'", dataType);
        cgio_close_file(cgioFile);
        return {};
    }

    std::vector<uint64_t> shape(numDimensions);
    std::copy(dimensions, dimensions + numDimensions, shape.begin());

    auto farray = m_fieldArrayUtils->createMutableFieldArray(etype, shape, -1, Order::fortran);
    CARB_LOG_INFO("About to read data as type: %s", getDataType(farray->getElementType()));
    if (cgio_read_all_data_type(
            cgioFile, currentNodeId, getDataType(farray->getElementType()), farray->getMutableData()) != CGIO_ERR_NONE)
    {
        CARB_LOG_ERROR("CGNS error: %s", get_cgio_error().c_str());
        cgio_close_file(cgioFile);
        return {};
    }
    cgio_close_file(cgioFile);
    CARB_LOG_INFO("read done");
    return carb::borrowObject<IFieldArray>(farray.get());
}

} // namespace cgns
} // namespace data
} // namespace cae
} // namespace omni
