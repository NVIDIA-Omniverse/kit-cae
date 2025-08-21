// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#include "Hdf5DataDelegate.h"

// .clang-format off
#include <omni/cae/data/DataDelegateUtilsIncludes.h>
// #include <omni/usd/UtilsIncludes.h>
// .clang-format on

#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <omni/cae/data/DataDelegateUtils.h>
#include <omni/cae/data/IFieldArrayUtils.h>
#include <omniCae/hdf5FieldArray.h>
// #include <omni/usd/UsdUtils.h>

// HDF5 dependencies
#if defined(_MSC_VER) /* MSVC Compiler Case */
#    pragma warning(push)
#    pragma warning(disable : 4251)
#    include <H5Cpp.h>
#    pragma warning(pop)
#else
#    include <H5Cpp.h>
#endif

#include <functional>
#include <numeric>

namespace omni
{
namespace cae
{
namespace data
{
namespace hdf5
{

bool DataDelegate::canProvide(pxr::UsdPrim prim) const
{
    return (prim.IsValid() && prim.IsA<pxr::OmniCaeHdf5FieldArray>());
}

ElementType getElementType(H5::DataSet dataset, bool convert_double_to_float)
{
    H5T_class_t cls = dataset.getTypeClass();
    size_t typeSize = dataset.getDataType().getSize();
    if (cls == H5T_FLOAT)
    {
        return ElementType::float32;
    }
    else if (cls == H5T_INTEGER)
    {
        H5::IntType intType(dataset);
        if (intType.getSign() == H5T_SGN_NONE)
        {
            return typeSize == 8 ? ElementType::uint64 : ElementType::uint32;
        }
        else
        {
            return typeSize == 8 ? ElementType::int64 : ElementType::int32;
        }
    }
    else if (cls == H5T_FLOAT)
    {
        return typeSize == 8 && !convert_double_to_float ? ElementType::float64 : ElementType::float32;
    }
    else
    {
        throw std::runtime_error("Unsupported HDF5 data type.");
    }
}

H5::DataType getDataType(ElementType eType)
{
    switch (eType)
    {
    case ElementType::int32:
        return H5::PredType::NATIVE_INT32;
    case ElementType::int64:
        return H5::PredType::NATIVE_INT64;
    case ElementType::uint32:
        return H5::PredType::NATIVE_UINT32;
    case ElementType::uint64:
        return H5::PredType::NATIVE_UINT64;
    case ElementType::float32:
        return H5::PredType::NATIVE_FLOAT;
    case ElementType::float64:
        return H5::PredType::NATIVE_DOUBLE;
    default:
        throw std::runtime_error("Unsupported element type.");
    }
}

carb::ObjectPtr<IFieldArray> DataDelegate::getFieldArray(pxr::UsdPrim prim, pxr::UsdTimeCode time)
{
    pxr::VtArray<pxr::SdfAssetPath> paths;
    if (!DataDelegateUtils::getFileNames(paths, prim, time))
    {
        return {};
    }

    if (paths.size() > 1)
    {
        CARB_LOG_WARN("Spatially split CGNS files are not supported yet. Only 1st file will be read.");
    }

    const std::string timeStr =
        time.IsDefault() ? "[default]" : (time.IsEarliestTime() ? "[earliest]" : std::to_string(time.GetValue()));

    const pxr::OmniCaeHdf5FieldArray hdf5Array(prim);
    std::string hdf5Path;
    if (!hdf5Array.GetHdf5PathAttr().Get(&hdf5Path, time))
    {
        CARB_LOG_ERROR("Failed to read 'hdf5Path' attribute on prim '%s'", prim.GetPath().GetString().c_str());
        return {};
    }

    if (hdf5Path.empty())
    {
        CARB_LOG_ERROR("Empty 'hdf5Path' attribute on prim '%s' at time '%s'", prim.GetPath().GetString().c_str(),
                       timeStr.c_str());
        return {};
    }

    CARB_LOG_INFO("Reading field '%s'", hdf5Path.c_str());
    std::string fname = paths[0].GetResolvedPath();
    if (fname.empty())
    {
        CARB_LOG_ERROR("Failed to resolve file path for prim '%s' at time '%s'", prim.GetPath().GetString().c_str(),
                       timeStr.c_str());
        return {};
    }

    fname = m_fileUtils->getLocalFilePath(fname);
    if (fname.empty())
    {
        CARB_LOG_ERROR("Failed to get local file path for prim '%s' at time '%s'", prim.GetPath().GetString().c_str(),
                       timeStr.c_str());
        return {};
    }

    try
    {
        CARB_LOG_INFO("open HDF5 file: '%s'", fname.c_str());
        H5::H5File file(fname, H5F_ACC_RDONLY);
        CARB_LOG_INFO("reading dataset '%s'", hdf5Path.c_str());
        H5::DataSet dataset = file.openDataSet(hdf5Path);

        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());

        bool convert_double_to_float = false;
        carb::settings::ISettings* settings = carb::getCachedInterface<carb::settings::ISettings>();
        if (settings->getAsBool("/exts/omni.cae.hdf5/convertDoubleToFloats"))
        {
            convert_double_to_float = true;
        }

        ElementType etype = getElementType(dataset, convert_double_to_float);

        std::vector<uint64_t> shape(rank);
        std::copy(dims.begin(), dims.end(), shape.begin());

        auto farray = m_fieldArrayUtils->createMutableFieldArray(etype, shape, -1, Order::c); // TODO: CONFIRM
        CARB_LOG_INFO("field array size:=%" PRIu64 " bytes", farray->getMutableDataSizeInBytes());

        dataset.read(farray->getMutableData(), getDataType(etype));
        return carb::borrowObject<IFieldArray>(farray.get());
    }
    catch (H5::Exception& err)
    {
        CARB_LOG_ERROR("HDF5 Error: %s", err.getDetailMsg().c_str());
    }
    catch (std::runtime_error& e)
    {
        CARB_LOG_ERROR("Runtime Error: %s", e.what());
    }
    return nullptr;
}

} // namespace hdf5
} // namespace data
} // namespace cae
} // namespace omni
