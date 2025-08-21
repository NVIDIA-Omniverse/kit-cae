// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
//  its affiliates is strictly prohibited.

#include "CGNSFileFormat.h"

#include "debugCodes.h"

#include <omniCae/cgnsFieldArray.h>
#include <omniCae/dataSet.h>
#include <omniCae/pointCloudAPI.h>
#include <omniCae/tokens.h>
#include <omniCaeSids/tokens.h>
#include <omniCaeSids/unstructuredAPI.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/pathUtils.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/usd/usd/specializes.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>

// CGNS dependencies
#include <cgns_io.h>
#include <cgnslib.h>
#include <iostream>
#include <set>

PXR_NAMESPACE_OPEN_SCOPE

namespace detail
{
void call_safe(int ierr, const char* details)
{
    if (ierr != CG_OK)
    {
        // CARB_LOG_ERROR("Failed %s (CGNS error: '%s')", details, cg_get_error());
        // std::cout << "ERROR : " << details << cg_get_error() << std::endl;
        TF_ERROR(KIT_CAE_CGNS_FILEFORMAT, "CGNS error: %s (%d) in %s", cg_get_error(), ierr, details);
        throw std::runtime_error(details);
    }
}

class ScopedPush
{
    std::vector<std::string>& m_vector;

public:
    ScopedPush(std::vector<std::string>& v, const std::string& s) : m_vector(v)
    {
        m_vector.push_back(s);
    }
    ~ScopedPush()
    {
        m_vector.pop_back();
    }
};

template <typename T>
SdfPath MakeChildPath(const T& parentPrim, const std::string& name)
{
    return parentPrim.GetPath().AppendChild(TfToken(TfMakeValidIdentifier(name)));
}

TfToken GetElementType(CGNS_ENUMT(ElementType_t) t)
{
#define DO_CASE(x)                                                                                                     \
    case CGNS_ENUMV(x):                                                                                                \
        return TfToken(#x);

    switch (t)
    {
        DO_CASE(ElementTypeNull);
        DO_CASE(ElementTypeUserDefined);
        DO_CASE(NODE);
        DO_CASE(BAR_2);
        DO_CASE(BAR_3);
        DO_CASE(TRI_3);
        DO_CASE(TRI_6);
        DO_CASE(QUAD_4);
        DO_CASE(QUAD_8);
        DO_CASE(QUAD_9);
        DO_CASE(TETRA_4);
        DO_CASE(TETRA_10);
        DO_CASE(PYRA_5);
        DO_CASE(PYRA_14);
        DO_CASE(PENTA_6);
        DO_CASE(PENTA_15);
        DO_CASE(PENTA_18);
        DO_CASE(HEXA_8);
        DO_CASE(HEXA_20);
        DO_CASE(HEXA_27);
        DO_CASE(MIXED);
        DO_CASE(PYRA_13);
        DO_CASE(NGON_n);
        DO_CASE(NFACE_n);
        DO_CASE(BAR_4);
        DO_CASE(TRI_9);
        DO_CASE(TRI_10);
        DO_CASE(QUAD_12);
        DO_CASE(QUAD_16);
        DO_CASE(TETRA_16);
        DO_CASE(TETRA_20);
        DO_CASE(PYRA_21);
        DO_CASE(PYRA_29);
        DO_CASE(PYRA_30);
        DO_CASE(PENTA_24);
        DO_CASE(PENTA_38);
        DO_CASE(PENTA_40);
        DO_CASE(PENTA_75);
        DO_CASE(HEXA_32);
        DO_CASE(HEXA_56);
        DO_CASE(HEXA_64);
        DO_CASE(HEXA_44);
        DO_CASE(HEXA_98);
        DO_CASE(HEXA_125);
    default:
        break;
    }
    return OmniCaeSidsTokens->ElementTypeNull;
}

TfToken GetGridLocation(CGNS_ENUMT(GridLocation_t) t)
{
    switch (t)
    {
    case CGNS_ENUMV(Vertex):
        return OmniCaeTokens->vertex;
    case CGNS_ENUMV(CellCenter):
        return OmniCaeTokens->cell;
    // case CGNS_ENUMV(IFaceCenter):
    // case CGNS_ENUMV(JFaceCenter):
    // case CGNS_ENUMV(KFaceCenter):
    // case CGNS_ENUMV(FaceCenter):
    //     return OmniCaeTokens->faceCenter;
    // case CGNS_ENUMV(EdgeCenter):
    //     return OmniCaeTokens->edgeCenter;
    // case CGNS_ENUMV(GridLocationUserDefined):
    // case CGNS_ENUMV(GridLocationNull):
    default:
        return OmniCaeTokens->none; // not supported yet.
    }
}

SdfLayerRefPtr ReadCGNS(int cgFile, const std::string& fname, const pxr::SdfLayer::FileFormatArguments& args)
{
    auto layer = pxr::SdfLayer::CreateAnonymous();
    auto stage = pxr::UsdStage::Open(layer);
    UsdGeomSetStageUpAxis(stage, pxr::UsdGeomTokens->z);

    auto world = UsdGeomXform::Define(stage, SdfPath("/World"));
    layer->SetDefaultPrim(world.GetPath().GetNameToken());

    auto iter = args.find("rootName");
    TfToken rootName((iter == args.end()) ? TfMakeValidIdentifier(TfGetBaseName(fname)) :
                                            TfMakeValidIdentifier(iter->second));

    iter = args.find("assetPath");
    auto assetPath = iter == args.end() ? SdfAssetPath(fname) : SdfAssetPath(iter->second);

    // add container for the file.
    auto scope = UsdGeomScope::Define(stage, world.GetPath().AppendChild(rootName));

    // Add class that can be used for all field arrays in this file.
    auto fieldArrayClass = stage->CreateClassPrim(MakeChildPath(scope, "CGNSFieldArrayClass"));
    OmniCaeFieldArray(fieldArrayClass).CreateFileNamesAttr().Set(VtArray<SdfAssetPath>{ assetPath });

    int nbases;
    call_safe(cg_nbases(cgFile, &nbases), "read num bases");

    std::vector<std::string> cgnsPath;
    const ScopedPush forRoot(cgnsPath, "");

    char name[1024];
    for (int base = 1; base <= nbases; ++base)
    {
        int cell_dim, phys_dim;
        call_safe(cg_base_read(cgFile, base, name, &cell_dim, &phys_dim), "read base");
        const ScopedPush forBase(cgnsPath, name);
        if (cell_dim != 3)
        {
            // warn: only 3d zones are supported;
            continue;
        }

        auto basePrim = stage->DefinePrim(MakeChildPath(scope, name), CGNSFileFormatTokens->Base_t);

        // let's determine if we have timesteps.
        int nsteps = 0;
        std::vector<double> timeValues;
        if (cg_biter_read(cgFile, base, name, &nsteps) == CG_OK)
        {
            timeValues.resize(nsteps);
            call_safe(cg_goto(cgFile, base, name, 0, NULL), "goto BaseIterativeData_t");
            int narrays;
            call_safe(cg_narrays(&narrays), "read narrays");
            for (int na = 1; na <= narrays; ++na)
            {
                CGNS_ENUMT(DataType_t) dtype;
                cgsize_t dims[12];
                int rank;
                call_safe(cg_array_info(na, name, &dtype, &rank, dims), "read array info");
                if (strcmp(name, "TimeValues") == 0)
                {
                    timeValues.resize(dims[0]);
                    call_safe(cg_array_read_as(na, CGNS_ENUMV(RealDouble), timeValues.data()), "read time values");
                    break;
                }
            }
        }

        int nzones;
        call_safe(cg_nzones(cgFile, base, &nzones), "read num zones");
        for (int zone = 1; zone <= nzones; ++zone)
        {
            CGNS_ENUMT(ZoneType_t) zonetype;
            call_safe(cg_zone_type(cgFile, base, zone, &zonetype), "read zone type");
            if (zonetype == CGNS_ENUMV(Structured))
            {
                // warn: skipping structured zones for now
                continue;
            }

            cgsize_t size[9];
            call_safe(cg_zone_read(cgFile, base, zone, name, size), "read zone");
            const ScopedPush forZone(cgnsPath, name);
            auto zonePrim = stage->DefinePrim(MakeChildPath(basePrim, name), CGNSFileFormatTokens->Zone_t);

            // for zone, get the grid coords first.
            int ngrids;
            call_safe(cg_ngrids(cgFile, base, zone, &ngrids), "read num grid coordinates");

            int ncoords;
            call_safe(cg_ncoords(cgFile, base, zone, &ncoords), "read num coords");

            SdfPathVector gridCoordinatePaths;
            SdfPathVector ngonPaths;
            SdfPathVector nfacePaths;
            OmniCaePointCloudAPI gridCoordinates;

            for (int grid = 1; grid <= std::min(1, ngrids); ++grid) // we only process 1st grid coordinate.
            {
                call_safe(cg_grid_read(cgFile, base, zone, grid, name), "read grid coordinate");
                const ScopedPush forGridCoordinate(cgnsPath, name);

                // create a CaeDataSet print for just the points.
                OmniCaeDataSet gcT = OmniCaeDataSet::Define(stage, MakeChildPath(zonePrim, name));
                auto gcPrim = gcT.GetPrim();
                OmniCaePointCloudAPI::Apply(gcPrim);
                for (int coord = 1; coord <= ncoords; ++coord)
                {
                    CGNS_ENUMT(DataType_t) datatype;
                    call_safe(cg_coord_info(cgFile, base, zone, coord, &datatype, name), "read coord info");
                    OmniCaeCgnsFieldArray arrayT = OmniCaeCgnsFieldArray::Define(stage, MakeChildPath(gcPrim, name));
                    const ScopedPush forGridCoordinateCoord(cgnsPath, name);
                    arrayT.GetPrim().GetSpecializes().SetSpecializes({ fieldArrayClass.GetPath() });
                    arrayT.CreateFieldAssociationAttr().Set(OmniCaeTokens->vertex);
                    arrayT.CreateFieldPathAttr().Set(TfStringJoin(cgnsPath, "/"));
                    gridCoordinatePaths.push_back(arrayT.GetPath());
                }

                OmniCaePointCloudAPI pcAPI(gcPrim);
                pcAPI.CreateCoordinatesRel().SetTargets(gridCoordinatePaths);
                gridCoordinates = pcAPI;
            }

            // Read Sections aka Element_t nodes
            int nsections;
            call_safe(cg_nsections(cgFile, base, zone, &nsections), "read num sections");
            std::vector<OmniCaeDataSet> sections;
            for (int section = 1; section <= nsections; ++section)
            {
                // printf("reading zone=%d, section=%d\n", zone, section);
                CGNS_ENUMT(ElementType_t) elementType;
                cgsize_t start, end;
                int nbndry, parent_flag;
                call_safe(cg_section_read(
                              cgFile, base, zone, section, name, &elementType, &start, &end, &nbndry, &parent_flag),
                          "read section info");
                const ScopedPush forSection(cgnsPath, name);

                OmniCaeDataSet datasetT = OmniCaeDataSet::Define(stage, MakeChildPath(zonePrim, name));
                OmniCaeSidsUnstructuredAPI::Apply(datasetT.GetPrim());
                OmniCaeSidsUnstructuredAPI sidsAPI(datasetT.GetPrim());
                sidsAPI.CreateElementTypeAttr().Set(GetElementType(elementType));
                sidsAPI.CreateElementRangeStartAttr().Set(static_cast<uint64_t>(start));
                sidsAPI.CreateElementRangeEndAttr().Set(static_cast<uint64_t>(end));
                sidsAPI.CreateGridCoordinatesRel().SetTargets(gridCoordinatePaths);

                {
                    const ScopedPush forConn(cgnsPath, "ElementConnectivity");
                    OmniCaeCgnsFieldArray arrayT =
                        OmniCaeCgnsFieldArray::Define(stage, MakeChildPath(datasetT, "ElementConnectivity"));
                    arrayT.GetPrim().GetSpecializes().SetSpecializes({ fieldArrayClass.GetPath() });
                    arrayT.CreateFieldAssociationAttr().Set(OmniCaeTokens->none);
                    arrayT.CreateFieldPathAttr().Set(TfStringJoin(cgnsPath, "/"));
                    sidsAPI.GetElementConnectivityRel().SetTargets({ SdfPath("ElementConnectivity") });
                }

                if (elementType == CGNS_ENUMV(NGON_n) || elementType == CGNS_ENUMV(NFACE_n))
                {
                    const ScopedPush forOffsets(cgnsPath, "ElementStartOffset");
                    OmniCaeCgnsFieldArray arrayT =
                        OmniCaeCgnsFieldArray::Define(stage, MakeChildPath(datasetT, "ElementStartOffset"));
                    arrayT.GetPrim().GetSpecializes().SetSpecializes({ fieldArrayClass.GetPath() });
                    arrayT.CreateFieldAssociationAttr().Set(OmniCaeTokens->none);
                    arrayT.CreateFieldPathAttr().Set(TfStringJoin(cgnsPath, "/"));
                    sidsAPI.GetElementStartOffsetRel().SetTargets({ SdfPath("ElementStartOffset") });
                }

                if (elementType == CGNS_ENUMV(NGON_n))
                {
                    ngonPaths.push_back(datasetT.GetPath());
                }
                else if (elementType == CGNS_ENUMV(NFACE_n))
                {
                    nfacePaths.push_back(datasetT.GetPath());
                }
                sections.push_back(datasetT);
                // printf("finished zone=%d, section=%d\n", zone, section);
                // layer->Export("/tmp/cgnslayer.usda");
            } // for (section; ; )

            // for each nface-n, point to other ngon-ns in the zone.
            for (const auto& nfacePath : nfacePaths)
            {
                OmniCaeSidsUnstructuredAPI sidsAPI(stage->GetPrimAtPath(nfacePath));
                sidsAPI.CreateNgonsRel().SetTargets(ngonPaths);
            }

            std::map<std::string, std::vector<std::string>> flowSolutionPointers;
            std::set<std::string> allFlowSolutionPointers;
            if (nsteps > 0)
            {
                // read flow solution ptrs.
                for (int zid = 1; cg_goto(cgFile, base, "Zone_t", zone, "ZoneIterativeData_t", zid, NULL) == CG_OK; ++zid)
                {
                    int narrays;
                    call_safe(cg_narrays(&narrays), "read narrays");
                    for (int na = 1; na <= narrays; ++na)
                    {
                        CGNS_ENUMT(DataType_t) dtype;
                        cgsize_t dims[12];
                        int rank;
                        call_safe(cg_array_info(na, name, &dtype, &rank, dims), "read array info");
                        // if name matches `FlowSolution.*Pointers`, then process it
                        if (TfStringStartsWith(name, "FlowSolution") && TfStringEndsWith(name, "Pointers"))
                        {
                            std::vector<char> buffer(dims[0] * dims[1]);
                            call_safe(cg_array_read_as(na, CGNS_ENUMV(Character), buffer.data()),
                                      "read flow solution pointers");
                            std::vector<std::string> fsps;
                            for (int d = 0; d < dims[1]; ++d)
                            {
                                std::vector<char> temp;
                                std::copy_n(std::next(buffer.begin(), d * dims[0]), dims[0], std::back_inserter(temp));
                                temp.push_back(0);
                                // FSPs are often padded.
                                const std::string tempStr = pxr::TfStringTrim(temp.data());
                                fsps.push_back(tempStr);
                            }

                            if (!fsps.empty())
                            {
                                std::copy(fsps.begin(), fsps.end(),
                                          std::inserter(allFlowSolutionPointers, allFlowSolutionPointers.end()));
                                flowSolutionPointers[fsps.front()] = fsps;
                            }
                        }
                    }
                }
            }

            // Read FlowSolutions.
            int nsols;
            call_safe(cg_nsols(cgFile, base, zone, &nsols), "read num solutions");
            for (int sol = 1; sol <= nsols; ++sol)
            {
                CGNS_ENUMT(GridLocation_t) location;
                call_safe(cg_sol_info(cgFile, base, zone, sol, name, &location), "read sol info");

                const std::string fsName = pxr::TfStringTrim(std::string(name));
                const auto fspIter = flowSolutionPointers.find(fsName);
                if (fspIter == flowSolutionPointers.end() &&
                    allFlowSolutionPointers.find(fsName) != allFlowSolutionPointers.end())
                {
                    // for FlowSolutions that are part of a iterative collection, we skip all but the
                    // first. When adding the first, we add all the iteratived paths.
                    continue;
                }

                // const ScopedPush forSol(cgnsPath, name);
                auto fsPrim = stage->DefinePrim(MakeChildPath(zonePrim, name), CGNSFileFormatTokens->FlowSolution_t);

                int nfields;
                call_safe(cg_nfields(cgFile, base, zone, sol, &nfields), "read nfields");
                for (int field = 1; field <= nfields; ++field)
                {
                    CGNS_ENUMT(DataType_t) datatype;
                    call_safe(cg_field_info(cgFile, base, zone, sol, field, &datatype, name), "read field info");

                    const std::string fieldName(name);

                    // const ScopedPush forField(cgnsPath, name);
                    OmniCaeCgnsFieldArray arrayT = OmniCaeCgnsFieldArray::Define(stage, MakeChildPath(fsPrim, name));
                    arrayT.GetPrim().GetSpecializes().SetSpecializes({ fieldArrayClass.GetPath() });
                    arrayT.CreateFieldAssociationAttr().Set(GetGridLocation(location));

                    if (fspIter != flowSolutionPointers.end())
                    {
                        // add paths for all time values.
                        size_t time_idx = 0;
                        for (const auto& ifsName : fspIter->second)
                        {
                            const ScopedPush forSol(cgnsPath, ifsName);
                            const ScopedPush forField(cgnsPath, fieldName);
                            arrayT.CreateFieldPathAttr().Set(
                                TfStringJoin(cgnsPath, "/"), pxr::UsdTimeCode(timeValues.at(time_idx)));
                            if (time_idx == 0)
                            {
                                // adding first value as the value for default time code until we
                                // start supporting time fully.
                                arrayT.CreateFieldPathAttr().Set(TfStringJoin(cgnsPath, "/"));
                            }
                            ++time_idx;
                        }
                    }
                    else
                    {
                        const ScopedPush forSol(cgnsPath, fsName);
                        const ScopedPush forField(cgnsPath, fieldName);
                        arrayT.CreateFieldPathAttr().Set(TfStringJoin(cgnsPath, "/"));
                    }

                    // FIXME: make unique in case name is repeated
                    auto tfFieldName = TfToken("field:" + TfMakeValidIdentifier(name));
                    for (auto& section : sections)
                    {
                        // TODO: we need to add API to OmniCaeDataSet to fields like UsdVolVolume.
                        // section.CreateFieldRelationship(fieldName, arrayT.GetPath());
                        section.GetPrim().CreateRelationship(tfFieldName).SetTargets({ arrayT.GetPath() });
                    }

                    if (location == CGNS_ENUMV(Vertex))
                    {
                        gridCoordinates.GetPrim().CreateRelationship(tfFieldName).SetTargets({ arrayT.GetPath() });
                    }
                }
            }
        }
    }

    return layer;
}

SdfLayerRefPtr ReadCGNS(const std::string& fname, const pxr::SdfLayer::FileFormatArguments& args)
{
    int cgFile;
    if (cg_open(fname.c_str(), CGIO_MODE_READ, &cgFile) == CG_OK)
    {
        // printf("opened: %s", fname.c_str());
        try
        {
            auto result = ReadCGNS(cgFile, fname, args);
            cg_close(cgFile);
            // result->Export("/tmp/cgnslayer.usda");
            return result;
        }
        catch (const std::exception&)
        {
            cg_close(cgFile);
        }
    }
    else
    {
        TF_ERROR(
            KIT_CAE_CGNS_FILEFORMAT, "Failed to open CGNS file: %s \n CGNS Error %s\n", fname.c_str(), cg_get_error());
    }
    return {};
}

} // namespace detail


CGNSFileFormat::CGNSFileFormat()
    : SdfFileFormat(CGNSFileFormatTokens->Id,
                    CGNSFileFormatTokens->Version,
                    CGNSFileFormatTokens->Target,
                    CGNSFileFormatTokens->Extension)
{
}

CGNSFileFormat::~CGNSFileFormat()
{
}

bool CGNSFileFormat::CanRead(const std::string& filePath) const
{
    // FIXME: don't think this ever gets called!
    std::string ext = TfGetExtension(filePath);
    if (ext != CGNSFileFormatTokens->Extension)
    {
        return false;
    }

    // FIXME: is filePath resolved path? otherwise the following check will fail!
    int cgioFile;
    if (cgio_open_file(filePath.c_str(), CGIO_MODE_READ, CG_FILE_NONE, &cgioFile) == CG_OK)
    {
        cgio_close_file(cgioFile);
        return true;
    }
    return false;
}

bool CGNSFileFormat::Read(SdfLayer* layer, const std::string& resolvedPath, bool metadataOnly) const
{
    PXR_NAMESPACE_USING_DIRECTIVE
    if (!TF_VERIFY(layer))
    {
        return false;
    }

    const FileFormatArguments& args = layer->GetFileFormatArguments();
    SdfLayerRefPtr cgnsLayer = detail::ReadCGNS(resolvedPath, args);
    if (!cgnsLayer)
    {
        return false;
    }

    layer->TransferContent(cgnsLayer);
    return true;
}

bool CGNSFileFormat::WriteToString(const SdfLayer& layer, std::string* str, const std::string& comment) const
{
    // this doesn't support writing
    return false;
}

bool CGNSFileFormat::WriteToStream(const SdfSpecHandle& spec, std::ostream& out, size_t indent) const
{
    // this doesn't support writing
    return false;
}

// clang-format off
TF_DEFINE_PUBLIC_TOKENS(
	CGNSFileFormatTokens,
	((Id, "cgnsFileFormat"))
	((Version, "1.0"))
	((Target, "usd"))
	((Extension, "cgns"))
    ((Base_t, "CGNSBase"))
	((Zone_t, "CGNSZone"))
	((FlowSolution_t, "CGNSFlowSolution"))
);
// clang-format on

PXR_NAMESPACE_CLOSE_SCOPE
