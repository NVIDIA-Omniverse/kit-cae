// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include "OpenFoamUtils.h"

#include <carb/logging/Log.h>

#include <cctype>
#include <fstream>
#include <map>
#include <string>

namespace omni
{
namespace cae
{
namespace data
{
namespace openfoam
{

std::istream& operator>>(std::istream& is, const WS&)
{
    std::istream::sentry s(is);
    if (!s)
    {
        return is;
    }

    bool line_comment = false;
    bool block_comment = false;
    char c;
    while (is.get(c))
    {
        if (block_comment && c == '*' && is.peek() == '/')
        {
            is.get(c); // Consume the closing slash
            block_comment = false; // End of block comment
            continue;
        }
        else if (block_comment)
        {
            continue; // Inside block comment, skip characters
        }
        else if (line_comment && (c == '\n' || c == '\r'))
        {
            line_comment = false; // End of line comment
            continue;
        }
        else if (line_comment)
        {
            continue; // Inside line comment, skip characters
        }
        else if (c == '/' && is.peek() == '*')
        {
            is.get(c); // Consume the asterisk
            block_comment = true; // Start of block comment
            continue;
        }
        else if (c == '/' && is.peek() == '/')
        {
            is.get(c); // Consume the second slash
            line_comment = true; // Start of line comment
            continue;
        }
        else if (std::isspace(static_cast<unsigned char>(c)))
        {
            continue; // Skip whitespace
        }
        else
        {
            // If we reach here, we have a valid character to read
            is.unget(); // Put back the character for reading later
            return is; // Return the ifp to continue reading
        }
    }
    return is;
}

std::istream& operator>>(std::istream& is, Keyword& kw)
{
    is >> WS(); // Skip whitespace, comments, etc.

    std::istream::sentry s(is);
    if (!s)
    {
        return is; // Return if the ifp is not in a good state
    }
    // read until semicolon or space or comment
    char c;
    kw.value.clear();
    if (is.peek() == ';')
    {
        // is.unget(); // Put back the semicolon for reading later
        // CARB_LOG_ERROR("Unexpected semicolon at the start of keyword");
        // throw std::runtime_error("Unexpected semicolon at the start of keyword");
        is.get(c); // Consume the semicolon
        kw.value = ";"; // Set the keyword value to semicolon
        return is; // Return if the keyword is just a semicolon
    }
    while (is.get(c))
    {
        if (c == ';' || std::isspace(static_cast<unsigned char>(c)))
        {
            is.unget(); // Put back the character for reading later
            break; // Stop reading on semicolon or whitespace
        }
        else if (c == '/' && is.peek() == '/')
        {
            is.unget(); // Put back the slash for reading later
            break;
        }
        else if (c == '/' && is.peek() == '*')
        {
            is.unget(); // Put back the slash for reading later
            break; // Stop reading on block comment start
        }
        else if (c == '"')
        {
            // Handle string literals
            while (is.get(c) && c != '"') // Read until closing quote
            {
                if (c == '\\') // Handle escape sequences
                {
                    kw.value += c; // Append backslash
                    if (is.get(c)) // Read the next character
                    {
                        kw.value += c; // Append escaped character
                    }
                }
                else
                {
                    kw.value += c; // Append regular character
                }
            }
            break; // Stop reading after string literal
        }
        else
        {
            kw.value += c; // Append character to value
        }
    }
    return is;
}

std::istream& operator>>(std::istream& is, Integer& ival)
{
    is >> WS(); // Skip whitespace, comments, etc.

    std::istream::sentry s(is);
    if (!s)
    {
        return is; // Return if the ifp is not in a good state
    }

    char c;
    ival.value = 0;
    if (is.get(c) && std::isdigit(static_cast<unsigned char>(c)))
    {
        do
        {
            ival.value = ival.value * 10 + (c - '0'); // Convert character to integer
        } while (is.get(c) && std::isdigit(static_cast<unsigned char>(c)));
        is.unget(); // Put back the last character for reading later
        return is; // Return if the integer is successfully read
    }
    else
    {
        is.unget(); // Put back the character for reading later
        CARB_LOG_ERROR("Expected an integer, but found '%c'", c);
        throw std::runtime_error("Expected integer not found");
    }
}

std::istream& operator>>(std::istream& is, const ExpectedChar& ex)
{
    is >> WS(); // Skip whitespace, comments, etc.

    std::istream::sentry s(is);
    if (!s)
    {
        return is; // Return if the ifp is not in a good state
    }

    char c;
    if (is.get(c) && c == ex.expected)
    {
        return is; // Return if the expected character is found
    }
    else
    {
        is.unget(); // Put back the character for reading later
        CARB_LOG_ERROR("Expected '%c', but found '%c'", ex.expected, c);
        throw std::runtime_error("Expected character not found");
    }
}

std::istream& operator>>(std::istream& is, const FF& ff)
{
    is >> WS(); // Skip whitespace, comments, etc.

    std::istream::sentry s(is);
    if (!s)
    {
        return is; // Return if the ifp is not in a good state
    }

    Keyword kw;
    do
    {
        is >> kw;
    } while (kw.str() != ff.value && is.good());
    return is; // Return if the expected keyword is found
}

std::map<std::string, std::string> readHeader(std::istream& ifp)
{
    std::map<std::string, std::string> header;
    std::ifstream::sentry s(ifp);
    if (!s)
    {
        CARB_LOG_ERROR("Failed to read from input stream");
        throw std::runtime_error("Failed to read from input stream");
    }

    // Read the FoamFile header
    Keyword kw;
    ifp >> kw; // Read the first keyword
    if (kw.str() != "FoamFile")
    {
        CARB_LOG_ERROR("Expected 'FoamFile', but found '%s'", kw.str().c_str());
        throw std::runtime_error("Expected 'FoamFile'");
    }
    ifp >> ExpectedChar('{'); // Expect an opening brace after 'FoamFile'}')

    Keyword key, value;
    while (ifp)
    {
        ifp >> key; // Read the next keyword
        if (key.str() == "}")
        {
            break; // End of header
        }

        ifp >> value; // Read the value associated with the keyword
        ifp >> ExpectedChar(';'); // Expect a semicolon after the value
        // Process the key-value pair as needed
        header[key.str()] = value.str(); // Store the key-value pair
    }
    return header;
}

bool isBinary(const std::map<std::string, std::string>& header)
{
    auto it = header.find("format");
    if (it != header.end())
    {
        const std::string& format = it->second;
        if (format == "binary")
        {
            return true; // The file is in binary format
        }
        else if (format == "ascii")
        {
            return false; // The file is in ASCII format
        }
    }
    CARB_LOG_ERROR("Format not found in header");
    throw std::runtime_error("Format not found in header");
}

ElementType getScalarType(const std::map<std::string, std::string>& header)
{
    auto it = header.find("arch");
    if (it == header.end())
    {
        CARB_LOG_ERROR("Element type not found in header");
        throw std::runtime_error("Element type not found in header");
    }

    std::string arch = it->second;
    // remove all whitespace characters from the arch string
    arch.erase(std::remove_if(arch.begin(), arch.end(), ::isspace), arch.end());
    if (arch.find("scalar=64") != std::string::npos)
    {
        return ElementType::float64; // 64-bit scalar
    }
    else if (arch.find("scalar=32") != std::string::npos)
    {
        return ElementType::float32; // 32-bit scalar
    }
    else
    {
        CARB_LOG_ERROR("Unsupported element type in header: %s", arch.c_str());
        throw std::runtime_error("Unsupported element type in header");
    }
}

ElementType getLabelType(const std::map<std::string, std::string>& header)
{
    auto it = header.find("arch");
    if (it == header.end())
    {
        CARB_LOG_ERROR("Element type not found in header");
        throw std::runtime_error("Element type not found in header");
    }

    std::string arch = it->second;
    // remove all whitespace characters from the arch string
    arch.erase(std::remove_if(arch.begin(), arch.end(), ::isspace), arch.end());
    if (arch.find("label=64") != std::string::npos)
    {
        return ElementType::int64; // 64-bit scalar
    }
    else if (arch.find("label=32") != std::string::npos)
    {
        return ElementType::int32; // 32-bit scalar
    }
    else
    {
        CARB_LOG_ERROR("Unsupported element type in header: %s", arch.c_str());
        throw std::runtime_error("Unsupported element type in header");
    }
}

template <typename T>
carb::ObjectPtr<IFieldArray> readScalar(std::ifstream& ifp,
                                        omni::cae::data::IFieldArrayUtils* utils,
                                        bool isBinaryFile,
                                        ElementType etype)
{
    Integer count;
    ifp >> count; // Read the count of owner or neighbor indices
    if (count.data() <= 0)
    {
        CARB_LOG_ERROR("Scalar count is not valid: %" PRIu64, count.data());
        throw std::runtime_error("Scalar count is not valid");
    }

    CARB_LOG_INFO("Scalar count: %" PRIu64, count.data());
    ifp >> ExpectedChar('('); // Expect an opening parenthesis for the list
    auto farray = utils->createMutableFieldArray(etype, { count.data() }, -1, Order::c);
    if (isBinaryFile)
    {
        // Read binary data
        char* ptr = farray->getMutableData<char>();
        ifp.read(ptr, count.data() * getElementSize(etype));
    }
    else
    {
        // Read ASCII data
        T* ptr = farray->getMutableData<T>();
        for (std::uint64_t i = 0; i < count.data(); ++i)
        {
            T val;
            ifp >> val; // Read the value
            *ptr++ = val; // Store the value in the field array
        }
    }
    ifp >> ExpectedChar(')'); // Expect a closing parenthesis for the list
    return carb::borrowObject<IFieldArray>(farray.get());
}

template <typename T>
carb::ObjectPtr<IFieldArray> readVector(std::ifstream& ifp,
                                        omni::cae::data::IFieldArrayUtils* utils,
                                        bool isBinaryFile,
                                        ElementType etype)
{
    Integer count;
    ifp >> count; // Read the count of owner or neighbor indices
    if (count.data() <= 0)
    {
        CARB_LOG_ERROR("Vector count is not valid: %" PRIu64, count.data());
        throw std::runtime_error("Vector count is not valid");
    }

    CARB_LOG_INFO("Vector count: %" PRIu64, count.data());
    ifp >> ExpectedChar('('); // Expect an opening parenthesis for the list
    auto farray = utils->createMutableFieldArray(etype, { count.data(), 3 }, -1, Order::c);
    if (isBinaryFile)
    {
        // Read binary data
        char* ptr = farray->getMutableData<char>();
        ifp.read(ptr, count.data() * 3 * getElementSize(etype));
    }
    else
    {
        // Read ASCII data
        T* ptr = farray->getMutableData<T>();
        for (std::uint64_t i = 0; i < count.data(); ++i)
        {
            ifp >> ExpectedChar('('); // Expect an opening parenthesis for each point
            T x, y, z;
            ifp >> x >> y >> z; // Read the coordinates
            ifp >> ExpectedChar(')'); // Expect a closing parenthesis for each point
            *ptr++ = x;
            *ptr++ = y;
            *ptr++ = z; // Store the coordinates in the field array
        }
    }
    ifp >> ExpectedChar(')'); // Expect a closing parenthesis for the list
    return carb::borrowObject<IFieldArray>(farray.get());
}

carb::ObjectPtr<IFieldArray> readPoints(const std::string& path, omni::cae::data::IFieldArrayUtils* utils)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    const bool isBinaryFile = isBinary(header);
    auto etype = isBinaryFile ? getScalarType(header) : ElementType::float32; // Default to float32 for ASCII files

    return etype == ElementType::float32 ? readVector<float>(ifp, utils, isBinaryFile, etype) :
                                           readVector<double>(ifp, utils, isBinaryFile, etype);
}

carb::ObjectPtr<IFieldArray> readOwnerOrNeighbour(const std::string& path, omni::cae::data::IFieldArrayUtils* utils)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    const bool isBinaryFile = isBinary(header);
    auto etype = isBinaryFile ? getLabelType(header) : ElementType::int32; // Default to int32 for ASCII files

    return etype == ElementType::int32 ? readScalar<int32_t>(ifp, utils, isBinaryFile, etype) :
                                         readScalar<int64_t>(ifp, utils, isBinaryFile, etype);
}

carb::ObjectPtr<IFieldArray> readFaceList(const std::string& path, omni::cae::data::IFieldArrayUtils* utils, bool offsets)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    const bool isBinaryFile = isBinary(header);
    if (isBinaryFile)
    {
        CARB_LOG_ERROR("Binary faceList files are not supported yet");
        throw std::runtime_error("Binary faceList files are not supported yet");
    }

    auto etype = ElementType::int32; // Default to int32 for ASCII files
    Integer count;
    ifp >> count; // Read the face count
    if (count.data() <= 0)
    {
        CARB_LOG_ERROR("Face count is not valid: %" PRIu64, count.data());
        throw std::runtime_error("Face count is not valid");
    }
    CARB_LOG_INFO("Face count: %" PRIu64, count.data());
    ifp >> ExpectedChar('('); // Expect an opening parenthesis for the face list

    std::vector<int32_t> v_offsets;
    std::vector<int32_t> v_ids;

    v_offsets.reserve(count.data() + 1); // Reserve space for offsets
    v_ids.reserve(count.data() * 3); // Reserve space for face indices (3 per face)
    for (std::uint64_t i = 0; i < count.data(); ++i)
    {
        int32_t numIndices;
        ifp >> numIndices; // Read the number of indices for the face
        if (numIndices < 3)
        {
            CARB_LOG_ERROR("Face must have at least 3 indices, but found %d", numIndices);
            throw std::runtime_error("Face must have at least 3 indices");
        }
        v_offsets.push_back(static_cast<int32_t>(v_ids.size())); // Store the current offset

        ifp >> ExpectedChar('('); // Expect an opening parenthesis for each face
        for (int32_t j = 0; j < numIndices; ++j)
        {
            int32_t index;
            ifp >> index; // Read the index
            v_ids.push_back(index); // Store the index in the list
        }
        ifp >> ExpectedChar(')'); // Expect a closing parenthesis for each face
    }
    ifp >> ExpectedChar(')'); // Expect a closing parenthesis for the face list
    v_offsets.push_back(static_cast<int32_t>(v_ids.size())); // Final offset for the last face
    ifp.close(); // Close the file after reading

    if (offsets)
    {
        // Create a field array for offsets
        auto farrayOffsets =
            utils->createMutableFieldArray(etype, { static_cast<std::uint64_t>(v_offsets.size()) }, -1, Order::c);
        int32_t* ptrOffsets = farrayOffsets->getMutableData<int32_t>();
        std::copy(v_offsets.begin(), v_offsets.end(), ptrOffsets);
        return carb::borrowObject<IFieldArray>(farrayOffsets.get());
    }
    else
    {
        // Create a field array for face indices
        auto farrayFaces =
            utils->createMutableFieldArray(etype, { static_cast<std::uint64_t>(v_ids.size()) }, -1, Order::c);
        int32_t* ptrFaces = farrayFaces->getMutableData<int32_t>();
        std::copy(v_ids.begin(), v_ids.end(), ptrFaces);
        return carb::borrowObject<IFieldArray>(farrayFaces.get());
    }
}

carb::ObjectPtr<IFieldArray> readFaceCompactList(const std::string& path,
                                                 omni::cae::data::IFieldArrayUtils* utils,
                                                 bool offsets)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    const bool isBinaryFile = isBinary(header);
    if (!isBinaryFile)
    {
        CARB_LOG_ERROR("ASCII faceCompactList files are not supported yet");
        throw std::runtime_error("ASCII faceCompactList files are not supported yet");
    }

    auto etype = getLabelType(header); // Get the label type from the header

    auto farray = etype == ElementType::int32 ? readScalar<int32_t>(ifp, utils, isBinaryFile, etype) :
                                                readScalar<int64_t>(ifp, utils, isBinaryFile, etype);
    if (offsets)
    {
        // Return offsets if requested
        return carb::borrowObject<IFieldArray>(farray.get());
    }
    else
    {
        // Return face indices if offsets are not requested
        farray = nullptr; // Clear the previous field array
        farray = etype == ElementType::int32 ? readScalar<int32_t>(ifp, utils, isBinaryFile, etype) :
                                               readScalar<int64_t>(ifp, utils, isBinaryFile, etype);
        return carb::borrowObject<IFieldArray>(farray.get());
    }
}

carb::ObjectPtr<IFieldArray> readFaces(const std::string& path, omni::cae::data::IFieldArrayUtils* utils, bool offsets)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    ifp.close(); // Close the file after reading the header

    if (header.find("class") != header.end())
    {
        if (header.at("class") == "faceList")
        {
            return readFaceList(path, utils, offsets);
        }
        else if (header.at("class") == "faceCompactList")
        {
            return readFaceCompactList(path, utils, offsets);
        }
        else
        {
            CARB_LOG_ERROR(
                "Expected 'class' to be 'faceList' or 'faceCompactList', but found '%s'", header.at("class").c_str());
            throw std::runtime_error("Expected 'class' to be 'faceList' or 'faceCompactList'");
        }
    }
    else
    {
        CARB_LOG_ERROR("Header does not contain 'class' key");
        throw std::runtime_error("Header does not contain 'class' key");
    }
}

carb::ObjectPtr<IFieldArray> readInternalField(const std::string& path, omni::cae::data::IFieldArrayUtils* utils)
{
    std::ifstream ifp(path.c_str(), std::ios::in | std::ios::binary);
    if (!ifp)
    {
        CARB_LOG_ERROR("Failed to open file '%s'", path.c_str());
        throw std::runtime_error("Failed to open file");
    }

    const auto header = readHeader(ifp);
    const bool isBinaryFile = isBinary(header);
    auto etype = isBinaryFile ? getScalarType(header) : ElementType::float32; // Default to float32 for ASCII files

    ifp >> FF("internalField"); // fast-forward past the 'internalField' keywor

    Keyword kw;
    ifp >> kw; // Read the uniform/non-uniform keyword
    if (kw.str() != "nonuniform")
    {
        CARB_LOG_ERROR("Expected 'nonuniform', but found '%s'", kw.str().c_str());
        throw std::runtime_error("Expected 'nonuniform'");
    }
    ifp >> kw;
    if (kw.str().find("scalar") != std::string::npos)
    {
        return etype == ElementType::float32 ? readScalar<float>(ifp, utils, isBinaryFile, etype) :
                                               readScalar<double>(ifp, utils, isBinaryFile, etype);
    }
    else if (kw.str().find("vector") != std::string::npos)
    {
        return etype == ElementType::float32 ? readVector<float>(ifp, utils, isBinaryFile, etype) :
                                               readVector<double>(ifp, utils, isBinaryFile, etype);
    }
    else
    {
        CARB_LOG_ERROR("Unsupported element type in internal field: %s", kw.str().c_str());
        throw std::runtime_error("Unsupported element type in internal field");
    }
}

} // namespace openfoam
} // namespace data
} // namespace cae
} // namespace omni
