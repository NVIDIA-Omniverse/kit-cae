// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.

#include <carb/logging/Log.h>

#include <omni/cae/data/IFieldArray.h>
#include <omni/cae/data/IFieldArrayUtils.h>

#include <iostream>
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


struct WS
{
};

struct Keyword
{
    std::string value;
    Keyword(const std::string& val) : value(val)
    {
    }
    Keyword() = default;
    const std::string& str() const
    {
        return value;
    }
};

struct Integer
{
    std::uint64_t value;
    Integer(std::uint64_t val) : value(val)
    {
    }
    Integer() = default;
    std::uint64_t data() const
    {
        return value;
    }
};

struct ExpectedChar
{
    char expected;
    ExpectedChar(char c) : expected(c)
    {
    }
};

struct FF
{
    std::string value;
    FF(const std::string& val) : value(val)
    {
    }
};

std::istream& operator>>(std::istream& is, const WS&);
std::istream& operator>>(std::istream& is, Keyword& keyword);
std::istream& operator>>(std::istream& is, Integer& ival);
std::istream& operator>>(std::istream& is, const ExpectedChar& expectedChar);
std::istream& operator>>(std::istream& is, const FF& ff);

carb::ObjectPtr<IFieldArray> readPoints(const std::string& path, omni::cae::data::IFieldArrayUtils* utils);
carb::ObjectPtr<IFieldArray> readOwnerOrNeighbour(const std::string& path, omni::cae::data::IFieldArrayUtils* utils);
carb::ObjectPtr<IFieldArray> readFaces(const std::string& path, omni::cae::data::IFieldArrayUtils* utils, bool offsets);
carb::ObjectPtr<IFieldArray> readInternalField(const std::string& path, omni::cae::data::IFieldArrayUtils* utils);

} // namespace openfoam
} // namespace data
} // namespace cae
} // namespace omni
