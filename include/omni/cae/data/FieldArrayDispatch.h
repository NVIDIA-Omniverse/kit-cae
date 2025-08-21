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

#include <omni/cae/data/IFieldArray.h>

namespace omni
{
namespace cae
{
namespace data
{

/// A FieldArrayTypeList for field arrays
template <typename... Types>
struct FieldArrayTypeList
{
};

/// Dispatcher to iterate over a each type in a FieldArrayTypeList.
template <typename List>
struct FieldArrayDispatcher;

template <typename First, typename... Rest>
struct FieldArrayDispatcher<FieldArrayTypeList<First, Rest...>>
{
    template <typename Func, typename Array, typename... Args>
    static bool dispatch(Func&& func, Array* array, Args&&... args)
    {
        if (array && array->getElementType() == ElementTypeTraits<First>::get())
        {
            func.template operator()<First>(array, std::forward<Args>(args)...);
            return true;
        }
        else
        {
            return FieldArrayDispatcher<FieldArrayTypeList<Rest...>>::dispatch(
                std::forward<Func>(func), array, std::forward<Args>(args)...);
        }
    }
};

// specialization to terminate template recursion
template <>
struct FieldArrayDispatcher<FieldArrayTypeList<>>
{
    template <typename Func, typename Array, typename... Args>
    static bool dispatch(Func&&, Array*, Args&&...)
    {
        return false;
    }
};

/// Dispatcher to iterate over a each type in a FieldArrayTypeList for 2 arrays.
template <typename List1, typename List2>
struct FieldArrayDispatcher2;

template <typename First1, typename... Rest1, typename First2, typename... Rest2>
struct FieldArrayDispatcher2<FieldArrayTypeList<First1, Rest1...>, FieldArrayTypeList<First2, Rest2...>>
{
    template <typename Func, typename Array1, typename Array2, typename... Args>
    static bool dispatch(Func&& func, Array1* array1, Array2* array2, Args&&... args)
    {
        if (array1 && array1->getElementType() == ElementTypeTraits<First1>::get() && array2 &&
            array2->getElementType() == ElementTypeTraits<First2>::get())
        {
            func.template operator()<First1, First2>(array1, array2, std::forward<Args>(args)...);
            return true;
        }
        else if (FieldArrayDispatcher2<FieldArrayTypeList<First1>, FieldArrayTypeList<Rest2...>>::dispatch(
                     std::forward<Func>(func), array1, array2, std::forward<Args>(args)...))
        {
            return true;
        }
        return FieldArrayDispatcher2<FieldArrayTypeList<Rest1...>, FieldArrayTypeList<First2, Rest2...>>::dispatch(
            std::forward<Func>(func), array1, array2, std::forward<Args>(args)...);
    }
};

// specialization to terminate template recursion
template <typename List2>
struct FieldArrayDispatcher2<FieldArrayTypeList<>, List2>
{
    template <typename Func, typename Array1, typename Array2, typename... Args>
    static bool dispatch(Func&&, Array1*, Array2*, Args&&...)
    {
        return false;
    }
};

template <typename List1>
struct FieldArrayDispatcher2<List1, FieldArrayTypeList<>>
{
    template <typename Func, typename Array1, typename Array2, typename... Args>
    static bool dispatch(Func&&, Array1*, Array2*, Args&&...)
    {
        return false;
    }
};

using FieldArrayTypes = FieldArrayTypeList<int32_t, int64_t, uint32_t, uint64_t, float, double>;
using FieldArrayIntegralTypes = FieldArrayTypeList<int32_t, int64_t, uint32_t, uint64_t>;
using FieldArrayRealTypes = FieldArrayTypeList<float, double>;

} // namespace data
} // namespace cae
} // namespace omni
