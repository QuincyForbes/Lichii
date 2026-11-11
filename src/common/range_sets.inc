// SPDX-FileCopyrightText: 2024 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/range_sets.h"

namespace Common {

// Sorted, disjoint, non-touching ranges in one array. The buffer cache looks these up tens of
// thousands of times a frame and most lookups miss; sets stay small (Ocarina of Time: ~200 ranges
// at a subtract, 763 at most), so moving a few hundred bytes on an insert costs less than tree
// nodes from a locked pool.
template <typename AddressType>
struct RangeSet<AddressType>::RangeSetImpl {
    struct Range {
        AddressType lower;
        AddressType upper;
    };
    using Iterator = typename std::vector<Range>::iterator;
    using ConstIterator = typename std::vector<Range>::const_iterator;

    /// The first range ending after `address`, the first that could overlap what starts there.
    ConstIterator FirstEndingAfter(AddressType address) const {
        return std::upper_bound(
            m_ranges.begin(), m_ranges.end(), address,
            [](AddressType value, const Range& range) { return value < range.upper; });
    }

    void Add(AddressType base_address, size_t size) {
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (end_address <= base_address) {
            return;
        }
        // Ranges that touch the new one join it: the first ending at or after its start...
        auto first = std::lower_bound(
            m_ranges.begin(), m_ranges.end(), base_address,
            [](const Range& range, AddressType value) { return range.upper < value; });
        if (first == m_ranges.end()) {
            m_ranges.push_back(Range{base_address, end_address});
            return;
        }
        // ...up to the first starting after its end.
        const auto last = std::upper_bound(
            first, m_ranges.end(), end_address,
            [](AddressType value, const Range& range) { return value < range.lower; });
        if (first == last) {
            m_ranges.insert(first, Range{base_address, end_address});
            return;
        }
        first->lower = (std::min)(first->lower, base_address);
        first->upper = (std::max)((last - 1)->upper, end_address);
        m_ranges.erase(first + 1, last);
    }

    void Subtract(AddressType base_address, size_t size) {
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (end_address <= base_address) {
            return;
        }
        const auto first = m_ranges.begin() + (FirstEndingAfter(base_address) - m_ranges.cbegin());
        const auto last = std::lower_bound(
            first, m_ranges.end(), end_address,
            [](const Range& range, AddressType value) { return range.lower < value; });
        if (first == last) {
            return;
        }
        const AddressType left_lower = first->lower;
        const AddressType right_upper = (last - 1)->upper;
        const bool keep_left = left_lower < base_address;
        const bool keep_right = end_address < right_upper;
        if (keep_left && keep_right && last - first == 1) {
            // Cut out of the middle of one range: it becomes two.
            first->upper = base_address;
            m_ranges.insert(last, Range{end_address, right_upper});
            return;
        }
        Iterator erase_from = first;
        Iterator erase_to = last;
        if (keep_left) {
            first->upper = base_address;
            ++erase_from;
        }
        if (keep_right) {
            --erase_to;
            erase_to->lower = end_address;
        }
        m_ranges.erase(erase_from, erase_to);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (const Range& range : m_ranges) {
            func(range.lower, range.upper);
        }
    }

    template <typename Func>
    void ForEachInRange(AddressType base_addr, size_t size, Func&& func) const {
        const AddressType start_address = base_addr;
        const AddressType end_address = start_address + static_cast<AddressType>(size);
        if (end_address <= start_address) {
            return;
        }
        for (auto it = FirstEndingAfter(start_address);
             it != m_ranges.end() && it->lower < end_address; ++it) {
            func((std::max)(it->lower, start_address), (std::min)(it->upper, end_address));
        }
    }

    std::vector<Range> m_ranges;
};

template <typename AddressType>
struct OverlapRangeSet<AddressType>::OverlapRangeSetImpl {
    struct Range {
        AddressType lower;
        AddressType upper;
        s32 count;
    };
    using Iterator = typename std::vector<Range>::iterator;
    using ConstIterator = typename std::vector<Range>::const_iterator;

    OverlapRangeSetImpl() = default;
    ~OverlapRangeSetImpl() = default;

    ConstIterator FirstEndingAfter(AddressType address) const {
        return std::upper_bound(
            m_ranges.begin(), m_ranges.end(), address,
            [](AddressType value, const Range& range) { return value < range.upper; });
    }

    static void AppendMerged(boost::container::small_vector<Range, 8>& out, AddressType lower,
                             AddressType upper, s32 count) {
        if (lower >= upper || count <= 0) {
            return;
        }
        if (!out.empty() && out.back().upper == lower && out.back().count == count) {
            out.back().upper = upper;
        } else {
            out.push_back(Range{lower, upper, count});
        }
    }

    void ReplaceSpan(Iterator first, Iterator last,
                     const boost::container::small_vector<Range, 8>& replacement) {
        const size_t old_count = static_cast<size_t>(last - first);
        if (replacement.size() <= old_count) {
            const auto mid = std::copy(replacement.begin(), replacement.end(), first);
            m_ranges.erase(mid, last);
            return;
        }
        const size_t offset = static_cast<size_t>(first - m_ranges.begin());
        std::copy(replacement.begin(), replacement.begin() + old_count, first);
        m_ranges.insert(m_ranges.begin() + offset + old_count, replacement.begin() + old_count,
                        replacement.end());
    }

    void Add(AddressType base_address, size_t size) {
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (end_address <= base_address) {
            return;
        }
        auto first = std::lower_bound(
            m_ranges.begin(), m_ranges.end(), base_address,
            [](const Range& range, AddressType value) { return range.upper < value; });
        if (first == m_ranges.end()) {
            m_ranges.push_back(Range{base_address, end_address, 1});
            return;
        }
        const auto last = std::upper_bound(
            first, m_ranges.end(), end_address,
            [](AddressType value, const Range& range) { return value < range.lower; });
        if (first == last) {
            m_ranges.insert(first, Range{base_address, end_address, 1});
            return;
        }
        boost::container::small_vector<Range, 8> out;
        if (first->lower < base_address) {
            AppendMerged(out, first->lower, base_address, first->count);
        }
        AddressType cursor = base_address;
        for (auto it = first; it != last; ++it) {
            const AddressType seg_lower = (std::max)(it->lower, base_address);
            const AddressType seg_upper = (std::min)(it->upper, end_address);
            if (cursor < seg_lower) {
                AppendMerged(out, cursor, seg_lower, 1);
                cursor = seg_lower;
            }
            if (seg_lower < seg_upper) {
                AppendMerged(out, seg_lower, seg_upper, it->count + 1);
                cursor = seg_upper;
            }
            if (it->upper > end_address) {
                if (cursor < end_address) {
                    AppendMerged(out, cursor, end_address, 1);
                    cursor = end_address;
                }
                AppendMerged(out, (std::max)(it->lower, end_address), it->upper, it->count);
            }
        }
        if (cursor < end_address) {
            AppendMerged(out, cursor, end_address, 1);
        }
        ReplaceSpan(first, last, out);
    }

    template <bool has_on_delete, typename Func>
    void Subtract(AddressType base_address, size_t size, s32 amount,
                  [[maybe_unused]] Func&& on_delete) {
        const AddressType end_address = base_address + static_cast<AddressType>(size);
        if (m_ranges.empty() || end_address <= base_address || amount <= 0) {
            return;
        }
        auto first = std::lower_bound(
            m_ranges.begin(), m_ranges.end(), base_address,
            [](const Range& range, AddressType value) { return range.upper < value; });
        const auto last = std::upper_bound(
            first, m_ranges.end(), end_address,
            [](AddressType value, const Range& range) { return value < range.lower; });
        if (first == last) {
            return;
        }
        boost::container::small_vector<Range, 8> out;
        for (auto it = first; it != last; ++it) {
            if (it->upper <= base_address || it->lower >= end_address) {
                AppendMerged(out, it->lower, it->upper, it->count);
                continue;
            }
            if (it->lower < base_address) {
                AppendMerged(out, it->lower, base_address, it->count);
            }
            const AddressType seg_lower = (std::max)(it->lower, base_address);
            const AddressType seg_upper = (std::min)(it->upper, end_address);
            if (it->count > amount) {
                AppendMerged(out, seg_lower, seg_upper, it->count - amount);
            } else if constexpr (has_on_delete) {
                if (it->count == amount) {
                    on_delete(seg_lower, seg_upper);
                }
            }
            if (it->upper > end_address) {
                AppendMerged(out, end_address, it->upper, it->count);
            }
        }
        ReplaceSpan(first, last, out);
    }

    template <typename Func>
    void ForEach(Func&& func) const {
        for (const Range& range : m_ranges) {
            func(range.lower, range.upper, range.count);
        }
    }

    template <typename Func>
    void ForEachInRange(AddressType base_address, size_t size, Func&& func) const {
        const AddressType start_address = base_address;
        const AddressType end_address = start_address + static_cast<AddressType>(size);
        if (m_ranges.empty() || end_address <= start_address) {
            return;
        }
        for (auto it = FirstEndingAfter(start_address);
             it != m_ranges.end() && it->lower < end_address; ++it) {
            func((std::max)(it->lower, start_address), (std::min)(it->upper, end_address),
                 it->count);
        }
    }

    std::vector<Range> m_ranges;
};

template <typename AddressType>
RangeSet<AddressType>::RangeSet() {
    m_impl = std::make_unique<RangeSet<AddressType>::RangeSetImpl>();
}

template <typename AddressType>
RangeSet<AddressType>::~RangeSet() = default;

template <typename AddressType>
RangeSet<AddressType>::RangeSet(RangeSet&& other) {
    m_impl = std::make_unique<RangeSet<AddressType>::RangeSetImpl>();
    m_impl->m_ranges = std::move(other.m_impl->m_ranges);
    // A moved-from container is only "valid but unspecified". Callers hand ranges over by
    // moving and then keep adding to the source (BufferCache::AccumulateFlushes).
    other.m_impl->m_ranges.clear();
}

template <typename AddressType>
RangeSet<AddressType>& RangeSet<AddressType>::operator=(RangeSet&& other) {
    if (this != &other) {
        m_impl->m_ranges = std::move(other.m_impl->m_ranges);
        other.m_impl->m_ranges.clear();
    }
    return *this;
}

template <typename AddressType>
void RangeSet<AddressType>::Add(AddressType base_address, size_t size) {
    m_impl->Add(base_address, size);
}

template <typename AddressType>
void RangeSet<AddressType>::Subtract(AddressType base_address, size_t size) {
    m_impl->Subtract(base_address, size);
}

template <typename AddressType>
void RangeSet<AddressType>::Clear() {
    m_impl->m_ranges.clear();
}

template <typename AddressType>
bool RangeSet<AddressType>::Empty() const {
    return m_impl->m_ranges.empty();
}

template <typename AddressType>
bool RangeSet<AddressType>::Intersects(AddressType base_address, size_t size) const {
    const auto& ranges = m_impl->m_ranges;
    if (ranges.empty() || size == 0) {
        return false;
    }
    const AddressType end_address = base_address + static_cast<AddressType>(size);
    const auto it = m_impl->FirstEndingAfter(base_address);
    return it != ranges.end() && it->lower < end_address;
}

template <typename AddressType>
template <typename Func>
void RangeSet<AddressType>::ForEach(Func&& func) const {
    m_impl->ForEach(std::move(func));
}

template <typename AddressType>
template <typename Func>
void RangeSet<AddressType>::ForEachInRange(AddressType base_address, size_t size,
                                           Func&& func) const {
    m_impl->ForEachInRange(base_address, size, std::move(func));
}

template <typename AddressType>
OverlapRangeSet<AddressType>::OverlapRangeSet() {
    m_impl = std::make_unique<OverlapRangeSet<AddressType>::OverlapRangeSetImpl>();
}

template <typename AddressType>
OverlapRangeSet<AddressType>::~OverlapRangeSet() = default;

template <typename AddressType>
OverlapRangeSet<AddressType>::OverlapRangeSet(OverlapRangeSet&& other) {
    m_impl = std::make_unique<OverlapRangeSet<AddressType>::OverlapRangeSetImpl>();
    m_impl->m_ranges = std::move(other.m_impl->m_ranges);
    other.m_impl->m_ranges.clear();
}

template <typename AddressType>
OverlapRangeSet<AddressType>& OverlapRangeSet<AddressType>::operator=(OverlapRangeSet&& other) {
    if (this != &other) {
        m_impl->m_ranges = std::move(other.m_impl->m_ranges);
        other.m_impl->m_ranges.clear();
    }
    return *this;
}

template <typename AddressType>
void OverlapRangeSet<AddressType>::Add(AddressType base_address, size_t size) {
    m_impl->Add(base_address, size);
}

template <typename AddressType>
void OverlapRangeSet<AddressType>::Subtract(AddressType base_address, size_t size) {
    m_impl->template Subtract<false>(base_address, size, 1, [](AddressType, AddressType) {});
}

template <typename AddressType>
template <typename Func>
void OverlapRangeSet<AddressType>::Subtract(AddressType base_address, size_t size,
                                            Func&& on_delete) {
    m_impl->template Subtract<true, Func>(base_address, size, 1, std::move(on_delete));
}

template <typename AddressType>
void OverlapRangeSet<AddressType>::DeleteAll(AddressType base_address, size_t size) {
    m_impl->template Subtract<false>(base_address, size, (std::numeric_limits<s32>::max)(),
                                     [](AddressType, AddressType) {});
}

template <typename AddressType>
void OverlapRangeSet<AddressType>::Clear() {
    m_impl->m_ranges.clear();
}

template <typename AddressType>
bool OverlapRangeSet<AddressType>::Empty() const {
    return m_impl->m_ranges.empty();
}

template <typename AddressType>
template <typename Func>
void OverlapRangeSet<AddressType>::ForEach(Func&& func) const {
    m_impl->ForEach(func);
}

template <typename AddressType>
template <typename Func>
void OverlapRangeSet<AddressType>::ForEachInRange(AddressType base_address, size_t size,
                                                  Func&& func) const {
    m_impl->ForEachInRange(base_address, size, std::move(func));
}

} // namespace Common
