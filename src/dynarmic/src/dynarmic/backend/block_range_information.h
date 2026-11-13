// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <vector>

#include <boost/icl/interval_set.hpp>
#include "common/container/unordered_set.h"

#include "dynarmic/ir/location_descriptor.h"

namespace Dynarmic::Backend {

template<typename P>
class BlockRangeInformation {
public:
    void AddRange(boost::icl::discrete_interval<P> range, IR::LocationDescriptor location);
    void ClearCache();
    ::Common::unordered_set<IR::LocationDescriptor> InvalidateRanges(const boost::icl::interval_set<P>& invalidated);

    /// For caches whose blocks also leave without InvalidateRanges or ClearCache (ring
    /// eviction, a block translated again after a fastmem fault): once the list has grown by a
    /// quarter, drops the ranges of blocks live(location) no longer knows and the repeats of
    /// the others.
    /// Without it the list grew by 24 bytes for every block ever translated.
    template<typename Live>
    void CompactIfGrown(Live&& live) {
        if (ranges.size() < compact_at) {
            return;
        }
        std::erase_if(ranges, [&live](const Range& range) { return !live(range.location); });
        const auto key = [](const Range& range) { return std::tuple{range.location.Value(), range.first, range.last}; };
        std::sort(ranges.begin(), ranges.end(), [&key](const Range& a, const Range& b) { return key(a) < key(b); });
        ranges.erase(std::unique(ranges.begin(), ranges.end(), [&key](const Range& a, const Range& b) { return key(a) == key(b); }),
                     ranges.end());
        // The list may grow by a quarter until the next time, and is given exactly that room.
        compact_at = ranges.size() + (std::max)(ranges.size() / 4, MinCompactSize);
        if (ranges.capacity() > compact_at + compact_at / 2) {
            ranges.shrink_to_fit();
        }
        ranges.reserve(compact_at);
    }

    /// Heap bytes of the list.
    std::size_t Bytes() const {
        return ranges.capacity() * sizeof(Range);
    }

private:
    /// The guest code [first, last] a block was translated from. Code is invalidated rarely (when
    /// it is unmapped or reprotected), so a list scanned then costs least: an interval map of
    /// block sets held ~300 bytes per block.
    struct Range {
        P first;
        P last;
        IR::LocationDescriptor location;
    };
    static constexpr std::size_t MinCompactSize = 65536;

    std::vector<Range> ranges;
    std::size_t compact_at = MinCompactSize;
};

}  // namespace Dynarmic::Backend
