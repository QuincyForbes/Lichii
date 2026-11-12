// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <fmt/format.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <malloc/malloc.h>
#endif

#include "common/common_types.h"
#include "common/logging.h"
#include "common/malloc_census.h"

namespace Common {

#ifdef __APPLE__
namespace {

struct SizeCount {
    u64 size;
    u64 count;
};

// Filled while a zone's locks are held: nothing here may allocate.
struct Census {
    static constexpr size_t TableSize = size_t{1} << 16;
    static constexpr std::array<u64, 7> Limits{256,     1 << 10, 16 << 10, 128 << 10,
                                               1 << 20, 16 << 20, ~u64{0}};
    std::vector<SizeCount> table = std::vector<SizeCount>(TableSize);
    std::array<SizeCount, Limits.size()> buckets{};
    std::array<u64, 8> largest{};
    u64 blocks{};
    u64 bytes{};
    u64 untabled_bytes{};

    void Add(u64 size) {
        ++blocks;
        bytes += size;
        for (size_t i = 0; i < Limits.size(); ++i) {
            if (size <= Limits[i]) {
                ++buckets[i].count;
                buckets[i].size += size;
                break;
            }
        }
        if (size > largest.back()) {
            auto it = std::upper_bound(largest.begin(), largest.end(), size, std::greater<u64>{});
            std::move_backward(it, largest.end() - 1, largest.end());
            *it = size;
        }
        size_t slot = static_cast<size_t>((size * 0x9E3779B97F4A7C15ULL) >> 48) & (TableSize - 1);
        for (size_t probe = 0; probe < 64; ++probe, slot = (slot + 1) & (TableSize - 1)) {
            if (table[slot].size == size) {
                ++table[slot].count;
                return;
            }
            if (table[slot].size == 0) {
                table[slot] = {size, 1};
                return;
            }
        }
        untabled_bytes += size;
    }
};

kern_return_t ReadLocal(task_t, vm_address_t address, vm_size_t, void** local) {
    *local = reinterpret_cast<void*>(address);
    return KERN_SUCCESS;
}

void Record(task_t, void* context, unsigned, vm_range_t* ranges, unsigned count) {
    auto* const census = static_cast<Census*>(context);
    for (unsigned i = 0; i < count; ++i) {
        census->Add(ranges[i].size);
    }
}

std::string MiB(u64 bytes) {
    return fmt::format("{:.1f}", static_cast<double>(bytes) / (1 << 20));
}

} // Anonymous namespace
#endif

void LogMallocCensus([[maybe_unused]] const char* reason) {
#ifdef __APPLE__
    static const bool enabled = [] {
        const char* const env = std::getenv("EDEN_NX2_MALLOC_CENSUS");
        return env == nullptr || env[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    vm_address_t* zones = nullptr;
    unsigned zone_count = 0;
    if (malloc_get_all_zones(mach_task_self(), ReadLocal, &zones, &zone_count) != KERN_SUCCESS) {
        LOG_INFO(Common_Memory, "Memory malloc census ({}): the zones could not be listed", reason);
        return;
    }
    Census census;
    std::string zone_text;
    for (unsigned i = 0; i < zone_count; ++i) {
        auto* const zone = reinterpret_cast<malloc_zone_t*>(zones[i]);
        if (zone == nullptr || zone->introspect == nullptr ||
            zone->introspect->enumerator == nullptr || zone->introspect->force_lock == nullptr ||
            zone->introspect->force_unlock == nullptr) {
            continue;
        }
        const u64 before = census.bytes;
        const u64 blocks_before = census.blocks;
        zone->introspect->force_lock(zone);
        zone->introspect->enumerator(mach_task_self(), &census, MALLOC_PTR_IN_USE_RANGE_TYPE,
                                     zones[i], ReadLocal, Record);
        zone->introspect->force_unlock(zone);
        const char* const name = malloc_get_zone_name(zone);
        zone_text += fmt::format("{}{} {} MiB in {}", zone_text.empty() ? "" : ", ",
                                 name != nullptr ? name : "?", MiB(census.bytes - before),
                                 census.blocks - blocks_before);
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();

    constexpr std::array<const char*, Census::Limits.size()> BucketNames{
        "<=256 B", "<=1 KiB", "<=16 KiB", "<=128 KiB", "<=1 MiB", "<=16 MiB", ">16 MiB"};
    std::string bucket_text;
    for (size_t i = 0; i < BucketNames.size(); ++i) {
        bucket_text += fmt::format("{}{} {} MiB in {}", i == 0 ? "" : ", ", BucketNames[i],
                                   MiB(census.buckets[i].size), census.buckets[i].count);
    }
    std::vector<SizeCount> sizes;
    for (const SizeCount& entry : census.table) {
        if (entry.size != 0) {
            sizes.push_back(entry);
        }
    }
    const size_t top = std::min<size_t>(sizes.size(), 24);
    std::partial_sort(sizes.begin(), sizes.begin() + static_cast<std::ptrdiff_t>(top), sizes.end(),
                      [](const SizeCount& a, const SizeCount& b) {
                          return a.size * a.count > b.size * b.count;
                      });
    std::string size_text;
    for (size_t i = 0; i < top; ++i) {
        size_text += fmt::format("{}{} B x {} = {} MiB", i == 0 ? "" : ", ", sizes[i].size,
                                 sizes[i].count, MiB(sizes[i].size * sizes[i].count));
    }
    std::string largest_text;
    for (const u64 size : census.largest) {
        if (size != 0) {
            largest_text += fmt::format("{}{}", largest_text.empty() ? "" : ", ", MiB(size));
        }
    }
    LOG_INFO(Common_Memory,
             "Memory malloc census ({}): {} MiB in {} blocks, {} ms | zones: {} | by size: {} | "
             "largest MiB: {} | most bytes by block size: {} | {} MiB not tabled",
             reason, MiB(census.bytes), census.blocks, ms, zone_text, bucket_text, largest_text,
             size_text, MiB(census.untabled_bytes));
#endif
}

} // namespace Common
