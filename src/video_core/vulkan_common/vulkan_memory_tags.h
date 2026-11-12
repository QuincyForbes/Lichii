// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/common_types.h"
#include "video_core/vulkan_common/vma.h"

namespace Vulkan {

/// What a VMA allocation is for. Set with MemoryTagScope around the allocator call; the
/// allocator stores it in the allocation's user data and counts bytes per tag until it is freed.
enum class MemoryTag : u8 {
    Untagged,
    GuestImage,
    RescaledImage,
    NullImage,
    TextureHelper,
    SparseImageMemory,
    BufferCache,
    StreamBuffer,
    StagingUpload,
    StagingDeviceLocal,
    StagingDownload,
    PresentFrame,
    PresentLayer,
    PresentFilter,
    Upscaler,
    FrameGeneration,
    QueryCache,
    DescriptorBuffer,
    MeshEmulation,
    MultiRangeBuffer,
    Readback,
    Diagnostics,
    TurboMode,
    Count,
};

constexpr std::array<std::string_view, static_cast<size_t>(MemoryTag::Count)> MemoryTagNames{
    "untagged",      "guest images",   "rescaled images", "null images",   "texture helpers",
    "sparse pages",  "buffer cache",   "stream buffer",   "staging upload", "staging device",
    "staging download", "present frames", "present layers", "present filters", "upscalers",   "frame generation",
    "queries",       "descriptor buffers", "mesh emulation", "multi-range buffers", "readback",
    "diagnostics",   "turbo mode",
};

namespace MemoryTags {

struct Counter {
    std::atomic<u64> bytes{};
    std::atomic<u64> count{};
    std::atomic<u64> peak{};
};

inline std::array<Counter, static_cast<size_t>(MemoryTag::Count)> counters{};
inline thread_local MemoryTag current{MemoryTag::Untagged};

/// The tag a caller set, or `fallback` when none is set.
inline MemoryTag CurrentOr(MemoryTag fallback) noexcept {
    return current == MemoryTag::Untagged ? fallback : current;
}

inline void* ToUserData(MemoryTag tag) noexcept {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(tag) + 1);
}

/// Records the current tag on a new allocation.
inline void Tag(VmaAllocator allocator, VmaAllocation allocation, u64 size) noexcept {
    if (allocation == nullptr) {
        return;
    }
    const MemoryTag tag = current;
    vmaSetAllocationUserData(allocator, allocation, ToUserData(tag));
    Counter& counter = counters[static_cast<size_t>(tag)];
    const u64 bytes = counter.bytes.fetch_add(size, std::memory_order_relaxed) + size;
    counter.count.fetch_add(1, std::memory_order_relaxed);
    u64 peak = counter.peak.load(std::memory_order_relaxed);
    while (bytes > peak &&
           !counter.peak.compare_exchange_weak(peak, bytes, std::memory_order_relaxed)) {
    }
}

/// Call before the allocation is freed.
inline void Untag(VmaAllocator allocator, VmaAllocation allocation) noexcept {
    if (allocation == nullptr || allocator == nullptr) {
        return;
    }
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(allocator, allocation, &info);
    const auto value = reinterpret_cast<uintptr_t>(info.pUserData);
    if (value == 0 || value > static_cast<uintptr_t>(MemoryTag::Count)) {
        return;
    }
    Counter& counter = counters[value - 1];
    counter.bytes.fetch_sub(info.size, std::memory_order_relaxed);
    counter.count.fetch_sub(1, std::memory_order_relaxed);
}

inline u64 Bytes(MemoryTag tag) noexcept {
    return counters[static_cast<size_t>(tag)].bytes.load(std::memory_order_relaxed);
}

/// "guest images 53.0 MiB (412, peak 80.2), stream buffer 32.1 MiB (1, peak 32.1), ..." in
/// falling order, with each tag's peak since the start; tags that never held memory are left out.
inline std::string Report() {
    std::array<size_t, static_cast<size_t>(MemoryTag::Count)> order{};
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::array<u64, static_cast<size_t>(MemoryTag::Count)> bytes{};
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = counters[i].bytes.load(std::memory_order_relaxed);
    }
    for (size_t i = 1; i < order.size(); ++i) {
        for (size_t j = i; j > 0 && bytes[order[j]] > bytes[order[j - 1]]; --j) {
            std::swap(order[j], order[j - 1]);
        }
    }
    std::string report;
    u64 total = 0;
    for (const size_t i : order) {
        const u64 peak = counters[i].peak.load(std::memory_order_relaxed);
        if (peak == 0) {
            continue;
        }
        total += bytes[i];
        report += fmt::format("{}{} {:.1f} MiB ({}, peak {:.1f})", report.empty() ? "" : ", ",
                              MemoryTagNames[i], static_cast<double>(bytes[i]) / (1 << 20),
                              counters[i].count.load(std::memory_order_relaxed),
                              static_cast<double>(peak) / (1 << 20));
    }
    return fmt::format("{:.1f} MiB: {}", static_cast<double>(total) / (1 << 20), report);
}

} // namespace MemoryTags

/// Tags the VMA allocations made on this thread while it lives.
class MemoryTagScope {
public:
    explicit MemoryTagScope(MemoryTag tag) noexcept : previous{MemoryTags::current} {
        MemoryTags::current = tag;
    }
    ~MemoryTagScope() {
        MemoryTags::current = previous;
    }
    MemoryTagScope(const MemoryTagScope&) = delete;
    MemoryTagScope& operator=(const MemoryTagScope&) = delete;

private:
    MemoryTag previous;
};

} // namespace Vulkan
