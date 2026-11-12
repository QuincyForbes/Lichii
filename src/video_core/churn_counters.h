// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"

/// What the caches make and destroy: per-frame churn of objects that should
/// be reused (render targets at drifted addresses, buffers joined over drifted ranges). Counted
/// always (one relaxed add where an object is made or destroyed, next to far larger work);
/// TextureCache::TickFrame logs "Churn (600 frames)" with EDEN_NX2_CHURN_LOG=1.
namespace VideoCommon::Churn {

enum Counter : size_t {
    ImagesCreated,
    ImagesDeleted,
    ImagesRelocated,
    ImagesPooled,      ///< New images that took a pooled VkImage (EXP42)
    ImageViewsCreated,
    ImageViewsDestroyed,
    BuffersCreated,    ///< BufferCache::CreateBuffer
    BuffersDeleted,    ///< BufferCache::DeleteBuffer
    BuffersJoined,     ///< BufferCache::JoinOverlap (each also deletes the joined buffer)
    BufferJoinBytes,   ///< Bytes copied by those joins
    BufferJoinsSkipped, ///< Joins without the copy (EDEN_NX2_JOIN_SKIP_CPU=1)
    BufferJoinSkippedBytes,
    StagingBuffers,    ///< StagingBufferPool::CreateStagingBuffer
    StagingBytes,
    DeviceAllocations, ///< vkAllocateMemory by the allocator (blocks and dedicated memory)
    DeviceFrees,
    DescriptorPools,   ///< New descriptor pools (a bank ran out)
    Count,
};

inline std::array<std::atomic<u64>, Count> g_counters{};
/// The counts at the last Take (the counters themselves only grow, see Read).
inline std::array<u64, Count> g_taken{};

inline void Add(Counter counter, u64 amount = 1) {
    g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
}

/// The counts since boot.
inline std::array<u64, Count> Read() {
    std::array<u64, Count> values{};
    for (size_t i = 0; i < Count; ++i) {
        values[i] = g_counters[i].load(std::memory_order_relaxed);
    }
    return values;
}

/// The counts since the last call (one caller: TextureCache::TickFrame).
inline std::array<u64, Count> Take() {
    std::array<u64, Count> values = Read();
    for (size_t i = 0; i < Count; ++i) {
        const u64 total = values[i];
        values[i] -= g_taken[i];
        g_taken[i] = total;
    }
    return values;
}

} // namespace VideoCommon::Churn
