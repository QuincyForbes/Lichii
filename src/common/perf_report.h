// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/common_types.h"

/// Where frame time goes, for a log line every few seconds ("Perf:"): on by default on iOS,
/// EDEN_NX2_PERF_REPORT=1 elsewhere, =0 turns it off. The counters only ever grow (one relaxed
/// add next to far larger work, none while the report is off); the report takes differences.
namespace Common::PerfReport {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_REPORT");
#if defined(__APPLE__) && TARGET_OS_IPHONE
        return value == nullptr || value[0] != '0';
#else
        return value != nullptr && value[0] != '0';
#endif
    }();
    return enabled;
}

enum Counter : std::size_t {
    Draws,
    DrawsSkipped,        ///< Draws dropped for want of a pipeline (async compiles)
    TextureUploadBytes,  ///< Guest texture data uploaded by the texture cache
    BufferUploadBytes,   ///< Guest buffer data uploaded by the buffer cache
    StagingUploadBytes,  ///< Every upload staging request (the sources above and the rest)
    StagingPooledBytes,  ///< Upload staging past the stream buffer, at its allocation size
    EarlySubmits,        ///< Command buffers submitted for their pooled uploads
    CapWaits,            ///< Waits for the GPU at the pooled upload cap
    TextureCollected,    ///< Images the texture collector deleted
    TextureCollectorDownloads, ///< Of them, downloaded first (a wait for the GPU each)
    BufferCollected,     ///< Buffers the buffer collector deleted
    MeshArenaWaits,      ///< Waits for a submission's mesh emulation arena
    Count,
};

inline std::array<std::atomic<u64>, Count> g_counters{};

inline void Add(Counter counter, u64 amount = 1) {
    if (!Enabled()) {
        return;
    }
    g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
}

inline std::array<u64, Count> Read() {
    std::array<u64, Count> values{};
    for (std::size_t i = 0; i < Count; ++i) {
        values[i] = g_counters[i].load(std::memory_order_relaxed);
    }
    return values;
}

} // namespace Common::PerfReport
