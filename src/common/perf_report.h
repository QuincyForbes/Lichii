// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
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
    StagingScratchBytes, ///< Device-local scratch from the staging pool (mesh emulation output)
    EarlySubmits,        ///< Command buffers submitted for their pooled uploads
    CapWaits,            ///< Waits for the GPU at the pooled upload cap
    TextureCollected,    ///< Images the texture collector deleted
    TextureCollectorDownloads, ///< Of them, downloaded first (a wait for the GPU each)
    BufferCollected,     ///< Buffers the buffer collector deleted
    MeshArenaWaits,      ///< Waits for a submission's mesh emulation arena
    UniformStreamBytes,  ///< Of StagingUploadBytes: uniform buffers streamed per draw
    UniformPadBytes,     ///< Of them, zeroed padding past the bound size (MoltenVK)
    IndexStreamBytes,    ///< Of StagingUploadBytes: inline and widened index data
    GpuBusyNs,           ///< GPU time of finished command buffers, overlaps counted once
    GpuTimedBuffers,     ///< Command buffers that GpuBusyNs covers
    RenderPasses,        ///< Render passes begun
    MeshEmulatedDraws,   ///< Mesh draws run as compute + a vertex draw
    MeshPassBreaks,      ///< Of them, ones that ended a render pass for their compute
    GuestRunNs,          ///< Guest cores: wall time in guest code (the JIT)
    GuestKernelNs,       ///< Guest cores: wall time in SVCs, the scheduler and fiber switches
    GuestIdleNs,         ///< Guest cores: wall time waiting with nothing to run
    GuestIdles,          ///< Guest cores: waits begun
    JitBlocks,           ///< Blocks the JIT translated
    JitBytes,            ///< Host code they took
    JitCompileNs,        ///< Wall time translating them (inside guest code time)
    JitCacheFullFlushes, ///< Code caches emptied because they were full
    JitCacheClears,      ///< Code caches emptied on request (guest instruction cache ops)
    JitRangeInvalidations, ///< Ranges of translated code dropped on request
    Count,
};

inline std::array<std::atomic<u64>, Count> g_counters{};

inline void Add(Counter counter, u64 amount = 1) {
    if (!Enabled()) {
        return;
    }
    g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
}

/// The fullest any guest core's code cache got since the last TakeJitPeak, and its size.
inline std::atomic<u64> g_jit_peak_used{};
inline std::atomic<u64> g_jit_cache_size{};

inline void NoteJitUse(u64 used, u64 cache_size) {
    if (!Enabled()) {
        return;
    }
    g_jit_cache_size.store(cache_size, std::memory_order_relaxed);
    u64 peak = g_jit_peak_used.load(std::memory_order_relaxed);
    while (used > peak &&
           !g_jit_peak_used.compare_exchange_weak(peak, used, std::memory_order_relaxed)) {
    }
}

inline u64 TakeJitPeak() {
    return g_jit_peak_used.exchange(0, std::memory_order_relaxed);
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Guest core time split: adds the time since this thread's previous mark to `closed`.
inline void MarkGuest(Counter closed) {
    if (!Enabled()) {
        return;
    }
    thread_local u64 last_ns = 0;
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (last_ns != 0) {
        g_counters[closed].fetch_add(now - last_ns, std::memory_order_relaxed);
    }
    last_ns = now;
}

inline std::array<u64, Count> Read() {
    std::array<u64, Count> values{};
    for (std::size_t i = 0; i < Count; ++i) {
        values[i] = g_counters[i].load(std::memory_order_relaxed);
    }
    return values;
}

} // namespace Common::PerfReport
