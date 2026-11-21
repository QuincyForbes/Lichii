// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::CpuReadStats {

/// Why a guest CPU read of GPU-tracked memory had to wait for the GPU (GPU::OnCPURead, the
/// "Perf CPU reads:" line next to "Perf memory callbacks:").
enum class Cause : u8 {
    None,
    Image,            ///< a GPU-modified image over the page (texture cache)
    PreflushInFlight, ///< a Preflushable page whose fence download is copied, not landed yet
    PreflushQueued,   ///< a Preflushable page written since its last fence download was taken
    Trapped,          ///< a GPU-written page the CPU had not read before (not Preflushable)
    Count,
};

inline constexpr size_t CauseCount = static_cast<size_t>(Cause::Count);

/// EDEN_NX2_GPU_READ_LAND. A guest read of a Preflushable page whose GPU bytes are not in guest
/// memory yet waits for them (GPU::OnCPURead).
/// 0: the GPU thread copies the page again and waits for the GPU to go idle, for each page.
/// 1: where a fence download of those bytes is already in flight, the GPU thread waits for that
///    download's own submission and lands it, with the other finished downloads within
///    EDEN_NX2_GPU_READ_LAND_KIB of the read (no copy, no submission, no wait for later work).
/// 2: also, when that download is already submitted, the reading core waits for it and lands
///    it itself: the GPU thread keeps recording and other readers are not held behind the wait.
/// PC phone sim, Ocarina of Time in Kokiri (3 interleaved sessions each): the GPU thread held by
/// these reads 79 -> 49 -> 5.5 ms/s, a GPU-tracked guest load 13.2 -> 9.7 -> 8.2 us, fps within
/// run-to-run noise. Apple hosts read-trap (no reactive flushing) and wait on a busier GPU.
inline constexpr int DefaultLand = 2;

inline int LandMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_READ_LAND");
        const int chosen = value == nullptr ? DefaultLand : std::atoi(value);
        LOG_INFO(HW_GPU, "CPU reads of GPU-written pages: land mode {}{}", chosen,
                 value == nullptr ? " (default)" : " (EDEN_NX2_GPU_READ_LAND)");
        return chosen;
    }();
    return mode;
}

inline u64 LandWindowBytes() {
    static const u64 bytes = [] {
        const char* value = std::getenv("EDEN_NX2_GPU_READ_LAND_KIB");
        return (value == nullptr ? u64{256} : std::strtoull(value, nullptr, 10)) * 1024;
    }();
    return bytes;
}

struct Counters {
    std::atomic<u64> asks{};
    std::atomic<u64> ask_ns{};
    /// Waiting for the caches' locks while asking (the GPU threads hold them through draws).
    std::atomic<u64> texture_asks{};
    std::atomic<u64> texture_lock_ns{};
    std::atomic<u64> buffer_asks{};
    std::atomic<u64> buffer_lock_ns{};
    std::array<std::atomic<u64>, CauseCount> waits{};
    /// From the request to the GPU thread starting it.
    std::array<std::atomic<u64>, CauseCount> queue_ns{};
    /// The flush itself on the GPU thread (downloads, their submit and wait).
    std::array<std::atomic<u64>, CauseCount> flush_ns{};
    std::atomic<u64> staged{};   ///< buffer downloads through staging: submit + wait for idle
    std::atomic<u64> staged_ns{};
    std::atomic<u64> mapped{};   ///< mapped buffers: wait for their last use
    std::atomic<u64> mapped_ns{};
    std::atomic<u64> bytes{};
    std::atomic<u64> landed{};   ///< fence downloads in flight landed early: wait for their tick
    std::atomic<u64> landed_ns{};
    std::atomic<u64> landed_bytes{};
    std::atomic<u64> core_landed{};  ///< reads whose core waited and landed the download itself
    std::atomic<u64> core_ns{};
    std::atomic<u64> core_refused{}; ///< tried that, went to the GPU thread after all
};

inline Counters g_counters;

/// Set on the GPU thread while it runs a CPU read's flush.
inline thread_local bool t_in_read = false;

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void Add(std::atomic<u64>& counter, u64 value) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

inline const char* CauseName(size_t cause) {
    switch (static_cast<Cause>(cause)) {
    case Cause::Image:
        return "image";
    case Cause::PreflushInFlight:
        return "preflushed, download in flight";
    case Cause::PreflushQueued:
        return "preflushed, written since its download";
    case Cause::Trapped:
        return "first read";
    default:
        return "other";
    }
}

} // namespace VideoCommon::CpuReadStats
