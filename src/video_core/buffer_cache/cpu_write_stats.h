// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <utility>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::CpuWriteStats {

/// What BufferCache::OnCPUWrite found for a guest CPU write of a tracked page (the
/// "Perf CPU writes:" line next to "Perf memory callbacks:").
enum class Outcome : u32 {
    Unregistered,   ///< no buffer there (texture or shader pages)
    GpuPage,        ///< a GPU-modified page: collected, the page stays protected
    GpuBytes,       ///< ... and the written bytes themselves are GPU-written
    PageNoGpuBytes, ///< ... and no byte of the page is GPU-written (a stale tracker bit)
    PageCpuRead,    ///< ... and the CPU read GPU-written data of the page before
    PagePreflushable,
    AlreadyCpu,     ///< page already CPU-modified, yet trapped (another tracker holds it)
    Marked,         ///< marked CPU-modified here
    StillProtected, ///< after WriteMemory the page's tracking count is still above 0
    Reclaimed,      ///< EDEN_NX2_CPU_WRITE_RECLAIM: a stale GPU mark dropped, written as CPU page
    DownloadQueued, ///< EDEN_NX2_CPU_WRITE_RECLAIM=2: the page's GPU bytes queued for download
    Count,
};

/// EDEN_NX2_CPU_WRITE_RECLAIM when unset (BufferCache::ReclaimGpuPagesForCpu). PC phone sim,
/// Ocarina of Time in Kokiri: GPU-tracked guest stores 2.5 M/s -> 1.2 M/s, standing +2 %.
inline constexpr int DefaultReclaim = 2;

inline int ReclaimMode() {
    static const int mode = [] {
        const char* value = std::getenv("EDEN_NX2_CPU_WRITE_RECLAIM");
        const int chosen = value == nullptr ? DefaultReclaim : std::atoi(value);
        LOG_INFO(HW_GPU, "CPU writes of GPU-written pages: reclaim mode {}{}", chosen,
                 value == nullptr ? " (default)" : " (EDEN_NX2_CPU_WRITE_RECLAIM)");
        return chosen;
    }();
    return mode;
}

inline std::array<std::atomic<u64>, static_cast<size_t>(Outcome::Count)> g_outcomes{};

/// GPU marks of the GPU-page outcome by age in frames: same frame, 1, 2-15, 16+.
inline std::array<std::atomic<u64>, 4> g_gpu_mark_age{};
/// GPU marks of the GPU-page outcome by source: written binding, DMA copy, unknown.
inline std::array<std::atomic<u64>, 3> g_gpu_mark_source{};

inline void Note(Outcome outcome) {
    g_outcomes[static_cast<size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
}

/// Last GPU mark per page, direct-mapped (diagnostics only; under the buffer cache mutex).
struct PageMark {
    u64 page{~u64{0}};
    u64 frame{};
    u64 begin{};
    u64 size{};
    u32 source{};
};
inline constexpr size_t PageMarkSlots = 4096;
inline std::array<PageMark, PageMarkSlots> g_page_marks{};

inline PageMark& Slot(u64 page) {
    return g_page_marks[(page ^ (page >> 12)) & (PageMarkSlots - 1)];
}

/// EDEN_NX2_CPU_WRITE_STATS=1: GPU mark ages and sources, the busiest pages.
inline bool Detailed() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_CPU_WRITE_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

struct PageSample {
    u64 addr{};
    u64 count{};
    PageMark mark{};
    u64 age{};
    /// Hull of the page's GPU-written bytes (gpu_modified_ranges), as offsets in the page.
    u64 gpu_first{};
    u64 gpu_end{};
};
inline std::mutex g_sample_mutex;
inline std::array<PageSample, 16> g_samples{};

inline void NoteSample(const PageSample& sample_in) {
    std::scoped_lock lock{g_sample_mutex};
    const u64 page = sample_in.addr >> 12;
    PageSample* lowest = &g_samples[0];
    for (PageSample& sample : g_samples) {
        if (sample.count != 0 && (sample.addr >> 12) == page) {
            const u64 count = sample.count + 1;
            sample = sample_in;
            sample.count = count;
            return;
        }
        if (sample.count < lowest->count) {
            lowest = &sample;
        }
    }
    *lowest = sample_in;
    lowest->count = 1;
}

inline std::array<PageSample, 16> TakeSamples() {
    std::scoped_lock lock{g_sample_mutex};
    return std::exchange(g_samples, {});
}

} // namespace VideoCommon::CpuWriteStats
