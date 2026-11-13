// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>

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
    TextureCollectorRedrawnDrops, ///< Of them, GPU-written but redrawn: no download
                                  ///< (EDEN_NX2_TC_REDRAWN_FRAMES)
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
    AttachmentLoadBytes, ///< Attachment bytes render passes load (neither cleared nor don't-care)
    AttachmentStoreBytes, ///< Attachment bytes render passes store
    RenderPassReopens,   ///< Render passes begun on the framebuffer the last one ended on, no clear
    GuestRunNs,          ///< Guest cores: wall time in guest code (the JIT)
    GuestKernelNs,       ///< Guest cores: wall time in SVCs, the scheduler and fiber switches
    GuestIdleNs,         ///< Guest cores: wall time waiting with nothing to run
    GuestIdles,          ///< Guest cores: waits begun
    JitBlocks,           ///< Blocks the JIT translated
    JitBytes,            ///< Host code they took
    JitCompileNs,        ///< Wall time translating them (inside guest code time)
    JitCacheFullFlushes, ///< Code caches emptied because they were full
    JitCacheClears,      ///< Code caches emptied on request (guest instruction cache ops)
    JitCacheEvictions,   ///< Oldest slices of full code caches dropped (EDEN_NX2_JIT_EVICT)
    JitRangeInvalidations, ///< Ranges of translated code dropped on request
    JitFastmemFaults,    ///< Host faults in fastmem accesses (each recompiles its block)
    // Per guest core, added when the JIT returns (FlushGuestEvents), not one by one.
    GuestSvcs,           ///< Supervisor calls
    GuestSlowReads,      ///< Loads through the JIT's memory callbacks
    GuestSlowWrites,     ///< Stores through them
    GuestClockReads,     ///< CNTPCT reads through the JIT's callback (not inline)
    GuestSpinHints,      ///< WFE and YIELD hints
    GuestClockWaits,     ///< Clock polls the guest core slept in (guest_wait_backoff)
    GuestClockWaitNs,    ///< Time slept there
    GuestFutileYields,   ///< Yields that found nothing else to run
    GuestYieldSleeps,    ///< Of them, slept in (guest_wait_backoff)
    GuestYieldSleepNs,   ///< Time slept there
    Count,
};

inline std::array<std::atomic<u64>, Count> g_counters{};
/// Supervisor calls by number (guest cores add theirs every 1024 calls).
inline std::array<std::atomic<u64>, 256> g_svc_counts{};

inline void Add(Counter counter, u64 amount = 1) {
    if (!Enabled()) {
        return;
    }
    g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
}

/// The fullest any guest core's code cache got since the last TakeJitPeak, and its size.
inline std::atomic<u64> g_jit_peak_used{};
inline std::atomic<u64> g_jit_cache_size{};

/// Per guest core (the JIT's processor id): code in its cache now, the most since the last
/// report, full flushes and partial evictions (EDEN_NX2_JIT_EVICT) with the bytes they dropped.
constexpr std::size_t JitCores = 8;
struct JitCore {
    std::atomic<u64> live{};
    std::atomic<u64> peak{};
    std::atomic<u64> flushes{};
    std::atomic<u64> evictions{};
    std::atomic<u64> evicted_bytes{};
    std::atomic<u64> blocks{};
    std::atomic<u64> compile_ns{};
    std::atomic<u64> faults{};      ///< Fastmem faults in this core's code
    std::atomic<u64> dead_bytes{};  ///< Code of blocks invalidated in place (faults, range drops)
    std::atomic<u64> live_blocks{}; ///< Blocks the cache keeps bookkeeping for
    std::atomic<u64> table_bytes{}; ///< Heap bytes of the per-block tables
    std::atomic<u64> list_bytes{};  ///< Heap bytes of the blocks' link and fastmem lists
};
inline std::array<JitCore, JitCores> g_jit_cores{};

inline JitCore* JitCoreOf(std::size_t core) {
    return Enabled() && core < JitCores ? &g_jit_cores[core] : nullptr;
}

inline void NoteJitUse(std::size_t core, u64 used, u64 cache_size) {
    if (!Enabled()) {
        return;
    }
    g_jit_cache_size.store(cache_size, std::memory_order_relaxed);
    u64 peak = g_jit_peak_used.load(std::memory_order_relaxed);
    while (used > peak &&
           !g_jit_peak_used.compare_exchange_weak(peak, used, std::memory_order_relaxed)) {
    }
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->live.store(used, std::memory_order_relaxed);
        if (used > jc->peak.load(std::memory_order_relaxed)) {
            jc->peak.store(used, std::memory_order_relaxed);
        }
    }
}

/// The heap a core's cache keeps beside its code (arm64 backend).
inline void NoteJitTables(std::size_t core, u64 blocks, u64 table_bytes, u64 list_bytes) {
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->live_blocks.store(blocks, std::memory_order_relaxed);
        jc->table_bytes.store(table_bytes, std::memory_order_relaxed);
        jc->list_bytes.store(list_bytes, std::memory_order_relaxed);
    }
}

/// The fastmem markers all guest cores share: accesses, and blocks that gave up fastmem.
inline std::atomic<u64> g_jit_markers{};
inline std::atomic<u64> g_jit_marker_blocks{};
inline void NoteJitMarkers(u64 markers, u64 blocks) {
    g_jit_markers.store(markers, std::memory_order_relaxed);
    g_jit_marker_blocks.store(blocks, std::memory_order_relaxed);
}

inline void NoteJitCompile(std::size_t core, u64 ns) {
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->blocks.fetch_add(1, std::memory_order_relaxed);
        jc->compile_ns.fetch_add(ns, std::memory_order_relaxed);
    }
}

inline void NoteJitFlush(std::size_t core) {
    Add(JitCacheFullFlushes);
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->flushes.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void NoteJitFault(std::size_t core) {
    Add(JitFastmemFaults);
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->faults.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void NoteJitDeadCode(std::size_t core, u64 bytes) {
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->dead_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

inline void NoteJitEviction(std::size_t core, u64 bytes) {
    Add(JitCacheEvictions);
    if (JitCore* const jc = JitCoreOf(core)) {
        jc->evictions.fetch_add(1, std::memory_order_relaxed);
        jc->evicted_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
}

struct JitCoreCounts {
    u64 flushes, evictions, evicted_bytes, blocks, compile_ns, faults, dead_bytes;
    u64 live_blocks, table_bytes, list_bytes;
};

inline std::array<JitCoreCounts, JitCores> ReadJitCores() {
    std::array<JitCoreCounts, JitCores> out{};
    for (std::size_t i = 0; i < JitCores; ++i) {
        const JitCore& jc = g_jit_cores[i];
        out[i] = {jc.flushes.load(std::memory_order_relaxed),
                  jc.evictions.load(std::memory_order_relaxed),
                  jc.evicted_bytes.load(std::memory_order_relaxed),
                  jc.blocks.load(std::memory_order_relaxed),
                  jc.compile_ns.load(std::memory_order_relaxed),
                  jc.faults.load(std::memory_order_relaxed),
                  jc.dead_bytes.load(std::memory_order_relaxed),
                  jc.live_blocks.load(std::memory_order_relaxed),
                  jc.table_bytes.load(std::memory_order_relaxed),
                  jc.list_bytes.load(std::memory_order_relaxed)};
    }
    return out;
}

/// A core's code now and the most since the previous call.
inline std::pair<u64, u64> TakeJitCoreUse(std::size_t core) {
    JitCore& jc = g_jit_cores[core];
    const u64 live = jc.live.load(std::memory_order_relaxed);
    return {live, (std::max)(live, jc.peak.exchange(live, std::memory_order_relaxed))};
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

/// The process's CPU time and where it ran (Apple: proc_pid_rusage). Empty where the OS does
/// not say.
struct ProcessCpu {
    u64 cpu_ns;         ///< User + system time on all cores
    u64 performance_ns; ///< Of it, on performance cores
    u64 energy_nj;      ///< CPU energy (0 if not reported)
};
std::optional<ProcessCpu> ReadProcessCpu();

constexpr std::size_t PlacementCores = 6;
/// Guest core threads, sampled now and then: the host CPU each ran on (Apple).
struct Placement {
    std::array<u64, PlacementCores> samples;
    std::array<u64, PlacementCores> high; ///< On a CPU id at or above HighCpuFirst()
};
/// Called by guest core `core` between JIT runs; samples every 64th call.
void SampleGuestCorePlacement(std::size_t core);
Placement ReadPlacement();
/// The first CPU id counted as "high" (Apple: the number of efficiency cores, which come first);
/// 0 when unknown.
u32 HighCpuFirst();
/// Apple: the OS thermal pressure level (0 nominal, 10 moderate, 20 heavy, 30 trapping, 40
/// sleeping), finer than the thermal state apps are told; -1 where unknown.
int ThermalPressure();

inline std::array<u64, Count> Read() {
    std::array<u64, Count> values{};
    for (std::size_t i = 0; i < Count; ++i) {
        values[i] = g_counters[i].load(std::memory_order_relaxed);
    }
    return values;
}

} // namespace Common::PerfReport
