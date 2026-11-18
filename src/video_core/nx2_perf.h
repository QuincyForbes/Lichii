// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <string_view>

#include "common/common_types.h"
#include "common/logging.h"

// Switches for the NX2 GPU optimizations, all on by default (names in opt_names) except
// fence_flush and fragment_barrier (reverted with zortos293/Lichii#161: a slab of wrong geometry
// in Kokiri Forest; EDEN_NX2_PERF_ON=<name>,... turns them on) and split_overlap (quarantined,
// never on). EDEN_NX2_PERF_FRAMES_AHEAD=0 turns frames_ahead off.
// EDEN_NX2_PERF_OPT=0 turns every one off; EDEN_NX2_PERF_SAFE=1 turns off the experimental ones
// (DefaultOffMask); EDEN_NX2_PERF_ON=<name>,... turns named ones back on and
// EDEN_NX2_PERF_OFF=<name>,... turns named ones off. EDEN_NX2_PERF_VERIFY=1 computes each
// optimized answer the old way too, logs any difference and uses the old answer.
namespace VideoCommon::Nx2Perf {

enum class Opt : unsigned {
    Bounds,            ///< MaxContinuousRange / GetMemoryLayoutSize from map-time runs
    Method,            ///< Maxwell3D fast method dispatch on the back
    CbBatch,           ///< Consecutive constant buffer words uploaded in one write
    CleanWords,        ///< SynchronizeBuffer skips clean tracker words
    IndexRebind,       ///< Repeated vkCmdBindIndexBuffer skipped
    UsageTracker,      ///< UsageTracker::Track skips pages known to be used
    FrontSleep,        ///< The split front sleeps instead of yielding while it waits
    SubmitBudget,      ///< 4096-draw native submission budget outside MoltenVK
    FragmentBarrier,   ///< FragmentBarrier deferred to the next draw that reads the pass
    FenceFlush,        ///< 5 ms DeferFlush, command list copies submitted at once
    PipelineRetention, ///< Sixteen sessions of used pipelines, saved every 15 s
    SortedPageBatches,
    QmdScratch,
    QueryDiagnostics,
    QueryUploadScratch,
    MacroReadEarly, ///< MME DMA reads outside GPU-written 1 MiB regions skip the full drain
    MacroReadCbOverlay, ///< Early MME reads overlay queued constant buffer words instead of waiting
    MacroReadWriterSet, ///< Early MME reads wait only for entries that can write memory
    PreflushAging,      ///< Preflush marks the CPU stopped reading age out
    EagerPublish,       ///< The front publishes ring entries while the back is idle, and at writers
    FrontSameFirst,     ///< Split front: same-value state writes skipped before any other work
    FrontClassTable,    ///< Split front: executable method classes from one precomputed table
    FrontRun,           ///< Split front: a command's direct 3D register words written in one call
    MacroBatch,  ///< Front macro writes buffered in the JIT, same-value ones dropped inline
    MacroBorrow, ///< Front macros read their parameters from the command words in place
    MacroState,  ///< The front macro JIT keeps its State across calls
    DownloadBatch,      ///< Fence-time downloads: one copy per buffer, ranges <= 4 KiB apart merged
    PipelineFastCache,  ///< Direct-mapped graphics pipeline cache in front of the slow path
    DrawPreamble,       ///< GPU::TickWork skips the sync mutex when no request is queued
    BindMemo,           ///< Unchanged non-written binds skip touch, sync and usage across draws
    /// Split back, Ampere: submits recorded draws while the front waits for the back or the
    /// host GPU has run dry, so a command list copy waits only for the tail behind it
    SubmitWhileWaited,
    /// Native 64 KiB uniform banks: guest bytes compared with the shadow on the frame's first
    /// bind only, not on every draw (a later CPU rewrite in the frame shows one frame late)
    BankOncePerFrame,
    /// GPUDirtyMemoryManager::Gather returns without its locks when nothing was collected
    DirtyGatherFast,
    /// Split front: a native macro call whose MME DMA read only feeds its last writes (render
    /// enable) runs without waiting for the back; the back writes those from the real words
    DmaForward,
    /// Split front: an instance loop's draw.begin right after the same one, still unpublished,
    /// is counted into that ring entry (bits 24-31); the back adds the instances
    DrawBeginCoalesce,
    /// A suspended channel's 1 ms front sleep ends at once on a new submit or on a GPU release
    /// that latches a waiter (Tegra::Nx2PendingWake in nx2.h)
    PendingWake,
    /// Diagnostics (EDEN_NX2_DIAG*): the back counts draws in a thread-local table with cached
    /// ablate/inspect answers and folds them into the shared state under one lock
    DiagDrawLite,
    /// Async shaders: when a depth-tested color draw is skipped, record its geometry identity and
    /// skip depth-only draws for the same geometry for up to 2 frames or until the material
    /// pipeline is built
    GeometryConsistency,
    /// Native 64 KiB banks: per-frame set checked first
    BankCheck,
    /// Quarantined (never on): pending writes are not a proof of safe early MME reads
    SplitOverlap,
    /// Ocarina presents a frame ahead
    FramesAhead,
    WriteIntersections,
    /// Native graphics SPIR-V emitted by the pipeline worker, not the back
    AsyncPipelines,
    UnswizzleWorkers,
    Count,
};

inline constexpr std::array<std::string_view, static_cast<size_t>(Opt::Count)> opt_names{
    "bounds", "method", "cb_batch", "clean_words", "index_rebind", "usage_tracker", "front_sleep",
    "submit_budget", "fragment_barrier", "fence_flush", "pipeline_retention", "sorted_page_batches",
    "qmd_scratch", "query_diagnostics", "query_upload_scratch", "macro_read_early",
    "macro_read_cb_overlay", "macro_read_writer_set", "preflush_aging", "eager_publish",
    "front_same_first", "front_class_table", "front_run", "macro_batch", "macro_borrow",
    "macro_state", "download_batch", "pipeline_fast_cache", "draw_preamble", "bind_memo",
    "submit_while_waited", "bank_once_per_frame", "dirty_gather_fast",
    "dma_forward", "draw_begin_coalesce", "pending_wake", "diag_draw_lite",
    "geometry_consistency", "bank_check",
    "split_overlap", "frames_ahead", "write_intersections", "async_pipelines",
    "unswizzle_workers",
};

static_assert(static_cast<size_t>(Opt::Count) <= 64, "the off-mask is 64 bits");
static_assert(
    [] {
        for (const std::string_view name : opt_names) {
            if (name.empty()) {
                return false;
            }
        }
        return true;
    }(),
    "every switch needs a name in opt_names");

constexpr u64 Bit(Opt opt) {
    return u64{1} << static_cast<unsigned>(opt);
}

/// Bit i of an off-mask is opt_names[i]: set means off.
constexpr bool IsOff(u64 mask, size_t index) {
    return ((mask >> index) & 1) != 0;
}

/// Off under EDEN_NX2_PERF_SAFE unless EDEN_NX2_PERF_ON names them: the experimental switches.
inline constexpr u64 DefaultOffMask() {
    u64 mask = 0;
    for (const Opt opt : {Opt::MacroReadCbOverlay, Opt::MacroReadWriterSet, Opt::PreflushAging,
                          Opt::EagerPublish, Opt::FrontSameFirst, Opt::FrontClassTable,
                          Opt::FrontRun, Opt::MacroBatch, Opt::MacroBorrow, Opt::MacroState,
                          Opt::DownloadBatch, Opt::PipelineFastCache, Opt::DrawPreamble,
                          Opt::BindMemo, Opt::SubmitWhileWaited, Opt::DmaForward,
                          Opt::BankOncePerFrame, Opt::DrawBeginCoalesce, Opt::PendingWake,
                          Opt::DiagDrawLite}) {
        mask |= Bit(opt);
    }
    return mask;
}

/// The off-mask from EDEN_NX2_PERF_SAFE (safe), EDEN_NX2_PERF_ON (on) and EDEN_NX2_PERF_OFF (off).
/// `base_off` switches start off (EDEN_NX2_PERF_ON can turn them on), `forced_off` stay off.
inline u64 ParseOffMask(bool safe, std::string_view on, std::string_view off, u64 base_off = 0,
                        u64 forced_off = 0) {
    u64 mask = (safe ? DefaultOffMask() : u64{0}) | base_off;
    while (!on.empty()) {
        const size_t comma = on.find(',');
        const std::string_view name = on.substr(0, comma);
        for (size_t i = 0; i < opt_names.size(); ++i) {
            if (name == opt_names[i]) {
                mask &= ~Bit(static_cast<Opt>(i));
            }
        }
        on = comma == std::string_view::npos ? std::string_view{} : on.substr(comma + 1);
    }
    while (!off.empty()) {
        const size_t comma = off.find(',');
        const std::string_view name = off.substr(0, comma);
        bool known = false;
        for (size_t i = 0; i < opt_names.size(); ++i) {
            if (name == opt_names[i]) {
                mask |= Bit(static_cast<Opt>(i));
                known = true;
            }
        }
        if (!known && !name.empty()) {
            LOG_WARNING(HW_GPU, "Nx2Perf: unknown name {} in EDEN_NX2_PERF_OFF", name);
        }
        off = comma == std::string_view::npos ? std::string_view{} : off.substr(comma + 1);
    }
    mask |= forced_off;
    return mask;
}

inline bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_OPT");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

namespace detail {
/// Off-mask in effect (bit set = off; every switch when EDEN_NX2_PERF_OPT=0); ~0 before the
/// first On().
inline std::atomic<u64> live_off_mask{~u64{0}};
u64 InitOffMask();
} // namespace detail

inline bool On(Opt opt) {
    u64 off = detail::live_off_mask.load(std::memory_order_relaxed);
    if (off == ~u64{0}) [[unlikely]] {
        off = detail::InitOffMask();
    }
    return !IsOff(off, static_cast<size_t>(opt));
}

inline bool Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("EDEN_NX2_PERF_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    return verify;
}

/// The frame period the game asks for (nx2:vi2 present word 12 at the speed limit), 0 while
/// unknown or unpaced. The host frame pacer (vk_host_pacer.h) never holds frames longer.
inline std::atomic<long long> g_guest_frame_interval_ns{0};

/// The display's measured refresh period while the host frame pacer runs, else 0. The game's
/// present pacing (nx2_gsv.h) times its frames to whole multiples of it (genlock).
inline std::atomic<long long> g_host_refresh_ns{0};

/// The refresh period of the display the game window is on, paced or not (0: unknown). Frame
/// generation fills up to this rate by default.
inline std::atomic<long long> g_display_refresh_ns{0};

/// Whether the Vulkan device runs the guest's tensor operations on native cooperative matrices
/// (set at device creation). Ocarina's movie upscaler setting picks by it on Automatic.
inline std::atomic<bool> g_native_tensor_ops{false};

/// At 30 FPS and below the host frame pacer presents frames as soon as they are rendered and
/// publishes how long before the display's next refresh the last one was ready (ns), or -1
/// while it isn't pacing that way. The game's present pacing (nx2_gsv.h) moves its frames until
/// that is ~6 ms: each lands on a refresh with little wait (input lag) and evenly spaced.
inline std::atomic<long long> g_host_phase_ns{-1};

/// Host frame times: the interval between frames handed to the swapchain, counted in 0.25 ms
/// buckets (the last holds 63.75 ms and up). The debug server's status returns the counts; the
/// difference between two calls gives that window's percentiles (1% and 0.1% lows).
inline constexpr size_t FrameTimeBuckets = 256;
inline constexpr long long FrameTimeBucketNs = 250'000;
inline std::array<std::atomic<unsigned long long>, FrameTimeBuckets> g_frame_times{};

inline void NoteFrameTime(long long interval_ns) {
    // Longer gaps are loading screens or a paused game, not frames.
    if (interval_ns <= 0 || interval_ns > 1'000'000'000) {
        return;
    }
    const auto bucket =
        (std::min)(static_cast<size_t>(interval_ns / FrameTimeBucketNs), FrameTimeBuckets - 1);
    g_frame_times[bucket].fetch_add(1, std::memory_order_relaxed);
}

} // namespace VideoCommon::Nx2Perf
