// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>

#include "common/common_types.h"
#include "common/perf_report.h"

/// CPU scheduling figures of the 10 s report (EDEN_NX2_PERF_REPORT): the "Perf threads:" line
/// (host threads by CPU time) and the "Perf sched:" line (guest threads, wake latency, waits on
/// the GPU, yield backoff).
namespace Common::PerfSched {

enum Counter : u32 {
    YieldRealYields,  ///< Yields that let another thread run
    YieldRunStarts,   ///< Futile yields that started a run: none before, or after a real yield
    YieldGapResets,   ///< ...: longer than the gap since the previous one
    YieldRegResets,   ///< ...: the callee-saved registers changed
    YieldUnderGate,   ///< Futile yields of a run still shorter than the gate
    SyncptArmed,      ///< Syncpoint events the game asked for that the GPU had not reached
    SyncptReached,    ///< ...that it had reached already
    SyncptWaitNs,     ///< Asked for until signalled
    Presents,
    PresentFenceNs,   ///< Present blocked until the GPU thread reached the frame's fence
    PresentPaceNs,    ///< Present held for the vblank
    Count,
};

inline std::array<std::atomic<u64>, Counter::Count> g_counters{};

inline void Add(Counter counter, u64 amount = 1) {
    if (PerfReport::Enabled()) {
        g_counters[counter].fetch_add(amount, std::memory_order_relaxed);
    }
}

/// Whether SleepThread's yield backoff is on (svc_thread.cpp): -1 until the first futile yield.
inline std::atomic<s32> g_yield_backoff{-1};

/// A distribution of waits: two buckets per power of two of nanoseconds.
struct Latency {
    static constexpr size_t Buckets = 96;

    std::atomic<u64> count{};
    std::atomic<u64> sum_ns{};
    std::atomic<u64> max_ns{};
    std::array<std::atomic<u32>, Buckets> buckets{};

    static size_t Bucket(u64 ns) {
        if (ns < 2) {
            return 0;
        }
        const size_t octave = static_cast<size_t>(std::bit_width(ns)) - 1;
        const size_t index = octave * 2 + static_cast<size_t>((ns >> (octave - 1)) & 1);
        return index < Buckets ? index : Buckets - 1;
    }

    /// The bucket's upper bound.
    static u64 BucketEndNs(size_t index) {
        const u64 base = u64{1} << (index / 2);
        return (index & 1) != 0 ? base * 2 : base + base / 2;
    }

    void Add(u64 ns) {
        count.fetch_add(1, std::memory_order_relaxed);
        sum_ns.fetch_add(ns, std::memory_order_relaxed);
        buckets[Bucket(ns)].fetch_add(1, std::memory_order_relaxed);
        u64 max = max_ns.load(std::memory_order_relaxed);
        while (ns > max && !max_ns.compare_exchange_weak(max, ns, std::memory_order_relaxed)) {
        }
    }
};

/// A guest core's host thread: asked to wake from its idle sleep until it runs again.
inline Latency g_core_wakes;
/// The host timer thread: an event's time until the thread is back from its sleep.
inline Latency g_timer_late;

/// One guest thread (KScheduler fills it in).
struct GuestThread {
    std::atomic<u64> id{};
    std::atomic<s32> priority{};
    std::atomic<bool> application{};
    std::atomic<u64> run_ns{};       ///< On a guest core
    std::atomic<u64> ready_ns{};     ///< Runnable without a guest core
    std::atomic<u64> ready_max_ns{}; ///< Longest such wait since the last report
    std::atomic<u64> switches{};     ///< Times it was given a core
    std::atomic<u64> core_changes{}; ///< ...a different one than before
    std::atomic<u32> cores{};        ///< Mask of the cores since the last report
};

inline constexpr s32 GuestThreads = 512;
inline std::array<GuestThread, GuestThreads> g_guest_threads{};
inline std::atomic<u32> g_guest_thread_count{};

/// A slot for a new guest thread. They go round: the oldest thread's is taken once all are used,
/// by when that thread has long exited.
inline s32 AllocateGuestThread(u64 id, bool application) {
    const s32 slot = static_cast<s32>(g_guest_thread_count.fetch_add(1, std::memory_order_relaxed) %
                                      static_cast<u32>(GuestThreads));
    g_guest_threads[static_cast<size_t>(slot)].id.store(id, std::memory_order_relaxed);
    g_guest_threads[static_cast<size_t>(slot)].application.store(application,
                                                                 std::memory_order_relaxed);
    return slot;
}

/// Once per frame on the GPU thread: logs the lines every 10 s.
void TickFrame();

} // namespace Common::PerfSched
