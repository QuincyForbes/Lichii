// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Port of nxnative's frame time constraints (seedds/nxnative PR #2569 and #2137, MIT, Copyright
// (c) 2026 seedds); the full notice is in frame_deadline.h.

#include <algorithm>

#include "common/frame_deadline.h"

#ifdef __APPLE__
#include <atomic>
#include <cstdlib>
#include <TargetConditionals.h>
#include <fmt/format.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_info.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <time.h>
#include "common/logging.h"
#endif

namespace Common::FrameDeadline {

namespace {
constexpr u64 MinComputationNs = 1'000'000;

u64 ClampComputation(u64 computation_ns, u64 period_ns) {
    return (std::clamp)(computation_ns, MinComputationNs,
                        (std::max)(MinComputationNs, period_ns / 10 * 9));
}
} // Anonymous namespace

u64 Window::Need(std::span<u64> samples_ns, u64 period_ns) {
    if (samples_ns.empty()) {
        return 0;
    }
    std::sort(samples_ns.begin(), samples_ns.end());
    const u64 p90 = samples_ns[(std::min)(samples_ns.size() * 9 / 10, samples_ns.size() - 1)];
    const u64 peak = samples_ns.back();
    return ClampComputation((std::max)(p90 + p90 / 4, peak + peak / 10), period_ns);
}

u64 Window::Sample(u64 frame, u64 cpu_ns, u64 period_ns) {
    if (!primed) {
        primed = true;
        last_frame = frame;
        last_cpu_ns = cpu_ns;
        return 0;
    }
    if (frame <= last_frame || cpu_ns < last_cpu_ns) {
        return 0;
    }
    // A thread that sat out frames spreads its time over them.
    samples[count++] = (cpu_ns - last_cpu_ns) / (frame - last_frame);
    last_frame = frame;
    last_cpu_ns = cpu_ns;
    if (count < Size) {
        return 0;
    }
    count = 0;
    u64 sum = 0;
    for (const u64 sample : samples) {
        sum += sample;
    }
    mean_ns = sum / Size;
    const u64 need = Need(samples, period_ns);
    if (declared_ns != 0 && need * 100 <= declared_ns * 115 && need * 100 >= declared_ns * 85) {
        return 0;
    }
    declared_ns = need;
    return need;
}

#ifdef __APPLE__
namespace {

// Below this the kernel took the thread out of the real-time band (BASEPRI_RTQUEUES): it ran
// through its computation without blocking for too long.
constexpr s32 RealTimePriority = 97;

struct Slot {
    std::atomic<bool> used{};
    std::atomic<u32> role{};
    std::atomic<int> index{-1};
    std::atomic<u64> computation_ns{};
    std::atomic<u64> period_ns{};
    std::atomic<u64> mean_ns{};
    std::atomic<s32> priority{};
    std::atomic<u32> frames{};
    std::atomic<u32> demoted{};
    std::atomic<s32> error{};
};

std::array<Slot, 16> g_slots;
std::atomic<u64> g_frame{};
std::atomic<u64> g_period_ns{};
std::atomic<u64> g_last_frame_ns{};
std::atomic<u64> g_measured_ns{};
std::atomic<u64> g_fallback_period_ns{};

struct Member {
    Slot* slot{};
    Window window;
    u64 seen_frame{};
    u64 declared_period_ns{};
    thread_act_t port{};
    ~Member() {
        if (slot != nullptr) {
            slot->used.store(false, std::memory_order_release);
        }
    }
};
thread_local Member t_member;

u64 Ticks(u64 ns) {
    static const mach_timebase_info_data_t timebase = [] {
        mach_timebase_info_data_t info{};
        mach_timebase_info(&info);
        return info;
    }();
    return timebase.numer == 0 ? ns : ns * timebase.denom / timebase.numer;
}

// XNU rejects THREAD_TIME_CONSTRAINT_POLICY periods above 50 ms (KERN_INVALID_ARGUMENT = 4),
// and declaring a 50 ms (20 FPS) period during a brief frame dip signals iOS power management
// to step down SoC/GPU clocks into a self-reinforcing 20-22 FPS spiral. Never declare slower
// than 30 FPS (33.3 ms).
constexpr u64 MaxPeriodNs = 33'333'334;

void Declare(Member& member, u64 computation_ns, u64 period_ns) {
    period_ns = (std::min)(period_ns, MaxPeriodNs);
    computation_ns = ClampComputation(computation_ns, period_ns);
    thread_time_constraint_policy_data_t policy{};
    policy.period = static_cast<u32>(Ticks(period_ns));
    policy.computation = static_cast<u32>(Ticks(computation_ns));
    policy.constraint = static_cast<u32>(Ticks(period_ns));
    policy.preemptible = 1;
    const kern_return_t result =
        thread_policy_set(member.port, THREAD_TIME_CONSTRAINT_POLICY,
                          reinterpret_cast<thread_policy_t>(&policy),
                          THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    Slot& slot = *member.slot;
    if (result != KERN_SUCCESS) {
        if (slot.error.exchange(result, std::memory_order_relaxed) != result) {
            LOG_WARNING(Common, "Frame deadlines: thread_policy_set failed ({})", result);
        }
        return;
    }
    member.declared_period_ns = period_ns;
    slot.error.store(0, std::memory_order_relaxed);
    slot.computation_ns.store(computation_ns, std::memory_order_relaxed);
    slot.period_ns.store(period_ns, std::memory_order_relaxed);
}

void TickFrame(Member& member, u64 frame) {
    u64 period_ns = g_period_ns.load(std::memory_order_relaxed);
    if (period_ns == 0) {
        period_ns = g_fallback_period_ns.load(std::memory_order_relaxed);
    }
    if (period_ns == 0) {
        return;
    }
    Slot& slot = *member.slot;
    if (member.declared_period_ns != 0) {
        thread_extended_info_data_t info{};
        mach_msg_type_number_t count = THREAD_EXTENDED_INFO_COUNT;
        if (thread_info(member.port, THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&info),
                        &count) == KERN_SUCCESS) {
            slot.priority.store(info.pth_curpri, std::memory_order_relaxed);
            slot.frames.fetch_add(1, std::memory_order_relaxed);
            if (info.pth_curpri < RealTimePriority) {
                slot.demoted.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    u64 computation_ns =
        member.window.Sample(frame, clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID), period_ns);
    slot.mean_ns.store(member.window.MeanNs(), std::memory_order_relaxed);
    if (computation_ns == 0 && member.declared_period_ns != 0) {
        // The game changed its frame rate: the same work against the new period, at once.
        const u64 declared = member.declared_period_ns;
        const u64 change = period_ns > declared ? period_ns - declared : declared - period_ns;
        if (change * 20 > declared) {
            computation_ns = ClampComputation(member.window.Declared(), period_ns);
        }
    }
    if (computation_ns != 0) {
        Declare(member, ClampComputation(computation_ns, period_ns), period_ns);
    }
}

std::string Name(const Slot& slot) {
    switch (static_cast<Role>(slot.role.load(std::memory_order_relaxed))) {
    case Role::GpuBack:
        return "GPUBack";
    case Role::GpuFront:
        return "GPU";
    case Role::VulkanWorker:
        return "VulkanWorker";
    case Role::GuestCore:
        return fmt::format("CPUCore_{}", slot.index.load(std::memory_order_relaxed));
    }
    return "?";
}

} // Anonymous namespace

u32 Roles() {
    static const u32 roles = [] {
        const char* const value = std::getenv("EDEN_NX2_APPLE_FRAME_DEADLINE");
        if (value != nullptr && value[0] != '\0') {
            return static_cast<u32>(std::strtoul(value, nullptr, 0));
        }
#if TARGET_OS_IPHONE
        return u32{7};
#else
        return u32{0};
#endif
    }();
    return roles;
}

void SetPeriodNs(u64 period_ns) {
    g_period_ns.store(period_ns == 0 ? 0 : (std::min)(period_ns, MaxPeriodNs),
                      std::memory_order_relaxed);
}

void NoteFrame() {
    // An unpaced game has no period of its own: the frame time measured, in whole 60 Hz refreshes
    // (clamped to 30-60 FPS so transient dips never tell iOS to target 20 FPS).
    constexpr u64 RefreshNs = 16'666'667;
    if (Roles() == 0) {
        return;
    }
    const u64 now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const u64 last = g_last_frame_ns.exchange(now, std::memory_order_relaxed);
    if (last != 0 && now > last) {
        const u64 old = g_measured_ns.load(std::memory_order_relaxed);
        const u64 frame_ns = (std::min<u64>)(now - last, 2 * RefreshNs);
        g_measured_ns.store(old == 0 ? frame_ns : (old * 31 + frame_ns) / 32,
                            std::memory_order_relaxed);
    }
    if ((g_frame.fetch_add(1, std::memory_order_relaxed) + 1) % Window::Size == 0) {
        const u64 refreshes =
            (g_measured_ns.load(std::memory_order_relaxed) + RefreshNs / 2) / RefreshNs;
        g_fallback_period_ns.store(std::clamp<u64>(refreshes, 1, 2) * RefreshNs,
                                   std::memory_order_relaxed);
    }
}

void Join(Role role, int index) {
    Member& member = t_member;
    if ((Roles() & static_cast<u32>(role)) == 0 || member.slot != nullptr) {
        return;
    }
    for (Slot& slot : g_slots) {
        bool expected = false;
        if (!slot.used.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            continue;
        }
        slot.role.store(static_cast<u32>(role), std::memory_order_relaxed);
        slot.index.store(index, std::memory_order_relaxed);
        slot.computation_ns.store(0, std::memory_order_relaxed);
        slot.period_ns.store(0, std::memory_order_relaxed);
        slot.mean_ns.store(0, std::memory_order_relaxed);
        slot.frames.store(0, std::memory_order_relaxed);
        slot.demoted.store(0, std::memory_order_relaxed);
        slot.error.store(0, std::memory_order_relaxed);
        member.port = pthread_mach_thread_np(pthread_self());
        member.seen_frame = g_frame.load(std::memory_order_relaxed);
        member.slot = &slot;
        return;
    }
}

// Guest cores call this from guest fibers, which change host threads: inlined, the caller could
// keep one host thread's member across the switch.
__attribute__((noinline)) void Tick() {
    Member& member = t_member;
    if (member.slot == nullptr) {
        return;
    }
    const u64 frame = g_frame.load(std::memory_order_relaxed);
    if (frame == member.seen_frame) {
        return;
    }
    member.seen_frame = frame;
    TickFrame(member, frame);
}

std::string TakeStatusText() {
    std::string text;
    for (Slot& slot : g_slots) {
        if (!slot.used.load(std::memory_order_acquire)) {
            continue;
        }
        const u32 frames = slot.frames.exchange(0, std::memory_order_relaxed);
        const u32 demoted = slot.demoted.exchange(0, std::memory_order_relaxed);
        const u64 computation_ns = slot.computation_ns.load(std::memory_order_relaxed);
        text += text.empty() ? " | frame deadlines:" : ",";
        if (const s32 error = slot.error.load(std::memory_order_relaxed); error != 0) {
            text += fmt::format(" {} refused ({})", Name(slot), error);
        } else if (computation_ns == 0) {
            text += fmt::format(" {} measuring", Name(slot));
        } else {
            text += fmt::format(
                " {} {:.1f} of {:.1f} ms (used {:.1f}), priority {}, below real-time in {} of {} "
                "frames",
                Name(slot), static_cast<double>(computation_ns) / 1e6,
                static_cast<double>(slot.period_ns.load(std::memory_order_relaxed)) / 1e6,
                static_cast<double>(slot.mean_ns.load(std::memory_order_relaxed)) / 1e6,
                slot.priority.load(std::memory_order_relaxed), demoted, frames);
        }
    }
    return text;
}

#else

u32 Roles() {
    return 0;
}

void SetPeriodNs(u64) {}

void NoteFrame() {}

std::string TakeStatusText() {
    return {};
}

#endif

} // namespace Common::FrameDeadline
