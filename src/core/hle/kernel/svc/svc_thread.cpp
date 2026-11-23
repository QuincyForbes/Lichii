// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/cpu_features.h"
#include "common/nx2_poll.h"
#include "common/perf_report.h"
#include "common/perf_sched.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/spin_deadline.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/debugger/trace_hooks.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_memory_layout.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_resource_reservation.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/svc/svc_thread_yield.h"
#include "core/hle/kernel/svc_results.h"
#include "core/hle/kernel/thread_startup_policy.h"

namespace Kernel::Svc {
namespace {

bool IsValidProcessCoreId(const KProcess& process, int32_t core_id) {
    const int32_t num_cores =
        (process.IsSwitch2() || (process.GetCoreMask() >> 4) != 0)
            ? static_cast<int32_t>(Core::Hardware::NUM_CPU_CORES)
            : 4;
    return 0 <= core_id && core_id < num_cores;
}

void FutileYieldPauseOnce() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

void FutileYieldRelax(u16 streak) {
    const u32 pauses = streak < 2U ? 0U : streak < 4U ? 1U : streak < 8U ? 4U : streak < 32U ? 8U : 16U;
    for (u32 i = 0; i < pauses; ++i) {
        FutileYieldPauseOnce();
    }
}

u64 SteadyNowNs() {
#if defined(__aarch64__) && !defined(_MSC_VER) && defined(__SIZEOF_INT128__)
    if (Common::SpinDeadline::RawClockEnabled()) {
        static const unsigned __int128 ns_factor =
            (static_cast<unsigned __int128>(1'000'000'000ULL) << 64) /
            Common::SpinDeadline::RawFrequency();
        const u64 ticks = Common::SpinDeadline::RawTicks();
        const u64 ns = static_cast<u64>((static_cast<unsigned __int128>(ticks) * ns_factor) >> 64);
        return ns != 0 ? ns : 1ULL;
    }
#elif defined(ARCHITECTURE_x86_64) && (defined(_M_X64) || defined(__x86_64__))
    if (Common::g_wall_clock.invariant) {
#if defined(_MSC_VER)
        const u64 tsc = __rdtsc();
        const u64 delta =
            tsc >= Common::g_wall_clock.base_rdtsc ? tsc - Common::g_wall_clock.base_rdtsc : 0ULL;
        const u64 ns = __umulh(delta, Common::g_wall_clock.ns_rdtsc_factor);
#else
        const u64 tsc = __builtin_ia32_rdtsc();
        const u64 delta =
            tsc >= Common::g_wall_clock.base_rdtsc ? tsc - Common::g_wall_clock.base_rdtsc : 0ULL;
        const u64 ns = static_cast<u64>(
            (static_cast<unsigned __int128>(delta) * Common::g_wall_clock.ns_rdtsc_factor) >> 64);
#endif
        return ns != 0 ? ns : 1ULL;
    }
#endif
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

u8* TryGetPlainUserPointer(const KProcess& process, u64 addr, size_t size) {
    if ((addr & (size - 1)) != 0) {
        return nullptr;
    }
    const auto& pt = process.GetPageTable().GetImpl();
    const u64 masked = addr & 0xffffffffffffULL;
    if ((masked >> pt.GetAddressSpaceBits()) != 0) {
        return nullptr;
    }
    const uintptr_t base = pt.entries[masked >> Core::Memory::YUZU_PAGEBITS].Pointer();
    return base != 0 ? reinterpret_cast<u8*>(base + masked) : nullptr;
}

struct YieldBackoff {
    std::optional<bool> forced; ///< From the environment; otherwise the guest_wait_backoff setting
    u64 gate_ns; ///< Yielding to nobody this long before the first sleep
    u64 min_ns;
    u64 cap_ns;
    u64 gap_ns; ///< A longer pause between two yields starts a new run
    bool same_regs; ///< Changed callee-saved registers start a new run
};

struct SvcFastFlags {
    bool fast_futile_yield;
    bool svc_trace_configured;
    bool starved_yield_handoff;
    s64 fast_arb_spin_us;
    YieldBackoff backoff;
};

/// EDEN_NX2_YIELD_BACKOFF: 0 off, 1 on, or <gate us>:<cap us>; unset follows guest_wait_backoff.
/// EDEN_NX2_YIELD_BACKOFF_GAP_US=<n>: the pause between two yields that still counts as the same
/// run (500). A host thread that lost its CPU between two yields exceeds a narrow gap, so a
/// loaded host needs headroom for the backoff to accumulate across OS preemptions.
/// EDEN_NX2_YIELD_BACKOFF_REGS=0: changed callee-saved registers no longer start a new run.
/// Ocarina of Time changes them at every one of its ~100,000 yields to nobody per second (its
/// job workers do work between yields), so the backoff never sleeps there; forced to (REGS=0 and
/// a 100 us gate), Kokiri Forest fell from ~105 to ~38 fps on the PC.
const SvcFastFlags& SvcFastConfig() {
    static const SvcFastFlags config = [] {
        const char* fast_yield_env = std::getenv("EDEN_NX2_FAST_FUTILE_YIELD");
        const char* trace_env = std::getenv("EDEN_NX2_SVC_TRACE");
        const char* handoff_env = std::getenv("EDEN_NX2_YIELD_HANDOFF");
        const char* arb_env = std::getenv("EDEN_NX2_ARB_SPIN");
        const s64 arb_raw =
            (arb_env == nullptr || arb_env[0] == '\0')
                ? s64{8}
                : std::max<s64>(s64{0}, static_cast<s64>(std::strtoll(arb_env, nullptr, 10)));

        YieldBackoff backoff{std::nullopt, 2'000'000, 10'000, 200'000, 500'000, true};
        if (const char* regs = std::getenv("EDEN_NX2_YIELD_BACKOFF_REGS"); regs != nullptr) {
            backoff.same_regs = regs[0] != '0';
        }
        if (const char* gap = std::getenv("EDEN_NX2_YIELD_BACKOFF_GAP_US"); gap != nullptr) {
            backoff.gap_ns = static_cast<u64>(std::strtoull(gap, nullptr, 10)) * 1000;
        }
        if (const char* value = std::getenv("EDEN_NX2_YIELD_BACKOFF");
            value != nullptr && value[0] != '\0') {
            backoff.forced = value[0] != '0';
            if (const char* colon = std::strchr(value, ':')) {
                backoff.gate_ns = static_cast<u64>(std::strtoull(value, nullptr, 10)) * 1000;
                backoff.cap_ns =
                    std::max<u64>(u64{10}, static_cast<u64>(std::strtoull(colon + 1, nullptr, 10))) *
                    1000;
            }
            LOG_INFO(
                Kernel_SVC,
                "Yield backoff: {} (after {} us of yields to nobody, {}-{} us, gaps to {} us, {})",
                *backoff.forced ? "on" : "off", backoff.gate_ns / 1000, backoff.min_ns / 1000,
                backoff.cap_ns / 1000, backoff.gap_ns / 1000,
                backoff.same_regs ? "same registers" : "any registers");
        }

        return SvcFastFlags{
            .fast_futile_yield = fast_yield_env == nullptr || fast_yield_env[0] != '0',
            .svc_trace_configured = trace_env != nullptr && trace_env[0] != '\0',
            .starved_yield_handoff = handoff_env == nullptr || handoff_env[0] != '0',
            .fast_arb_spin_us = std::min<s64>(arb_raw, s64{2}),
            .backoff = backoff,
        };
    }();
    return config;
}

bool IsYieldBackoffActive(const SvcFastFlags& flags) {
    return flags.backoff.forced ? *flags.backoff.forced
                                : Settings::values.guest_wait_backoff.GetValue();
}

constexpr u16 kStarvedHandoffStreak = 6;
constexpr u64 kStarvedHandoffGapNs = 500'000;
constexpr u64 kStarvedHandoffLeaseNs = 200'000;
constexpr u64 kFailedHandoffCooldownNs = 100'000;

struct FailedHandoffCache {
    s32 core_id{-1};
    u32 suggested_count{};
    s64 sched_count{-1};
    u64 failed_ns{};
};
thread_local FailedHandoffCache g_failed_handoff{};

bool IsFailedHandoffCoolingDown(KernelCore& kernel, const KThread& thread, u64 now) {
    const s32 core_id = thread.GetActiveCore();
    if (core_id < 0 || g_failed_handoff.core_id != core_id || g_failed_handoff.failed_ns == 0 ||
        now < g_failed_handoff.failed_ns) {
        return false;
    }
    if (KScheduler::GetCoreScheduledCount(kernel, core_id) > 1U ||
        KScheduler::GetCoreSuggestedCount(kernel, core_id) > g_failed_handoff.suggested_count) {
        return false;
    }
    const u64 elapsed = now - g_failed_handoff.failed_ns;
    if (elapsed < kFailedHandoffCooldownNs) {
        return true;
    }
    const KProcess* const process = thread.GetOwnerProcess();
    return process != nullptr && process->GetScheduledCount() == g_failed_handoff.sched_count &&
           elapsed < kStarvedHandoffLeaseNs;
}

void RecordFailedHandoff(KernelCore& kernel, const KThread& thread, u64 now) {
    const s32 core_id = thread.GetActiveCore();
    if (core_id >= 0 && KScheduler::GetCoreScheduledCount(kernel, core_id) <= 1U) {
        const KProcess* const process = thread.GetOwnerProcess();
        g_failed_handoff.core_id = core_id;
        g_failed_handoff.suggested_count = KScheduler::GetCoreSuggestedCount(kernel, core_id);
        g_failed_handoff.sched_count = process != nullptr ? process->GetScheduledCount() : s64{-1};
        g_failed_handoff.failed_ns = now;
    }
}

bool CanFastPathSvc64(Core::System& system, const SvcFastFlags& flags, KernelCore*& out_kernel,
                      KProcess*& out_process, KThread*& out_thread, PhysicalCore*& out_phys_core,
                      s32& out_core_id) {
    if (!flags.fast_futile_yield || flags.svc_trace_configured) {
        return false;
    }
    auto& kernel = system.Kernel();
    if (!kernel.IsMulticore() ||
        Core::Trace::g_svc_hook.load(std::memory_order_relaxed) != nullptr) {
        return false;
    }
    KThread* const thread = GetCurrentThreadPointer(kernel);
    if (thread == nullptr) {
        return false;
    }
    KProcess* const process = thread->GetOwnerProcess();
    if (process == nullptr || !process->Is64Bit() || !thread->HasEnteredKernel() ||
        thread->HasDpc() || thread->GetStepState() != StepState::NotStepping ||
        thread->GetDisableDispatchCount() != 0 ||
        thread->GetRawState() != ThreadState::Runnable) {
        return false;
    }
    PhysicalCore& phys_core = kernel.CurrentPhysicalCore();
    const s32 core_id = thread->GetActiveCore();
    if (core_id != static_cast<s32>(phys_core.CoreIndex()) || phys_core.IsInterrupted()) {
        return false;
    }
    const auto& sched = kernel.Scheduler(static_cast<size_t>(core_id));
    if (sched.NeedsScheduling()) {
        return false;
    }
    if (const u64 exp = sched.GetHandoffExpiryNs();
        exp != 0 && Common::PerfReport::NowNs() >= exp) {
        return false;
    }
    out_kernel = &kernel;
    out_process = process;
    out_thread = thread;
    out_phys_core = &phys_core;
    out_core_id = core_id;
    return true;
}

u64 ComputeThreadSpinStateHash(KernelCore& kernel, const KProcess& process) {
    Svc::ThreadContext context;
    u64 regs = 0;
    if (auto* arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex())) {
        arm->GetContext(context);
        regs = HashYieldSpinWord(regs, context.pc);
        regs = HashYieldSpinWord(regs, context.sp);
        for (size_t i = 19; i <= 28; ++i) {
            regs = HashYieldSpinWord(regs, context.r[i]);
        }
        regs = HashYieldSpinWord(regs, context.fp);
    }
    return regs != 0 ? regs : 1ULL;
}

void SleepThreadForNs(KernelCore& kernel, KThread& thread, u64 sleep_ns) {
    s64 timeout = kernel.HardwareTimer().GetTick() + static_cast<s64>(sleep_ns) + 2;
    if (timeout <= 0) {
        timeout = (std::numeric_limits<s64>::max)();
    }
    static_cast<void>(thread.Sleep(kernel, timeout));
}

void NoteYieldBackoffState(s32 active) {
    if (Common::PerfSched::g_yield_backoff.load(std::memory_order_relaxed) != active) {
        Common::PerfSched::g_yield_backoff.store(active, std::memory_order_relaxed);
    }
}

bool HandleFutileYield(KernelCore& kernel, KThread& thread, const SvcFastFlags& flags,
                       u32 yield_state, u64 spin_hash, bool allow_schedule) {
    const bool backoff_on = IsYieldBackoffActive(flags);
    const bool has_starved =
        flags.starved_yield_handoff &&
        (yield_state & KScheduler::kYieldStateStarvedCandidate) != 0;

    const u64 now = SteadyNowNs();
    const u64 since_last = now >= thread.m_nx2_yield_last ? now - thread.m_nx2_yield_last : 0;
    const bool same_spin = spin_hash == 0 || spin_hash == thread.m_nx2_yield_regs;
    const u16 next_streak =
        same_spin
            ? (thread.m_nx2_futile_yield_streak < 0xFFFFU
                   ? static_cast<u16>(thread.m_nx2_futile_yield_streak + 1U)
                   : u16{0xFFFFU})
            : u16{1U};

    if (!backoff_on && !has_starved) {
        thread.m_nx2_last_yield_futile = true;
        Common::PerfReport::Add(Common::PerfReport::GuestFutileYields);
        thread.m_nx2_futile_yield_streak = next_streak;
        thread.m_nx2_yield_spin_streak = 0;
        thread.m_nx2_yield_last = now;
        if (spin_hash != 0) {
            thread.m_nx2_yield_regs = spin_hash;
        }
        NoteYieldBackoffState(0);
        if (next_streak >= 2U) {
            FutileYieldRelax(next_streak);
        }
        return true;
    }

    const bool recent_yield = thread.m_nx2_yield_last != 0 && since_last <= kStarvedHandoffGapNs;
    const bool spin_continues =
        recent_yield && (spin_hash == 0 || spin_hash == thread.m_nx2_yield_regs);
    const u16 next_spin_streak =
        spin_continues
            ? (thread.m_nx2_yield_spin_streak < 0xFFFFU
                   ? static_cast<u16>(thread.m_nx2_yield_spin_streak + 1U)
                   : u16{0xFFFFU})
            : u16{1U};
    const bool handoff_wanted = has_starved && next_spin_streak >= kStarvedHandoffStreak &&
                                !IsFailedHandoffCoolingDown(kernel, thread, now);

    const YieldBackoff& setting = flags.backoff;
    const bool regs_changed = backoff_on && setting.same_regs && spin_hash != thread.m_nx2_yield_regs;
    const bool run_restarts = thread.m_nx2_yield_run_start == 0 || since_last > setting.gap_ns ||
                              regs_changed;
    const u64 run_elapsed =
        now >= thread.m_nx2_yield_run_start ? now - thread.m_nx2_yield_run_start : 0;
    const bool backoff_sleep = backoff_on && !run_restarts && run_elapsed >= setting.gate_ns;

    if (!allow_schedule && (handoff_wanted || backoff_sleep)) {
        return false;
    }

    thread.m_nx2_last_yield_futile = true;
    Common::PerfReport::Add(Common::PerfReport::GuestFutileYields);
    thread.m_nx2_futile_yield_streak = next_streak;
    thread.m_nx2_yield_spin_streak = next_spin_streak;

    const auto try_handoff = [&]() -> bool {
        if (!handoff_wanted) {
            return false;
        }
        thread.m_nx2_yield_spin_streak = 0;
        if (KScheduler::TryHandoffStarvedOnYield(kernel, kStarvedHandoffLeaseNs)) {
            g_failed_handoff.failed_ns = 0;
            thread.m_nx2_futile_yield_streak = 0;
            thread.m_nx2_yield_last = SteadyNowNs();
            return true;
        }
        RecordFailedHandoff(kernel, thread, SteadyNowNs());
        return false;
    };

    if (!backoff_on) {
        NoteYieldBackoffState(0);
        thread.m_nx2_yield_last = now;
        if (spin_hash != 0) {
            thread.m_nx2_yield_regs = spin_hash;
        }
        if (try_handoff()) {
            return true;
        }
        if (next_streak >= 2U) {
            FutileYieldRelax(next_streak);
        }
        return true;
    }

    NoteYieldBackoffState(1);
    if (run_restarts) {
        Common::PerfSched::Add(thread.m_nx2_yield_run_start == 0
                                   ? Common::PerfSched::YieldRunStarts
                                   : regs_changed
                                         ? Common::PerfSched::YieldRegResets
                                         : Common::PerfSched::YieldGapResets);
        thread.m_nx2_yield_run_start = now;
        thread.m_nx2_yield_last = now;
        thread.m_nx2_yield_regs = spin_hash;
        if (try_handoff()) {
            return true;
        }
        if (next_streak >= 2U) {
            FutileYieldRelax(next_streak);
        }
        return true;
    }

    thread.m_nx2_yield_last = now;
    if (!backoff_sleep) {
        Common::PerfSched::Add(Common::PerfSched::YieldUnderGate);
        if (try_handoff()) {
            return true;
        }
        if (next_streak >= 2U) {
            FutileYieldRelax(next_streak);
        }
        return true;
    }

    thread.m_nx2_futile_yield_streak = 0;
    thread.m_nx2_yield_spin_streak = 0;
    const u64 sleep_ns = std::clamp<u64>(run_elapsed / 8, setting.min_ns, setting.cap_ns);
    SleepThreadForNs(kernel, thread, sleep_ns);
    const u64 after = SteadyNowNs();
    thread.m_nx2_yield_last = after;
    Common::PerfReport::Add(Common::PerfReport::GuestYieldSleeps);
    Common::PerfReport::Add(Common::PerfReport::GuestYieldSleepNs, after >= now ? after - now : 0);
    return true;
}

void YieldBackoffAfterYield(KernelCore& kernel, KThread& thread, const u32* cached_yield_state) {
    const auto& flags = SvcFastConfig();
    if (!thread.m_nx2_last_yield_futile) {
        // Another same-or-higher priority thread ran: reset backoff, but track rapid yield
        // ping-pongs so two priority-44 workers on one core still hand off to starved lower
        // priority threads queued behind them.
        thread.m_nx2_yield_run_start = 0;
        thread.m_nx2_futile_yield_streak = 0;
        Common::PerfSched::Add(Common::PerfSched::YieldRealYields);
        const u64 now = SteadyNowNs();
        const u64 since_last =
            now >= thread.m_nx2_yield_last ? now - thread.m_nx2_yield_last : 0;
        if (thread.m_nx2_yield_last != 0 && since_last <= kStarvedHandoffGapNs) {
            if (thread.m_nx2_yield_spin_streak < 0xFFFFU) {
                ++thread.m_nx2_yield_spin_streak;
            }
            if (flags.starved_yield_handoff &&
                thread.m_nx2_yield_spin_streak >= kStarvedHandoffStreak) {
                thread.m_nx2_yield_spin_streak = 0;
                if (KScheduler::HasStarvedRunnableForCore(kernel, thread.GetActiveCore()) &&
                    KScheduler::TryHandoffStarvedOnYield(kernel, kStarvedHandoffLeaseNs)) {
                    g_failed_handoff.failed_ns = 0;
                    thread.m_nx2_yield_last = SteadyNowNs();
                    return;
                }
            }
        } else {
            thread.m_nx2_yield_spin_streak = 1;
        }
        thread.m_nx2_yield_last = now;
        return;
    }

    u32 ys = KScheduler::kInitialYieldState;
    if (cached_yield_state != nullptr) {
        ys = *cached_yield_state;
    } else if (const s32 core_id = thread.GetActiveCore();
               core_id >= 0 && core_id < static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        ys = kernel.Scheduler(static_cast<size_t>(core_id)).GetYieldState();
    }

    const bool backoff_on = IsYieldBackoffActive(flags);
    const bool has_starved =
        flags.starved_yield_handoff && (ys & KScheduler::kYieldStateStarvedCandidate) != 0;
    const KProcess* const process = thread.GetOwnerProcess();
    const bool need_spin_hash =
        process != nullptr &&
        ((backoff_on && flags.backoff.same_regs) ||
         (has_starved &&
          static_cast<u32>(thread.m_nx2_yield_spin_streak) + 1U >= kStarvedHandoffStreak));
    const u64 spin_hash = need_spin_hash ? ComputeThreadSpinStateHash(kernel, *process) : 0;
    static_cast<void>(HandleFutileYield(kernel, thread, flags, ys, spin_hash, true));
}

// Ocarina of Time's retail NPDM exposes cores 0..5 (0x3F), pinning six priority-44 job workers to
// cores 0..5 while restricting its main render/frame threads (priorities 47 and 55) to 0x10/0x20/
// 0x30 (cores 4-5) and 0x38 (cores 3-5). With 8 emulated cores (0..7), Dedicated places those
// render/frame threads onto dedicated physical cores 6 and 7 while preserving virtual masks.
// EDEN_NX2_OOT_WIDE_AFFINITY: 0=Game, 1=Wide (0..5), 2=Ideal, 3=Dedicated (default when 8 cores).
enum class OotAffinity { Game, Wide, Ideal, Dedicated };

constexpr u64 kOotProgramId = 0x0400486024DEC000ULL;

OotAffinity OotAffinityMode(const KProcess& process) {
    static const OotAffinity mode = [] {
        constexpr OotAffinity default_mode =
            Core::Hardware::NUM_CPU_CORES >= 8 ? OotAffinity::Dedicated : OotAffinity::Wide;
        const char* value = std::getenv("EDEN_NX2_OOT_WIDE_AFFINITY");
        return value == nullptr   ? default_mode
               : value[0] == '0'  ? OotAffinity::Game
               : value[0] == '1'  ? OotAffinity::Wide
               : value[0] == '2'  ? OotAffinity::Ideal
               : value[0] == '3'  ? OotAffinity::Dedicated
                                  : default_mode;
    }();
    return process.GetProgramId() == kOotProgramId ? mode : OotAffinity::Game;
}

bool OotCore0WorkerOnCore7Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_WORKER_CORE7");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

s64 OotShortPollSleepNs() {
    static const s64 clamp_ns = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_SHORT_POLL_SLEEP");
        if (value == nullptr || value[0] == '\0') {
            return s64{150'000};
        }
        if (value[0] == '0' && value[1] == '\0') {
            return s64{0};
        }
        if (value[0] == '1' && value[1] == '\0') {
            return s64{150'000};
        }
        const long long parsed_us = std::strtoll(value, nullptr, 10);
        if (parsed_us <= 0) {
            return s64{0};
        }
        return std::clamp<s64>(static_cast<s64>(parsed_us) * 1000, s64{10'000}, s64{1'500'000});
    }();
    return clamp_ns;
}

bool IsProcessMainThread(KProcess& process, const KThread& thread) {
    KScopedLightLock lk{process.GetListLock()};
    for (const auto& candidate : process.GetThreadList()) {
        if (candidate.GetThreadType() == ThreadType::User) {
            return candidate.GetThreadId() == thread.GetThreadId();
        }
    }
    return false;
}

void ComputeOotDedicatedOverrides(u64 virtual_mask, s32 base_priority, bool is_main_thread,
                                  s32& out_phys_ideal, u64& out_phys_mask) {
    out_phys_ideal = -3;
    out_phys_mask = 0;
    if constexpr (Core::Hardware::NUM_CPU_CORES >= 8) {
        constexpr u64 all_phys = (1ULL << Core::Hardware::NUM_CPU_CORES) - 1;
        if (std::popcount(virtual_mask) > 1) {
            if (virtual_mask == 0x30ULL) {
                if (base_priority == 47) {
                    out_phys_ideal = 6;
                    out_phys_mask = (1ULL << 6);
                } else if (base_priority > 47) {
                    out_phys_ideal = 7;
                    out_phys_mask = all_phys & ~(1ULL << 6);
                }
            } else if (virtual_mask == 0x38ULL) {
                out_phys_ideal = 7;
                out_phys_mask = all_phys & ~(1ULL << 6);
            } else {
                out_phys_mask = all_phys & ~(1ULL << 6);
            }
        } else if ((virtual_mask & 0x30ULL) != 0 && base_priority == 47) {
            // Ocarina's main frame/render threads (priority 47) use single-core masks 0x10 and
            // 0x20, getting preempted ~22% of every frame by the priority-44 per-core job workers
            // on cores 4-5. Give them dedicated physical core 6 while keeping virtual masks intact.
            out_phys_ideal = 6;
            out_phys_mask = (1ULL << 6);
        } else if (virtual_mask == 0x1ULL && base_priority == 44 && !is_main_thread &&
                   OotCore0WorkerOnCore7Enabled()) {
            // Ocarina's main thread (created at process boot on core 0 at priority 44) otherwise
            // shares core 0 with job worker #0 (also core 0, priority 44), causing ~2.5-3.0 ms per
            // frame of same-core mutex priority-inheritance preemption ping-pong while core 7 has
            // zero priority-44 workers. Pin secondary core-0 priority-44 workers onto core 7.
            out_phys_ideal = 7;
            out_phys_mask = (1ULL << 7);
        } else if (base_priority >= 45) {
            // Lower-priority single-core service/I/O threads (such as t103 IStorage reader, audio
            // and decompression workers at priority >= 45) starve for full 10 ms preemption quanta
            // behind pairs of priority-44 job spinners on cores 0..5 while the main render thread
            // blocks on them; let them run on core 7 or any free non-render core while keeping all
            // priority <= 44 per-core job workers pinned to cores 0..5 (and core 7 for worker #0).
            out_phys_ideal = 7;
            out_phys_mask = all_phys & ~(1ULL << 6);
        }
    }
}

} // Anonymous namespace

bool YieldBackoffEnabled() {
    return IsYieldBackoffActive(SvcFastConfig());
}

bool TryFastFutileYield64(Core::System& system, s64 ns, u64 spin_state_hash) {
    if (static_cast<u64>(ns) + 2ULL > 2ULL) {
        return false;
    }
    const auto yield_type = static_cast<Svc::YieldType>(ns);
    const auto& flags = SvcFastConfig();
    KernelCore* kernel{};
    KProcess* process{};
    KThread* thread{};
    PhysicalCore* phys_core{};
    s32 core_id{};
    if (!CanFastPathSvc64(system, flags, kernel, process, thread, phys_core, core_id)) {
        return false;
    }
    u32 ys = KScheduler::kInitialYieldState;
    const bool is_futile =
        KScheduler::IsCoreYieldFutile(*kernel, *process, *thread, core_id, yield_type, &ys) ||
        (yield_type == Svc::YieldType::WithoutCoreMigration &&
         thread->GetYieldCoreId() == core_id &&
         thread->GetYieldCoreSeq() == KScheduler::GetCoreScheduledQueueVersion(*kernel, core_id)) ||
        thread->GetYieldScheduleCount() == process->GetScheduledCount();
    if (!is_futile) {
        return false;
    }
    return HandleFutileYield(*kernel, *thread, flags, ys, spin_state_hash, false);
}

bool TryFastGetCurrentProcessorNumber64(Core::System& system, u64* out_core) {
    const auto& flags = SvcFastConfig();
    KernelCore* kernel{};
    KProcess* process{};
    KThread* thread{};
    PhysicalCore* phys_core{};
    s32 core_id{};
    if (!CanFastPathSvc64(system, flags, kernel, process, thread, phys_core, core_id)) {
        return false;
    }
    *out_core = static_cast<u64>(phys_core->CoreIndex());
    return true;
}

bool TryFastSignalToAddress64(Core::System& system, u64 address, u32 signal_type, s32 value,
                              s32 count, u64* out_result) {
    if (IsKernelAddress(address) || (address & (sizeof(s32) - 1)) != 0) {
        return false;
    }
    const auto type = static_cast<Svc::SignalType>(signal_type);
    if (type != Svc::SignalType::Signal &&
        type != Svc::SignalType::SignalAndIncrementIfEqual &&
        type != Svc::SignalType::SignalAndModifyByWaitingCountIfEqual) {
        return false;
    }
    const auto& flags = SvcFastConfig();
    KernelCore* kernel{};
    KProcess* process{};
    KThread* thread{};
    PhysicalCore* phys_core{};
    s32 core_id{};
    if (!CanFastPathSvc64(system, flags, kernel, process, thread, phys_core, core_id)) {
        return false;
    }
    Result res = ResultSuccess;
    if (process->TryFastSignalAddressArbiter(address, type, value, count, &res)) {
        *out_result = res.raw;
        return true;
    }
    if (type == Svc::SignalType::SignalAndIncrementIfEqual ||
        type == Svc::SignalType::SignalAndModifyByWaitingCountIfEqual) {
        if (u8* const ptr = TryGetPlainUserPointer(*process, address, sizeof(s32));
            ptr != nullptr &&
            std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire) !=
                value) {
            *out_result = ResultInvalidState.raw;
            return true;
        }
    }
    return false;
}

bool TryFastWaitForAddress64(Core::System& system, u64 address, u32 arb_type, s64 value,
                             s64 timeout_ns, u64* out_result) {
    if (IsKernelAddress(address)) {
        return false;
    }
    const auto type = static_cast<Svc::ArbitrationType>(arb_type);
    const size_t align =
        type == Svc::ArbitrationType::WaitIfEqual64 ? sizeof(s64) : sizeof(s32);
    if ((address & (align - 1)) != 0) {
        return false;
    }
    const auto& flags = SvcFastConfig();
    KernelCore* kernel{};
    KProcess* process{};
    KThread* thread{};
    PhysicalCore* phys_core{};
    s32 core_id{};
    if (!CanFastPathSvc64(system, flags, kernel, process, thread, phys_core, core_id) ||
        thread->IsTerminationRequested()) {
        return false;
    }
    if (type == Svc::ArbitrationType::DecrementAndWaitIfLessThan) {
        if (u8* const ptr = TryGetPlainUserPointer(*process, address, sizeof(s32));
            ptr != nullptr &&
            std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire) >=
                static_cast<s32>(value)) {
            *out_result = ResultInvalidState.raw;
            return true;
        }
        return false;
    }
    Result res = ResultSuccess;
    if (process->TryFastWaitAddressArbiter(address, type, value, timeout_ns, &res)) {
        *out_result = res.raw;
        return true;
    }
    if (timeout_ns == 0 ||
        (type != Svc::ArbitrationType::WaitIfLessThan &&
         type != Svc::ArbitrationType::WaitIfEqual &&
         type != Svc::ArbitrationType::WaitIfEqual64)) {
        return false;
    }
    const s64 spin_us = flags.fast_arb_spin_us;
    u8& credit = thread->m_nx2_arb_spin_credit;
    if (spin_us <= 0 || credit == 0 || KScheduler::HasStarvedRunnableForCore(*kernel, core_id)) {
        return false;
    }
    u8* const ptr = TryGetPlainUserPointer(*process, address, align);
    if (ptr == nullptr) {
        return false;
    }
    const bool is_64 = type == Svc::ArbitrationType::WaitIfEqual64;
    const auto still_waits = [&]() -> bool {
        if (type == Svc::ArbitrationType::WaitIfLessThan) {
            return std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(
                       std::memory_order_acquire) < static_cast<s32>(value);
        }
        if (is_64) {
            return std::atomic_ref<s64>{*reinterpret_cast<s64*>(ptr)}.load(
                       std::memory_order_acquire) == value;
        }
        return std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire) ==
               static_cast<s32>(value);
    };
    const Common::SpinDeadline::Deadline deadline{std::chrono::microseconds{spin_us}};
    for (u32 spin = 1;; ++spin) {
        FutileYieldPauseOnce();
        if (!still_waits()) {
            if (thread->IsTerminationRequested()) {
                return false;
            }
            credit = static_cast<u8>(std::min<u32>(static_cast<u32>(credit) + 2U, 16U));
            *out_result = ResultInvalidState.raw;
            return true;
        }
        if ((spin & 31U) == 0) {
            if (phys_core->IsInterrupted() ||
                KScheduler::HasStarvedRunnableForCore(*kernel, core_id) || deadline.Expired()) {
                return false;
            }
        }
    }
}

bool TryFastArbitrateLock64(Core::System& system, u32 thread_handle, u64 address, u32 /*tag*/,
                            u64* out_result) {
    if (IsKernelAddress(address) || (address & (sizeof(u32) - 1)) != 0) {
        return false;
    }
    const auto& flags = SvcFastConfig();
    KernelCore* kernel{};
    KProcess* process{};
    KThread* thread{};
    PhysicalCore* phys_core{};
    s32 core_id{};
    if (!CanFastPathSvc64(system, flags, kernel, process, thread, phys_core, core_id) ||
        thread->IsTerminationRequested()) {
        return false;
    }
    u8* const ptr = TryGetPlainUserPointer(*process, address, sizeof(u32));
    if (ptr == nullptr) {
        return false;
    }
    const u32 cur =
        std::atomic_ref<u32>{*reinterpret_cast<u32*>(ptr)}.load(std::memory_order_acquire);
    if (cur == (thread_handle | Svc::HandleWaitMask)) {
        return false;
    }
    *out_result = ResultSuccess.raw;
    return true;
}

/// Creates a new thread
Result CreateThread(Core::System& system, Handle* out_handle, u64 entry_point, u64 arg,
                    u64 stack_bottom, s32 priority, s32 core_id) {
    LOG_DEBUG(Kernel_SVC,
              "called entry_point={:#08x}, arg={:#08x}, stack_bottom={:#08x}, "
              "priority={:#08x}, core_id={:#08x}",
              entry_point, arg, stack_bottom, priority, core_id);

    // Adjust core id, if it's the default magic.
    auto& kernel = system.Kernel();
    auto& process = GetCurrentProcess(kernel);
    if (core_id == IdealCoreUseProcessValue) {
        core_id = process.GetIdealCoreId();
    }

    // Validate arguments.
    R_UNLESS(IsValidProcessCoreId(process, core_id), ResultInvalidCoreId);
    R_UNLESS(((1ull << core_id) & process.GetCoreMask()) != 0, ResultInvalidCoreId);

    R_UNLESS(HighestThreadPriority <= priority && priority <= LowestThreadPriority,
             ResultInvalidPriority);
    R_UNLESS(process.CheckThreadPriority(priority), ResultInvalidPriority);

    // Reserve a new thread from the process resource limit (waiting up to 100ms).
    KScopedResourceReservation thread_reservation(system.Kernel(), std::addressof(process),
                                                  LimitableResource::ThreadCountMax, 1,
                                                  kernel.HardwareTimer().GetTick() + 100000000);
    R_UNLESS(thread_reservation.Succeeded(), ResultLimitReached);

    // Create the thread.
    KThread* thread = KThread::Create(kernel);
    R_UNLESS(thread != nullptr, ResultOutOfResource)
    SCOPE_EXIT {
        thread->Close(system.Kernel());
    };

    // Initialize the thread.
    {
        KScopedLightLock lk{process.GetStateLock()};
        R_TRY(KThread::InitializeUserThread(system, thread, entry_point, arg, stack_bottom,
                                            priority, core_id, std::addressof(process)));
    }

    // Commit the thread reservation.
    thread_reservation.Commit();

    // Clone the current fpu status to the new thread.
    thread->CloneFpuStatus(system.Kernel());

    // Register the new thread.
    KThread::Register(kernel, thread);

    if (OotAffinityMode(process) == OotAffinity::Dedicated && Core::Hardware::NUM_CPU_CORES >= 8) {
        const u64 virtual_mask = 1ULL << core_id;
        s32 phys_ideal = -3;
        u64 phys_mask = 0;
        ComputeOotDedicatedOverrides(virtual_mask, priority, false, phys_ideal, phys_mask);
        if (phys_mask != 0) {
            static_cast<void>(
                thread->SetCoreMask(kernel, core_id, virtual_mask, phys_ideal, phys_mask));
        }
    }

    // Add the thread to the handle table.
    R_TRY(process.GetHandleTable().Add(system.Kernel(), out_handle, thread));

    // Pass the thread handle to the thread local region.
    process.GetMemory().Write32(GetInteger(thread->GetTlsAddress()) + 0x110, *out_handle);

    R_SUCCEED();
}

/// Starts the thread for the provided handle
Result StartThread(Core::System& system, Handle thread_handle) {
    LOG_DEBUG(Kernel_SVC, "called thread={:#08x}", thread_handle);

    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Try to start the thread.
    R_TRY(thread->Run(system.Kernel()));

    if (GetCurrentProcess(system.Kernel()).GetProgramId() == MarioTennisFeverProgramId) {
        KScopedSchedulerLock sl{system.Kernel()};
        KThread& caller = GetCurrentThread(system.Kernel());
        if (NeedsStartupGrace(MarioTennisFeverProgramId, caller.GetActiveCore(), caller.GetPriority(),
                              thread->GetActiveCore(), thread->GetPriority())) {
            caller.RecordStartupThread(system.Kernel(), thread.GetPointerUnsafe());
        }
    }

    // On hardware a started thread that can run begins within microseconds. An emulated core takes
    // longer (wake-up, JIT), so let the thread reach its first SVC before the caller carries on.
    // Mario Tennis Fever's boot stub reuses the stack arguments it gave its threads right after
    // starting them.
    auto& kernel = system.Kernel();
    const s32 core = thread->GetActiveCore();
    if (core == GetCurrentCoreId(kernel) || !kernel.IsMulticore()) {
        // Sleeping lets this core run the new thread; in single-core mode the idle core passes
        // control to the next one.
        for (int i = 0; i < 10 && !thread->HasEnteredKernel() && !thread->IsTerminationRequested();
             ++i) {
            const s64 timeout = kernel.HardwareTimer().GetTick() + 100'000 + 2;
            static_cast<void>(GetCurrentThread(kernel).Sleep(kernel, timeout));
        }
    } else if (core >= 0 && core < static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        const KThread* const running = kernel.Scheduler(core).GetSchedulerCurrentThread();
        const bool preempts = running == nullptr || running == thread.GetPointerUnsafe() ||
                              running == kernel.Scheduler(core).GetIdleThread() ||
                              thread->GetPriority() < running->GetPriority();
        if (preempts) {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(5);
            while (!thread->HasEnteredKernel() && !thread->IsTerminationRequested() &&
                   std::chrono::steady_clock::now() < deadline) {
                Common::PollWait();
            }
        }
    }

    R_SUCCEED();
}

/// Called when a thread exits
void ExitThread(Core::System& system) {
    auto* const current_thread = GetCurrentThreadPointer(system.Kernel());
    system.GlobalSchedulerContext().RemoveThread(current_thread);
    current_thread->Exit(system.Kernel());
}

/// Sleep the current thread
static void SleepThreadImpl(Core::System& system, s64 ns, bool from_guest_svc) {
    auto& kernel = system.Kernel();
    const auto yield_type = static_cast<Svc::YieldType>(ns);

    LOG_TRACE(Kernel_SVC, "called nanoseconds={}", ns);

    // When the input tick is positive, sleep.
    if (ns > 0) {
        auto& cur_thread = GetCurrentThread(kernel);
        s64 effective_ns = ns;
        if (from_guest_svc && effective_ns >= 500'000 && effective_ns <= 1'500'000 &&
            cur_thread.GetBasePriority() == 45) {
            if (const KProcess* const cur_process = cur_thread.GetOwnerProcess();
                cur_process != nullptr && cur_process->GetProgramId() == kOotProgramId) {
                if (const s64 clamp_ns = OotShortPollSleepNs();
                    clamp_ns > 0 && effective_ns > clamp_ns) {
                    effective_ns = clamp_ns;
                }
            }
        }

        // Convert the timeout from nanoseconds to ticks.
        // NOTE: Nintendo does not use this conversion logic in WaitSynchronization...
        s64 timeout;

        const s64 offset_tick(effective_ns);
        if (offset_tick > 0) {
            timeout = kernel.HardwareTimer().GetTick() + offset_tick + 2;
            if (timeout <= 0) {
                timeout = (std::numeric_limits<s64>::max)();
            }
        } else {
            timeout = (std::numeric_limits<s64>::max)();
        }

        // Sleep.
        // NOTE: Nintendo does not check the result of this sleep.
        cur_thread.m_nx2_futile_yield_streak = 0;
        if (ns >= 250'000) {
            cur_thread.m_nx2_yield_spin_streak = 0;
        }
        static_cast<void>(cur_thread.Sleep(kernel, timeout));
    } else if (static_cast<u64>(ns) + 2ULL <= 2ULL) {
        KThread& cur_thread = GetCurrentThread(kernel);
        if (kernel.IsMulticore() && cur_thread.GetRawState() == ThreadState::Runnable) {
            const KProcess* const cur_process = cur_thread.GetOwnerProcess();
            u32 ys = KScheduler::kInitialYieldState;
            if (cur_process != nullptr &&
                KScheduler::IsCoreYieldFutile(kernel, *cur_process, cur_thread,
                                              cur_thread.GetActiveCore(), yield_type, &ys)) {
                cur_thread.m_nx2_last_yield_futile = true;
                YieldBackoffAfterYield(kernel, cur_thread, &ys);
                return;
            }
        }
        if (yield_type == Svc::YieldType::WithoutCoreMigration) {
            KScheduler::YieldWithoutCoreMigration(kernel);
        } else if (yield_type == Svc::YieldType::WithCoreMigration) {
            KScheduler::YieldWithCoreMigration(kernel);
        } else {
            KScheduler::YieldToAnyThread(kernel);
        }
        YieldBackoffAfterYield(kernel, cur_thread, nullptr);
    } else {
        // Nintendo does nothing at all if an otherwise invalid value is passed.
    }
}

void SleepThread(Core::System& system, s64 ns) {
    SleepThreadImpl(system, ns, false);
}

/// Gets the thread context
Result GetThreadContext3(Core::System& system, u64 out_context, Handle thread_handle) {
    LOG_DEBUG(Kernel_SVC, "called, out_context={:#08x}, thread_handle={:#x}", out_context, thread_handle);

    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Require the handle be to a non-current thread in the current process.
    R_UNLESS(thread->GetOwnerProcess() == GetCurrentProcessPointer(system.Kernel()), ResultInvalidHandle);
    R_UNLESS(thread.GetPointerUnsafe() != GetCurrentThreadPointer(system.Kernel()), ResultBusy);

    // Get the thread context.
    Svc::ThreadContext context{};
    R_TRY(thread->GetThreadContext3(system.Kernel(), std::addressof(context)));

    // Copy the thread context to user space.
    R_UNLESS(
        GetCurrentMemory(system.Kernel()).WriteBlock(out_context, std::addressof(context), sizeof(context)),
        ResultInvalidPointer);

    R_SUCCEED();
}

/// Gets the priority for the specified thread
Result GetThreadPriority(Core::System& system, s32* out_priority, Handle handle) {
    LOG_TRACE(Kernel_SVC, "called");

    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Get the thread's priority.
    *out_priority = thread->GetPriority();
    R_SUCCEED();
}

/// Sets the priority for the specified thread
Result SetThreadPriority(Core::System& system, Handle thread_handle, s32 priority) {
    // Get the current process.
    auto& kernel = system.Kernel();
    KProcess& process = GetCurrentProcess(kernel);

    // Validate the priority.
    R_UNLESS(HighestThreadPriority <= priority && priority <= LowestThreadPriority,
             ResultInvalidPriority);
    R_UNLESS(process.CheckThreadPriority(priority), ResultInvalidPriority);

    // Get the thread from its handle.
    KScopedAutoObject thread = process.GetHandleTable().GetObject<KThread>(kernel, thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    const s32 old_base_priority = thread->GetBasePriority();

    // Set the thread priority.
    thread->SetBasePriority(kernel, priority);

    if (KProcess* const owner = thread->GetOwnerProcess();
        owner != nullptr && OotAffinityMode(*owner) == OotAffinity::Dedicated &&
        Core::Hardware::NUM_CPU_CORES >= 8 && old_base_priority != priority) {
        s32 virtual_ideal{};
        u64 virtual_mask{};
        if (R_SUCCEEDED(thread->GetCoreMask(kernel, &virtual_ideal, &virtual_mask)) &&
            virtual_mask != 0) {
            s32 phys_ideal = -3;
            u64 phys_mask = 0;
            const bool is_main = IsProcessMainThread(*owner, *thread);
            ComputeOotDedicatedOverrides(virtual_mask, thread->GetBasePriority(), is_main,
                                         phys_ideal, phys_mask);
            static_cast<void>(thread->SetCoreMask(kernel, virtual_ideal, virtual_mask, phys_ideal,
                                                  phys_mask));
        }
    }
    R_SUCCEED();
}

Result GetThreadList(Core::System& system, s32* out_num_threads, u64 out_thread_ids,
                     s32 out_thread_ids_size, Handle debug_handle) {
    // TODO: Handle this case when debug events are supported.
    UNIMPLEMENTED_IF(debug_handle != InvalidHandle);

    LOG_DEBUG(Kernel_SVC, "called. out_thread_ids={:#016x}, out_thread_ids_size={}",
              out_thread_ids, out_thread_ids_size);

    // If the size is negative or larger than INT32_MAX / sizeof(u64)
    if ((out_thread_ids_size & 0xF0000000) != 0) {
        LOG_ERROR(Kernel_SVC, "Supplied size outside [0, 0x0FFFFFFF] range. size={}",
                  out_thread_ids_size);
        R_THROW(ResultOutOfRange);
    }

    auto* const current_process = GetCurrentProcessPointer(system.Kernel());
    const auto total_copy_size = out_thread_ids_size * sizeof(u64);

    if (out_thread_ids_size > 0 &&
        !current_process->GetPageTable().Contains(out_thread_ids, total_copy_size)) {
        LOG_ERROR(Kernel_SVC, "Address range outside address space. begin={:#016x}, end={:#016x}",
                  out_thread_ids, out_thread_ids + total_copy_size);
        R_THROW(ResultInvalidCurrentMemory);
    }

    auto& memory = GetCurrentMemory(system.Kernel());
    const auto& thread_list = current_process->GetThreadList();
    const auto num_threads = thread_list.size();
    const auto copy_amount = (std::min)(static_cast<std::size_t>(out_thread_ids_size), num_threads);

    auto list_iter = thread_list.cbegin();
    for (std::size_t i = 0; i < copy_amount; ++i, ++list_iter) {
        memory.Write64(out_thread_ids, list_iter->GetThreadId());
        out_thread_ids += sizeof(u64);
    }

    *out_num_threads = static_cast<u32>(num_threads);
    R_SUCCEED();
}

Result GetThreadCoreMask(Core::System& system, s32* out_core_id, u64* out_affinity_mask,
                         Handle thread_handle) {
    LOG_TRACE(Kernel_SVC, "called, handle={:#08x}", thread_handle);

    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Get the core mask.
    R_RETURN(thread->GetCoreMask(system.Kernel(), out_core_id, out_affinity_mask));
}

Result SetThreadCoreMask(Core::System& system, Handle thread_handle, s32 core_id,
                         u64 affinity_mask) {
    auto& process = GetCurrentProcess(system.Kernel());
    KScopedAutoObject thread =
        process.GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    KProcess* const owner = thread->GetOwnerProcess();
    s32 phys_ideal_override = -3;
    u64 phys_affinity_override = 0;
    // Determine the core id/affinity mask.
    if (core_id == IdealCoreUseProcessValue) {
        core_id = process.GetIdealCoreId();
        affinity_mask = (1ULL << core_id);
        if (owner != nullptr && OotAffinityMode(*owner) == OotAffinity::Dedicated &&
            Core::Hardware::NUM_CPU_CORES >= 8) {
            const bool is_main = IsProcessMainThread(*owner, *thread);
            ComputeOotDedicatedOverrides(affinity_mask, thread->GetBasePriority(), is_main,
                                         phys_ideal_override, phys_affinity_override);
        }
    } else {
        // Validate the affinity mask.
        const u64 process_core_mask = process.GetCoreMask();
        R_UNLESS((affinity_mask | process_core_mask) == process_core_mask, ResultInvalidCoreId);
        R_UNLESS(affinity_mask != 0, ResultInvalidCombination);

        const u64 requested_mask = affinity_mask;
        const OotAffinity mode = owner != nullptr ? OotAffinityMode(*owner) : OotAffinity::Game;
        if (mode == OotAffinity::Dedicated && Core::Hardware::NUM_CPU_CORES >= 8) {
            const bool is_main = IsProcessMainThread(*owner, *thread);
            ComputeOotDedicatedOverrides(requested_mask, thread->GetBasePriority(), is_main,
                                         phys_ideal_override, phys_affinity_override);
        } else if (mode == OotAffinity::Wide && std::popcount(requested_mask) > 1) {
            affinity_mask = process_core_mask & ((1ULL << Core::Hardware::NUM_CPU_CORES) - 1);
            if (core_id >= 3 && core_id < 6) {
                const s32 remapped = 1 + ((core_id - 3) % 2);
                if (((1ULL << remapped) & affinity_mask) != 0) {
                    core_id = remapped;
                }
            }
        } else if (mode == OotAffinity::Ideal && std::popcount(requested_mask) > 1 &&
                   IsValidProcessCoreId(process, core_id) &&
                   ((1ULL << core_id) & affinity_mask) != 0) {
            affinity_mask = 1ULL << core_id;
        }

        // Validate the core id.
        if (IsValidProcessCoreId(process, core_id)) {
            R_UNLESS(((1ULL << core_id) & affinity_mask) != 0, ResultInvalidCombination);
        } else {
            R_UNLESS(core_id == IdealCoreNoUpdate || core_id == IdealCoreDontCare,
                     ResultInvalidCoreId);
        }
    }

    // Set the core mask.
    R_RETURN(thread->SetCoreMask(system.Kernel(), core_id, affinity_mask, phys_ideal_override,
                                 phys_affinity_override));
}

/// Get the ID for the specified thread.
Result GetThreadId(Core::System& system, u64* out_thread_id, Handle thread_handle) {
    // Get the thread from its handle.
    KScopedAutoObject thread = GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Get the thread's id.
    *out_thread_id = thread->GetId();
    R_SUCCEED();
}

Result CreateThread64(Core::System& system, Handle* out_handle, uint64_t func, uint64_t arg,
                      uint64_t stack_bottom, int32_t priority, int32_t core_id) {
    R_RETURN(CreateThread(system, out_handle, func, arg, stack_bottom, priority, core_id));
}

Result StartThread64(Core::System& system, Handle thread_handle) {
    R_RETURN(StartThread(system, thread_handle));
}

void ExitThread64(Core::System& system) {
    return ExitThread(system);
}

void SleepThread64(Core::System& system, int64_t ns) {
    SleepThreadImpl(system, ns, true);
}

Result GetThreadPriority64(Core::System& system, int32_t* out_priority, Handle thread_handle) {
    R_RETURN(GetThreadPriority(system, out_priority, thread_handle));
}

Result SetThreadPriority64(Core::System& system, Handle thread_handle, int32_t priority) {
    R_RETURN(SetThreadPriority(system, thread_handle, priority));
}

Result GetThreadCoreMask64(Core::System& system, int32_t* out_core_id, uint64_t* out_affinity_mask,
                           Handle thread_handle) {
    R_RETURN(GetThreadCoreMask(system, out_core_id, out_affinity_mask, thread_handle));
}

Result SetThreadCoreMask64(Core::System& system, Handle thread_handle, int32_t core_id,
                           uint64_t affinity_mask) {
    R_RETURN(SetThreadCoreMask(system, thread_handle, core_id, affinity_mask));
}

Result GetThreadId64(Core::System& system, uint64_t* out_thread_id, Handle thread_handle) {
    R_RETURN(GetThreadId(system, out_thread_id, thread_handle));
}

Result GetThreadContext364(Core::System& system, uint64_t out_context, Handle thread_handle) {
    R_RETURN(GetThreadContext3(system, out_context, thread_handle));
}

Result GetThreadList64(Core::System& system, int32_t* out_num_threads, uint64_t out_thread_ids,
                       int32_t max_out_count, Handle debug_handle) {
    R_RETURN(GetThreadList(system, out_num_threads, out_thread_ids, max_out_count, debug_handle));
}

Result CreateThread64From32(Core::System& system, Handle* out_handle, uint32_t func, uint32_t arg,
                            uint32_t stack_bottom, int32_t priority, int32_t core_id) {
    R_RETURN(CreateThread(system, out_handle, func, arg, stack_bottom, priority, core_id));
}

Result StartThread64From32(Core::System& system, Handle thread_handle) {
    R_RETURN(StartThread(system, thread_handle));
}

void ExitThread64From32(Core::System& system) {
    return ExitThread(system);
}

void SleepThread64From32(Core::System& system, int64_t ns) {
    SleepThreadImpl(system, ns, true);
}

Result GetThreadPriority64From32(Core::System& system, int32_t* out_priority,
                                 Handle thread_handle) {
    R_RETURN(GetThreadPriority(system, out_priority, thread_handle));
}

Result SetThreadPriority64From32(Core::System& system, Handle thread_handle, int32_t priority) {
    R_RETURN(SetThreadPriority(system, thread_handle, priority));
}

Result GetThreadCoreMask64From32(Core::System& system, int32_t* out_core_id,
                                 uint64_t* out_affinity_mask, Handle thread_handle) {
    R_RETURN(GetThreadCoreMask(system, out_core_id, out_affinity_mask, thread_handle));
}

Result SetThreadCoreMask64From32(Core::System& system, Handle thread_handle, int32_t core_id,
                                 uint64_t affinity_mask) {
    R_RETURN(SetThreadCoreMask(system, thread_handle, core_id, affinity_mask));
}

Result GetThreadId64From32(Core::System& system, uint64_t* out_thread_id, Handle thread_handle) {
    R_RETURN(GetThreadId(system, out_thread_id, thread_handle));
}

Result GetThreadContext364From32(Core::System& system, uint32_t out_context, Handle thread_handle) {
    R_RETURN(GetThreadContext3(system, out_context, thread_handle));
}

Result GetThreadList64From32(Core::System& system, int32_t* out_num_threads,
                             uint32_t out_thread_ids, int32_t max_out_count, Handle debug_handle) {
    R_RETURN(GetThreadList(system, out_num_threads, out_thread_ids, max_out_count, debug_handle));
}

} // namespace Kernel::Svc
