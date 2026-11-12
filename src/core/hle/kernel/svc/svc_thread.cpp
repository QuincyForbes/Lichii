// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <thread>

#include "common/perf_report.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_resource_reservation.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/svc.h"
#include "core/hle/kernel/thread_startup_policy.h"

namespace Kernel::Svc {
namespace {

constexpr bool IsValidVirtualCoreId(int32_t core_id) {
    return (0 <= core_id && core_id < static_cast<int32_t>(Core::Hardware::NUM_CPU_CORES));
}

// Switch 2 hardware has eight cores. Six are emulated (the most an application is given), so
// fold any id past them back onto the low cores, both in core ids and in affinity masks.
constexpr int32_t FoldExtraCoreId(int32_t core_id) {
    constexpr auto num_cores = static_cast<int32_t>(Core::Hardware::NUM_CPU_CORES);
    return core_id >= num_cores && core_id < 2 * num_cores ? core_id - num_cores : core_id;
}

constexpr u64 FoldExtraCoreMask(u64 mask) {
    constexpr u64 low = (1ULL << Core::Hardware::NUM_CPU_CORES) - 1;
    return (mask & low) | ((mask >> Core::Hardware::NUM_CPU_CORES) & low);
}

struct YieldBackoff {
    std::optional<bool> forced; ///< From the environment; otherwise the guest_wait_backoff setting
    u64 gate_ns; ///< Yielding to nobody this long before the first sleep
    u64 min_ns;
    u64 cap_ns;
};

/// EDEN_NX2_YIELD_BACKOFF: 0 off, 1 on, or <gate us>:<cap us>; unset follows guest_wait_backoff.
const YieldBackoff& YieldBackoffSetting() {
    static const YieldBackoff setting = [] {
        YieldBackoff result{std::nullopt, 2'000'000, 10'000, 200'000};
        const char* value = std::getenv("EDEN_NX2_YIELD_BACKOFF");
        if (value == nullptr || value[0] == '\0') {
            return result;
        }
        result.forced = value[0] != '0';
        if (const char* colon = std::strchr(value, ':')) {
            result.gate_ns = std::strtoull(value, nullptr, 10) * 1000;
            result.cap_ns = (std::max)(u64{10}, std::strtoull(colon + 1, nullptr, 10)) * 1000;
        }
        LOG_INFO(Kernel_SVC, "Yield backoff: {} (after {} us of yields to nobody, {}-{} us)",
                 *result.forced ? "on" : "off", result.gate_ns / 1000, result.min_ns / 1000,
                 result.cap_ns / 1000);
        return result;
    }();
    return setting;
}

/// On the Switch a yield with nothing else to run costs nothing, so guest code waits for other
/// threads by yielding in a loop. Here each pass is a host round trip that keeps a host core busy
/// (on a phone, one of its two performance cores). A thread that has yielded to nobody for the
/// whole gate with the same callee-saved registers (a poll keeps its state there; a loop that
/// works between yields changes them) sleeps an eighth of the run so far instead: its guest core
/// idles, and anything that becomes runnable there runs at once.
void YieldBackoffAfterYield(Core::System& system) {
    auto& kernel = system.Kernel();
    KThread& thread = GetCurrentThread(kernel);
    KProcess& process = GetCurrentProcess(kernel);
    if (thread.GetYieldScheduleCount() != process.GetScheduledCount()) {
        // Another thread runs next: a real yield.
        thread.m_nx2_yield_run_start = 0;
        return;
    }
    Common::PerfReport::Add(Common::PerfReport::GuestFutileYields);
    if (!YieldBackoffEnabled()) {
        return;
    }
    const YieldBackoff& setting = YieldBackoffSetting();
    const auto steady_ns = [] {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    };
    const u64 now = steady_ns();
    Svc::ThreadContext context{};
    if (auto* arm = process.GetArmInterface(kernel.CurrentPhysicalCoreIndex())) {
        arm->GetContext(context);
    }
    u64 regs = 0;
    for (size_t i = 19; i <= 28; ++i) {
        regs = (regs ^ context.r[i]) * 0x100000001B3ULL;
    }
    constexpr u64 MaxGapNs = 100'000;
    if (thread.m_nx2_yield_run_start == 0 || now - thread.m_nx2_yield_last > MaxGapNs ||
        regs != thread.m_nx2_yield_regs) {
        thread.m_nx2_yield_run_start = now;
        thread.m_nx2_yield_last = now;
        thread.m_nx2_yield_regs = regs;
        return;
    }
    thread.m_nx2_yield_last = now;
    const u64 run = now - thread.m_nx2_yield_run_start;
    if (run < setting.gate_ns) {
        return;
    }
    const u64 sleep_ns = std::clamp(run / 8, setting.min_ns, setting.cap_ns);
    s64 timeout = kernel.HardwareTimer().GetTick() + static_cast<s64>(sleep_ns) + 2;
    if (timeout <= 0) {
        timeout = (std::numeric_limits<s64>::max)();
    }
    static_cast<void>(thread.Sleep(kernel, timeout));
    const u64 after = steady_ns();
    thread.m_nx2_yield_last = after;
    Common::PerfReport::Add(Common::PerfReport::GuestYieldSleeps);
    Common::PerfReport::Add(Common::PerfReport::GuestYieldSleepNs, after - now);
}

// Ocarina of Time's two main threads (priorities 55 and 47) may only run on two of cores 3-5,
// where the per-core job workers (priority 44) keep preempting them while cores 0-2 sit idle
// about 30% of the time. Any mask of more than one core is widened to all of the process's cores,
// so an idle core can take a preempted main thread. Pinned (single-core) threads are unchanged.
// EDEN_NX2_OOT_WIDE_AFFINITY=0 turns it off.
bool WidenAffinity(const KProcess& process, u64 affinity_mask) {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_OOT_WIDE_AFFINITY");
        return value == nullptr || value[0] != '0';
    }();
    return enabled && std::popcount(affinity_mask) > 1 &&
           process.GetProgramId() == 0x0400486024DEC000ULL;
}

} // Anonymous namespace

bool YieldBackoffEnabled() {
    const YieldBackoff& setting = YieldBackoffSetting();
    return setting.forced ? *setting.forced : Settings::values.guest_wait_backoff.GetValue();
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
    core_id = FoldExtraCoreId(core_id);

    // Validate arguments.
    R_UNLESS(IsValidVirtualCoreId(core_id), ResultInvalidCoreId);
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
                std::this_thread::yield();
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
void SleepThread(Core::System& system, s64 ns) {
    auto& kernel = system.Kernel();
    const auto yield_type = static_cast<Svc::YieldType>(ns);

    LOG_TRACE(Kernel_SVC, "called nanoseconds={}", ns);

    // When the input tick is positive, sleep.
    if (ns > 0) {
        // Convert the timeout from nanoseconds to ticks.
        // NOTE: Nintendo does not use this conversion logic in WaitSynchronization...
        s64 timeout;

        const s64 offset_tick(ns);
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
        static_cast<void>(GetCurrentThread(kernel).Sleep(kernel, timeout));
    } else if (yield_type == Svc::YieldType::WithoutCoreMigration) {
        KScheduler::YieldWithoutCoreMigration(kernel);
        YieldBackoffAfterYield(system);
    } else if (yield_type == Svc::YieldType::WithCoreMigration) {
        KScheduler::YieldWithCoreMigration(kernel);
        YieldBackoffAfterYield(system);
    } else if (yield_type == Svc::YieldType::ToAnyThread) {
        KScheduler::YieldToAnyThread(kernel);
        YieldBackoffAfterYield(system);
    } else {
        // Nintendo does nothing at all if an otherwise invalid value is passed.
    }
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
    KProcess& process = GetCurrentProcess(system.Kernel());

    // Validate the priority.
    R_UNLESS(HighestThreadPriority <= priority && priority <= LowestThreadPriority,
             ResultInvalidPriority);
    R_UNLESS(process.CheckThreadPriority(priority), ResultInvalidPriority);

    // Get the thread from its handle.
    KScopedAutoObject thread = process.GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Set the thread priority.
    thread->SetBasePriority(system.Kernel(), priority);
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
    // Determine the core id/affinity mask.
    if (core_id == IdealCoreUseProcessValue) {
        core_id = GetCurrentProcess(system.Kernel()).GetIdealCoreId();
        affinity_mask = (1ULL << core_id);
    } else {
        // Validate the affinity mask.
        const u64 process_core_mask = GetCurrentProcess(system.Kernel()).GetCoreMask();
        R_UNLESS((affinity_mask | process_core_mask) == process_core_mask, ResultInvalidCoreId);
        R_UNLESS(affinity_mask != 0, ResultInvalidCombination);

        // Validated against the real mask above; fold to the emulated cores from here on.
        affinity_mask = FoldExtraCoreMask(affinity_mask);
        core_id = FoldExtraCoreId(core_id);
        if (WidenAffinity(GetCurrentProcess(system.Kernel()), affinity_mask)) {
            affinity_mask = FoldExtraCoreMask(process_core_mask);
        }

        // Validate the core id.
        if (IsValidVirtualCoreId(core_id)) {
            R_UNLESS(((1ULL << core_id) & affinity_mask) != 0, ResultInvalidCombination);
        } else {
            R_UNLESS(core_id == IdealCoreNoUpdate || core_id == IdealCoreDontCare,
                     ResultInvalidCoreId);
        }
    }

    // Get the thread from its handle.
    KScopedAutoObject thread =
        GetCurrentProcess(system.Kernel()).GetHandleTable().GetObject<KThread>(system.Kernel(), thread_handle);
    R_UNLESS(thread.IsNotNull(), ResultInvalidHandle);

    // Set the core mask.
    R_RETURN(thread->SetCoreMask(system.Kernel(), core_id, affinity_mask));
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
    return SleepThread(system, ns);
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
    return SleepThread(system, ns);
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
