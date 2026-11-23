// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <bit>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/assert.h"
#include "common/bit_util.h"
#include "common/fiber.h"
#include "common/logging.h"
#include "common/perf_sched.h"
#include "core/arm/arm_interface.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/cpu_manager.h"
#include "core/hle/kernel/k_interrupt_manager.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"

namespace Kernel {

static void IncrementScheduledCount(Kernel::KThread* thread) {
    if (auto process = thread->GetOwnerProcess(); process) {
        process->IncrementScheduledCount();
    }
}

static void SpinPause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

u32 KScheduler::GetCoreScheduledCount(KernelCore& kernel, s32 core_id) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        return 0;
    }
    return GetPriorityQueue(kernel).GetScheduledCount(core_id);
}

u32 KScheduler::GetCoreSuggestedCount(KernelCore& kernel, s32 core_id) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        return 0;
    }
    return GetPriorityQueue(kernel).GetSuggestedCount(core_id);
}

u64 KScheduler::GetCoreScheduledQueueVersion(KernelCore& kernel, s32 core_id) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        return 0;
    }
    return GetPriorityQueue(kernel).GetScheduledQueueVersion(core_id);
}

bool KScheduler::HasStarvedRunnableForCore(KernelCore& kernel, s32 core_id) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        return false;
    }
    return (kernel.Scheduler(static_cast<size_t>(core_id)).GetYieldState() &
            kYieldStateStarvedCandidate) != 0;
}

bool KScheduler::IsYieldUncontended(KernelCore& kernel, const KProcess& process, s32 core_id) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        return false;
    }
    if (process.HasAnyPinnedThread()) {
        return false;
    }
    const u32 ys = kernel.Scheduler(static_cast<size_t>(core_id)).GetYieldState();
    return ys == kInitialYieldState;
}

bool KScheduler::IsCoreYieldFutile(KernelCore& kernel, const KProcess& process,
                                   const KThread& thread, s32 core_id, Svc::YieldType yield_type,
                                   u32* out_yield_state) {
    if (core_id < 0 || core_id >= static_cast<s32>(Core::Hardware::NUM_CPU_CORES)) {
        if (out_yield_state != nullptr) {
            *out_yield_state = kInitialYieldState;
        }
        return false;
    }
    const auto& sched = kernel.Scheduler(static_cast<size_t>(core_id));
    const u32 ys = sched.GetYieldState();
    if (out_yield_state != nullptr) {
        *out_yield_state = ys;
    }
    if (process.HasAnyPinnedThread() || sched.NeedsScheduling() ||
        (ys & kYieldStateActiveHandoff) != 0 ||
        sched.GetHighestPriorityThread() != std::addressof(thread)) {
        return false;
    }
    const s32 prio = thread.GetPriority();
    const s32 local_prio = static_cast<s32>(ys & 0xFFU);
    const s32 mig_prio = static_cast<s32>((ys >> 8) & 0xFFU);
    if (yield_type == Svc::YieldType::WithoutCoreMigration) {
        return local_prio > prio;
    }
    if (yield_type == Svc::YieldType::WithCoreMigration) {
        return local_prio > prio && mig_prio > prio;
    }
    return local_prio >= kNoWaiterPriority && mig_prio > prio;
}

bool KScheduler::TryHandoffStarvedOnYield(KernelCore& kernel, u64 lease_ns) {
    ASSERT(CanSchedule(kernel));
    KThread& cur_thread = GetCurrentThread(kernel);
    const u64 now_ns = Common::PerfReport::NowNs();

    KScopedSchedulerLock sl{kernel};
    if (cur_thread.GetRawState() != ThreadState::Runnable ||
        cur_thread.GetNumKernelWaiters() > 0 || cur_thread.IsTerminationRequested()) {
        return false;
    }
    const s32 core_id = cur_thread.GetActiveCore();
    if (core_id < 0) {
        return false;
    }
    auto& pq = GetPriorityQueue(kernel);

    bool has_local_waiter = false;
    for (KThread* next = pq.GetScheduledNext(core_id, std::addressof(cur_thread)); next != nullptr;
         next = pq.GetScheduledNext(core_id, next)) {
        if (next->m_nx2_yield_handoff_until_ns <= now_ns) {
            has_local_waiter = true;
            break;
        }
    }

    if (!has_local_waiter) {
        KThread* migrated = nullptr;
        for (KThread* s = pq.GetSuggestedFront(core_id); s != nullptr;
             s = pq.GetSuggestedNext(core_id, s)) {
            if (s->m_nx2_yield_handoff_until_ns > now_ns) {
                continue;
            }
            const s32 s_core = s->GetActiveCore();
            KThread* const running_on_s =
                (s_core >= 0) ? kernel.Scheduler(static_cast<size_t>(s_core)).GetHighestPriorityThread()
                              : nullptr;
            if (running_on_s != s &&
                (running_on_s == nullptr ||
                 running_on_s->GetPriority() >= HighestCoreMigrationAllowedPriority)) {
                s->SetActiveCore(core_id);
                pq.ChangeCore(s_core, s);
                IncrementScheduledCount(s);
                migrated = s;
                break;
            }
        }
        if (migrated == nullptr) {
            return false;
        }
    }

    cur_thread.m_nx2_yield_handoff_until_ns = now_ns + lease_ns;
    IncrementScheduledCount(std::addressof(cur_thread));
    SetSchedulerUpdateNeeded(kernel);
    return true;
}

void KScheduler::RefreshExpiredHandoff(KernelCore& kernel) {
    auto* const sched = kernel.CurrentScheduler();
    if (sched == nullptr) {
        return;
    }
    const u64 exp = sched->GetHandoffExpiryNs();
    if (exp == 0 || Common::PerfReport::NowNs() < exp) {
        return;
    }
    KScopedSchedulerLock sl{kernel};
    if (const u64 recheck = sched->GetHandoffExpiryNs();
        recheck != 0 && Common::PerfReport::NowNs() >= recheck) {
        SetSchedulerUpdateNeeded(kernel);
    }
}

KScheduler::KScheduler(KernelCore& kernel) {
    m_switch_fiber = std::make_shared<Common::Fiber>([this, &kernel] {
        while (true) {
            ScheduleImplFiber(kernel);
        }
    });
    m_state.needs_scheduling = true;
}

KScheduler::~KScheduler() = default;

void KScheduler::SetInterruptTaskRunnable(KernelCore& kernel) {
    m_state.interrupt_task_runnable = true;
    m_state.needs_scheduling = true;
}

void KScheduler::RequestScheduleOnInterrupt(KernelCore& kernel) {
    m_state.needs_scheduling = true;
    if (CanSchedule(kernel)) {
        ScheduleOnInterrupt(kernel);
    }
}

void KScheduler::DisableScheduling(KernelCore& kernel) {
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() >= 0);
    GetCurrentThread(kernel).DisableDispatch(kernel);
}

void KScheduler::EnableScheduling(KernelCore& kernel, u64 cores_needing_scheduling) {
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() >= 1);

    auto* scheduler{kernel.CurrentScheduler()};

    if (!scheduler || kernel.IsPhantomModeForSingleCore()) {
        KScheduler::RescheduleCores(kernel, cores_needing_scheduling);
        KScheduler::RescheduleCurrentHLEThread(kernel);
        return;
    }

    scheduler->RescheduleOtherCores(kernel, cores_needing_scheduling);

    if (GetCurrentThread(kernel).GetDisableDispatchCount() > 1) {
        GetCurrentThread(kernel).EnableDispatch(kernel);
    } else {
        scheduler->RescheduleCurrentCore(kernel);
    }
}

void KScheduler::RescheduleCurrentHLEThread(KernelCore& kernel) {
    // HACK: we cannot schedule from this thread, it is not a core thread
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() == 1);

    // Ensure dummy threads that are waiting block.
    GetCurrentThread(kernel).DummyThreadBeginWait(kernel);

    ASSERT(GetCurrentThread(kernel).GetState() != ThreadState::Waiting);
    GetCurrentThread(kernel).EnableDispatch(kernel);
}

u64 KScheduler::UpdateHighestPriorityThreads(KernelCore& kernel) {
    if (IsSchedulerUpdateNeeded(kernel)) {
        return UpdateHighestPriorityThreadsImpl(kernel);
    }
    const u32 mask =
        kernel.GlobalSchedulerContext().m_active_handoff_mask.load(std::memory_order_relaxed);
    if (mask != 0) {
        const u64 now = Common::PerfReport::NowNs();
        u32 bits = mask;
        while (bits != 0) {
            const u32 c = static_cast<u32>(std::countr_zero(bits));
            bits &= bits - 1;
            const u64 exp = kernel.Scheduler(c).GetHandoffExpiryNs();
            if (exp != 0 && now >= exp) {
                return UpdateHighestPriorityThreadsImpl(kernel);
            }
        }
    }
    return 0;
}

void KScheduler::Schedule(KernelCore& kernel) {
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() == 1);
    ASSERT(m_core_id == GetCurrentCoreId(kernel));

    ScheduleImpl(kernel);
}

void KScheduler::ScheduleOnInterrupt(KernelCore& kernel) {
    GetCurrentThread(kernel).DisableDispatch(kernel);
    Schedule(kernel);
    GetCurrentThread(kernel).EnableDispatch(kernel);
}

void KScheduler::PreemptSingleCore(KernelCore& kernel) {
    GetCurrentThread(kernel).DisableDispatch(kernel);

    auto* thread = GetCurrentThreadPointer(kernel);
    auto& previous_scheduler = kernel.Scheduler(thread->GetCurrentCore());
    previous_scheduler.Unload(kernel, thread);

    Common::Fiber::YieldTo(thread->GetHostContext(), *m_switch_fiber);

    GetCurrentThread(kernel).EnableDispatch(kernel);
}

void KScheduler::RescheduleCurrentCore(KernelCore& kernel) {
    ASSERT(!kernel.IsPhantomModeForSingleCore());
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() == 1);

    GetCurrentThread(kernel).EnableDispatch(kernel);

    if (m_state.needs_scheduling.load(std::memory_order_acquire)) {
        // Disable interrupts, and then check again if rescheduling is needed.
        // KScopedInterruptDisable intr_disable;

        this->RescheduleCurrentCoreImpl(kernel);
    }
}

void KScheduler::RescheduleCurrentCoreImpl(KernelCore& kernel) {
    // Check that scheduling is needed.
    if (m_state.needs_scheduling.load(std::memory_order_acquire)) [[likely]] {
        GetCurrentThread(kernel).DisableDispatch(kernel);
        Schedule(kernel);
        GetCurrentThread(kernel).EnableDispatch(kernel);
    }
}

void KScheduler::Initialize(KernelCore& kernel, KThread* main_thread, KThread* idle_thread, s32 core_id) {
    // Set core ID/idle thread/interrupt task manager.
    m_core_id = core_id;
    m_idle_thread = idle_thread;
    // m_state.idle_thread_stack = m_idle_thread->GetStackTop();
    // m_state.interrupt_task_manager = std::addressof(kernel.GetInterruptTaskManager());

    // Insert the main thread into the priority queue.
    // {
    //     KScopedSchedulerLock lk{kernel};
    //     GetPriorityQueue(kernel).PushBack(GetCurrentThreadPointer(kernel));
    //     SetSchedulerUpdateNeeded(kernel);
    // }

    // Bind interrupt handler.
    // kernel.GetInterruptManager().BindHandler(
    //     GetSchedulerInterruptHandler(kernel), KInterruptName::Scheduler, m_core_id,
    //     KInterruptController::PriorityLevel::Scheduler, false, false);

    // Set the current thread.
    m_current_thread = main_thread;
}

void KScheduler::Activate(KernelCore& kernel) {
    ASSERT(GetCurrentThread(kernel).GetDisableDispatchCount() == 1);

    // m_state.should_count_idle = KTargetSystem::IsDebugMode();
    m_is_active = true;
    RescheduleCurrentCore(kernel);
}

void KScheduler::OnThreadStart(KernelCore& kernel) {
    GetCurrentThread(kernel).EnableDispatch(kernel);
}

u64 KScheduler::UpdateHighestPriorityThread(KernelCore& kernel, KThread* highest_thread) {
    if (KThread* prev_highest_thread = GetHighestPriorityThread();
        prev_highest_thread != highest_thread) [[likely]] {
        if (prev_highest_thread != nullptr) [[likely]] {
            IncrementScheduledCount(prev_highest_thread);
            prev_highest_thread->SetLastScheduledTick(
                kernel.System().CoreTiming().GetClockTicks());
        }
        if (m_state.should_count_idle) {
            if (highest_thread != nullptr) [[likely]] {
                if (KProcess* process = highest_thread->GetOwnerProcess(); process != nullptr) {
                    process->SetRunningThread(m_core_id, highest_thread, m_state.idle_count, 0);
                }
            } else {
                m_state.idle_count++;
            }
        }

        m_state.highest_priority_thread.store(highest_thread, std::memory_order_relaxed);
        m_state.needs_scheduling.store(true, std::memory_order_relaxed);
        return (1ULL << m_core_id);
    } else {
        return 0;
    }
}

u64 KScheduler::UpdateHighestPriorityThreadsImpl(KernelCore& kernel) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));

    // Clear that we need to update.
    ClearSchedulerUpdateNeeded(kernel);

    u64 cores_needing_scheduling = 0, idle_cores = 0;
    u32 new_handoff_mask = 0;
    u64 now_ns = 0;
    std::array<KThread*, Core::Hardware::NUM_CPU_CORES> top_threads{};
    std::array<u64, Core::Hardware::NUM_CPU_CORES> core_expiries{};
    auto& priority_queue = GetPriorityQueue(kernel);

    // We want to go over all cores, finding the highest priority thread and determining if
    // scheduling is needed for that core.
    for (size_t core_id = 0; core_id < Core::Hardware::NUM_CPU_CORES; core_id++) {
        KThread* top_thread = priority_queue.GetScheduledFront(static_cast<s32>(core_id));
        u64 core_expiry_ns = 0;
        if (top_thread != nullptr) {
            if (top_thread->m_nx2_yield_handoff_until_ns != 0) {
                if (now_ns == 0) {
                    now_ns = Common::PerfReport::NowNs();
                }
                KThread* candidate = top_thread;
                while (candidate != nullptr && candidate->m_nx2_yield_handoff_until_ns != 0) {
                    if (now_ns >= candidate->m_nx2_yield_handoff_until_ns ||
                        candidate->GetNumKernelWaiters() > 0 ||
                        candidate->IsTerminationRequested()) {
                        candidate->m_nx2_yield_handoff_until_ns = 0;
                        break;
                    }
                    KThread* const next =
                        priority_queue.GetScheduledNext(static_cast<s32>(core_id), candidate);
                    if (next == nullptr) {
                        for (KThread* clr = top_thread; clr != nullptr;
                             clr = priority_queue.GetScheduledNext(static_cast<s32>(core_id), clr)) {
                            clr->m_nx2_yield_handoff_until_ns = 0;
                        }
                        candidate = top_thread;
                        core_expiry_ns = 0;
                        break;
                    }
                    if (core_expiry_ns == 0 ||
                        candidate->m_nx2_yield_handoff_until_ns < core_expiry_ns) {
                        core_expiry_ns = candidate->m_nx2_yield_handoff_until_ns;
                    }
                    candidate = next;
                }
                top_thread = candidate;
            }

            // We need to check if the thread's process has a pinned thread.
            if (KProcess* parent = top_thread->GetOwnerProcess()) {
                // Check that there's a pinned thread other than the current top thread.
                if (KThread* pinned = parent->GetPinnedThread(static_cast<s32>(core_id));
                    pinned != nullptr && pinned != top_thread) {
                    // We need to prefer threads with kernel waiters to the pinned thread.
                    if (top_thread->GetNumKernelWaiters() ==
                        0 /* && top_thread != parent->GetExceptionThread() */) {
                        // If the pinned thread is runnable, use it.
                        if (pinned->GetRawState() == ThreadState::Runnable) {
                            top_thread = pinned;
                        } else {
                            top_thread = nullptr;
                        }
                    }
                }
            }
        } else {
            idle_cores |= (1ULL << core_id);
        }

        core_expiries[core_id] = core_expiry_ns;
        if (core_expiry_ns != 0) {
            new_handoff_mask |= (1U << core_id);
        }
        top_threads[core_id] = top_thread;
    }

    // Idle cores are bad. We're going to try to migrate threads to each idle core in turn,
    // preferring higher core IDs first so multi-core background workers stay off Core 0 when
    // higher cores are also idle (Djihads80/Eglantine#910), while reusing the thread's last
    // physical core when it is also among the non-zero idle cores.
    while (idle_cores != 0) {
        const s32 core_id = static_cast<s32>(Common::BitSize<u64>() - 1 - std::countl_zero(idle_cores));
        bool retry_core = false;

        if (KThread* suggested = priority_queue.GetSuggestedFront(core_id); suggested != nullptr) {
            s32 migration_candidates[Core::Hardware::NUM_CPU_CORES];
            size_t num_candidates = 0;

            // While we have a suggested thread, try to migrate it!
            while (suggested != nullptr) {
                // Check if the suggested thread is the top thread on its core.
                const s32 suggested_core = suggested->GetActiveCore();
                if (KThread* top_thread =
                        (suggested_core >= 0) ? top_threads[static_cast<size_t>(suggested_core)]
                                              : nullptr;
                    top_thread != suggested) {
                    // Make sure we're not dealing with threads too high priority for migration.
                    if (top_thread != nullptr &&
                        top_thread->GetPriority() < HighestCoreMigrationAllowedPriority) {
                        break;
                    }

                    s32 target_core = core_id;
                    if (const s32 last_core = suggested->GetCurrentCore();
                        last_core > 0 && last_core != core_id &&
                        (idle_cores & (1ULL << last_core)) != 0 &&
                        suggested->GetAffinityMask().GetAffinity(last_core) &&
                        priority_queue.GetSuggestedFront(last_core) == suggested) {
                        target_core = last_core;
                    }

                    // The suggested thread isn't bound to its core, so we can migrate it!
                    suggested->m_nx2_yield_handoff_until_ns = 0;
                    suggested->SetActiveCore(target_core);
                    priority_queue.ChangeCore(suggested_core, suggested);
                    top_threads[static_cast<size_t>(target_core)] = suggested;
                    if (target_core != core_id) {
                        idle_cores &= ~(1ULL << target_core);
                        retry_core = true;
                    }
                    break;
                }

                // Note this core as a candidate for migration.
                ASSERT(num_candidates < Core::Hardware::NUM_CPU_CORES);
                migration_candidates[num_candidates++] = suggested_core;
                suggested = priority_queue.GetSuggestedNext(core_id, suggested);
            }

            // If suggested is nullptr, we failed to migrate a specific thread. So let's try all our
            // candidate cores' top threads.
            if (suggested == nullptr) {
                for (size_t i = 0; i < num_candidates; i++) {
                    // Check if there's some other thread that can run on the candidate core.
                    const s32 candidate_core = migration_candidates[i];
                    suggested = top_threads[static_cast<size_t>(candidate_core)];
                    KThread* const front_on_candidate =
                        priority_queue.GetScheduledFront(candidate_core);
                    KThread* next_on_candidate_core =
                        (front_on_candidate != suggested)
                            ? front_on_candidate
                            : priority_queue.GetScheduledNext(candidate_core, suggested);
                    if (next_on_candidate_core != nullptr) {
                        if (front_on_candidate != suggested) {
                            for (KThread* clr = front_on_candidate;
                                 clr != nullptr && clr != suggested;
                                 clr = priority_queue.GetScheduledNext(candidate_core, clr)) {
                                clr->m_nx2_yield_handoff_until_ns = 0;
                            }
                            core_expiries[static_cast<size_t>(candidate_core)] = 0;
                            new_handoff_mask &= ~(1U << candidate_core);
                        }
                        // The candidate core can run some other thread! We'll migrate its current
                        // top thread to us.
                        top_threads[static_cast<size_t>(candidate_core)] = next_on_candidate_core;

                        // Perform the migration.
                        suggested->SetActiveCore(core_id);
                        priority_queue.ChangeCore(candidate_core, suggested);
                        top_threads[static_cast<size_t>(core_id)] = suggested;
                        break;
                    }
                }
            }
        }

        if (!retry_core) {
            idle_cores &= ~(1ULL << core_id);
        }
    }

    // Resolve cross-core priority inversions: if a runnable multi-core waiter is queued behind a
    // higher-priority thread on src_core while another allowed unpinned core runs a strictly
    // lower-priority thread, migrate the waiter to preempt that lower-priority core.
    for (s32 src_core = 0; src_core < static_cast<s32>(Core::Hardware::NUM_CPU_CORES); ++src_core) {
        if (priority_queue.GetScheduledCount(src_core) <= 1) {
            continue;
        }
        KThread* const src_top = top_threads[static_cast<size_t>(src_core)];
        if (src_top == nullptr || src_top->GetPriority() < HighestCoreMigrationAllowedPriority) {
            continue;
        }
        for (KThread* waiter = priority_queue.GetScheduledFront(src_core); waiter != nullptr;) {
            KThread* const next_waiter = priority_queue.GetScheduledNext(src_core, waiter);
            if (waiter == src_top || waiter->m_nx2_yield_handoff_until_ns != 0) {
                waiter = next_waiter;
                continue;
            }
            const u64 mask = waiter->GetAffinityMask().GetAffinityMask() & ~(1ULL << src_core);
            if (mask != 0) {
                const s32 preferred_core = waiter->GetCurrentCore();
                s32 best_dst = -1;
                s32 worst_prio = waiter->GetPriority();
                u64 avail = mask;
                while (avail != 0) {
                    const s32 dst = std::countr_zero(avail);
                    avail &= avail - 1;
                    KThread* const dst_top = top_threads[static_cast<size_t>(dst)];
                    if (dst_top == nullptr) {
                        continue;
                    }
                    if (KProcess* const parent = dst_top->GetOwnerProcess();
                        parent != nullptr && parent->GetPinnedThread(dst) != nullptr) {
                        continue;
                    }
                    const s32 dst_prio = dst_top->GetPriority();
                    if (dst_prio > worst_prio ||
                        (best_dst >= 0 && dst_prio == worst_prio &&
                         (dst == preferred_core ||
                          (best_dst != preferred_core && dst > best_dst)))) {
                        worst_prio = dst_prio;
                        best_dst = dst;
                    }
                }
                if (best_dst >= 0) {
                    waiter->SetActiveCore(best_dst);
                    priority_queue.ChangeCore(src_core, waiter);
                    IncrementScheduledCount(waiter);
                    top_threads[static_cast<size_t>(best_dst)] = waiter;
                }
            }
            waiter = next_waiter;
        }
    }

    // Summarize non-running runnable waiters per core so uncontended yields and pre-wait spins
    // check one L1D-local word on their own KScheduler instead of scanning queues across cores.
    std::array<u8, Core::Hardware::NUM_CPU_CORES> local_waiter_prio{};
    std::array<u8, Core::Hardware::NUM_CPU_CORES> migratable_prio{};
    local_waiter_prio.fill(static_cast<u8>(kNoWaiterPriority));
    migratable_prio.fill(static_cast<u8>(kNoWaiterPriority));
    u32 starved_mask = 0;

    for (s32 src_core = 0; src_core < static_cast<s32>(Core::Hardware::NUM_CPU_CORES); ++src_core) {
        const u32 sched_count = priority_queue.GetScheduledCount(src_core);
        if (sched_count == 0) {
            continue;
        }
        KThread* const src_top = top_threads[static_cast<size_t>(src_core)];
        if (sched_count == 1 && src_top == priority_queue.GetScheduledFront(src_core)) {
            continue;
        }
        const bool src_allows_migration =
            src_top == nullptr || src_top->GetPriority() >= HighestCoreMigrationAllowedPriority;
        for (KThread* w = priority_queue.GetScheduledFront(src_core); w != nullptr;
             w = priority_queue.GetScheduledNext(src_core, w)) {
            if (w == src_top) {
                continue;
            }
            const u8 prio =
                static_cast<u8>(std::clamp<s32>(w->GetPriority(), 0, kNoWaiterPriority));
            if (prio < local_waiter_prio[static_cast<size_t>(src_core)]) {
                local_waiter_prio[static_cast<size_t>(src_core)] = prio;
            }
            const bool unleased = w->m_nx2_yield_handoff_until_ns == 0;
            if (unleased) {
                starved_mask |= (1U << src_core);
            }
            u64 dst_mask = w->GetAffinityMask().GetAffinityMask() & ~(1ULL << src_core);
            if (dst_mask != 0) {
                if (unleased && src_allows_migration) {
                    starved_mask |= static_cast<u32>(dst_mask);
                }
                while (dst_mask != 0) {
                    const size_t dst = static_cast<size_t>(std::countr_zero(dst_mask));
                    dst_mask &= dst_mask - 1;
                    if (prio < migratable_prio[dst]) {
                        migratable_prio[dst] = prio;
                    }
                }
            }
        }
    }

    if (priority_queue.GetDetachedCount() > 0) {
        for (s32 c = 0; c < static_cast<s32>(Core::Hardware::NUM_CPU_CORES); ++c) {
            if (priority_queue.GetSuggestedCount(c) == 0) {
                continue;
            }
            for (KThread* s = priority_queue.GetSuggestedFront(c); s != nullptr;
                 s = priority_queue.GetSuggestedNext(c, s)) {
                if (s->GetActiveCore() < 0) {
                    const u8 prio =
                        static_cast<u8>(std::clamp<s32>(s->GetPriority(), 0, kNoWaiterPriority));
                    if (prio < migratable_prio[static_cast<size_t>(c)]) {
                        migratable_prio[static_cast<size_t>(c)] = prio;
                    }
                    if (s->m_nx2_yield_handoff_until_ns == 0) {
                        starved_mask |= (1U << c);
                    }
                }
            }
        }
    }

    for (size_t core_id = 0; core_id < Core::Hardware::NUM_CPU_CORES; ++core_id) {
        auto& sched = kernel.Scheduler(core_id);
        const u64 exp = core_expiries[core_id];
        if (sched.m_handoff_expiry_ns.load(std::memory_order_relaxed) != exp) {
            sched.m_handoff_expiry_ns.store(exp, std::memory_order_relaxed);
        }
        const u32 new_yield_state =
            static_cast<u32>(local_waiter_prio[core_id]) |
            (static_cast<u32>(migratable_prio[core_id]) << 8) |
            (exp != 0 ? kYieldStateActiveHandoff : 0U) |
            (((starved_mask >> core_id) & 1U) != 0 ? kYieldStateStarvedCandidate : 0U);
        if (sched.m_yield_state.load(std::memory_order_relaxed) != new_yield_state) {
            sched.m_yield_state.store(new_yield_state, std::memory_order_relaxed);
        }
        cores_needing_scheduling |= sched.UpdateHighestPriorityThread(kernel, top_threads[core_id]);
    }

    auto& active_handoff = kernel.GlobalSchedulerContext().m_active_handoff_mask;
    if (active_handoff.load(std::memory_order_relaxed) != new_handoff_mask) {
        active_handoff.store(new_handoff_mask, std::memory_order_relaxed);
    }

    // HACK: any waiting dummy threads can wake up now.
    kernel.GlobalSchedulerContext().WakeupWaitingDummyThreads(kernel);

    // HACK: if we are a dummy thread, and we need to go sleep, indicate
    // that for when the lock is released.
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    if (cur_thread->IsDummyThread() && cur_thread->GetState() != ThreadState::Runnable) {
        cur_thread->RequestDummyThreadWait(kernel);
    }

    return cores_needing_scheduling;
}

void KScheduler::SwitchThread(KernelCore& kernel, KThread* next_thread) {
    KProcess* const cur_process = GetCurrentProcessPointer(kernel);
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);

    // We never want to schedule a null thread, so use the idle thread if we don't have a next.
    if (next_thread == nullptr) {
        next_thread = m_idle_thread;
    }

    if (next_thread->GetCurrentCore() != m_core_id) {
        next_thread->SetCurrentCore(m_core_id);
    }

    // If we're not actually switching thread, there's nothing to do.
    if (next_thread == cur_thread) {
        return;
    }

    // Next thread is now known not to be nullptr, and must not be dispatchable.
    ASSERT(next_thread->GetDisableDispatchCount() == 1);
    ASSERT(!next_thread->IsDummyThread());

    // Update the CPU time tracking variables.
    const s64 prev_tick = m_last_context_switch_time;
    const s64 cur_tick = kernel.System().CoreTiming().GetClockTicks();
    const s64 tick_diff = cur_tick - prev_tick;
    cur_thread->AddCpuTime(m_core_id, tick_diff);
    if (cur_process != nullptr) {
        cur_process->AddCpuTime(tick_diff);
    }
    m_last_context_switch_time = cur_tick;

    // Update our previous thread.
    if (cur_process != nullptr) {
        if (!cur_thread->IsTerminationRequested() && cur_thread->GetActiveCore() == m_core_id)
            [[likely]] {
            m_state.prev_thread = cur_thread;
        } else {
            m_state.prev_thread = nullptr;
        }
    }

    // Switch the current process, if we're switching processes.
    // if (KProcess *next_process = next_thread->GetOwnerProcess(); next_process != cur_process) {
    //     KProcess::Switch(cur_process, next_process);
    // }

    // Set the new thread.
    SetCurrentThread(kernel, next_thread);
    m_current_thread = next_thread;

    // Set the new Thread Local region.
    // cpu::SwitchThreadLocalRegion(GetInteger(next_thread->GetThreadLocalRegionAddress()));

    // Update the thread's cpu time differential in TLS, if relevant.
    next_thread->UpdateTlsThreadCpuTime(kernel, cur_tick);
}

void KScheduler::ScheduleImpl(KernelCore& kernel) {
    // First, clear the needs scheduling bool.
    m_state.needs_scheduling.store(false, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);

    // Load the appropriate thread pointers for scheduling.
    KThread* const cur_thread{GetCurrentThreadPointer(kernel)};
    KThread* highest_priority_thread{GetHighestPriorityThread()};

    // Check whether there are runnable interrupt tasks.
    if (m_state.interrupt_task_runnable) {
        // The interrupt task is runnable.
        // We want to switch to the interrupt task/idle thread.
        highest_priority_thread = nullptr;
    }

    // If there aren't, we want to check if the highest priority thread is the same as the current
    // thread.
    if (highest_priority_thread == cur_thread) {
        // If they're the same, then we can just issue a memory barrier and return.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return;
    }

    // The highest priority thread is not the same as the current thread.
    // Jump to the switcher and continue executing from there.
    m_switch_cur_thread = cur_thread;
    m_switch_highest_priority_thread = highest_priority_thread;
    m_switch_from_schedule = true;
    Common::Fiber::YieldTo(cur_thread->m_host_context, *m_switch_fiber);

    // Returning from ScheduleImpl occurs after this thread has been scheduled again.
}

void KScheduler::ScheduleImplFiber(KernelCore& kernel) {
    KThread* const cur_thread{m_switch_cur_thread};
    KThread* highest_priority_thread{m_switch_highest_priority_thread};

    // If we're not coming from scheduling (i.e., we came from SC preemption),
    // we should restart the scheduling loop directly. Not accurate to HOS.
    if (!m_switch_from_schedule) {
        goto retry;
    }

    // Mark that we are not coming from scheduling anymore.
    m_switch_from_schedule = false;

    // Save the original thread context.
    Unload(kernel, cur_thread);

    // The current thread's context has been entirely taken care of.
    // Now we want to loop until we successfully switch the thread context.
    while (true) {
        // We're starting to try to do the context switch.
        // Check if the highest priority thread is null.
        if (!highest_priority_thread) {
            // The next thread is nullptr!

            // Switch to the idle thread. Note: HOS treats idling as a special case for
            // performance. This is not *required* for yuzu's purposes, and for singlecore
            // compatibility, we can just move the logic that would go here into the execution
            // of the idle thread. If we ever remove singlecore, we should implement this
            // accurately to HOS.
            highest_priority_thread = m_idle_thread;
        }

        // We want to try to lock the highest priority thread's context.
        // Try to take it.
        while (!highest_priority_thread->m_context_guard.try_lock()) {
            // The highest priority thread's context is already locked.
            // Check if we need scheduling. If we don't, we can retry directly.
            if (m_state.needs_scheduling.load(std::memory_order_acquire)) {
                // If we do, another core is interfering, and we must start again.
                goto retry;
            }
            SpinPause();
        }

        // It's time to switch the thread.
        // Switch to the highest priority thread.
        SwitchThread(kernel, highest_priority_thread);

        // Check if we need scheduling. If we do, then we can't complete the switch and should
        // retry.
        if (m_state.needs_scheduling.load(std::memory_order_seq_cst)) {
            // Some libc++ lazily init mutex
            [[maybe_unused]] auto const can_lock = highest_priority_thread->m_context_guard.try_lock();
            DEBUG_ASSERT(!can_lock);

            // Our switch failed.
            // We should unlock the thread context, and then retry.
            highest_priority_thread->m_context_guard.unlock();
            goto retry;
        } else {
            break;
        }

    retry:

        // We failed to successfully do the context switch, and need to retry.
        // Clear needs_scheduling.
        m_state.needs_scheduling.store(false, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);

        // Refresh the highest priority thread.
        highest_priority_thread = GetHighestPriorityThread();
    }

    // Reload the guest thread context.
    Reload(kernel, highest_priority_thread);

    // Reload the host thread.
    Common::Fiber::YieldTo(m_switch_fiber, *highest_priority_thread->m_host_context);
}

void KScheduler::Unload(KernelCore& kernel, KThread* thread) {
    kernel.PhysicalCore(m_core_id).SaveContext(thread);

    // Before the context is unlocked: another core may take the thread right after.
    if (thread->m_nx2_perf_slot >= 0) {
        auto& perf = Common::PerfSched::g_guest_threads[static_cast<size_t>(thread->m_nx2_perf_slot)];
        const u64 now = Common::PerfReport::NowNs();
        if (thread->m_nx2_perf_run_since != 0) {
            perf.run_ns.fetch_add(now - thread->m_nx2_perf_run_since, std::memory_order_relaxed);
            thread->m_nx2_perf_run_since = 0;
        }
        if (thread->GetRawState() == ThreadState::Runnable) {
            thread->m_nx2_perf_ready_since.store(now, std::memory_order_relaxed);
        }
    }

    // Check if the thread is terminated by checking the DPC flags.
    if ((thread->GetStackParameters().dpc_flags & static_cast<u32>(DpcFlag::Terminated)) == 0) {
        // Some libc++ lazily init mutex
        [[maybe_unused]] auto const can_lock = thread->m_context_guard.try_lock();
        DEBUG_ASSERT(!can_lock);

        // The thread isn't terminated, so we want to unlock it.
        thread->m_context_guard.unlock();
    }
}

void KScheduler::Reload(KernelCore& kernel, KThread* thread) {
    if (thread->m_nx2_perf_slot >= 0) {
        auto& perf = Common::PerfSched::g_guest_threads[static_cast<size_t>(thread->m_nx2_perf_slot)];
        const u64 now = Common::PerfReport::NowNs();
        if (const u64 since = thread->m_nx2_perf_ready_since.exchange(0, std::memory_order_relaxed);
            since != 0 && now > since) {
            const u64 waited = now - since;
            perf.ready_ns.fetch_add(waited, std::memory_order_relaxed);
            if (waited > perf.ready_max_ns.load(std::memory_order_relaxed)) {
                perf.ready_max_ns.store(waited, std::memory_order_relaxed);
            }
        }
        perf.switches.fetch_add(1, std::memory_order_relaxed);
        if (thread->m_nx2_perf_core >= 0 && thread->m_nx2_perf_core != m_core_id) {
            perf.core_changes.fetch_add(1, std::memory_order_relaxed);
        }
        perf.cores.fetch_or(1U << m_core_id, std::memory_order_relaxed);
        perf.priority.store(thread->GetPriority(), std::memory_order_relaxed);
        thread->m_nx2_perf_core = m_core_id;
        thread->m_nx2_perf_run_since = now;
    }
    kernel.PhysicalCore(m_core_id).LoadContext(thread);
}

void KScheduler::ClearPreviousThread(KernelCore& kernel, KThread* thread) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));
    for (size_t i = 0; i < Core::Hardware::NUM_CPU_CORES; ++i) {
        // Get an atomic reference to the core scheduler's previous thread.
        auto& prev_thread{kernel.Scheduler(i).m_state.prev_thread};

        // Atomically clear the previous thread if it's our target.
        if (prev_thread.load(std::memory_order_relaxed) == thread) {
            KThread* compare = thread;
            prev_thread.compare_exchange_strong(compare, nullptr, std::memory_order_seq_cst);
        }
    }
}

void KScheduler::OnThreadStateChanged(KernelCore& kernel, KThread* thread, ThreadState old_state) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));
    // Check if the state has changed, because if it hasn't there's nothing to do.
    if (const ThreadState cur_state = thread->GetRawState(); cur_state != old_state) {
        // Update the priority queues.
        if (old_state == ThreadState::Runnable) {
            // If we were previously runnable, then we're not runnable now, and we should remove.
            thread->m_nx2_futile_yield_streak = 0;
            thread->m_nx2_yield_spin_streak = 0;
            thread->m_nx2_yield_handoff_until_ns = 0;
            GetPriorityQueue(kernel).Remove(thread);
            IncrementScheduledCount(thread);
            SetSchedulerUpdateNeeded(kernel);
            // HACK: if this is a dummy thread, it should no longer wake up when the
            // scheduler lock is released.
            if (thread->IsDummyThread())
                kernel.GlobalSchedulerContext().UnregisterDummyThreadForWakeup(thread);
        } else if (cur_state == ThreadState::Runnable) {
            // If we're now runnable, then we weren't previously, and we should add.
            GetPriorityQueue(kernel).PushBack(thread);
            IncrementScheduledCount(thread);
            SetSchedulerUpdateNeeded(kernel);
            if (Common::PerfSched::Enabled() && !thread->IsDummyThread()) {
                if (thread->m_nx2_perf_slot == -2) {
                    const KProcess* const owner = thread->GetOwnerProcess();
                    thread->m_nx2_perf_slot = Common::PerfSched::AllocateGuestThread(
                        thread->GetThreadId(),
                        owner != nullptr && owner == kernel.ApplicationProcess());
                }
                thread->m_nx2_perf_ready_since.store(Common::PerfReport::NowNs(),
                                                     std::memory_order_relaxed);
            }
            // HACK: if this is a dummy thread, it should wake up when the scheduler
            // lock is released.
            if (thread->IsDummyThread())
                kernel.GlobalSchedulerContext().RegisterDummyThreadForWakeup(thread);
        }
    }
}

void KScheduler::OnThreadPriorityChanged(KernelCore& kernel, KThread* thread, s32 old_priority) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));

    // If the thread is runnable, we want to change its priority in the queue.
    if (thread->GetRawState() == ThreadState::Runnable) {
        GetPriorityQueue(kernel).ChangePriority(old_priority,
                                                thread == GetCurrentThreadPointer(kernel), thread);
        IncrementScheduledCount(thread);
        SetSchedulerUpdateNeeded(kernel);
    }
}

void KScheduler::OnThreadAffinityMaskChanged(KernelCore& kernel, KThread* thread,
                                             const KAffinityMask& old_affinity, s32 old_core) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));

    // If the thread is runnable, we want to change its affinity in the queue.
    if (thread->GetRawState() == ThreadState::Runnable) {
        GetPriorityQueue(kernel).ChangeAffinityMask(old_core, old_affinity, thread);
        IncrementScheduledCount(thread);
        SetSchedulerUpdateNeeded(kernel);
    }
}

void KScheduler::RotateScheduledQueue(KernelCore& kernel, s32 core_id, s32 priority) {
    ASSERT(IsSchedulerLockedByCurrentThread(kernel));

    // Get a reference to the priority queue.
    auto& priority_queue = GetPriorityQueue(kernel);
    bool queue_changed = false;

    // Rotate the front of the queue to the end.
    KThread* top_thread = priority_queue.GetScheduledFront(core_id, priority);
    KThread* next_thread = nullptr;
    if (top_thread != nullptr) {
        next_thread = priority_queue.MoveToScheduledBack(top_thread);
        if (next_thread != top_thread) {
            IncrementScheduledCount(top_thread);
            IncrementScheduledCount(next_thread);
            queue_changed = true;
        }
    }

    // While we have a suggested thread, try to migrate it!
    {
        KThread* suggested = priority_queue.GetSuggestedFront(core_id, priority);
        while (suggested != nullptr) {
            // Check if the suggested thread is the top thread on its core.
            const s32 suggested_core = suggested->GetActiveCore();
            KThread* const top_on_suggested_core =
                (suggested_core >= 0) ? priority_queue.GetScheduledFront(suggested_core) : nullptr;
            KThread* const running_on_suggested_core =
                (suggested_core >= 0)
                    ? kernel.Scheduler(static_cast<size_t>(suggested_core)).GetHighestPriorityThread()
                    : nullptr;
            if (top_on_suggested_core != suggested && running_on_suggested_core != suggested) {
                // If the next thread is a new thread that has been waiting longer than our
                // suggestion, we prefer it to our suggestion.
                if (top_thread != next_thread && next_thread != nullptr &&
                    next_thread->GetLastScheduledTick() < suggested->GetLastScheduledTick()) {
                    suggested = nullptr;
                    break;
                }

                // If we're allowed to do a migration, do one.
                // NOTE: Unlike migrations in UpdateHighestPriorityThread, this moves the suggestion
                // to the front of the queue.
                if (top_on_suggested_core == nullptr ||
                    top_on_suggested_core->GetPriority() >= HighestCoreMigrationAllowedPriority) {
                    suggested->SetActiveCore(core_id);
                    priority_queue.ChangeCore(suggested_core, suggested, true);
                    IncrementScheduledCount(suggested);
                    queue_changed = true;
                    break;
                }
            }

            // Get the next suggestion.
            suggested = priority_queue.GetSamePriorityNext(core_id, suggested);
        }
    }

    // Now that we might have migrated a thread with the same priority, check if we can do better.
    if (priority_queue.GetSuggestedCount(core_id) > 0) {
        KThread* best_thread = priority_queue.GetScheduledFront(core_id);
        if (best_thread == GetCurrentThreadPointer(kernel)) {
            best_thread = priority_queue.GetScheduledNext(core_id, best_thread);
        }

        // If the best thread we can choose has a priority the same or worse than ours, try to
        // migrate a higher priority thread.
        if (best_thread != nullptr && best_thread->GetPriority() >= priority) {
            KThread* suggested = priority_queue.GetSuggestedFront(core_id);
            while (suggested != nullptr) {
                // If the suggestion's priority is the same as ours, don't bother.
                if (suggested->GetPriority() >= best_thread->GetPriority()) {
                    break;
                }

                // Check if the suggested thread is the top thread on its core.
                const s32 suggested_core = suggested->GetActiveCore();
                KThread* const top_on_suggested_core =
                    (suggested_core >= 0) ? priority_queue.GetScheduledFront(suggested_core)
                                          : nullptr;
                KThread* const running_on_suggested_core =
                    (suggested_core >= 0)
                        ? kernel.Scheduler(static_cast<size_t>(suggested_core)).GetHighestPriorityThread()
                        : nullptr;
                if (top_on_suggested_core != suggested && running_on_suggested_core != suggested) {
                    // If we're allowed to do a migration, do one.
                    // NOTE: Unlike migrations in UpdateHighestPriorityThread, this moves the
                    // suggestion to the front of the queue.
                    if (top_on_suggested_core == nullptr ||
                        top_on_suggested_core->GetPriority() >=
                            HighestCoreMigrationAllowedPriority) {
                        suggested->SetActiveCore(core_id);
                        priority_queue.ChangeCore(suggested_core, suggested, true);
                        IncrementScheduledCount(suggested);
                        queue_changed = true;
                        break;
                    }
                }

                // Get the next suggestion.
                suggested = priority_queue.GetSuggestedNext(core_id, suggested);
            }
        }
    }

    // After a rotation that changed a queue, we need a scheduler update.
    if (queue_changed) {
        SetSchedulerUpdateNeeded(kernel);
    }
}

void KScheduler::YieldWithoutCoreMigration(KernelCore& kernel) {
    // Validate preconditions.
    ASSERT(CanSchedule(kernel));
    ASSERT(GetCurrentProcessPointer(kernel) != nullptr);

    // Get the current thread and process.
    KThread& cur_thread = GetCurrentThread(kernel);
    KProcess& cur_process = GetCurrentProcess(kernel);

    // If the thread's yield count matches, there's nothing for us to do.
    if (cur_thread.GetYieldScheduleCount() == cur_process.GetScheduledCount()) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }

    // Get a reference to the priority queue.
    auto& priority_queue = GetPriorityQueue(kernel);

    // YieldWithoutCoreMigration only rotates this core's scheduled queue: activity on other
    // cores cannot make another thread runnable here unless this core's scheduled queue changed.
    if (const s32 fast_core = cur_thread.GetActiveCore();
        fast_core >= 0 && cur_thread.GetRawState() == ThreadState::Runnable &&
        (IsCoreYieldFutile(kernel, cur_process, cur_thread, fast_core,
                           Svc::YieldType::WithoutCoreMigration) ||
         (cur_thread.GetYieldCoreId() == fast_core &&
          cur_thread.GetYieldCoreSeq() == priority_queue.GetScheduledQueueVersion(fast_core)))) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }

    // Perform the yield.
    {
        KScopedSchedulerLock sl{kernel};

        const auto cur_state = cur_thread.GetRawState();
        if (cur_state == ThreadState::Runnable) {
            // Put the current thread at the back of the queue.
            KThread* next_thread = priority_queue.MoveToScheduledBack(std::addressof(cur_thread));

            // If the next thread is different, we have an update to perform.
            if (next_thread != std::addressof(cur_thread)) {
                IncrementScheduledCount(std::addressof(cur_thread));
                SetSchedulerUpdateNeeded(kernel);
                cur_thread.m_nx2_last_yield_futile = false;
            } else {
                // Otherwise, record both process and per-core counts without bumping the
                // process-wide count (which would invalidate other cores' yield fast paths).
                const s32 active_core = cur_thread.GetActiveCore();
                cur_thread.SetYieldScheduleCount(cur_process.GetScheduledCount());
                if (active_core >= 0) {
                    cur_thread.SetYieldCoreSeq(active_core,
                                               priority_queue.GetScheduledQueueVersion(active_core));
                }
                cur_thread.m_nx2_last_yield_futile = true;
            }
        } else {
            cur_thread.m_nx2_last_yield_futile = false;
        }
    }
}

void KScheduler::YieldWithCoreMigration(KernelCore& kernel) {
    // Validate preconditions.
    ASSERT(CanSchedule(kernel));
    ASSERT(GetCurrentProcessPointer(kernel) != nullptr);

    // Get the current thread and process.
    KThread& cur_thread = GetCurrentThread(kernel);
    KProcess& cur_process = GetCurrentProcess(kernel);

    // If the thread's yield count matches, there's nothing for us to do.
    if (cur_thread.GetYieldScheduleCount() == cur_process.GetScheduledCount()) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }
    if (const s32 fast_core = cur_thread.GetActiveCore();
        fast_core >= 0 && cur_thread.GetRawState() == ThreadState::Runnable &&
        IsCoreYieldFutile(kernel, cur_process, cur_thread, fast_core,
                          Svc::YieldType::WithCoreMigration)) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }

    // Get a reference to the priority queue.
    auto& priority_queue = GetPriorityQueue(kernel);

    // Perform the yield.
    {
        KScopedSchedulerLock sl{kernel};

        const auto cur_state = cur_thread.GetRawState();
        if (cur_state == ThreadState::Runnable) {
            // Get the current active core.
            const s32 core_id = cur_thread.GetActiveCore();

            // Put the current thread at the back of the queue.
            KThread* next_thread = priority_queue.MoveToScheduledBack(std::addressof(cur_thread));

            // While we have a suggested thread, try to migrate it!
            bool recheck = false;
            const s32 cur_prio = cur_thread.GetPriority();
            KThread* suggested = priority_queue.GetSuggestedFront(core_id);
            while (suggested != nullptr) {
                if (suggested->GetPriority() > cur_prio) {
                    suggested = nullptr;
                    break;
                }

                // Check if the suggested thread is the thread running on its core.
                const s32 suggested_core = suggested->GetActiveCore();

                if (KThread* running_on_suggested_core =
                        (suggested_core >= 0)
                            ? kernel.Scheduler(static_cast<size_t>(suggested_core)).GetHighestPriorityThread()
                            : nullptr;
                    running_on_suggested_core != suggested) {
                    // Prefer the next thread when the current thread's priority is equal to the
                    // suggestion's, but the next thread has been waiting longer.
                    if (suggested->GetPriority() == cur_prio &&
                        next_thread != std::addressof(cur_thread) &&
                        next_thread->GetLastScheduledTick() < suggested->GetLastScheduledTick()) {
                        suggested = nullptr;
                        break;
                    }

                    // If we're allowed to do a migration, do one.
                    // NOTE: Unlike migrations in UpdateHighestPriorityThread, this moves the
                    // suggestion to the front of the queue.
                    if (running_on_suggested_core == nullptr ||
                        running_on_suggested_core->GetPriority() >=
                            HighestCoreMigrationAllowedPriority) {
                        suggested->SetActiveCore(core_id);
                        priority_queue.ChangeCore(suggested_core, suggested, true);
                        IncrementScheduledCount(suggested);
                        break;
                    } else {
                        // We couldn't perform a migration, but we should check again on a future
                        // yield.
                        recheck = true;
                    }
                }

                // Get the next suggestion.
                suggested = priority_queue.GetSuggestedNext(core_id, suggested);
            }

            // If we still have a suggestion or the next thread is different, we have an update to
            // perform.
            if (suggested != nullptr || next_thread != std::addressof(cur_thread)) {
                IncrementScheduledCount(std::addressof(cur_thread));
                SetSchedulerUpdateNeeded(kernel);
                cur_thread.m_nx2_last_yield_futile = false;
            } else {
                if (!recheck) {
                    // Otherwise if we don't need to re-check, set the thread's yield count so that
                    // we won't waste work until the process is scheduled again.
                    cur_thread.SetYieldScheduleCount(cur_process.GetScheduledCount());
                }
                cur_thread.m_nx2_last_yield_futile = true;
            }
        } else {
            cur_thread.m_nx2_last_yield_futile = false;
        }
    }
}

void KScheduler::YieldToAnyThread(KernelCore& kernel) {
    // Validate preconditions.
    ASSERT(CanSchedule(kernel));
    ASSERT(GetCurrentProcessPointer(kernel) != nullptr);

    // Get the current thread and process.
    KThread& cur_thread = GetCurrentThread(kernel);
    KProcess& cur_process = GetCurrentProcess(kernel);

    // If the thread's yield count matches, there's nothing for us to do.
    if (cur_thread.GetYieldScheduleCount() == cur_process.GetScheduledCount()) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }
    if (const s32 fast_core = cur_thread.GetActiveCore();
        fast_core >= 0 && cur_thread.GetRawState() == ThreadState::Runnable &&
        IsCoreYieldFutile(kernel, cur_process, cur_thread, fast_core,
                          Svc::YieldType::ToAnyThread)) {
        cur_thread.m_nx2_last_yield_futile = true;
        return;
    }

    // Get a reference to the priority queue.
    auto& priority_queue = GetPriorityQueue(kernel);

    // Perform the yield.
    {
        KScopedSchedulerLock sl{kernel};

        const auto cur_state = cur_thread.GetRawState();
        if (cur_state == ThreadState::Runnable) {
            // Get the current active core.
            const s32 core_id = cur_thread.GetActiveCore();

            // Fast check: if cur_thread is alone on core_id and no non-running suggested thread
            // with priority <= cur_thread's exists on core_id, detaching to -1 would only walk
            // down to cur_thread and re-attach it right back to core_id.
            if (core_id >= 0 &&
                priority_queue.GetScheduledFront(core_id) == std::addressof(cur_thread) &&
                priority_queue.GetScheduledNext(core_id, std::addressof(cur_thread)) == nullptr) {
                bool has_candidate = false;
                for (KThread* probe = priority_queue.GetSuggestedFront(core_id);
                     probe != nullptr && probe->GetPriority() <= cur_thread.GetPriority();
                     probe = priority_queue.GetSuggestedNext(core_id, probe)) {
                    const s32 probe_core = probe->GetActiveCore();
                    KThread* const top_on_probe =
                        (probe_core >= 0) ? priority_queue.GetScheduledFront(probe_core) : nullptr;
                    KThread* const running_on_probe =
                        (probe_core >= 0)
                            ? kernel.Scheduler(static_cast<size_t>(probe_core)).GetHighestPriorityThread()
                            : nullptr;
                    if (top_on_probe != probe && running_on_probe != probe) {
                        has_candidate = true;
                        break;
                    }
                }
                if (!has_candidate) {
                    cur_thread.SetYieldScheduleCount(cur_process.GetScheduledCount());
                    cur_thread.m_nx2_last_yield_futile = true;
                    return;
                }
            }

            // Migrate the current thread to core -1.
            cur_thread.SetActiveCore(-1);
            priority_queue.ChangeCore(core_id, std::addressof(cur_thread));
            IncrementScheduledCount(std::addressof(cur_thread));

            // If there's nothing scheduled, we can try to perform a migration.
            if (priority_queue.GetScheduledFront(core_id) == nullptr) {
                // While we have a suggested thread, try to migrate it!
                KThread* suggested = priority_queue.GetSuggestedFront(core_id);
                while (suggested != nullptr) {
                    // Check if the suggested thread is the top thread on its core.
                    const s32 suggested_core = suggested->GetActiveCore();
                    KThread* const top_on_suggested_core =
                        (suggested_core >= 0) ? priority_queue.GetScheduledFront(suggested_core)
                                              : nullptr;
                    KThread* const running_on_suggested_core =
                        (suggested_core >= 0)
                            ? kernel.Scheduler(static_cast<size_t>(suggested_core)).GetHighestPriorityThread()
                            : nullptr;
                    if (top_on_suggested_core != suggested &&
                        running_on_suggested_core != suggested) {
                        // If we're allowed to do a migration, do one.
                        if (top_on_suggested_core == nullptr ||
                            top_on_suggested_core->GetPriority() >=
                                HighestCoreMigrationAllowedPriority) {
                            suggested->SetActiveCore(core_id);
                            priority_queue.ChangeCore(suggested_core, suggested);
                            IncrementScheduledCount(suggested);
                        }

                        // Regardless of whether we migrated, we had a candidate, so we're done.
                        break;
                    }

                    // Get the next suggestion.
                    suggested = priority_queue.GetSuggestedNext(core_id, suggested);
                }

                // If the suggestion is different from the current thread, we need to perform an
                // update.
                if (suggested != std::addressof(cur_thread)) {
                    SetSchedulerUpdateNeeded(kernel);
                    cur_thread.m_nx2_last_yield_futile = false;
                } else {
                    // Otherwise, set the thread's yield count so that we won't waste work until the
                    // process is scheduled again.
                    cur_thread.SetYieldScheduleCount(cur_process.GetScheduledCount());
                    cur_thread.m_nx2_last_yield_futile = true;
                }
            } else {
                // Otherwise, we have an update to perform.
                SetSchedulerUpdateNeeded(kernel);
                cur_thread.m_nx2_last_yield_futile = false;
            }
        } else {
            cur_thread.m_nx2_last_yield_futile = false;
        }
    }
}

void KScheduler::RescheduleOtherCores(KernelCore& kernel, u64 cores_needing_scheduling) {
    if (const u64 core_mask = cores_needing_scheduling & ~(1ULL << m_core_id); core_mask != 0) {
        RescheduleCores(kernel, core_mask);
    }
}

void KScheduler::RescheduleCores(KernelCore& kernel, u64 core_mask) {
    // Send IPI
    core_mask &= (1ULL << Core::Hardware::NUM_CPU_CORES) - 1;
    while (core_mask != 0) {
        const size_t i = static_cast<size_t>(std::countr_zero(core_mask));
        core_mask &= core_mask - 1;
        kernel.PhysicalCore(i).Interrupt();
    }
}

} // namespace Kernel
