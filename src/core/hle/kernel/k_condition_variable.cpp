// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/spin_deadline.h"
#include "core/core.h"
#include "core/hle/kernel/k_condition_variable.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc_common.h"
#include "core/hle/kernel/svc_results.h"
#include "core/memory.h"

namespace Kernel {

namespace {

bool ReadFromUser(KernelCore& kernel, u32* out, KProcessAddress address) {
    *out = GetCurrentMemory(kernel).Read32(GetInteger(address));
    return true;
}

bool WriteToUser(KernelCore& kernel, KProcessAddress address, u32 val) {
    GetCurrentMemory(kernel).Write32(GetInteger(address), val);
    return true;
}

bool UpdateLockAtomic(KernelCore& kernel, u32* out, KProcessAddress address, u32 if_zero,
                      u32 new_orr_mask) {
    auto& memory = GetCurrentMemory(kernel);
    const u64 raw_addr = GetInteger(address);

    u32 expected = memory.Read32(raw_addr);
    while (true) {
        const u32 value = expected != 0 ? (expected | new_orr_mask) : if_zero;
        if (memory.WriteExclusive32(raw_addr, value, expected)) {
            break;
        }
        expected = memory.Read32(raw_addr);
    }

    *out = expected;
    return true;
}

void LockSpinPause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

s64 LockSpinMicroseconds() {
    static const s64 microseconds = [] {
        const char* value = std::getenv("EDEN_NX2_LOCK_SPIN");
        if (value == nullptr || value[0] == '\0') {
            return s64{4};
        }
        return (std::max)(s64{0}, static_cast<s64>(std::strtoll(value, nullptr, 10)));
    }();
    return microseconds;
}

class ThreadQueueImplForKConditionVariableWaitForAddress final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKConditionVariableWaitForAddress(KernelCore& kernel)
        : KThreadQueue(kernel)
    {}

    virtual void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Remove the thread as a waiter from its owner.
        waiting_thread->GetLockOwner(kernel)->RemoveWaiter(kernel, waiting_thread);

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }
};

class ThreadQueueImplForKConditionVariableWaitConditionVariable final : public KThreadQueue {
private:
    KConditionVariable::ThreadTree* m_tree;

public:
    explicit ThreadQueueImplForKConditionVariableWaitConditionVariable(KernelCore& kernel, KConditionVariable::ThreadTree* t)
        : KThreadQueue(kernel)
        , m_tree(t)
    {}

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Remove the thread as a waiter from its owner.
        if (KThread* owner = waiting_thread->GetLockOwner(kernel); owner != nullptr) {
            owner->RemoveWaiter(kernel, waiting_thread);
        }

        // If the thread is waiting on a condvar, remove it from the tree.
        if (waiting_thread->IsWaitingForConditionVariable()) {
            m_tree->erase(m_tree->iterator_to(*waiting_thread));
            waiting_thread->ClearConditionVariable();
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }
};

} // namespace

KConditionVariable::KConditionVariable(Core::System& system)
    : m_system{system}
{}

KConditionVariable::~KConditionVariable() = default;

Result KConditionVariable::SignalToAddress(KernelCore& kernel, KProcessAddress addr) {
    KThread* owner_thread = GetCurrentThreadPointer(kernel);

    // Signal the address.
    {
        KScopedSchedulerLock sl(kernel);

        // Remove waiter thread.
        bool has_waiters{};
        KThread* const next_owner_thread = owner_thread->RemoveUserWaiterByKey(kernel, std::addressof(has_waiters), addr);

        // Determine the next tag.
        u32 next_value{};
        if (next_owner_thread != nullptr) {
            next_value = next_owner_thread->GetAddressKeyValue();
            if (has_waiters) {
                next_value |= Svc::HandleWaitMask;
            }
        }

        // Synchronize memory before proceeding.
        std::atomic_thread_fence(std::memory_order_release);

        // Write the value to userspace.
        Result result{ResultSuccess};
        if (WriteToUser(kernel, addr, next_value)) {
            result = ResultSuccess;
        } else {
            result = ResultInvalidCurrentMemory;
        }

        // If necessary, signal the next owner thread.
        if (next_owner_thread != nullptr) {
            next_owner_thread->EndWait(kernel, result);
        }

        R_RETURN(result);
    }
}

Result KConditionVariable::WaitForAddress(KernelCore& kernel, Handle handle, KProcessAddress addr, u32 value) {
    KThread* cur_thread = GetCurrentThreadPointer(kernel);
    ThreadQueueImplForKConditionVariableWaitForAddress wait_queue(kernel);
    const u32 expected_tag = handle | Svc::HandleWaitMask;

    // Resolve handle and optionally watch the lock word briefly outside KScopedSchedulerLock
    // when the lock owner is actively running on another physical core (so priority inheritance
    // is never delayed for a preempted owner).
    KScopedAutoObject<KThread> pre_owner{kernel};
    if (!cur_thread->IsTerminationRequested() && (GetInteger(addr) & 3) == 0) {
        if (u8* const pointer = GetCurrentMemory(kernel).GetPointerSilent(GetInteger(addr));
            pointer != nullptr) {
            const auto load_tag = [pointer] {
                return std::atomic_ref<u32>{*reinterpret_cast<u32*>(pointer)}.load(
                    std::memory_order_acquire);
            };
            if (load_tag() != expected_tag) {
                R_SUCCEED();
            }
            pre_owner = GetCurrentProcess(kernel)
                            .GetHandleTable()
                            .GetObjectWithoutPseudoHandle<KThread>(kernel, handle);
            const s64 spin_us = LockSpinMicroseconds();
            const s32 current_core = GetCurrentCoreId(kernel);
            if (spin_us > 0 && pre_owner.IsNotNull() && kernel.IsMulticore() && current_core >= 0 &&
                KScheduler::GetCoreScheduledCount(kernel, current_core) <= 1) {
                const s32 owner_core = pre_owner->GetCurrentCore();
                if (owner_core >= 0 && owner_core != current_core &&
                    kernel.Scheduler(owner_core).GetSchedulerCurrentThread() ==
                        pre_owner.GetPointerUnsafe()) {
                    u8& credit = cur_thread->m_nx2_lock_spin_credit;
                    if (credit > 0 || (++cur_thread->m_nx2_lock_spin_probe & 15) == 0) {
                        const PhysicalCore& core = kernel.CurrentPhysicalCore();
                        const Common::SpinDeadline::Deadline deadline{
                            std::chrono::microseconds{spin_us}};
                        bool released = false;
                        for (u32 spin = 1;; ++spin) {
                            LockSpinPause();
                            if (load_tag() != expected_tag) {
                                credit = static_cast<u8>((std::min)(credit + 2, 16));
                                released = true;
                                break;
                            }
                            if ((spin & 31) == 0) {
                                if (core.IsInterrupted() ||
                                    KScheduler::GetCoreScheduledCount(kernel, current_core) > 1 ||
                                    kernel.Scheduler(owner_core).GetSchedulerCurrentThread() !=
                                        pre_owner.GetPointerUnsafe() ||
                                    deadline.Expired()) {
                                    break;
                                }
                            }
                        }
                        if (released && !cur_thread->IsTerminationRequested()) {
                            R_SUCCEED();
                        }
                        if (!released && credit > 0) {
                            --credit;
                        }
                    }
                }
            }
        }
    }

    // Wait for the address.
    KThread* owner_thread{};
    {
        KScopedSchedulerLock sl(kernel);

        // Check if the thread should terminate.
        R_UNLESS(!cur_thread->IsTerminationRequested(), ResultTerminationRequested);

        // Read the tag from userspace.
        u32 test_tag{};
        R_UNLESS(ReadFromUser(kernel, std::addressof(test_tag), addr), ResultInvalidCurrentMemory);

        // If the tag isn't the handle (with wait mask), we're done.
        R_SUCCEED_IF(test_tag != expected_tag);

        // Get the lock owner thread.
        owner_thread = pre_owner.IsNotNull()
                           ? pre_owner.ReleasePointerUnsafe()
                           : GetCurrentProcess(kernel)
                                 .GetHandleTable()
                                 .GetObjectWithoutPseudoHandle<KThread>(kernel, handle)
                                 .ReleasePointerUnsafe();
        R_UNLESS(owner_thread != nullptr, ResultInvalidHandle);

        // Update the lock.
        cur_thread->SetUserAddressKey(addr, value);
        owner_thread->AddWaiter(kernel, cur_thread);

        // Begin waiting.
        cur_thread->BeginWait(kernel, std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::ConditionVar);
    }

    // Close our reference to the owner thread, now that the wait is over.
    owner_thread->Close(kernel);

    // Get the wait result.
    R_RETURN(cur_thread->GetWaitResult());
}

void KConditionVariable::SignalImpl(KernelCore& kernel, KThread* thread) {
    // Check pre-conditions.
    ASSERT(KScheduler::IsSchedulerLockedByCurrentThread(kernel));

    // Update the tag.
    KProcessAddress address = thread->GetAddressKey();
    u32 own_tag = thread->GetAddressKeyValue();

    u32 prev_tag{};
    bool can_access{};
    {
        // NOTE: If scheduler lock is not held here, interrupt disable is required.
        // KScopedInterruptDisable di;

        // TODO(bunnei): We should call CanAccessAtomic(..) here.
        can_access = true;
        if (can_access) {
            UpdateLockAtomic(kernel, std::addressof(prev_tag), address, own_tag, Svc::HandleWaitMask);
        }
    }

    if (can_access) {
        if (prev_tag == Svc::InvalidHandle) {
            // If nobody held the lock previously, we're all good.
            thread->EndWait(kernel, ResultSuccess);
        } else {
            // Get the previous owner.
            KThread* owner_thread = GetCurrentProcess(kernel)
                .GetHandleTable()
                .GetObjectWithoutPseudoHandle<KThread>(kernel, Handle(prev_tag & ~Svc::HandleWaitMask))
                .ReleasePointerUnsafe();

            if (owner_thread) {
                // Add the thread as a waiter on the owner.
                owner_thread->AddWaiter(kernel, thread);
                owner_thread->Close(kernel);
            } else {
                // The lock was tagged with a thread that doesn't exist.
                thread->EndWait(kernel, ResultInvalidState);
            }
        }
    } else {
        // If the address wasn't accessible, note so.
        thread->EndWait(kernel, ResultInvalidCurrentMemory);
    }
}

void KConditionVariable::Signal(u64 cv_key, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(m_system.Kernel());

        auto it = m_tree.empty() ? m_tree.end() : m_tree.nfind_key({cv_key, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) && (it->GetConditionVariableKey() == cv_key)) {
            KThread* target_thread = std::addressof(*it);

            it = m_tree.erase(it);
            target_thread->ClearConditionVariable();

            this->SignalImpl(m_system.Kernel(), target_thread);

            ++num_waiters;
        }

        // If we have no waiters, clear the has waiter flag.
        if (it == m_tree.end() || it->GetConditionVariableKey() != cv_key) {
            constexpr u32 HasNoWaiterFlag = 0;
            WriteToUser(m_system.Kernel(), cv_key, HasNoWaiterFlag);
        }
    }
}

Result KConditionVariable::Wait(KProcessAddress addr, u64 key, u32 value, s64 timeout) {
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(m_system.Kernel());
    KHardwareTimer* timer{};
    ThreadQueueImplForKConditionVariableWaitConditionVariable wait_queue(m_system.Kernel(),
                                                                         std::addressof(m_tree));

    {
        KScopedSchedulerLockAndSleep slp(m_system.Kernel(), std::addressof(timer), cur_thread, timeout);

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Update the value and process for the next owner.
        {
            // Remove waiter thread.
            bool has_waiters{};
            KThread* next_owner_thread =
                cur_thread->RemoveUserWaiterByKey(m_system.Kernel(), std::addressof(has_waiters), addr);

            // Update for the next owner thread.
            u32 next_value{};
            if (next_owner_thread != nullptr) {
                // Get the next tag value.
                next_value = next_owner_thread->GetAddressKeyValue();
                if (has_waiters) {
                    next_value |= Svc::HandleWaitMask;
                }

                // Wake up the next owner.
                next_owner_thread->EndWait(m_system.Kernel(), ResultSuccess);
            }

            // Write to the cv key.
            {
                constexpr u32 HasWaiterFlag = 1;
                WriteToUser(m_system.Kernel(), key, HasWaiterFlag);
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            // Write the value to userspace.
            if (!WriteToUser(m_system.Kernel(), addr, next_value)) {
                slp.CancelSleep();
                R_THROW(ResultInvalidCurrentMemory);
            }
        }

        // If timeout is zero, time out.
        R_UNLESS(timeout != 0, ResultTimedOut);

        // Update condition variable tracking.
        cur_thread->SetConditionVariable(std::addressof(m_tree), addr, key, value);
        m_tree.insert(*cur_thread);

        // Begin waiting.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(m_system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::ConditionVar);
    }

    // Get the wait result.
    R_RETURN(cur_thread->GetWaitResult());
}

} // namespace Kernel
