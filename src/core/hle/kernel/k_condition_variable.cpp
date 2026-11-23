// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/spin_deadline.h"
#include "core/core.h"
#include "core/hle/kernel/k_condition_variable.h"
#include "core/hle/kernel/k_memory_layout.h"
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

#if defined(_MSC_VER)
#define NX2_CV_NOINLINE __declspec(noinline)
#else
#define NX2_CV_NOINLINE __attribute__((noinline))
#endif

constexpr size_t CondVarStripeBits = 12;
constexpr size_t NumCondVarWaiterStripes = size_t{1} << CondVarStripeBits;
static_assert(NumCondVarWaiterStripes == 4096);

alignas(128) std::array<std::atomic<u32>, NumCondVarWaiterStripes> g_cv_waiter_stripes_a{};
alignas(128) std::array<std::atomic<u32>, NumCondVarWaiterStripes> g_cv_waiter_stripes_b{};

inline size_t CondVarStripeIndexA(const KConditionVariable* cv, u64 key) noexcept {
    u64 h = (key >> 2) ^ (static_cast<u64>(reinterpret_cast<uintptr_t>(cv)) >> 4);
    h ^= (h >> 23) ^ (h >> 41);
    return static_cast<size_t>((h * u64{0x9E3779B97F4A7C15ULL}) >> (64 - CondVarStripeBits));
}

inline size_t CondVarStripeIndexB(const KConditionVariable* cv, u64 key) noexcept {
    u64 h = (key >> 2) ^ (static_cast<u64>(reinterpret_cast<uintptr_t>(cv)) >> 9);
    h ^= (h >> 17) ^ (h >> 35);
    return static_cast<size_t>((h * u64{0xBF58476D1CE4E5B9ULL}) >> (64 - CondVarStripeBits));
}

inline bool MayHaveCondVarWaiters(const KConditionVariable* cv, u64 key) noexcept {
    if (g_cv_waiter_stripes_a[CondVarStripeIndexA(cv, key)].load(std::memory_order_relaxed) == 0) {
        return false;
    }
    return g_cv_waiter_stripes_b[CondVarStripeIndexB(cv, key)].load(std::memory_order_relaxed) != 0;
}

inline void NoteCondVarWaiterRemoved(const KConditionVariable* cv, u64 key) noexcept {
    g_cv_waiter_stripes_a[CondVarStripeIndexA(cv, key)].fetch_sub(1, std::memory_order_relaxed);
    g_cv_waiter_stripes_b[CondVarStripeIndexB(cv, key)].fetch_sub(1, std::memory_order_relaxed);
}

struct ScopedCondVarStripePreparing {
    std::atomic<u32>& stripe_a;
    std::atomic<u32>& stripe_b;
    bool inserted = false;

    ScopedCondVarStripePreparing(const KConditionVariable* cv, u64 key) noexcept
        : stripe_a{g_cv_waiter_stripes_a[CondVarStripeIndexA(cv, key)]},
          stripe_b{g_cv_waiter_stripes_b[CondVarStripeIndexB(cv, key)]} {
        stripe_a.fetch_add(1, std::memory_order_relaxed);
        stripe_b.fetch_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    ~ScopedCondVarStripePreparing() {
        if (!inserted) {
            stripe_a.fetch_sub(1, std::memory_order_relaxed);
            stripe_b.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    YUZU_NON_COPYABLE(ScopedCondVarStripePreparing);
    YUZU_NON_MOVEABLE(ScopedCondVarStripePreparing);
};

struct PlainUserPageCache {
    u64 page{~u64{0}};
    uintptr_t base{};
};

u32* TryGetPlainUserU32(const KProcess& process, KProcessAddress address) noexcept {
    const u64 raw_addr = GetInteger(address);
    if (IsKernelAddress(address) || (raw_addr & (sizeof(u32) - 1)) != 0) {
        return nullptr;
    }
    const auto& pt = process.GetPageTable().GetImpl();
    const u64 masked = raw_addr & u64{0xffffffffffffULL};
    if ((masked >> pt.GetAddressSpaceBits()) != 0) {
        return nullptr;
    }
    const uintptr_t base = pt.entries[masked >> Core::Memory::YUZU_PAGEBITS].Pointer();
    return base != 0 ? reinterpret_cast<u32*>(base + masked) : nullptr;
}

u32* TryGetPlainUserU32Cached(const KProcess& process, KProcessAddress address,
                              PlainUserPageCache& cache) noexcept {
    const u64 raw_addr = GetInteger(address);
    if (IsKernelAddress(address) || (raw_addr & (sizeof(u32) - 1)) != 0) {
        return nullptr;
    }
    const u64 masked = raw_addr & u64{0xffffffffffffULL};
    const u64 page = masked >> Core::Memory::YUZU_PAGEBITS;
    if (page == cache.page) {
        return cache.base != 0 ? reinterpret_cast<u32*>(cache.base + masked) : nullptr;
    }
    const auto& pt = process.GetPageTable().GetImpl();
    if ((masked >> pt.GetAddressSpaceBits()) != 0) {
        return nullptr;
    }
    const uintptr_t base = pt.entries[page].Pointer();
    cache.page = page;
    cache.base = base;
    return base != 0 ? reinterpret_cast<u32*>(base + masked) : nullptr;
}

bool ReadFromUser(KProcess& process, u32* plain_ptr, u32* out, KProcessAddress address) {
    if (plain_ptr != nullptr) {
        *out = std::atomic_ref<u32>{*plain_ptr}.load(std::memory_order_acquire);
        return true;
    }
    if (u32* const ptr = TryGetPlainUserU32(process, address); ptr != nullptr) {
        *out = std::atomic_ref<u32>{*ptr}.load(std::memory_order_acquire);
        return true;
    }
    *out = process.GetMemory().Read32(GetInteger(address));
    return true;
}

bool WriteToUser(KProcess& process, u32* plain_ptr, KProcessAddress address, u32 val,
                 std::memory_order order = std::memory_order_release) {
    if (plain_ptr != nullptr) {
        std::atomic_ref<u32>{*plain_ptr}.store(val, order);
        return true;
    }
    if (u32* const ptr = TryGetPlainUserU32(process, address); ptr != nullptr) {
        std::atomic_ref<u32>{*ptr}.store(val, order);
        return true;
    }
    if (order != std::memory_order_relaxed) {
        std::atomic_thread_fence(std::memory_order_release);
    }
    process.GetMemory().Write32(GetInteger(address), val);
    return true;
}

void ClearUserWaiterFlag(KProcess& process, u32* plain_ptr, KProcessAddress address,
                         u32 zero_flag) {
    if (plain_ptr != nullptr) {
        std::atomic_ref<u32> word{*plain_ptr};
        if (word.load(std::memory_order_relaxed) != zero_flag) {
            word.store(zero_flag, std::memory_order_release);
        }
        return;
    }
    if (u32* const ptr = TryGetPlainUserU32(process, address); ptr != nullptr) {
        std::atomic_ref<u32> word{*ptr};
        if (word.load(std::memory_order_relaxed) != zero_flag) {
            word.store(zero_flag, std::memory_order_release);
        }
        return;
    }
    std::atomic_thread_fence(std::memory_order_release);
    process.GetMemory().Write32(GetInteger(address), zero_flag);
}

bool UpdateLockAtomic(KProcess& process, u32* plain_ptr, u32* out, KProcessAddress address,
                      u32 if_zero, u32 new_orr_mask) {
    u32* const ptr = plain_ptr != nullptr ? plain_ptr : TryGetPlainUserU32(process, address);
    if (ptr != nullptr) {
        std::atomic_ref<u32> word{*ptr};
        u32 expected = word.load(std::memory_order_acquire);
        while (true) {
            const u32 value = expected != 0 ? (expected | new_orr_mask) : if_zero;
            if (value == expected && expected != 0) {
                *out = expected;
                return true;
            }
            if (word.compare_exchange_weak(expected, value, std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
                *out = expected;
                return true;
            }
        }
    }

    auto& memory = process.GetMemory();
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

struct SignalOwnerCache {
    Handle handle{Svc::InvalidHandle};
    KThread* thread{};
};

void SignalWaiterThread(KernelCore& kernel, KProcess& process, KThread* thread,
                        PlainUserPageCache& page_cache, SignalOwnerCache& owner_cache) {
    ASSERT(KScheduler::IsSchedulerLockedByCurrentThread(kernel));

    const KProcessAddress address = thread->GetAddressKey();
    const u32 own_tag = thread->GetAddressKeyValue();
    u32* const addr_ptr = TryGetPlainUserU32Cached(process, address, page_cache);

    u32 prev_tag{};
    UpdateLockAtomic(process, addr_ptr, std::addressof(prev_tag), address, own_tag,
                     Svc::HandleWaitMask);

    if (prev_tag == Svc::InvalidHandle) {
        thread->EndWait(kernel, ResultSuccess);
        return;
    }

    const Handle owner_handle = static_cast<Handle>(prev_tag & ~Svc::HandleWaitMask);
    if (owner_cache.thread != nullptr && owner_handle == owner_cache.handle) {
        owner_cache.thread->AddWaiter(kernel, thread);
        return;
    }

    KThread* const owner_thread = process.GetHandleTable()
                                      .GetObjectWithoutPseudoHandle<KThread>(kernel, owner_handle)
                                      .ReleasePointerUnsafe();
    if (owner_thread != nullptr) {
        if (owner_cache.thread != nullptr) {
            owner_cache.thread->Close(kernel);
        }
        owner_cache.handle = owner_handle;
        owner_cache.thread = owner_thread;
        owner_thread->AddWaiter(kernel, thread);
    } else {
        thread->EndWait(kernel, ResultInvalidState);
    }
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
    KConditionVariable* m_cond_var;
    KConditionVariable::ThreadTree* m_tree;

public:
    explicit ThreadQueueImplForKConditionVariableWaitConditionVariable(
        KernelCore& kernel, KConditionVariable* cv, KConditionVariable::ThreadTree* t)
        : KThreadQueue(kernel)
        , m_cond_var(cv)
        , m_tree(t)
    {}

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // Remove the thread as a waiter from its owner.
        if (KThread* owner = waiting_thread->GetLockOwner(kernel); owner != nullptr) {
            owner->RemoveWaiter(kernel, waiting_thread);
        }

        // If the thread is waiting on a condvar, remove it from the tree.
        if (waiting_thread->IsWaitingForConditionVariable()) {
            const u64 key = waiting_thread->GetConditionVariableKey();
            m_tree->erase(m_tree->iterator_to(*waiting_thread));
            waiting_thread->ClearConditionVariable();
            NoteCondVarWaiterRemoved(m_cond_var, key);
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }
};

} // namespace

KConditionVariable::KConditionVariable(Core::System& system)
    : m_system{system}
{}

KConditionVariable::~KConditionVariable() {
    for (const auto& thread : m_tree) {
        NoteCondVarWaiterRemoved(this, thread.GetConditionVariableKey());
    }
}

Result KConditionVariable::SignalToAddress(KernelCore& kernel, KProcessAddress addr) {
    KThread* const owner_thread = GetCurrentThreadPointer(kernel);
    KProcess& process = *owner_thread->GetOwnerProcess();
    u32* const plain_ptr = TryGetPlainUserU32(process, addr);

    // Signal the address.
    {
        KScopedSchedulerLock sl(kernel);

        // Remove waiter thread.
        bool has_waiters{};
        KThread* const next_owner_thread =
            owner_thread->RemoveUserWaiterByKey(kernel, std::addressof(has_waiters), addr);

        // Determine the next tag.
        u32 next_value{};
        if (next_owner_thread != nullptr) {
            next_value = next_owner_thread->GetAddressKeyValue();
            if (has_waiters) {
                next_value |= Svc::HandleWaitMask;
            }
        }

        // Write the value to userspace with release ordering.
        Result result{ResultSuccess};
        if (WriteToUser(process, plain_ptr, addr, next_value, std::memory_order_release)) {
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
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    KProcess& process = *cur_thread->GetOwnerProcess();
    const u32 expected_tag = handle | Svc::HandleWaitMask;
    u32* const plain_ptr = TryGetPlainUserU32(process, addr);
    bool did_spin = false;

    // Resolve handle and optionally watch the lock word briefly outside KScopedSchedulerLock
    // when the lock owner is actively running on another physical core (so priority inheritance
    // is never delayed for a preempted owner).
    KScopedAutoObject<KThread> pre_owner{kernel};
    if (!cur_thread->IsTerminationRequested() && (GetInteger(addr) & 3) == 0) {
        u32* pointer = plain_ptr;
        if (pointer == nullptr) {
            pointer = reinterpret_cast<u32*>(process.GetMemory().GetPointerSilent(GetInteger(addr)));
        }
        if (pointer != nullptr) {
            const auto load_tag = [pointer] {
                return std::atomic_ref<u32>{*pointer}.load(std::memory_order_acquire);
            };
            if (load_tag() != expected_tag) {
                R_SUCCEED();
            }
            const s64 spin_us = LockSpinMicroseconds();
            const s32 current_core = cur_thread->GetCurrentCore();
            u8& credit = cur_thread->m_nx2_lock_spin_credit;
            if (spin_us > 0 && kernel.IsMulticore() && current_core >= 0 &&
                !KScheduler::HasStarvedRunnableForCore(kernel, current_core) &&
                (credit > 0 || (++cur_thread->m_nx2_lock_spin_probe & 15) == 0)) {
                did_spin = true;
                pre_owner =
                    process.GetHandleTable().GetObjectWithoutPseudoHandle<KThread>(kernel, handle);
                if (pre_owner.IsNotNull()) {
                    const s32 owner_core = pre_owner->GetCurrentCore();
                    if (owner_core >= 0 && owner_core != current_core &&
                        kernel.Scheduler(owner_core).GetSchedulerCurrentThread() ==
                            pre_owner.GetPointerUnsafe()) {
                        const PhysicalCore& core = kernel.CurrentPhysicalCore();
                        const Common::SpinDeadline::Deadline deadline{
                            std::chrono::microseconds{spin_us}};
                        bool released = false;
                        for (u32 spin = 1;; ++spin) {
                            LockSpinPause();
                            if (load_tag() != expected_tag) {
                                credit = static_cast<u8>((std::min)(static_cast<u32>(credit) + 2U, 16U));
                                released = true;
                                break;
                            }
                            if ((spin & 31U) == 0) {
                                if (core.IsInterrupted() ||
                                    KScheduler::HasStarvedRunnableForCore(kernel, current_core) ||
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
                    } else if (load_tag() != expected_tag &&
                               !cur_thread->IsTerminationRequested()) {
                        R_SUCCEED();
                    }
                }
            }
        }
    }

    // Wait for the address.
    ThreadQueueImplForKConditionVariableWaitForAddress wait_queue(kernel);
    KThread* owner_thread{};
    {
        KScopedSchedulerLock sl(kernel);

        // Check if the thread should terminate.
        R_UNLESS(!cur_thread->IsTerminationRequested(), ResultTerminationRequested);

        // Read the tag from userspace.
        u32* const locked_ptr = did_spin ? TryGetPlainUserU32(process, addr) : plain_ptr;
        u32 test_tag{};
        R_UNLESS(ReadFromUser(process, locked_ptr, std::addressof(test_tag), addr),
                 ResultInvalidCurrentMemory);

        // If the tag isn't the handle (with wait mask), we're done.
        R_SUCCEED_IF(test_tag != expected_tag);

        // Get the lock owner thread.
        owner_thread = pre_owner.IsNotNull()
                           ? pre_owner.ReleasePointerUnsafe()
                           : process.GetHandleTable()
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
    PlainUserPageCache page_cache{};
    SignalOwnerCache owner_cache{};
    SignalWaiterThread(kernel, GetCurrentProcess(kernel), thread, page_cache, owner_cache);
    if (owner_cache.thread != nullptr) {
        owner_cache.thread->Close(kernel);
    }
}

void KConditionVariable::Signal(u64 cv_key, s32 count) {
    // Dekker-linearizable fast path when no waiter is queued or preparing on cv_key and the
    // userspace waiter flag is already clear (writes nothing outside KScopedSchedulerLock).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!MayHaveCondVarWaiters(this, cv_key)) {
        auto& kernel = m_system.Kernel();
        if (const KThread* const cur_thread = GetCurrentThreadPointer(kernel);
            cur_thread != nullptr) {
            if (const KProcess* const process = cur_thread->GetOwnerProcess(); process != nullptr) {
                if (u32* const cv_ptr = TryGetPlainUserU32(*process, cv_key);
                    cv_ptr != nullptr &&
                    std::atomic_ref<u32>{*cv_ptr}.load(std::memory_order_acquire) == 0 &&
                    !MayHaveCondVarWaiters(this, cv_key)) {
                    return;
                }
            }
        }
    }

    SignalSlow(cv_key, count);
}

NX2_CV_NOINLINE void KConditionVariable::SignalSlow(u64 cv_key, s32 count) {
    auto& kernel = m_system.Kernel();
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    KProcess* const process = cur_thread != nullptr ? cur_thread->GetOwnerProcess() : nullptr;
    PlainUserPageCache page_cache{};
    u32* const cv_ptr =
        process != nullptr ? TryGetPlainUserU32Cached(*process, cv_key, page_cache) : nullptr;

    // Perform signaling.
    s32 num_waiters{};
    SignalOwnerCache owner_cache{};
    {
        KScopedSchedulerLock sl(kernel);
        KProcess& target_process = process != nullptr ? *process : GetCurrentProcess(kernel);

        const bool may_have_waiters = !m_tree.empty() && MayHaveCondVarWaiters(this, cv_key);
        auto it = may_have_waiters ? m_tree.nfind_key({cv_key, -1}) : m_tree.end();
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetConditionVariableKey() == cv_key)) {
            KThread* target_thread = std::addressof(*it);

            it = m_tree.erase(it);
            target_thread->ClearConditionVariable();
            NoteCondVarWaiterRemoved(this, cv_key);

            SignalWaiterThread(kernel, target_process, target_thread, page_cache, owner_cache);

            ++num_waiters;
        }

        // If we have no waiters, clear the has waiter flag.
        if (it == m_tree.end() || it->GetConditionVariableKey() != cv_key) {
            constexpr u32 HasNoWaiterFlag = 0;
            ClearUserWaiterFlag(target_process, cv_ptr, cv_key, HasNoWaiterFlag);
        }
    }
    if (owner_cache.thread != nullptr) {
        owner_cache.thread->Close(kernel);
    }
}

Result KConditionVariable::Wait(KProcessAddress addr, u64 key, u32 value, s64 timeout) {
    // Prepare to wait.
    auto& kernel = m_system.Kernel();
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    KProcess& process = *cur_thread->GetOwnerProcess();
    PlainUserPageCache page_cache{};
    u32* const key_ptr = TryGetPlainUserU32Cached(process, key, page_cache);
    u32* const addr_ptr = TryGetPlainUserU32Cached(process, addr, page_cache);
    KHardwareTimer* timer{};
    ThreadQueueImplForKConditionVariableWaitConditionVariable wait_queue(
        kernel, this, std::addressof(m_tree));

    {
        ScopedCondVarStripePreparing stripe_guard{this, key};
        KScopedSchedulerLockAndSleep slp(kernel, std::addressof(timer), cur_thread, timeout);

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
                cur_thread->RemoveUserWaiterByKey(kernel, std::addressof(has_waiters), addr);

            // Update for the next owner thread.
            u32 next_value{};
            if (next_owner_thread != nullptr) {
                // Get the next tag value.
                next_value = next_owner_thread->GetAddressKeyValue();
                if (has_waiters) {
                    next_value |= Svc::HandleWaitMask;
                }

                // Wake up the next owner.
                next_owner_thread->EndWait(kernel, ResultSuccess);
            }

            // Write to the cv key.
            {
                constexpr u32 HasWaiterFlag = 1;
                WriteToUser(process, key_ptr, key, HasWaiterFlag, std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }

            // Write the value to userspace.
            if (!WriteToUser(process, addr_ptr, addr, next_value, std::memory_order_release)) {
                slp.CancelSleep();
                R_THROW(ResultInvalidCurrentMemory);
            }
        }

        // If timeout is zero, time out.
        R_UNLESS(timeout != 0, ResultTimedOut);

        // Update condition variable tracking.
        cur_thread->SetConditionVariable(std::addressof(m_tree), addr, key, value);
        m_tree.insert(*cur_thread);
        stripe_guard.inserted = true;

        // Begin waiting.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(kernel, std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::ConditionVar);
    }

    // Get the wait result.
    R_RETURN(cur_thread->GetWaitResult());
}

} // namespace Kernel
