// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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
#include "core/hle/kernel/k_address_arbiter.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/k_typed_address.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/kernel/svc_results.h"
#include "core/memory.h"

namespace Kernel {

KAddressArbiter::KAddressArbiter(Core::System& system_)
    : system{system_}
{}
KAddressArbiter::~KAddressArbiter() = default;

namespace {

bool ReadFromUser(KernelCore& kernel, s32* out, KProcessAddress address) {
    *out = GetCurrentMemory(kernel).Read32(GetInteger(address));
    return true;
}

bool ReadFromUser(KernelCore& kernel, s64* out, KProcessAddress address) {
    auto& memory = GetCurrentMemory(kernel);
    const u64 addr = GetInteger(address);

    // Reject ranges that wrap past the end of the address space; the range check accepts them.
    if (addr + sizeof(s64) < addr || !memory.IsValidVirtualAddressRange(addr, sizeof(s64))) {
        return false;
    }

    *out = static_cast<s64>(memory.Read64(addr));
    return true;
}

bool DecrementIfLessThan(KernelCore& kernel, s32* out, KProcessAddress address, s32 value) {
    auto& memory = GetCurrentMemory(kernel);
    const u64 raw_addr = GetInteger(address);

    s32 current_value = static_cast<s32>(memory.Read32(raw_addr));
    while (current_value < value) {
        const s32 decrement_value = current_value - 1;
        if (memory.WriteExclusive32(raw_addr, static_cast<u32>(decrement_value),
                                    static_cast<u32>(current_value))) {
            break;
        }
        current_value = static_cast<s32>(memory.Read32(raw_addr));
    }

    *out = current_value;
    return true;
}

bool UpdateIfEqual(KernelCore& kernel, s32* out, KProcessAddress address, s32 value,
                   s32 new_value) {
    auto& memory = GetCurrentMemory(kernel);
    const u64 raw_addr = GetInteger(address);

    s32 current_value = static_cast<s32>(memory.Read32(raw_addr));
    while (current_value == value) {
        if (memory.WriteExclusive32(raw_addr, static_cast<u32>(new_value),
                                    static_cast<u32>(current_value))) {
            break;
        }
        current_value = static_cast<s32>(memory.Read32(raw_addr));
    }

    *out = current_value;
    return true;
}

void ArbSpinPause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

struct ScopedStripePreparing {
    std::atomic<u32>& stripe;
    bool inserted = false;

    explicit ScopedStripePreparing(std::atomic<u32>& s) : stripe{s} {
        stripe.fetch_add(1, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    ~ScopedStripePreparing() {
        if (!inserted) {
            stripe.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    YUZU_NON_COPYABLE(ScopedStripePreparing);
    YUZU_NON_MOVEABLE(ScopedStripePreparing);
};

/// EDEN_NX2_ARB_SPIN: microseconds a thread about to wait on an address first watches the word
/// (default 8; 0 turns it off).
s64 ArbSpinMicroseconds() {
    static const s64 microseconds = [] {
        const char* value = std::getenv("EDEN_NX2_ARB_SPIN");
        if (value == nullptr || value[0] == '\0') {
            return s64{8};
        }
        return (std::max)(s64{0}, static_cast<s64>(std::strtoll(value, nullptr, 10)));
    }();
    return microseconds;
}

/// Guest job systems hand work between threads through these waits many times a frame, and the
/// signal often comes a few microseconds later. Before the thread sleeps (scheduler lock, context
/// switch, then the signaler's EndWait, interrupt and switch back), it watches the word briefly
/// without any lock. If the word stops satisfying the wait, the wait ends with ResultInvalidState,
/// exactly what the kernel returns to a thread that entered the SVC that much later. Threads whose
/// spins rarely pay off stop spinning (m_nx2_arb_spin_credit).
template <typename StillWaits>
bool SpinBeforeWait(KernelCore& kernel, KThread* cur_thread, uint64_t addr, bool is_64_bit,
                    StillWaits&& still_waits) {
    const s64 spin_us = ArbSpinMicroseconds();
    if (spin_us == 0 || !kernel.IsMulticore() || (addr & (is_64_bit ? 7 : 3)) != 0) {
        return false;
    }
    // Yield the core immediately when another runnable thread is already queued on it.
    if (const s32 core_id = GetCurrentCoreId(kernel);
        core_id >= 0 && KScheduler::GetCoreScheduledCount(kernel, core_id) > 1) {
        return false;
    }
    u8& credit = cur_thread->m_nx2_arb_spin_credit;
    if (credit == 0 && (++cur_thread->m_nx2_arb_spin_probe & 15) != 0) {
        return false;
    }
    u8* const pointer = GetCurrentMemory(kernel).GetPointerSilent(addr);
    if (pointer == nullptr) {
        return false;
    }
    const auto load_word = [&]() -> s64 {
        if (is_64_bit) {
            return std::atomic_ref<s64>{*reinterpret_cast<s64*>(pointer)}.load(
                std::memory_order_acquire);
        }
        return std::atomic_ref<s32>{*reinterpret_cast<s32*>(pointer)}.load(
            std::memory_order_acquire);
    };
    if (!still_waits(load_word())) {
        credit = static_cast<u8>((std::min)(credit + 2, 16));
        return !cur_thread->IsTerminationRequested();
    }
    const PhysicalCore& core = kernel.CurrentPhysicalCore();
    const s32 current_core = GetCurrentCoreId(kernel);
    const Common::SpinDeadline::Deadline deadline{std::chrono::microseconds{spin_us}};
    for (u32 spin = 1;; ++spin) {
        ArbSpinPause();
        if (!still_waits(load_word())) {
            credit = static_cast<u8>((std::min)(credit + 2, 16));
            // A terminating thread takes the normal path, which reports the termination first.
            return !cur_thread->IsTerminationRequested();
        }
        if ((spin & 31) == 0) {
            // Something else must run on this core: stop watching.
            if (core.IsInterrupted() ||
                (current_core >= 0 && KScheduler::GetCoreScheduledCount(kernel, current_core) > 1)) {
                return false;
            }
            if (deadline.Expired()) {
                break;
            }
        }
    }
    if (credit > 0) {
        --credit;
    }
    return false;
}

class ThreadQueueImplForKAddressArbiter final : public KThreadQueue {
public:
    explicit ThreadQueueImplForKAddressArbiter(KernelCore& kernel, KAddressArbiter* arbiter,
                                               KAddressArbiter::ThreadTree* t)
        : KThreadQueue(kernel), m_arbiter(arbiter), m_tree(t) {}

    void CancelWait(KernelCore& kernel, KThread* waiting_thread, Result wait_result, bool cancel_timer_task) override {
        // If the thread is waiting on an address arbiter, remove it from the tree.
        if (waiting_thread->IsWaitingForAddressArbiter()) {
            const uint64_t key = waiting_thread->GetAddressArbiterKey();
            m_tree->erase(m_tree->iterator_to(*waiting_thread));
            waiting_thread->ClearAddressArbiter();
            m_arbiter->NoteWaiterRemoved(key);
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    KAddressArbiter* m_arbiter{};
    KAddressArbiter::ThreadTree* m_tree{};
};

} // namespace

Result KAddressArbiter::Signal(uint64_t addr, s32 count) {
    // Dekker-linearizable fast path: pairs with ScopedStripePreparing's seq_cst increment+fence
    // before ReadFromUser in WaitIfLessThan/WaitIfEqual.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (WaiterStripe(addr).load(std::memory_order_seq_cst) == 0) {
        R_SUCCEED();
    }

    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());
        if (m_tree.empty()) {
            R_SUCCEED();
        }

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteWaiterRemoved(addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndIncrementIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        // Check the userspace value.
        s32 user_value{};
        R_UNLESS(UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, value + 1),
                 ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        if (m_tree.empty()) {
            R_SUCCEED();
        }

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteWaiterRemoved(addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndModifyByWaitingCountIfEqual(uint64_t addr, s32 value, s32 count) {
    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(system.Kernel());

        auto it = m_tree.empty() ? m_tree.end() : m_tree.nfind_key({addr, -1});
        // Determine the updated value.
        s32 new_value{};
        if (count <= 0) {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                new_value = value - 2;
            } else {
                new_value = value + 1;
            }
        } else {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                auto tmp_it = it;
                s32 tmp_num_waiters{};
                while (++tmp_it != m_tree.end() && tmp_it->GetAddressArbiterKey() == addr) {
                    if (tmp_num_waiters++ >= count) {
                        break;
                    }
                }

                if (tmp_num_waiters < count) {
                    new_value = value - 1;
                } else {
                    new_value = value;
                }
            } else {
                new_value = value + 1;
            }
        }

        // Check the userspace value.
        s32 user_value{};
        bool succeeded{};
        if (value != new_value) {
            succeeded = UpdateIfEqual(system.Kernel(), std::addressof(user_value), addr, value, new_value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        R_UNLESS(succeeded, ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(system.Kernel(), ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteWaiterRemoved(addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::WaitIfLessThan(uint64_t addr, s32 value, bool decrement, s64 timeout) {
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());
    if (timeout != 0 && SpinBeforeWait(system.Kernel(), cur_thread, addr, false,
                                       [value](s64 user_value) { return user_value < value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), this, std::addressof(m_tree));

    {
        ScopedStripePreparing stripe_guard{WaiterStripe(addr)};
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        s32 user_value{};
        bool succeeded{};
        if (decrement) {
            succeeded = DecrementIfLessThan(system.Kernel(), std::addressof(user_value), addr, value);
        } else {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        }

        if (!succeeded) {
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is less than the specified one.
        if (user_value >= value) {
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);
        stripe_guard.inserted = true;

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

Result KAddressArbiter::WaitIfEqual(uint64_t addr, s64 value, s64 timeout, bool is_64_bit) {
    // Prepare to wait.
    KThread* cur_thread = GetCurrentThreadPointer(system.Kernel());
    if (timeout != 0 && SpinBeforeWait(system.Kernel(), cur_thread, addr, is_64_bit,
                                       [value](s64 user_value) { return user_value == value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(system.Kernel(), this, std::addressof(m_tree));

    {
        ScopedStripePreparing stripe_guard{WaiterStripe(addr)};
        KScopedSchedulerLockAndSleep slp{system.Kernel(), std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        s64 user_value{};
        bool succeeded{};
        if (is_64_bit) {
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value), addr);
        } else {
            s32 user_value32{};
            succeeded = ReadFromUser(system.Kernel(), std::addressof(user_value32), addr);
            user_value = user_value32;
        }
        if (!succeeded) {
            slp.CancelSleep();
            R_THROW(ResultInvalidCurrentMemory);
        }

        // Check that the value is equal.
        if (value != user_value) {
            slp.CancelSleep();
            R_THROW(ResultInvalidState);
        }

        // Check that the timeout is non-zero.
        if (timeout == 0) {
            slp.CancelSleep();
            R_THROW(ResultTimedOut);
        }

        // Set the arbiter.
        cur_thread->SetAddressArbiter(std::addressof(m_tree), addr);
        m_tree.insert(*cur_thread);
        stripe_guard.inserted = true;

        // Wait for the thread to finish.
        wait_queue.SetHardwareTimer(timer);
        cur_thread->BeginWait(system.Kernel(), std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

} // namespace Kernel
