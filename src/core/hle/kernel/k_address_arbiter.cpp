// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

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
#include "core/hle/kernel/k_address_arbiter.h"
#include "core/hle/kernel/k_memory_layout.h"
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

namespace {

#if defined(_MSC_VER)
#define NX2_ARB_NOINLINE __declspec(noinline)
#else
#define NX2_ARB_NOINLINE __attribute__((noinline))
#endif

constexpr size_t ArbiterStripeBits = 12;
constexpr size_t NumArbiterWaiterStripes = size_t{1} << ArbiterStripeBits;
static_assert(NumArbiterWaiterStripes == 4096);

alignas(128) std::array<std::atomic<u32>, NumArbiterWaiterStripes> g_arb_waiter_stripes_a{};
alignas(128) std::array<std::atomic<u32>, NumArbiterWaiterStripes> g_arb_waiter_stripes_b{};

inline size_t ArbiterStripeIndexA(const KAddressArbiter* arb, uint64_t addr) noexcept {
    uint64_t h = (addr >> 2) ^ (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arb)) >> 4);
    h ^= (h >> 23) ^ (h >> 41);
    return static_cast<size_t>((h * uint64_t{0x9E3779B97F4A7C15ULL}) >> (64 - ArbiterStripeBits));
}

inline size_t ArbiterStripeIndexB(const KAddressArbiter* arb, uint64_t addr) noexcept {
    uint64_t h = (addr >> 2) ^ (static_cast<uint64_t>(reinterpret_cast<uintptr_t>(arb)) >> 9);
    h ^= (h >> 17) ^ (h >> 35);
    return static_cast<size_t>((h * uint64_t{0xBF58476D1CE4E5B9ULL}) >> (64 - ArbiterStripeBits));
}

inline bool MayHaveArbiterWaiters(const KAddressArbiter* arb, uint64_t addr) noexcept {
    if (g_arb_waiter_stripes_a[ArbiterStripeIndexA(arb, addr)].load(std::memory_order_relaxed) ==
        0) {
        return false;
    }
    return g_arb_waiter_stripes_b[ArbiterStripeIndexB(arb, addr)].load(std::memory_order_relaxed) !=
           0;
}

inline void NoteArbiterWaiterRemoved(const KAddressArbiter* arb, uint64_t addr) noexcept {
    g_arb_waiter_stripes_a[ArbiterStripeIndexA(arb, addr)].fetch_sub(1, std::memory_order_relaxed);
    g_arb_waiter_stripes_b[ArbiterStripeIndexB(arb, addr)].fetch_sub(1, std::memory_order_relaxed);
}

struct ScopedStripePreparing {
    std::atomic<u32>& stripe_a;
    std::atomic<u32>& stripe_b;
    bool inserted = false;

    ScopedStripePreparing(const KAddressArbiter* arb, uint64_t addr) noexcept
        : stripe_a{g_arb_waiter_stripes_a[ArbiterStripeIndexA(arb, addr)]},
          stripe_b{g_arb_waiter_stripes_b[ArbiterStripeIndexB(arb, addr)]} {
        stripe_a.fetch_add(1, std::memory_order_relaxed);
        stripe_b.fetch_add(1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    ~ScopedStripePreparing() {
        if (!inserted) {
            stripe_a.fetch_sub(1, std::memory_order_relaxed);
            stripe_b.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    YUZU_NON_COPYABLE(ScopedStripePreparing);
    YUZU_NON_MOVEABLE(ScopedStripePreparing);
};

u8* TryGetPlainUserPointer(const KProcess& process, uint64_t addr, size_t size) noexcept {
    if (IsKernelAddress(addr) || (addr & (size - 1)) != 0) {
        return nullptr;
    }
    const auto& pt = process.GetPageTable().GetImpl();
    const uint64_t masked = addr & uint64_t{0xffffffffffffULL};
    if ((masked >> pt.GetAddressSpaceBits()) != 0) {
        return nullptr;
    }
    const uintptr_t base = pt.entries[masked >> Core::Memory::YUZU_PAGEBITS].Pointer();
    return base != 0 ? reinterpret_cast<u8*>(base + masked) : nullptr;
}

bool ReadFromUser(KProcess& process, u8* plain_ptr, s32* out, KProcessAddress address) {
    if (plain_ptr != nullptr) {
        *out =
            std::atomic_ref<s32>{*reinterpret_cast<s32*>(plain_ptr)}.load(std::memory_order_acquire);
        return true;
    }
    const u64 raw_addr = GetInteger(address);
    if (u8* const ptr = TryGetPlainUserPointer(process, raw_addr, sizeof(s32)); ptr != nullptr) {
        *out = std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire);
        return true;
    }
    *out = static_cast<s32>(process.GetMemory().Read32(raw_addr));
    return true;
}

bool ReadFromUser(KProcess& process, u8* plain_ptr, s64* out, KProcessAddress address) {
    if (plain_ptr != nullptr) {
        *out =
            std::atomic_ref<s64>{*reinterpret_cast<s64*>(plain_ptr)}.load(std::memory_order_acquire);
        return true;
    }
    const u64 addr = GetInteger(address);
    if (u8* const ptr = TryGetPlainUserPointer(process, addr, sizeof(s64)); ptr != nullptr) {
        *out = std::atomic_ref<s64>{*reinterpret_cast<s64*>(ptr)}.load(std::memory_order_acquire);
        return true;
    }

    auto& memory = process.GetMemory();
    // Reject ranges that wrap past the end of the address space; the range check accepts them.
    if (addr + sizeof(s64) < addr || !memory.IsValidVirtualAddressRange(addr, sizeof(s64))) {
        return false;
    }

    *out = static_cast<s64>(memory.Read64(addr));
    return true;
}

bool DecrementIfLessThan(KProcess& process, u8* plain_ptr, s32* out, KProcessAddress address,
                         s32 value) {
    const u64 raw_addr = GetInteger(address);
    u8* const ptr =
        plain_ptr != nullptr ? plain_ptr : TryGetPlainUserPointer(process, raw_addr, sizeof(s32));
    if (ptr != nullptr) {
        std::atomic_ref<s32> word{*reinterpret_cast<s32*>(ptr)};
        s32 current_value = word.load(std::memory_order_acquire);
        while (current_value < value) {
            const s32 decrement_value = static_cast<s32>(static_cast<u32>(current_value) - 1U);
            if (word.compare_exchange_weak(current_value, decrement_value,
                                           std::memory_order_acq_rel, std::memory_order_acquire)) {
                break;
            }
        }
        *out = current_value;
        return true;
    }

    auto& memory = process.GetMemory();
    s32 current_value = static_cast<s32>(memory.Read32(raw_addr));
    while (current_value < value) {
        const s32 decrement_value = static_cast<s32>(static_cast<u32>(current_value) - 1U);
        if (memory.WriteExclusive32(raw_addr, static_cast<u32>(decrement_value),
                                    static_cast<u32>(current_value))) {
            break;
        }
        current_value = static_cast<s32>(memory.Read32(raw_addr));
    }

    *out = current_value;
    return true;
}

bool UpdateIfEqual(KProcess& process, u8* plain_ptr, s32* out, KProcessAddress address, s32 value,
                   s32 new_value) {
    const u64 raw_addr = GetInteger(address);
    u8* const ptr =
        plain_ptr != nullptr ? plain_ptr : TryGetPlainUserPointer(process, raw_addr, sizeof(s32));
    if (ptr != nullptr) {
        std::atomic_ref<s32> word{*reinterpret_cast<s32*>(ptr)};
        s32 current_value = word.load(std::memory_order_acquire);
        while (current_value == value) {
            if (word.compare_exchange_weak(current_value, new_value, std::memory_order_acq_rel,
                                           std::memory_order_acquire)) {
                break;
            }
        }
        *out = current_value;
        return true;
    }

    auto& memory = process.GetMemory();
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
bool SpinBeforeWait(KernelCore& kernel, KThread* cur_thread, KProcess& process, uint64_t addr,
                    u8* plain_ptr, bool is_64_bit, StillWaits&& still_waits) {
    const s64 spin_us = ArbSpinMicroseconds();
    if (spin_us == 0 || !kernel.IsMulticore() || (addr & (is_64_bit ? 7 : 3)) != 0) {
        return false;
    }
    const s32 current_core = cur_thread->GetCurrentCore();
    // Yield the core immediately when another runnable thread needs it.
    if (current_core >= 0 && KScheduler::HasStarvedRunnableForCore(kernel, current_core)) {
        return false;
    }
    u8& credit = cur_thread->m_nx2_arb_spin_credit;
    if (credit == 0 && (++cur_thread->m_nx2_arb_spin_probe & 15) != 0) {
        return false;
    }
    u8* const pointer =
        plain_ptr != nullptr ? plain_ptr : process.GetMemory().GetPointerSilent(addr);
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
    if (plain_ptr == nullptr && !still_waits(load_word())) {
        credit = static_cast<u8>((std::min)(static_cast<u32>(credit) + 2U, 16U));
        return !cur_thread->IsTerminationRequested();
    }
    const PhysicalCore& core = kernel.CurrentPhysicalCore();
    const Common::SpinDeadline::Deadline deadline{std::chrono::microseconds{spin_us}};
    for (u32 spin = 1;; ++spin) {
        ArbSpinPause();
        if (!still_waits(load_word())) {
            credit = static_cast<u8>((std::min)(static_cast<u32>(credit) + 2U, 16U));
            // A terminating thread takes the normal path, which reports the termination first.
            return !cur_thread->IsTerminationRequested();
        }
        if ((spin & 31U) == 0) {
            // Something else must run on this core: stop watching.
            if (core.IsInterrupted() ||
                (current_core >= 0 && KScheduler::HasStarvedRunnableForCore(kernel, current_core))) {
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
            NoteArbiterWaiterRemoved(m_arbiter, key);
        }

        // Invoke the base cancel wait handler.
        KThreadQueue::CancelWait(kernel, waiting_thread, wait_result, cancel_timer_task);
    }

private:
    KAddressArbiter* m_arbiter{};
    KAddressArbiter::ThreadTree* m_tree{};
};

bool TryFastWaitWithPointer(KThread* cur_thread, u8* ptr, Svc::ArbitrationType type, s64 value,
                            s64 timeout_ns, Result* out_result) noexcept {
    if (ptr == nullptr || cur_thread->IsTerminationRequested()) {
        return false;
    }
    switch (type) {
    case Svc::ArbitrationType::WaitIfLessThan: {
        const s32 cur =
            std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire);
        if (cur >= static_cast<s32>(value)) {
            cur_thread->m_nx2_arb_spin_credit =
                static_cast<u8>((std::min)(static_cast<u32>(cur_thread->m_nx2_arb_spin_credit) + 2U, 16U));
            *out_result = ResultInvalidState;
            return true;
        }
        if (timeout_ns == 0) {
            *out_result = ResultTimedOut;
            return true;
        }
        return false;
    }
    case Svc::ArbitrationType::DecrementAndWaitIfLessThan: {
        const s32 cur =
            std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(std::memory_order_acquire);
        if (cur >= static_cast<s32>(value)) {
            cur_thread->m_nx2_arb_spin_credit =
                static_cast<u8>((std::min)(static_cast<u32>(cur_thread->m_nx2_arb_spin_credit) + 2U, 16U));
            *out_result = ResultInvalidState;
            return true;
        }
        return false;
    }
    case Svc::ArbitrationType::WaitIfEqual:
    case Svc::ArbitrationType::WaitIfEqual64: {
        const bool is_64 = type == Svc::ArbitrationType::WaitIfEqual64;
        const s64 target = is_64 ? value : static_cast<s64>(static_cast<s32>(value));
        const s64 cur = is_64
                            ? std::atomic_ref<s64>{*reinterpret_cast<s64*>(ptr)}.load(
                                  std::memory_order_acquire)
                            : static_cast<s64>(std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}
                                                   .load(std::memory_order_acquire));
        if (cur != target) {
            cur_thread->m_nx2_arb_spin_credit =
                static_cast<u8>((std::min)(static_cast<u32>(cur_thread->m_nx2_arb_spin_credit) + 2U, 16U));
            *out_result = ResultInvalidState;
            return true;
        }
        if (timeout_ns == 0) {
            *out_result = ResultTimedOut;
            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

NX2_ARB_NOINLINE bool TryFastConditionalSignalMismatch(Core::System& sys, uint64_t addr, s32 value,
                                                       Result* out_result) {
    if (const KThread* const cur_thread = GetCurrentThreadPointer(sys.Kernel());
        cur_thread != nullptr) {
        if (const KProcess* const process = cur_thread->GetOwnerProcess(); process != nullptr) {
            if (u8* const ptr = TryGetPlainUserPointer(*process, addr, sizeof(s32));
                ptr != nullptr &&
                std::atomic_ref<s32>{*reinterpret_cast<s32*>(ptr)}.load(
                    std::memory_order_acquire) != value) {
                *out_result = ResultInvalidState;
                return true;
            }
        }
    }
    return false;
}

} // namespace

KAddressArbiter::KAddressArbiter(Core::System& system_)
    : system{system_}
{}

KAddressArbiter::~KAddressArbiter() {
    for (const auto& thread : m_tree) {
        NoteArbiterWaiterRemoved(this, thread.GetAddressArbiterKey());
    }
}

bool KAddressArbiter::TryFastSignalToAddress(uint64_t addr, Svc::SignalType type, s32 value,
                                             s32 /*count*/, Result* out_result) {
    if (type == Svc::SignalType::Signal) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!MayHaveArbiterWaiters(this, addr)) {
            *out_result = ResultSuccess;
            return true;
        }
        return false;
    }
    if (type == Svc::SignalType::SignalAndIncrementIfEqual ||
        type == Svc::SignalType::SignalAndModifyByWaitingCountIfEqual) {
        return TryFastConditionalSignalMismatch(system, addr, value, out_result);
    }
    return false;
}

bool KAddressArbiter::TryFastWaitForAddress(uint64_t addr, Svc::ArbitrationType type, s64 value,
                                            s64 timeout_ns, Result* out_result) {
    KThread* const cur_thread = GetCurrentThreadPointer(system.Kernel());
    if (cur_thread == nullptr) {
        return false;
    }
    const KProcess* const process = cur_thread->GetOwnerProcess();
    if (process == nullptr) {
        return false;
    }
    const size_t size = type == Svc::ArbitrationType::WaitIfEqual64 ? sizeof(s64) : sizeof(s32);
    u8* const ptr = TryGetPlainUserPointer(*process, addr, size);
    return TryFastWaitWithPointer(cur_thread, ptr, type, value, timeout_ns, out_result);
}

Result KAddressArbiter::Signal(uint64_t addr, s32 count) {
    // Dekker-linearizable fast path: pairs with ScopedStripePreparing's increment + seq_cst fence
    // before ReadFromUser in WaitIfLessThan/WaitIfEqual.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!MayHaveArbiterWaiters(this, addr)) {
        R_SUCCEED();
    }
    return SignalSlow(addr, count);
}

NX2_ARB_NOINLINE Result KAddressArbiter::SignalSlow(uint64_t addr, s32 count) {
    auto& kernel = system.Kernel();
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(kernel);
        if (m_tree.empty() || !MayHaveArbiterWaiters(this, addr)) {
            R_SUCCEED();
        }

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(kernel, ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteArbiterWaiterRemoved(this, addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndIncrementIfEqual(uint64_t addr, s32 value, s32 count) {
    auto& kernel = system.Kernel();
    KProcess& process = *GetCurrentThreadPointer(kernel)->GetOwnerProcess();
    u8* const plain_ptr = TryGetPlainUserPointer(process, addr, sizeof(s32));
    if (plain_ptr != nullptr &&
        std::atomic_ref<s32>{*reinterpret_cast<s32*>(plain_ptr)}.load(std::memory_order_acquire) !=
            value) {
        R_THROW(ResultInvalidState);
    }

    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(kernel);

        // Check the userspace value.
        s32 user_value{};
        const s32 incremented = static_cast<s32>(static_cast<u32>(value) + 1U);
        R_UNLESS(UpdateIfEqual(process, plain_ptr, std::addressof(user_value), addr, value,
                               incremented),
                 ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        if (m_tree.empty() || !MayHaveArbiterWaiters(this, addr)) {
            R_SUCCEED();
        }

        auto it = m_tree.nfind_key({addr, -1});
        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(kernel, ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteArbiterWaiterRemoved(this, addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::SignalAndModifyByWaitingCountIfEqual(uint64_t addr, s32 value, s32 count) {
    auto& kernel = system.Kernel();
    KProcess& process = *GetCurrentThreadPointer(kernel)->GetOwnerProcess();
    u8* const plain_ptr = TryGetPlainUserPointer(process, addr, sizeof(s32));
    if (plain_ptr != nullptr &&
        std::atomic_ref<s32>{*reinterpret_cast<s32*>(plain_ptr)}.load(std::memory_order_acquire) !=
            value) {
        R_THROW(ResultInvalidState);
    }

    // Perform signaling.
    s32 num_waiters{};
    {
        KScopedSchedulerLock sl(kernel);

        const bool may_have_waiters = !m_tree.empty() && MayHaveArbiterWaiters(this, addr);
        auto it = may_have_waiters ? m_tree.nfind_key({addr, -1}) : m_tree.end();
        // Determine the updated value.
        s32 new_value{};
        const u32 uval = static_cast<u32>(value);
        if (count <= 0) {
            if (it != m_tree.end() && it->GetAddressArbiterKey() == addr) {
                new_value = static_cast<s32>(uval - 2U);
            } else {
                new_value = static_cast<s32>(uval + 1U);
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
                    new_value = static_cast<s32>(uval - 1U);
                } else {
                    new_value = value;
                }
            } else {
                new_value = static_cast<s32>(uval + 1U);
            }
        }

        // Check the userspace value.
        s32 user_value{};
        bool succeeded{};
        if (value != new_value) {
            succeeded = UpdateIfEqual(process, plain_ptr, std::addressof(user_value), addr, value,
                                      new_value);
        } else {
            succeeded = ReadFromUser(process, plain_ptr, std::addressof(user_value), addr);
        }

        R_UNLESS(succeeded, ResultInvalidCurrentMemory);
        R_UNLESS(user_value == value, ResultInvalidState);

        while ((it != m_tree.end()) && (count <= 0 || num_waiters < count) &&
               (it->GetAddressArbiterKey() == addr)) {
            // End the thread's wait.
            KThread* target_thread = std::addressof(*it);
            target_thread->EndWait(kernel, ResultSuccess);

            ASSERT(target_thread->IsWaitingForAddressArbiter());
            target_thread->ClearAddressArbiter();

            it = m_tree.erase(it);
            NoteArbiterWaiterRemoved(this, addr);
            ++num_waiters;
        }
    }
    R_SUCCEED();
}

Result KAddressArbiter::WaitIfLessThan(uint64_t addr, s32 value, bool decrement, s64 timeout) {
    auto& kernel = system.Kernel();
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    KProcess& process = *cur_thread->GetOwnerProcess();
    u8* const plain_ptr = TryGetPlainUserPointer(process, addr, sizeof(s32));
    {
        Result fast_res = ResultSuccess;
        const auto arb_type = decrement ? Svc::ArbitrationType::DecrementAndWaitIfLessThan
                                        : Svc::ArbitrationType::WaitIfLessThan;
        if (TryFastWaitWithPointer(cur_thread, plain_ptr, arb_type, value, timeout, &fast_res)) {
            return fast_res;
        }
    }
    if (timeout != 0 && SpinBeforeWait(kernel, cur_thread, process, addr, plain_ptr, false,
                                       [value](s64 user_value) { return user_value < value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(kernel, this, std::addressof(m_tree));

    {
        ScopedStripePreparing stripe_guard{this, addr};
        KScopedSchedulerLockAndSleep slp{kernel, std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        u8* const locked_ptr = TryGetPlainUserPointer(process, addr, sizeof(s32));
        s32 user_value{};
        bool succeeded{};
        if (decrement) {
            succeeded =
                DecrementIfLessThan(process, locked_ptr, std::addressof(user_value), addr, value);
        } else {
            succeeded = ReadFromUser(process, locked_ptr, std::addressof(user_value), addr);
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
        cur_thread->BeginWait(kernel, std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

Result KAddressArbiter::WaitIfEqual(uint64_t addr, s64 value, s64 timeout, bool is_64_bit) {
    auto& kernel = system.Kernel();
    KThread* const cur_thread = GetCurrentThreadPointer(kernel);
    KProcess& process = *cur_thread->GetOwnerProcess();
    const size_t word_size = is_64_bit ? sizeof(s64) : sizeof(s32);
    u8* const plain_ptr = TryGetPlainUserPointer(process, addr, word_size);
    {
        Result fast_res = ResultSuccess;
        if (TryFastWaitWithPointer(cur_thread, plain_ptr,
                                   is_64_bit ? Svc::ArbitrationType::WaitIfEqual64
                                             : Svc::ArbitrationType::WaitIfEqual,
                                   value, timeout, &fast_res)) {
            return fast_res;
        }
    }
    if (timeout != 0 && SpinBeforeWait(kernel, cur_thread, process, addr, plain_ptr, is_64_bit,
                                       [value](s64 user_value) { return user_value == value; })) {
        R_THROW(ResultInvalidState);
    }
    KHardwareTimer* timer{};
    ThreadQueueImplForKAddressArbiter wait_queue(kernel, this, std::addressof(m_tree));

    {
        ScopedStripePreparing stripe_guard{this, addr};
        KScopedSchedulerLockAndSleep slp{kernel, std::addressof(timer), cur_thread, timeout};

        // Check that the thread isn't terminating.
        if (cur_thread->IsTerminationRequested()) {
            slp.CancelSleep();
            R_THROW(ResultTerminationRequested);
        }

        // Read the value from userspace.
        u8* const locked_ptr = TryGetPlainUserPointer(process, addr, word_size);
        s64 user_value{};
        bool succeeded{};
        if (is_64_bit) {
            succeeded = ReadFromUser(process, locked_ptr, std::addressof(user_value), addr);
        } else {
            s32 user_value32{};
            succeeded = ReadFromUser(process, locked_ptr, std::addressof(user_value32), addr);
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
        cur_thread->BeginWait(kernel, std::addressof(wait_queue));
        cur_thread->SetWaitReasonForDebugging(ThreadWaitReasonForDebugging::Arbitration);
    }

    // Get the result.
    return cur_thread->GetWaitResult();
}

} // namespace Kernel
