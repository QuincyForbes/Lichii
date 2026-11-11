// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/hle/kernel/k_spin_lock.h"

#include <chrono>
#include <cstdlib>
#include <thread>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "common/nx2_address_wait.h"

namespace Kernel {

namespace {

bool IsLowCoreDevice() {
#if defined(__APPLE__) && defined(TARGET_OS_IOS) && TARGET_OS_IOS
    return true;
#else
    static const bool low_cores = [] {
        const unsigned int hc = std::thread::hardware_concurrency();
        return hc != 0 && hc <= 6;
    }();
    return low_cores;
#endif
}

int FallbackSpinCount() {
    return IsLowCoreDevice() ? 32 : 128;
}

void Pause() {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__APPLE__) && defined(__aarch64__)
    __builtin_arm_isb(0xF);
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

} // Anonymous namespace

namespace Nx2Wait {

bool Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_KSPIN");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

s64 SpinMicroseconds() {
    static const s64 microseconds = [] {
        const char* value = std::getenv("EDEN_NX2_KSPIN_US");
        if (value != nullptr) {
            return static_cast<s64>(std::strtoll(value, nullptr, 10));
        }
        return IsLowCoreDevice() ? s64{2} : s64{20};
    }();
    return microseconds;
}

void Wait(std::atomic<u32>& word, u32 expected) {
    Common::Nx2AddressWait::Wait(word, expected);
}

void WakeOne(std::atomic<u32>& word) {
    Common::Nx2AddressWait::WakeOne(word);
}

void WakeAll(std::atomic<u32>& word) {
    Common::Nx2AddressWait::WakeAll(word);
}

} // namespace Nx2Wait

void KSpinLock::Lock() {
    u32 expected = 0;
    if (m_state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
        return;
    }
    if (Nx2Wait::Enabled()) {
        LockContendedNx2();
        return;
    }
    const int max_spins = FallbackSpinCount();
    for (int spin = 0; spin < max_spins; ++spin) {
        Pause();
        if (m_state.load(std::memory_order_relaxed) == 0) {
            expected = 0;
            if (m_state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                              std::memory_order_relaxed)) {
                return;
            }
        }
    }
    // Mark the lock contended and sleep until it is released; taken this way it stays marked,
    // so its Unlock wakes the next sleeper.
    while (m_state.exchange(2, std::memory_order_acquire) != 0) {
        m_state.wait(2, std::memory_order_relaxed);
    }
}

void KSpinLock::LockContendedNx2() {
    // Holders keep the lock for a few microseconds; sleeping costs the waiter the host's wake-up
    // latency and the holder a wake syscall. Spin for a bounded time first.
    const s64 budget_us = Nx2Wait::SpinMicroseconds();
    if (budget_us > 0) {
        const auto budget = std::chrono::microseconds{budget_us};
        const u32 max_spins = IsLowCoreDevice() ? 64U : 4096U;
        std::chrono::steady_clock::time_point deadline{};
        for (u32 spin = 1; spin <= max_spins; ++spin) {
            Pause();
            if (m_state.load(std::memory_order_relaxed) == 0) {
                u32 expected = 0;
                if (m_state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                                  std::memory_order_relaxed)) {
                    return;
                }
            }
            if ((spin & 15U) == 0) {
                const auto now = std::chrono::steady_clock::now();
                if (spin == 16U) {
                    deadline = now + budget;
                } else if (now >= deadline) {
                    break;
                }
            }
        }
    }
    while (m_state.exchange(2, std::memory_order_acquire) != 0) {
        Nx2Wait::Wait(m_state, 2);
    }
}

void KSpinLock::Unlock() {
    if (m_state.exchange(0, std::memory_order_release) == 2) {
        if (Nx2Wait::Enabled()) {
            Nx2Wait::WakeOne(m_state);
        } else {
            m_state.notify_one();
        }
    }
}

bool KSpinLock::TryLock() {
    u32 expected = 0;
    return m_state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                           std::memory_order_relaxed);
}

} // namespace Kernel
