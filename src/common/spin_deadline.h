// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <cstdlib>

#include "common/common_types.h"

/// Deadlines for the bounded spins of the guest cores (Idle, address arbiter waits, the kernel's
/// contended locks). std::chrono::steady_clock::now() goes through clock_gettime and
/// mach_continuous_time on macOS (~40 ns, ~75 of 6000 guest-core samples in an Ocarina profile,
/// all inside these spins); on arm64 the virtual counter is read directly instead, without the
/// ISB that exact time reads need (a deadline may be read a few cycles early or late).
/// EDEN_NX2_SPIN_CLOCK=0 restores steady_clock.
namespace Common::SpinDeadline {

[[nodiscard]] inline bool RawClockEnabled() {
#if defined(__aarch64__) || defined(_M_ARM64)
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_SPIN_CLOCK");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
#else
    return false;
#endif
}

#if defined(__aarch64__) && !defined(_MSC_VER)
[[nodiscard]] inline u64 RawTicks() {
    u64 ticks;
    asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
}

[[nodiscard]] inline u64 RawFrequency() {
    static const u64 frequency = [] {
        u64 value;
        asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
        return value != 0 ? value : u64{24'000'000};
    }();
    return frequency;
}
#endif

/// A point a bounded spin stops at. Cheap to test; start it once per spin.
class Deadline {
public:
    explicit Deadline(std::chrono::nanoseconds budget) {
#if defined(__aarch64__) && !defined(_MSC_VER)
        if (RawClockEnabled()) {
            raw = true;
            const u64 ns = static_cast<u64>(budget.count() > 0 ? budget.count() : 0);
            // Budgets are microseconds: ns * frequency stays far below 2^64.
            end_ticks = RawTicks() + (ns * RawFrequency()) / 1'000'000'000ULL;
            return;
        }
#endif
        end_time = std::chrono::steady_clock::now() + budget;
    }

    [[nodiscard]] bool Expired() const {
#if defined(__aarch64__) && !defined(_MSC_VER)
        if (raw) {
            return static_cast<s64>(RawTicks() - end_ticks) >= 0;
        }
#endif
        return std::chrono::steady_clock::now() >= end_time;
    }

private:
    std::chrono::steady_clock::time_point end_time{};
    u64 end_ticks{};
    bool raw{};
};

} // namespace Common::SpinDeadline
