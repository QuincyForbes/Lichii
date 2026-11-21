// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace Core {
class System;
}

namespace Kernel::Svc {

constexpr u64 HashYieldSpinWord(u64 hash, u64 word) noexcept {
    return (hash ^ word) * 0x100000001B3ULL;
}

/// Lock-free in-JIT fast path for `svcSleepThread` yield calls (`ns` in `{-2, -1, 0}`) on 64-bit
/// multicore guest threads when the yield is provably futile and neither backoff nor runnable
/// handoff needs to sleep the calling thread. Returns true when handled in place without halting
/// the JIT, or false when the caller must perform a normal SupervisorCall halt.
bool TryFastFutileYield64(Core::System& system, s64 ns, u64 spin_state_hash);

} // namespace Kernel::Svc
