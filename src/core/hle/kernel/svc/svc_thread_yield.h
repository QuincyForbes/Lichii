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

bool YieldBackoffEnabled();

/// Lock-free in-JIT fast paths for hot non-blocking 64-bit multicore SVCs.
/// Every fast path is strictly side-effect-free when returning false so the caller can fall back
/// to a standard SupervisorCall halt.
bool TryFastFutileYield64(Core::System& system, s64 ns, u64 spin_state_hash);
bool TryFastGetCurrentProcessorNumber64(Core::System& system, u64* out_core);
bool TryFastSignalToAddress64(Core::System& system, u64 address, u32 signal_type, s32 value,
                              s32 count, u64* out_result);
bool TryFastWaitForAddress64(Core::System& system, u64 address, u32 arb_type, s64 value,
                             s64 timeout_ns, u64* out_result);
bool TryFastArbitrateLock64(Core::System& system, u32 thread_handle, u64 address, u32 tag,
                            u64* out_result);

} // namespace Kernel::Svc
