// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace Common {

/// EDEN_NX2_POLL_YIELD=0: threads that poll for another thread sleep 50 us instead of yielding.
/// Darwin's sched_yield is swtch_pri (xnu osfmk/kern/syscall_subr.c): with anything else runnable
/// the caller drops to priority 0 for up to 10 ms, with nothing runnable it spins on its core.
[[nodiscard]] inline bool PollYield() {
    static const bool yield = [] {
        const char* value = std::getenv("EDEN_NX2_POLL_YIELD");
        return value == nullptr || value[0] != '0';
    }();
    return yield;
}

} // namespace Common
