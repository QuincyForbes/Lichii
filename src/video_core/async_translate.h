// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>

namespace VideoCommon {

/// EDEN_NX2_ASYNC_TRANSLATE, read on first use (main() has applied nx2_switches.txt by then):
/// 0 translates every new graphics pipeline on the GPU thread, as before.
/// 1 (default) translates the new pipelines of draws that may be skipped on the translation
///   workers, from a snapshot of what the translation reads; the draws are skipped until the
///   pipeline is ready.
/// 2 (Fritchii's verify mode, not in this build) translates as 1.
/// 3 translates on the GPU thread as before, from the snapshot, and checks every recorded read
///   against the live state (the snapshot test, without workers).
[[nodiscard]] inline int AsyncTranslateMode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_ASYNC_TRANSLATE");
        if (value == nullptr) {
            // 1 by default (Fritchii 2026-11-08: verify mode 362 compared, 0 differ, no invalid
            // SPIR-V under the validation layer; empty-cache walk: GPU-thread pipeline time 7.5 ->
            // 4.4 s, frames over 50 ms 77 -> 64).
            return 1;
        }
        const int parsed = std::atoi(value);
        if (parsed == 2) {
            return 1;
        }
        return parsed >= 0 && parsed <= 3 ? parsed : 0;
    }();
    return mode;
}

/// Whether translation workers read guest memory (mode 1): the GPU memory manager's writers then
/// take its page table lock.
[[nodiscard]] inline bool AsyncTranslateWorkers() {
    return AsyncTranslateMode() == 1;
}

} // namespace VideoCommon
