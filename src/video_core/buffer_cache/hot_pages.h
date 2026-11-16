// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>

#include "common/common_types.h"

namespace VideoCommon::HotPages {

/// EDEN_NX2_HOT_PAGES (from Djihads80/Eglantine f27522bfb9). A page the guest writes again after
/// each upload costs a write trap and two protection changes per upload. Such a page (trapped
/// twice within WindowFrames frames, never written by the GPU) turns hot: its uploads leave it
/// unprotected, and it counts as written again at the next epoch, so the first use of each epoch
/// uploads it whether the guest wrote it or not.
/// 0: off. 1: an epoch is a frame (as upstream). 2: an epoch also ends with each guest command
/// list, for games that fill pages shared by two lists of a frame between the two.
/// Between two epochs a write to a hot page is not seen: the buffer cache tells its write memos
/// at each epoch instead (BufferCache::AdvanceHotPages).
#ifdef __APPLE__
inline constexpr int DefaultMode = 2;
#else
inline constexpr int DefaultMode = 0;
#endif

/// Hot pages cool down together after this many frames and have to trap twice again.
inline constexpr u32 WindowFrames = 128;

[[nodiscard]] inline int ReadMode() noexcept {
    const char* const value = std::getenv("EDEN_NX2_HOT_PAGES");
    return value == nullptr ? DefaultMode : std::atoi(value);
}

/// EDEN_NX2_HOT_PAGES_VERIFY=1 (diagnostics): the buffer cache keeps the bytes each hot page was
/// uploaded with and compares them at every later use within the epoch. A difference is a use
/// that write protection would have uploaded first ("stale uses" in "Perf CPU writes:").
[[nodiscard]] inline bool ReadVerify() noexcept {
    const char* const value = std::getenv("EDEN_NX2_HOT_PAGES_VERIFY");
    return value != nullptr && value[0] == '1';
}

/// Plain values, not function statics: the tracker reads the mode at every page state change.
inline int g_mode = ReadMode();
inline bool g_verify = g_mode != 0 && ReadVerify();

/// Called by VideoCore::CreateGPU, once the frontend has loaded nx2_switches.txt.
inline void ReadSwitches() noexcept {
    g_mode = ReadMode();
    g_verify = g_mode != 0 && ReadVerify();
}

/// Guest command lists finished (GPU::OnCommandListEnd). Mode 2 ends an epoch at each.
inline std::atomic<u32> g_command_lists{0};

struct Counters {
    std::atomic<u64> promoted{};    ///< pages turned hot
    std::atomic<u64> cooled{};      ///< hot pages cooled down at the end of a window
    std::atomic<u64> gpu_written{}; ///< hot pages the GPU wrote: tracked again, for good
    std::atomic<s64> hot{};         ///< pages hot now
    /// Uploads of a hot page that left it unprotected: about one trapped store and two
    /// protection changes saved each.
    std::atomic<u64> kept{};
    std::atomic<u64> epochs{};
    /// Tracker protection changes sent to the device memory: ranges and their pages.
    std::atomic<u64> protect_ranges{};
    std::atomic<u64> protect_pages{};
    std::atomic<u64> unprotect_ranges{};
    std::atomic<u64> unprotect_pages{};
    /// EDEN_NX2_HOT_PAGES_VERIFY: uses of an uploaded hot page compared, those whose bytes had
    /// changed, and uploads of a hot page with the bytes of its previous upload.
    std::atomic<u64> verify_uses{};
    std::atomic<u64> verify_stale{};
    std::atomic<u64> verify_uploads{};
    std::atomic<u64> verify_same{};
};
inline Counters g_counters;

inline void Add(std::atomic<u64>& counter, u64 count) noexcept {
    counter.fetch_add(count, std::memory_order_relaxed);
}

} // namespace VideoCommon::HotPages
