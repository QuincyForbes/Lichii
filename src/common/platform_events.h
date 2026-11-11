// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>

#include "common/common_types.h"

// Host platform signals the engine reacts to: memory pressure (iOS memory warnings and the
// per-process limit), the app leaving the foreground and the display's refresh rate. A frontend
// feeds them; nothing here changes behaviour until it does.
namespace Common::PlatformEvents {

/// A low-memory warning from the OS: the caches collect aggressively for a while.
void SignalMemoryPressure();
/// Bumped by every SignalMemoryPressure, including those PollMemoryPressure raises.
u32 MemoryPressureGeneration();

/// Bytes the OS still lets this process allocate before it is killed (iOS), 0 where unknown.
u64 AvailableProcessMemory();
/// Signal pressure whenever AvailableProcessMemory falls below `bytes` (0, the default: never).
void SetMemoryLowWater(u64 bytes);
/// Checks the low-water mark at most every 250 ms. Called once per frame by the caches.
void PollMemoryPressure();

struct ProcessMemory {
    u64 footprint;      ///< What jetsam compares with the limit (phys_footprint)
    u64 footprint_peak;
    u64 compressed;
    u64 reusable;       ///< Marked reusable (MADV_FREE_REUSABLE), outside the footprint
    u64 internal;       ///< Anonymous memory mapped by the process, per mapping (pmap ledger)
    u64 purgeable_nonvolatile;
    u64 external;       ///< Resident pages of mapped files, outside the footprint
};
/// Apple: the process footprint and parts of it. Windows: the working set and its peak. Zeros
/// elsewhere.
ProcessMemory QueryProcessMemory();
/// Apple: hands the memory malloc holds free back to the OS. Returns the bytes released.
u64 RelieveMallocPressure();

/// Work to finish before the app is suspended (write caches out). Listeners run on the thread
/// calling NotifyEnteringBackground; they must not wait for the emulated CPU or the GPU thread.
std::size_t AddBackgroundListener(std::function<void()> listener);
void RemoveBackgroundListener(std::size_t id);
void NotifyEnteringBackground();

/// The display's highest refresh rate (UIScreen.maximumFramesPerSecond), 0 when unknown.
void SetDisplayMaxFps(u32 fps);
u32 DisplayMaxFps();

} // namespace Common::PlatformEvents
