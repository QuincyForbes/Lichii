// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <cstring>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/common_types.h"
#include "common/literals.h"
#include "common/platform_events.h"

/// The caches' budgets where they share the process's memory limit with everything else (iOS).
/// EDEN_NX2_CACHE_BUDGET=ios gives a PC the phone's budgets, with EDEN_NX2_CACHE_BUDGET_MIB
/// (default 4352) as what graphics may use before the process limit: what is left is that less
/// the device memory in use, as on an iPhone with guest memory in a file.
namespace VideoCommon::HostBudget {

using namespace Common::Literals;

inline bool Simulated() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return false;
#else
    static const bool simulated = [] {
        const char* value = std::getenv("EDEN_NX2_CACHE_BUDGET");
        return value != nullptr && std::strcmp(value, "ios") == 0;
    }();
    return simulated;
#endif
}

inline bool Active() {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return true;
#else
    return Simulated();
#endif
}

/// The texture collector compares its images' own memory with its thresholds, and both caches'
/// thresholds follow the memory left to the process. The device heap the texture collector
/// compared before also holds buffers, staging, presentation and the driver's own memory (an
/// iPhone: 2254 MiB in use for 271 MiB of images), so it collected without end, and the buffer
/// cache collected at a fixed 207 MiB with 400-600 MiB of buffers. EDEN_NX2_CACHE_FIXED=1: both
/// as before.
inline bool FollowsHeadroom() {
    static const bool follows = [] {
        const char* value = std::getenv("EDEN_NX2_CACHE_FIXED");
        return Active() && (value == nullptr || value[0] == '0');
    }();
    return follows;
}

/// Memory left to the process (0: not known).
inline u64 Available([[maybe_unused]] u64 device_usage) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    return Common::PlatformEvents::AvailableProcessMemory();
#else
    if (!Simulated()) {
        return 0;
    }
    static const u64 limit = [] {
        const char* value = std::getenv("EDEN_NX2_CACHE_BUDGET_MIB");
        const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
        return (mib != 0 ? mib : 4352) << 20;
    }();
    // Never 0, which means not known.
    return limit > device_usage + 1_MiB ? limit - device_usage : 1_MiB;
#endif
}

/// What ApplyMemoryBudgets gives an iPhone 18 Pro Max, for a simulated phone (the buffer cache's
/// fixed threshold was 207 MiB).
inline constexpr u64 SimulatedTextureExpected = 1793_MiB;
inline constexpr u64 SimulatedTextureCritical = 2391_MiB;
inline u64 SimulatedBufferCollect() {
    return FollowsHeadroom() ? 414_MiB : 207_MiB;
}

/// Room past this is the caches' to use.
inline constexpr u64 Reserve = 768_MiB;

} // namespace VideoCommon::HostBudget
