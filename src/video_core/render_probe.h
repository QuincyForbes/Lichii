// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// Numbers the renderer publishes for a frontend's performance overlay (the iOS app's). Each is one
// relaxed store per frame by the thread that owns the value, and a relaxed load by the reader.
namespace VideoCore::Probe {

/// Guest framebuffer width and height, then the output (swapchain) width and height, 16 bits each.
inline std::atomic<u64> frame_extents{};
/// Device memory the texture cache holds, and the level at which it starts collecting.
inline std::atomic<u64> texture_cache_bytes{};
inline std::atomic<u64> texture_cache_expected{};
/// Bumped to have the renderer log its memory report at the next frame (debug server).
inline std::atomic<u32> memory_report_requests{};

inline void StoreFrameExtents(u32 guest_width, u32 guest_height, u32 output_width,
                              u32 output_height) {
    const auto clamp = [](u32 value) { return static_cast<u64>(value > 0xFFFF ? 0xFFFF : value); };
    frame_extents.store(clamp(guest_width) | clamp(guest_height) << 16 |
                            clamp(output_width) << 32 | clamp(output_height) << 48,
                        std::memory_order_relaxed);
}

} // namespace VideoCore::Probe
