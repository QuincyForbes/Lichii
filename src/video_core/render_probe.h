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
/// The 3D scene's width and height, then the width and height the pass that scales it writes, 16
/// bits each. Zero until a game's recognised upscale pass ran (Ocarina of Time); scene_upscaler
/// names what drew that pass.
inline std::atomic<u64> scene_extents{};
inline std::atomic<const char*> scene_upscaler{};
/// Device memory the texture cache holds, and the level at which it starts collecting.
inline std::atomic<u64> texture_cache_bytes{};
inline std::atomic<u64> texture_cache_expected{};
/// Bumped to have the renderer log its memory report at the next frame (debug server).
inline std::atomic<u32> memory_report_requests{};

[[nodiscard]] inline u64 PackExtents(u32 first_width, u32 first_height, u32 second_width,
                                     u32 second_height) {
    const auto clamp = [](u32 value) { return static_cast<u64>(value > 0xFFFF ? 0xFFFF : value); };
    return clamp(first_width) | clamp(first_height) << 16 | clamp(second_width) << 32 |
           clamp(second_height) << 48;
}

inline void StoreFrameExtents(u32 guest_width, u32 guest_height, u32 output_width,
                              u32 output_height) {
    frame_extents.store(PackExtents(guest_width, guest_height, output_width, output_height),
                        std::memory_order_relaxed);
}

/// upscaler is a string literal, or null to forget the scene (a new game).
inline void StoreScene(u32 scene_width, u32 scene_height, u32 target_width, u32 target_height,
                       const char* upscaler) {
    scene_upscaler.store(upscaler, std::memory_order_relaxed);
    scene_extents.store(upscaler ? PackExtents(scene_width, scene_height, target_width,
                                               target_height)
                                 : 0,
                        std::memory_order_relaxed);
}

} // namespace VideoCore::Probe
