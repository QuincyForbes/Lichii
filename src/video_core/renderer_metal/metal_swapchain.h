// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (renderer_metal/*.mm).

#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <chrono>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_work_policy.h"

namespace Metal {

class Device;

/// The frontend's CAMetalLayer (WindowSystemInfo::render_surface, the same layer MoltenVK takes).
class Swapchain {
public:
    /// Throws std::runtime_error when the surface is not a CAMetalLayer.
    explicit Swapchain(const Device& device, void* render_surface);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// The next drawable at this size in pixels, or nil when the layer has none to give
    /// (minimised window, app in the background).
    [[nodiscard]] id<CAMetalDrawable> NextDrawable(u32 width, u32 height);

    [[nodiscard]] MTLPixelFormat PixelFormat() const noexcept {
        return MTLPixelFormatBGRA8Unorm;
    }

    /// Presents `drawable` with `command_buffer`. A guest frame is paced as Vulkan's host pacer
    /// does (EDEN_NX2_HOST_PACE, EDEN_NX2_METAL_HOST_PACE): no earlier than the game's frame
    /// period in whole refreshes after the one before.
    void Present(id<MTLCommandBuffer> command_buffer, id<CAMetalDrawable> drawable,
                 bool guest_frame);

    /// Presents `drawable` with `command_buffer` once the one before has been shown for
    /// `hold_seconds` (frame generation spaces its frames itself; no guest pacing).
    void PresentHeld(id<MTLCommandBuffer> command_buffer, id<CAMetalDrawable> drawable,
                     f64 hold_seconds);

    /// The main screen's refresh period in ns, 0 when it can't be read.
    [[nodiscard]] f64 DisplayRefreshNs();

    /// Since the last call: guest frames presented, those held for pacing, and the refreshes
    /// each was held for ("Metal pacing:" line); empty without pacing.
    [[nodiscard]] std::string TakePacingLine();

private:
    void ApplyVsync();

    CAMetalLayer* layer = nil;
    u32 drawable_width{};
    u32 drawable_height{};
    bool vsync{};
    WorkPolicy::PresentPacer pacer;
    std::optional<std::chrono::steady_clock::time_point> last_guest_frame;
    u64 paced_frames{};
    u64 held_frames{};
    f64 held_refreshes{};
};

} // namespace Metal
