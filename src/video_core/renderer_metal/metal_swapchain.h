// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (renderer_metal/*.mm).

#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "common/common_types.h"

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

private:
    void ApplyVsync();

    CAMetalLayer* layer = nil;
    u32 drawable_width{};
    u32 drawable_height{};
    bool vsync{};
};

} // namespace Metal
