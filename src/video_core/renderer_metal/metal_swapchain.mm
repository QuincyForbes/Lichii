// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>

#include <TargetConditionals.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_swapchain.h"

namespace Metal {

Swapchain::Swapchain(const Device& device, void* render_surface) {
    id surface = (__bridge id)render_surface;
    if (surface == nil || ![surface isKindOfClass:[CAMetalLayer class]]) {
        throw std::runtime_error("The window has no CAMetalLayer");
    }
    layer = (CAMetalLayer*)surface;
    // Set from the renderer's thread, as MoltenVK does.
    layer.device = device.GetDevice();
    layer.pixelFormat = PixelFormat();
    layer.framebufferOnly = YES;
    layer.maximumDrawableCount = 3;
    ApplyVsync();
    LOG_INFO(Render_Metal, "Metal: presenting to a CAMetalLayer, {} drawables, vsync {}",
             layer.maximumDrawableCount, vsync ? "on" : "off");
}

Swapchain::~Swapchain() = default;

void Swapchain::ApplyVsync() {
    vsync = Settings::values.vsync_mode.GetValue() != Settings::VSyncMode::Immediate;
#if !TARGET_OS_IPHONE
    layer.displaySyncEnabled = vsync ? YES : NO;
#else
    // iOS presents on the display's refresh only.
    vsync = true;
#endif
}

id<CAMetalDrawable> Swapchain::NextDrawable(u32 width, u32 height) {
    if (width == 0 || height == 0) {
        return nil;
    }
    if (width != drawable_width || height != drawable_height) {
        drawable_width = width;
        drawable_height = height;
        layer.drawableSize = CGSizeMake(width, height);
    }
    return [layer nextDrawable];
}

} // namespace Metal
