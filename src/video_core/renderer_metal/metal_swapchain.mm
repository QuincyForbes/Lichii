// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <stdexcept>

#include <TargetConditionals.h>
#include <fmt/format.h>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_swapchain.h"
#include "video_core/renderer_metal/metal_switches.h"

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
    // As Vulkan's swapchain (B8G8R8A8_UNORM in SRGB_NONLINEAR, which MoltenVK turns into this):
    // the pictures are sRGB, matched to the display's colours. Neither renderer outputs HDR.
    // EDEN_NX2_METAL_COLORSPACE=0 leaves the layer untagged, as before.
    static const bool tag_srgb = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_COLORSPACE");
        return value == nullptr || value[0] != '0';
    }();
    if (tag_srgb) {
        if (@available(iOS 16.0, macOS 10.11, *)) {
            const CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            layer.colorspace = srgb;
            CGColorSpaceRelease(srgb);
        }
    }
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

namespace {

// Vulkan's EDEN_NX2_HOST_PACE (on unless 0, read under EDEN_NX2_METAL_VULKAN_SWITCHES) and
// EDEN_NX2_METAL_HOST_PACE=0 to leave it off on Metal alone.
bool HostPaceEnabled() {
    static const bool enabled = [] {
        const char* const metal = std::getenv("EDEN_NX2_METAL_HOST_PACE");
        if (metal != nullptr && metal[0] == '0') {
            return false;
        }
        const char* const vulkan = VulkanSwitch("EDEN_NX2_HOST_PACE");
        return vulkan == nullptr || vulkan[0] != '0';
    }();
    return enabled;
}

// The main screen's refresh rate (UIScreen on iOS, NSScreen on macOS 12 and later), looked up
// by name so video_core links neither UIKit nor AppKit; 0 when it can't be read.
f64 ScreenRefreshNs() {
    Class screen_class = NSClassFromString(@"UIScreen");
    if (screen_class == nil) {
        screen_class = NSClassFromString(@"NSScreen");
    }
    if (screen_class == nil || ![screen_class respondsToSelector:@selector(mainScreen)]) {
        return 0.0;
    }
    id screen = [screen_class valueForKey:@"mainScreen"];
    if (screen == nil || ![screen respondsToSelector:NSSelectorFromString(@"maximumFramesPerSecond")]) {
        return 0.0;
    }
    const NSInteger fps = [[screen valueForKey:@"maximumFramesPerSecond"] integerValue];
    return fps > 0 ? 1e9 / static_cast<f64>(fps) : 0.0;
}

} // Anonymous namespace

void Swapchain::Present(id<MTLCommandBuffer> command_buffer, id<CAMetalDrawable> drawable,
                        bool guest_frame) {
    f64 minimum_ns = 0.0;
    // Not in the Turbo speed mode, as Vulkan's present thread.
    if (guest_frame && vsync && HostPaceEnabled() &&
        Settings::values.current_speed_mode.GetValue() != Settings::SpeedMode::Turbo) {
        if (pacer.refresh_ns == 0.0) {
            pacer.refresh_ns = ScreenRefreshNs();
            LOG_INFO(Render_Metal, "Metal pacing: display refresh {:.2f} ms",
                     pacer.refresh_ns / 1e6);
            if (pacer.refresh_ns == 0.0) {
                pacer.refresh_ns = -1.0; // unknown: presented at once
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (last_guest_frame) {
            pacer.NoteFrame(static_cast<f64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - *last_guest_frame)
                    .count()));
        }
        last_guest_frame = now;
        minimum_ns = pacer.MinimumDurationNs();
        ++paced_frames;
    }
    if (minimum_ns > 0.0) {
        ++held_frames;
        held_refreshes += minimum_ns / pacer.refresh_ns + 0.5;
        [command_buffer presentDrawable:drawable afterMinimumDuration:minimum_ns / 1e9];
    } else {
        [command_buffer presentDrawable:drawable];
    }
}

std::string Swapchain::TakePacingLine() {
    if (paced_frames == 0) {
        return {};
    }
    const std::string line = fmt::format(
        "{} guest frames, {} held to whole refreshes ({:.1f} refreshes each), game frame {:.2f} "
        "ms, refresh {:.2f} ms",
        paced_frames, held_frames, held_frames != 0 ? held_refreshes / held_frames : 0.0,
        pacer.content_ns / 1e6, pacer.refresh_ns / 1e6);
    paced_frames = 0;
    held_frames = 0;
    held_refreshes = 0.0;
    return line;
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
