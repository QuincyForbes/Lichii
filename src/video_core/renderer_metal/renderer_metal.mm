// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_swapchain.h"
#include "video_core/renderer_metal/renderer_metal.h"

namespace Metal {

struct RendererMetal::Impl {
    explicit Impl(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu)
        : swapchain{device, emu_window.GetWindowInfo().render_surface}, rasterizer{gpu} {}

    Device device;
    Swapchain swapchain;
    RasterizerMetal rasterizer;

    std::chrono::steady_clock::time_point report_start = std::chrono::steady_clock::now();
    u64 presented{};
    u64 skipped{};
};

RendererMetal::RendererMetal(Core::Frontend::EmuWindow& emu_window,
                             [[maybe_unused]] Tegra::MaxwellDeviceMemoryManager& device_memory,
                             Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), gpu{gpu_},
      impl{std::make_unique<Impl>(emu_window, gpu_)} {
    LOG_WARNING(Render_Metal,
                "Metal renderer (experimental): milestone 1, frames are cleared and presented, "
                "guest drawing is not shown yet");
}

RendererMetal::~RendererMetal() {
    // Command buffers still running reference the layer's drawables.
    @autoreleasepool {
        id<MTLCommandBuffer> last = [impl->device.GetQueue() commandBuffer];
        [last commit];
        [last waitUntilCompleted];
    }
}

void RendererMetal::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    SCOPE_EXIT {
        render_window.OnFrameDisplayed();
    };
    if (renderer_settings.screenshot_requested) {
        renderer_settings.screenshot_requested = false;
        if (renderer_settings.screenshot_complete_callback) {
            renderer_settings.screenshot_complete_callback(false);
        }
    }
    if (!render_window.IsShown()) {
        return;
    }

    @autoreleasepool {
        const auto& layout = render_window.GetFramebufferLayout();
        id<CAMetalDrawable> drawable = impl->swapchain.NextDrawable(layout.width, layout.height);
        if (drawable != nil) {
            id<MTLCommandBuffer> command_buffer = [impl->device.GetQueue() commandBuffer];
            command_buffer.label = @"Present";
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = drawable.texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            // Dark blue: the renderer runs, the guest's frame is not composited yet.
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0.06, 0.09, 0.22, 1.0);
            id<MTLRenderCommandEncoder> encoder =
                [command_buffer renderCommandEncoderWithDescriptor:pass];
            [encoder endEncoding];
            [command_buffer presentDrawable:drawable];
            [command_buffer commit];
            ++impl->presented;
        } else {
            ++impl->skipped;
        }
    }

    ++m_current_frame;
    gpu.RendererFrameEndNotify();
    impl->rasterizer.TickFrame();

    const auto now = std::chrono::steady_clock::now();
    if (now - impl->report_start >= std::chrono::seconds{10}) {
        const auto [draws, dispatches] = impl->rasterizer.TakeWorkCounts();
        LOG_INFO(Render_Metal,
                 "Metal: {} frames presented, {} without a drawable, {} draws and {} dispatches "
                 "taken in the last 10 s",
                 impl->presented, impl->skipped, draws, dispatches);
        impl->report_start = now;
        impl->presented = 0;
        impl->skipped = 0;
    }
}

std::vector<u8> RendererMetal::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

VideoCore::RasterizerInterface* RendererMetal::ReadRasterizer() {
    return &impl->rasterizer;
}

std::string RendererMetal::GetDeviceVendor() const {
    return impl->device.GetName();
}

} // namespace Metal
