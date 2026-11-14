// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstring>
#include <optional>
#include <vector>

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_swapchain.h"
#include "video_core/renderer_metal/renderer_metal.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"

namespace Metal {

namespace {

[[nodiscard]] MTLPixelFormat FramebufferFormat(Service::android::PixelFormat format) {
    switch (format) {
    case Service::android::PixelFormat::Rgba1010102:
        return MTLPixelFormatRGB10A2Unorm;
    case Service::android::PixelFormat::Rgb565:
        return MTLPixelFormatB5G6R5Unorm;
    case Service::android::PixelFormat::Bgra8888:
        return MTLPixelFormatBGRA8Unorm;
    default:
        return MTLPixelFormatRGBA8Unorm;
    }
}

} // Anonymous namespace

struct RendererMetal::Impl {
    explicit Impl(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu,
                  Tegra::MaxwellDeviceMemoryManager& device_memory_)
        : device_memory{device_memory_}, scheduler{device},
          swapchain{device, emu_window.GetWindowInfo().render_surface},
          rasterizer{gpu, device_memory_, device, scheduler} {}

    /// The guest framebuffer as a texture: the texture cache's image, else read from guest
    /// memory (block linear) into a texture of our own.
    std::optional<DisplayTexture> PrepareLayer(const Tegra::FramebufferConfig& framebuffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Device device;
    Scheduler scheduler;
    Swapchain swapchain;
    RasterizerMetal rasterizer;

    Object cpu_texture;
    Object cpu_upload;
    u8* cpu_upload_contents{};
    u64 cpu_upload_size{};
    u64 cpu_upload_tick{};
    u32 cpu_width{};
    u32 cpu_height{};
    Service::android::PixelFormat cpu_format{};
    std::vector<u8> unswizzled;

    std::chrono::steady_clock::time_point report_start = std::chrono::steady_clock::now();
    u64 presented{};
    u64 skipped{};
    u64 from_cache{};
    u64 from_memory{};
};

std::optional<DisplayTexture> RendererMetal::Impl::PrepareLayer(
    const Tegra::FramebufferConfig& framebuffer) {
    const DAddr framebuffer_addr = framebuffer.address + framebuffer.offset;
    if (auto accelerated = rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr)) {
        ++from_cache;
        return accelerated;
    }
    if (framebuffer.width == 0 || framebuffer.height == 0) {
        return std::nullopt;
    }
    if (!cpu_texture || cpu_width != framebuffer.width || cpu_height != framebuffer.height ||
        cpu_format != framebuffer.pixel_format) {
        cpu_texture = CreateTexture(device, TextureDesc{
                                                .kind = TextureKind::e2D,
                                                .format = static_cast<u32>(
                                                    FramebufferFormat(framebuffer.pixel_format)),
                                                .width = framebuffer.width,
                                                .height = framebuffer.height,
                                            });
        cpu_width = framebuffer.width;
        cpu_height = framebuffer.height;
        cpu_format = framebuffer.pixel_format;
    }
    // The guest's framebuffers are block linear with 16-GOB blocks (as OpenGL reads them).
    constexpr u32 block_height_log2 = 4;
    const auto pixel_format =
        VideoCore::Surface::PixelFormatFromGPUPixelFormat(framebuffer.pixel_format);
    const u32 bytes_per_pixel = VideoCore::Surface::BytesPerBlock(pixel_format);
    const u64 size_in_bytes = Tegra::Texture::CalculateSize(
        true, bytes_per_pixel, framebuffer.stride, framebuffer.height, 1, block_height_log2, 0);
    const u8* const host_ptr = device_memory.GetPointer<u8>(framebuffer_addr);
    if (host_ptr == nullptr || !cpu_texture) {
        return std::nullopt;
    }
    const u64 bytes_per_row = u64{framebuffer.stride} * bytes_per_pixel;
    const u64 linear_size = bytes_per_row * framebuffer.height;
    unswizzled.resize(linear_size);
    Tegra::Texture::UnswizzleTexture(unswizzled, std::span<const u8>(host_ptr, size_in_bytes),
                                     bytes_per_pixel, framebuffer.stride, framebuffer.height, 1,
                                     block_height_log2, 0);
    // One upload buffer: the previous frame's copy must be done before it is written again.
    scheduler.Wait(cpu_upload_tick);
    if (linear_size > cpu_upload_size) {
        cpu_upload = CreateBuffer(device, linear_size, true, &cpu_upload_contents);
        cpu_upload_size = cpu_upload_contents != nullptr ? linear_size : 0;
    }
    if (cpu_upload_contents == nullptr) {
        return std::nullopt;
    }
    std::memcpy(cpu_upload_contents, unswizzled.data(), linear_size);
    scheduler.CopyBufferToTexture(cpu_upload.Get(), 0, static_cast<u32>(bytes_per_row), 0,
                                  cpu_texture.Get(), 0, 0, Offset3D{},
                                  Extent3D{framebuffer.width, framebuffer.height, 1});
    cpu_upload_tick = scheduler.CurrentTick();
    ++from_memory;
    return DisplayTexture{
        .texture = cpu_texture.Get(),
        .width = framebuffer.width,
        .height = framebuffer.height,
        .scaled_width = framebuffer.width,
        .scaled_height = framebuffer.height,
    };
}

RendererMetal::RendererMetal(Core::Frontend::EmuWindow& emu_window,
                             Tegra::MaxwellDeviceMemoryManager& device_memory, Tegra::GPU& gpu_,
                             std::unique_ptr<Core::Frontend::GraphicsContext> context_)
    : RendererBase(emu_window, std::move(context_)), gpu{gpu_},
      impl{std::make_unique<Impl>(emu_window, gpu_, device_memory)} {
    LOG_WARNING(Render_Metal,
                "Metal renderer (experimental): milestone 2, the texture and buffer caches, "
                "copies, clears and the guest framebuffer run on Metal; draws are not drawn yet");
}

RendererMetal::~RendererMetal() = default;

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
        // Only the bottom layer for now: overlays need blending (milestone 4).
        const std::optional<DisplayTexture> source =
            framebuffers.empty() ? std::nullopt : impl->PrepareLayer(framebuffers.front());
        const id<CAMetalDrawable> drawable =
            impl->swapchain.NextDrawable(layout.width, layout.height);
        if (drawable != nil) {
            const Attachment target{.texture = (__bridge void*)drawable.texture};
            bool drawn = false;
            if (source) {
                const auto crop =
                    Tegra::NormalizeCrop(framebuffers.front(), source->width, source->height);
                const f32 width = static_cast<f32>(source->scaled_width);
                const f32 height = static_cast<f32>(source->scaled_height);
                const auto& screen = layout.screen;
                drawn = impl->scheduler.Blit(BlitPass{
                    .src = source->texture,
                    .src_rect = {static_cast<s32>(crop.left * width),
                                 static_cast<s32>(crop.top * height),
                                 static_cast<s32>(crop.right * width),
                                 static_cast<s32>(crop.bottom * height)},
                    .dst = target,
                    .dst_format = DrawableFormat(),
                    .dst_rect = {static_cast<s32>(screen.left), static_cast<s32>(screen.top),
                                 static_cast<s32>(screen.right), static_cast<s32>(screen.bottom)},
                    .dst_width = layout.width,
                    .dst_height = layout.height,
                    .linear = true,
                    .keep_dst = false,
                });
            }
            if (!drawn) {
                // Dark blue: the renderer runs, the guest has no framebuffer to show.
                ClearPass clear{};
                clear.color[0] = target;
                clear.clear_color[0] = true;
                clear.color_value[0] = {0.06, 0.09, 0.22, 1.0};
                clear.width = layout.width;
                clear.height = layout.height;
                impl->scheduler.Clear(clear);
            }
            const id<MTLCommandBuffer> command_buffer =
                (__bridge id<MTLCommandBuffer>)impl->scheduler.CommandBuffer();
            [command_buffer presentDrawable:drawable];
            impl->scheduler.Flush();
            ++impl->presented;
        } else {
            ++impl->skipped;
            impl->scheduler.Flush();
        }
    }

    ++m_current_frame;
    gpu.RendererFrameEndNotify();
    impl->rasterizer.TickFrame();

    const auto now = std::chrono::steady_clock::now();
    if (now - impl->report_start >= std::chrono::seconds{10}) {
        const auto [draws, dispatches, clears] = impl->rasterizer.TakeWorkCounts();
        const auto [encoders, command_buffers, blits, passes] = impl->scheduler.TakeCounts();
        LOG_INFO(Render_Metal,
                 "Metal: {} frames presented ({} from the texture cache, {} from guest memory), "
                 "{} without a drawable; {} draws and {} dispatches taken (not drawn yet), {} "
                 "clears; {} command buffers, {} encoders ({} render passes), {} blit commands; "
                 "textures {} MiB, buffers {} MiB",
                 impl->presented, impl->from_cache, impl->from_memory, impl->skipped, draws,
                 dispatches, clears, command_buffers, encoders, passes, blits,
                 impl->rasterizer.TextureRuntime().allocated_bytes >> 20,
                 impl->rasterizer.BufferRuntime().GetDeviceMemoryUsage() >> 20);
        impl->report_start = now;
        impl->presented = 0;
        impl->skipped = 0;
        impl->from_cache = 0;
        impl->from_memory = 0;
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
