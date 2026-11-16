// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#include <fmt/format.h>

#include "common/console_capture.h"
#include "common/frame_deadline.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/scope_exit.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_coverage.h"
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

/// MetalFX spatial upscaling of the first layer (the MetalFxSpatial scaling filter), on the
/// renderer's own textures and command buffer.
class SpatialUpscaler {
public:
    /// The upscaled picture (output_width x output_height, the drawable's format), or nullptr
    /// when MetalFX can't run (the layer is presented as it is).
    TextureHandle Upscale(const Device& device, Scheduler& scheduler, TextureHandle source,
                          const std::array<s32, 4>& source_rect, u32 output_width,
                          u32 output_height) {
        if (failed || output_width == 0 || output_height == 0) {
            return nullptr;
        }
        if (@available(iOS 16.0, macOS 13.0, *)) {
            const u32 input_width = static_cast<u32>(std::abs(source_rect[2] - source_rect[0]));
            const u32 input_height = static_cast<u32>(std::abs(source_rect[3] - source_rect[1]));
            const MTLPixelFormat format = AsTexture(source).pixelFormat;
            if (input_width == 0 || input_height == 0) {
                return nullptr;
            }
            if (scaler == nil || input_width != in_width || input_height != in_height ||
                output_width != out_width || output_height != out_height ||
                format != in_format) {
                if (!Prepare(device, format, input_width, input_height, output_width,
                             output_height)) {
                    failed = true;
                    return nullptr;
                }
            }
            // The crop (with its flips) into an input of exactly its size.
            const bool copied = scheduler.Blit(BlitPass{
                .src = source,
                .src_rect = source_rect,
                .dst = {.texture = (__bridge void*)input},
                .dst_format = static_cast<u32>(format),
                .dst_rect = {0, 0, static_cast<s32>(input_width), static_cast<s32>(input_height)},
                .dst_width = input_width,
                .dst_height = input_height,
                .linear = false,
                .keep_dst = false,
            });
            if (!copied) {
                return nullptr;
            }
            const id<MTLCommandBuffer> command_buffer =
                (__bridge id<MTLCommandBuffer>)scheduler.CommandBuffer();
            id<MTLFXSpatialScaler> spatial = scaler;
            spatial.colorTexture = input;
            spatial.outputTexture = output;
            spatial.inputContentWidth = input_width;
            spatial.inputContentHeight = input_height;
            [spatial encodeToCommandBuffer:command_buffer];
            return (__bridge void*)output;
        }
        failed = true;
        return nullptr;
    }

private:
    bool Prepare(const Device& device, MTLPixelFormat format, u32 input_width, u32 input_height,
                 u32 output_width, u32 output_height) API_AVAILABLE(ios(16.0), macos(13.0)) {
        if (![MTLFXSpatialScalerDescriptor supportsDevice:device.GetDevice()]) {
            LOG_WARNING(Render_Metal, "MetalFX spatial: not supported on this GPU");
            return false;
        }
        MTLFXSpatialScalerDescriptor* descriptor = [[MTLFXSpatialScalerDescriptor alloc] init];
        descriptor.inputWidth = input_width;
        descriptor.inputHeight = input_height;
        descriptor.outputWidth = output_width;
        descriptor.outputHeight = output_height;
        descriptor.colorTextureFormat = format;
        descriptor.outputTextureFormat = static_cast<MTLPixelFormat>(DrawableFormat());
        // The present chain is in display (gamma) space.
        descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        id<MTLFXSpatialScaler> spatial = [descriptor newSpatialScalerWithDevice:device.GetDevice()];
        if (spatial == nil) {
            LOG_WARNING(Render_Metal, "MetalFX spatial: no scaler for {}x{} -> {}x{}", input_width,
                        input_height, output_width, output_height);
            return false;
        }
        const auto make = [&](u32 width, u32 height, MTLPixelFormat texture_format,
                              MTLTextureUsage usage) {
            MTLTextureDescriptor* texture = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:texture_format
                                             width:width
                                            height:height
                                         mipmapped:NO];
            texture.storageMode = MTLStorageModePrivate;
            texture.usage = usage;
            return [device.GetDevice() newTextureWithDescriptor:texture];
        };
        input = make(input_width, input_height, format,
                     spatial.colorTextureUsage | MTLTextureUsageRenderTarget |
                         MTLTextureUsageShaderRead);
        output = make(output_width, output_height,
                      static_cast<MTLPixelFormat>(DrawableFormat()),
                      spatial.outputTextureUsage | MTLTextureUsageShaderRead);
        if (input == nil || output == nil) {
            return false;
        }
        scaler = spatial;
        in_width = input_width;
        in_height = input_height;
        out_width = output_width;
        out_height = output_height;
        in_format = format;
        LOG_INFO(Render_Metal, "MetalFX spatial: {}x{} -> {}x{}", input_width, input_height,
                 output_width, output_height);
        return true;
    }

    id scaler = nil;
    id<MTLTexture> input = nil;
    id<MTLTexture> output = nil;
    u32 in_width{};
    u32 in_height{};
    u32 out_width{};
    u32 out_height{};
    MTLPixelFormat in_format{MTLPixelFormatInvalid};
    bool failed{};
};

/// Host code addresses as "image+offset" (the offsets symbolicate against the image's dSYM).
std::string DescribeFrames(void* const* frames, int count) {
    std::string text;
    for (int index = 0; index < count; ++index) {
        const char* const separator = index != 0 ? " < " : "";
        const auto address = reinterpret_cast<uintptr_t>(frames[index]);
        Dl_info info{};
        if (dladdr(frames[index], &info) != 0 && info.dli_fname != nullptr) {
            const char* const slash = std::strrchr(info.dli_fname, '/');
            text += fmt::format("{}{}+{:#x}", separator,
                                slash != nullptr ? slash + 1 : info.dli_fname,
                                address - reinterpret_cast<uintptr_t>(info.dli_fbase));
        } else {
            text += fmt::format("{}{:#x}", separator, address);
        }
    }
    return text;
}

/// What the process printed last, on one line (iOS: the end of console.txt; nothing where the
/// console isn't captured). Metal names the descriptor or call it rejects there, then aborts.
std::string ConsoleTail() {
    static constexpr size_t MaxBytes = 2048;
    std::string text = Common::ConsoleCaptureTail(2 * MaxBytes);
    if (const size_t assertion = text.rfind("failed assertion"); assertion != std::string::npos) {
        // From the assertion's own line on, without the log lines before it.
        const size_t line = text.rfind('\n', assertion);
        text.erase(0, line != std::string::npos ? line + 1 : 0);
        text.resize((std::min)(text.size(), MaxBytes));
    } else if (text.size() > MaxBytes) {
        text.erase(0, text.size() - MaxBytes);
    }
    for (char& character : text) {
        if (static_cast<unsigned char>(character) < 0x20 || character == 0x7f) {
            character = ' ';
        }
    }
    return text.empty() ? "nothing captured" : text;
}

struct sigaction previous_abort {};
NSUncaughtExceptionHandler* previous_exception_handler = nullptr;

void OnAbort(int, siginfo_t*, void*) {
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set()) {
        char thread_name[64]{};
        pthread_getname_np(pthread_self(), thread_name, sizeof(thread_name));
        void* frames[32];
        const int count = backtrace(frames, 32);
        LOG_CRITICAL(Render_Metal,
                     "Process aborted (SIGABRT) on thread \"{}\"; frames: {}; the end of "
                     "console.txt, where Metal prints its reason first: {}",
                     thread_name, DescribeFrames(frames, count), ConsoleTail());
    }
    // The abort goes on as it would have without this handler.
    sigaction(SIGABRT, &previous_abort, nullptr);
    raise(SIGABRT);
}

void OnUncaughtException(NSException* exception) {
    void* frames[32];
    int count = 0;
    for (NSNumber* address in exception.callStackReturnAddresses) {
        if (count == 32) {
            break;
        }
        frames[count++] = reinterpret_cast<void*>(address.unsignedIntegerValue);
    }
    const char* const name = exception.name.UTF8String;
    const char* const reason = exception.reason.UTF8String;
    LOG_CRITICAL(Render_Metal,
                 "Uncaught Objective-C exception {}: {}; frames: {}; the end of console.txt: {}",
                 name != nullptr ? name : "?", reason != nullptr ? reason : "no reason",
                 DescribeFrames(frames, count), ConsoleTail());
    if (previous_exception_handler != nullptr) {
        previous_exception_handler(exception);
    }
}

/// Metal answers a descriptor or call it rejects by aborting the process, and nothing names a
/// SIGABRT or an uncaught Objective-C exception in the log otherwise: the log just stops. With
/// the Metal renderer both leave a line first. EDEN_NX2_METAL_ABORT_TRACE=0 leaves them alone.
struct AbortTrace {
    AbortTrace() {
        static std::once_flag installed;
        std::call_once(installed, [] {
            const char* const value = std::getenv("EDEN_NX2_METAL_ABORT_TRACE");
            if (value != nullptr && value[0] == '0') {
                return;
            }
            previous_exception_handler = NSGetUncaughtExceptionHandler();
            NSSetUncaughtExceptionHandler(&OnUncaughtException);
            struct sigaction action {};
            action.sa_sigaction = &OnAbort;
            action.sa_flags = SA_SIGINFO;
            sigemptyset(&action.sa_mask);
            sigaction(SIGABRT, &action, &previous_abort);
        });
    }
};

} // Anonymous namespace

struct RendererMetal::Impl {
    explicit Impl(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu,
                  Tegra::MaxwellDeviceMemoryManager& device_memory_)
        : device_memory{device_memory_}, scheduler{device},
          swapchain{device, emu_window.GetWindowInfo().render_surface},
          rasterizer{gpu, device_memory_, device, scheduler} {}

    /// First: the device and swapchain are created under it.
    AbortTrace abort_trace;

    /// The guest framebuffer as a texture: the texture cache's image, else read from guest
    /// memory (block linear) into a texture of our own.
    std::optional<DisplayTexture> PrepareLayer(const Tegra::FramebufferConfig& framebuffer,
                                               size_t layer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Device device;
    Scheduler scheduler;
    Swapchain swapchain;
    RasterizerMetal rasterizer;

    /// A layer read from guest memory (no texture cache image at its address).
    struct MemoryLayer {
        Object texture;
        Object upload;
        u8* upload_contents{};
        u64 upload_size{};
        u64 upload_tick{};
        u32 width{};
        u32 height{};
        Service::android::PixelFormat format{};
    };
    std::vector<MemoryLayer> memory_layers;
    SpatialUpscaler spatial;
    std::vector<u8> unswizzled;

    std::chrono::steady_clock::time_point report_start = std::chrono::steady_clock::now();
    u64 presented{};
    u64 skipped{};
    u64 from_cache{};
    u64 from_memory{};
};

std::optional<DisplayTexture> RendererMetal::Impl::PrepareLayer(
    const Tegra::FramebufferConfig& framebuffer, size_t layer) {
    const DAddr framebuffer_addr = framebuffer.address + framebuffer.offset;
    if (auto accelerated = rasterizer.AccelerateDisplay(framebuffer, framebuffer_addr)) {
        ++from_cache;
        return accelerated;
    }
    if (framebuffer.width == 0 || framebuffer.height == 0) {
        return std::nullopt;
    }
    if (memory_layers.size() <= layer) {
        memory_layers.resize(layer + 1);
    }
    MemoryLayer& state = memory_layers[layer];
    if (!state.texture || state.width != framebuffer.width || state.height != framebuffer.height ||
        state.format != framebuffer.pixel_format) {
        state.texture = CreateTexture(device, TextureDesc{
                                                  .kind = TextureKind::e2D,
                                                  .format = static_cast<u32>(
                                                      FramebufferFormat(framebuffer.pixel_format)),
                                                  .width = framebuffer.width,
                                                  .height = framebuffer.height,
                                              });
        state.width = framebuffer.width;
        state.height = framebuffer.height;
        state.format = framebuffer.pixel_format;
    }
    // The guest's framebuffers are block linear with 16-GOB blocks (as OpenGL reads them).
    constexpr u32 block_height_log2 = 4;
    const auto pixel_format =
        VideoCore::Surface::PixelFormatFromGPUPixelFormat(framebuffer.pixel_format);
    const u32 bytes_per_pixel = VideoCore::Surface::BytesPerBlock(pixel_format);
    const u64 size_in_bytes = Tegra::Texture::CalculateSize(
        true, bytes_per_pixel, framebuffer.stride, framebuffer.height, 1, block_height_log2, 0);
    const u8* const host_ptr = device_memory.GetPointer<u8>(framebuffer_addr);
    if (host_ptr == nullptr || !state.texture) {
        return std::nullopt;
    }
    const u64 bytes_per_row = u64{framebuffer.stride} * bytes_per_pixel;
    const u64 linear_size = bytes_per_row * framebuffer.height;
    unswizzled.resize(linear_size);
    Tegra::Texture::UnswizzleTexture(unswizzled, std::span<const u8>(host_ptr, size_in_bytes),
                                     bytes_per_pixel, framebuffer.stride, framebuffer.height, 1,
                                     block_height_log2, 0);
    // One upload buffer a layer: the previous frame's copy must be done before it is written.
    scheduler.Wait(state.upload_tick);
    if (linear_size > state.upload_size) {
        state.upload = CreateBuffer(device, linear_size, true, &state.upload_contents);
        state.upload_size = state.upload_contents != nullptr ? linear_size : 0;
    }
    if (state.upload_contents == nullptr) {
        return std::nullopt;
    }
    std::memcpy(state.upload_contents, unswizzled.data(), linear_size);
    scheduler.CopyBufferToTexture(state.upload.Get(), 0, static_cast<u32>(bytes_per_row), 0,
                                  state.texture.Get(), 0, 0, Offset3D{},
                                  Extent3D{framebuffer.width, framebuffer.height, 1});
    state.upload_tick = scheduler.CurrentTick();
    ++from_memory;
    return DisplayTexture{
        .texture = state.texture.Get(),
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
                "Metal renderer (experimental): draws, compute dispatches, occlusion queries, "
                "tessellation and mesh shaders (emulated) run on Metal; geometry shaders and "
                "transform feedback do not yet");
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
        // The guest's layers in order: the first replaces the picture, the others are blended
        // over it as their own blend mode says.
        std::vector<std::pair<DisplayTexture, size_t>> sources;
        for (size_t layer = 0; layer < framebuffers.size(); ++layer) {
            if (auto texture = impl->PrepareLayer(framebuffers[layer], layer)) {
                sources.emplace_back(*texture, layer);
            }
        }
        const id<CAMetalDrawable> drawable =
            impl->swapchain.NextDrawable(layout.width, layout.height);
        if (drawable != nil) {
            const Attachment target{.texture = (__bridge void*)drawable.texture};
            bool drawn = false;
            for (const auto& [source, layer] : sources) {
                const Tegra::FramebufferConfig& framebuffer = framebuffers[layer];
                const auto crop = Tegra::NormalizeCrop(framebuffer, source.width, source.height);
                const f32 width = static_cast<f32>(source.scaled_width);
                const f32 height = static_cast<f32>(source.scaled_height);
                const auto& screen = layout.screen;
                BlitBlend blend = BlitBlend::None;
                if (drawn) {
                    switch (framebuffer.blending) {
                    case Tegra::BlendMode::Premultiplied:
                        blend = BlitBlend::Premultiplied;
                        break;
                    case Tegra::BlendMode::Coverage:
                        blend = BlitBlend::Coverage;
                        break;
                    default:
                        break;
                    }
                }
                std::array<s32, 4> src_rect{static_cast<s32>(crop.left * width),
                                            static_cast<s32>(crop.top * height),
                                            static_cast<s32>(crop.right * width),
                                            static_cast<s32>(crop.bottom * height)};
                TextureHandle texture = source.texture;
                bool linear = true;
                if (!drawn && Settings::values.scaling_filter.GetValue() ==
                                  Settings::ScalingFilter::MetalFxSpatial) {
                    const u32 out_width = screen.right - screen.left;
                    const u32 out_height = screen.bottom - screen.top;
                    if (TextureHandle upscaled = impl->spatial.Upscale(
                            impl->device, impl->scheduler, texture, src_rect, out_width,
                            out_height)) {
                        texture = upscaled;
                        src_rect = {0, 0, static_cast<s32>(out_width),
                                    static_cast<s32>(out_height)};
                        linear = false;
                    }
                }
                const bool layer_drawn = impl->scheduler.Blit(BlitPass{
                    .src = texture,
                    .src_rect = src_rect,
                    .dst = target,
                    .dst_format = DrawableFormat(),
                    .dst_rect = {static_cast<s32>(screen.left), static_cast<s32>(screen.top),
                                 static_cast<s32>(screen.right), static_cast<s32>(screen.bottom)},
                    .dst_width = layout.width,
                    .dst_height = layout.height,
                    .linear = linear,
                    .keep_dst = drawn,
                    .blend = blend,
                });
                drawn = drawn || layer_drawn;
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
            impl->scheduler.NotePresent(drawn);
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
        const auto work = impl->rasterizer.TakeWorkCounts();
        const auto [encoders, command_buffers, blits, passes, clears_folded, state_sets,
                    state_skips, splits, ahead_copies, ahead_kernels, idle_fences] =
            impl->scheduler.TakeCounts();
        const auto [graphics_made, compute_made, refused] =
            impl->rasterizer.Pipelines().TakeCounts();
        LOG_INFO(Render_Metal,
                 "Metal: {} frames presented ({} from the texture cache, {} from guest memory), "
                 "{} without a drawable; {} draws ({} with a built pipeline, {} drawn, {} with "
                 "rewritten indices, {} of them by a kernel) and {} dispatches ({} with a "
                 "pipeline); {} clears, {} texel buffer copies, {} occlusion queries, {} emulated "
                 "mesh draws ({} workgroups), {} tessellated draws ({} patches), {} Ocarina "
                 "upscales bilinear; pipelines made {} graphics, {} compute, {} refused; {} command "
                 "buffers, {} encoders ({} render passes, {} ended by a fragment barrier, {} "
                 "starting with a clear), {} blit commands; state and bindings {} set, {} left as "
                 "they were; {} copies and {} kernels ahead of their segment, {} segments split "
                 "off; {} fences without a command buffer; staging {} MiB ({} early commits, {} "
                 "waited for, {} buffers freed); textures {} MiB, buffers {} MiB",
                 impl->presented, impl->from_cache, impl->from_memory, impl->skipped, work.draws,
                 work.draws_with_pipeline, work.draws_encoded, work.draws_converted,
                 work.indices_rewritten_on_gpu, work.dispatches, work.dispatches_with_pipeline,
                 work.clears, work.texel_copies, work.occlusion_queries, work.mesh_draws,
                 work.mesh_groups, work.tessellated_draws, work.tessellated_patches,
                 work.ocarina_bilinear, graphics_made, compute_made, refused, command_buffers,
                 encoders, passes, work.barrier_pass_ends, clears_folded, blits, state_sets,
                 state_skips, ahead_copies, ahead_kernels, splits, idle_fences,
                 work.staging_bytes >> 20, work.staging_flushes, work.staging_waits,
                 work.staging_freed,
                 impl->rasterizer.TextureRuntime().allocated_bytes >> 20,
                 impl->rasterizer.BufferRuntime().GetDeviceMemoryUsage() >> 20);
        if (const std::string coverage = Coverage::TakeLine(); !coverage.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal coverage: {}", coverage);
        }
        if (const auto mesh = impl->rasterizer.TakeMeshCounts();
            mesh.native_draws + mesh.emulated_draws != 0) {
            LOG_INFO(Render_Metal,
                     "Perf Metal mesh: {} draws with mesh functions ({} with a task count the GPU "
                     "wrote, {} from indirect records), {} through the compute emulation",
                     mesh.native_draws, mesh.from_gpu, mesh.indirect, mesh.emulated_draws);
        }
        if (const std::string deadlines = Common::FrameDeadline::TakeStatusText();
            !deadlines.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal threads{}", deadlines);
        }
        // The pipeline archive: builds it served, and the build time that saved so far (each hit
        // at the average miss's cost, less what the hit took).
        PipelineArchive& archive = impl->rasterizer.Pipelines().Archive();
        const PipelineArchive::Counts recent = archive.TakeCounts();
        if (recent.hits + recent.misses != 0) {
            const PipelineArchive::Counts total = archive.Totals();
            const f64 miss_ms = total.misses != 0
                                    ? static_cast<f64>(total.miss_ns) / total.misses / 1e6
                                    : 0.0;
            const f64 saved_ms = (std::max)(
                miss_ms * static_cast<f64>(total.hits) - static_cast<f64>(total.hit_ns) / 1e6,
                0.0);
            LOG_INFO(Render_Metal,
                     "Metal pipeline archive: {} hits ({:.1f} ms), {} misses ({:.1f} ms), {} added, "
                     "{} not added; since boot {} hits, {} misses, ~{:.0f} ms of builds saved",
                     recent.hits, static_cast<f64>(recent.hit_ns) / 1e6, recent.misses,
                     static_cast<f64>(recent.miss_ns) / 1e6, recent.added, recent.add_failures,
                     total.hits, total.misses, saved_ms);
        }
        // Per frame, to set against MoltenVK's Perf: and Perf passes: lines.
        const f64 seconds = std::chrono::duration<f64>(now - impl->report_start).count();
        const f64 frames = static_cast<f64>((std::max)(impl->presented, u64{1}));
        LOG_INFO(Render_Metal,
                 "Metal perf: {:.1f} fps, GPU {:.2f} ms/frame, {:.1f} command buffers, {:.1f} "
                 "encoders and {:.1f} render passes per frame, {:.0f} draws per frame",
                 static_cast<f64>(impl->presented) / seconds,
                 static_cast<f64>(impl->scheduler.TakeGpuNanoseconds()) / 1e6 / frames,
                 static_cast<f64>(command_buffers) / frames, static_cast<f64>(encoders) / frames,
                 static_cast<f64>(passes) / frames, static_cast<f64>(work.draws_encoded) / frames);
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
