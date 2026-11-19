// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
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
#include "core/hle/service/nvnflinger/hwc_layer.h"
#include "video_core/capture.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_perf.h"
#ifdef HAS_RESHADE
#include "video_core/post_processing/fx_chain.h"
#endif
#include "video_core/render_probe.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_coverage.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_frame_gen.h"
#include "video_core/renderer_metal/metal_post_process.h"
#include "video_core/renderer_metal/metal_present.h"
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

[[nodiscard]] bool LazyAppletCaptureEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_LAZY_APPLET_CAPTURE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] bool NativeLayerTilingEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_NATIVE_TILING");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

[[nodiscard]] bool AsyncPresentEnabled() {
    static const bool allowed = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ASYNC_PRESENT");
        return value == nullptr || value[0] != '0';
    }();
    return allowed && Settings::values.async_presentation.GetValue();
}

constexpr const char* PresentCopySource = R"(
#include <metal_stdlib>
using namespace metal;
struct PresentOut {
    float4 position [[position]];
    float2 uv;
};
vertex PresentOut present_vertex(uint id [[vertex_id]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    PresentOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = corner;
    return out;
}
fragment float4 present_fragment(PresentOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                                 sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv);
}
)";

/// MetalFX spatial upscaling of the first layer (the MetalFxSpatial scaling filter), on the
/// renderer's own textures and command buffer.
class SpatialUpscaler {
public:
    /// The upscaled picture (output_width x output_height, the drawable's format), or nullptr
    /// when MetalFX can't run (the layer is presented as it is).
    TextureHandle Upscale(const Device& device, Scheduler& scheduler, TextureHandle source,
                          const std::array<s32, 4>& source_rect, u32 output_width,
                          u32 output_height) {
        if (device_unsupported || source == nullptr || output_width == 0 || output_height == 0) {
            return nullptr;
        }
        if (@available(iOS 16.0, macOS 13.0, *)) {
            const u32 input_width = static_cast<u32>(std::abs(source_rect[2] - source_rect[0]));
            const u32 input_height = static_cast<u32>(std::abs(source_rect[3] - source_rect[1]));
            // MetalFX spatial only upscales strictly larger outputs; 1:1, downscales and tiny
            // transient crops fall back for the frame without disabling future frames.
            if (input_width == 0 || input_height == 0 || output_width <= input_width ||
                output_height <= input_height) {
                return nullptr;
            }
            const id<MTLTexture> src_tex = AsTexture(source);
            if (src_tex == nil) {
                return nullptr;
            }
            const MTLPixelFormat src_format = src_tex.pixelFormat;
            const MTLPixelFormat format = SupportedInputFormat(src_format);
            if (format == MTLPixelFormatInvalid) {
                return nullptr;
            }
            if (scaler == nil || input_width != in_width || input_height != in_height ||
                output_width != out_width || output_height != out_height ||
                format != in_format) {
                if (!Prepare(device, scheduler, format, input_width, input_height, output_width,
                             output_height)) {
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
                .linear = format != src_format,
                .keep_dst = false,
            });
            if (!copied) {
                return nullptr;
            }
            const id<MTLCommandBuffer> command_buffer =
                (__bridge id<MTLCommandBuffer>)scheduler.CommandBuffer();
            if (command_buffer == nil) {
                return nullptr;
            }
            id<MTLFXSpatialScaler> spatial = scaler;
            spatial.colorTexture = input;
            spatial.outputTexture = output;
            spatial.inputContentWidth = input_width;
            spatial.inputContentHeight = input_height;
            @try {
                [spatial encodeToCommandBuffer:command_buffer];
            } @catch (NSException* exception) {
                const char* const reason = exception.reason.UTF8String;
                LOG_WARNING(Render_Metal, "MetalFX spatial: encoding raised: {}",
                            reason != nullptr ? reason : "no reason");
                scaler = nil;
                return nullptr;
            }
            return (__bridge void*)output;
        }
        device_unsupported = true;
        return nullptr;
    }

private:
    static MTLPixelFormat SupportedInputFormat(MTLPixelFormat format) noexcept {
        switch (format) {
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGB10A2Unorm:
        case MTLPixelFormatRGBA16Float:
        case MTLPixelFormatRG11B10Float:
            return format;
        default:
            // Any other displayable format blits cleanly into RGBA16Float for MetalFX.
            return format != MTLPixelFormatInvalid ? MTLPixelFormatRGBA16Float
                                                   : MTLPixelFormatInvalid;
        }
    }

    bool Prepare(const Device& device, Scheduler& scheduler, MTLPixelFormat format,
                 u32 input_width, u32 input_height, u32 output_width,
                 u32 output_height) API_AVAILABLE(ios(16.0), macos(13.0)) {
        if (![MTLFXSpatialScalerDescriptor supportsDevice:device.GetDevice()]) {
            LOG_WARNING(Render_Metal, "MetalFX spatial: not supported on this GPU");
            device_unsupported = true;
            return false;
        }
        if (input != nil || output != nil) {
            scheduler.Finish();
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
        id<MTLFXSpatialScaler> spatial = nil;
        @try {
            spatial = [descriptor newSpatialScalerWithDevice:device.GetDevice()];
        } @catch (NSException* exception) {
            const char* const reason = exception.reason.UTF8String;
            LOG_WARNING(Render_Metal, "MetalFX spatial: creating scaler {}x{} -> {}x{} raised: {}",
                        input_width, input_height, output_width, output_height,
                        reason != nullptr ? reason : "no reason");
            spatial = nil;
        }
        if (spatial == nil) {
            if (warned_sizes < 8) {
                ++warned_sizes;
                LOG_WARNING(Render_Metal, "MetalFX spatial: no scaler for {}x{} -> {}x{}",
                            input_width, input_height, output_width, output_height);
            }
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
        id<MTLTexture> new_input =
            make(input_width, input_height, format,
                 spatial.colorTextureUsage | MTLTextureUsageRenderTarget |
                     MTLTextureUsageShaderRead);
        id<MTLTexture> new_output =
            make(output_width, output_height, static_cast<MTLPixelFormat>(DrawableFormat()),
                 spatial.outputTextureUsage | MTLTextureUsageShaderRead);
        if (new_input == nil || new_output == nil) {
            return false;
        }
        scaler = spatial;
        input = new_input;
        output = new_output;
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
    u32 warned_sizes{};
    MTLPixelFormat in_format{MTLPixelFormatInvalid};
    bool device_unsupported{};
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

    ~Impl() {
        StopAsyncPresent();
    }

    /// First: the device and swapchain are created under it.
    AbortTrace abort_trace;

    /// The guest framebuffer as a texture: the texture cache's image, else read from guest
    /// memory (block linear) into a texture of our own.
    std::optional<DisplayTexture> PrepareLayer(const Tegra::FramebufferConfig& framebuffer,
                                               size_t layer);

    /// The layers drawn into `target` (`target_format`, `layout.width` x `layout.height`) at the
    /// layout's screen; with `apply_filters` the scaling filter and anti-aliasing settings apply.
    /// Whether a layer was drawn.
    bool DrawLayers(std::span<const Tegra::FramebufferConfig> framebuffers, Attachment target,
                    u32 target_format, const Layout::FramebufferLayout& layout,
                    bool apply_filters);

#ifdef HAS_LSFG
    /// Frame generation (FSR 3 or MetalFX): the layers drawn into a texture of our own, the
    /// generated frames and then the real one blitted to drawables, each held for its share of
    /// the real frame interval (Vulkan's Composite and present thread). Whether a real frame was
    /// drawn.
    bool CompositeGenerated(std::span<const Tegra::FramebufferConfig> framebuffers,
                            id<CAMetalDrawable> drawable,
                            const Layout::FramebufferLayout& layout);

    /// `texture` into `drawable` and presented on its own command buffer, after the previous
    /// drawable has been shown `hold_seconds`.
    void PresentTexture(TextureHandle texture, id<CAMetalDrawable> drawable,
                        const Layout::FramebufferLayout& layout, f64 hold_seconds, bool guest);

    FrameGen frame_gen{device, scheduler};
    Object compose;
    std::array<u32, 2> compose_size{};
    bool refresh_published{};
    u64 generated_since_report{};
#endif

    /// Vulkan's RenderAppletCaptureLayer: the LastFrame layer stack drawn into the capture texture
    /// (1280x720 RGBA8) each frame, for GetAppletCaptureBuffer.
    void RenderAppletCapture(std::span<const Tegra::FramebufferConfig> framebuffers);

    /// A texture `width` x `height` read back into memory of `size` bytes (rows of width * 4).
    bool ReadTexture(TextureHandle texture, u32 width, u32 height, std::span<u8> out);

    Object applet_capture;
    std::vector<Tegra::FramebufferConfig> applet_layers;
    std::vector<Tegra::FramebufferConfig> last_framebuffers;
    Object screenshot_target;
    u32 screenshot_width{};
    u32 screenshot_height{};
    std::vector<Tegra::FramebufferConfig> screenshot_layers;
    Object readback;
    u64 readback_size{};
    u8* readback_contents{};

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Device device;
    Scheduler scheduler;
    Swapchain swapchain;
    RasterizerMetal rasterizer;

    /// A layer read from guest memory (no texture cache image at its address).
    struct MemoryLayer {
        static constexpr size_t UploadSlots = 3;
        struct UploadSlot {
            Object buffer;
            u8* contents{};
            u64 size{};
            u64 tick{};
        };
        Object texture;
        std::array<UploadSlot, UploadSlots> uploads{};
        size_t next_upload{};
        u32 width{};
        u32 height{};
        Service::android::PixelFormat format{};
    };
    std::vector<MemoryLayer> memory_layers;
    SpatialUpscaler spatial;
    PresentPasses present{device, scheduler};
#ifdef HAS_RESHADE
    PostProcess post_process{device, scheduler};
#endif

    std::chrono::steady_clock::time_point report_start = std::chrono::steady_clock::now();
    std::atomic<u64> presented{0};
    std::atomic<u64> skipped{0};
    std::atomic<u64> drawable_wait_ns{0}; ///< nextDrawable waits since the last report
    std::atomic<u64> drawable_wait_max_ns{0};
    u64 from_cache{};
    u64 from_memory{};

    static constexpr size_t AsyncPresentSlots = 3;
    struct AsyncSlot {
        Object texture;
        u32 width{};
        u32 height{};
        u64 render_tick{};
    };
    struct PresentJob {
        size_t slot_index{};
        TextureHandle texture{};
        u32 width{};
        u32 height{};
        u64 ready_tick{};
        bool drawn{};
        bool nearest{};
    };
    struct AsyncPresentSync {
        std::mutex mutex;
        std::condition_variable work_cv;
        std::condition_variable free_cv;
        std::deque<PresentJob> jobs;
        std::array<bool, AsyncPresentSlots> slot_busy{};
        bool worker_busy{};
        bool stop{};
    };
    std::mutex swapchain_pacing_mutex;
    std::shared_ptr<AsyncPresentSync> async_sync = std::make_shared<AsyncPresentSync>();
    std::array<AsyncSlot, AsyncPresentSlots> async_slots{};
    size_t next_async_slot{};
    std::thread async_present_thread;
    id<MTLCommandQueue> async_present_queue = nil;
    id<MTLRenderPipelineState> async_present_pipeline = nil;
    id<MTLSamplerState> async_present_sampler = nil;
    id<MTLSamplerState> async_present_nearest_sampler = nil;
    bool async_present_failed{};

    void NoteDrawableWait(u64 waited_ns) noexcept;
    void ClearAttachment(Attachment target, u32 width, u32 height,
                         std::array<f64, 4> color);
    bool EnsureAsyncPresenter();
    void RunAsyncPresentWorker();
    bool SubmitAsyncPresent(std::span<const Tegra::FramebufferConfig> framebuffers,
                            const Layout::FramebufferLayout& layout);
    void DrainAsyncPresent();
    void StopAsyncPresent();
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
                                                  .pixel_format_view = false,
                                              });
        state.width = framebuffer.width;
        state.height = framebuffer.height;
        state.format = framebuffer.pixel_format;
    }
    // The guest's framebuffers are block linear with 16-GOB blocks (Switch 2: native GOB tiling,
    // matching Vulkan's Layer::UpdateRawImage).
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
    const u64 bytes_per_row = u64{framebuffer.width} * bytes_per_pixel;
    const u64 linear_size = bytes_per_row * framebuffer.height;
    MemoryLayer::UploadSlot& slot = state.uploads[state.next_upload];
    state.next_upload = (state.next_upload + 1) % MemoryLayer::UploadSlots;
    scheduler.Wait(slot.tick);
    if (linear_size > slot.size) {
        slot.buffer = CreateBuffer(device, linear_size, true, &slot.contents);
        slot.size = slot.contents != nullptr ? linear_size : 0;
    }
    if (slot.contents == nullptr) {
        return std::nullopt;
    }
    Tegra::Texture::UnswizzleTexture(
        std::span<u8>(slot.contents, linear_size), std::span<const u8>(host_ptr, size_in_bytes),
        bytes_per_pixel, framebuffer.width, framebuffer.height, 1, block_height_log2, 0, 1,
        NativeLayerTilingEnabled() && framebuffer.native_tiling);
    scheduler.CopyBufferToTexture(slot.buffer.Get(), 0, static_cast<u32>(bytes_per_row), 0,
                                  state.texture.Get(), 0, 0, Offset3D{},
                                  Extent3D{framebuffer.width, framebuffer.height, 1});
    slot.tick = scheduler.CurrentTick();
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
#ifdef HAS_RESHADE
    // The post-processing chain the settings name, as Vulkan's renderer loads it.
    VideoCore::FxChain::Instance().LoadFromSettings();
#endif
    LOG_WARNING(Render_Metal,
                "Metal renderer (experimental): draws, compute dispatches, occlusion queries, "
                "tessellation, mesh shaders and transform feedback (emulated) run on Metal; "
                "geometry shaders do not yet");
}

RendererMetal::~RendererMetal() = default;

bool RendererMetal::Impl::DrawLayers(std::span<const Tegra::FramebufferConfig> framebuffers,
                                     Attachment target, u32 target_format,
                                     const Layout::FramebufferLayout& layout, bool apply_filters) {
    // The guest's layers in order: the first replaces the picture, the others are blended over
    // it as their own blend mode says.
    std::vector<std::pair<DisplayTexture, size_t>> sources;
    for (size_t layer = 0; layer < framebuffers.size(); ++layer) {
        if (auto texture = PrepareLayer(framebuffers[layer], layer)) {
            sources.emplace_back(*texture, layer);
        }
    }
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
        const Settings::ScalingFilter filter = Settings::values.scaling_filter.GetValue();
        bool linear = filter != Settings::ScalingFilter::NearestNeighbor &&
                      filter != Settings::ScalingFilter::Mmpx;
        const bool is_applet = (framebuffer.layer_stack_mask &
                                Service::Nvnflinger::LayerStackBit(
                                    Service::Nvnflinger::LayerStackId::Recording)) == 0;
        const bool already_upscaled = source.already_upscaled;
        const bool anti_aliased = already_upscaled || source.anti_aliased;
        const s32 screen_width = static_cast<s32>(screen.right - screen.left);
        const s32 screen_height = static_cast<s32>(screen.bottom - screen.top);
        if (apply_filters && PresentPasses::Enabled() && screen_width > 0 && screen_height > 0) {
            // Vulkan's Layer::ConfigureDraw: anti-aliasing over the layer's texture (skipped when
            // Ocarina temporal AA/upscaling already ran), then FSR/FSR3/SGSR to the screen's size,
            // then the window filter.
            PresentPasses::Source passes_source{
                .texture = texture,
                .size = {source.scaled_width, source.scaled_height},
                .render = {source.scaled_width, source.scaled_height},
                .crop = {crop.left, crop.top, crop.right, crop.bottom},
            };
            if (!anti_aliased && !is_applet) {
                passes_source = present.AntiAlias(
                    passes_source, Settings::values.anti_aliasing.GetValue());
            }
#ifdef HAS_RESHADE
            // The post-processing chain (Vulkan's Layer::ConfigureDraw): the guest's own layers,
            // not the applet's, before the upscaler.
            if (!already_upscaled && !is_applet && PostProcess::Wanted()) {
                passes_source.texture =
                    post_process.Apply(passes_source.texture, passes_source.size);
            }
#endif
            const bool skip_sgsr = is_applet &&
                                   (filter == Settings::ScalingFilter::Sgsr ||
                                    filter == Settings::ScalingFilter::SgsrEdge);
            if (!already_upscaled && !skip_sgsr) {
                passes_source = present.Upscale(
                    passes_source, filter,
                    {static_cast<u32>(screen_width), static_cast<u32>(screen_height)});
            }
            if (PresentPasses::HasWindowShader(filter) &&
                present.DrawWindow(
                    passes_source, filter, target, target_format,
                    {layout.width, layout.height},
                    {static_cast<s32>(screen.left), static_cast<s32>(screen.top),
                     static_cast<s32>(screen.right), static_cast<s32>(screen.bottom)},
                    drawn, blend)) {
                drawn = true;
                continue;
            }
            texture = passes_source.texture;
            const f32 pass_width = static_cast<f32>(passes_source.size[0]);
            const f32 pass_height = static_cast<f32>(passes_source.size[1]);
            src_rect = {static_cast<s32>(passes_source.crop[0] * pass_width),
                        static_cast<s32>(passes_source.crop[1] * pass_height),
                        static_cast<s32>(passes_source.crop[2] * pass_width),
                        static_cast<s32>(passes_source.crop[3] * pass_height)};
            linear = filter != Settings::ScalingFilter::NearestNeighbor &&
                     filter != Settings::ScalingFilter::Mmpx;
        }
        if (!drawn && apply_filters && !already_upscaled && !is_applet &&
            filter == Settings::ScalingFilter::MetalFxSpatial) {
            const u32 out_width = screen.right - screen.left;
            const u32 out_height = screen.bottom - screen.top;
            if (TextureHandle upscaled = spatial.Upscale(
                    device, scheduler, texture, src_rect, out_width,
                    out_height)) {
                texture = upscaled;
                src_rect = {0, 0, static_cast<s32>(out_width),
                            static_cast<s32>(out_height)};
                linear = false;
            }
        }
        const bool layer_drawn = scheduler.Blit(BlitPass{
            .src = texture,
            .src_rect = src_rect,
            .dst = target,
            .dst_format = target_format,
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
    return drawn;
}

#ifdef HAS_LSFG
void RendererMetal::Impl::PresentTexture(TextureHandle texture, id<CAMetalDrawable> drawable,
                                         const Layout::FramebufferLayout& layout,
                                         f64 hold_seconds, bool guest) {
    const id<MTLTexture> dst_tex = drawable.texture;
    const u32 dst_w = dst_tex != nil ? static_cast<u32>(dst_tex.width) : layout.width;
    const u32 dst_h = dst_tex != nil ? static_cast<u32>(dst_tex.height) : layout.height;
    scheduler.Blit(BlitPass{
        .src = texture,
        .src_rect = {0, 0, static_cast<s32>(layout.width), static_cast<s32>(layout.height)},
        .dst = {.texture = (__bridge void*)dst_tex},
        .dst_format = DrawableFormat(),
        .dst_rect = {0, 0, static_cast<s32>(dst_w), static_cast<s32>(dst_h)},
        .dst_width = dst_w,
        .dst_height = dst_h,
        .linear = dst_w != layout.width || dst_h != layout.height,
        .keep_dst = false,
    });
    const id<MTLCommandBuffer> command_buffer =
        (__bridge id<MTLCommandBuffer>)scheduler.CommandBuffer(true);
    {
        std::scoped_lock lock{swapchain_pacing_mutex};
        swapchain.PresentHeld(command_buffer, drawable, hold_seconds);
    }
    scheduler.NotePresent(guest);
    // Each present on a command buffer of its own: the hold counts from the one before.
    scheduler.Flush(CommitReason::Present);
    presented.fetch_add(1, std::memory_order_relaxed);
}

bool RendererMetal::Impl::CompositeGenerated(std::span<const Tegra::FramebufferConfig> framebuffers,
                                             id<CAMetalDrawable> drawable,
                                             const Layout::FramebufferLayout& layout) {
    const std::array<u32, 2> size{layout.width, layout.height};
    if (!compose || compose_size != size) {
        scheduler.Finish();
        compose = CreateTexture(device, {.kind = TextureKind::e2D,
                                         .format = DrawableFormat(),
                                         .width = size[0],
                                         .height = size[1],
                                         .render_target = true,
                                         .pixel_format_view = false});
        compose_size = size;
    }
    if (!compose) {
        return false;
    }
    const Attachment target{.texture = compose.Get()};
    const bool drawn = DrawLayers(framebuffers, target, DrawableFormat(), layout, true);
    if (!drawn) {
        ClearAttachment(target, layout.width, layout.height, {0.06, 0.09, 0.22, 1.0});
    }
    if (!refresh_published) {
        refresh_published = true;
        f64 refresh_ns = 0.0;
        {
            std::scoped_lock lock{swapchain_pacing_mutex};
            refresh_ns = swapchain.DisplayRefreshNs();
        }
        if (refresh_ns > 0.0) {
            // The pacer follows the display: no more frames than it shows.
            VideoCommon::Nx2Perf::g_display_refresh_ns.store(static_cast<long long>(refresh_ns),
                                                            std::memory_order_relaxed);
        }
    }
    std::array<u32, 2> guest_size{};
    if (!framebuffers.empty()) {
        const auto& guest = framebuffers.front();
        const bool cropped = !guest.crop_rect.IsEmpty();
        guest_size = {cropped ? static_cast<u32>(guest.crop_rect.GetWidth()) : guest.width,
                      cropped ? static_cast<u32>(guest.crop_rect.GetHeight()) : guest.height};
    }
    const size_t count =
        frame_gen.Process(compose.Get(), size, DrawableFormat(), guest_size);
    // Each generated frame gets an equal share of the real frame interval before the next
    // present (zortos293/Lichii#783).
    const f64 spacing = static_cast<f64>((std::min)(frame_gen.FrameInterval(), 0.1f)) /
                        static_cast<f64>(count + 1);
    id<CAMetalDrawable> next = drawable;
    f64 hold = 0.0;
    for (size_t generation = 0; generation < count; ++generation) {
        bool presented_gen = false;
        @autoreleasepool {
            const TextureHandle generated = frame_gen.Generate(generation);
            if (generated != nullptr) {
                if (next == nil) {
                    next = swapchain.NextDrawable(layout.width, layout.height);
                }
                if (next != nil) {
                    PresentTexture(generated, next, layout, hold, false);
                    next = nil;
                    ++generated_since_report;
                    hold = spacing;
                    presented_gen = true;
                }
            }
        }
        if (!presented_gen) {
            break;
        }
    }
    @autoreleasepool {
        if (next == nil) {
            next = swapchain.NextDrawable(layout.width, layout.height);
        }
        if (next != nil) {
            PresentTexture(compose.Get(), next, layout, hold, drawn);
        } else {
            skipped.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return drawn;
}
#endif

void RendererMetal::Impl::NoteDrawableWait(u64 waited_ns) noexcept {
    drawable_wait_ns.fetch_add(waited_ns, std::memory_order_relaxed);
    u64 prev_max = drawable_wait_max_ns.load(std::memory_order_relaxed);
    while (waited_ns > prev_max &&
           !drawable_wait_max_ns.compare_exchange_weak(prev_max, waited_ns,
                                                       std::memory_order_relaxed)) {
    }
}

void RendererMetal::Impl::ClearAttachment(Attachment target, u32 width, u32 height,
                                          std::array<f64, 4> color) {
    ClearPass clear{};
    clear.color[0] = target;
    clear.clear_color[0] = true;
    clear.color_value[0] = color;
    clear.width = width;
    clear.height = height;
    scheduler.Clear(clear);
}

bool RendererMetal::Impl::EnsureAsyncPresenter() {
    if (async_present_thread.joinable()) {
        return true;
    }
    if (async_present_failed) {
        return false;
    }
    const id<MTLDevice> mtl = device.GetDevice();
    async_present_queue = [mtl newCommandQueueWithMaxCommandBufferCount:16];
    if (async_present_queue != nil) {
        async_present_queue.label = @"LichiiPresent";
    }
    NSError* error = nil;
    const id<MTLLibrary> lib = [mtl newLibraryWithSource:@(PresentCopySource)
                                                 options:nil
                                                   error:&error];
    if (lib != nil) {
        MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
        desc.label = @"Async Present";
        desc.vertexFunction = [lib newFunctionWithName:@"present_vertex"];
        desc.fragmentFunction = [lib newFunctionWithName:@"present_fragment"];
        desc.colorAttachments[0].pixelFormat = swapchain.PixelFormat();
        async_present_pipeline = [mtl newRenderPipelineStateWithDescriptor:desc error:&error];
    }
    MTLSamplerDescriptor* sdesc = [MTLSamplerDescriptor new];
    sdesc.minFilter = MTLSamplerMinMagFilterLinear;
    sdesc.magFilter = MTLSamplerMinMagFilterLinear;
    sdesc.sAddressMode = MTLSamplerAddressModeClampToEdge;
    sdesc.tAddressMode = MTLSamplerAddressModeClampToEdge;
    async_present_sampler = [mtl newSamplerStateWithDescriptor:sdesc];
    sdesc.minFilter = MTLSamplerMinMagFilterNearest;
    sdesc.magFilter = MTLSamplerMinMagFilterNearest;
    async_present_nearest_sampler = [mtl newSamplerStateWithDescriptor:sdesc];
    if (async_present_queue == nil || async_present_pipeline == nil ||
        async_present_sampler == nil || async_present_nearest_sampler == nil) {
        LOG_WARNING(Render_Metal, "Metal: async present worker unavailable ({}); presenting inline",
                    error != nil ? error.localizedDescription.UTF8String : "resource init failed");
        async_present_queue = nil;
        async_present_pipeline = nil;
        async_present_sampler = nil;
        async_present_nearest_sampler = nil;
        async_present_failed = true;
        return false;
    }
    async_present_thread = std::thread([this] { RunAsyncPresentWorker(); });
    return true;
}

void RendererMetal::Impl::RunAsyncPresentWorker() {
    pthread_setname_np("MetalPresent");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    const id<MTLSharedEvent> shared_event = (__bridge id<MTLSharedEvent>)scheduler.SharedEvent();
    for (;;) {
        PresentJob job{};
        {
            std::unique_lock lock{async_sync->mutex};
            async_sync->work_cv.wait(lock, [this] {
                return async_sync->stop || !async_sync->jobs.empty();
            });
            if (async_sync->stop || async_sync->jobs.empty()) {
                return;
            }
            job = async_sync->jobs.front();
            async_sync->jobs.pop_front();
            async_sync->worker_busy = true;
        }
        @autoreleasepool {
            const auto drawable_start = std::chrono::steady_clock::now();
            const id<CAMetalDrawable> drawable = swapchain.NextDrawable(job.width, job.height);
            const u64 waited = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - drawable_start)
                    .count());
            NoteDrawableWait(waited);
            bool submitted_cb = false;
            if (drawable != nil) {
                const id<MTLTexture> dst_tex = drawable.texture;
                const u32 dst_w = dst_tex != nil ? static_cast<u32>(dst_tex.width) : 0;
                const u32 dst_h = dst_tex != nil ? static_cast<u32>(dst_tex.height) : 0;
                const id<MTLCommandBuffer> cb =
                    (dst_w > 0 && dst_h > 0) ? [async_present_queue commandBuffer] : nil;
                if (cb != nil) {
                    cb.label = @"Lichii present";
                    if (shared_event != nil && job.ready_tick != 0) {
                        [cb encodeWaitForEvent:shared_event value:job.ready_tick];
                    }
                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.colorAttachments[0].texture = dst_tex;
                    pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
                    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
                    pass.renderTargetWidth = dst_w;
                    pass.renderTargetHeight = dst_h;
                    const id<MTLRenderCommandEncoder> enc =
                        [cb renderCommandEncoderWithDescriptor:pass];
                    if (enc != nil) {
                        enc.label = @"Present Blit";
                        [enc setRenderPipelineState:async_present_pipeline];
                        [enc setViewport:MTLViewport{0.0, 0.0, static_cast<f64>(dst_w),
                                                     static_cast<f64>(dst_h), 0.0, 1.0}];
                        [enc setScissorRect:MTLScissorRect{0, 0, dst_w, dst_h}];
                        [enc setFragmentTexture:AsTexture(job.texture) atIndex:0];
                        [enc setFragmentSamplerState:job.nearest ? async_present_nearest_sampler
                                                                 : async_present_sampler
                                             atIndex:0];
                        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
                        [enc endEncoding];
                    }
                    {
                        std::scoped_lock lock{swapchain_pacing_mutex};
                        swapchain.Present(cb, drawable, job.drawn);
                    }
                    std::shared_ptr<AsyncPresentSync> sync_ref = async_sync;
                    const size_t slot_idx = job.slot_index;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
                        {
                            std::scoped_lock lock{sync_ref->mutex};
                            sync_ref->slot_busy[slot_idx] = false;
                        }
                        sync_ref->free_cv.notify_all();
                    }];
                    [cb commit];
                    submitted_cb = true;
                    presented.fetch_add(1, std::memory_order_relaxed);
                }
            }
            if (!submitted_cb) {
                skipped.fetch_add(1, std::memory_order_relaxed);
                {
                    std::scoped_lock lock{async_sync->mutex};
                    async_sync->slot_busy[job.slot_index] = false;
                }
                async_sync->free_cv.notify_all();
            }
        }
        {
            std::scoped_lock lock{async_sync->mutex};
            async_sync->worker_busy = false;
        }
        async_sync->free_cv.notify_all();
    }
}

bool RendererMetal::Impl::SubmitAsyncPresent(
    std::span<const Tegra::FramebufferConfig> framebuffers,
    const Layout::FramebufferLayout& layout) {
    if (!EnsureAsyncPresenter()) {
        return false;
    }
    if (layout.width == 0 || layout.height == 0) {
        skipped.fetch_add(1, std::memory_order_relaxed);
        scheduler.Flush(CommitReason::Present);
        return true;
    }
    const size_t slot_idx = next_async_slot;
    next_async_slot = (next_async_slot + 1) % AsyncPresentSlots;
    {
        const auto wait_start = std::chrono::steady_clock::now();
        std::unique_lock lock{async_sync->mutex};
        if (async_sync->slot_busy[slot_idx]) {
            async_sync->free_cv.wait(lock, [&] { return !async_sync->slot_busy[slot_idx]; });
            const u64 waited = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_start)
                    .count());
            VideoCommon::SplitStats::AddPresentBlock(waited);
        }
    }
    AsyncSlot& slot = async_slots[slot_idx];
    if (!slot.texture || slot.width != layout.width || slot.height != layout.height) {
        if (slot.render_tick != 0 && !scheduler.IsFree(slot.render_tick)) {
            scheduler.Wait(slot.render_tick);
        }
        slot.texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                              .format = DrawableFormat(),
                                              .width = layout.width,
                                              .height = layout.height,
                                              .render_target = true,
                                              .pixel_format_view = false});
        slot.width = layout.width;
        slot.height = layout.height;
    }
    if (!slot.texture) {
        return false;
    }
    const Attachment target{.texture = slot.texture.Get()};
    const bool drawn = DrawLayers(framebuffers, target, DrawableFormat(), layout, true);
    if (!drawn) {
        ClearAttachment(target, layout.width, layout.height, {0.06, 0.09, 0.22, 1.0});
    }
    scheduler.NotePresent(drawn);
    const u64 ready_tick = scheduler.Flush(CommitReason::Present);
    slot.render_tick = ready_tick;
    const Settings::ScalingFilter filter = Settings::values.scaling_filter.GetValue();
    const bool nearest = filter == Settings::ScalingFilter::NearestNeighbor ||
                         filter == Settings::ScalingFilter::Mmpx;
    {
        std::scoped_lock lock{async_sync->mutex};
        async_sync->slot_busy[slot_idx] = true;
        async_sync->jobs.push_back(PresentJob{
            .slot_index = slot_idx,
            .texture = slot.texture.Get(),
            .width = layout.width,
            .height = layout.height,
            .ready_tick = ready_tick,
            .drawn = drawn,
            .nearest = nearest,
        });
    }
    async_sync->work_cv.notify_one();
    return true;
}

void RendererMetal::Impl::DrainAsyncPresent() {
    if (!async_present_thread.joinable()) {
        return;
    }
    std::unique_lock lock{async_sync->mutex};
    async_sync->free_cv.wait(lock, [this] {
        return async_sync->jobs.empty() && !async_sync->worker_busy &&
               std::ranges::none_of(async_sync->slot_busy, [](bool b) { return b; });
    });
}

void RendererMetal::Impl::StopAsyncPresent() {
    if (async_present_thread.joinable()) {
        {
            std::scoped_lock lock{async_sync->mutex};
            async_sync->stop = true;
            for (const PresentJob& pending : async_sync->jobs) {
                async_sync->slot_busy[pending.slot_index] = false;
            }
            async_sync->jobs.clear();
        }
        async_sync->work_cv.notify_all();
        async_present_thread.join();
    }
    std::unique_lock lock{async_sync->mutex};
    async_sync->free_cv.wait(lock, [this] {
        return std::ranges::none_of(async_sync->slot_busy, [](bool b) { return b; });
    });
}

void RendererMetal::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    SCOPE_EXIT {
        render_window.OnFrameDisplayed();
    };
    @autoreleasepool {
        if (LazyAppletCaptureEnabled()) {
            impl->last_framebuffers.assign(framebuffers.begin(), framebuffers.end());
        } else {
            // The last frame for the applet capture (LayerStackId::LastFrame), shown or not.
            impl->RenderAppletCapture(framebuffers);
        }
    }
    if (!render_window.IsShown()) {
        return;
    }

    @autoreleasepool {
        RenderScreenshot(framebuffers);
        const auto& layout = render_window.GetFramebufferLayout();
        if (!framebuffers.empty()) {
            const auto& guest = framebuffers.front();
            const bool cropped = !guest.crop_rect.IsEmpty();
            const u32 game_width =
                cropped ? static_cast<u32>(guest.crop_rect.GetWidth()) : guest.width;
            const u32 game_height =
                cropped ? static_cast<u32>(guest.crop_rect.GetHeight()) : guest.height;
            VideoCore::Probe::StoreFrameExtents(game_width, game_height, layout.width,
                                                layout.height);
        }
        const bool want_frame_gen =
#ifdef HAS_LSFG
            FrameGen::Wanted();
#else
            false;
#endif
        if (!want_frame_gen && AsyncPresentEnabled() &&
            impl->SubmitAsyncPresent(framebuffers, layout)) {
            // Composite rendered into the handoff ring; MetalPresent handles nextDrawable/present.
        } else {
            impl->DrainAsyncPresent();
            // nextDrawable blocks the compositing thread (the GPU back) while every drawable is
            // queued for display (display sync on, game faster than the display): timed for the
            // nx2:vi2 critical path line and the report below.
            const auto drawable_start = std::chrono::steady_clock::now();
            const id<CAMetalDrawable> drawable =
                impl->swapchain.NextDrawable(layout.width, layout.height);
            {
                const u64 waited = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - drawable_start)
                        .count());
                VideoCommon::SplitStats::AddPresentBlock(waited);
                impl->NoteDrawableWait(waited);
            }
            if (drawable != nil) {
#ifdef HAS_LSFG
                if (want_frame_gen) {
                    impl->CompositeGenerated(framebuffers, drawable, layout);
                } else
#endif
                {
                    const id<MTLTexture> dst_tex = drawable.texture;
                    const u32 dst_w =
                        dst_tex != nil ? static_cast<u32>(dst_tex.width) : layout.width;
                    const u32 dst_h =
                        dst_tex != nil ? static_cast<u32>(dst_tex.height) : layout.height;
                    Layout::FramebufferLayout draw_layout = layout;
                    draw_layout.width = dst_w;
                    draw_layout.height = dst_h;
                    if ((dst_w != layout.width || dst_h != layout.height) &&
                        layout.width > 0 && layout.height > 0) {
                        const f64 sx = static_cast<f64>(dst_w) / static_cast<f64>(layout.width);
                        const f64 sy = static_cast<f64>(dst_h) / static_cast<f64>(layout.height);
                        draw_layout.screen.left =
                            static_cast<u32>(std::lround(layout.screen.left * sx));
                        draw_layout.screen.right =
                            static_cast<u32>(std::lround(layout.screen.right * sx));
                        draw_layout.screen.top =
                            static_cast<u32>(std::lround(layout.screen.top * sy));
                        draw_layout.screen.bottom =
                            static_cast<u32>(std::lround(layout.screen.bottom * sy));
                    }
                    const Attachment target{.texture = (__bridge void*)dst_tex};
                    const bool drawn =
                        impl->DrawLayers(framebuffers, target, DrawableFormat(), draw_layout, true);
                    if (!drawn) {
                        // Dark blue: the renderer runs, the guest has no framebuffer to show.
                        impl->ClearAttachment(target, dst_w, dst_h, {0.06, 0.09, 0.22, 1.0});
                    }
                    const id<MTLCommandBuffer> command_buffer =
                        (__bridge id<MTLCommandBuffer>)impl->scheduler.CommandBuffer(true);
                    {
                        std::scoped_lock lock{impl->swapchain_pacing_mutex};
                        impl->swapchain.Present(command_buffer, drawable, drawn);
                    }
                    impl->scheduler.NotePresent(drawn);
                    impl->scheduler.Flush(CommitReason::Present);
                    impl->presented.fetch_add(1, std::memory_order_relaxed);
                }
            } else {
                impl->skipped.fetch_add(1, std::memory_order_relaxed);
                impl->scheduler.Flush(CommitReason::Present);
            }
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
        const u64 presented_count = impl->presented.exchange(0, std::memory_order_relaxed);
        const u64 skipped_count = impl->skipped.exchange(0, std::memory_order_relaxed);
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
                 "waited for, {} buffers freed); textures {} MiB (+{} MiB pooled), buffers {} MiB",
                 presented_count, impl->from_cache, impl->from_memory, skipped_count, work.draws,
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
                 impl->rasterizer.TextureRuntime().PooledImageBytes() >> 20,
                 impl->rasterizer.BufferRuntime().GetDeviceMemoryUsage() >> 20);
        LOG_INFO(Render_Metal,
                 "Metal present ({}): waited {:.1f} ms for drawables in {} frames "
                 "(longest {:.2f} ms)",
                 impl->async_present_thread.joinable() ? "async worker" : "inline",
                 static_cast<double>(impl->drawable_wait_ns.exchange(0, std::memory_order_relaxed)) /
                     1e6,
                 presented_count + skipped_count,
                 static_cast<double>(
                     impl->drawable_wait_max_ns.exchange(0, std::memory_order_relaxed)) /
                     1e6);
        if (const std::string coverage = Coverage::TakeLine(); !coverage.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal coverage: {}", coverage);
        }
#ifdef HAS_LSFG
        if (const u64 generated = std::exchange(impl->generated_since_report, u64{0});
            generated != 0) {
            LOG_INFO(Render_Metal, "Perf Metal frame generation: {} generated frames in {} s ({})",
                     generated, 10, impl->frame_gen.LastMaker());
        }
#endif
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
        if (const std::string builds = impl->rasterizer.Pipelines().TakeBuildLine();
            !builds.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal pipelines: {}", builds);
        }
        if (const std::string decisions = impl->rasterizer.TakeGpuDecisionLine();
            !decisions.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal GPU decisions: {}", decisions);
        }
        if (const std::string reduction = impl->rasterizer.TakeReductionLine();
            !reduction.empty()) {
            LOG_INFO(Render_Metal, "Perf Metal sampler reduction variants: {}", reduction);
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
        const f64 frames = static_cast<f64>((std::max)(presented_count, u64{1}));
        LOG_INFO(Render_Metal,
                 "Metal perf: {:.1f} fps, GPU {:.2f} ms/frame, {:.1f} command buffers, {:.1f} "
                 "encoders and {:.1f} render passes per frame, {:.0f} draws per frame",
                 static_cast<f64>(presented_count) / seconds,
                 static_cast<f64>(impl->scheduler.TakeGpuNanoseconds()) / 1e6 / frames,
                 static_cast<f64>(command_buffers) / frames, static_cast<f64>(encoders) / frames,
                 static_cast<f64>(passes) / frames, static_cast<f64>(work.draws_encoded) / frames);
        LOG_INFO(Render_Metal, "Metal commits: {}", impl->scheduler.TakeCommitLine(frames));
        LOG_INFO(Render_Metal, "Metal pass ends: {}", impl->scheduler.TakePassEndLine(frames));
        if (const u64 present_passes = impl->present.TakePasses(); present_passes != 0) {
            LOG_INFO(Render_Metal,
                     "Metal present filters: {:.1f} passes per frame (scaling filter {}, "
                     "anti-aliasing {})",
                     static_cast<f64>(present_passes) / frames,
                     Settings::CanonicalizeEnum(Settings::values.scaling_filter.GetValue()),
                     Settings::CanonicalizeEnum(Settings::values.anti_aliasing.GetValue()));
        }
        std::string pacing;
        {
            std::scoped_lock lock{impl->swapchain_pacing_mutex};
            pacing = impl->swapchain.TakePacingLine();
        }
        if (!pacing.empty()) {
            LOG_INFO(Render_Metal, "Metal pacing: {}", pacing);
        }
        if (const auto memo = impl->rasterizer.TakeBindMemoCounts(); memo.hits + memo.misses != 0) {
            LOG_INFO(Render_Metal,
                     "Metal binding memo: {} bindings from the memo, {} resolved ({} draws or "
                     "dispatches resolved all again)",
                     memo.hits, memo.misses, memo.full_resolves);
        }
        if (const u64 draw_flushes = impl->rasterizer.TakeDrawFlushes(); draw_flushes != 0) {
            LOG_INFO(Render_Metal, "Metal: {} command buffers committed every so many draws "
                                   "(EDEN_NX2_FLUSH_DRAWS)",
                     draw_flushes);
        }
        if (const auto [deferred, idle_commits] = impl->scheduler.TakeDeferredFences();
            deferred != 0) {
            LOG_INFO(Render_Metal,
                     "Metal fences: {} left pending to go with later work "
                     "(EDEN_NX2_FENCE_FLUSH_US), {} commits of such work by the idle GPU thread",
                     deferred, idle_commits);
        }
        impl->report_start = now;
        impl->from_cache = 0;
        impl->from_memory = 0;
    }
}

bool RendererMetal::Impl::ReadTexture(TextureHandle texture, u32 width, u32 height,
                                      std::span<u8> out) {
    const u64 size = u64{width} * height * 4;
    if (texture == nullptr || out.size() < size) {
        return false;
    }
    if (readback_size < size) {
        readback = CreateBuffer(device, size, true, &readback_contents);
        readback_size = readback_contents != nullptr ? size : 0;
    }
    if (readback_contents == nullptr) {
        return false;
    }
    scheduler.CopyTextureToBuffer(texture, 0, 0, {}, {width, height, 1}, readback.Get(), 0,
                                  width * 4, 0);
    // As Vulkan: the copy is waited for before the capture is handed on.
    scheduler.Finish();
    std::memcpy(out.data(), readback_contents, size);
    return true;
}

void RendererMetal::Impl::RenderAppletCapture(
    std::span<const Tegra::FramebufferConfig> framebuffers) {
    const auto layers = Tegra::FilterLayerStack(
        framebuffers, Service::Nvnflinger::LayerStackId::LastFrame, applet_layers);
    if (layers.empty()) {
        return;
    }
    using namespace VideoCore::Capture;
    if (!applet_capture) {
        applet_capture = CreateTexture(device, {.kind = TextureKind::e2D,
                                                .format = Fmt::RGBA8Unorm,
                                                .width = LinearWidth,
                                                .height = LinearHeight,
                                                .render_target = true,
                                                .pixel_format_view = false});
    }
    if (!applet_capture) {
        return;
    }
    const Attachment target{.texture = applet_capture.Get()};
    if (!DrawLayers(layers, target, Fmt::RGBA8Unorm, VideoCore::Capture::Layout, false)) {
        ClearAttachment(target, LinearWidth, LinearHeight, {0.0, 0.0, 0.0, 1.0});
    }
}

std::vector<u8> RendererMetal::GetAppletCaptureBuffer() {
    using namespace VideoCore::Capture;
    std::vector<u8> out(TiledSize);
    if (LazyAppletCaptureEnabled() && !impl->last_framebuffers.empty()) {
        @autoreleasepool {
            impl->RenderAppletCapture(impl->last_framebuffers);
        }
    }
    if (!impl->applet_capture) {
        return out;
    }
    // Vulkan's GetAppletCaptureBuffer: the capture read back and swizzled.
    std::vector<u8> linear(u64{LinearWidth} * LinearHeight * BytesPerPixel);
    if (impl->ReadTexture(impl->applet_capture.Get(), LinearWidth, LinearHeight, linear)) {
        Tegra::Texture::SwizzleTexture(out, linear, BytesPerPixel, LinearWidth, LinearHeight,
                                       LinearDepth, BlockHeight, BlockDepth);
    }
    return out;
}

void RendererMetal::RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (!renderer_settings.screenshot_requested) {
        return;
    }
    const auto layers = Tegra::FilterLayerStack(
        framebuffers, renderer_settings.screenshot_layer_stack, impl->screenshot_layers);
    const auto& layout = renderer_settings.screenshot_framebuffer_layout;
    if (layout.width != 0 && layout.height != 0) {
        if (!impl->screenshot_target || impl->screenshot_width != layout.width ||
            impl->screenshot_height != layout.height) {
            impl->screenshot_target = CreateTexture(impl->device, {.kind = TextureKind::e2D,
                                                                   .format = Fmt::BGRA8Unorm,
                                                                   .width = layout.width,
                                                                   .height = layout.height,
                                                                   .render_target = true,
                                                                   .pixel_format_view = false});
            impl->screenshot_width = layout.width;
            impl->screenshot_height = layout.height;
        }
        if (impl->screenshot_target) {
            const Attachment target{.texture = impl->screenshot_target.Get()};
            if (!impl->DrawLayers(layers, target, Fmt::BGRA8Unorm, layout, true)) {
                impl->ClearAttachment(target, layout.width, layout.height, {0.0, 0.0, 0.0, 1.0});
            }
            // B8G8R8A8 rows, as Vulkan's screenshot buffer.
            (void)impl->ReadTexture(
                impl->screenshot_target.Get(), layout.width, layout.height,
                std::span(static_cast<u8*>(renderer_settings.screenshot_bits),
                          u64{layout.width} * layout.height * 4));
        }
    }
    renderer_settings.screenshot_requested = false;
    if (renderer_settings.screenshot_complete_callback) {
        // Its argument is whether the picture is upside down (never, as on Vulkan).
        renderer_settings.screenshot_complete_callback(false);
    }
}

VideoCore::RasterizerInterface* RendererMetal::ReadRasterizer() {
    return &impl->rasterizer;
}

std::string RendererMetal::GetDeviceVendor() const {
    return impl->device.GetName();
}

} // namespace Metal
