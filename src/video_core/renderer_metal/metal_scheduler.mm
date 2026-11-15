// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

#import <Metal/Metal.h>
#include <TargetConditionals.h>
#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {

/// Apple GPUs' largest render target side.
constexpr u32 MaxRenderTargetSize = 16384;

/// A stage's argument tables as Metal's validation layer names them on Apple's own GPUs (a Mac's
/// other GPUs take 128 textures).
constexpr u32 BufferSlots = 31;
constexpr u32 TextureSlots = 96;
constexpr u32 SamplerSlots = 16;

/// Metal checks a slot past a stage's tables in its validation layer only.
bool SlotFits(u32 slot, u32 slots, const char* kind) {
    if (slot < slots || !PipelineCheckEnabled()) {
        return true;
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Metal, "Metal: {} slot {} not bound, a stage has {}", kind, slot,
                    slots);
    }
    return false;
}

constexpr const char* BlitSource = R"(
#include <metal_stdlib>
using namespace metal;

struct BlitOut {
    float4 position [[position]];
    float2 uv;
};

// One triangle over the viewport; uv_rect is the source rectangle (normalized) at its corners.
vertex BlitOut blit_vertex(uint id [[vertex_id]], constant float4& uv_rect [[buffer(0)]]) {
    const float2 corner = float2(float((id << 1) & 2), float(id & 2));
    BlitOut out;
    out.position = float4(corner.x * 2.0 - 1.0, 1.0 - corner.y * 2.0, 0.0, 1.0);
    out.uv = mix(uv_rect.xy, uv_rect.zw, corner);
    return out;
}

fragment float4 blit_float(BlitOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                           sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv);
}

fragment uint4 blit_uint(BlitOut in [[stage_in]], texture2d<uint> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

fragment int4 blit_sint(BlitOut in [[stage_in]], texture2d<int> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    return src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
}

struct DepthOut {
    float depth [[depth(any)]];
};

fragment DepthOut blit_depth(BlitOut in [[stage_in]], depth2d<float> src [[texture(0)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    DepthOut out;
    out.depth = src.read(uint2(clamp(in.uv * size, float2(0.0), size - 1.0)));
    return out;
}

struct DepthStencilOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

// The stencil goes out as the reference value of a Replace stencil operation.
fragment DepthStencilOut blit_depth_stencil(BlitOut in [[stage_in]],
                                            depth2d<float> src [[texture(0)]],
                                            texture2d<uint> src_stencil [[texture(1)]]) {
    const float2 size = float2(src.get_width(), src.get_height());
    const uint2 texel = uint2(clamp(in.uv * size, float2(0.0), size - 1.0));
    DepthStencilOut out;
    out.depth = src.read(texel);
    out.stencil = src_stencil.read(texel).x;
    return out;
}
)";

/// The stencil-only view format of a depth-stencil format (invalid when it has none).
MTLPixelFormat StencilViewFormat(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatDepth32Float_Stencil8:
        return MTLPixelFormatX32_Stencil8;
#if !TARGET_OS_IPHONE
    case MTLPixelFormatDepth24Unorm_Stencil8:
        return MTLPixelFormatX24_Stencil8;
#endif
    default:
        return MTLPixelFormatInvalid;
    }
}

void ApplyAttachment(MTLRenderPassAttachmentDescriptor* descriptor, const Attachment& attachment) {
    descriptor.texture = AsTexture(attachment.texture);
    descriptor.level = attachment.level;
    descriptor.slice = attachment.slice;
    descriptor.depthPlane = attachment.depth_plane;
}

} // Anonymous namespace

struct Scheduler::Impl {
    explicit Impl(const Device& device_) : device{device_}, queue{device_.GetQueue()} {
        NSError* error = nil;
        library = [device.GetDevice() newLibraryWithSource:@(BlitSource) options:nil error:&error];
        if (library == nil) {
            LOG_ERROR(Render_Metal, "Metal: blit shaders failed to build: {}",
                      error.localizedDescription.UTF8String);
        }
        MTLSamplerDescriptor* sampler_desc = [MTLSamplerDescriptor new];
        sampler_desc.minFilter = MTLSamplerMinMagFilterNearest;
        sampler_desc.magFilter = MTLSamplerMinMagFilterNearest;
        nearest = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        sampler_desc.minFilter = MTLSamplerMinMagFilterLinear;
        sampler_desc.magFilter = MTLSamplerMinMagFilterLinear;
        linear = [device.GetDevice() newSamplerStateWithDescriptor:sampler_desc];
        visibility = [device.GetDevice() newBufferWithLength:VisibilitySlots * sizeof(u64)
                                                     options:MTLResourceStorageModeShared];
        std::memset(visibility.contents, 0, VisibilitySlots * sizeof(u64));
    }

    id<MTLCommandBuffer> Commands() {
        if (command_buffer == nil) {
            // EDEN_NX2_METAL_ENCODER_STATUS=0: no per-encoder status in command buffer errors.
            static const bool encoder_status = [] {
                const char* const value = std::getenv("EDEN_NX2_METAL_ENCODER_STATUS");
                return value == nullptr || value[0] != '0';
            }();
            if (encoder_status) {
                MTLCommandBufferDescriptor* descriptor = [MTLCommandBufferDescriptor new];
                descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
                command_buffer = [queue commandBufferWithDescriptor:descriptor];
            } else {
                command_buffer = [queue commandBuffer];
            }
            command_buffer.label = [NSString stringWithFormat:@"Lichii %llu", current_tick];
        }
        return command_buffer;
    }

    /// The progress trace: the first of each kind of work at Info, so a log that stops shows how
    /// far the renderer got.
    void CountDraw() {
        if (draws++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first draw encoded (command buffer {}, render pass {})",
                     current_tick, render_pass_serial);
        }
    }

    void CountDispatch() {
        if (dispatches++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first dispatch encoded (command buffer {})",
                     current_tick);
        }
    }

    void EndEncoder() {
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
        if (render != nil) {
            [render endEncoding];
            render = nil;
        }
        if (compute != nil) {
            [compute endEncoding];
            compute = nil;
        }
    }

    id<MTLBlitCommandEncoder> BlitEncoder() {
        if (blit == nil) {
            EndEncoder();
            blit = [Commands() blitCommandEncoder];
            blit.label = @"Blit";
            ++encoders;
        }
        ++blit_commands;
        return blit;
    }

    id<MTLRenderCommandEncoder> RenderEncoder(MTLRenderPassDescriptor* pass) {
        EndEncoder();
        ++encoders;
        ++render_passes;
        const id<MTLRenderCommandEncoder> encoder =
            [Commands() renderCommandEncoderWithDescriptor:pass];
        if (encoder == nil) {
            // Metal refused the pass (draws into it are dropped): name it once.
            if (refused_passes++ < 4) {
                LOG_ERROR(Render_Metal, "Metal: render pass refused ({}x{}, colour 0 {}, depth {})",
                          pass.renderTargetWidth, pass.renderTargetHeight,
                          pass.colorAttachments[0].texture != nil
                              ? static_cast<u32>(pass.colorAttachments[0].texture.pixelFormat)
                              : 0U,
                          pass.depthAttachment.texture != nil
                              ? static_cast<u32>(pass.depthAttachment.texture.pixelFormat)
                              : 0U);
            }
            return nil;
        }
        encoder.label = [NSString stringWithFormat:@"Pass %llu", render_pass_serial + 1];
        return encoder;
    }

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format, BlitBlend blend) {
        const u64 key = (static_cast<u64>(format) << 8) | static_cast<u64>(blend);
        if (const auto it = blit_pipelines.find(key); it != blit_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float: fragment = @"blit_float"; break;
        case FormatClass::Uint: fragment = @"blit_uint"; break;
        case FormatClass::Sint: fragment = @"blit_sint"; break;
        case FormatClass::Depth: fragment = @"blit_depth"; break;
        case FormatClass::DepthStencil:
            fragment = StencilViewFormat(format) != MTLPixelFormatInvalid ? @"blit_depth_stencil"
                                                                          : @"blit_depth";
            break;
        default: break;
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Blit";
            desc.vertexFunction = [library newFunctionWithName:@"blit_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            if (format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil) {
                desc.depthAttachmentPixelFormat = format;
                if (format_class == FormatClass::DepthStencil) {
                    desc.stencilAttachmentPixelFormat = format;
                }
            } else {
                desc.colorAttachments[0].pixelFormat = format;
            }
            if (blend != BlitBlend::None && format_class == FormatClass::Float) {
                MTLRenderPipelineColorAttachmentDescriptor* color = desc.colorAttachments[0];
                color.blendingEnabled = YES;
                color.sourceRGBBlendFactor = blend == BlitBlend::Premultiplied
                                                 ? MTLBlendFactorOne
                                                 : MTLBlendFactorSourceAlpha;
                color.sourceAlphaBlendFactor = color.sourceRGBBlendFactor;
                color.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
                color.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            }
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: blit pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        blit_pipelines.emplace(key, pipeline);
        return pipeline;
    }

    /// The depth blits' state: every fragment writes its depth, and the stencil it outputs.
    id<MTLDepthStencilState> DepthWriteAlways() {
        if (depth_write_always == nil) {
            MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
            desc.depthCompareFunction = MTLCompareFunctionAlways;
            desc.depthWriteEnabled = YES;
            MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
            stencil.stencilCompareFunction = MTLCompareFunctionAlways;
            stencil.depthStencilPassOperation = MTLStencilOperationReplace;
            stencil.readMask = 0xFF;
            stencil.writeMask = 0xFF;
            desc.frontFaceStencil = stencil;
            desc.backFaceStencil = stencil;
            depth_write_always = [device.GetDevice() newDepthStencilStateWithDescriptor:desc];
        }
        return depth_write_always;
    }

    const Device& device;
    id<MTLCommandQueue> queue;
    id<MTLDepthStencilState> depth_write_always = nil;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    /// The render encoder draws go to, open until other work needs the command buffer.
    id<MTLRenderCommandEncoder> render = nil;
    RenderTargetSet render_targets{};
    id<MTLComputeCommandEncoder> compute = nil;
    id<MTLBuffer> visibility = nil;
    u64 render_pass_serial{};
    /// The open render encoder's counting slot (-1: not counting).
    s64 visibility_slot{-1};
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    /// By format << 8 | BlitBlend.
    std::unordered_map<u64, id<MTLRenderPipelineState>> blit_pipelines;

    u64 current_tick{1};
    std::atomic<u64> submitted_tick{0};
    std::atomic<u64> completed_tick{0};
    /// Wakes WaitSubmitted when completed_tick moves.
    std::mutex completed_mutex;
    std::condition_variable completed_cv;
    /// GPU time of the command buffers completed since TakeGpuNanoseconds.
    std::atomic<u64> gpu_ns{0};
    std::deque<std::pair<u64, id<MTLCommandBuffer>>> in_flight;

    u64 encoders{};
    u64 command_buffers{};
    u64 blit_commands{};
    u64 render_passes{};
    /// Since boot (the progress trace).
    u64 draws{};
    u64 dispatches{};
    u64 committed{};
    /// Presents showing a guest layer, and those that had none to show.
    u64 presents{};
    u64 blank_presents{};
    u32 refused_passes{};
    std::chrono::steady_clock::time_point heartbeat = std::chrono::steady_clock::now();
    std::atomic<bool> completed_logged{false};
    std::atomic<u32> errors_logged{0};
};

Scheduler::Scheduler(const Device& device) : impl{std::make_unique<Impl>(device)} {}

Scheduler::~Scheduler() {
    Finish();
}

u64 Scheduler::CurrentTick() const noexcept {
    return impl->current_tick;
}

bool Scheduler::IsFree(u64 tick) const noexcept {
    return impl->completed_tick.load(std::memory_order_acquire) >= tick;
}

void Scheduler::Wait(u64 tick) {
    if (tick >= impl->current_tick) {
        Flush();
    }
    while (!impl->in_flight.empty() && impl->in_flight.front().first <= tick) {
        [impl->in_flight.front().second waitUntilCompleted];
        impl->in_flight.pop_front();
    }
}

bool Scheduler::IsSubmitted(u64 tick) const noexcept {
    return impl->submitted_tick.load(std::memory_order_acquire) >= tick;
}

void Scheduler::WaitSubmitted(u64 tick) {
    if (IsFree(tick)) {
        return;
    }
    std::unique_lock lock{impl->completed_mutex};
    impl->completed_cv.wait(lock, [&] { return IsFree(tick); });
}

bool Scheduler::HasPendingWork() const noexcept {
    return impl->command_buffer != nil;
}

u64 Scheduler::Flush() {
    impl->EndEncoder();
    const id<MTLCommandBuffer> command_buffer = impl->Commands();
    const u64 tick = impl->current_tick++;
    std::atomic<u64>* const completed = &impl->completed_tick;
    std::atomic<u64>* const gpu_ns = &impl->gpu_ns;
    std::atomic<bool>* const completed_logged = &impl->completed_logged;
    std::atomic<u32>* const errors_logged = &impl->errors_logged;
    std::mutex* const completed_mutex = &impl->completed_mutex;
    std::condition_variable* const completed_cv = &impl->completed_cv;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        const CFTimeInterval busy = finished.GPUEndTime - finished.GPUStartTime;
        if (busy > 0.0) {
            gpu_ns->fetch_add(static_cast<u64>(busy * 1e9), std::memory_order_relaxed);
        }
        if (!completed_logged->exchange(true)) {
            LOG_INFO(Render_Metal, "Metal: first command buffer completed (tick {}, status {}, "
                                   "GPU {:.2f} ms)",
                     tick, static_cast<int>(finished.status), busy * 1e3);
        }
        if (finished.status == MTLCommandBufferStatusError &&
            errors_logged->fetch_add(1) < 16) {
            NSError* const error = finished.error;
            std::string encoders;
            NSArray* const infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
            for (id<MTLCommandBufferEncoderInfo> info in infos) {
                // MTLCommandEncoderErrorState: 0 unknown, 1 completed, 2 affected, 3 pending,
                // 4 faulted.
                if (info.errorState != MTLCommandEncoderErrorStateCompleted) {
                    encoders += fmt::format("{}{} (state {})", encoders.empty() ? "" : ", ",
                                            info.label.UTF8String != nullptr ? info.label.UTF8String
                                                                             : "?",
                                            static_cast<int>(info.errorState));
                }
            }
            LOG_ERROR(Render_Metal, "Metal: command buffer {} failed: code {}: {}; encoders: {}",
                      tick, static_cast<long>(error.code),
                      error.localizedDescription.UTF8String != nullptr
                          ? error.localizedDescription.UTF8String
                          : "no description",
                      encoders.empty() ? "none reported" : encoders);
        }
        u64 previous = completed->load(std::memory_order_relaxed);
        while (previous < tick &&
               !completed->compare_exchange_weak(previous, tick, std::memory_order_release)) {
        }
        {
            std::scoped_lock lock{*completed_mutex};
        }
        completed_cv->notify_all();
    }];
    [command_buffer commit];
    impl->submitted_tick.store(tick, std::memory_order_release);
    ++impl->command_buffers;
    if (impl->committed++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first command buffer committed (tick {})", tick);
    }
    // Until the first guest frame, a line every 5 s: where a boot that never shows one stands.
    if (impl->presents == 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - impl->heartbeat >= std::chrono::seconds{5}) {
            impl->heartbeat = now;
            LOG_INFO(Render_Metal,
                     "Metal: no guest frame presented yet; {} command buffers committed, {} "
                     "completed, {} draws, {} dispatches, {} presents without a guest picture",
                     impl->committed, impl->completed_tick.load(std::memory_order_relaxed),
                     impl->draws, impl->dispatches, impl->blank_presents);
        }
    }
    impl->command_buffer = nil;
    impl->in_flight.emplace_back(tick, command_buffer);
    while (!impl->in_flight.empty() && IsFree(impl->in_flight.front().first)) {
        impl->in_flight.pop_front();
    }
    return tick;
}

void Scheduler::Finish() {
    Wait(Flush());
}

void Scheduler::CopyBuffer(BufferHandle src, u64 src_offset, BufferHandle dst, u64 dst_offset,
                           u64 size) {
    if (size == 0 || src == nullptr || dst == nullptr) {
        return;
    }
    [impl->BlitEncoder() copyFromBuffer:AsBuffer(src)
                           sourceOffset:src_offset
                               toBuffer:AsBuffer(dst)
                      destinationOffset:dst_offset
                                   size:size];
}

void Scheduler::FillBuffer(BufferHandle dst, u64 offset, u64 size, u8 value) {
    if (size == 0 || dst == nullptr) {
        return;
    }
    [impl->BlitEncoder() fillBuffer:AsBuffer(dst) range:NSMakeRange(offset, size) value:value];
}

void Scheduler::CopyBufferToTexture(BufferHandle src, u64 offset, u32 bytes_per_row,
                                    u32 bytes_per_image, TextureHandle dst, u32 level, u32 slice,
                                    Offset3D origin, Extent3D extent) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    [impl->BlitEncoder() copyFromBuffer:AsBuffer(src)
                           sourceOffset:offset
                      sourceBytesPerRow:bytes_per_row
                    sourceBytesPerImage:bytes_per_image
                             sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                              toTexture:AsTexture(dst)
                       destinationSlice:slice
                       destinationLevel:level
                      destinationOrigin:MTLOriginMake(origin.x, origin.y, origin.z)];
}

void Scheduler::CopyTextureToBuffer(TextureHandle src, u32 level, u32 slice, Offset3D origin,
                                    Extent3D extent, BufferHandle dst, u64 offset,
                                    u32 bytes_per_row, u32 bytes_per_image) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:slice
                             sourceLevel:level
                            sourceOrigin:MTLOriginMake(origin.x, origin.y, origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                                toBuffer:AsBuffer(dst)
                       destinationOffset:offset
                  destinationBytesPerRow:bytes_per_row
                destinationBytesPerImage:bytes_per_image];
}

void Scheduler::CopyTexture(TextureHandle src, u32 src_level, u32 src_slice, Offset3D src_origin,
                            Extent3D extent, TextureHandle dst, u32 dst_level, u32 dst_slice,
                            Offset3D dst_origin) {
    if (src == nullptr || dst == nullptr) {
        return;
    }
    [impl->BlitEncoder() copyFromTexture:AsTexture(src)
                             sourceSlice:src_slice
                             sourceLevel:src_level
                            sourceOrigin:MTLOriginMake(src_origin.x, src_origin.y, src_origin.z)
                              sourceSize:MTLSizeMake(extent.width, extent.height, extent.depth)
                               toTexture:AsTexture(dst)
                        destinationSlice:dst_slice
                        destinationLevel:dst_level
                       destinationOrigin:MTLOriginMake(dst_origin.x, dst_origin.y, dst_origin.z)];
}

void Scheduler::BreakBlits() {
    impl->EndEncoder();
}

void Scheduler::Clear(const ClearPass& clear) {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    bool any = false;
    for (size_t index = 0; index < clear.color.size(); ++index) {
        if (clear.color[index].texture == nullptr || !clear.clear_color[index]) {
            continue;
        }
        MTLRenderPassColorAttachmentDescriptor* color = pass.colorAttachments[index];
        ApplyAttachment(color, clear.color[index]);
        const auto& value = clear.color_value[index];
        color.loadAction = MTLLoadActionClear;
        color.storeAction = MTLStoreActionStore;
        color.clearColor = MTLClearColorMake(value[0], value[1], value[2], value[3]);
        any = true;
    }
    if (clear.depth.texture != nullptr && clear.clear_depth) {
        ApplyAttachment(pass.depthAttachment, clear.depth);
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        pass.depthAttachment.clearDepth = clear.depth_value;
        any = true;
    }
    if (clear.stencil.texture != nullptr && clear.clear_stencil) {
        ApplyAttachment(pass.stencilAttachment, clear.stencil);
        pass.stencilAttachment.loadAction = MTLLoadActionClear;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
        pass.stencilAttachment.clearStencil = clear.stencil_value;
        any = true;
    }
    if (!any) {
        return;
    }
    pass.renderTargetWidth = std::clamp(clear.width, 1U, MaxRenderTargetSize);
    pass.renderTargetHeight = std::clamp(clear.height, 1U, MaxRenderTargetSize);
    [impl->RenderEncoder(pass) endEncoding];
}

bool Scheduler::Blit(const BlitPass& blit) {
    const id<MTLTexture> src = AsTexture(blit.src);
    if (src == nil || blit.dst.texture == nullptr) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(blit.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->BlitPipeline(format, blit.blend);
    if (pipeline == nil) {
        return false;
    }
    const auto [dx0, dy0, dx1, dy1] = blit.dst_rect;
    const auto [sx0, sy0, sx1, sy1] = blit.src_rect;
    const f32 width = static_cast<f32>(src.width);
    const f32 height = static_cast<f32>(src.height);
    // A mirrored destination becomes a mirrored source rectangle over a forward viewport.
    std::array<f32, 4> uv{sx0 / width, sy0 / height, sx1 / width, sy1 / height};
    if (dx1 < dx0) {
        std::swap(uv[0], uv[2]);
    }
    if (dy1 < dy0) {
        std::swap(uv[1], uv[3]);
    }
    const s32 x = (std::min)(dx0, dx1);
    const s32 y = (std::min)(dy0, dy1);
    const s32 w = std::abs(dx1 - dx0);
    const s32 h = std::abs(dy1 - dy0);
    if (w == 0 || h == 0) {
        return true;
    }

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    const FormatClass format_class = ClassOf(blit.dst_format);
    const bool depth = format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
    if (depth) {
        // Depth only: the fragment writes it; stencil keeps what it held or is cleared to 0.
        ApplyAttachment(pass.depthAttachment, blit.dst);
        pass.depthAttachment.loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        if (format_class == FormatClass::DepthStencil) {
            ApplyAttachment(pass.stencilAttachment, blit.dst);
            pass.stencilAttachment.loadAction =
                blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
            pass.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    } else {
        ApplyAttachment(pass.colorAttachments[0], blit.dst);
        pass.colorAttachments[0].loadAction =
            blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    // A depth-stencil source's stencil, read through a stencil view.
    id<MTLTexture> src_stencil = nil;
    if (format_class == FormatClass::DepthStencil &&
        StencilViewFormat(format) != MTLPixelFormatInvalid) {
        src_stencil = [src newTextureViewWithPixelFormat:StencilViewFormat(src.pixelFormat)];
        if (src_stencil == nil) {
            return false;
        }
    }
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    [encoder setRenderPipelineState:pipeline];
    if (depth) {
        [encoder setDepthStencilState:impl->DepthWriteAlways()];
        [encoder setFragmentTexture:src_stencil atIndex:1];
    }
    [encoder setViewport:MTLViewport{static_cast<f64>(x), static_cast<f64>(y),
                                     static_cast<f64>(w), static_cast<f64>(h), 0.0, 1.0}];
    const s32 limit_w = static_cast<s32>(blit.dst_width);
    const s32 limit_h = static_cast<s32>(blit.dst_height);
    const s32 sx = std::clamp(x, 0, limit_w);
    const s32 sy = std::clamp(y, 0, limit_h);
    const s32 sw = std::clamp(x + w, 0, limit_w) - sx;
    const s32 sh = std::clamp(y + h, 0, limit_h) - sy;
    if (sw <= 0 || sh <= 0) {
        [encoder endEncoding];
        return true;
    }
    [encoder setScissorRect:MTLScissorRect{static_cast<NSUInteger>(sx),
                                           static_cast<NSUInteger>(sy),
                                           static_cast<NSUInteger>(sw),
                                           static_cast<NSUInteger>(sh)}];
    [encoder setVertexBytes:uv.data() length:sizeof(uv) atIndex:0];
    [encoder setFragmentTexture:src atIndex:0];
    [encoder setFragmentSamplerState:blit.linear ? impl->linear : impl->nearest atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    return true;
}

void Scheduler::BeginRendering(const RenderTargetSet& targets) {
    if (impl->render != nil && impl->render_targets == targets) {
        return;
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    for (size_t index = 0; index < targets.color.size(); ++index) {
        if (targets.color[index].texture == nullptr) {
            continue;
        }
        ApplyAttachment(pass.colorAttachments[index], targets.color[index]);
        pass.colorAttachments[index].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[index].storeAction = MTLStoreActionStore;
    }
    if (targets.has_depth) {
        ApplyAttachment(pass.depthAttachment, targets.depth);
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    if (targets.has_stencil) {
        ApplyAttachment(pass.stencilAttachment, targets.depth);
        pass.stencilAttachment.loadAction = MTLLoadActionLoad;
        pass.stencilAttachment.storeAction = MTLStoreActionStore;
    }
    // Draws without attachments (occlusion queries) still need a size, within Metal's limit.
    pass.renderTargetWidth = std::clamp(targets.width, 1U, MaxRenderTargetSize);
    pass.renderTargetHeight = std::clamp(targets.height, 1U, MaxRenderTargetSize);
    const bool targetless = !targets.has_depth && !targets.has_stencil &&
                            std::ranges::all_of(targets.color, [](const Attachment& color) {
                                return color.texture == nullptr;
                            });
    if (targetless && PipelineCheckEnabled()) {
        // Without attachments nothing else tells Metal the pass has one sample per pixel, as
        // the pipelines drawn into it have (MoltenVK sets it for such a pass too).
        pass.defaultRasterSampleCount = 1;
    }
    pass.visibilityResultBuffer = impl->visibility;
    impl->render = impl->RenderEncoder(pass);
    impl->render_targets = targets;
    impl->visibility_slot = -1;
    ++impl->render_pass_serial;
}

void Scheduler::SetVisibility(std::optional<u32> slot) {
    const s64 wanted = slot ? static_cast<s64>(*slot) : -1;
    if (impl->render == nil || wanted == impl->visibility_slot) {
        return;
    }
    impl->visibility_slot = wanted;
    if (slot) {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeCounting
                                       offset:*slot * sizeof(u64)];
    } else {
        [impl->render setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
    }
}

u64* Scheduler::VisibilityResults() noexcept {
    return static_cast<u64*>(impl->visibility.contents);
}

u64 Scheduler::RenderPassSerial() const noexcept {
    return impl->render_pass_serial;
}

void Scheduler::SetDrawState(const DrawState& state) {
    const id<MTLRenderCommandEncoder> encoder = impl->render;
    if (encoder == nil) {
        return;
    }
    [encoder setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)state.pipeline];
    if (state.depth_stencil != nullptr) {
        [encoder setDepthStencilState:(__bridge id<MTLDepthStencilState>)state.depth_stencil];
    }
    [encoder setViewport:MTLViewport{state.viewport.x, state.viewport.y, state.viewport.width,
                                     state.viewport.height, state.viewport.znear,
                                     state.viewport.zfar}];
    // Metal rejects scissors past the attachments.
    const u32 target_width = std::clamp(impl->render_targets.width, 1U, MaxRenderTargetSize);
    const u32 target_height = std::clamp(impl->render_targets.height, 1U, MaxRenderTargetSize);
    const u32 x = (std::min)(state.scissor.x, target_width - 1);
    const u32 y = (std::min)(state.scissor.y, target_height - 1);
    const u32 width = (std::min)(state.scissor.width, target_width - x);
    const u32 height = (std::min)(state.scissor.height, target_height - y);
    [encoder setScissorRect:MTLScissorRect{x, y, (std::max)(width, 1U), (std::max)(height, 1U)}];
    [encoder setCullMode:static_cast<MTLCullMode>(state.cull_mode)];
    [encoder setFrontFacingWinding:state.front_ccw ? MTLWindingCounterClockwise
                                                   : MTLWindingClockwise];
    [encoder setDepthClipMode:state.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
    [encoder setDepthBias:state.depth_bias slopeScale:state.slope_scale clamp:state.bias_clamp];
    [encoder setStencilFrontReferenceValue:state.stencil_front_ref
                        backReferenceValue:state.stencil_back_ref];
    [encoder setBlendColorRed:state.blend_color[0]
                        green:state.blend_color[1]
                         blue:state.blend_color[2]
                        alpha:state.blend_color[3]];
}

void Scheduler::BindVertexBuffer(u32 slot, BufferHandle buffer, u32 offset) {
    if (impl->render != nil && SlotFits(slot, BufferSlots, "vertex buffer")) {
        [impl->render setVertexBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
    }
}

void Scheduler::BindBuffer(ShaderStage stage, u32 slot, BufferHandle buffer, u32 offset) {
    if (!SlotFits(slot, BufferSlots, "buffer")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        [impl->render setVertexBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [impl->render setFragmentBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
        break;
    case ShaderStage::Compute:
        [impl->compute setBuffer:AsBuffer(buffer) offset:offset atIndex:slot];
        break;
    }
}

void Scheduler::BindBytes(ShaderStage stage, u32 slot, const void* data, u32 size) {
    if (!SlotFits(slot, BufferSlots, "buffer")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        [impl->render setVertexBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [impl->render setFragmentBytes:data length:size atIndex:slot];
        break;
    case ShaderStage::Compute:
        [impl->compute setBytes:data length:size atIndex:slot];
        break;
    }
}

void Scheduler::BindTexture(ShaderStage stage, u32 slot, TextureHandle texture) {
    if (!SlotFits(slot, TextureSlots, "texture")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        [impl->render setVertexTexture:AsTexture(texture) atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [impl->render setFragmentTexture:AsTexture(texture) atIndex:slot];
        break;
    case ShaderStage::Compute:
        [impl->compute setTexture:AsTexture(texture) atIndex:slot];
        break;
    }
}

void Scheduler::BindSampler(ShaderStage stage, u32 slot, void* sampler) {
    if (!SlotFits(slot, SamplerSlots, "sampler")) {
        return;
    }
    switch (stage) {
    case ShaderStage::Vertex:
        [impl->render setVertexSamplerState:AsSampler(sampler) atIndex:slot];
        break;
    case ShaderStage::Fragment:
        [impl->render setFragmentSamplerState:AsSampler(sampler) atIndex:slot];
        break;
    case ShaderStage::Compute:
        [impl->compute setSamplerState:AsSampler(sampler) atIndex:slot];
        break;
    }
}

void Scheduler::DrawIndirect(u32 primitive, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                  indirectBuffer:AsBuffer(buffer)
            indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::DrawIndexedIndirect(u32 primitive, u32 index_size, BufferHandle index_buffer,
                                    u64 index_offset, BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                         indirectBuffer:AsBuffer(buffer)
                   indirectBufferOffset:offset];
    impl->CountDraw();
}

void Scheduler::BeginCompute(void* pipeline) {
    if (impl->compute == nil) {
        impl->EndEncoder();
        impl->compute = [impl->Commands() computeCommandEncoder];
        impl->compute.label = @"Compute";
        ++impl->encoders;
    }
    [impl->compute setComputePipelineState:(__bridge id<MTLComputePipelineState>)pipeline];
}

void Scheduler::Dispatch(std::array<u32, 3> groups, std::array<u32, 3> group_size) {
    if (impl->compute == nil || groups[0] == 0 || groups[1] == 0 || groups[2] == 0) {
        return;
    }
    [impl->compute dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
                  threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1], group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchIndirect(BufferHandle buffer, u64 offset, std::array<u32, 3> group_size) {
    if (impl->compute == nil || buffer == nullptr) {
        return;
    }
    [impl->compute dispatchThreadgroupsWithIndirectBuffer:AsBuffer(buffer)
                                     indirectBufferOffset:offset
                                    threadsPerThreadgroup:MTLSizeMake(group_size[0], group_size[1],
                                                                      group_size[2])];
    impl->CountDispatch();
}

void Scheduler::DispatchStageIn(u32 first, u32 count, u32 first_instance, u32 instances) {
    if (impl->compute == nil || count == 0 || instances == 0) {
        return;
    }
    [impl->compute setStageInRegion:MTLRegionMake2D(first, first_instance, count, instances)];
    [impl->compute dispatchThreads:MTLSizeMake(count, instances, 1)
             threadsPerThreadgroup:MTLSizeMake((std::min)(count, 64U), 1, 1)];
    impl->CountDispatch();
}

void Scheduler::SetTessellationFactorBuffer(BufferHandle buffer, u64 offset) {
    if (impl->render == nil || buffer == nullptr) {
        return;
    }
    [impl->render setTessellationFactorBuffer:AsBuffer(buffer) offset:offset instanceStride:0];
}

void Scheduler::DrawPatches(u32 control_points, u32 patch_count) {
    if (impl->render == nil || patch_count == 0) {
        return;
    }
    [impl->render drawPatches:control_points
                   patchStart:0
                   patchCount:patch_count
             patchIndexBuffer:nil
       patchIndexBufferOffset:0
                instanceCount:1
                 baseInstance:0];
    impl->CountDraw();
}

void Scheduler::Draw(u32 primitive, u32 first_vertex, u32 vertex_count, u32 instance_count,
                     u32 base_instance) {
    if (impl->render == nil || vertex_count == 0 || instance_count == 0) {
        return;
    }
    [impl->render drawPrimitives:static_cast<MTLPrimitiveType>(primitive)
                     vertexStart:first_vertex
                     vertexCount:vertex_count
                   instanceCount:instance_count
                    baseInstance:base_instance];
    impl->CountDraw();
}

void Scheduler::DrawIndexed(u32 primitive, u32 index_count, u32 index_size,
                            BufferHandle index_buffer, u64 index_offset, u32 instance_count,
                            s32 base_vertex, u32 base_instance) {
    if (impl->render == nil || index_count == 0 || instance_count == 0 || index_buffer == nullptr) {
        return;
    }
    [impl->render drawIndexedPrimitives:static_cast<MTLPrimitiveType>(primitive)
                             indexCount:index_count
                              indexType:index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                            indexBuffer:AsBuffer(index_buffer)
                      indexBufferOffset:index_offset
                          instanceCount:instance_count
                             baseVertex:base_vertex
                           baseInstance:base_instance];
    impl->CountDraw();
}

void* Scheduler::CommandBuffer() {
    impl->EndEncoder();
    return (__bridge void*)impl->Commands();
}

u64 Scheduler::TakeGpuNanoseconds() noexcept {
    return impl->gpu_ns.exchange(0, std::memory_order_relaxed);
}

void Scheduler::NotePresent(bool guest_picture) {
    if (!guest_picture) {
        if (impl->blank_presents++ == 0) {
            LOG_INFO(Render_Metal, "Metal: first drawable presented, without a guest picture");
        }
        return;
    }
    if (impl->presents++ == 0) {
        LOG_INFO(Render_Metal, "Metal: first guest frame presented ({} command buffers, {} draws, "
                               "{} dispatches before it)",
                 impl->committed, impl->draws, impl->dispatches);
    }
}

std::array<u64, 4> Scheduler::TakeCounts() noexcept {
    return {std::exchange(impl->encoders, 0), std::exchange(impl->command_buffers, 0),
            std::exchange(impl->blit_commands, 0), std::exchange(impl->render_passes, 0)};
}

} // namespace Metal
