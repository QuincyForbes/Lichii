// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <deque>
#include <unordered_map>
#include <utility>

#import <Metal/Metal.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {

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
)";

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
    }

    id<MTLCommandBuffer> Commands() {
        if (command_buffer == nil) {
            command_buffer = [queue commandBuffer];
            command_buffer.label = @"Lichii";
        }
        return command_buffer;
    }

    void EndEncoder() {
        if (blit != nil) {
            [blit endEncoding];
            blit = nil;
        }
    }

    id<MTLBlitCommandEncoder> BlitEncoder() {
        if (blit == nil) {
            blit = [Commands() blitCommandEncoder];
            ++encoders;
        }
        ++blit_commands;
        return blit;
    }

    id<MTLRenderCommandEncoder> RenderEncoder(MTLRenderPassDescriptor* pass) {
        EndEncoder();
        ++encoders;
        ++render_passes;
        return [Commands() renderCommandEncoderWithDescriptor:pass];
    }

    id<MTLRenderPipelineState> BlitPipeline(MTLPixelFormat format) {
        if (const auto it = blit_pipelines.find(format); it != blit_pipelines.end()) {
            return it->second;
        }
        id<MTLRenderPipelineState> pipeline = nil;
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        NSString* fragment = nil;
        switch (format_class) {
        case FormatClass::Float: fragment = @"blit_float"; break;
        case FormatClass::Uint: fragment = @"blit_uint"; break;
        case FormatClass::Sint: fragment = @"blit_sint"; break;
        default: break;
        }
        if (library != nil && fragment != nil) {
            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = @"Blit";
            desc.vertexFunction = [library newFunctionWithName:@"blit_vertex"];
            desc.fragmentFunction = [library newFunctionWithName:fragment];
            desc.colorAttachments[0].pixelFormat = format;
            NSError* error = nil;
            pipeline = [device.GetDevice() newRenderPipelineStateWithDescriptor:desc error:&error];
            if (pipeline == nil) {
                LOG_ERROR(Render_Metal, "Metal: blit pipeline for format {} failed: {}",
                          static_cast<u32>(format), error.localizedDescription.UTF8String);
            }
        }
        blit_pipelines.emplace(format, pipeline);
        return pipeline;
    }

    const Device& device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit = nil;
    id<MTLLibrary> library = nil;
    id<MTLSamplerState> nearest = nil;
    id<MTLSamplerState> linear = nil;
    std::unordered_map<MTLPixelFormat, id<MTLRenderPipelineState>> blit_pipelines;

    u64 current_tick{1};
    std::atomic<u64> completed_tick{0};
    std::deque<std::pair<u64, id<MTLCommandBuffer>>> in_flight;

    u64 encoders{};
    u64 command_buffers{};
    u64 blit_commands{};
    u64 render_passes{};
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

bool Scheduler::HasPendingWork() const noexcept {
    return impl->command_buffer != nil;
}

u64 Scheduler::Flush() {
    impl->EndEncoder();
    const id<MTLCommandBuffer> command_buffer = impl->Commands();
    const u64 tick = impl->current_tick++;
    std::atomic<u64>* const completed = &impl->completed_tick;
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        if (finished.status == MTLCommandBufferStatusError) {
            LOG_ERROR(Render_Metal, "Metal: command buffer {} failed: {}", tick,
                      finished.error.localizedDescription.UTF8String);
        }
        u64 previous = completed->load(std::memory_order_relaxed);
        while (previous < tick &&
               !completed->compare_exchange_weak(previous, tick, std::memory_order_release)) {
        }
    }];
    [command_buffer commit];
    ++impl->command_buffers;
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
    pass.renderTargetWidth = clear.width;
    pass.renderTargetHeight = clear.height;
    [impl->RenderEncoder(pass) endEncoding];
}

bool Scheduler::Blit(const BlitPass& blit) {
    const id<MTLTexture> src = AsTexture(blit.src);
    if (src == nil || blit.dst.texture == nullptr) {
        return false;
    }
    const MTLPixelFormat format = static_cast<MTLPixelFormat>(blit.dst_format);
    const id<MTLRenderPipelineState> pipeline = impl->BlitPipeline(format);
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
    ApplyAttachment(pass.colorAttachments[0], blit.dst);
    pass.colorAttachments[0].loadAction = blit.keep_dst ? MTLLoadActionLoad : MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    const id<MTLRenderCommandEncoder> encoder = impl->RenderEncoder(pass);
    [encoder setRenderPipelineState:pipeline];
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

void* Scheduler::CommandBuffer() {
    impl->EndEncoder();
    return (__bridge void*)impl->Commands();
}

std::array<u64, 4> Scheduler::TakeCounts() noexcept {
    return {std::exchange(impl->encoders, 0), std::exchange(impl->command_buffers, 0),
            std::exchange(impl->blit_commands, 0), std::exchange(impl->render_passes, 0)};
}

} // namespace Metal
