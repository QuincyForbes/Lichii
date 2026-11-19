// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_frame_gen_metalfx.h"
#include "video_core/renderer_metal/metal_scheduler.h"

// MTLFXFrameInterpolator came with the iOS 26 and macOS 26 SDKs.
#if __has_include(<MetalFX/MTLFXFrameInterpolator.h>)
#define LICHII_METALFX_FRAME_INTERPOLATOR 1
#endif

namespace Metal {

#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
namespace {

// The flow is at half the frame's size: sample it bilinearly at every pixel of the frame. The
// interpolator's motionVectorScale turns flow texels into pixels (Vulkan's UPSAMPLE_MOTION_SOURCE).
constexpr const char* UPSAMPLE_MOTION_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void upsample_motion(texture2d<float, access::sample> flow [[texture(0)]],
                            texture2d<float, access::write> motion [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= motion.get_width() || gid.y >= motion.get_height()) {
        return;
    }
    constexpr sampler linear(coord::normalized, address::clamp_to_edge, filter::linear);
    const float2 uv = (float2(gid) + 0.5) / float2(motion.get_width(), motion.get_height());
    motion.write(float4(flow.sample(linear, uv).xy, 0.0, 0.0), gid);
}
)";

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format, std::array<u32, 2> size,
                           MTLTextureUsage usage) {
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:size[0]
                                                          height:size[1]
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = usage;
    return [device newTextureWithDescriptor:descriptor];
}

bool HasUsage(id<MTLTexture> texture, MTLTextureUsage usage) {
    return (texture.usage & usage) == usage;
}

} // Anonymous namespace
#endif

struct FrameInterpolator::Impl {
    bool failed{};
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    bool Prepare(id<MTLDevice> device, id<MTLTexture> color, id<MTLTexture> destination,
                 std::array<u32, 2> size);

    std::array<u32, 2> extent{};
    MTLPixelFormat format = MTLPixelFormatInvalid;
    id interpolator = nil; ///< id<MTLFXFrameInterpolator>, iOS 26 and macOS 26 only
    id<MTLComputePipelineState> upsample = nil;
    id<MTLTexture> depth = nil;
    id<MTLTexture> motion = nil;
    id<MTLTexture> output = nil; ///< Where the destination lacks the usage MetalFX needs
    bool depth_cleared{};
#endif
};

#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
bool FrameInterpolator::Impl::Prepare(id<MTLDevice> device, id<MTLTexture> color,
                                      id<MTLTexture> destination, std::array<u32, 2> size) {
    if (@available(iOS 26.0, macOS 26.0, *)) {
        if (interpolator != nil && extent == size && format == color.pixelFormat) {
            return true;
        }
        interpolator = nil;
        extent = size;
        format = color.pixelFormat;
        if (upsample == nil) {
            NSError* error = nil;
            id<MTLLibrary> library =
                [device newLibraryWithSource:[NSString stringWithUTF8String:UPSAMPLE_MOTION_SOURCE]
                                     options:nil
                                       error:&error];
            id<MTLFunction> function = [library newFunctionWithName:@"upsample_motion"];
            if (function != nil) {
                upsample = [device newComputePipelineStateWithFunction:function error:&error];
            }
            if (upsample == nil) {
                LOG_ERROR(Render_Metal, "MetalFX frame interpolation: motion kernel failed: {}",
                          error != nil ? error.localizedDescription.UTF8String : "unknown");
                return false;
            }
        }
        MTLFXFrameInterpolatorDescriptor* descriptor =
            [[MTLFXFrameInterpolatorDescriptor alloc] init];
        descriptor.colorTextureFormat = format;
        descriptor.outputTextureFormat = format;
        descriptor.depthTextureFormat = MTLPixelFormatDepth32Float;
        descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
        descriptor.inputWidth = size[0];
        descriptor.inputHeight = size[1];
        descriptor.outputWidth = size[0];
        descriptor.outputHeight = size[1];
        id<MTLFXFrameInterpolator> created = nil;
        @try {
            created = [descriptor newFrameInterpolatorWithDevice:device];
        } @catch (NSException* exception) {
            LOG_ERROR(Render_Metal, "MetalFX frame interpolation: creating it raised: {}",
                      exception.reason.UTF8String != nullptr ? exception.reason.UTF8String
                                                             : "no reason given");
        }
        if (created == nil) {
            LOG_WARNING(Render_Metal, "MetalFX frame interpolation: no interpolator for {}x{}",
                        size[0], size[1]);
            return false;
        }
        if (!HasUsage(color, created.colorTextureUsage)) {
            LOG_WARNING(Render_Metal,
                        "MetalFX frame interpolation: the frame copies lack the texture usage");
            return false;
        }
        depth = MakeTexture(device, MTLPixelFormatDepth32Float, size,
                            created.depthTextureUsage | MTLTextureUsageRenderTarget);
        motion = MakeTexture(device, MTLPixelFormatRG16Float, size,
                             created.motionTextureUsage | MTLTextureUsageShaderWrite);
        const bool direct = HasUsage(destination, created.outputTextureUsage);
        output = direct ? nil : MakeTexture(device, format, size, created.outputTextureUsage);
        if (depth == nil || motion == nil || (!direct && output == nil)) {
            return false;
        }
        depth_cleared = false;
        interpolator = created;
        LOG_INFO(Render_Metal, "MetalFX frame interpolation: {}x{}{}", size[0], size[1],
                 output != nil ? " (through a copy)" : "");
        return true;
    }
    return false;
}
#endif

FrameInterpolator::FrameInterpolator() : impl{std::make_unique<Impl>()} {}

FrameInterpolator::~FrameInterpolator() = default;

bool FrameInterpolator::IsSupported([[maybe_unused]] const Device& device) {
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    if (@available(iOS 26.0, macOS 26.0, *)) {
        return [MTLFXFrameInterpolatorDescriptor supportsDevice:device.GetDevice()];
    }
#endif
    return false;
}

bool FrameInterpolator::Prepare([[maybe_unused]] const Device& device,
                                [[maybe_unused]] TextureHandle color,
                                [[maybe_unused]] TextureHandle destination,
                                [[maybe_unused]] std::array<u32, 2> size) {
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    Impl& state = *impl;
    if (state.failed) {
        return false;
    }
    if (@available(iOS 26.0, macOS 26.0, *)) {
        @autoreleasepool {
            id<MTLTexture> color_texture = AsTexture(color);
            id<MTLTexture> destination_texture = AsTexture(destination);
            if (color_texture == nil || destination_texture == nil ||
                !state.Prepare(device.GetDevice(), color_texture, destination_texture, size)) {
                state.failed = true;
                return false;
            }
            return true;
        }
    }
#endif
    return false;
}

bool FrameInterpolator::Generate([[maybe_unused]] const Device& device,
                                 [[maybe_unused]] Scheduler& scheduler,
                                 [[maybe_unused]] TextureHandle previous,
                                 [[maybe_unused]] TextureHandle current,
                                 [[maybe_unused]] std::array<u32, 2> size,
                                 [[maybe_unused]] TextureHandle motion_texture,
                                 [[maybe_unused]] std::array<u32, 2> motion_size,
                                 [[maybe_unused]] TextureHandle destination,
                                 [[maybe_unused]] f32 delta_time, [[maybe_unused]] bool reset) {
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    Impl& state = *impl;
    if (state.failed) {
        return false;
    }
    if (@available(iOS 26.0, macOS 26.0, *)) {
        @autoreleasepool {
            id<MTLTexture> previous_texture = AsTexture(previous);
            id<MTLTexture> current_texture = AsTexture(current);
            id<MTLTexture> flow_texture = AsTexture(motion_texture);
            id<MTLTexture> destination_texture = AsTexture(destination);
            if (previous_texture == nil || current_texture == nil || flow_texture == nil ||
                destination_texture == nil ||
                destination_texture.pixelFormat != current_texture.pixelFormat ||
                !state.Prepare(device.GetDevice(), current_texture, destination_texture, size)) {
                state.failed = true;
                LOG_WARNING(Render_Metal,
                            "MetalFX frame interpolation unavailable; using FSR 3 frame generation");
                return false;
            }
            id<MTLFXFrameInterpolator> interpolator = state.interpolator;
            id<MTLTexture> output = state.output != nil ? state.output : destination_texture;
            const id<MTLCommandBuffer> command_buffer =
                (__bridge id<MTLCommandBuffer>)scheduler.CommandBuffer();
            @try {
                if (!state.depth_cleared) {
                    // No depth from the game: one plane at the far end.
                    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
                    pass.depthAttachment.texture = state.depth;
                    pass.depthAttachment.loadAction = MTLLoadActionClear;
                    pass.depthAttachment.storeAction = MTLStoreActionStore;
                    pass.depthAttachment.clearDepth = 1.0;
                    [[command_buffer renderCommandEncoderWithDescriptor:pass] endEncoding];
                    state.depth_cleared = true;
                }
                id<MTLComputeCommandEncoder> compute = [command_buffer computeCommandEncoder];
                [compute setComputePipelineState:state.upsample];
                [compute setTexture:flow_texture atIndex:0];
                [compute setTexture:state.motion atIndex:1];
                const MTLSize group = MTLSizeMake(16, 16, 1);
                const MTLSize groups = MTLSizeMake((size[0] + 15) / 16, (size[1] + 15) / 16, 1);
                [compute dispatchThreadgroups:groups threadsPerThreadgroup:group];
                [compute endEncoding];

                interpolator.colorTexture = current_texture;
                interpolator.prevColorTexture = previous_texture;
                interpolator.depthTexture = state.depth;
                interpolator.motionTexture = state.motion;
                interpolator.outputTexture = output;
                interpolator.motionVectorScaleX =
                    static_cast<float>(size[0]) / static_cast<float>(motion_size[0]);
                interpolator.motionVectorScaleY =
                    static_cast<float>(size[1]) / static_cast<float>(motion_size[1]);
                interpolator.deltaTime = delta_time;
                // A flat image with no camera: values that keep the depth plane in range.
                interpolator.nearPlane = 0.1f;
                interpolator.farPlane = 1000.0f;
                interpolator.fieldOfView = 60.0f;
                interpolator.aspectRatio = static_cast<float>(size[0]) / static_cast<float>(size[1]);
                interpolator.jitterOffsetX = 0.0f;
                interpolator.jitterOffsetY = 0.0f;
                interpolator.shouldResetHistory = reset;
                [interpolator encodeToCommandBuffer:command_buffer];

                if (state.output != nil) {
                    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
                    [blit copyFromTexture:state.output toTexture:destination_texture];
                    [blit endEncoding];
                }
            } @catch (NSException* exception) {
                state.failed = true;
                LOG_WARNING(Render_Metal,
                            "MetalFX frame interpolation raised ({}); using FSR 3 frame generation",
                            exception.reason.UTF8String != nullptr ? exception.reason.UTF8String
                                                                   : "no reason given");
                return false;
            }
            return true;
        }
    }
#endif
    return false;
}

} // namespace Metal
