// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <MetalFX/MetalFX.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/metalfx_context.h"
#include "video_core/renderer_vulkan/present/metalfx_frame_gen.h"
#include "video_core/vulkan_common/vulkan_device.h"

// MTLFXFrameInterpolator came with the iOS 26 and macOS 26 SDKs.
#if __has_include(<MetalFX/MTLFXFrameInterpolator.h>)
#define LICHII_METALFX_FRAME_INTERPOLATOR 1
#endif

namespace Vulkan {

#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
namespace {

// The flow is at half the frame's size: sample it bilinearly at every pixel of the frame. The
// interpolator's motionVectorScale turns flow texels into pixels.
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

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format, VkExtent2D extent,
                           MTLTextureUsage usage) {
    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:extent.width
                                                          height:extent.height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = usage;
    return [device newTextureWithDescriptor:descriptor];
}

} // Anonymous namespace
#endif

struct MetalFxFrameGen::Impl {
    Impl(const Device& device, Scheduler& scheduler) : context{device, scheduler} {}

    MetalFxContext context;
    bool failed{};
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    bool Prepare(id<MTLTexture> color, id<MTLTexture> destination, VkExtent2D extent);

    VkExtent2D extent{};
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
bool MetalFxFrameGen::Impl::Prepare(id<MTLTexture> color, id<MTLTexture> destination,
                                    VkExtent2D size) {
    if (@available(iOS 26.0, macOS 26.0, *)) {
        if (interpolator != nil && extent.width == size.width && extent.height == size.height &&
            format == color.pixelFormat) {
            return true;
        }
        interpolator = nil;
        extent = size;
        format = color.pixelFormat;
        id<MTLDevice> device = context.MtlDevice();
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
                LOG_ERROR(Render_Vulkan, "MetalFX frame interpolation: motion kernel failed: {}",
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
        descriptor.inputWidth = size.width;
        descriptor.inputHeight = size.height;
        descriptor.outputWidth = size.width;
        descriptor.outputHeight = size.height;
        id<MTLFXFrameInterpolator> created = [descriptor newFrameInterpolatorWithDevice:device];
        if (created == nil) {
            LOG_WARNING(Render_Vulkan, "MetalFX frame interpolation: no interpolator for {}x{}",
                        size.width, size.height);
            return false;
        }
        if (!HasUsage(color, created.colorTextureUsage)) {
            LOG_WARNING(Render_Vulkan,
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
        LOG_INFO(Render_Vulkan, "MetalFX frame interpolation: {}x{}{}", size.width, size.height,
                 output != nil ? " (through a copy)" : "");
        return true;
    }
    return false;
}
#endif

MetalFxFrameGen::MetalFxFrameGen(const Device& device, Scheduler& scheduler)
    : impl{std::make_unique<Impl>(device, scheduler)} {}

MetalFxFrameGen::~MetalFxFrameGen() = default;

bool MetalFxFrameGen::IsSupported([[maybe_unused]] const Device& device) {
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    if (!device.IsExtMetalObjectsEnabled()) {
        return false;
    }
    if (@available(iOS 26.0, macOS 26.0, *)) {
        id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
        return mtl_device != nil && [MTLFXFrameInterpolatorDescriptor supportsDevice:mtl_device];
    }
#endif
    return false;
}

bool MetalFxFrameGen::Generate([[maybe_unused]] VkImage previous,
                               [[maybe_unused]] VkImage current,
                               [[maybe_unused]] VkExtent2D extent,
                               [[maybe_unused]] VkImage motion_image,
                               [[maybe_unused]] VkExtent2D motion_extent,
                               [[maybe_unused]] VkImage destination,
                               [[maybe_unused]] f32 delta_time, [[maybe_unused]] bool reset) {
#ifdef LICHII_METALFX_FRAME_INTERPOLATOR
    Impl& state = *impl;
    if (state.failed || !state.context.IsValid()) {
        return false;
    }
    if (@available(iOS 26.0, macOS 26.0, *)) {
        @autoreleasepool {
            id<MTLTexture> previous_texture = state.context.Texture(previous);
            id<MTLTexture> current_texture = state.context.Texture(current);
            id<MTLTexture> flow_texture = state.context.Texture(motion_image);
            id<MTLTexture> destination_texture = state.context.Texture(destination);
            if (previous_texture == nil || current_texture == nil || flow_texture == nil ||
                destination_texture == nil ||
                destination_texture.pixelFormat != current_texture.pixelFormat ||
                !state.Prepare(current_texture, destination_texture, extent)) {
                state.failed = true;
                LOG_WARNING(Render_Vulkan,
                            "MetalFX frame interpolation unavailable; using FSR 3 frame generation");
                return false;
            }
            id<MTLFXFrameInterpolator> interpolator = state.interpolator;
            id<MTLTexture> output = state.output != nil ? state.output : destination_texture;
            const bool ran = state.context.Run([&](id<MTLCommandBuffer> command_buffer) {
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
                const MTLSize groups =
                    MTLSizeMake((extent.width + 15) / 16, (extent.height + 15) / 16, 1);
                [compute dispatchThreadgroups:groups threadsPerThreadgroup:group];
                [compute endEncoding];

                interpolator.colorTexture = current_texture;
                interpolator.prevColorTexture = previous_texture;
                interpolator.depthTexture = state.depth;
                interpolator.motionTexture = state.motion;
                interpolator.outputTexture = output;
                interpolator.motionVectorScaleX =
                    static_cast<float>(extent.width) / static_cast<float>(motion_extent.width);
                interpolator.motionVectorScaleY =
                    static_cast<float>(extent.height) / static_cast<float>(motion_extent.height);
                interpolator.deltaTime = delta_time;
                // A flat image with no camera: values that keep the depth plane in range.
                interpolator.nearPlane = 0.1f;
                interpolator.farPlane = 1000.0f;
                interpolator.fieldOfView = 60.0f;
                interpolator.aspectRatio =
                    static_cast<float>(extent.width) / static_cast<float>(extent.height);
                interpolator.jitterOffsetX = 0.0f;
                interpolator.jitterOffsetY = 0.0f;
                interpolator.shouldResetHistory = reset;
                [interpolator encodeToCommandBuffer:command_buffer];

                if (state.output != nil) {
                    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
                    [blit copyFromTexture:state.output toTexture:destination_texture];
                    [blit endEncoding];
                }
                return true;
            });
            if (!ran) {
                state.failed = true;
                LOG_WARNING(Render_Vulkan, "MetalFX frame interpolation failed; using FSR 3 "
                                           "frame generation");
            }
            return ran;
        }
    }
#endif
    return false;
}

} // namespace Vulkan
