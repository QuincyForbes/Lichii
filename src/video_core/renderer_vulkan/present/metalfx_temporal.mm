// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <MetalFX/MetalFX.h>

#include <algorithm>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/metalfx_context.h"
#include "video_core/renderer_vulkan/present/metalfx_temporal.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan {

namespace {

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

/// The 2D texture of one level and layer of a Vulkan image's Metal texture.
id<MTLTexture> Plane(id<MTLTexture> texture, const VkImageSubresourceRange& range) {
    if (texture == nil) {
        return nil;
    }
    if (texture.textureType == MTLTextureType2D && range.baseMipLevel == 0 &&
        range.baseArrayLayer == 0) {
        return texture;
    }
    return [texture newTextureViewWithPixelFormat:texture.pixelFormat
                                      textureType:MTLTextureType2D
                                           levels:NSMakeRange(range.baseMipLevel, 1)
                                           slices:NSMakeRange(range.baseArrayLayer, 1)];
}

struct ScalerKey {
    MTLPixelFormat color{};
    MTLPixelFormat depth{};
    MTLPixelFormat motion{};
    MTLPixelFormat output{};
    MTLPixelFormat mask{};
    NSUInteger input_width{};
    NSUInteger input_height{};
    VkExtent2D render{};
    VkExtent2D upscale{};
    bool hdr{};

    bool operator==(const ScalerKey& other) const {
        return color == other.color && depth == other.depth && motion == other.motion &&
               output == other.output && mask == other.mask &&
               input_width == other.input_width && input_height == other.input_height &&
               render.width == other.render.width && render.height == other.render.height &&
               upscale.width == other.upscale.width && upscale.height == other.upscale.height &&
               hdr == other.hdr;
    }
};

} // Anonymous namespace

struct MetalFxTemporalUpscaler::Impl {
    Impl(const Device& device, Scheduler& scheduler) : context{device, scheduler} {}

    bool Prepare(const ScalerKey& key);

    MetalFxContext context;
    id<MTLFXTemporalScaler> scaler = nil;
    ScalerKey key{};
    bool content_properties{};
    bool first_evaluation{};
};

bool MetalFxTemporalUpscaler::Impl::Prepare(const ScalerKey& wanted) {
    if (scaler != nil && key == wanted) {
        return true;
    }
    scaler = nil;
    key = wanted;
    id<MTLDevice> device = context.MtlDevice();
    MTLFXTemporalScalerDescriptor* descriptor = [[MTLFXTemporalScalerDescriptor alloc] init];
    descriptor.colorTextureFormat = wanted.color;
    descriptor.depthTextureFormat = wanted.depth;
    descriptor.motionTextureFormat = wanted.motion;
    descriptor.outputTextureFormat = wanted.output;
    descriptor.inputWidth = wanted.input_width;
    descriptor.inputHeight = wanted.input_height;
    descriptor.outputWidth = wanted.upscale.width;
    descriptor.outputHeight = wanted.upscale.height;
    descriptor.autoExposureEnabled = wanted.hdr;
    // The game renders into part of larger images: MetalFX reads the render size from them.
    content_properties = wanted.render.width != wanted.input_width ||
                         wanted.render.height != wanted.input_height;
    if (content_properties) {
        const f32 scale_x =
            static_cast<f32>(wanted.upscale.width) / static_cast<f32>(wanted.render.width);
        const f32 scale_y =
            static_cast<f32>(wanted.upscale.height) / static_cast<f32>(wanted.render.height);
        descriptor.inputContentPropertiesEnabled = YES;
        descriptor.inputContentMinScale = std::min(scale_x, scale_y);
        descriptor.inputContentMaxScale = std::max(scale_x, scale_y);
    }
    if (wanted.mask != MTLPixelFormatInvalid) {
        if (@available(iOS 18.0, macOS 15.0, *)) {
            descriptor.reactiveMaskTextureEnabled = YES;
            descriptor.reactiveMaskTextureFormat = wanted.mask;
        }
    }
    scaler = [descriptor newTemporalScalerWithDevice:device];
    if (scaler == nil) {
        LOG_WARNING(Render_Vulkan, "MetalFX temporal: no scaler for {}x{} -> {}x{}",
                    wanted.render.width, wanted.render.height, wanted.upscale.width,
                    wanted.upscale.height);
        return false;
    }
    first_evaluation = true;
    LOG_INFO(Render_Vulkan, "MetalFX temporal: {}x{} -> {}x{}{}", wanted.render.width,
             wanted.render.height, wanted.upscale.width, wanted.upscale.height,
             wanted.mask != MTLPixelFormatInvalid ? " with a reactive mask" : "");
    return true;
}

MetalFxTemporalUpscaler::MetalFxTemporalUpscaler(const Device& device_, Scheduler& scheduler_,
                                                 MemoryAllocator& memory_allocator_)
    : device{device_}, scheduler{scheduler_}, memory_allocator{memory_allocator_} {
    if (!IsSupported(device)) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    impl = std::make_unique<Impl>(device, scheduler);
    if (!impl->context.IsValid()) {
        throw vk::Exception(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    point_sampler = device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

MetalFxTemporalUpscaler::~MetalFxTemporalUpscaler() = default;

bool MetalFxTemporalUpscaler::IsSupported(const Device& device) {
    if (!device.IsExtMetalObjectsEnabled()) {
        return false;
    }
    if (@available(iOS 16.0, macOS 13.0, *)) {
        id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
        return mtl_device != nil && [MTLFXTemporalScalerDescriptor supportsDevice:mtl_device];
    }
    return false;
}

void MetalFxTemporalUpscaler::CreateOutput(VkExtent2D upscale_size) {
    if (output_image) {
        scheduler.Finish();
    }
    upscale_extent = upscale_size;
    output_image = CreateMetalFxImage(memory_allocator, upscale_size, OUTPUT_FORMAT,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                          VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    output_view = CreateWrappedImageView(device, output_image, OUTPUT_FORMAT);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, barrier);
    });
}

bool MetalFxTemporalUpscaler::Upscale(const Inputs& inputs) {
    if (failed) {
        return false;
    }
    if (!output_image || upscale_extent.width != inputs.upscale_size.width ||
        upscale_extent.height != inputs.upscale_size.height) {
        CreateOutput(inputs.upscale_size);
    }
    Impl& state = *impl;
    @autoreleasepool {
        id<MTLTexture> color =
            Plane(state.context.Texture(inputs.color.image), inputs.color.range);
        id<MTLTexture> depth =
            Plane(state.context.Texture(inputs.depth.image), inputs.depth.range);
        id<MTLTexture> motion =
            Plane(state.context.Texture(inputs.motion_vectors.image), inputs.motion_vectors.range);
        id<MTLTexture> output = state.context.Texture(*output_image);
        id<MTLTexture> mask = nil;
        if (inputs.bias_current_color) {
            if (@available(iOS 18.0, macOS 15.0, *)) {
                mask = Plane(state.context.Texture(inputs.bias_current_color->image),
                             inputs.bias_current_color->range);
            }
        }
        if (color == nil || depth == nil || motion == nil || output == nil) {
            LOG_WARNING(Render_Vulkan, "MetalFX temporal: no Metal textures for the inputs");
            failed = true;
            return false;
        }
        const ScalerKey key{
            .color = color.pixelFormat,
            .depth = depth.pixelFormat,
            .motion = motion.pixelFormat,
            .output = output.pixelFormat,
            .mask = mask != nil ? mask.pixelFormat : MTLPixelFormatInvalid,
            .input_width = color.width,
            .input_height = color.height,
            .render = inputs.render_size,
            .upscale = inputs.upscale_size,
            .hdr = inputs.hdr,
        };
        if (!state.Prepare(key)) {
            failed = true;
            return false;
        }
        id<MTLFXTemporalScaler> scaler = state.scaler;
        bool usable = HasUsage(color, scaler.colorTextureUsage) &&
                      HasUsage(depth, scaler.depthTextureUsage) &&
                      HasUsage(motion, scaler.motionTextureUsage) &&
                      HasUsage(output, scaler.outputTextureUsage);
        if (mask != nil) {
            if (@available(iOS 18.0, macOS 15.0, *)) {
                usable = usable && HasUsage(mask, scaler.reactiveTextureUsage);
            }
        }
        if (!usable) {
            LOG_WARNING(Render_Vulkan, "MetalFX temporal: the inputs lack the texture usage it "
                                       "needs");
            failed = true;
            return false;
        }
        const bool reset = inputs.reset || state.first_evaluation;
        state.first_evaluation = false;
        const bool ran = state.context.Run([&](id<MTLCommandBuffer> command_buffer) {
            scaler.colorTexture = color;
            scaler.depthTexture = depth;
            scaler.motionTexture = motion;
            scaler.outputTexture = output;
            if (state.content_properties) {
                scaler.inputContentWidth = inputs.render_size.width;
                scaler.inputContentHeight = inputs.render_size.height;
            }
            scaler.jitterOffsetX = inputs.jitter[0];
            scaler.jitterOffsetY = inputs.jitter[1];
            scaler.motionVectorScaleX = inputs.motion_vector_scale[0];
            scaler.motionVectorScaleY = inputs.motion_vector_scale[1];
            scaler.depthReversed = inputs.depth_inverted;
            scaler.preExposure = 1.0f;
            scaler.reset = reset;
            if (@available(iOS 18.0, macOS 15.0, *)) {
                if (mask != nil) {
                    scaler.reactiveMaskTexture = mask;
                }
            }
            [scaler encodeToCommandBuffer:command_buffer];
            return true;
        });
        if (!ran) {
            LOG_WARNING(Render_Vulkan, "MetalFX temporal: the Metal work failed");
            failed = true;
            return false;
        }
    }
    return true;
}

} // namespace Vulkan
