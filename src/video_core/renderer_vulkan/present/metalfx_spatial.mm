// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#import <MetalFX/MetalFX.h>

#include <algorithm>
#include <cmath>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/metalfx_context.h"
#include "video_core/renderer_vulkan/present/metalfx_spatial.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

// The present chain's sources vary (RGBA8, BGRA8, A2B10G10R10, FXAA's output); a blit copies the
// crop into one format MetalFX takes.
constexpr VkFormat INPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;

VkImageMemoryBarrier MakeBarrier(VkImage image, VkAccessFlags src_access, VkAccessFlags dst_access,
                                 VkImageLayout old_layout, VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
}

} // Anonymous namespace

struct MetalFxSpatial::Impl {
    Impl(const Device& device_, MemoryAllocator& allocator_, Scheduler& scheduler,
         VkExtent2D extent_)
        : device{device_}, allocator{allocator_}, context{device_, scheduler}, extent{extent_} {}

    bool Prepare(Scheduler& scheduler, VkExtent2D input_size);

    const Device& device;
    MemoryAllocator& allocator;
    MetalFxContext context;
    VkExtent2D extent;
    VkExtent2D input_extent{};
    vk::Image input;
    vk::Image output;
    vk::ImageView output_view;
    id<MTLFXSpatialScaler> scaler = nil;
    bool failed{};
};

bool MetalFxSpatial::Impl::Prepare(Scheduler& scheduler, VkExtent2D input_size) {
    if (scaler != nil && input_extent.width == input_size.width &&
        input_extent.height == input_size.height) {
        return true;
    }
    if (input) {
        scheduler.Finish();
    }
    scaler = nil;
    input_extent = input_size;
    input = CreateMetalFxImage(allocator, input_size, INPUT_FORMAT,
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_STORAGE_BIT);
    if (!output) {
        output = CreateMetalFxImage(allocator, extent, OUTPUT_FORMAT,
                                    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        output_view = CreateWrappedImageView(device, output, OUTPUT_FORMAT);
    }
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([input_image = *input, output_image = *output](vk::CommandBuffer cmdbuf) {
        const std::array barriers{
            MakeBarrier(input_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_GENERAL),
            MakeBarrier(output_image, 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_GENERAL),
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                               0, {}, {}, barriers);
    });

    id<MTLTexture> input_texture = context.Texture(*input);
    id<MTLTexture> output_texture = context.Texture(*output);
    if (input_texture == nil || output_texture == nil) {
        return false;
    }
    MTLFXSpatialScalerDescriptor* descriptor = [[MTLFXSpatialScalerDescriptor alloc] init];
    descriptor.inputWidth = input_size.width;
    descriptor.inputHeight = input_size.height;
    descriptor.outputWidth = extent.width;
    descriptor.outputHeight = extent.height;
    descriptor.colorTextureFormat = input_texture.pixelFormat;
    descriptor.outputTextureFormat = output_texture.pixelFormat;
    // The present chain is in display (gamma) space.
    descriptor.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
    scaler = [descriptor newSpatialScalerWithDevice:context.MtlDevice()];
    if (scaler == nil) {
        LOG_WARNING(Render_Vulkan, "MetalFX spatial: no scaler for {}x{} -> {}x{}",
                    input_size.width, input_size.height, extent.width, extent.height);
        return false;
    }
    if (!HasUsage(input_texture, scaler.colorTextureUsage) ||
        !HasUsage(output_texture, scaler.outputTextureUsage)) {
        LOG_WARNING(Render_Vulkan, "MetalFX spatial: the images lack the texture usage it needs");
        scaler = nil;
        return false;
    }
    LOG_INFO(Render_Vulkan, "MetalFX spatial: {}x{} -> {}x{}", input_size.width,
             input_size.height, extent.width, extent.height);
    return true;
}

MetalFxSpatial::MetalFxSpatial(const Device& device, MemoryAllocator& memory_allocator,
                               Scheduler& scheduler, VkExtent2D extent)
    : impl{std::make_unique<Impl>(device, memory_allocator, scheduler, extent)} {}

MetalFxSpatial::~MetalFxSpatial() = default;

MetalFxSpatial::MetalFxSpatial(MetalFxSpatial&&) noexcept = default;

MetalFxSpatial& MetalFxSpatial::operator=(MetalFxSpatial&&) noexcept = default;

bool MetalFxSpatial::IsSupported(const Device& device) {
    if (!device.IsExtMetalObjectsEnabled()) {
        return false;
    }
    if (@available(iOS 16.0, macOS 13.0, *)) {
        id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
        return mtl_device != nil && [MTLFXSpatialScalerDescriptor supportsDevice:mtl_device];
    }
    return false;
}

std::optional<VkImageView> MetalFxSpatial::Draw(Scheduler& scheduler, VkImage source_image,
                                                VkExtent2D input_image_extent,
                                                const Common::Rectangle<f32>& crop_rect) {
    Impl& state = *impl;
    if (state.failed || !state.context.IsValid()) {
        return std::nullopt;
    }
    const f32 width = static_cast<f32>(input_image_extent.width);
    const f32 height = static_cast<f32>(input_image_extent.height);
    // A crop can flip the image (right < left or bottom < top): the blit keeps the flip.
    const s32 left = static_cast<s32>(std::lround(crop_rect.left * width));
    const s32 right = static_cast<s32>(std::lround(crop_rect.right * width));
    const s32 top = static_cast<s32>(std::lround(crop_rect.top * height));
    const s32 bottom = static_cast<s32>(std::lround(crop_rect.bottom * height));
    const VkExtent2D input_size{
        .width = static_cast<u32>(std::max(1, std::abs(right - left))),
        .height = static_cast<u32>(std::max(1, std::abs(bottom - top))),
    };
    @autoreleasepool {
        if (!state.Prepare(scheduler, input_size)) {
            state.failed = true;
            return std::nullopt;
        }

        const VkImageBlit blit{
            .srcSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets{{left, top, 0}, {right, bottom, 1}},
            .dstSubresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets{{0, 0, 0},
                        {static_cast<s32>(input_size.width), static_cast<s32>(input_size.height),
                         1}},
        };
        scheduler.RequestOutsideRenderPassOperationContext();
        scheduler.Record([source_image, input = *state.input, blit](vk::CommandBuffer cmdbuf) {
            const std::array before{
                MakeBarrier(source_image,
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
                MakeBarrier(input, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);
            cmdbuf.BlitImage(source_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, input,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blit, VK_FILTER_LINEAR);
            const std::array after{
                MakeBarrier(source_image, VK_ACCESS_TRANSFER_READ_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
                MakeBarrier(input, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, {}, {}, after);
        });

        id<MTLTexture> input_texture = state.context.Texture(*state.input);
        id<MTLTexture> output_texture = state.context.Texture(*state.output);
        id<MTLFXSpatialScaler> scaler = state.scaler;
        const bool ran = state.context.Run([&](id<MTLCommandBuffer> command_buffer) {
            scaler.colorTexture = input_texture;
            scaler.outputTexture = output_texture;
            scaler.inputContentWidth = input_size.width;
            scaler.inputContentHeight = input_size.height;
            [scaler encodeToCommandBuffer:command_buffer];
            return true;
        });
        if (!ran) {
            LOG_WARNING(Render_Vulkan, "MetalFX spatial: the Metal work failed; presenting "
                                       "without it");
            state.failed = true;
            return std::nullopt;
        }
    }
    return *state.output_view;
}

} // namespace Vulkan
