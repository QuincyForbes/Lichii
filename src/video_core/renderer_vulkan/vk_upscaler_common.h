// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

inline constexpr VkFormat UPSCALER_OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
inline constexpr VkImageUsageFlags UPSCALER_OUTPUT_USAGE =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
    VK_IMAGE_USAGE_TRANSFER_DST_BIT;

inline constexpr VkImageSubresourceRange UPSCALER_COLOR_RANGE{
    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    .baseMipLevel = 0,
    .levelCount = 1,
    .baseArrayLayer = 0,
    .layerCount = 1,
};

inline constexpr VkAccessFlags UPSCALER_INPUT_WRITES =
    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;

[[nodiscard]] constexpr VkImageMemoryBarrier MakeUpscalerImageBarrier(
    VkImage image, VkImageSubresourceRange range, VkAccessFlags src_access,
    VkAccessFlags dst_access, VkImageLayout old_layout, VkImageLayout new_layout) noexcept {
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
        .subresourceRange = range,
    };
}

template <typename InputImage>
[[nodiscard]] VkImageMemoryBarrier MakeUpscalerInputToReadBarrier(
    const InputImage& image) noexcept {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return MakeUpscalerImageBarrier(image.image, range, UPSCALER_INPUT_WRITES,
                                    VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

template <typename InputImage>
[[nodiscard]] VkImageMemoryBarrier MakeUpscalerInputToGeneralBarrier(
    const InputImage& image) noexcept {
    VkImageSubresourceRange range = image.range;
    range.aspectMask = image.barrier_aspect;
    return MakeUpscalerImageBarrier(image.image, range, VK_ACCESS_SHADER_READ_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | UPSCALER_INPUT_WRITES,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL);
}

[[nodiscard]] inline vk::Sampler CreateUpscalerSampler(const Device& device, VkFilter filter) {
    return device.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = filter,
        .minFilter = filter,
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
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
}

inline void CreateUpscalerOutput(const Device& device, MemoryAllocator& memory_allocator,
                                 Scheduler& scheduler, VkExtent2D upscale_size,
                                 vk::Image& output_image, vk::ImageView& output_view) {
    if (output_image) {
        scheduler.Finish(); // the old output may still be in use
    }
    const MemoryTagScope memory_tag{MemoryTag::Upscaler};
    output_image = memory_allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = UPSCALER_OUTPUT_FORMAT,
        .extent = {upscale_size.width, upscale_size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = UPSCALER_OUTPUT_USAGE,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    output_view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *output_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = UPSCALER_OUTPUT_FORMAT,
        .components{},
        .subresourceRange = UPSCALER_COLOR_RANGE,
    });
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *output_image](vk::CommandBuffer cmdbuf) {
        cmdbuf.PipelineBarrier(
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            MakeUpscalerImageBarrier(image, UPSCALER_COLOR_RANGE, 0,
                                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
    });
}

} // namespace Vulkan
