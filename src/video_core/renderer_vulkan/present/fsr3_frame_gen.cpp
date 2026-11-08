// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/present/fsr3_frame_gen.h"
#include "video_core/renderer_vulkan/present/util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/vulkan_common/vulkan_device.h"

#include "video_core/host_shaders/fsr3_frame_interpolation_comp_spv.h"
#include "video_core/host_shaders/fsr3_optical_flow_comp_spv.h"

namespace Vulkan {

namespace {

// ΓöÇΓöÇ Precise pipeline barriers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

VkImageMemoryBarrier MakeBarrier(VkImage image, VkAccessFlags src_access,
                                 VkAccessFlags dst_access, VkImageLayout old_layout,
                                 VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
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

void BarrierComputeWriteToRead(vk::CommandBuffer cmdbuf, VkImage) {
    static constexpr VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, barrier, {}, {});
}

// ΓöÇΓöÇ Batched dual-copy ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
// Copy one source image into TWO destinations with only 3 barrier calls
// instead of 6 (avoiding redundant source transitions).
void CopyImageToTwo(vk::CommandBuffer cmdbuf, VkImage source,
                    VkImage dest_a, VkImage dest_b, VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    // 1. Transition source ΓåÆ TRANSFER_SRC, both dests ΓåÆ TRANSFER_DST (batched)
    const std::array before{
        MakeBarrier(source, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeBarrier(dest_a, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
        MakeBarrier(dest_b, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    // 2. Copy source ΓåÆ dest_a, then source ΓåÆ dest_b (source stays TRANSFER_SRC)
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_a,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dest_b,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    // 3. Transition all three back to GENERAL (batched)
    const std::array after{
        MakeBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(dest_a, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(dest_b, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

void CopyImageToImage(vk::CommandBuffer cmdbuf, VkImage source, VkImage destination,
                      VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    const std::array before{
        MakeBarrier(source, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeBarrier(destination, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    const std::array after{
        MakeBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

void CopyFrameAndPrepareMotion(vk::CommandBuffer cmdbuf, VkImage source, VkImage destination,
                               VkImage motion, VkExtent2D extent) {
    const VkImageCopy region{
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {extent.width, extent.height, 1},
    };

    const std::array before{
        MakeBarrier(source, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
        MakeBarrier(destination, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, {}, {}, before);

    cmdbuf.CopyImage(source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);

    const std::array after{
        MakeBarrier(source, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(destination, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL),
        MakeBarrier(motion, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL),
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           0, {}, {}, after);
}

} // anonymous namespace

struct FlowPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float search_radius;
    float block_size;
};

struct InterpPushConstants {
    float inv_render_w;
    float inv_render_h;
    float render_w;
    float render_h;
    float blend_factor;
    float flow_scale;
};

Fsr3FrameGen::Fsr3FrameGen(const Device& device, MemoryAllocator& memory_allocator,
                           VkExtent2D extent, VkFormat format)
    : m_allocator{memory_allocator}, m_extent{extent},
      m_flow_extent{(std::max)(1u, extent.width / 2), (std::max)(1u, extent.height / 2)},
      m_format{format} {
    CreateResources(device);
    CreateShaders(device);
    CreatePipelines(device);
    m_initialized = true;
}

Fsr3FrameGen::~Fsr3FrameGen() = default;

void Fsr3FrameGen::CreateResources(const Device& device) {
    m_linear_sampler = CreateBilinearSampler(device);

    for (size_t i = 0; i < NUM_PARITIES; ++i) {
        m_frame_images[i] = CreateWrappedImage(m_allocator, m_extent, m_format);
        m_frame_views[i] = CreateWrappedImageView(device, m_frame_images[i], m_format);
    }

    // Half-resolution motion vector image
    m_motion_image = CreateWrappedImage(m_allocator, m_flow_extent, VK_FORMAT_R16G16_SFLOAT);
    m_motion_view = CreateWrappedImageView(device, m_motion_image, VK_FORMAT_R16G16_SFLOAT);

    // Descriptor pool ΓÇö sized for flow (3 bindings) + interp (4 bindings) ├ù 2 sets
    const std::array pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                             static_cast<u32>(2 * NUM_PARITIES + 3 * NUM_PARITIES * INTERP_CACHE_SIZE)},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                             static_cast<u32>(NUM_PARITIES + NUM_PARITIES * INTERP_CACHE_SIZE)},
    };
    m_desc_pool = device.GetLogical().CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = static_cast<u32>(NUM_PARITIES + NUM_PARITIES * INTERP_CACHE_SIZE),
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
}

void Fsr3FrameGen::CreateShaders(const Device& device) {
    m_flow_shader = BuildShader(device, FSR3_OPTICAL_FLOW_COMP_SPV);
    m_interp_shader = BuildShader(device, FSR3_FRAME_INTERPOLATION_COMP_SPV);
}

void Fsr3FrameGen::CreatePipelines(const Device& device) {
    // ΓöÇΓöÇ Optical Flow Pipeline ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    const std::array flow_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    m_flow_desc_layout = device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<u32>(flow_bindings.size()),
        .pBindings = flow_bindings.data(),
    });

    const VkPushConstantRange flow_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(FlowPushConstants),
    };

    m_flow_pipe_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_flow_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &flow_push_range,
    });

    m_flow_pipeline = device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *m_flow_shader,
            .pName = "main",
        },
        .layout = *m_flow_pipe_layout,
    });

    // ΓöÇΓöÇ Frame Interpolation Pipeline ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    const std::array interp_bindings{
        VkDescriptorSetLayoutBinding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        VkDescriptorSetLayoutBinding{3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };

    m_interp_desc_layout = device.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<u32>(interp_bindings.size()),
        .pBindings = interp_bindings.data(),
    });

    const VkPushConstantRange interp_push_range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(InterpPushConstants),
    };

    m_interp_pipe_layout = device.GetLogical().CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = m_interp_desc_layout.address(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &interp_push_range,
    });

    m_interp_pipeline = device.GetLogical().CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *m_interp_shader,
            .pName = "main",
        },
        .layout = *m_interp_pipe_layout,
    });

    // Allocate descriptor sets
    std::array<VkDescriptorSetLayout, NUM_PARITIES> flow_layouts;
    flow_layouts.fill(*m_flow_desc_layout);
    m_flow_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, flow_layouts);

    std::array<VkDescriptorSetLayout, NUM_PARITIES * INTERP_CACHE_SIZE> interp_layouts;
    interp_layouts.fill(*m_interp_desc_layout);
    m_interp_desc_sets = CreateWrappedDescriptorSets(m_desc_pool, interp_layouts);

    const VkSampler sampler = *m_linear_sampler;
    const VkDescriptorImageInfo motion_sample_info{
        .sampler = sampler,
        .imageView = *m_motion_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo motion_store_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = *m_motion_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    for (size_t parity = 0; parity < NUM_PARITIES; ++parity) {
        const VkDescriptorImageInfo cur_info{
            .sampler = sampler,
            .imageView = *m_frame_views[parity],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo hist_info{
            .sampler = sampler,
            .imageView = *m_frame_views[parity ^ 1],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorSet flow_set = m_flow_desc_sets[parity];
        const std::array<VkWriteDescriptorSet, 3> flow_writes{{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, flow_set, 0, 0, 1,
             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cur_info, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, flow_set, 1, 0, 1,
             VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hist_info, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, flow_set, 2, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &motion_store_info, nullptr, nullptr},
        }};
        device.GetLogical().UpdateDescriptorSets(flow_writes, {});

        for (size_t slot = 0; slot < INTERP_CACHE_SIZE; ++slot) {
            const VkDescriptorSet interp_set =
                m_interp_desc_sets[parity * INTERP_CACHE_SIZE + slot];
            const std::array<VkWriteDescriptorSet, 3> interp_writes{{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, interp_set, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &cur_info, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, interp_set, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &hist_info, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, interp_set, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &motion_sample_info, nullptr, nullptr},
            }};
            device.GetLogical().UpdateDescriptorSets(interp_writes, {});
        }
    }
}

void Fsr3FrameGen::Process([[maybe_unused]] const Device& device, Scheduler& scheduler,
                           VkImage current_frame, [[maybe_unused]] VkImageView current_view,
                           u64 frame_count, bool dispatch_flow) {
    if (!m_has_history) {
        // First frame: populate history and current-frame copy via batched dual-copy
        scheduler.Record([this, current_frame](vk::CommandBuffer cmdbuf) {
            CopyImageToTwo(cmdbuf, current_frame, *m_frame_images[0], *m_frame_images[1], m_extent);
        });
        m_has_history = true;
        m_parity = 0;
        m_last_frame_count = frame_count;
        return;
    }

    if (!dispatch_flow) {
        m_parity ^= 1U;
        const VkImage cur_image = *m_frame_images[m_parity];
        scheduler.Record([this, current_frame, cur_image](vk::CommandBuffer cmdbuf) {
            CopyImageToImage(cmdbuf, current_frame, cur_image, m_extent);
        });
        m_last_frame_count = frame_count;
        m_flow_valid = false;
        return;
    }

    // ΓöÇΓöÇ Duplicate-frame detection ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if (frame_count == m_last_frame_count && m_flow_valid) {
        return;
    }
    m_last_frame_count = frame_count;

    // ΓöÇΓöÇ Descriptor caching ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    m_parity ^= 1U;
    const u32 parity = m_parity;
    const VkImage cur_image = *m_frame_images[parity];
    const VkDescriptorSet flow_set = m_flow_desc_sets[parity];

    // Dispatch Optical Flow at half resolution
    const u32 flow_w = m_flow_extent.width;
    const u32 flow_h = m_flow_extent.height;
    const FlowPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(flow_w),
        .inv_render_h = 1.0f / static_cast<float>(flow_h),
        .render_w = static_cast<float>(flow_w),
        .render_h = static_cast<float>(flow_h),
        .search_radius = 2.0f,
        .block_size = 2.0f,
    };

    scheduler.Record([this, constants, current_frame, cur_image, flow_set, flow_w,
                      flow_h](vk::CommandBuffer cmdbuf) {
        CopyFrameAndPrepareMotion(cmdbuf, current_frame, cur_image, *m_motion_image, m_extent);

        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_flow_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_flow_pipe_layout, 0,
                                  flow_set, {});
        cmdbuf.PushConstants(*m_flow_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((flow_w + 15) / 16, (flow_h + 15) / 16, 1);

        // Motion vectors: GENERAL ΓåÆ SHADER_READ_ONLY
        BarrierComputeWriteToRead(cmdbuf, *m_motion_image);
    });

    m_flow_valid = true;
}

void Fsr3FrameGen::Generate(const Device& device, Scheduler& scheduler,
                            VkImageView destination_storage_view, float blend_factor,
                            float flow_scale) {
    if (!m_has_history || !m_flow_valid) {
        return;
    }

    // ΓöÇΓöÇ Descriptor caching for interpolation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    const u32 parity = m_parity;
    auto& cache = m_cached_interp_dest_views[parity];
    u32 slot = INTERP_CACHE_SIZE;
    for (u32 i = 0; i < INTERP_CACHE_SIZE; ++i) {
        if (cache[i] == destination_storage_view) {
            slot = i;
            break;
        }
    }
    if (slot == INTERP_CACHE_SIZE) {
        slot = m_interp_next_slot[parity];
        m_interp_next_slot[parity] = (slot + 1) % INTERP_CACHE_SIZE;
        cache[slot] = destination_storage_view;

        const VkDescriptorSet dst_set = m_interp_desc_sets[parity * INTERP_CACHE_SIZE + slot];
        const VkDescriptorImageInfo dest_info{
            .sampler = VK_NULL_HANDLE,
            .imageView = destination_storage_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = dst_set,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &dest_info,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        device.GetLogical().UpdateDescriptorSets(std::span{&write, 1}, {});
    }
    const VkDescriptorSet interp_set = m_interp_desc_sets[parity * INTERP_CACHE_SIZE + slot];

    // Interpolation dispatches at full resolution
    const u32 out_w = m_extent.width;
    const u32 out_h = m_extent.height;
    const InterpPushConstants constants{
        .inv_render_w = 1.0f / static_cast<float>(out_w),
        .inv_render_h = 1.0f / static_cast<float>(out_h),
        .render_w = static_cast<float>(out_w),
        .render_h = static_cast<float>(out_h),
        .blend_factor = blend_factor,
        .flow_scale = flow_scale,
    };

    scheduler.Record([this, constants, interp_set, out_w, out_h](vk::CommandBuffer cmdbuf) {
        cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipeline);
        cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *m_interp_pipe_layout, 0,
                                  interp_set, {});
        cmdbuf.PushConstants(*m_interp_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
        cmdbuf.Dispatch((out_w + 15) / 16, (out_h + 15) / 16, 1);
    });
}

} // namespace Vulkan
