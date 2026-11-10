// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// AMD FidelityFX Super Resolution 3 (FSR 3) Frame Generation pipeline.
/// Uses compute-based optical flow and bilateral frame interpolation
/// to synthesize intermediate frames without requiring engine motion vectors.
///
/// Optimizations:
///  - Quarter-resolution optical flow (16├ù fewer compute invocations)
///  - Shared-memory tile caching in both shaders
///  - Early-out for static pixels (skip diamond search)
///  - Duplicate-frame detection (skip optical flow entirely)
///  - Precise pipeline barriers instead of full-pipeline TransitionImageLayout
///  - Descriptor-write caching (only update when image views change)
class Fsr3FrameGen {
public:
    explicit Fsr3FrameGen(const Device& device, MemoryAllocator& memory_allocator,
                          VkExtent2D extent, VkFormat format);
    ~Fsr3FrameGen();

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count, bool dispatch_flow = true);

    void Generate(const Device& device, Scheduler& scheduler, VkImageView destination_storage_view,
                  float blend_factor, float flow_scale);

    [[nodiscard]] bool IsReady() const {
        return m_initialized && m_has_history;
    }

    /// Whether the last Process left two frames and the flow between them (Generate can run).
    [[nodiscard]] bool HasFlow() const {
        return m_has_history && m_flow_valid;
    }
    [[nodiscard]] VkImage CurrentFrame() const {
        return *m_frame_images[m_parity];
    }
    [[nodiscard]] VkImage PreviousFrame() const {
        return *m_frame_images[m_parity ^ 1];
    }
    /// R16G16F at FlowExtent: per current pixel, the offset in flow texels to the previous frame.
    [[nodiscard]] VkImage MotionImage() const {
        return *m_motion_image;
    }
    [[nodiscard]] VkExtent2D FlowExtent() const {
        return m_flow_extent;
    }

private:
    static constexpr size_t NUM_PARITIES = 2;
    static constexpr size_t INTERP_CACHE_SIZE = 8;

    void CreateResources(const Device& device);
    void CreateShaders(const Device& device);
    void CreatePipelines(const Device& device);

    MemoryAllocator& m_allocator;
    VkExtent2D m_extent;
    VkExtent2D m_flow_extent;   // half-res for optical flow
    VkFormat m_format;

    vk::Sampler m_linear_sampler;
    vk::ShaderModule m_flow_shader;
    vk::ShaderModule m_interp_shader;

    vk::DescriptorSetLayout m_flow_desc_layout;
    vk::PipelineLayout m_flow_pipe_layout;
    vk::Pipeline m_flow_pipeline;

    vk::DescriptorSetLayout m_interp_desc_layout;
    vk::PipelineLayout m_interp_pipe_layout;
    vk::Pipeline m_interp_pipeline;

    vk::DescriptorPool m_desc_pool;
    vk::DescriptorSets m_flow_desc_sets;
    vk::DescriptorSets m_interp_desc_sets;

    std::array<vk::Image, NUM_PARITIES> m_frame_images;
    std::array<vk::ImageView, NUM_PARITIES> m_frame_views;
    u32 m_parity{};

    vk::Image m_motion_image;
    vk::ImageView m_motion_view;

    bool m_initialized{};
    bool m_has_history{};

    // Duplicate-frame detection: skip optical flow when the frame hasn't changed
    u64 m_last_frame_count{};
    bool m_flow_valid{};

    std::array<std::array<VkImageView, INTERP_CACHE_SIZE>, NUM_PARITIES> m_cached_interp_dest_views{};
    std::array<u32, NUM_PARITIES> m_interp_next_slot{};
};

} // namespace Vulkan
