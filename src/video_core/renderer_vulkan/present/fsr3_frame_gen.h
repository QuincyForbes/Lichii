// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/present/nv_optical_flow.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;
struct Frame;

/// AMD FidelityFX Super Resolution 3 (FSR 3) Frame Generation pipeline.
/// Uses NVIDIA hardware optical flow (VK_NV_optical_flow) when available,
/// or compute-based optical flow and bilateral frame interpolation.
class Fsr3FrameGen {
public:
    explicit Fsr3FrameGen(const Device& device, MemoryAllocator& memory_allocator,
                          VkExtent2D extent, VkFormat format);
    ~Fsr3FrameGen();

    void Process(const Device& device, Scheduler& scheduler, VkImage current_frame,
                 VkImageView current_view, u64 frame_count, bool dispatch_flow = true);

    void Generate(const Device& device, Scheduler& scheduler, VkImage destination,
                  VkImageView destination_storage_view, float blend_factor, float flow_scale);

    /// The semaphore the next graphics submission must wait on before reading the hardware
    /// optical flow, or a null handle. Taking it hands that wait to the caller. Generation on
    /// the compute queue keeps it.
    [[nodiscard]] VkSemaphore TakeFlowWait() {
        return m_async ? VK_NULL_HANDLE : std::exchange(m_flow_wait, VK_NULL_HANDLE);
    }

    /// Generated frames are built and submitted on the device's frame generation compute queue
    /// (GenerateAsync) instead of the scheduler.
    [[nodiscard]] bool IsAsync() const {
        return m_async;
    }

    /// Records and submits the generation into `destination` on the compute queue, signalling
    /// its render_ready semaphore and generated_ready fence.
    void GenerateAsync(const Device& device, Frame& destination, float blend_factor,
                       float flow_scale);

    [[nodiscard]] bool IsReady() const {
        return m_initialized && m_has_history && m_flow_valid;
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
    static constexpr u32 MaxAsyncFrames = 8;

    void CreateResources(const Device& device);
    void CreateShaders(const Device& device);
    void CreatePipelines(const Device& device);
    void CreateAsyncResources(const Device& device);

    MemoryAllocator& m_allocator;
    VkExtent2D m_extent;
    VkExtent2D m_flow_extent; // half-res for shader optical flow
    VkFormat m_format;

    vk::Sampler m_linear_sampler;
    vk::Sampler m_nearest_sampler;
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
    std::optional<DescriptorPool> m_interp_pool;
    DescriptorAllocator m_interp_allocator;

    std::array<vk::Image, NUM_PARITIES> m_frame_images;
    std::array<vk::ImageView, NUM_PARITIES> m_frame_views;
    u32 m_parity{};

    vk::Image m_motion_image;
    vk::ImageView m_motion_view;

    bool m_initialized{};
    bool m_has_history{};

    u64 m_last_frame_count{};
    bool m_flow_valid{};

    /// NVIDIA optical flow hardware in place of the flow shader, when available.
    std::unique_ptr<NvOpticalFlow> m_nvof;
    VkSemaphore m_flow_wait{};
    bool m_flow_copied{};

    // Generation on the compute queue: a command buffer and descriptor set per frame slot.
    std::vector<u32> m_shared_families;
    bool m_try_async{};
    const Device* m_device{};
    bool m_async{};
    vk::CommandPool m_async_pool;
    vk::CommandBuffers m_async_cmdbufs;
    vk::DescriptorPool m_async_desc_pool;
    vk::DescriptorSets m_async_sets;
};

} // namespace Vulkan
