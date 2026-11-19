// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <span>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

class Device;
class PipelineStatistics;
class Scheduler;

/// EDEN_NX2_DISPATCH_DEPS=1: dispatches that end an open render pass are counted by what ties
/// them to it (its attachments bound, storage writes in it), every 10 s.
bool DispatchDepsTrace();

class ComputePipeline {
public:
    explicit ComputePipeline(const Device& device, Scheduler& scheduler, vk::PipelineCache& pipeline_cache,
                             DescriptorPool& descriptor_pool,
                             GuestDescriptorQueue& guest_descriptor_queue,
                             DescriptorBufferRing& descriptor_buffer_ring,
                             Common::ThreadWorker* thread_worker,
                             PipelineStatistics* pipeline_statistics,
                             VideoCore::ShaderNotify* shader_notify, const Shader::Info& info,
                             vk::ShaderModule spv_module, u64 shader_hash);

    ComputePipeline& operator=(ComputePipeline&&) noexcept = delete;
    ComputePipeline(ComputePipeline&&) noexcept = delete;

    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;

    using BindCommand = std::function<void(vk::CommandBuffer)>;

    [[nodiscard]] bool Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                 Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                 BufferCache& buffer_cache, TextureCache& texture_cache,
                                 TextureCacheRuntime& texture_runtime,
                                 std::optional<ByteAddressedShadow>& shadow,
                                 BindCommand* deferred_bind = nullptr);

    /// Whether the bindings fit an Ocarina upscale pass: one written image, no other writes.
    [[nodiscard]] bool CanReplaceOcarinaUpscale() const;

    bool IsBound() const noexcept {
        return static_cast<bool>(pipeline);
    }

    /// Whether no build of this pipeline is left queued on a worker thread, which holds it
    [[nodiscard]] bool IsBuildTaskDone() const noexcept {
        return build_task_done.load(std::memory_order::acquire);
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return is_built.load(std::memory_order::acquire);
    }

    [[nodiscard]] u64 ShaderHash() const noexcept {
        return shader_hash;
    }

    /// Storage images written by the last configured dispatch
    [[nodiscard]] std::span<const VideoCommon::ImageViewId> WrittenImages() const noexcept {
        return written_images;
    }

    struct SampledImage {
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
        bool gpu_modified{};
        bool recent_gpu_modified{};
    };

    /// Sampled textures, kept during frame dumps or writer diagnostics. Diagnostic state is
    /// captured before the dispatch's own storage-image writes are marked.
    [[nodiscard]] std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    [[nodiscard]] const Shader::Info& Info() const noexcept {
        return info;
    }

    /// Every image view the last configured dispatch bound, kept while EDEN_NX2_DISPATCH_DEPS runs.
    [[nodiscard]] std::span<const VideoCommon::ImageViewInOut> BoundImages() const noexcept {
        return bound_images;
    }

    /// Advances the frame window of EDEN_NX2_UPSCALER_TRACE
    static void TickUpscalerTrace();

    /// Whether the build ended without creating a pipeline (driver rejected the shader).
    [[nodiscard]] bool BuildFailed() const noexcept {
        return is_built.load(std::memory_order::acquire) && !pipeline;
    }

    [[nodiscard]] size_t StorageBufferCount() const noexcept {
        return info.storage_buffers_descriptors.size();
    }

    [[nodiscard]] u32 StorageImageCount() const noexcept {
        u32 count{};
        for (const auto& desc : info.image_descriptors) {
            count += desc.count;
        }
        return count;
    }

private:
    /// Builds the pipeline once, on whichever thread gets here first.
    void RunBuild();

    const Device& device;
    vk::PipelineCache& pipeline_cache;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    Shader::Info info;
    u64 shader_hash{};
    u32 num_descriptor_entries{};
    bool has_image_resources{false};
    bool has_rescaling_push_constants{false};
    boost::container::small_vector<VideoCommon::ImageViewId, 8> written_images;
    std::vector<SampledImage> sampled_images;
    std::vector<VideoCommon::ImageViewInOut> bound_images;
    boost::container::small_vector<u8, 64> texture_is_storage_ptr;
    bool has_storage_ptr_textures{};

    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};

    vk::ShaderModule spv_module;
    VkDescriptorSetLayout descriptor_set_layout{}; ///< Owned by the PipelineLayoutCache
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<u32> binding_kinds; ///< EDEN_NX2_BINDMEMO: per texture/image view, in read order
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    std::vector<u8> last_descriptor_bytes;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};
    bool has_last_descriptor_buffer{};
    DescriptorAllocator descriptor_allocator;
    VkPipelineLayout pipeline_layout{}; ///< Owned by the PipelineLayoutCache
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    std::function<void()> build_task;
    std::atomic_bool build_claimed{false};
    std::atomic_bool build_task_done{true};
    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
};

} // namespace Vulkan
