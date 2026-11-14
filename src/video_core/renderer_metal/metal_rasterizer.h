// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <utility>

#include "common/common_types.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_fence_manager.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace Tegra {
class GPU;
struct FramebufferConfig;
} // namespace Tegra

namespace Metal {

class Device;
class Scheduler;

class AccelerateDMA : public Tegra::Engines::AccelerateDMAInterface {
public:
    explicit AccelerateDMA(BufferCache& buffer_cache, TextureCache& texture_cache);

    bool BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) override;
    bool BufferClear(GPUVAddr src_address, u64 amount, u32 value) override;
    bool ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::ImageOperand& src,
                       const Tegra::DMA::BufferOperand& dst) override;
    bool BufferToImage(const Tegra::DMA::ImageCopy& copy_info, const Tegra::DMA::BufferOperand& src,
                       const Tegra::DMA::ImageOperand& dst) override;

private:
    template <bool IS_IMAGE_UPLOAD>
    bool DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                            const Tegra::DMA::BufferOperand& buffer_operand,
                            const Tegra::DMA::ImageOperand& image_operand);

    BufferCache& buffer_cache;
    TextureCache& texture_cache;
};

/// The guest framebuffer a present shows, found in the texture cache.
struct DisplayTexture {
    TextureHandle texture{};
    u32 width{};
    u32 height{};
    u32 scaled_width{};
    u32 scaled_height{};
};

/// Takes the GPU's work into the texture and buffer caches. Milestone 2: memory, copies, blits,
/// clears and DMA run on Metal; draws and dispatches are counted, not drawn (milestone 3).
class RasterizerMetal final : public VideoCore::RasterizerInterface,
                              protected VideoCommon::ChannelSetupCaches<VideoCommon::ChannelInfo> {
public:
    explicit RasterizerMetal(Tegra::GPU& gpu, Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device, Scheduler& scheduler);
    ~RasterizerMetal() override;

    void Draw(bool is_indexed, u32 instance_count) override;
    void DrawTexture() override;
    void Clear(u32 layer_count) override;
    void DispatchCompute() override;
    void ResetCounter(VideoCommon::QueryType type) override;
    void Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
               VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) override;
    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size) override;
    void DisableGraphicsUniformBuffer(size_t stage, u32 index) override;
    void FlushAll() override;
    void FlushRegion(DAddr addr, u64 size,
                     VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    bool MustFlushRegion(DAddr addr, u64 size,
                         VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void InvalidateRegion(DAddr addr, u64 size,
                          VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void OnCacheInvalidation(DAddr addr, u64 size) override;
    bool OnCPUWrite(DAddr addr, u64 size) override;
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override;
    void InvalidateGPUCache() override;
    void UnmapMemory(DAddr addr, u64 size) override;
    void ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) override;
    void SignalFence(std::function<void()>&& func) override;
    void SyncOperation(std::function<void()>&& func) override;
    void SignalSyncPoint(u32 value) override;
    void SignalReference() override;
    void ReleaseFences(bool force) override;
    void FlushAndInvalidateRegion(
        DAddr addr, u64 size, VideoCommon::CacheType which = VideoCommon::CacheType::All) override;
    void WaitForIdle() override;
    void FragmentBarrier() override;
    void TiledCacheBarrier() override;
    void FlushCommands() override;
    void TickFrame() override;
    bool AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                               const Tegra::Engines::Fermi2D::Surface& dst,
                               const Tegra::Engines::Fermi2D::Config& copy_config) override;
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override;
    void AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                  std::span<const u8> memory) override;
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;
    void InitializeChannel(Tegra::Control::ChannelState& channel) override;
    void BindChannel(Tegra::Control::ChannelState& channel) override;
    void ReleaseChannel(s32 channel_id) override;

    /// The texture cache's image at the framebuffer's address, if it has one.
    [[nodiscard]] std::optional<DisplayTexture> AccelerateDisplay(
        const Tegra::FramebufferConfig& config, DAddr framebuffer_addr);

    struct WorkCounts {
        u64 draws;
        u64 draws_with_pipeline; ///< Their pipeline translated and built by Metal
        u64 draws_encoded;       ///< Recorded on a render encoder
        u64 dispatches;
        u64 dispatches_with_pipeline;
        u64 clears;
    };

    /// Work taken since the last call (the renderer's 10 s report).
    [[nodiscard]] WorkCounts TakeWorkCounts() noexcept {
        return {std::exchange(draws, 0), std::exchange(draws_with_pipeline, 0),
                std::exchange(draws_encoded, 0),
                std::exchange(dispatches, 0), std::exchange(dispatches_with_pipeline, 0),
                std::exchange(clears, 0)};
    }

    [[nodiscard]] PipelineCache& Pipelines() noexcept {
        return pipeline_cache;
    }

    [[nodiscard]] const TextureCacheRuntime& TextureRuntime() const noexcept {
        return texture_cache_runtime;
    }

    [[nodiscard]] const BufferCacheRuntime& BufferRuntime() const noexcept {
        return buffer_cache_runtime;
    }

private:
    /// Binds everything the draw needs and opens its render pass; false to skip it.
    bool PrepareDraw(GraphicsPipeline& pipeline, bool is_indexed);

    void QueryFallback(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                       VideoCommon::QueryPropertiesFlags flags, u32 payload);

    Tegra::GPU& gpu;
    const Device& device;
    Scheduler& scheduler;

    StagingBufferPool staging_pool;
    TextureCacheRuntime texture_cache_runtime;
    TextureCache texture_cache;
    BufferCacheRuntime buffer_cache_runtime;
    BufferCache buffer_cache;
    PipelineCache pipeline_cache;
    QueryCache query_cache;
    AccelerateDMA accelerate_dma;
    FenceManager fence_manager;

    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    ::Common::unordered_map<u64, Object> depth_stencil_states;

    u64 draws{};
    u64 draws_with_pipeline{};
    u64 draws_encoded{};
    u64 dispatches{};
    u64 dispatches_with_pipeline{};
    u64 clears{};
};

} // namespace Metal
