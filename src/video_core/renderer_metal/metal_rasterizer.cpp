// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <mutex>

#include "common/alignment.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/framebuffer_config.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_rasterizer.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

AccelerateDMA::AccelerateDMA(BufferCache& buffer_cache_, TextureCache& texture_cache_)
    : buffer_cache{buffer_cache_}, texture_cache{texture_cache_} {}

bool AccelerateDMA::BufferCopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMACopy(src_address, dest_address, amount);
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    std::scoped_lock lock{buffer_cache.mutex};
    return buffer_cache.DMAClear(src_address, amount, value);
}

template <bool IS_IMAGE_UPLOAD>
bool AccelerateDMA::DmaBufferImageCopy(const Tegra::DMA::ImageCopy& copy_info,
                                       const Tegra::DMA::BufferOperand& buffer_operand,
                                       const Tegra::DMA::ImageOperand& image_operand) {
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    const auto image_id = texture_cache.DmaImageId(image_operand, IS_IMAGE_UPLOAD);
    if (image_id == VideoCommon::NULL_IMAGE_ID) {
        return false;
    }
    const u32 buffer_size = static_cast<u32>(buffer_operand.pitch * buffer_operand.height);
    static constexpr auto sync_info = VideoCommon::ObtainBufferSynchronize::FullSynchronize;
    const auto post_op = IS_IMAGE_UPLOAD ? VideoCommon::ObtainBufferOperation::DoNothing
                                         : VideoCommon::ObtainBufferOperation::MarkAsWritten;
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(buffer_operand.address, buffer_size, sync_info, post_op);

    const auto [image, copy] = texture_cache.DmaBufferImageCopy(
        copy_info, buffer_operand, image_operand, image_id, IS_IMAGE_UPLOAD);
    const std::span copy_span{&copy, 1};

    if constexpr (IS_IMAGE_UPLOAD) {
        texture_cache.PrepareImage(image_id, true, false);
        image->UploadMemory(buffer->Handle(), offset, copy_span);
    } else {
        if (offset % VideoCore::Surface::BytesPerBlock(image->info.format)) {
            return false;
        }
        texture_cache.DownloadImageIntoBuffer(image, buffer->Handle(), offset, copy_span,
                                              buffer_operand.address, buffer_size);
    }
    return true;
}

bool AccelerateDMA::ImageToBuffer(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::ImageOperand& image_operand,
                                  const Tegra::DMA::BufferOperand& buffer_operand) {
    return DmaBufferImageCopy<false>(copy_info, buffer_operand, image_operand);
}

bool AccelerateDMA::BufferToImage(const Tegra::DMA::ImageCopy& copy_info,
                                  const Tegra::DMA::BufferOperand& buffer_operand,
                                  const Tegra::DMA::ImageOperand& image_operand) {
    return DmaBufferImageCopy<true>(copy_info, buffer_operand, image_operand);
}

RasterizerMetal::RasterizerMetal(Tegra::GPU& gpu_, Tegra::MaxwellDeviceMemoryManager& device_memory,
                                 const Device& device_, Scheduler& scheduler_)
    : gpu{gpu_}, device{device_}, scheduler{scheduler_}, staging_pool{device_, scheduler_},
      texture_cache_runtime{device_, scheduler_, staging_pool},
      texture_cache{texture_cache_runtime, device_memory},
      buffer_cache_runtime{device_, scheduler_, staging_pool},
      buffer_cache{device_memory, buffer_cache_runtime}, pipeline_cache{device_memory, device_},
      accelerate_dma{buffer_cache, texture_cache},
      fence_manager{*this, gpu_, texture_cache, buffer_cache, query_cache, scheduler_} {}

RasterizerMetal::~RasterizerMetal() = default;

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    ++draws;
    // Milestone 3 step: the pipeline is translated and built, the draw is not encoded yet.
    GraphicsPipeline* const pipeline = pipeline_cache.CurrentGraphicsPipeline();
    if (pipeline != nullptr && pipeline->BuildFunctions(device)) {
        ++draws_with_pipeline;
    }
}

void RasterizerMetal::DrawTexture() {
    ++draws;
}

void RasterizerMetal::Clear(u32 layer_count) {
    if (!maxwell3d) {
        return;
    }
    gpu_memory->FlushCaching();
    const auto& regs = maxwell3d->regs;
    const bool any_color = regs.clear_surface.R || regs.clear_surface.G ||
                           regs.clear_surface.B || regs.clear_surface.A;
    const bool all_color = regs.clear_surface.R && regs.clear_surface.G &&
                           regs.clear_surface.B && regs.clear_surface.A;
    if (!any_color && !regs.clear_surface.Z && !regs.clear_surface.S) {
        return;
    }
    if (any_color && !all_color) {
        // A masked clear is a draw on Metal (milestone 3).
        static std::atomic_bool warned{false};
        if (!warned.exchange(true)) {
            LOG_WARNING(Render_Metal, "Metal: clears of some colour channels are not done yet");
        }
    }
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UpdateRenderTargets(true);
    const Framebuffer* const framebuffer = texture_cache.GetFramebuffer();
    if (framebuffer == nullptr) {
        return;
    }
    ClearPass pass{};
    pass.width = framebuffer->Width();
    pass.height = framebuffer->Height();
    const u32 rt = regs.clear_surface.RT;
    if (all_color && rt < NUM_RT && framebuffer->Colors()[rt].texture != nullptr) {
        pass.color[rt] = framebuffer->Colors()[rt];
        pass.clear_color[rt] = true;
        for (size_t component = 0; component < 4; ++component) {
            pass.color_value[rt][component] = regs.clear_color[component];
        }
    }
    if (regs.clear_surface.Z && framebuffer->HasDepth()) {
        pass.depth = framebuffer->Depth();
        pass.clear_depth = true;
        pass.depth_value = regs.clear_depth;
    }
    if (regs.clear_surface.S && framebuffer->HasStencil()) {
        pass.stencil = framebuffer->Depth();
        pass.clear_stencil = true;
        pass.stencil_value = static_cast<u32>(regs.clear_stencil);
    }
    scheduler.Clear(pass);
    ++clears;
}

void RasterizerMetal::DispatchCompute() {
    ++dispatches;
    ComputePipeline* const pipeline = pipeline_cache.CurrentComputePipeline();
    if (pipeline != nullptr && pipeline->State(device) != nullptr) {
        ++dispatches_with_pipeline;
    }
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType) {}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32) {
    QueryFallback(gpu_addr, type, flags, payload);
}

void RasterizerMetal::QueryFallback(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                                    VideoCommon::QueryPropertiesFlags flags, u32 payload) {
    if (!gpu_memory) {
        return;
    }
    // Host queries come with milestone 4: counters read as 1 (something passed), as OpenGL's
    // fallback does.
    if (type != VideoCommon::QueryType::Payload) {
        payload = 1U;
    }
    std::function<void()> func([this, gpu_addr, flags, memory_manager = gpu_memory, payload] {
        if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
            memory_manager->Write<u64>(gpu_addr + 8, gpu.GetTicks());
            memory_manager->Write<u64>(gpu_addr, static_cast<u64>(payload));
        } else {
            memory_manager->Write<u32>(gpu_addr, payload);
        }
    });
    if (True(flags & VideoCommon::QueryPropertiesFlags::IsAFence)) {
        SignalFence(std::move(func));
        return;
    }
    func();
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {
    std::scoped_lock lock{buffer_cache.mutex};
    buffer_cache.BindGraphicsUniformBuffer(stage, index, gpu_addr, size);
}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    buffer_cache.DisableGraphicsUniformBuffer(stage, index);
}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.DownloadMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.DownloadMemory(addr, size);
    }
}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.IsRegionGpuModified(addr, size)) {
            return true;
        }
    }
    if (!Settings::IsGPULevelHigh()) {
        return false;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        return texture_cache.IsRegionGpuModified(addr, size);
    }
    return false;
}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        if (const auto area = texture_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (const auto area = buffer_cache.GetFlushArea(addr, size)) {
            return *area;
        }
    }
    return VideoCore::RasterizerDownloadArea{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType which) {
    if (addr == 0 || size == 0) {
        return;
    }
    if (True(which & VideoCommon::CacheType::TextureCache)) {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::BufferCache)) {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    if (True(which & VideoCommon::CacheType::ShaderCache)) {
        pipeline_cache.InvalidateRegion(addr, size);
    }
}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{buffer_cache.mutex};
        if (buffer_cache.OnCPUWrite(addr, size)) {
            return true;
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
    return false;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {
    if (addr == 0 || size == 0) {
        return;
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size);
    }
    pipeline_cache.InvalidateRegion(addr, size);
}

void RasterizerMetal::InvalidateGPUCache() {
    gpu.InvalidateGPUCache();
}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.UnmapMemory(addr, size);
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.WriteMemory(addr, size, false);
    }
    pipeline_cache.OnCacheInvalidation(addr, size);
}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {
    std::scoped_lock lock{texture_cache.mutex};
    texture_cache.UnmapGPUMemory(as_id, addr, size);
}

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    fence_manager.SignalFence(std::move(func));
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    fence_manager.SyncOperation(std::move(func));
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    fence_manager.SignalSyncPoint(value);
}

void RasterizerMetal::SignalReference() {
    fence_manager.SignalOrdering();
}

void RasterizerMetal::ReleaseFences(bool force) {
    fence_manager.WaitPendingFences(force);
}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size,
                                               VideoCommon::CacheType which) {
    if (Settings::IsGPULevelHigh()) {
        FlushRegion(addr, size, which);
    }
    InvalidateRegion(addr, size, which);
}

void RasterizerMetal::WaitForIdle() {
    SignalReference();
}

void RasterizerMetal::FragmentBarrier() {}

void RasterizerMetal::TiledCacheBarrier() {}

void RasterizerMetal::FlushCommands() {
    if (scheduler.HasPendingWork()) {
        scheduler.Flush();
    }
}

void RasterizerMetal::TickFrame() {
    fence_manager.TickFrame();
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.TickFrame();
    }
    {
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.TickFrame();
    }
    staging_pool.TickFrame();
}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.BlitImage(dst, src, copy_config);
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {
    const auto cpu_addr = gpu_memory->GpuToCpuAddress(address);
    if (!cpu_addr) {
        gpu_memory->WriteBlock(address, memory.data(), copy_size);
        return;
    }
    gpu_memory->WriteBlockUnsafe(address, memory.data(), copy_size);
    {
        std::unique_lock lock{buffer_cache.mutex};
        if (!buffer_cache.InlineMemory(*cpu_addr, copy_size, memory)) {
            buffer_cache.WriteMemory(*cpu_addr, copy_size);
        }
    }
    {
        std::scoped_lock lock{texture_cache.mutex};
        texture_cache.WriteMemory(*cpu_addr, copy_size);
    }
    pipeline_cache.InvalidateRegion(*cpu_addr, copy_size);
}

void RasterizerMetal::LoadDiskResources(u64, std::stop_token,
                                        const VideoCore::DiskResourceLoadCallback&) {}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.CreateChannel(channel);
    buffer_cache.CreateChannel(channel);
    pipeline_cache.CreateChannel(channel);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    const s32 channel_id = channel.bind_id;
    BindToChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.BindToChannel(channel_id);
    buffer_cache.BindToChannel(channel_id);
    pipeline_cache.BindToChannel(channel_id);
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    texture_cache.EraseChannel(channel_id);
    buffer_cache.EraseChannel(channel_id);
    pipeline_cache.EraseChannel(channel_id);
}

std::optional<DisplayTexture> RasterizerMetal::AccelerateDisplay(
    const Tegra::FramebufferConfig& config, DAddr framebuffer_addr) {
    if (framebuffer_addr == 0) {
        return std::nullopt;
    }
    std::scoped_lock lock{texture_cache.mutex};
    const auto [image_view, scaled] =
        texture_cache.TryFindFramebufferImageView(config, framebuffer_addr);
    if (image_view == nullptr) {
        return std::nullopt;
    }
    const auto& resolution = Settings::values.resolution_info;
    DisplayTexture texture{
        .texture = image_view->Handle(Shader::TextureType::Color2D),
        .width = image_view->size.width,
        .height = image_view->size.height,
    };
    texture.scaled_width = scaled ? resolution.ScaleUp(texture.width) : texture.width;
    texture.scaled_height = scaled ? resolution.ScaleUp(texture.height) : texture.height;
    if (texture.texture == nullptr) {
        return std::nullopt;
    }
    return texture;
}

} // namespace Metal
