// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/alignment.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_rasterizer.h"

namespace Metal {

bool AccelerateDMA::BufferCopy(GPUVAddr start_address, GPUVAddr end_address, u64 amount) {
    return true;
}

bool AccelerateDMA::BufferClear(GPUVAddr src_address, u64 amount, u32 value) {
    return true;
}

RasterizerMetal::RasterizerMetal(Tegra::GPU& gpu_) : gpu{gpu_} {}

RasterizerMetal::~RasterizerMetal() = default;

void RasterizerMetal::Draw(bool is_indexed, u32 instance_count) {
    ++draws;
}

void RasterizerMetal::DrawTexture() {
    ++draws;
}

void RasterizerMetal::Clear(u32 layer_count) {}

void RasterizerMetal::DispatchCompute() {
    ++dispatches;
}

void RasterizerMetal::ResetCounter(VideoCommon::QueryType type) {}

void RasterizerMetal::Query(GPUVAddr gpu_addr, VideoCommon::QueryType type,
                            VideoCommon::QueryPropertiesFlags flags, u32 payload, u32 subreport) {
    if (!gpu_memory) {
        return;
    }
    if (True(flags & VideoCommon::QueryPropertiesFlags::HasTimeout)) {
        gpu_memory->Write<u64>(gpu_addr + 8, gpu.GetTicks());
        gpu_memory->Write<u64>(gpu_addr, static_cast<u64>(payload));
    } else {
        gpu_memory->Write<u32>(gpu_addr, payload);
    }
}

void RasterizerMetal::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                                u32 size) {}

void RasterizerMetal::DisableGraphicsUniformBuffer(size_t stage, u32 index) {}

void RasterizerMetal::FlushAll() {}

void RasterizerMetal::FlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}

bool RasterizerMetal::MustFlushRegion(DAddr addr, u64 size, VideoCommon::CacheType) {
    return false;
}

void RasterizerMetal::InvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}

bool RasterizerMetal::OnCPUWrite(DAddr addr, u64 size) {
    return false;
}

void RasterizerMetal::OnCacheInvalidation(DAddr addr, u64 size) {}

VideoCore::RasterizerDownloadArea RasterizerMetal::GetFlushArea(DAddr addr, u64 size) {
    return VideoCore::RasterizerDownloadArea{
        .start_address = Common::AlignDown(addr, Core::DEVICE_PAGESIZE),
        .end_address = Common::AlignUp(addr + size, Core::DEVICE_PAGESIZE),
        .preemtive = true,
    };
}

void RasterizerMetal::InvalidateGPUCache() {}

void RasterizerMetal::UnmapMemory(DAddr addr, u64 size) {}

void RasterizerMetal::ModifyGPUMemory(size_t as_id, GPUVAddr addr, u64 size) {}

void RasterizerMetal::SignalFence(std::function<void()>&& func) {
    func();
}

void RasterizerMetal::SyncOperation(std::function<void()>&& func) {
    func();
}

void RasterizerMetal::SignalSyncPoint(u32 value) {
    auto& syncpoint_manager = gpu.Host1x().GetSyncpointManager();
    syncpoint_manager.IncrementGuest(value);
    syncpoint_manager.IncrementHost(value);
}

void RasterizerMetal::SignalReference() {}

void RasterizerMetal::ReleaseFences(bool) {}

void RasterizerMetal::FlushAndInvalidateRegion(DAddr addr, u64 size, VideoCommon::CacheType) {}

void RasterizerMetal::WaitForIdle() {}

void RasterizerMetal::FragmentBarrier() {}

void RasterizerMetal::TiledCacheBarrier() {}

void RasterizerMetal::FlushCommands() {}

void RasterizerMetal::TickFrame() {}

bool RasterizerMetal::AccelerateSurfaceCopy(const Tegra::Engines::Fermi2D::Surface& src,
                                            const Tegra::Engines::Fermi2D::Surface& dst,
                                            const Tegra::Engines::Fermi2D::Config& copy_config) {
    return true;
}

Tegra::Engines::AccelerateDMAInterface& RasterizerMetal::AccessAccelerateDMA() {
    return accelerate_dma;
}

void RasterizerMetal::AccelerateInlineToMemory(GPUVAddr address, size_t copy_size,
                                               std::span<const u8> memory) {}

void RasterizerMetal::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                        const VideoCore::DiskResourceLoadCallback& callback) {}

void RasterizerMetal::InitializeChannel(Tegra::Control::ChannelState& channel) {
    CreateChannel(channel);
}

void RasterizerMetal::BindChannel(Tegra::Control::ChannelState& channel) {
    BindToChannel(channel.bind_id);
}

void RasterizerMetal::ReleaseChannel(s32 channel_id) {
    EraseChannel(channel_id);
}

} // namespace Metal
