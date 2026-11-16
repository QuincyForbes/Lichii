// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/gpu_split.h"
#include "video_core/renderer_metal/metal_fence_manager.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

void InnerFence::Queue() {
    if (!is_stubbed) {
        tick = scheduler.FenceTick();
    }
}

bool InnerFence::IsSignaled() const {
    return is_stubbed || scheduler.IsFree(tick);
}

void InnerFence::Wait() {
    if (!is_stubbed) {
        scheduler.Wait(tick);
    }
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

void FenceManager::SubmitDeferredWork() {
    // Only the split's back leaves fence work pending (Scheduler::FenceTick) and commits.
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.FlushDeferred();
    }
}

} // namespace Metal
