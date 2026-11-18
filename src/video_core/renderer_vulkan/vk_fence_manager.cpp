// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <thread>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue() {
    if (is_stubbed) {
        return;
    }
    if (!scheduler.HasPendingCommands() && scheduler.CurrentTick() > 1) {
        wait_tick = scheduler.CurrentTick() - 1;
        return;
    }
    // Get the current tick so we can wait for it
    wait_tick = scheduler.CurrentTick();
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    } else {
        queued = std::chrono::steady_clock::now();
    }
}

namespace {
// EDEN_NX2_FENCE_PATIENCE_US=<us>: the fencing thread lets a deferred fence submission wait up
// to that long after the fence was made before it asks the back for it (0 = asks at once). The
// submissions the back makes meanwhile take the fence along.
std::chrono::microseconds FencePatience() {
    static const std::chrono::microseconds patience{[] {
        const char* const value = std::getenv("EDEN_NX2_FENCE_PATIENCE_US");
        return value != nullptr ? (std::max)(std::atoll(value), 0LL) : 0LL;
    }()};
    return patience;
}
} // Anonymous namespace

bool InnerFence::IsSignaled() const {
    if (is_stubbed) {
        return true;
    }
    return scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    if (is_stubbed) {
        return;
    }
    // A deferred fence flush (Scheduler::DeferFlush) may not be submitted yet. Only the
    // split's back may submit; other threads (the fencing thread) ask it to and wait. If the back
    // has stopped, nothing else uses the scheduler and Wait below submits it.
    if (const auto patience = FencePatience();
        patience.count() != 0 && !scheduler.IsSubmitted(wait_tick) &&
        VideoCommon::GpuSplit::Get().BackRunning() && !VideoCommon::GpuSplit::OnBackThread()) {
        const auto deadline = queued + patience;
        while (!scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().BackRunning() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    if (!scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread() && scheduler.WaitSubmitted(wait_tick)) {
        scheduler.GetMasterSemaphore().Wait(wait_tick);
        return;
    }
    scheduler.Wait(wait_tick);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, const Device& device_, Scheduler& scheduler_)
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
    // Only the split's back defers (Scheduler::DeferFlush) and submits.
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.FlushDeferred();
    }
}

} // namespace Vulkan
