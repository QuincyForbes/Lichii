// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_metal/metal_fence_manager.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

bool FenceManagerParams::AsyncCheck() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ASYNC_FENCES");
        const bool on = value == nullptr || value[0] != '0';
        LOG_INFO(Render_Metal, "Metal: fences released on the fencing thread: {} "
                               "(EDEN_NX2_METAL_ASYNC_FENCES)",
                 on ? "on" : "off");
        return on;
    }();
    return enabled;
}

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
    if (is_stubbed) {
        return;
    }
    if (FenceManagerParams::AsyncCheck()) {
        // The fencing thread: waits for the completion handlers, never commits (Scheduler::Wait
        // walks and commits the GPU thread's command buffers). FenceTick left the tick committed
        // or pending for the GPU thread's next commit point, which WaitCompleted asks for after
        // the fence batching window.
        scheduler.WaitAnyThread(tick, CommitReason::Fence, true);
    } else {
        scheduler.Wait(tick, CommitReason::Fence);
    }
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {
    Scheduler::SetSplitHooks({
        .back_running = [] { return VideoCommon::GpuSplit::Get().BackRunning(); },
        .on_back_thread = [] { return VideoCommon::GpuSplit::OnBackThread(); },
        .owns_scheduler = [] { return VideoCommon::GpuSplit::OwnsScheduler(); },
        .request_idle_hook = [] { VideoCommon::GpuSplit::Get().RequestIdleHook(); },
    });
}

FenceManager::~FenceManager() {
    if (AsyncFences()) {
        // The base joins the fencing thread next; nothing commits pending work any more.
        scheduler.AbandonWaits();
    }
}

void FenceManager::WaitPendingFences(bool force) {
    if (AsyncFences() && !force && !VideoCommon::GpuSplit::OnBackThread() &&
        VideoCommon::GpuSplit::OwnsScheduler()) {
        // Without the split nothing runs an idle hook: work left pending for the fencing
        // thread (FenceTick with EDEN_NX2_METAL_FENCE_BATCH=2, RequestCommit) is committed
        // at each command list end.
        scheduler.FlushDeferred();
    }
    static const bool force_release_submit = [] {
        const char* const value = std::getenv("EDEN_NX2_RELEASE_SUBMIT");
        return value != nullptr && value[0] == '1';
    }();
    const bool skip_release_submit = !force_release_submit && force && AsyncFences() &&
                                     VideoCommon::GpuSplit::OnBackThread() &&
                                     !VideoCore::Nx2Debug::DelayedFences();
    if (skip_release_submit) {
        in_nonblocking_release_wait = true;
    }
    GenericFenceManager::WaitPendingFences(force);
    in_nonblocking_release_wait = false;
}

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
    if (in_nonblocking_release_wait && !VideoCore::Nx2Debug::DelayedFences()) {
        return;
    }
    // Only the split's back leaves fence work pending (Scheduler::FenceTick) and commits; with
    // asynchronous fences, the GPU thread without the split too (RequestCommit).
    if (VideoCommon::GpuSplit::OnBackThread() ||
        (AsyncFences() && VideoCommon::GpuSplit::OwnsScheduler())) {
        scheduler.FlushDeferred();
    }
}

} // namespace Metal
