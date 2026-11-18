// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_wait_site.h"
#include "video_core/host1x/syncpoint_manager.h"
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

void FenceGate::Open() {
    {
        std::scoped_lock lock{mutex};
        open = true;
    }
    cv.notify_all();
}

bool FenceGate::IsOpen() const {
    std::scoped_lock lock{mutex};
    return open;
}

void FenceGate::WaitOpen() {
    std::unique_lock lock{mutex};
    cv.wait(lock, [this] { return open; });
}

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

void InnerFence::Queue() {
    if (!is_stubbed) {
        tick = scheduler.FenceTick();
    }
}

bool InnerFence::IsSignaled() const {
    if (handoff && !handoff->IsOpen()) {
        return false;
    }
    return is_stubbed || scheduler.IsFree(tick);
}

void InnerFence::Wait() {
    if (handoff) {
        // The GPU thread still uses the scheduler: this fence's texture downloads record on it.
        handoff->WaitOpen();
    }
    if (is_stubbed) {
        return;
    }
    if (FenceManagerParams::AsyncCheck()) {
        // The fencing thread: waits for the completion handlers, never commits (Scheduler::Wait
        // walks and commits the GPU thread's command buffers). FenceTick left the tick committed
        // or pending for the GPU thread's next commit point, which WaitCompleted asks for.
        scheduler.WaitAnyThread(tick, CommitReason::Fence);
    } else {
        scheduler.Wait(tick, CommitReason::Fence);
    }
}

void InnerFence::SetHandoff(std::shared_ptr<FenceGate> gate) {
    handoff = std::move(gate);
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

bool FenceManager::TextureDownloadsPending() {
    std::scoped_lock lock{texture_cache.mutex};
    return texture_cache.HasAsyncFlushWork();
}

template <typename Signal>
void FenceManager::SignalWithTextureDownloads(Signal&& signal) {
    if (!AsyncFences() || !TextureDownloadsPending()) {
        signal();
        return;
    }
    // The texture cache downloads synchronously (IMPLEMENTS_ASYNC_DOWNLOADS false): the fencing
    // thread pops the fence's downloads by recording copies and Scheduler::Finish. The fence is
    // held (IsSignaled false, Wait blocks) until this thread is done with the scheduler, and this
    // thread then waits for an operation the fencing thread runs after the fence's pops.
    const auto handoff = std::make_shared<FenceGate>();
    const auto popped = std::make_shared<FenceGate>();
    SyncOperation([popped] { popped->Open(); });
    next_handoff = handoff;
    signal();
    const bool handed = next_handoff == nullptr;
    next_handoff.reset();
    if (!handed) {
        // No fence took the downloads; the operation goes with a later fence, nobody waits.
        return;
    }
    // Nothing commits while this thread waits: every tick the fencing thread waits for first
    // (earlier fences', RequestCommit's) is committed now. FenceTick leaves them pending only
    // as flush_pending, which FlushDeferred commits.
    scheduler.FlushDeferred();
    handoff->Open();
    VideoCommon::BackWaitScope wait_scope{"Metal::FenceManager (texture downloads popped on the "
                                          "fencing thread)"};
    popped->WaitOpen();
}

void FenceManager::SignalFence(std::function<void()>&& func, bool force_completion) {
    SignalWithTextureDownloads(
        [&] { GenericFenceManager::SignalFence(std::move(func), force_completion); });
}

void FenceManager::SignalSyncPoint(u32 value) {
    if (value >= Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) {
        // Ignored by the base without a fence.
        GenericFenceManager::SignalSyncPoint(value);
        return;
    }
    SignalWithTextureDownloads([&] { GenericFenceManager::SignalSyncPoint(value); });
}

void FenceManager::WaitPendingFences(bool force) {
    if (AsyncFences()) {
        if (force && TextureDownloadsPending()) {
            // The base's own fence would have the fencing thread pop the downloads while this
            // thread still submits (SubmitDeferredWork): a fence of their own first.
            SignalFence([] {});
        }
        if (!force && !VideoCommon::GpuSplit::OnBackThread() &&
            VideoCommon::GpuSplit::OwnsScheduler()) {
            // Without the split nothing runs an idle hook: work left pending for the fencing
            // thread (FenceTick with EDEN_NX2_METAL_FENCE_BATCH=2, RequestCommit) is committed
            // at each command list end.
            scheduler.FlushDeferred();
        }
    }
    GenericFenceManager::WaitPendingFences(force);
}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    if (next_handoff && !fence->IsStubbed()) {
        fence->SetHandoff(std::move(next_handoff));
        next_handoff.reset();
    }
    fence->Queue();
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

void FenceManager::SubmitDeferredWork() {
    // Only the split's back leaves fence work pending (Scheduler::FenceTick) and commits; with
    // asynchronous fences, the GPU thread without the split too (RequestCommit).
    if (VideoCommon::GpuSplit::OnBackThread() ||
        (AsyncFences() && VideoCommon::GpuSplit::OwnsScheduler())) {
        scheduler.FlushDeferred();
    }
}

} // namespace Metal
