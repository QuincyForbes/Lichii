// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>

#include "video_core/fence_manager.h"
#include "video_core/renderer_metal/metal_buffer_cache.h"
#include "video_core/renderer_metal/metal_texture_cache.h"

namespace Metal {

class Scheduler;

/// Host queries are not recorded yet (milestone 4): the rasterizer writes query results at
/// once, so the fence manager has nothing of theirs to wait for.
class QueryCache {
public:
    void CommitAsyncFlushes() {}
    bool HasUncommittedFlushes() const {
        return false;
    }
    bool ShouldWaitAsyncFlushes() const {
        return false;
    }
    void PopAsyncFlushes() {}
};

/// A one-shot event between the GPU thread and the fencing thread.
class FenceGate {
public:
    void Open();
    bool IsOpen() const;
    void WaitOpen();

private:
    mutable std::mutex mutex;
    std::condition_variable cv;
    bool open{};
};

/// A guest fence: the scheduler tick of the command buffer that ends with it.
class InnerFence : public VideoCommon::FenceBase {
public:
    explicit InnerFence(Scheduler& scheduler, bool is_stubbed);

    void Queue();
    bool IsSignaled() const;
    void Wait();

    /// Not signalled before `gate` opens (FenceManager::SignalWithTextureDownloads).
    void SetHandoff(std::shared_ptr<FenceGate> gate);

private:
    Scheduler& scheduler;
    u64 tick{};
    std::shared_ptr<FenceGate> handoff;
};

using Fence = std::shared_ptr<InnerFence>;

struct FenceManagerParams {
    using FenceType = Fence;
    using BufferCacheType = BufferCache;
    using TextureCacheType = TextureCache;
    using QueryCacheType = QueryCache;

    /// The fencing thread waits for fences and runs their operations (Vulkan's way); the GPU
    /// thread no longer waits for the host GPU at each command list end.
    static constexpr bool HAS_ASYNC_CHECK = true;
    /// EDEN_NX2_METAL_ASYNC_FENCES=0: fences are released on the GPU thread as before (waiting
    /// for the host GPU at command list ends). Read once.
    static bool AsyncCheck();
};

using GenericFenceManager = VideoCommon::FenceManager<FenceManagerParams>;

class FenceManager final : public GenericFenceManager {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
                          TextureCache& texture_cache, BufferCache& buffer_cache,
                          QueryCache& query_cache, Scheduler& scheduler);
    ~FenceManager() override;

    // Hide the base's: with asynchronous fences, a fence carrying texture downloads has the
    // fencing thread record and wait for them on the scheduler (the texture cache's synchronous
    // downloads), which only one thread may use at a time. The GPU thread hands the scheduler
    // over and waits for those downloads (SignalWithTextureDownloads).
    void SignalFence(std::function<void()>&& func, bool force_completion = false);
    void SignalSyncPoint(u32 value);
    void WaitPendingFences(bool force);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;
    void SubmitDeferredWork() override;

private:
    /// Whether the texture cache has downloads the next fence would commit.
    bool TextureDownloadsPending();

    template <typename Signal>
    void SignalWithTextureDownloads(Signal&& signal);

    Scheduler& scheduler;
    /// GPU thread: given to the next fence queued (SignalWithTextureDownloads).
    std::shared_ptr<FenceGate> next_handoff;
};

} // namespace Metal
