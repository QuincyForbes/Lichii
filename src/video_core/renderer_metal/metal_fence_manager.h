// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

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
    [[nodiscard]] bool HasPendingFenceWork() const noexcept {
        return false;
    }
    void PopAsyncFlushes() {}
};

/// A guest fence: the scheduler tick of the command buffer that ends with it.
class InnerFence : public VideoCommon::FenceBase {
public:
    explicit InnerFence(Scheduler& scheduler, bool is_stubbed);

    void Queue();
    bool IsSignaled() const;
    void Wait();

private:
    Scheduler& scheduler;
    u64 tick{};
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

    void WaitPendingFences(bool force);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;
    void SubmitDeferredWork() override;

private:
    Scheduler& scheduler;
    bool in_nonblocking_release_wait{};
};

} // namespace Metal
