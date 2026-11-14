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

    static constexpr bool HAS_ASYNC_CHECK = false;
};

using GenericFenceManager = VideoCommon::FenceManager<FenceManagerParams>;

class FenceManager final : public GenericFenceManager {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
                          TextureCache& texture_cache, BufferCache& buffer_cache,
                          QueryCache& query_cache, Scheduler& scheduler);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;

private:
    Scheduler& scheduler;
};

} // namespace Metal
