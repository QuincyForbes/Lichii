// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <climits>
#include <vector>

#include "common/common_types.h"

#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct StagingBufferRef {
    VkBuffer buffer;
    VkDeviceAddress device_address;
    VkDeviceSize offset;
    std::span<u8> mapped_span;
    MemoryUsage usage;
    u32 log2_level;
    u64 index;
};

class StagingBufferPool {
public:
    static constexpr size_t NUM_SYNCS = 16;

    explicit StagingBufferPool(const Device& device, MemoryAllocator& memory_allocator,
                               Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferRef Request(size_t size, MemoryUsage usage, bool deferred = false);
    StagingBufferRef RequestUploadBuffer(size_t size) {
        return Request(size, MemoryUsage::Upload);
    }
    StagingBufferRef GetUploadBuffer(size_t size) {
        return Request(size, MemoryUsage::Upload);
    }
    void FreeDeferred(StagingBufferRef& ref);

    [[nodiscard]] VkBuffer StreamBuf() const noexcept {
        return *stream_buffer;
    }

    [[nodiscard]] bool IsRebarEnabled() const noexcept {
        return use_rebar;
    }

    void TickFrame();

    /// Bytes held by the pooled staging buffers (the stream buffer not included).
    [[nodiscard]] u64 CachedBytes() const;

    /// Called between draws and dispatches, when every staging buffer handed out has its copies
    /// recorded: submits the command buffer once the uploads made for it pass the submit mark,
    /// and waits for the GPU while the pooled upload buffers pass the cap. True if it submitted.
    bool BoundUploads();

    /// While a draw's vertex and index buffers are recorded and the draw is not, BoundUploads
    /// does not submit: the draw would land in a command buffer without them.
    void HoldSubmits(bool hold) noexcept {
        submits_held = hold;
    }

    [[nodiscard]] VkDeviceSize StreamBufferSize() const noexcept {
        return stream_buffer_size;
    }

private:
    struct StreamBufferCommit {
        size_t upper_bound;
        u64 tick;
    };

    struct StagingBuffer {
        vk::Buffer buffer;
        VkDeviceAddress device_address;
        std::span<u8> mapped_span;
        MemoryUsage usage;
        u32 log2_level;
        u64 index;
        u64 tick = 0;
        bool deferred{};

        StagingBufferRef Ref() const noexcept {
            return {
                .buffer = *buffer,
                .device_address = device_address,
                .offset = 0,
                .mapped_span = mapped_span,
                .usage = usage,
                .log2_level = log2_level,
                .index = index,
            };
        }
    };

    struct StagingBuffers {
        std::vector<StagingBuffer> entries;
        size_t delete_index = 0;
        size_t iterate_index = 0;
        u64 known_busy_gpu_tick = ~u64{0};
    };

    static constexpr size_t NUM_LEVELS = sizeof(size_t) * CHAR_BIT;
    using StagingBuffersCache = std::array<StagingBuffers, NUM_LEVELS>;

    struct ChunkPointer {
        std::atomic<size_t> offset{0};
        std::atomic<size_t> limit{0};
        mutable std::atomic<u64> blocked_tick{0};
    };

    StagingBufferRef GetStreamBuffer(size_t size);

    bool AreRegionsActive(size_t region_begin, size_t region_end) const;

    StagingBufferRef GetStagingBuffer(size_t size, MemoryUsage usage, bool deferred = false);

    std::optional<StagingBufferRef> TryGetReservedBuffer(size_t size, MemoryUsage usage,
                                                         bool deferred);

    StagingBufferRef CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred);

    StagingBuffersCache& GetCache(MemoryUsage usage);

    void ReleaseCache(MemoryUsage usage);

    void ReleaseLevel(StagingBuffersCache& cache, size_t log2);

    /// Frees every staging buffer the GPU is done with.
    void ReleaseIdleBuffers();

    /// Frees idle upload and device-local buffers, largest first, until they hold at most `bytes`.
    void ReleaseIdleUploadsDownTo(u64 bytes);

    /// The oldest tick a pooled upload or device-local buffer waits for (0: none busy).
    [[nodiscard]] u64 OldestBusyUploadTick() const;

    /// Bytes held by the pooled upload and device-local buffers.
    [[nodiscard]] u64 UploadPoolBytes() const;

    /// Bytes a buffer of this level takes (MoltenVK pads small upload buffers).
    [[nodiscard]] u64 AllocationBytes(MemoryUsage usage, size_t log2) const noexcept;

    size_t Region(size_t iter) const noexcept {
        return iter / region_size;
    }

    const Device& device;
    MemoryAllocator& memory_allocator;
    Scheduler& scheduler;

    vk::Buffer stream_buffer;
    VkDeviceAddress stream_buffer_address{};
    std::span<u8> stream_pointer;
    VkDeviceSize stream_buffer_size;
    VkDeviceSize region_size;

    size_t iterator = 0;
    size_t used_iterator = 0;
    size_t free_iterator = 0;
    std::array<u64, NUM_SYNCS> sync_ticks{};

    mutable ChunkPointer current_chunk{};
    bool use_rebar = false;
    u32 seen_host_pressure{};
    bool low_memory{};         ///< Under the low-water mark at the last frame (iOS)
    u64 low_memory_waits{};    ///< GPU waits in CreateStagingBuffer since the last log line
    u64 frame_count{};

    u64 upload_submit_bytes{}; ///< Uploads per command buffer before BoundUploads submits (0: off)
    u64 upload_cap_bytes{};    ///< Pooled upload buffers before BoundUploads waits for the GPU
    u64 configured_submit_bytes{}; ///< The marks set at boot, which room to spare raises (iOS)
    u64 configured_cap_bytes{};
    u64 tick_upload_bytes{};   ///< Pooled upload bytes handed out for upload_tick
    u64 upload_tick{};
    u64 early_submits{};       ///< Since the last log line
    u64 early_submit_bytes{};
    u64 cap_waits{};
    u64 low_memory_cap_waits{}; ///< Early submits that found under twice the low-water mark left
    u64 upload_pool_peak{};
    bool submits_held{};
    u64 held_submits{};        ///< Early submits put off inside a draw since the last log line

    StagingBuffersCache device_local_cache;
    StagingBuffersCache upload_cache;
    StagingBuffersCache download_cache;

    size_t current_delete_level = 0;
    u64 buffer_index = 0;
    u64 unique_ids{};
};

} // namespace Vulkan
