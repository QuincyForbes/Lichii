// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

class Device;
class Scheduler;

struct StagingBufferRef {
    BufferHandle buffer{};
    u64 offset{};
    std::span<u8> mapped_span;
    size_t index{};
};

/// Shared-storage buffers for uploads and downloads, reused once the GPU is past the command
/// buffer that last used them. Sizes are rounded to powers of two.
/// EDEN_NX2_METAL_STAGING_BOUND: the buffers one command buffer takes are bounded by committing
/// it early (WantsFlush), all of them by waiting for the GPU (OverCap), with Vulkan's marks
/// (EDEN_NX2_STAGING_SUBMIT_MIB, EDEN_NX2_STAGING_CAP_MIB).
class StagingBufferPool {
public:
    explicit StagingBufferPool(const Device& device, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// A deferred buffer stays taken until FreeDeferred, whatever the GPU does.
    [[nodiscard]] StagingBufferRef Request(size_t size, bool deferred = false,
                                           bool host_to_device = false);

    void FreeDeferred(StagingBufferRef& ref);

    /// A slice of one ring buffer for GPU scratch or CPU-readable data used by the command buffer
    /// being recorded: no rounding to powers of two, no padding past `size`. A slice larger than
    /// a sixteenth of the ring is a buffer of its own (Request).
    [[nodiscard]] StagingBufferRef Stream(size_t size, size_t alignment);

    /// A slice of the write-combined, hazard-untracked CPU upload ring (append-only CPU writes,
    /// never written by the GPU). Falls back to Request when larger than a region or on a full lap.
    [[nodiscard]] StagingBufferRef RequestUpload(size_t size, size_t alignment = 16);

    [[nodiscard]] u64 StreamSize() const noexcept {
        return stream_size;
    }

    /// Frees buffers unused for a while.
    void TickFrame();

    [[nodiscard]] u64 HeldBytes() const noexcept {
        return held_bytes;
    }

    /// Whether the command buffer being recorded took enough buffers to be committed now, so the
    /// GPU can give them back.
    [[nodiscard]] bool WantsFlush() const noexcept;

    /// Whether the pool holds more than its cap: the caller waits for the GPU, then Trim.
    [[nodiscard]] bool OverCap() const noexcept {
        return submit_bytes != 0 && held_bytes > cap_bytes;
    }

    /// Frees buffers the GPU is done with until half the cap is held.
    void Trim();

    /// Buffers freed since the last call.
    [[nodiscard]] u64 TakeFreed() noexcept {
        return std::exchange(freed, 0);
    }

private:
    struct Entry {
        Object buffer;
        u8* contents{};
        u64 size{};
        u64 tick{};
        u64 last_frame{};
        bool deferred{};
    };

    static constexpr u64 StreamRegions = 16;

    /// Takes entries[index] for the command buffer being recorded.
    StagingBufferRef Take(size_t index, size_t size, bool deferred);

    void Free(Entry& entry);

    const Device& device;
    Scheduler& scheduler;
    std::vector<Entry> entries;
    /// The ring's size: 32 MiB as Vulkan's on iOS, or EDEN_NX2_STREAM_BUFFER_MIB (4 to 1024).
    u64 stream_size{};
    u64 stream_region_size{};
    Object stream;
    u8* stream_contents{};
    u64 stream_offset{};
    /// The last tick that used each region of the ring.
    std::array<u64, StreamRegions> stream_ticks{};
    Object upload_stream;
    u8* upload_stream_contents{};
    u64 upload_stream_offset{};
    std::array<u64, StreamRegions> upload_stream_ticks{};
    u64 frame{};
    u64 held_bytes{};
    /// 0: not bounded.
    u64 submit_bytes{};
    u64 cap_bytes{};
    /// Bytes of the buffers taken in taken_tick.
    u64 taken_tick{};
    u64 taken_bytes{};
    u64 freed{};
};

} // namespace Metal
