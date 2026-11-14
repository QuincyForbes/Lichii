// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
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
class StagingBufferPool {
public:
    explicit StagingBufferPool(const Device& device, Scheduler& scheduler);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// A deferred buffer stays taken until FreeDeferred, whatever the GPU does.
    [[nodiscard]] StagingBufferRef Request(size_t size, bool deferred = false);

    void FreeDeferred(StagingBufferRef& ref);

    /// A slice of one ring buffer for data used by the command buffer being recorded (streamed
    /// uniforms): no rounding to powers of two, no padding past `size`.
    [[nodiscard]] StagingBufferRef Stream(size_t size, size_t alignment);

    /// Frees buffers unused for a while.
    void TickFrame();

    [[nodiscard]] u64 HeldBytes() const noexcept {
        return held_bytes;
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

    static constexpr u64 StreamSize = u64{32} << 20;
    static constexpr u64 StreamRegions = 16;
    static constexpr u64 StreamRegionSize = StreamSize / StreamRegions;

    const Device& device;
    Scheduler& scheduler;
    std::vector<Entry> entries;
    Object stream;
    u8* stream_contents{};
    u64 stream_offset{};
    /// The last tick that used each region of the ring.
    std::array<u64, StreamRegions> stream_ticks{};
    u64 frame{};
    u64 held_bytes{};
};

} // namespace Metal
