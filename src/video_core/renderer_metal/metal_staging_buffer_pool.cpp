// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <limits>

#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"

namespace Metal {

namespace {
/// Frames a free buffer is kept for the next request of its size.
constexpr u64 KeepFrames = 300;
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Request(size_t size, bool deferred) {
    const u64 rounded = std::bit_ceil((std::max)(size, size_t{4096}));
    const u64 tick = scheduler.CurrentTick();
    for (size_t index = 0; index < entries.size(); ++index) {
        Entry& entry = entries[index];
        if (entry.size != rounded || entry.deferred || !scheduler.IsFree(entry.tick)) {
            continue;
        }
        entry.tick = tick;
        entry.last_frame = frame;
        entry.deferred = deferred;
        return StagingBufferRef{
            .buffer = entry.buffer.Get(),
            .offset = 0,
            .mapped_span = std::span<u8>(entry.contents, size),
            .index = index,
        };
    }
    u8* contents = nullptr;
    Object buffer = CreateBuffer(device, rounded, true, &contents);
    ASSERT_MSG(buffer && contents != nullptr, "Metal staging buffer of {} bytes failed", rounded);
    held_bytes += rounded;
    entries.push_back(Entry{
        .buffer = std::move(buffer),
        .contents = contents,
        .size = rounded,
        .tick = tick,
        .last_frame = frame,
        .deferred = deferred,
    });
    const Entry& entry = entries.back();
    return StagingBufferRef{
        .buffer = entry.buffer.Get(),
        .offset = 0,
        .mapped_span = std::span<u8>(entry.contents, size),
        .index = entries.size() - 1,
    };
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    if (ref.index < entries.size()) {
        Entry& entry = entries[ref.index];
        entry.deferred = false;
        entry.tick = scheduler.CurrentTick();
    }
}

StagingBufferRef StagingBufferPool::Stream(size_t size, size_t alignment) {
    if (size > StreamRegionSize) {
        return Request(size);
    }
    if (!stream) {
        stream = CreateBuffer(device, StreamSize, true, &stream_contents);
        ASSERT_MSG(stream && stream_contents != nullptr, "Metal stream buffer failed");
    }
    u64 offset = Common::AlignUp(stream_offset, (std::max)(alignment, size_t{16}));
    const u64 region = offset / StreamRegionSize;
    // A slice never spans two regions, so one tick per region covers it.
    if (region >= StreamRegions || (offset + size - 1) / StreamRegionSize != region) {
        offset = region + 1 >= StreamRegions ? 0 : (region + 1) * StreamRegionSize;
    }
    const u64 first = offset / StreamRegionSize;
    const u64 tick = scheduler.CurrentTick();
    if (stream_ticks[first] != tick && offset % StreamRegionSize == 0) {
        // Entering a region: the GPU must be done with what it held a lap ago.
        scheduler.Wait(stream_ticks[first]);
    }
    stream_ticks[first] = tick;
    stream_offset = offset + size;
    return StagingBufferRef{
        .buffer = stream.Get(),
        .offset = offset,
        .mapped_span = std::span<u8>(stream_contents + offset, size),
        .index = (std::numeric_limits<size_t>::max)(),
    };
}

void StagingBufferPool::TickFrame() {
    ++frame;
    // Only from the back, so the indices handed out stay valid.
    while (!entries.empty()) {
        const Entry& entry = entries.back();
        if (entry.deferred || frame - entry.last_frame < KeepFrames ||
            !scheduler.IsFree(entry.tick)) {
            break;
        }
        held_bytes -= entry.size;
        entries.pop_back();
    }
}

} // namespace Metal
