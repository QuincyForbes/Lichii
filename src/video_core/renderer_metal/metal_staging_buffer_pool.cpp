// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>

#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/host_budget.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"
#include "video_core/renderer_metal/metal_switches.h"

namespace Metal {

namespace {
/// Frames a free buffer is kept for the next request of its size.
constexpr u64 KeepFrames = 300;
/// The same while the pool holds more than a command buffer may take.
constexpr u64 PressureKeepFrames = 8;

u64 BoundBytes(const char* name, u64 default_mib) {
    if (const char* const env = std::getenv(name)) {
        const long mib = std::strtol(env, nullptr, 10);
        if (mib >= 0) {
            return static_cast<u64>(mib) << 20;
        }
    }
    return default_mib << 20;
}

/// EDEN_NX2_STREAM_BUFFER_MIB as Vulkan reads it (4 to 1024 MiB), else 32 MiB.
u64 StreamBufferBytes() {
    if (const char* const env = VulkanSwitch("EDEN_NX2_STREAM_BUFFER_MIB")) {
        if (const long mib = std::strtol(env, nullptr, 10); mib >= 4 && mib <= 1024) {
            return static_cast<u64>(mib) << 20;
        }
    }
    return u64{32} << 20;
}
} // Anonymous namespace

StagingBufferPool::StagingBufferPool(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, stream_size{StreamBufferBytes()},
      stream_region_size{stream_size / StreamRegions} {
    // EDEN_NX2_METAL_STAGING_BOUND=0: no bounds, and free buffers go only from the list's end.
    const char* const bound = std::getenv("EDEN_NX2_METAL_STAGING_BOUND");
    if (bound == nullptr || bound[0] != '0') {
        const bool phone = VideoCommon::HostBudget::Active();
        submit_bytes = BoundBytes("EDEN_NX2_STAGING_SUBMIT_MIB", phone ? 24 : 256);
        cap_bytes = (std::max)(BoundBytes("EDEN_NX2_STAGING_CAP_MIB", phone ? 96 : 1024),
                               submit_bytes * 2);
    }
}

StagingBufferPool::~StagingBufferPool() = default;

StagingBufferRef StagingBufferPool::Take(size_t index, size_t size, bool deferred) {
    Entry& entry = entries[index];
    const u64 tick = scheduler.CurrentTick();
    entry.tick = tick;
    entry.last_frame = frame;
    entry.deferred = deferred;
    // A deferred buffer comes back with FreeDeferred, not with a commit.
    if (!deferred) {
        if (taken_tick != tick) {
            taken_tick = tick;
            taken_bytes = 0;
        }
        taken_bytes += entry.size;
    }
    return StagingBufferRef{
        .buffer = entry.buffer.Get(),
        .offset = 0,
        .mapped_span = std::span<u8>(entry.contents, size),
        .index = index,
    };
}

StagingBufferRef StagingBufferPool::Request(size_t size, bool deferred) {
    const u64 rounded = std::bit_ceil((std::max)(size, size_t{4096}));
    size_t empty = entries.size();
    for (size_t index = 0; index < entries.size(); ++index) {
        const Entry& entry = entries[index];
        if (entry.size == 0) {
            empty = (std::min)(empty, index);
            continue;
        }
        if (entry.size != rounded || entry.deferred || !scheduler.IsFree(entry.tick)) {
            continue;
        }
        return Take(index, size, deferred);
    }
    u8* contents = nullptr;
    Object buffer = CreateBuffer(device, rounded, true, &contents);
    ASSERT_MSG(buffer && contents != nullptr, "Metal staging buffer of {} bytes failed", rounded);
    held_bytes += rounded;
    if (empty == entries.size()) {
        entries.emplace_back();
    }
    entries[empty] = Entry{
        .buffer = std::move(buffer),
        .contents = contents,
        .size = rounded,
        .tick = 0,
        .last_frame = frame,
        .deferred = false,
    };
    return Take(empty, size, deferred);
}

bool StagingBufferPool::WantsFlush() const noexcept {
    return submit_bytes != 0 && taken_tick == scheduler.CurrentTick() &&
           taken_bytes >= submit_bytes;
}

void StagingBufferPool::Free(Entry& entry) {
    held_bytes -= entry.size;
    entry = Entry{};
    ++freed;
}

void StagingBufferPool::Trim() {
    if (submit_bytes == 0) {
        return;
    }
    for (Entry& entry : entries) {
        if (held_bytes <= cap_bytes / 2) {
            break;
        }
        if (entry.size != 0 && !entry.deferred && scheduler.IsFree(entry.tick)) {
            Free(entry);
        }
    }
    while (!entries.empty() && entries.back().size == 0) {
        entries.pop_back();
    }
}

void StagingBufferPool::FreeDeferred(StagingBufferRef& ref) {
    if (ref.index < entries.size()) {
        Entry& entry = entries[ref.index];
        entry.deferred = false;
        entry.tick = scheduler.CurrentTick();
    }
}

StagingBufferRef StagingBufferPool::Stream(size_t size, size_t alignment) {
    if (size > stream_region_size) {
        return Request(size);
    }
    if (!stream) {
        stream = CreateBuffer(device, stream_size, true, &stream_contents);
        ASSERT_MSG(stream && stream_contents != nullptr, "Metal stream buffer failed");
    }
    u64 offset = Common::AlignUp(stream_offset, (std::max)(alignment, size_t{16}));
    const u64 region = offset / stream_region_size;
    // A slice never spans two regions, so one tick per region covers it.
    if (region >= StreamRegions || (offset + size - 1) / stream_region_size != region) {
        offset = region + 1 >= StreamRegions ? 0 : (region + 1) * stream_region_size;
    }
    const u64 first = offset / stream_region_size;
    const u64 tick = scheduler.CurrentTick();
    if (stream_ticks[first] != tick && offset % stream_region_size == 0) {
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
    if (submit_bytes != 0) {
        // A slot whose buffer went stays, so the indices handed out stay valid.
        const u64 keep = held_bytes > submit_bytes ? PressureKeepFrames : KeepFrames;
        for (Entry& entry : entries) {
            if (entry.size != 0 && !entry.deferred && frame - entry.last_frame >= keep &&
                scheduler.IsFree(entry.tick)) {
                Free(entry);
            }
        }
        while (!entries.empty() && entries.back().size == 0) {
            entries.pop_back();
        }
        return;
    }
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
