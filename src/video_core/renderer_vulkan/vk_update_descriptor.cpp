// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

UpdateDescriptorQueue::UpdateDescriptorQueue([[maybe_unused]] const Device& device_,
                                             size_t frame_payload_size_,
                                             bool supports_descriptor_buffer_)
    : maximum_frame_payload_size{frame_payload_size_ * 4},
      supports_descriptor_buffer{supports_descriptor_buffer_} {}

UpdateDescriptorQueue::~UpdateDescriptorQueue() = default;

void UpdateDescriptorQueue::TickFrame() {
    FrameArena& outgoing = frames[frame_index];
    outgoing.last_used_chunks = payload_end != nullptr ? outgoing.chunk_index + 1 : 0;

    if (++frame_index >= FRAMES_IN_FLIGHT) {
        frame_index = 0;
    }

    FrameArena& arena = frames[frame_index];
    if (arena.last_used_chunks < arena.chunks.size()) {
        arena.peak_recent_chunks = (std::max)(arena.peak_recent_chunks, arena.last_used_chunks);
        if (++arena.excess_turns >= 4) {
            arena.chunks.resize(arena.peak_recent_chunks);
            arena.peak_recent_chunks = 0;
            arena.excess_turns = 0;
        }
    } else {
        arena.peak_recent_chunks = arena.last_used_chunks;
        arena.excess_turns = 0;
    }

    arena.chunk_index = 0;
    arena.total_entries = 0;
    if (!arena.chunks.empty()) {
        payload_cursor = arena.chunks[0].data.get();
        payload_end = payload_cursor + arena.chunks[0].capacity;
    } else {
        payload_cursor = nullptr;
        payload_end = nullptr;
    }
    upload_start = nullptr;
}

void UpdateDescriptorQueue::Acquire(Scheduler& scheduler, size_t required_entries,
                                    bool use_descriptor_buffer_) {
    use_descriptor_buffer = supports_descriptor_buffer && use_descriptor_buffer_;
    static constexpr size_t DEFAULT_REQUIRED_ENTRIES = 0x400;
    const size_t reserve = required_entries > 0 ? required_entries : DEFAULT_REQUIRED_ENTRIES;
    if (payload_cursor != nullptr &&
        static_cast<size_t>(payload_end - payload_cursor) >= reserve) [[likely]] {
        upload_start = payload_cursor;
        return;
    }
    AcquireSlow(scheduler, reserve);
}

void UpdateDescriptorQueue::AcquireSlow(Scheduler& scheduler, size_t reserve) {
    if (reserve > maximum_frame_payload_size) {
        throw std::length_error("Descriptor reservation exceeds the bounded staging capacity");
    }

    FrameArena& arena = frames[frame_index];
    if (payload_end != nullptr) {
        const Chunk& current = arena.chunks[arena.chunk_index];
        arena.total_entries += static_cast<size_t>(payload_cursor - current.data.get());
    }

    const auto activate_chunk = [&](size_t idx) noexcept {
        arena.chunk_index = idx;
        payload_cursor = arena.chunks[idx].data.get();
        payload_end = payload_cursor + arena.chunks[idx].capacity;
        upload_start = payload_cursor;
    };

    const size_t next_index = payload_end != nullptr ? arena.chunk_index + 1 : 0;
    if (arena.total_entries + reserve <= maximum_frame_payload_size) {
        if (next_index < arena.chunks.size() && arena.chunks[next_index].capacity >= reserve) {
            activate_chunk(next_index);
            return;
        }
        const size_t alloc_cap = (std::max)(CHUNK_ENTRIES, reserve);
        std::unique_ptr<DescriptorUpdateEntry[]> storage;
        try {
            storage = std::make_unique_for_overwrite<DescriptorUpdateEntry[]>(alloc_cap);
        } catch (const std::bad_alloc&) {
            storage.reset();
        }
        if (storage) {
            if (next_index < arena.chunks.size()) {
                arena.chunks[next_index] = Chunk{std::move(storage), alloc_cap};
            } else {
                arena.chunks.push_back(Chunk{std::move(storage), alloc_cap});
            }
            activate_chunk(next_index);
            return;
        }
    }

    if (!fallback_reported) {
        LOG_WARNING(Render_Vulkan,
                    "Descriptor payload arena reached {} entries in frame {}; using worker drain fallback",
                    arena.total_entries, frame_index);
        fallback_reported = true;
    }
    scheduler.WaitWorker();
    arena.total_entries = 0;
    for (size_t i = 0; i < arena.chunks.size(); ++i) {
        if (arena.chunks[i].capacity >= reserve) {
            activate_chunk(i);
            return;
        }
    }
    const size_t alloc_cap = (std::max)(CHUNK_ENTRIES, reserve);
    auto storage = std::make_unique_for_overwrite<DescriptorUpdateEntry[]>(alloc_cap);
    if (arena.chunks.empty()) {
        arena.chunks.push_back(Chunk{std::move(storage), alloc_cap});
    } else {
        arena.chunks[0] = Chunk{std::move(storage), alloc_cap};
    }
    activate_chunk(0);
}

} // namespace Vulkan
