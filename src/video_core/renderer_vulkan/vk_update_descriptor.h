// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <memory>
#include <variant>
#include <vector>
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

struct DescriptorAddress {
    VkDeviceAddress address{};
    VkDeviceSize range{};
    VkFormat format{};
    u32 padding{};
};

union DescriptorUpdateEntry {
    DescriptorUpdateEntry() : raw{} {}
    DescriptorUpdateEntry(VkDescriptorImageInfo image_) : raw{} { image = image_; }
    DescriptorUpdateEntry(VkDescriptorBufferInfo buffer_) : raw{} { buffer = buffer_; }
    DescriptorUpdateEntry(VkBufferView texel_buffer_) : raw{} { texel_buffer = texel_buffer_; }
    DescriptorUpdateEntry(DescriptorAddress address_) : raw{} { address = address_; }
    std::array<u64, 3> raw;
    VkDescriptorImageInfo image;
    VkDescriptorBufferInfo buffer;
    VkBufferView texel_buffer;
    DescriptorAddress address;
};

class UpdateDescriptorQueue final {
    // This should be plenty for the vast majority of cases. Most desktop platforms only
    // provide up to 3 swapchain images.
    static constexpr size_t FRAMES_IN_FLIGHT = 8;
    static constexpr size_t CHUNK_ENTRIES = 0x1000; // 4,096 entries = 96 KiB per chunk

public:
    static constexpr size_t GUEST_FRAME_PAYLOAD_SIZE = 0x80000;
    static constexpr size_t COMPUTE_FRAME_PAYLOAD_SIZE = 0x20000;

    explicit UpdateDescriptorQueue(const Device& device_, size_t frame_payload_size_,
                                   bool supports_descriptor_buffer_ = false);
    ~UpdateDescriptorQueue();

    [[nodiscard]] bool UsesDescriptorBuffer() const noexcept {
        return use_descriptor_buffer;
    }

    void TickFrame();
    void Acquire(Scheduler& scheduler, size_t required_entries = 0,
                 bool use_descriptor_buffer_ = false);

    const DescriptorUpdateEntry* UpdateData() const noexcept {
        return upload_start;
    }

    size_t UpdateCount() const noexcept {
        return upload_start ? static_cast<size_t>(payload_cursor - upload_start) : 0;
    }

    void AddSampledImage(VkImageView image_view, VkSampler sampler) {
        *(payload_cursor++) = VkDescriptorImageInfo{
            .sampler = sampler,
            .imageView = image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    void AddImage(VkImageView image_view) {
        *(payload_cursor++) = VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    void AddBuffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        *(payload_cursor++) = VkDescriptorBufferInfo{
            .buffer = buffer,
            .offset = offset,
            .range = size,
        };
    }

    void AddBuffer(VkBuffer buffer, VkDeviceAddress base_address, VkDeviceSize offset,
                   VkDeviceSize size) {
        if (!use_descriptor_buffer) {
            AddBuffer(buffer, offset, size);
            return;
        }
        *(payload_cursor++) = DescriptorAddress{
            .address = base_address == 0 ? 0 : base_address + offset,
            .range = base_address == 0 ? VK_WHOLE_SIZE : size,
            .format = VK_FORMAT_UNDEFINED,
        };
    }

    void AddTexelBuffer(VkBufferView texel_buffer) {
        *(payload_cursor++) = texel_buffer;
    }

    /// Adds an entry kept as raw bytes (TextureCache::BindingMemo::payload).
    void AddPayload(const void* entry) {
        std::memcpy(static_cast<void*>(payload_cursor++), entry, sizeof(DescriptorUpdateEntry));
    }

    /// The entry added last.
    [[nodiscard]] const DescriptorUpdateEntry& Last() const noexcept {
        return payload_cursor[-1];
    }

    void AddTexelBuffer(VkBufferView texel_buffer, VkDeviceAddress base_address,
                        VkDeviceSize offset, VkDeviceSize size, VkFormat format) {
        if (!use_descriptor_buffer) {
            AddTexelBuffer(texel_buffer);
            return;
        }
        *(payload_cursor++) = DescriptorAddress{
            .address = base_address == 0 ? 0 : base_address + offset,
            .range = base_address == 0 ? VK_WHOLE_SIZE : size,
            .format = format,
        };
    }

private:
    struct Chunk {
        std::unique_ptr<DescriptorUpdateEntry[]> data;
        size_t capacity{};
    };

    struct FrameArena {
        std::vector<Chunk> chunks;
        size_t chunk_index{};
        size_t total_entries{};
        size_t last_used_chunks{};
        size_t peak_recent_chunks{};
        u32 excess_turns{};
    };

    void AcquireSlow(Scheduler& scheduler, size_t reserve);

    const size_t maximum_frame_payload_size;
    const bool supports_descriptor_buffer;
    bool use_descriptor_buffer{false};
    size_t frame_index{0};
    DescriptorUpdateEntry* payload_cursor = nullptr;
    DescriptorUpdateEntry* payload_end = nullptr;
    const DescriptorUpdateEntry* upload_start = nullptr;
    std::array<FrameArena, FRAMES_IN_FLIGHT> frames{};
    bool fallback_reported{};
};

// TODO: should these be separate classes instead?
using GuestDescriptorQueue = UpdateDescriptorQueue;
using ComputePassDescriptorQueue = UpdateDescriptorQueue;

} // namespace Vulkan
