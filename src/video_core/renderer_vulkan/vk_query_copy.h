// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

inline std::vector<VkBufferCopy> NonOverlappingQueryCopies(std::span<const VkBufferCopy> copies) {
    std::vector<VkBufferCopy> result;
    result.reserve(copies.size());
    for (const VkBufferCopy& copy : copies) {
        if (copy.size == 0) {
            continue;
        }
        const VkDeviceSize new_begin = copy.dstOffset;
        const VkDeviceSize new_end = copy.dstOffset + copy.size;
        const size_t existing_count = result.size();
        for (size_t i = 0; i < existing_count; ++i) {
            VkBufferCopy& existing = result[i];
            if (existing.size == 0) {
                continue;
            }
            const VkDeviceSize old_begin = existing.dstOffset;
            const VkDeviceSize old_end = existing.dstOffset + existing.size;
            if (new_end <= old_begin || new_begin >= old_end) {
                continue;
            }
            if (new_begin <= old_begin && new_end >= old_end) {
                existing.size = 0;
            } else if (new_begin <= old_begin) {
                const VkDeviceSize trim = new_end - old_begin;
                existing.srcOffset += trim;
                existing.dstOffset += trim;
                existing.size -= trim;
            } else if (new_end >= old_end) {
                existing.size = new_begin - old_begin;
            } else {
                const VkDeviceSize right_offset = new_end - old_begin;
                const VkBufferCopy right{
                    .srcOffset = existing.srcOffset + right_offset,
                    .dstOffset = new_end,
                    .size = old_end - new_end,
                };
                existing.size = new_begin - old_begin;
                result.push_back(right);
            }
        }
        result.push_back(copy);
    }
    std::erase_if(result, [](const VkBufferCopy& c) { return c.size == 0; });
    return result;
}

} // namespace Vulkan
