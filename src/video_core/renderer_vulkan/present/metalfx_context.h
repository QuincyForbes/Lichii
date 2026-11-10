// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (present/metalfx_*.mm): Metal work ordered with MoltenVK's queue.

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <functional>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Runs MetalFX on the Metal objects behind MoltenVK's Vulkan objects (VK_EXT_metal_objects).
///
/// Work goes to a command queue of its own. It waits on the GPU for the scheduler's timeline
/// semaphore (exported as an MTLSharedEvent) to reach the tick of everything recorded before it,
/// and signals a timeline semaphore of its own; a submission with no commands turns that into a
/// binary semaphore the scheduler's next submission waits for. That path is opt-in
/// (EDEN_NX2_METALFX_SYNC=event); by default, and without shared events, it waits on the CPU:
/// Finish, then waitUntilCompleted.
class MetalFxContext {
public:
    explicit MetalFxContext(const Device& device, Scheduler& scheduler);
    ~MetalFxContext();

    MetalFxContext(const MetalFxContext&) = delete;
    MetalFxContext& operator=(const MetalFxContext&) = delete;

    [[nodiscard]] bool IsValid() const noexcept {
        return queue != nil;
    }

    [[nodiscard]] id<MTLDevice> MtlDevice() const noexcept {
        return mtl_device;
    }

    /// The Metal texture behind a Vulkan image, or nil.
    [[nodiscard]] id<MTLTexture> Texture(VkImage image) const;

    /// Encodes work that runs after every Vulkan command recorded so far and before every one
    /// recorded after this returns. encode returns false to drop the command buffer. Returns
    /// whether the work was committed.
    bool Run(const std::function<bool(id<MTLCommandBuffer>)>& encode);

private:
    [[nodiscard]] id<MTLSharedEvent> SharedEvent(VkSemaphore semaphore) const;
    void SetUpEvents();

    const Device& device;
    Scheduler& scheduler;
    PFN_vkExportMetalObjectsEXT export_objects{};
    id<MTLDevice> mtl_device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> last_command_buffer = nil;
    id<MTLSharedEvent> vulkan_event = nil; ///< The scheduler's timeline semaphore
    id<MTLSharedEvent> metal_event = nil;  ///< metal_semaphore
    vk::Semaphore metal_semaphore;
    std::array<vk::Semaphore, 2> bridge_semaphores;
    size_t bridge_index{};
    u64 metal_value{};
};

/// Whether the texture can be used where MetalFX asks for these usage options.
[[nodiscard]] inline bool HasUsage(id<MTLTexture> texture, MTLTextureUsage usage) {
    return texture != nil && (texture.usage & usage) == usage;
}

/// A 2D image (in UNDEFINED layout) whose Metal texture MetalFX reads or writes.
[[nodiscard]] vk::Image CreateMetalFxImage(MemoryAllocator& allocator, VkExtent2D extent,
                                           VkFormat format, VkImageUsageFlags usage);

} // namespace Vulkan
