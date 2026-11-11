// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (present/metalfx_*.mm): Metal work ordered with MoltenVK's queue.

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <functional>
#include <memory>

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
/// binary semaphore the scheduler's next submission waits for. Without shared events, or when
/// EDEN_NX2_METALFX_SYNC=cpu is set, it waits on the CPU: Finish, then waitUntilCompleted.
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

    /// Whether Run orders the work with shared events on the GPU instead of waiting on the CPU.
    [[nodiscard]] bool UsesSharedEvents() const noexcept {
        return metal_event != nil;
    }

    /// The Metal texture behind a Vulkan image, or nil.
    [[nodiscard]] id<MTLTexture> Texture(VkImage image) const;

    /// Encodes work that runs after every Vulkan command recorded so far and before every one
    /// recorded after this returns. encode returns false to drop the command buffer. Returns
    /// whether the work was committed (with CPU waits: whether it completed), and false once a
    /// command buffer of the shared-event path has failed after Run returned. Logs the reason.
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
    std::array<vk::Semaphore, 8> bridge_semaphores;
    size_t bridge_index{};
    u64 metal_value{};
    /// Set by the completion handler of a command buffer that failed after Run returned.
    std::shared_ptr<std::atomic<bool>> async_failure = std::make_shared<std::atomic<bool>>(false);
};

/// Whether the texture can be used where MetalFX asks for these usage options.
[[nodiscard]] inline bool HasUsage(id<MTLTexture> texture, MTLTextureUsage usage) {
    return texture != nil && (texture.usage & usage) == usage;
}

/// A 2D image (in UNDEFINED layout) whose Metal texture MetalFX reads or writes.
[[nodiscard]] vk::Image CreateMetalFxImage(MemoryAllocator& allocator, VkExtent2D extent,
                                           VkFormat format, VkImageUsageFlags usage);

} // namespace Vulkan
