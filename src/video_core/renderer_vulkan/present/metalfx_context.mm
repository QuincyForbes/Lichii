// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_vulkan/present/metalfx_context.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

// Shared-event GPU synchronization is used by default where MTLSharedEvent export is available;
// EDEN_NX2_METALFX_SYNC=cpu forces synchronous CPU waits for A/B debugging.
bool ForceCpuSync() {
    static const bool force = [] {
        const char* const value = std::getenv("EDEN_NX2_METALFX_SYNC");
        return value != nullptr && std::string_view{value} == "cpu";
    }();
    return force;
}

/// Why a command buffer did not complete, for the log.
std::string FailureText(id<MTLCommandBuffer> buffer) {
    NSError* const error = buffer.error;
    if (error == nil) {
        return fmt::format("status {}, no error", static_cast<int>(buffer.status));
    }
    const char* const domain = error.domain.UTF8String;
    const char* const description = error.localizedDescription.UTF8String;
    return fmt::format("{} {}: {}", domain ? domain : "?", static_cast<long>(error.code),
                       description ? description : "?");
}

} // Anonymous namespace

vk::Image CreateMetalFxImage(MemoryAllocator& allocator, VkExtent2D extent, VkFormat format,
                             VkImageUsageFlags usage) {
    static constexpr VkExportMetalObjectCreateInfoEXT export_ci{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
        .pNext = nullptr,
        .exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_TEXTURE_BIT_EXT,
    };
    const MemoryTagScope memory_tag{MemoryTag::Upscaler};
    return allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &export_ci,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
}

MetalFxContext::MetalFxContext(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {
    if (!device.IsExtMetalObjectsEnabled()) {
        return;
    }
    const VkDevice handle = *device.GetLogical();
    export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        device.GetDispatchLoader().vkGetDeviceProcAddr(handle, "vkExportMetalObjectsEXT"));
    if (!export_objects) {
        return;
    }
    VkExportMetalDeviceInfoEXT device_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT,
        .pNext = nullptr,
        .mtlDevice = nil,
    };
    VkExportMetalObjectsInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
        .pNext = &device_info,
    };
    export_objects(handle, &info);
    mtl_device = device_info.mtlDevice;
    if (mtl_device == nil) {
        return;
    }
    queue = [mtl_device newCommandQueue];
    queue.label = @"Lichii MetalFX";
    if (!ForceCpuSync()) {
        SetUpEvents();
    }
    LOG_INFO(Render_Vulkan, "MetalFX: {} synchronization with MoltenVK",
             metal_event != nil ? "shared event" : "CPU");
}

MetalFxContext::~MetalFxContext() {
    // Consume any pending next_wait_semaphore and let in-flight bridge submissions finish before
    // metal_semaphore and bridge_semaphores are destroyed.
    if (metal_value != 0) {
        scheduler.Finish();
    }
    if (last_command_buffer != nil) {
        [last_command_buffer waitUntilCompleted];
    }
}

id<MTLSharedEvent> MetalFxContext::SharedEvent(VkSemaphore semaphore) const {
    VkExportMetalSharedEventInfoEXT event_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
        .pNext = nullptr,
        .semaphore = semaphore,
        .event = VK_NULL_HANDLE,
        .mtlSharedEvent = nil,
    };
    VkExportMetalObjectsInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
        .pNext = &event_info,
    };
    export_objects(*device.GetLogical(), &info);
    return event_info.mtlSharedEvent;
}

void MetalFxContext::SetUpEvents() {
    // MoltenVK backs timeline semaphores with MTLSharedEvent where the device has them.
    const VkSemaphore master = scheduler.GetMasterSemaphore().Handle();
    if (master == VK_NULL_HANDLE) {
        return;
    }
    id<MTLSharedEvent> master_event = SharedEvent(master);
    if (master_event == nil) {
        return;
    }
    const VkExportMetalObjectCreateInfoEXT export_ci{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECT_CREATE_INFO_EXT,
        .pNext = nullptr,
        .exportObjectType = VK_EXPORT_METAL_OBJECT_TYPE_METAL_SHARED_EVENT_BIT_EXT,
    };
    const VkSemaphoreTypeCreateInfo type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = &export_ci,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    vk::Semaphore semaphore = device.GetLogical().CreateSemaphore(VkSemaphoreCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &type_ci,
        .flags = 0,
    });
    id<MTLSharedEvent> event = SharedEvent(*semaphore);
    if (event == nil) {
        return;
    }
    for (vk::Semaphore& bridge : bridge_semaphores) {
        bridge = device.GetLogical().CreateSemaphore();
    }
    metal_semaphore = std::move(semaphore);
    vulkan_event = master_event;
    metal_event = event;
}

id<MTLTexture> MetalFxContext::Texture(VkImage image) const {
    if (!export_objects || image == VK_NULL_HANDLE) {
        return nil;
    }
    VkExportMetalTextureInfoEXT texture_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
        .pNext = nullptr,
        .image = image,
        .imageView = VK_NULL_HANDLE,
        .bufferView = VK_NULL_HANDLE,
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
        .mtlTexture = nil,
    };
    VkExportMetalObjectsInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
        .pNext = &texture_info,
    };
    export_objects(*device.GetLogical(), &info);
    return texture_info.mtlTexture;
}

bool MetalFxContext::Run(const std::function<bool(id<MTLCommandBuffer>)>& encode) {
    // The handler of the command buffer that failed has logged why.
    if (queue == nil || async_failure->load()) {
        return false;
    }
    @autoreleasepool {
        const bool gpu_sync = metal_event != nil;
        u64 tick = 0;
        if (gpu_sync) {
            tick = scheduler.Flush();
            // The bridge submission below has to reach MoltenVK's queue after the one that
            // signals tick: a wait queued ahead of its signal on the same queue never ends.
            scheduler.WaitWorker();
        } else {
            scheduler.Finish();
        }
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            return false;
        }
        command_buffer.label = @"MetalFX";
        if (gpu_sync) {
            [command_buffer encodeWaitForEvent:vulkan_event value:tick];
        }
        if (!encode(command_buffer)) {
            return false;
        }
        if (!gpu_sync) {
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            last_command_buffer = nil;
            if (command_buffer.status != MTLCommandBufferStatusCompleted) {
                LOG_ERROR(Render_Vulkan, "MetalFX: the command buffer failed ({})",
                          FailureText(command_buffer));
                return false;
            }
            return true;
        }
        const u64 value = ++metal_value;
        [command_buffer encodeSignalEvent:metal_event value:value];
        // A command buffer that faults may skip its signal, and MoltenVK's queue would wait for it
        // forever: signal from the CPU instead. Its output was not written, so later calls fail and
        // the callers fall back instead of presenting it.
        id<MTLSharedEvent> event = metal_event;
        const std::shared_ptr<std::atomic<bool>> failure = async_failure;
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
          if (buffer.status != MTLCommandBufferStatusError) {
              return;
          }
          if (event.signaledValue < value) {
              event.signaledValue = value;
          }
          if (!failure->exchange(true)) {
              LOG_ERROR(Render_Vulkan, "MetalFX: the command buffer failed ({})",
                        FailureText(buffer));
          }
        }];
        [command_buffer commit];
        last_command_buffer = command_buffer;

        const VkSemaphore metal = *metal_semaphore;
        const VkSemaphore bridge = *bridge_semaphores[bridge_index];
        bridge_index = (bridge_index + 1) % bridge_semaphores.size();
        static constexpr VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        static constexpr u64 binary_value = 0;
        const VkTimelineSemaphoreSubmitInfo timeline_info{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 1,
            .pWaitSemaphoreValues = &value,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &binary_value,
        };
        const VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timeline_info,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &metal,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 0,
            .pCommandBuffers = nullptr,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &bridge,
        };
        VkResult result;
        {
            std::scoped_lock lock{scheduler.submit_mutex};
            result = device.GetGraphicsQueue().Submit(submit_info);
        }
        if (result != VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "MetalFX: bridge submission failed ({}); waiting on the CPU",
                      static_cast<int>(result));
            [command_buffer waitUntilCompleted];
            return true;
        }
        scheduler.WaitOnNextSubmit(bridge);
        return true;
    }
}

} // namespace Vulkan
