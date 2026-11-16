// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <stdexcept>
#include <utility>

#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#include <os/proc.h>
#endif

#include "common/logging.h"
#include "video_core/renderer_metal/metal_device.h"

namespace Metal {

Device::Device() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        throw std::runtime_error("No Metal device");
    }
    // Command buffers kept alive by the queue: the present one and a few in flight.
    queue = [device newCommandQueueWithMaxCommandBufferCount:64];
    if (queue == nil) {
        throw std::runtime_error("Metal command queue creation failed");
    }
    queue.label = @"Lichii";
    name = device.name.UTF8String;

    static constexpr std::pair<MTLGPUFamily, u32> families[]{
        {MTLGPUFamilyApple9, 9}, {MTLGPUFamilyApple8, 8}, {MTLGPUFamilyApple7, 7},
        {MTLGPUFamilyApple6, 6}, {MTLGPUFamilyApple5, 5}, {MTLGPUFamilyApple4, 4},
    };
    for (const auto& [family, number] : families) {
        if ([device supportsFamily:family]) {
            apple_family = number;
            break;
        }
    }
    mesh_shaders = apple_family >= 7 || [device supportsFamily:MTLGPUFamilyMac2];
#if TARGET_OS_IPHONE
    bc_textures = false;
    if (@available(iOS 16.4, *)) {
        bc_textures = device.supportsBCTextureCompression;
    }
#else
    bc_textures = true;
#endif

    LOG_INFO(Render_Metal,
             "Metal: {} (Apple GPU family {}, Metal 3 {}, mesh shaders {}, BC textures {}, "
             "argument buffers tier {}, memory budget {} MiB)",
             name, apple_family, [device supportsFamily:MTLGPUFamilyMetal3] ? "yes" : "no",
             mesh_shaders ? "yes" : "no", bc_textures ? "yes" : "no",
             device.argumentBuffersSupport == MTLArgumentBuffersTier2 ? 2 : 1,
             MemoryBudget() >> 20);
}

Device::~Device() = default;

u64 Device::MemoryBudget() const {
#if TARGET_OS_IPHONE
    return os_proc_available_memory();
#else
    return device.recommendedMaxWorkingSetSize;
#endif
}

u64 Device::AllocatedSize() const {
    return device.currentAllocatedSize;
}

} // namespace Metal
