// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Objective-C++ only (renderer_metal/*.mm).

#pragma once

#import <Metal/Metal.h>

#include <string>

#include "common/common_types.h"

namespace Metal {

/// The system's Metal GPU and the one command queue the renderer submits to.
class Device {
public:
    /// Throws std::runtime_error when the system has no Metal GPU.
    explicit Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] id<MTLDevice> GetDevice() const noexcept {
        return device;
    }

    [[nodiscard]] id<MTLCommandQueue> GetQueue() const noexcept {
        return queue;
    }

    [[nodiscard]] const std::string& GetName() const noexcept {
        return name;
    }

    /// Highest MTLGPUFamilyAppleN the GPU supports (Apple9 = A17 Pro / M3 and newer), 0 if none.
    [[nodiscard]] u32 AppleFamily() const noexcept {
        return apple_family;
    }

    /// Metal mesh pipelines (MTLMeshRenderPipelineDescriptor): Apple7+ and Mac2 GPUs.
    [[nodiscard]] bool SupportsMeshShaders() const noexcept {
        return mesh_shaders;
    }

    /// BC1-BC7 textures: Macs, and iPads/iPhones with Apple9 GPUs since iOS 16.4.
    [[nodiscard]] bool SupportsBCTextures() const noexcept {
        return bc_textures;
    }

    /// Bytes the GPU should keep resident: recommendedMaxWorkingSetSize on macOS, the memory the
    /// process may still allocate on iOS (unified memory).
    [[nodiscard]] u64 MemoryBudget() const;

    /// Bytes the device has allocated for this process (currentAllocatedSize).
    [[nodiscard]] u64 AllocatedSize() const;

private:
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    std::string name;
    u32 apple_family{};
    bool mesh_shaders{};
    bool bc_textures{};
};

} // namespace Metal
