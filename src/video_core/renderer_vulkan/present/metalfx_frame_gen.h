// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Apple MetalFX frame interpolation (iOS 26 / macOS 26), with the optical flow of the FSR 3
/// frame generation as its motion vectors. It makes the one frame halfway between two.
class MetalFxFrameGen {
public:
    explicit MetalFxFrameGen(const Device& device, Scheduler& scheduler);
    ~MetalFxFrameGen();

    /// Whether this device and OS version run MetalFX frame interpolation.
    [[nodiscard]] static bool IsSupported(const Device& device);

    /// Writes the frame halfway between previous and current into destination (same size and
    /// format as both). motion holds the flow at motion_extent, in flow texels, pointing from a
    /// current pixel to where it was in the previous frame. False leaves the frame to FSR 3.
    bool Generate(VkImage previous, VkImage current, VkExtent2D extent, VkImage motion,
                  VkExtent2D motion_extent, VkImage destination, f32 delta_time, bool reset);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
