// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <optional>

#include "common/math_util.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Apple MetalFX spatial upscaling, the MetalFxSpatial scaling filter (Apple only).
class MetalFxSpatial {
public:
    explicit MetalFxSpatial(const Device& device, MemoryAllocator& memory_allocator,
                            Scheduler& scheduler, VkExtent2D extent);
    ~MetalFxSpatial();

    MetalFxSpatial(MetalFxSpatial&&) noexcept;
    MetalFxSpatial& operator=(MetalFxSpatial&&) noexcept;

    /// Whether this device and OS version run MetalFX spatial scaling.
    [[nodiscard]] static bool IsSupported(const Device& device);

    /// Upscales the crop of the source to the output size. Returns the upscaled view, or nothing
    /// when MetalFX can't run (the caller presents the source as it is).
    std::optional<VkImageView> Draw(Scheduler& scheduler, VkImage source_image,
                                    VkExtent2D input_image_extent,
                                    const Common::Rectangle<f32>& crop_rect);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
