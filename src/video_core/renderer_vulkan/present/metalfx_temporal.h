// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Apple MetalFX temporal upscaling in place of a game's DLSS evaluation and of Ocarina of Time's
// upscaler, where NVIDIA DLSS (vk_streamline_dlss.h, same interface) runs elsewhere.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

class MetalFxTemporalUpscaler {
public:
    struct Image {
        VkImageView view;
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The view's level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
    };

    struct Inputs {
        Image color;          ///< Jittered colour at the render size.
        Image depth;          ///< Device depth at the render size.
        Image motion_vectors; ///< Motion to the previous frame at the render size.
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        /// This frame's sample offset in render pixels.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels (the render size for UV offsets).
        std::array<f32, 2> motion_vector_scale;
        bool reset;
        bool depth_inverted = true;
        bool hdr = true; ///< Linear HDR colour. False for tonemapped, perceptually encoded colour.
        /// R16F at the render size: 1 where the history should give way to this frame's colour.
        std::optional<Image> bias_current_color;
    };

    /// Throws vk::Exception when MetalFX can't run on this device.
    explicit MetalFxTemporalUpscaler(const Device& device, Scheduler& scheduler,
                                     MemoryAllocator& memory_allocator);
    ~MetalFxTemporalUpscaler();

    /// Whether this device and OS version run MetalFX temporal upscaling.
    [[nodiscard]] static bool IsSupported(const Device& device);

    /// Runs one evaluation into the output image (RGBA16F at the upscale size, GENERAL layout).
    /// Returns false when MetalFX can't run, so the caller can fall back.
    bool Upscale(const Inputs& inputs);

    [[nodiscard]] bool Failed() const noexcept {
        return failed.load();
    }

    [[nodiscard]] VkImage OutputImage() const noexcept {
        return *output_image;
    }
    [[nodiscard]] VkImageView OutputView() const noexcept {
        return *output_view;
    }
    /// Nearest-neighbour clamp sampler (normalized coordinates), for reading the output.
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }

private:
    struct Impl;

    void CreateOutput(VkExtent2D upscale_size);

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    std::unique_ptr<Impl> impl;
    VkExtent2D upscale_extent{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Sampler point_sampler;
    std::atomic<bool> failed{};
};

} // namespace Vulkan
