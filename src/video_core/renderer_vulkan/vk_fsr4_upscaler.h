// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// FSR 4 through the FFX API (vulkan_common/ffx_runtime.h): on Windows AMD's FidelityFX runtime,
// which the user downloads from AMD in the Graphics settings; on Linux the fsr4-vk community Vulkan
// port the packages ship. Supports both Ocarina's LDR standard-depth profile and Switch 2 guest
// DLSS games' HDR reversed-infinite-depth profile, falling back to FSR 3.1 when off, missing or
// failing.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MemoryAllocator;
class Scheduler;

namespace FfxRuntime {
struct Api;
}

class Fsr4Upscaler {
public:
    enum class Profile {
        OcarinaLdr,
        HdrReversedDepth,
    };

    struct Image {
        VkImage image;
        VkFormat format;
        VkImageSubresourceRange range;     ///< The used level and layer, with the view's aspect.
        VkImageAspectFlags barrier_aspect; ///< Every aspect of the image's format.
        VkExtent2D extent{};               ///< Backing image dimensions (render_size when zero).
    };

    struct Inputs {
        Image color;          ///< Profile's HDR or tonemapped LDR colour at the render size.
        Image depth;          ///< Device depth in the profile's convention at the render size.
        Image motion_vectors; ///< Motion to the previous frame.
        std::optional<Image> reactive;
        VkExtent2D render_size;
        VkExtent2D upscale_size;
        std::optional<VkExtent2D> max_render_size;
        f32 frame_time_ms;
        bool reset;
        bool sharpen;
        f32 sharpness; ///< 0-1
        f32 vertical_fov = 1.04719755f;
        f32 camera_near = 0.1f;
        f32 camera_far = 25000.0f;
        std::array<f32, 2> jitter{}; ///< Offset the scene was drawn with, in render pixels.
        std::array<f32, 2> motion_vector_scale{1.0f, 1.0f};
    };

    /// Whether FSR 4 is turned on and the device can run it: AMD's driver on Windows, and on Linux
    /// what CheckFsr4Port asks of the device (the log says why not).
    [[nodiscard]] static bool IsWanted(const Device& device);

    /// Throws vk::Exception when FSR 4 can't start on this device: the runtime is missing or
    /// doesn't check out, or it offers no FSR 4 for the GPU.
    explicit Fsr4Upscaler(const Device& device, Scheduler& scheduler,
                          MemoryAllocator& memory_allocator,
                          Profile profile = Profile::OcarinaLdr);
    ~Fsr4Upscaler();

    /// Records one evaluation into the output image (RGBA16F at the upscale size, GENERAL
    /// layout). False when FSR 4 can't run (the caller then uses FSR 3.1 from now on).
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
    [[nodiscard]] VkSampler PointSampler() const noexcept {
        return *point_sampler;
    }

private:
    bool CreateContext(VkExtent2D max_render_size, VkExtent2D upscale_size, bool dynamic_resolution);
    void DestroyContext();
    void CreateOutput(VkExtent2D upscale_size);
    void CreateExposure();

    const Device& device;
    Scheduler& scheduler;
    MemoryAllocator& memory_allocator;
    const Profile profile;
    std::shared_ptr<const FfxRuntime::Api> api;
    u64 provider_id{};
    std::string provider_name;
    void* context{}; ///< ffxContext
    VkExtent2D context_render_capacity{};
    VkExtent2D upscale_extent{};
    bool context_dynamic_resolution{};
    vk::Image output_image;
    vk::ImageView output_view;
    vk::Image exposure_image; ///< 1x1 R32F holding 1 (OcarinaLdr unit exposure)
    vk::Sampler point_sampler;
    std::atomic<bool> failed{};
};

} // namespace Vulkan
