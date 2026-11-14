// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class DescriptorPool;
class Device;
class MemoryAllocator;
class Scheduler;
class StagingBufferPool;

/// The temporal upscaler Ocarina uses this session: the settings, EDEN_NX2_DLSS_REPLACEMENT,
/// DLSS / MetalFX availability, and native when FSR 3.1 is unsupported on the device.
[[nodiscard]] VideoCommon::OcarinaUpscaleBackend ResolveOcarinaUpscaler(const Device& device);

struct OcarinaActorMesh {
    std::span<const u8> vertices;
    std::span<const u16> indices;
    GPUVAddr vertex_address{};
    GPUVAddr index_address{};
    s32 base_vertex{};
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::span<const f32> current_bones;
    std::span<const f32> previous_bones;
};

/// Replaces the game's upscale pass with AMD FSR 3.1, or with the build's live upscaler (NVIDIA
/// DLSS, Apple MetalFX temporal) for the Dlss backend.
class OcarinaUpscaler {
public:
    OcarinaUpscaler(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                    ComputePassDescriptorQueue& queue, MemoryAllocator& allocator,
                    StagingBufferPool& staging, VideoCommon::OcarinaUpscaleBackend backend);
    ~OcarinaUpscaler();

    [[nodiscard]] bool EvaluationFailed() const;
    /// Save the 1440p upscaled copy before the game adds its HUD.
    void SaveSceneReference(VkImageView color);
    /// Composite the HUD from the guest image over the matching 4K scene.
    void Composite4k(VkImageView native_composite, u64 frame);
    /// Copy the scene colour once the fog is drawn, before the glows and particles that have no
    /// motion vectors of their own.
    void SaveEffectsBase(VkImageView scene, VkExtent2D extent, u64 frame);
    /// Mark the pixels those glows and particles changed since SaveEffectsBase, so Record can
    /// tell the upscaler to use this frame's colour there.
    void MarkEffects(VkImageView scene, u64 frame);
    VkImage CompositedImage() const;
    VkImageView CompositedView() const;

    /// Queue motion, the evaluation and guest writeback for one frame. A false return leaves
    /// the guest dispatch in charge. The scheduler checks EvaluationFailed before skipping it.
    bool Record(VkImageView color, VkImageView depth, VkImageView target, VkExtent2D extent,
                VkExtent2D output_extent, VkExtent2D target_extent, const std::array<f32, 16>& current,
                const std::array<f32, 16>& previous, const OcarinaActorMesh& mesh,
                VkImage depth_image, u64 frame, std::array<f32, 2> jitter = {},
                VkExtent2D depth_extent = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/// EDEN_NX2_OOT_BILINEAR_UPSCALE (boot, 1 = on; phones get it from their device profile): the
/// game's upscale pass becomes one bilinear tap whenever no temporal upscaler takes the frame.
[[nodiscard]] bool OcarinaBilinearUpscaleEnabled();

/// The game's upscale pass as one bilinear tap of the scene, squared as the game's pass writes it.
class OcarinaBilinearUpscale {
public:
    OcarinaBilinearUpscale(const Device& device, Scheduler& scheduler, DescriptorPool& pool,
                           ComputePassDescriptorQueue& queue);
    ~OcarinaBilinearUpscale();

    /// Queues the pass: the scene image's top-left extent (padded images included) to target.
    void Record(VkImageView color, VkExtent2D color_size, VkExtent2D extent, VkImageView target,
                VkExtent2D target_extent);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
