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
                VkImage depth_image, u64 frame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
