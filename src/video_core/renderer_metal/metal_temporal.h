// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// MetalFX temporal upscaling on the renderer's own textures and command buffer, in place of a
// game's DLSS (the texture cache's TemporalUpscale), as Vulkan's MetalFxTemporalUpscaler does
// through VK_EXT_metal_objects. Plain C++ header.

#pragma once

#include <array>
#include <memory>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"

namespace Metal {

class Device;
class Scheduler;

class TemporalUpscaler {
public:
    struct Inputs {
        TextureHandle color;  ///< Jittered colour at the render size.
        TextureHandle depth;  ///< Device depth at the render size.
        TextureHandle motion; ///< Motion to the previous frame at the render size.
        std::array<u32, 2> render_size;
        std::array<u32, 2> output_size;
        /// This frame's sample offset in render pixels.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels.
        std::array<f32, 2> motion_vector_scale;
        bool reset;
    };

    TemporalUpscaler();
    ~TemporalUpscaler();

    /// Whether this GPU and OS run MetalFX temporal upscaling.
    [[nodiscard]] static bool IsSupported(const Device& device);

    /// Encodes one evaluation on the scheduler's command buffer. The output (RGBA16F at the
    /// output size), or null when MetalFX can't run (logged once; later calls return null too).
    [[nodiscard]] TextureHandle Upscale(const Device& device, Scheduler& scheduler,
                                        const Inputs& inputs);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
