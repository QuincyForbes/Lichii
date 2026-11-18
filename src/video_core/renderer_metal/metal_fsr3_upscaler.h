// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// AMD FidelityFX Super Resolution 3.1 in place of a game's DLSS evaluation, as Vulkan's
// Fsr3Upscaler (vk_fsr3_upscaler.cpp): the same nine compute passes (the SDK's shaders through
// the SPIR-V -> MSL translator), internal textures and constants, recorded on the scheduler. The
// default profile: reversed infinite depth and HDR colour with auto exposure. Plain C++.

#pragma once

#include <array>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

class Device;

class Fsr3Upscaler {
public:
    struct Inputs {
        TextureHandle color;   ///< HDR colour at render size.
        TextureHandle depth;   ///< Device depth (a depth texture) at render size.
        TextureHandle motion;  ///< Motion to the previous frame at the render size.
        std::array<u32, 2> render_size;
        std::array<u32, 2> upscale_size;
        /// This frame's sample offset in render pixels.
        std::array<f32, 2> jitter;
        /// Turns a motion vector into render pixels.
        std::array<f32, 2> motion_vector_scale;
        f32 frame_time_ms;
        bool reset;
        bool sharpen;
        f32 sharpness; ///< 0 to 1 as FSR takes it: 2 stops of RCAS at 0, none at 1.
        f32 vertical_fov = 1.04719755f;
    };

    explicit Fsr3Upscaler(const Device& device, Scheduler& scheduler);
    ~Fsr3Upscaler();

    /// Whether this GPU runs the passes (texture atomics, the formats); logged once when not.
    [[nodiscard]] bool Prepare();

    /// Records one evaluation; the output (RGBA16F at the upscale size), valid until the next
    /// call, or null when it can't be made.
    [[nodiscard]] TextureHandle Upscale(const Inputs& inputs);

private:
    struct Texture {
        Object texture;
        std::vector<Object> levels; ///< One view per level, for textures with mips.
        std::array<u32, 2> size{};
        u32 format{};
        u32 level_count{1};
        TextureHandle Handle() const noexcept {
            return texture.Get();
        }
    };

    struct Pass {
        Object library;
        Object function;
        Object state;
        std::vector<ShaderTranslate::ResourceSlot> slots;
        std::array<u32, 3> group{};
    };

    /// One descriptor of a dispatch: a texture, or constants.
    struct Resource {
        TextureHandle texture{};
        const void* bytes{};
        u32 size{};
    };

    bool BuildPass(Pass& pass, std::span<const u32> code, bool atomics);
    Texture MakeTexture(u32 format, u32 width, u32 height, u32 levels = 1);
    bool CreateResources(std::array<u32, 2> render_size, std::array<u32, 2> upscale_size);
    void Dispatch(const Pass& pass, std::span<const Resource> resources, u32 groups_x,
                  u32 groups_y);
    void ClearTexture(const Texture& target, std::array<f64, 4> value);

    const Device& device;
    Scheduler& scheduler;
    Object point_sampler;
    Object linear_sampler;
    std::array<Pass, 9> passes;
    bool prepared{};
    bool failed{};

    std::array<Texture, 2> accumulation;
    std::array<Texture, 2> luma;
    std::array<Texture, 2> luma_history;
    std::array<Texture, 2> internal_upscaled;
    Texture farthest_depth;
    Texture shading_change;
    Texture new_locks;
    Texture spd_mips;
    Texture farthest_depth_mip1;
    Texture dilated_motion_vectors;
    Texture dilated_depth;
    Texture previous_nearest_depth;
    Texture dilated_reactive_masks;
    Texture frame_info;
    Texture spd_atomic;
    Texture lanczos_lut;
    Texture default_reactive;
    Texture unit_exposure;
    Texture output;
    Texture depth_color; ///< The scene depth as R32F

    std::array<u32, 2> render_extent{};
    std::array<u32, 2> upscale_extent{};
    std::array<s32, 2> previous_render_size{};
    std::array<s32, 2> previous_upscale_size{};
    std::array<f32, 2> previous_jitter{};
    f32 jitter_phase_count{};
    f32 frame_index{};
    u32 frame_parity{};
    bool first_execution{true};
};

} // namespace Metal
