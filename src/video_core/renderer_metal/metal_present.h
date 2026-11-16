// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The Vulkan renderer's present filters on Metal (EDEN_NX2_METAL_PRESENT_FILTERS): the window
// filters (present/filters.cpp), FSR 1 and SGSR (present/fsr.cpp, present/sgsr.cpp), FXAA and
// SMAA (present/fxaa.cpp, present/smaa.cpp), with Vulkan's host shaders through the SPIR-V ->
// MSL translator, drawn as render passes on the scheduler. Plain C++.

#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

class Device;

class PresentPasses {
public:
    /// A picture on its way to the window: `crop` (left, top, right, bottom, 0-1 of the
    /// texture) of a texture `size` large, which shows `render` pixels of the guest's.
    struct Source {
        TextureHandle texture{};
        std::array<u32, 2> size{};
        std::array<u32, 2> render{};
        std::array<f32, 4> crop{0.0f, 0.0f, 1.0f, 1.0f};
    };

    explicit PresentPasses(const Device& device, Scheduler& scheduler);
    ~PresentPasses();

    /// EDEN_NX2_METAL_PRESENT_FILTERS=0: the layers go through the bilinear or nearest blit
    /// whatever the filter and anti-aliasing settings say, as before.
    [[nodiscard]] static bool Enabled();

    /// Whether `filter` draws the window pass with a shader of its own (else the blit does).
    [[nodiscard]] static bool HasWindowShader(Settings::ScalingFilter filter);

    /// FXAA or SMAA over the source's whole texture, as Vulkan's layer applies them; the source
    /// unchanged with none (or when a pass can't be made).
    [[nodiscard]] Source AntiAlias(const Source& source, Settings::AntiAliasing anti_aliasing);

    /// FSR 1 (EASU, then RCAS at the FSR sharpness setting) or SGSR to `output` pixels; the
    /// source unchanged for the other filters (or when a pass can't be made).
    [[nodiscard]] Source Upscale(const Source& source, Settings::ScalingFilter filter,
                                 std::array<u32, 2> output);

    /// The window pass of `filter` (vulkan_present.vert and the filter's fragment shader) into
    /// `screen` (left, top, right, bottom) of `target`; false when it has none or can't run.
    bool DrawWindow(const Source& source, Settings::ScalingFilter filter, Attachment target,
                    u32 target_format, std::array<u32, 2> target_size, std::array<s32, 4> screen,
                    bool keep_target, BlitBlend blend);

    /// Vulkan's temporal stand-in for the game's DLSS (TextureCacheRuntime::TemporalUpscale
    /// without FSR 3.1): temporal_upscale.frag accumulates this frame into a history at the
    /// output's size, then temporal_sharpen.frag (RCAS at the FSR sharpness setting) draws it.
    struct TemporalInputs {
        TextureHandle color{};
        TextureHandle motion{};
        Attachment target{};
        u32 format{};
        std::array<u32, 2> size{};
        std::array<f32, 2> jitter{};
        std::array<f32, 2> motion_scale{};
        bool use_history{};
    };
    bool TemporalFilter(const TemporalInputs& inputs);

    /// Passes drawn since the last call.
    [[nodiscard]] u64 TakePasses() noexcept;

    /// The shaders the passes use, by role.
    enum class Shader : u8 {
        PresentVertex,
        Bicubic,
        ZeroTangent,
        BSpline,
        Mitchell,
        Spline1,
        Gaussian,
        Lanczos,
        ScaleForce,
        Area,
        Mmpx,
        FsrVertex,
        Easu,
        Rcas,
        SgsrVertex,
        Sgsr,
        SgsrEdge,
        FxaaVertex,
        Fxaa,
        SmaaEdgeVertex,
        SmaaEdge,
        SmaaWeightVertex,
        SmaaWeight,
        SmaaBlendVertex,
        SmaaBlend,
        FullScreenVertex,
        TemporalUpscale,
        TemporalSharpen,
        Count,
    };

    /// Translates and builds one shader; false (logged once) when it can't. For the smoke test.
    bool Prepare(Shader shader);

private:
    struct Program {
        Object library;
        Object function;
        std::vector<ShaderTranslate::ResourceSlot> slots;
        bool failed{};
    };

    struct Pass {
        Shader vertex;
        Shader fragment;
        /// Textures by binding: [binding] is sampled from inputs[binding].
        std::span<const TextureHandle> inputs;
        bool nearest{};
        Attachment target;
        u32 format{};
        std::array<u32, 2> size{};
        bool keep_target{};
        BlitBlend blend{BlitBlend::None};
        std::span<const u8> push_constants;
        u32 vertices{3};
        /// Triangle strip (the window pass's quad) instead of a triangle.
        bool strip{};
    };

    /// A render target of ours, made again when `size` changes.
    struct Target {
        Object texture;
        std::array<u32, 2> size{};
        u32 format{};
    };

    const Program* Get(Shader shader);
    bool Draw(const Pass& pass);
    TextureHandle Make(Target& target, std::array<u32, 2> size, u32 format);
    bool PrepareSmaaTextures();

    const Device& device;
    Scheduler& scheduler;
    std::array<Program, static_cast<size_t>(Shader::Count)> programs;
    std::map<std::tuple<Shader, Shader, u32, BlitBlend>, Object> pipelines;
    Object linear;
    Object nearest;
    Target anti_aliased;
    Target smaa_edges;
    Target smaa_weights;
    Target easu;
    Target rcas;
    Target sgsr;
    Object smaa_area;
    Object smaa_search;
    std::optional<bool> smaa_textures;
    /// The stand-in's history: read from one, stored to the other.
    std::array<Object, 2> history;
    std::array<u32, 2> history_size{};
    size_t history_current{};
    u64 passes{};
};

} // namespace Metal
