// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The post-processing chain (dist/post_shaders effects and presets, ReShade FX) on Metal, as
// Vulkan's PostProcessChain (present/post_process.cpp) runs it over a layer before the upscaler:
// the effects compile through the shared reshadefx front end to SPIR-V, which goes through the
// SPIR-V -> MSL translator; each pass is a render pass on the scheduler. Only built with
// HAS_RESHADE. Plain C++.

#pragma once

#ifdef HAS_RESHADE

#include <array>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

class Device;

class PostProcess {
public:
    explicit PostProcess(const Device& device, Scheduler& scheduler);
    ~PostProcess();

    /// Whether the settings ask for a chain (post_shader_enabled and a non-empty chain).
    [[nodiscard]] static bool Wanted();

    /// The texture after the chain, an RGBA16F one of `size`; `source` itself when there is no
    /// chain, or nothing in it could be built.
    [[nodiscard]] TextureHandle Apply(TextureHandle source, std::array<u32, 2> size);

    /// Effects of the chain that built (for the smoke test).
    [[nodiscard]] size_t BuiltEffects() const noexcept {
        return effects.size();
    }

    /// Passes drawn since the last call.
    [[nodiscard]] u64 TakePasses() noexcept;

private:
    enum class UniformKind : u8 { Boolean, Integer, Floating };
    enum class UniformSource : u8 { Value, FrameTime, FrameCount, Timer, Random, PingPong };

    struct UniformWrite {
        std::string name;
        u32 offset{};
        u32 components{};
        UniformKind kind{UniformKind::Floating};
        UniformSource source{UniformSource::Value};
        std::array<f32, 4> fallback{};
        std::array<f32, 4> args{};
        f32 state{};
        f32 direction{1.0f};
    };

    struct Texture {
        std::string name;
        Object texture;
        std::array<u32, 2> size{};
        u32 format{};
        bool is_backbuffer{};
        bool cleared{};
    };

    struct Sampler {
        Object sampler;
        size_t texture_index{};
    };

    struct Program {
        Object library;
        Object function;
        std::vector<ShaderTranslate::ResourceSlot> slots;
    };

    struct Pass {
        Program vertex;
        Program fragment;
        Object pipeline;
        /// Entry point binding -> index into the effect's samplers.
        std::vector<std::pair<u32, size_t>> sampler_bindings;
        size_t target_texture{};
        std::array<u32, 2> size{};
        u32 vertices{3};
        u32 topology{};
        bool clear{};
        bool writes_backbuffer{};
        u32 target_format{};
    };

    struct Effect {
        size_t entry_index{};
        std::string file;
        std::vector<Texture> textures;
        std::vector<Sampler> samplers;
        std::vector<Pass> passes;
        std::vector<UniformWrite> uniforms;
        u32 uniform_size{};
    };

    bool BuildEffects();
    void UpdateUniforms(Effect& effect, std::vector<u8>& out, f32 delta_seconds);
    bool DrawPass(Effect& effect, Pass& pass, TextureHandle backbuffer, TextureHandle target,
                  const std::vector<u8>& uniforms);

    const Device& device;
    Scheduler& scheduler;

    std::vector<Effect> effects;
    std::array<Object, 2> frames;
    std::array<u32, 2> extent{};
    u64 generation{};
    bool built{};
    Object fallback;
    Object fallback_sampler;

    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point previous;
    u64 frame_count{};
    u64 passes{};
};

} // namespace Metal

#endif // HAS_RESHADE
