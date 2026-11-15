// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time's upscale dispatch replaced on Metal, as Vulkan's OcarinaBilinearUpscale does:
// the same host shader (ocarina_bilinear_upscale.comp) through the SPIR-V -> MSL translator.
// Plain C++.

#pragma once

#include <array>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

class Device;
class Scheduler;

class OcarinaUpscale {
public:
    explicit OcarinaUpscale(const Device& device, Scheduler& scheduler);
    ~OcarinaUpscale();

    /// EDEN_NX2_OOT_BILINEAR_UPSCALE=1 (boot; phones get it from their device profile).
    [[nodiscard]] static bool BilinearEnabled();

    /// Whether a compute program is one of the game's upscale passes the bilinear pass takes:
    /// a known shader sampling one texture into one written storage image, nothing else written.
    [[nodiscard]] static bool IsBilinearPass(u64 shader_hash, const Shader::Info& info);

    /// Encodes the pass: `extent` of `color` (color_size) squared into all of `target`
    /// (target_size). False when it can't run (the game's pass runs instead).
    bool Bilinear(TextureHandle color, std::array<u32, 2> color_size, std::array<u32, 2> extent,
                  TextureHandle target, std::array<u32, 2> target_size);

    /// Passes replaced since the last call.
    [[nodiscard]] u64 TakePasses() noexcept;

private:
    bool Prepare();

    const Device& device;
    Scheduler& scheduler;
    Object library;
    Object function;
    Object state;
    Object sampler;
    std::vector<ShaderTranslate::ResourceSlot> slots;
    bool failed{};
    u64 passes{};
};

} // namespace Metal
