// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time's upscale dispatch replaced on Metal, as Vulkan's OcarinaBilinearUpscale does:
// the same host shader (ocarina_bilinear_upscale.comp) through the SPIR-V -> MSL translator.
// Plain C++.

#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/ocarina_motion.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_metal/metal_temporal.h"

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

/// The upscale dispatch replaced by MetalFX temporal upscaling, as Vulkan's OcarinaUpscaler does
/// with its live upscaler on Apple: motion from the scene depth and the camera
/// (ocarina_camera_motion.comp), the scene colour as RGBA16F (ocarina_dlss_color.comp), MetalFX,
/// then the game's target written from its output (ocarina_dlss_output.comp). Link's skinned
/// motion and the effects mask Vulkan adds are not done here: those pixels get the camera's
/// motion only.
class OcarinaTemporal {
public:
    /// The game's pass that reads the scene depth; its descriptors name the depth image.
    static constexpr u64 DepthReader = 0x1fc7a0423116281fULL;
    /// The 1440p pass, which the handheld switch leaves to the 4K rules.
    static constexpr u64 Ocarina1440pShader = 0x6bd913e1f34e83ebULL;

    struct Frame {
        TextureHandle color;  ///< The scene (E5B9G9R9), the extent at its top left
        TextureHandle depth;  ///< The scene depth (D32)
        TextureHandle target; ///< The game's target (B10G11R11, storage)
        std::array<u32, 2> extent;
        std::array<u32, 2> target_size;
        VideoCommon::OcarinaMatrix current;
        VideoCommon::OcarinaMatrix previous;
        u64 frame;
    };

    explicit OcarinaTemporal(const Device& device, Scheduler& scheduler);
    ~OcarinaTemporal();

    /// EDEN_NX2_METAL_OOT_TEMPORAL=0: never (the bilinear pass or the game's pass runs).
    [[nodiscard]] static bool Enabled();

    /// EDEN_NX2_OOT_TEMPORAL_HANDHELD (as on Vulkan; on unless 0): the scaled passes to 1920x1080.
    [[nodiscard]] static bool HandheldEnabled();

    /// A pass Vulkan's DLSS / MetalFX path replaces (CanReplaceOcarinaUpscale without the passes
    /// only FSR 3.1 takes) with one sampled texture and one written image.
    [[nodiscard]] static bool IsTemporalPass(u64 shader_hash, const Shader::Info& info);

    /// The passes that scale to 2560x1440 (Vulkan's upscale_4k set).
    [[nodiscard]] static bool IsScaledPass(u64 shader_hash);

    /// Whether this GPU runs MetalFX temporal upscaling and it has not failed this session.
    [[nodiscard]] bool Available();

    /// Encodes the replacement; false (with `reason`) when the game's pass must run instead.
    bool Record(const Frame& frame, const char*& reason);

    /// The motion pass alone into Motion() (the smoke test).
    bool EncodeMotion(TextureHandle depth, std::array<u32, 2> extent,
                      const VideoCommon::OcarinaMatrix& clip_to_previous_clip);

    [[nodiscard]] TextureHandle Motion() const noexcept {
        return motion.Get();
    }

    /// Passes replaced since the last call.
    [[nodiscard]] u64 TakePasses() noexcept;

private:
    struct Pass {
        Object library;
        Object function;
        Object state;
        std::vector<ShaderTranslate::ResourceSlot> slots;
    };

    bool Prepare();
    bool Resize(std::array<u32, 2> extent);
    void Run(const Pass& pass, TextureHandle input, TextureHandle output,
             std::span<const u8> push_constants, std::array<u32, 2> size);

    const Device& device;
    Scheduler& scheduler;
    TemporalUpscaler upscaler;
    Pass motion_pass;
    Pass color_pass;
    Pass output_pass;
    Object sampler;
    Object motion;
    Object color;
    std::array<u32, 2> size{};
    std::optional<u64> last_frame;
    VideoCommon::OcarinaMatrix last_view_projection{};
    std::optional<bool> supported;
    bool prepared{};
    bool failed{};
    u64 passes{};
};

} // namespace Metal
