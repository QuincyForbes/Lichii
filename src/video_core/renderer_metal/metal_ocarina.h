// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Ocarina of Time's upscale dispatch replaced on Metal, as Vulkan's OcarinaBilinearUpscale does:
// the same host shader (ocarina_bilinear_upscale.comp) through the SPIR-V -> MSL translator.
// Plain C++.

#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
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

/// Link's skinned mesh and his poses of this frame and the one before (Vulkan's
/// OcarinaActorMesh): vertices of 48 bytes from the lowest index the draw reads, indices rebased
/// to it, and 280 row-major 4x3 bone matrices per pose.
struct OcarinaActorMesh {
    static constexpr u32 Stride = 48;
    static constexpr u32 Bones = 280;

    std::span<const u8> vertices;
    std::span<const u16> indices;
    /// Changes whenever the vertices or indices do (their guest addresses), so they are
    /// uploaded again only then.
    u64 mesh_key{};
    std::array<f32, 4> current_origin{};
    std::array<f32, 4> previous_origin{};
    std::span<const f32> current_bones;
    std::span<const f32> previous_bones;
};

/// The upscale dispatch replaced by MetalFX temporal upscaling, as Vulkan's OcarinaUpscaler does
/// with its live upscaler on Apple: motion from the scene depth and the camera
/// (ocarina_camera_motion.comp), Link's skinned motion drawn over it where his pixels match the
/// scene depth (ocarina_actor_motion.vert/.frag, EDEN_NX2_METAL_OOT_ACTOR), the scene colour as
/// RGBA16F (ocarina_dlss_color.comp), the glows and particles drawn after the fog as MetalFX's
/// reactive mask (ocarina_dlss_effects.comp, ocarina_dlss_bias.comp, EDEN_NX2_METAL_OOT_EFFECTS),
/// MetalFX, then the game's target written from its output (ocarina_dlss_output.comp).
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
        /// Link's mesh and poses, null without them (dialogue, cutscenes, first person).
        const OcarinaActorMesh* actor{};
        /// The scene draws' viewport offset this frame (render pixels; EDEN_NX2_DLSS_JITTER).
        std::array<f32, 2> jitter{};
    };

    explicit OcarinaTemporal(const Device& device, Scheduler& scheduler);
    ~OcarinaTemporal();

    /// EDEN_NX2_METAL_OOT_TEMPORAL=0: never (the bilinear pass or the game's pass runs).
    [[nodiscard]] static bool Enabled();

    /// EDEN_NX2_OOT_TEMPORAL_HANDHELD (as on Vulkan; on unless 0): the scaled passes to 1920x1080.
    [[nodiscard]] static bool HandheldEnabled();

    /// EDEN_NX2_METAL_OOT_ACTOR=0: Link's pixels get the camera's motion only (O09).
    [[nodiscard]] static bool ActorEnabled();

    /// EDEN_NX2_METAL_OOT_EFFECTS=0: no reactive mask for the glows and particles (O10).
    [[nodiscard]] static bool EffectsEnabled();

    /// Vulkan's TrackOcarinaActor match: the vertex program that skins Link.
    static constexpr u64 ActorVertexShader = 0x28c6e527634f814aULL;
    /// The fog's fragment program: the glows and particles come right after it.
    static constexpr u64 FogFragmentShader = 0x3ddfb1d7a854c849ULL;

    /// A pass Vulkan's DLSS / MetalFX path replaces (CanReplaceOcarinaUpscale without the passes
    /// only FSR 3.1 takes) with one sampled texture and one written image.
    [[nodiscard]] static bool IsTemporalPass(u64 shader_hash, const Shader::Info& info);

    /// The passes that scale to 2560x1440 (Vulkan's upscale_4k set).
    [[nodiscard]] static bool IsScaledPass(u64 shader_hash);

    /// Whether this GPU runs MetalFX temporal upscaling and it has not failed this session.
    [[nodiscard]] bool Available();

    /// Encodes the replacement; false (with `reason`) when the game's pass must run instead.
    bool Record(const Frame& frame, const char*& reason);

    /// Link's pose for the overlay: his mesh, this frame's and the last frame's camera.
    struct ActorMotion {
        const OcarinaActorMesh& mesh;
        const VideoCommon::OcarinaMatrix& current;
        const VideoCommon::OcarinaMatrix& previous;
    };

    /// The motion pass into Motion(), then Link's skinned motion over it with `actor` (the
    /// smoke test runs this alone).
    bool EncodeMotion(TextureHandle depth, std::array<u32, 2> extent,
                      const VideoCommon::OcarinaMatrix& clip_to_previous_clip,
                      const ActorMotion* actor = nullptr, std::array<f32, 2> jitter = {});

    /// Vulkan's SaveEffectsBase: the scene (B10G11R11, `extent` at its top left) copied once the
    /// fog is drawn, before the glows and particles.
    void SaveEffectsBase(TextureHandle scene, std::array<u32, 2> extent, u64 frame);

    /// Vulkan's MarkEffects: the pixels the glows and particles changed since SaveEffectsBase of
    /// the same frame, before the game's next pass writes the scene.
    void MarkEffects(TextureHandle scene, u64 frame);

    /// The reactive mask of `frame` into EffectsBias() from its marks (and the last frame's,
    /// moved along Motion()); false without marks of this frame at `extent`.
    bool EncodeEffectsBias(u64 frame, std::array<u32, 2> extent);

    [[nodiscard]] TextureHandle Motion() const noexcept {
        return motion.Get();
    }

    [[nodiscard]] TextureHandle EffectsBias() const noexcept {
        return effects.bias.Get();
    }

    /// Passes replaced since the last call.
    [[nodiscard]] u64 TakePasses() noexcept;

    /// Since the last call: frames with Link's motion drawn, frames with a reactive mask.
    [[nodiscard]] std::array<u64, 2> TakeExtras() noexcept;

    /// The actor motion programs as the translator makes them (vertex, fragment), with their
    /// resource slots appended; for the smoke test.
    [[nodiscard]] static std::array<std::string, 2> ActorMsl(u32 apple_family);

    /// Why Link's motion was last not drawn (empty when it was), for the smoke test.
    [[nodiscard]] const std::string& ActorError() const noexcept {
        return actor_error;
    }

private:
    struct Pass {
        Object library;
        Object function;
        Object state;
        std::vector<ShaderTranslate::ResourceSlot> slots;
    };

    /// The overlay's buffers; one per command buffer in flight.
    struct ActorSlot {
        Object vertices;
        Object indices;
        Object current;
        Object previous;
        u8* vertex_data{};
        u8* index_data{};
        u8* current_data{};
        u8* previous_data{};
        size_t vertex_bytes{};
        size_t index_count{};
        u64 mesh_key{};
        u64 tick{};
        bool used{};
    };

    struct Effects {
        std::array<u32, 2> extent{};
        Object base;
        std::array<Object, 2> marks;
        Object bias;
        std::optional<u64> base_frame;
        std::optional<u64> marked_frame;
        std::optional<u64> previous_marked_frame;
    };

    bool Prepare();
    bool PrepareActor();
    bool PrepareEffects();
    bool Resize(std::array<u32, 2> extent);
    void Run(const Pass& pass, std::span<const TextureHandle> inputs, TextureHandle output,
             std::span<const u8> push_constants, std::array<u32, 2> size);
    void Run(const Pass& pass, TextureHandle input, TextureHandle output,
             std::span<const u8> push_constants, std::array<u32, 2> pass_size) {
        Run(pass, std::span(&input, 1), output, push_constants, pass_size);
    }
    bool DrawActor(TextureHandle depth, std::array<u32, 2> extent, const ActorMotion& actor,
                   std::array<f32, 2> jitter);
    [[nodiscard]] ActorSlot* AcquireActorSlot(const OcarinaActorMesh& mesh);

    const Device& device;
    Scheduler& scheduler;
    TemporalUpscaler upscaler;
    Pass motion_pass;
    Pass color_pass;
    Pass output_pass;
    Pass effects_mark_pass;
    Pass effects_bias_pass;
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

    Object actor_vertex_library;
    Object actor_fragment_library;
    Object actor_vertex;
    Object actor_fragment;
    Object actor_state;
    std::vector<ShaderTranslate::ResourceSlot> actor_vertex_slots;
    std::vector<ShaderTranslate::ResourceSlot> actor_fragment_slots;
    std::array<ActorSlot, 4> actor_slots;
    bool actor_prepared{};
    bool actor_failed{};
    std::string actor_error;
    u64 actor_frames{};

    Effects effects;
    bool effects_prepared{};
    bool effects_failed{};
    u64 effects_frames{};
};

} // namespace Metal
