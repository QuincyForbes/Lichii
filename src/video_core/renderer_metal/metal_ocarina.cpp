// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/logging.h"
#include "video_core/host_shaders/ocarina_actor_motion_frag_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_vert_spv.h"
#include "video_core/host_shaders/ocarina_bilinear_upscale_comp_spv.h"
#include "video_core/host_shaders/ocarina_camera_motion_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_bias_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_color_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_effects_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_output_comp_spv.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

/// ocarina_actor_motion's Transforms block (std140), as Vulkan's ActorTransforms.
struct ActorTransforms {
    VideoCommon::OcarinaMatrix current;
    VideoCommon::OcarinaMatrix previous;
    std::array<f32, 4> current_origin;
    std::array<f32, 4> previous_origin;
    std::array<f32, 2> extent;
    std::array<f32, 2> jitter;
};
static_assert(sizeof(ActorTransforms) == 176);

/// Bits of float depth Link's replayed pixels may differ from the scene's by (Vulkan's
/// ActorOverlay: matches in the captured replay leveled off there).
constexpr u32 ActorDepthTolerance = 16;
// The passes Vulkan's TryOcarinaBilinear takes: 2560x1440 (menus, gameplay), the gameplay one
// when the scene is under half the target (above 0.4 of it, and 0.4 and below: oot_resolution
// 40 % and 33 %) and the 0xe40f5efb5480e692 one.
constexpr std::array<u64, 5> BilinearShaders{0x4c1ae2fe58f30a97ULL, 0x9302bc0d6d1d9bc1ULL,
                                             0x03bf0ffc7d2d0bedULL, 0x31c0acb7c5e87c89ULL,
                                             0xe40f5efb5480e692ULL};
// Vulkan's ComputePipeline::CanReplaceOcarinaUpscale passes but the full-size and 3/4 ones only
// FSR 3.1 replaces; the first five scale to 2560x1440 (TryOcarinaUpscale's upscale_4k).
constexpr std::array<u64, 6> TemporalShaders{
    OcarinaTemporal::Ocarina1440pShader, 0x4c1ae2fe58f30a97ULL, 0x9302bc0d6d1d9bc1ULL,
    0x03bf0ffc7d2d0bedULL,               0x31c0acb7c5e87c89ULL, 0xe40f5efb5480e692ULL};
constexpr u32 GroupSize = 8;

/// One sampled texture into one written storage image, nothing else written.
bool SamplesIntoOneImage(const Shader::Info& info) {
    return !info.stores_global_memory && info.texture_buffer_descriptors.empty() &&
           info.image_buffer_descriptors.empty() && info.texture_descriptors.size() == 1 &&
           info.texture_descriptors.front().count == 1 && info.image_descriptors.size() == 1 &&
           info.image_descriptors.front().count == 1 && info.image_descriptors.front().is_written &&
           !info.image_descriptors.front().is_read &&
           !info.image_descriptors.front().is_byte_addressed &&
           std::ranges::none_of(info.storage_buffers_descriptors,
                                &Shader::StorageBufferDescriptor::is_written);
}

/// ocarina_camera_motion.comp's push constants: the matrix, then the jitter, padded as MSL lays
/// the block out.
struct MotionParameters {
    VideoCommon::OcarinaMatrix clip_to_previous_clip;
    std::array<f32, 2> jitter;
    std::array<f32, 2> padding;
};
static_assert(sizeof(MotionParameters) == 80);

bool EnvironmentOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

/// One of the host compute shaders through the translator into `pass` (OcarinaTemporal::Pass).
template <typename PassT>
bool BuildPass(const Device& device, PassT& pass, std::span<const u32> code, std::string& error) {
    const ShaderTranslate::Result msl =
        ShaderTranslate::SpirvToMsl(code, ShaderTranslate::Stage::Compute, {});
    error = msl.error;
    if (msl.source.empty()) {
        return false;
    }
    pass.library = CreateLibrary(device, msl.source, error);
    pass.function = pass.library ? CreateFunction(pass.library.Get(), msl.entry_point) : Object{};
    pass.state = pass.function ? CreateComputePipelineState(device, pass.function.Get(), error)
                               : Object{};
    pass.slots = msl.slots;
    return static_cast<bool>(pass.state);
}
} // Anonymous namespace

OcarinaUpscale::OcarinaUpscale(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

OcarinaUpscale::~OcarinaUpscale() = default;

bool OcarinaUpscale::BilinearEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_OOT_BILINEAR_UPSCALE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

bool OcarinaUpscale::IsBilinearPass(u64 shader_hash, const Shader::Info& info) {
    // As Vulkan's ComputePipeline::CanReplaceOcarinaUpscale.
    return std::ranges::find(BilinearShaders, shader_hash) != BilinearShaders.end() &&
           SamplesIntoOneImage(info);
}

bool OcarinaUpscale::Prepare() {
    if (state || failed) {
        return !failed;
    }
    failed = true;
    const ShaderTranslate::Result msl = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(OCARINA_BILINEAR_UPSCALE_COMP_SPV), ShaderTranslate::Stage::Compute,
        {});
    std::string error = msl.error;
    if (!msl.source.empty()) {
        library = CreateLibrary(device, msl.source, error);
        function = library ? CreateFunction(library.Get(), msl.entry_point) : Object{};
        state = function ? CreateComputePipelineState(device, function.Get(), error) : Object{};
    }
    sampler = CreateLinearClampSampler(device);
    if (!state || !sampler) {
        LOG_WARNING(Render_Metal, "Metal: Ocarina bilinear upscale unavailable: {}", error);
        state = {};
        return false;
    }
    slots = msl.slots;
    failed = false;
    return true;
}

bool OcarinaUpscale::Bilinear(TextureHandle color, std::array<u32, 2> color_size,
                              std::array<u32, 2> extent, TextureHandle target,
                              std::array<u32, 2> target_size) {
    if (color == nullptr || target == nullptr || color_size[0] == 0 || color_size[1] == 0 ||
        !Prepare()) {
        return false;
    }
    // ocarina_bilinear_upscale.comp: scene (binding 0), guest_color (binding 1), scene_scale.
    const std::array<f32, 2> scene_scale{
        static_cast<f32>(extent[0]) / static_cast<f32>(color_size[0]),
        static_cast<f32>(extent[1]) / static_cast<f32>(color_size[1]),
    };
    scheduler.BeginCompute(state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, scene_scale.data(),
                                    static_cast<u32>(sizeof(scene_scale)));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding == 0 ? color : target);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({(target_size[0] + GroupSize - 1) / GroupSize,
                        (target_size[1] + GroupSize - 1) / GroupSize, 1},
                       {GroupSize, GroupSize, 1});
    ++passes;
    return true;
}

u64 OcarinaUpscale::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

OcarinaTemporal::OcarinaTemporal(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

OcarinaTemporal::~OcarinaTemporal() = default;

bool OcarinaTemporal::Enabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_OOT_TEMPORAL");
    return enabled;
}

bool OcarinaTemporal::HandheldEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_OOT_TEMPORAL_HANDHELD");
    return enabled;
}

bool OcarinaTemporal::ActorEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_OOT_ACTOR");
    return enabled;
}

bool OcarinaTemporal::EffectsEnabled() {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_OOT_EFFECTS");
    return enabled;
}

bool OcarinaTemporal::IsTemporalPass(u64 shader_hash, const Shader::Info& info) {
    return std::ranges::find(TemporalShaders, shader_hash) != TemporalShaders.end() &&
           SamplesIntoOneImage(info);
}

bool OcarinaTemporal::IsScaledPass(u64 shader_hash) {
    return std::ranges::find(TemporalShaders.begin(), TemporalShaders.begin() + 5, shader_hash) !=
           TemporalShaders.begin() + 5;
}

bool OcarinaTemporal::Available() {
    if (!supported) {
        supported = TemporalUpscaler::IsSupported(device);
    }
    return *supported && !failed;
}

bool OcarinaTemporal::Prepare() {
    if (prepared || failed) {
        return !failed;
    }
    failed = true;
    std::string error;
    sampler = CreateLinearClampSampler(device);
    if (!BuildPass(device, motion_pass, OCARINA_CAMERA_MOTION_COMP_SPV, error) ||
        !BuildPass(device, color_pass, OCARINA_DLSS_COLOR_COMP_SPV, error) ||
        !BuildPass(device, output_pass, OCARINA_DLSS_OUTPUT_COMP_SPV, error) || !sampler) {
        LOG_WARNING(Render_Metal, "Metal: Ocarina temporal upscale unavailable: {}", error);
        return false;
    }
    prepared = true;
    failed = false;
    return true;
}

bool OcarinaTemporal::PrepareActor() {
    if (actor_prepared || actor_failed) {
        return !actor_failed;
    }
    actor_failed = true;
    // No y flip: the viewport below is Vulkan's flipped one (y down from the top), which is
    // Metal's own.
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = DeviceAppleFamily(device);
    options.flip_vertex_y = false;
    std::string error;
    const ShaderTranslate::Result vertex = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(OCARINA_ACTOR_MOTION_VERT_SPV), ShaderTranslate::Stage::Vertex,
        options);
    const ShaderTranslate::Result fragment = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(OCARINA_ACTOR_MOTION_FRAG_SPV), ShaderTranslate::Stage::Fragment,
        options);
    error = !vertex.error.empty() ? vertex.error : fragment.error;
    if (!vertex.source.empty() && !fragment.source.empty()) {
        actor_vertex_library = CreateLibrary(device, vertex.source, error);
        actor_fragment_library =
            actor_vertex_library ? CreateLibrary(device, fragment.source, error) : Object{};
        if (actor_fragment_library) {
            actor_vertex = CreateFunction(actor_vertex_library.Get(), vertex.entry_point);
            actor_fragment = CreateFunction(actor_fragment_library.Get(), fragment.entry_point);
        }
    }
    if (actor_vertex && actor_fragment) {
        // Vulkan's ActorOverlay pipeline: one 48-byte stream, the motion's red and green only,
        // no depth test (the fragment program compares with the scene depth itself), no culling.
        using Attribute = Maxwell::VertexAttribute;
        auto desc = Zeroed<RenderPipelineDesc>();
        desc.vertex_function = actor_vertex.Get();
        desc.fragment_function = actor_fragment.Get();
        desc.color_formats[0] = Fmt::RG16Float;
        desc.blend[0].write_mask = 0b0011;
        desc.samples = 1;
        desc.topology_class = 2;
        desc.rasterize = 1;
        desc.attributes[0] = {1, 0, 0, Attribute::Type::Float, Attribute::Size::Size_R32_G32_B32};
        desc.attributes[4] = {1, 0, 40, Attribute::Type::UNorm, Attribute::Size::Size_R8_G8_B8_A8};
        desc.attributes[5] = {1, 0, 44, Attribute::Type::UNorm, Attribute::Size::Size_R8_G8_B8_A8};
        desc.attributes[6] = {1, 0, 12, Attribute::Type::UInt,
                              Attribute::Size::Size_R16_G16_B16_A16};
        desc.attributes[7] = {1, 0, 20, Attribute::Type::UInt,
                              Attribute::Size::Size_R16_G16_B16_A16};
        desc.streams[0] = {1, OcarinaActorMesh::Stride, 0};
        actor_state = CreateRenderPipelineState(device, desc, error);
    }
    if (!sampler) {
        sampler = CreateLinearClampSampler(device);
    }
    if (!actor_state || !sampler) {
        actor_error = error.empty() ? "no sampler" : error;
        LOG_WARNING(Render_Metal, "Metal: Ocarina actor motion unavailable: {}", error);
        return false;
    }
    actor_vertex_slots = vertex.slots;
    actor_fragment_slots = fragment.slots;
    actor_prepared = true;
    actor_failed = false;
    return true;
}

bool OcarinaTemporal::PrepareEffects() {
    if (effects_prepared || effects_failed) {
        return !effects_failed;
    }
    effects_failed = true;
    std::string error;
    if (!BuildPass(device, effects_mark_pass, OCARINA_DLSS_EFFECTS_COMP_SPV, error) ||
        !BuildPass(device, effects_bias_pass, OCARINA_DLSS_BIAS_COMP_SPV, error) ||
        !Prepare()) {
        LOG_WARNING(Render_Metal, "Metal: Ocarina effects mask unavailable: {}", error);
        return false;
    }
    effects_prepared = true;
    effects_failed = false;
    return true;
}

bool OcarinaTemporal::Resize(std::array<u32, 2> extent) {
    if (extent == size && motion && color) {
        return true;
    }
    const auto make = [&](u32 format, bool render_target) {
        return CreateTexture(device, {.kind = TextureKind::e2D,
                                      .format = format,
                                      .width = extent[0],
                                      .height = extent[1],
                                      .render_target = render_target,
                                      .shader_write = true});
    };
    // The actor overlay draws into the motion.
    motion = make(Fmt::RG16Float, true);
    color = make(Fmt::RGBA16Float, false);
    size = extent;
    return motion && color;
}

void OcarinaTemporal::Run(const Pass& pass, std::span<const TextureHandle> inputs,
                          TextureHandle output, std::span<const u8> push_constants,
                          std::array<u32, 2> extent) {
    // The host shaders: the sampled inputs at bindings 0.., the written image after them.
    scheduler.BeginCompute(pass.state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : pass.slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet && !push_constants.empty()) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, push_constants.data(),
                                    static_cast<u32>(push_constants.size()));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding < inputs.size() ? inputs[slot.binding] : output);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({(extent[0] + GroupSize - 1) / GroupSize,
                        (extent[1] + GroupSize - 1) / GroupSize, 1},
                       {GroupSize, GroupSize, 1});
}

OcarinaTemporal::ActorSlot* OcarinaTemporal::AcquireActorSlot(const OcarinaActorMesh& mesh) {
    // A slot the GPU is done with, the one that holds this mesh first.
    ActorSlot* found = nullptr;
    for (ActorSlot& slot : actor_slots) {
        if (slot.used && !scheduler.IsFree(slot.tick)) {
            continue;
        }
        if (found == nullptr || (slot.used && slot.mesh_key == mesh.mesh_key &&
                                 slot.vertex_bytes == mesh.vertices.size() &&
                                 slot.index_count == mesh.indices.size())) {
            found = &slot;
        }
    }
    if (found == nullptr) {
        return nullptr;
    }
    ActorSlot& slot = *found;
    const size_t bone_bytes = size_t{OcarinaActorMesh::Bones} * 12 * sizeof(f32);
    if (!slot.current) {
        slot.current = CreateBuffer(device, bone_bytes, true, &slot.current_data);
        slot.previous = CreateBuffer(device, bone_bytes, true, &slot.previous_data);
    }
    const bool same_mesh = slot.used && slot.mesh_key == mesh.mesh_key &&
                           slot.vertex_bytes == mesh.vertices.size() &&
                           slot.index_count == mesh.indices.size();
    if (!same_mesh) {
        if (!slot.vertices || slot.vertex_bytes < mesh.vertices.size()) {
            slot.vertices = CreateBuffer(device, mesh.vertices.size(), true, &slot.vertex_data);
        }
        if (!slot.indices || slot.index_count < mesh.indices.size()) {
            slot.indices =
                CreateBuffer(device, mesh.indices.size() * sizeof(u16), true, &slot.index_data);
        }
        if (slot.vertex_data == nullptr || slot.index_data == nullptr) {
            slot = {};
            return nullptr;
        }
        std::memcpy(slot.vertex_data, mesh.vertices.data(), mesh.vertices.size());
        std::memcpy(slot.index_data, mesh.indices.data(), mesh.indices.size() * sizeof(u16));
        slot.vertex_bytes = mesh.vertices.size();
        slot.index_count = mesh.indices.size();
        slot.mesh_key = mesh.mesh_key;
    }
    if (slot.current_data == nullptr || slot.previous_data == nullptr) {
        return nullptr;
    }
    std::memcpy(slot.current_data, mesh.current_bones.data(), bone_bytes);
    std::memcpy(slot.previous_data, mesh.previous_bones.data(), bone_bytes);
    slot.tick = scheduler.CurrentTick();
    slot.used = true;
    return &slot;
}

bool OcarinaTemporal::DrawActor(TextureHandle depth, std::array<u32, 2> extent,
                                const ActorMotion& actor) {
    const OcarinaActorMesh& mesh = actor.mesh;
    const size_t bone_floats = size_t{OcarinaActorMesh::Bones} * 12;
    if (mesh.indices.empty() || mesh.vertices.size() < OcarinaActorMesh::Stride ||
        mesh.current_bones.size() < bone_floats || mesh.previous_bones.size() < bone_floats) {
        actor_error = "mesh or poses incomplete";
        return false;
    }
    if (!PrepareActor()) {
        return false;
    }
    ActorSlot* const slot = AcquireActorSlot(mesh);
    if (slot == nullptr) {
        actor_error = "no free actor slot";
        return false;
    }
    actor_error.clear();
    const ActorTransforms transforms{
        .current = actor.current,
        .previous = actor.previous,
        .current_origin = mesh.current_origin,
        .previous_origin = mesh.previous_origin,
        .extent = {static_cast<f32>(extent[0]), static_cast<f32>(extent[1])},
        .jitter = {},
    };
    RenderTargetSet targets{};
    targets.color[0] = {.texture = motion.Get()};
    targets.width = extent[0];
    targets.height = extent[1];
    scheduler.BeginRendering(targets);
    scheduler.SetDrawState({
        .pipeline = actor_state.Get(),
        .viewport = {0.0f, 0.0f, static_cast<f32>(extent[0]), static_cast<f32>(extent[1]), 0.0f,
                     1.0f},
        .scissor = {0, 0, extent[0], extent[1]},
        .cull_mode = 0, // MTLCullModeNone
        .front_ccw = true,
    });
    scheduler.BindVertexBuffer(VertexBufferSlot(0), slot->vertices.Get(), 0);
    // Vertex: the poses at bindings 0 and 1, the transforms at 2. Fragment: the transforms at
    // 2, the scene depth at 3, the depth tolerance as its push constant.
    for (const ShaderTranslate::ResourceSlot& resource : actor_vertex_slots) {
        if (resource.kind != ShaderTranslate::ResourceKind::Buffer) {
            continue;
        }
        if (resource.binding == 0 || resource.binding == 1) {
            scheduler.BindBuffer(ShaderStage::Vertex, resource.index,
                                 resource.binding == 0 ? slot->current.Get()
                                                       : slot->previous.Get(),
                                 0);
        } else if (resource.binding == 2) {
            scheduler.BindBytes(ShaderStage::Vertex, resource.index, &transforms,
                                static_cast<u32>(sizeof(transforms)));
        }
    }
    for (const ShaderTranslate::ResourceSlot& resource : actor_fragment_slots) {
        switch (resource.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (resource.set == ShaderTranslate::PushConstantSet) {
                scheduler.BindBytes(ShaderStage::Fragment, resource.index, &ActorDepthTolerance,
                                    static_cast<u32>(sizeof(ActorDepthTolerance)));
            } else if (resource.binding == 2) {
                scheduler.BindBytes(ShaderStage::Fragment, resource.index, &transforms,
                                    static_cast<u32>(sizeof(transforms)));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Fragment, resource.index, depth);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Fragment, resource.index, sampler.Get());
            break;
        }
    }
    scheduler.DrawIndexed(*PrimitiveType(Maxwell::PrimitiveTopology::Triangles),
                          static_cast<u32>(mesh.indices.size()), 2, slot->indices.Get(), 0, 1, 0,
                          0);
    scheduler.EndRendering();
    ++actor_frames;
    return true;
}

bool OcarinaTemporal::EncodeMotion(TextureHandle depth, std::array<u32, 2> extent,
                                   const VideoCommon::OcarinaMatrix& clip_to_previous_clip,
                                   const ActorMotion* actor) {
    if (depth == nullptr || extent[0] == 0 || extent[1] == 0 || !Prepare() || !Resize(extent)) {
        return false;
    }
    // No jitter: Vulkan gives MetalFX none either unless EDEN_NX2_DLSS_JITTER is set.
    const MotionParameters parameters{
        .clip_to_previous_clip = clip_to_previous_clip,
        .jitter = {},
        .padding = {},
    };
    Run(motion_pass, depth, motion.Get(),
        std::span(reinterpret_cast<const u8*>(&parameters), sizeof(parameters)), extent);
    // Without Link's pose (dialogue, cutscenes, first person) the motion is the camera's alone.
    if (actor != nullptr && ActorEnabled()) {
        DrawActor(depth, extent, *actor);
    }
    return true;
}

void OcarinaTemporal::SaveEffectsBase(TextureHandle scene, std::array<u32, 2> extent, u64 frame) {
    if (scene == nullptr || extent[0] == 0 || extent[1] == 0 || !PrepareEffects()) {
        return;
    }
    if (effects.extent != extent || !effects.base) {
        const auto make = [&](u32 format) {
            return CreateTexture(device, {.kind = TextureKind::e2D,
                                          .format = format,
                                          .width = extent[0],
                                          .height = extent[1],
                                          .shader_write = true});
        };
        effects = {};
        effects.extent = extent;
        effects.base = make(Fmt::RGBA16Float);
        effects.marks = {make(Fmt::R8Unorm), make(Fmt::R8Unorm)};
        effects.bias = make(Fmt::R16Float);
        if (!effects.base || !effects.marks[0] || !effects.marks[1] || !effects.bias) {
            effects = {};
            return;
        }
    }
    Run(color_pass, scene, effects.base.Get(), {}, extent);
    effects.base_frame = frame;
}

void OcarinaTemporal::MarkEffects(TextureHandle scene, u64 frame) {
    if (scene == nullptr || effects.base_frame != frame) {
        return;
    }
    const std::array<TextureHandle, 2> inputs{effects.base.Get(), scene};
    Run(effects_mark_pass, inputs, effects.marks[frame % 2].Get(), {}, effects.extent);
    effects.previous_marked_frame = effects.marked_frame;
    effects.marked_frame = frame;
}

bool OcarinaTemporal::EncodeEffectsBias(u64 frame, std::array<u32, 2> extent) {
    if (effects.marked_frame != frame || effects.extent != extent || !motion ||
        size != extent) {
        return false;
    }
    const bool has_previous = effects.previous_marked_frame == frame - 1;
    const u64 previous_mark = has_previous ? frame - 1 : frame;
    const std::array<TextureHandle, 3> inputs{effects.marks[frame % 2].Get(),
                                              effects.marks[previous_mark % 2].Get(),
                                              motion.Get()};
    const u32 control = has_previous ? 1U : 0U;
    Run(effects_bias_pass, inputs, effects.bias.Get(),
        std::span(reinterpret_cast<const u8*>(&control), sizeof(control)), extent);
    return true;
}

bool OcarinaTemporal::Record(const Frame& frame, const char*& reason) {
    if (!Available()) {
        reason = "MetalFX temporal unavailable";
        return false;
    }
    if (last_frame == frame.frame) {
        reason = "a second upscale pass this frame";
        return false;
    }
    const auto transform = VideoCommon::OcarinaClipToPreviousClip(frame.current, frame.previous);
    if (!transform) {
        reason = "camera matrix not invertible";
        return false;
    }
    std::optional<ActorMotion> actor;
    if (frame.actor != nullptr) {
        actor.emplace(ActorMotion{*frame.actor, frame.current, frame.previous});
    }
    if (!EncodeMotion(frame.depth, frame.extent, *transform, actor ? &*actor : nullptr)) {
        reason = "motion pass not made";
        return false;
    }
    Run(color_pass, frame.color, color.Get(), {}, frame.extent);
    const bool reactive = EffectsEnabled() && TemporalUpscaler::ReactiveMaskSupported() &&
                          EncodeEffectsBias(frame.frame, frame.extent);
    if (reactive) {
        ++effects_frames;
    }
    const bool reset = !last_frame || frame.frame != *last_frame + 1 ||
                       !std::ranges::equal(frame.previous, last_view_projection,
                                           [](f32 a, f32 b) { return std::abs(a - b) < 0.001f; });
    const TextureHandle output = upscaler.Upscale(device, scheduler,
                                                  {
                                                      .color = color.Get(),
                                                      .depth = frame.depth,
                                                      .motion = motion.Get(),
                                                      .render_size = frame.extent,
                                                      .output_size = frame.target_size,
                                                      .jitter = {},
                                                      .motion_vector_scale = {1.0f, 1.0f},
                                                      .reset = reset,
                                                      .depth_reversed = false,
                                                      .auto_exposure = false,
                                                      .reactive = reactive
                                                                      ? effects.bias.Get()
                                                                      : nullptr,
                                                  });
    if (output == nullptr) {
        failed = true;
        reason = "MetalFX declined the frame";
        return false;
    }
    Run(output_pass, output, frame.target, {}, frame.target_size);
    last_frame = frame.frame;
    last_view_projection = frame.current;
    ++passes;
    return true;
}

u64 OcarinaTemporal::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

std::array<u64, 2> OcarinaTemporal::TakeExtras() noexcept {
    return {std::exchange(actor_frames, 0), std::exchange(effects_frames, 0)};
}

} // namespace Metal
