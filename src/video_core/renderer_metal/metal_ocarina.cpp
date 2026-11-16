// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <span>
#include <string>
#include <utility>

#include "common/logging.h"
#include "video_core/host_shaders/ocarina_bilinear_upscale_comp_spv.h"
#include "video_core/host_shaders/ocarina_camera_motion_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_color_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_output_comp_spv.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"

namespace Metal {

namespace {
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
    const auto build = [&](Pass& pass, std::span<const u32> code) {
        const ShaderTranslate::Result msl =
            ShaderTranslate::SpirvToMsl(code, ShaderTranslate::Stage::Compute, {});
        error = msl.error;
        if (msl.source.empty()) {
            return false;
        }
        pass.library = CreateLibrary(device, msl.source, error);
        pass.function =
            pass.library ? CreateFunction(pass.library.Get(), msl.entry_point) : Object{};
        pass.state =
            pass.function ? CreateComputePipelineState(device, pass.function.Get(), error)
                          : Object{};
        pass.slots = msl.slots;
        return static_cast<bool>(pass.state);
    };
    sampler = CreateLinearClampSampler(device);
    if (!build(motion_pass, OCARINA_CAMERA_MOTION_COMP_SPV) ||
        !build(color_pass, OCARINA_DLSS_COLOR_COMP_SPV) ||
        !build(output_pass, OCARINA_DLSS_OUTPUT_COMP_SPV) || !sampler) {
        LOG_WARNING(Render_Metal, "Metal: Ocarina temporal upscale unavailable: {}", error);
        return false;
    }
    prepared = true;
    failed = false;
    return true;
}

bool OcarinaTemporal::Resize(std::array<u32, 2> extent) {
    if (extent == size && motion && color) {
        return true;
    }
    const auto make = [&](u32 format) {
        return CreateTexture(device, {.kind = TextureKind::e2D,
                                      .format = format,
                                      .width = extent[0],
                                      .height = extent[1],
                                      .shader_write = true});
    };
    motion = make(Fmt::RG16Float);
    color = make(Fmt::RGBA16Float);
    size = extent;
    return motion && color;
}

void OcarinaTemporal::Run(const Pass& pass, TextureHandle input, TextureHandle output,
                          std::span<const u8> push_constants, std::array<u32, 2> extent) {
    // The host shaders: the sampled input at binding 0, the written image at binding 1.
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
                                  slot.binding == 0 ? input : output);
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

bool OcarinaTemporal::EncodeMotion(TextureHandle depth, std::array<u32, 2> extent,
                                   const VideoCommon::OcarinaMatrix& clip_to_previous_clip) {
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
    if (!EncodeMotion(frame.depth, frame.extent, *transform)) {
        reason = "motion pass not made";
        return false;
    }
    Run(color_pass, frame.color, color.Get(), {}, frame.extent);
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

} // namespace Metal
