// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numbers>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/div_ceil.h"
#include "common/logging.h"
#include "video_core/fsr.h"
#include "video_core/host_shaders/fsr3_upscaler_accumulate_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_accumulate_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_accumulate_sharpen_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_accumulate_sharpen_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_luma_instability_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_luma_instability_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_luma_pyramid_nowave_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_luma_pyramid_nowave_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_prepare_inputs_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_prepare_inputs_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_prepare_reactivity_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_prepare_reactivity_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_rcas_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_rcas_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_shading_change_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_shading_change_ldr_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_shading_change_pyramid_nowave_comp_spv.h"
#include "video_core/host_shaders/fsr3_upscaler_shading_change_pyramid_nowave_ldr_comp_spv.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_fsr3_upscaler.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

// cbFSR3Upscaler (std140; the SDK's Fsr3UpscalerConstants), as Vulkan's.
struct Constants {
    std::array<s32, 2> render_size;
    std::array<s32, 2> previous_render_size;
    std::array<s32, 2> upscale_size;
    std::array<s32, 2> previous_upscale_size;
    std::array<s32, 2> max_render_size;
    std::array<s32, 2> max_upscale_size;
    std::array<f32, 4> device_to_view_depth;
    std::array<f32, 2> jitter;
    std::array<f32, 2> previous_jitter;
    std::array<f32, 2> motion_vector_scale;
    std::array<f32, 2> downscale_factor;
    std::array<f32, 2> motion_vector_jitter_cancellation;
    f32 tan_half_fov;
    f32 jitter_phase_count;
    f32 delta_time;
    f32 delta_pre_exposure;
    f32 view_space_to_meters;
    f32 frame_index;
    f32 velocity_factor;
    f32 reactiveness_scale;
    f32 shading_change_scale;
    f32 accumulation_added_per_frame;
    f32 min_disocclusion_accumulation;
};
static_assert(sizeof(Constants) == 148);

struct SpdConstants {
    u32 mips;
    u32 num_work_groups;
    std::array<u32, 2> work_group_offset;
    std::array<u32, 2> render_size;
};

/// What setBytes is given: the MSL structs are padded past the std140 ones.
constexpr u32 ConstantsBytes = 192;
constexpr u32 SpdBytes = 64;
constexpr u32 RcasBytes = 32;

enum PassId : size_t {
    PassPrepareInputs,
    PassLumaPyramid,
    PassShadingChangePyramid,
    PassShadingChange,
    PassPrepareReactivity,
    PassLumaInstability,
    PassAccumulate,
    PassAccumulateSharpen,
    PassRcas,
};

constexpr u32 SPD_STORED_MIPS = 6;
constexpr f32 CAMERA_NEAR = 0.1f;

s32 JitterPhaseCount(u32 render_width, u32 display_width) {
    const f32 ratio = static_cast<f32>(display_width) / static_cast<f32>(render_width);
    return static_cast<s32>(8.0f * ratio * ratio);
}

/// The compute program's LocalSize execution mode.
std::array<u32, 3> LocalSize(std::span<const u32> words) {
    size_t offset = 5;
    while (offset < words.size()) {
        const u32 length = words[offset] >> 16;
        const u32 opcode = words[offset] & 0xffff;
        if (length == 0 || offset + length > words.size()) {
            break;
        }
        // OpExecutionMode, LocalSize.
        if (opcode == 16 && length >= 6 && words[offset + 2] == 17) {
            return {words[offset + 3], words[offset + 4], words[offset + 5]};
        }
        offset += length;
    }
    return {8, 8, 1};
}

} // Anonymous namespace

Fsr3Upscaler::Fsr3Upscaler(const Device& device_, Scheduler& scheduler_, Profile profile_)
    : device{device_}, scheduler{scheduler_}, profile{profile_} {}

Fsr3Upscaler::~Fsr3Upscaler() = default;

bool Fsr3Upscaler::BuildPass(Pass& pass, std::span<const u32> code, bool atomics) {
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = DeviceAppleFamily(device);
    if (atomics) {
        // Texture atomics (SPD's counter).
        options.msl_minor = 1;
    }
    const ShaderTranslate::Result msl =
        ShaderTranslate::SpirvToMsl(code, ShaderTranslate::Stage::Compute, options);
    std::string error = msl.error;
    if (!msl.source.empty()) {
        pass.library = CreateLibrary(device, msl.source, error);
        pass.function =
            pass.library ? CreateFunction(pass.library.Get(), msl.entry_point) : Object{};
        pass.state =
            pass.function ? CreateComputePipelineState(device, pass.function.Get(), error)
                          : Object{};
    }
    if (!pass.state) {
        if (const char* const dump = std::getenv("LICHII_DUMP_MSL")) {
            std::ofstream{dump, std::ios::app} << "// ---- failing pass\n" << msl.source << "\n";
        }
        LOG_WARNING(Render_Metal, "Metal: FSR 3.1 pass not built: {}", error);
        return false;
    }
    pass.slots = msl.slots;
    pass.group = LocalSize(code);
    return true;
}

bool Fsr3Upscaler::Prepare() {
    if (prepared || failed) {
        return prepared;
    }
    Tegra::Texture::TSCEntry tsc{};
    tsc.wrap_u.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.wrap_v.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.wrap_p.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Nearest);
    point_sampler = CreateSampler(device, tsc);
    tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    linear_sampler = CreateSampler(device, tsc);
    // Metal's SPD reduces without quad shuffles (the SDK's "no wave" variants).
    const bool ldr = profile == Profile::OcarinaLdr;
    failed = !point_sampler || !linear_sampler ||
             !BuildPass(passes[PassPrepareInputs],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_PREPARE_INPUTS_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_PREPARE_INPUTS_COMP_SPV),
                        true) ||
             !BuildPass(passes[PassLumaPyramid],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_LUMA_PYRAMID_NOWAVE_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_LUMA_PYRAMID_NOWAVE_COMP_SPV),
                        true) ||
             !BuildPass(
                 passes[PassShadingChangePyramid],
                 ldr ? std::span<const u32>(FSR3_UPSCALER_SHADING_CHANGE_PYRAMID_NOWAVE_LDR_COMP_SPV)
                     : std::span<const u32>(FSR3_UPSCALER_SHADING_CHANGE_PYRAMID_NOWAVE_COMP_SPV),
                 true) ||
             !BuildPass(passes[PassShadingChange],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_SHADING_CHANGE_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_SHADING_CHANGE_COMP_SPV),
                        false) ||
             !BuildPass(passes[PassPrepareReactivity],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_PREPARE_REACTIVITY_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_PREPARE_REACTIVITY_COMP_SPV),
                        false) ||
             !BuildPass(passes[PassLumaInstability],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_LUMA_INSTABILITY_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_LUMA_INSTABILITY_COMP_SPV),
                        false) ||
             !BuildPass(passes[PassAccumulate],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_ACCUMULATE_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_ACCUMULATE_COMP_SPV),
                        false) ||
             !BuildPass(passes[PassAccumulateSharpen],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_ACCUMULATE_SHARPEN_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_ACCUMULATE_SHARPEN_COMP_SPV),
                        false) ||
             !BuildPass(passes[PassRcas],
                        ldr ? std::span<const u32>(FSR3_UPSCALER_RCAS_LDR_COMP_SPV)
                            : std::span<const u32>(FSR3_UPSCALER_RCAS_COMP_SPV),
                        false);
    prepared = !failed;
    return prepared;
}

Fsr3Upscaler::Texture Fsr3Upscaler::MakeTexture(u32 format, u32 width, u32 height, u32 levels) {
    Texture out;
    out.size = {(std::max)(width, 1U), (std::max)(height, 1U)};
    out.format = format;
    out.level_count = levels;
    const bool lut = format == Fmt::R16Snorm;
    out.texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                         .format = format,
                                         .width = out.size[0],
                                         .height = out.size[1],
                                         .levels = levels,
                                         .render_target = !lut,
                                         .shader_write = !lut,
                                         .pixel_format_view = false});
    if (out.texture && levels > 1) {
        for (u32 level = 0; level < levels; ++level) {
            out.levels.push_back(CreateTextureView(out.texture.Get(), format, TextureKind::e2D,
                                                   level, 1, 0, 1, IdentitySwizzle));
        }
    }
    return out;
}

void Fsr3Upscaler::ClearTexture(const Texture& target, std::array<f64, 4> value) {
    for (u32 level = 0; level < target.level_count; ++level) {
        ClearPass clear{};
        clear.color[0] = Attachment{.texture = target.Handle(), .level = level};
        clear.clear_color[0] = true;
        clear.color_value[0] = value;
        clear.width = (std::max)(target.size[0] >> level, 1U);
        clear.height = (std::max)(target.size[1] >> level, 1U);
        scheduler.Clear(clear);
    }
}

bool Fsr3Upscaler::CreateResources(std::array<u32, 2> render_size,
                                   std::array<u32, 2> upscale_size) {
    const u32 rw = render_size[0];
    const u32 rh = render_size[1];
    const u32 uw = upscale_size[0];
    const u32 uh = upscale_size[1];
    for (size_t i = 0; i < 2; ++i) {
        accumulation[i] = MakeTexture(Fmt::R8Unorm, rw, rh);
        luma[i] = MakeTexture(Fmt::R16Float, rw, rh);
        luma_history[i] = MakeTexture(Fmt::RGBA8Unorm, rw, rh);
        internal_upscaled[i] = MakeTexture(Fmt::RGBA16Float, uw, uh);
    }
    farthest_depth = MakeTexture(Fmt::R16Float, rw, rh);
    shading_change = MakeTexture(Fmt::R8Unorm, rw / 2, rh / 2);
    new_locks = MakeTexture(Fmt::R8Unorm, uw, uh);
    const u32 spd_levels =
        static_cast<u32>(std::floor(std::log2(static_cast<f32>((std::max)(rw / 2, rh / 2))))) + 1;
    spd_mips = MakeTexture(Fmt::RG16Float, rw / 2, rh / 2, (std::max)(spd_levels, SPD_STORED_MIPS));
    farthest_depth_mip1 = MakeTexture(Fmt::R16Float, rw / 2, rh / 2);
    dilated_motion_vectors = MakeTexture(Fmt::RG16Float, rw, rh);
    dilated_depth = MakeTexture(Fmt::R32Float, rw, rh);
    previous_nearest_depth = MakeTexture(Fmt::R32Uint, rw, rh);
    dilated_reactive_masks = MakeTexture(Fmt::RGBA8Unorm, rw, rh);
    frame_info = MakeTexture(Fmt::RGBA32Float, 1, 1);
    spd_atomic = MakeTexture(Fmt::R32Uint, 1, 1);
    lanczos_lut = MakeTexture(Fmt::R16Snorm, 128, 1);
    default_reactive = MakeTexture(Fmt::R8Unorm, rw, rh);
    unit_exposure = MakeTexture(Fmt::RGBA32Float, 1, 1);
    output = MakeTexture(Fmt::RGBA16Float, uw, uh);
    depth_color = MakeTexture(Fmt::R32Float, rw, rh);
    render_extent = render_size;
    upscale_extent = upscale_size;
    first_execution = true;
    frame_parity = 0;
    const auto made = [](const Texture& texture) { return static_cast<bool>(texture.texture); };
    bool ok = true;
    for (const auto* group : {&accumulation, &luma, &luma_history, &internal_upscaled}) {
        for (const Texture& texture : *group) {
            ok = ok && made(texture);
        }
    }
    for (const Texture* texture :
         {&farthest_depth, &shading_change, &new_locks, &spd_mips, &farthest_depth_mip1,
          &dilated_motion_vectors, &dilated_depth, &previous_nearest_depth,
          &dilated_reactive_masks, &frame_info, &spd_atomic, &lanczos_lut, &default_reactive,
          &unit_exposure, &output, &depth_color}) {
        ok = ok && made(*texture);
    }
    if (ok) {
        LOG_INFO(Render_Metal, "FSR 3.1 upscaler: {}x{} -> {}x{}", rw, rh, uw, uh);
    }
    return ok;
}

void Fsr3Upscaler::Dispatch(const Pass& pass, std::span<const Resource> resources, u32 groups_x,
                            u32 groups_y) {
    scheduler.BeginCompute(pass.state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : pass.slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.binding < resources.size() && resources[slot.binding].bytes != nullptr) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index,
                                    resources[slot.binding].bytes, resources[slot.binding].size);
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding < resources.size() ? resources[slot.binding].texture
                                                                  : nullptr);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            // 1000 point clamp, 1001 linear clamp.
            scheduler.BindSampler(ShaderStage::Compute, slot.index,
                                  slot.binding == 1000 ? point_sampler.Get() : linear_sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({groups_x, groups_y, 1}, pass.group);
}

TextureHandle Fsr3Upscaler::Upscale(const Inputs& in) {
    if (!Prepare() || in.color == nullptr || in.depth == nullptr || in.motion == nullptr ||
        in.render_size[0] == 0 || in.render_size[1] == 0) {
        return nullptr;
    }
    if (!output.texture || in.render_size != render_extent || in.upscale_size != upscale_extent) {
        if (output.texture) {
            scheduler.Finish(); // the old textures may still be in use
        }
        if (!CreateResources(in.render_size, in.upscale_size)) {
            failed = true;
            return nullptr;
        }
    }
    const bool first = first_execution;
    const bool reset = in.reset || first;
    first_execution = false;

    const s32 rw = static_cast<s32>(render_extent[0]);
    const s32 rh = static_cast<s32>(render_extent[1]);
    const s32 uw = static_cast<s32>(upscale_extent[0]);
    const s32 uh = static_cast<s32>(upscale_extent[1]);

    // Constants, as fsr3upscalerDispatch computes them (Vulkan's Fsr3Upscaler::Upscale).
    Constants constants{};
    constants.render_size = {rw, rh};
    constants.previous_render_size = first ? constants.render_size : previous_render_size;
    constants.upscale_size = {uw, uh};
    constants.previous_upscale_size = first ? constants.upscale_size : previous_upscale_size;
    constants.max_render_size = constants.render_size;
    constants.max_upscale_size = constants.upscale_size;
    {
        const f32 near_plane = CAMERA_NEAR;
        const f32 fov = in.vertical_fov;
        const f32 aspect = static_cast<f32>(rw) / static_cast<f32>(rh);
        const f32 cot_half_fov = std::cos(0.5f * fov) / std::sin(0.5f * fov);
        constants.device_to_view_depth = {-1.0f * (0.0f + FLT_EPSILON), near_plane,
                                          1.0f / (cot_half_fov / aspect), 1.0f / cot_half_fov};
        if (profile == Profile::OcarinaLdr) {
            constexpr f32 far_plane = 25000.0f;
            constants.device_to_view_depth[0] = far_plane / (far_plane - near_plane);
            constants.device_to_view_depth[1] = -far_plane * near_plane / (far_plane - near_plane);
        }
        constants.tan_half_fov = std::tan(std::atan(std::tan(fov * 0.5f) * aspect));
    }
    constants.jitter = in.jitter;
    constants.previous_jitter = first ? in.jitter : previous_jitter;
    constants.motion_vector_scale = {in.motion_vector_scale[0] / static_cast<f32>(rw),
                                     in.motion_vector_scale[1] / static_cast<f32>(rh)};
    constants.downscale_factor = {static_cast<f32>(rw) / static_cast<f32>(uw),
                                  static_cast<f32>(rh) / static_cast<f32>(uh)};
    constants.motion_vector_jitter_cancellation = {0.0f, 0.0f};
    const s32 phase_count = JitterPhaseCount(render_extent[0], upscale_extent[0]);
    if (reset || jitter_phase_count == 0.0f) {
        jitter_phase_count = static_cast<f32>(phase_count);
    } else if (static_cast<f32>(phase_count) > jitter_phase_count) {
        jitter_phase_count += 1.0f;
    } else if (static_cast<f32>(phase_count) < jitter_phase_count) {
        jitter_phase_count -= 1.0f;
    }
    constants.jitter_phase_count = jitter_phase_count;
    constants.delta_time = std::clamp(in.frame_time_ms / 1000.0f, 0.0f, 1.0f);
    constants.delta_pre_exposure = 1.0f;
    constants.view_space_to_meters = 1.0f;
    frame_index = reset ? 0.0f : frame_index + 1.0f;
    constants.frame_index = frame_index;
    // The SDK's defaults (FFX_FSR3UPSCALER_CONFIGURE_UPSCALE_KEY_*).
    constants.velocity_factor = 1.0f;
    constants.reactiveness_scale = 1.0f;
    constants.shading_change_scale = 1.0f;
    constants.accumulation_added_per_frame = 1.0f / 3.0f;
    constants.min_disocclusion_accumulation = -1.0f / 3.0f;
    previous_render_size = constants.render_size;
    previous_upscale_size = constants.upscale_size;
    previous_jitter = in.jitter;

    // ffxSpdSetup over the whole render area.
    const u32 spd_groups_x = (render_extent[0] - 1) / 64 + 1;
    const u32 spd_groups_y = (render_extent[1] - 1) / 64 + 1;
    const SpdConstants spd{
        .mips = static_cast<u32>((std::min)(
            std::floor(std::log2(static_cast<f32>((std::max)(render_extent[0], render_extent[1])))),
            12.0f)),
        .num_work_groups = spd_groups_x * spd_groups_y,
        .work_group_offset = {0, 0},
        .render_size = {render_extent[0], render_extent[1]},
    };
    std::array<u32, 4> rcas_config{};
    FSR::FsrRcasCon(rcas_config.data(), 2.0f - 2.0f * std::clamp(in.sharpness, 0.0f, 1.0f));

    // The constants go to setBytes padded to what the MSL structs take.
    std::array<u8, ConstantsBytes> constants_bytes{};
    std::array<u8, SpdBytes> spd_bytes{};
    std::array<u8, RcasBytes> rcas_bytes{};
    std::memcpy(constants_bytes.data(), &constants, sizeof(constants));
    std::memcpy(spd_bytes.data(), &spd, sizeof(spd));
    std::memcpy(rcas_bytes.data(), rcas_config.data(), sizeof(rcas_config));
    const Resource cb_fsr{.bytes = constants_bytes.data(), .size = ConstantsBytes};
    const Resource cb_spd{.bytes = spd_bytes.data(), .size = SpdBytes};
    const Resource cb_rcas{.bytes = rcas_bytes.data(), .size = RcasBytes};

    // Clears: everything when the textures are new, the history on a reset, and the per-frame
    // scratch (previous nearest depth, the SPD counter and mips) always.
    const size_t p = frame_parity;
    const size_t q = p ^ 1;
    if (first) {
        for (const auto* group : {&accumulation, &luma, &luma_history, &internal_upscaled}) {
            for (const Texture& texture : *group) {
                ClearTexture(texture, {0, 0, 0, 0});
            }
        }
        for (const Texture* texture :
             {&farthest_depth, &shading_change, &new_locks, &farthest_depth_mip1,
              &dilated_motion_vectors, &dilated_depth, &dilated_reactive_masks, &default_reactive,
              &output}) {
            ClearTexture(*texture, {0, 0, 0, 0});
        }
        ClearTexture(unit_exposure, {1, 0, 0, 0});
        // FSR's Lanczos-2 table (fsr3upscalerCreate).
        std::array<s16, 128> weights{};
        for (size_t i = 0; i < weights.size(); ++i) {
            const f32 x = 2.0f * static_cast<f32>(i) / static_cast<f32>(weights.size() - 1);
            const f32 pi_x = std::numbers::pi_v<f32> * x;
            const f32 y = std::abs(x) < FLT_EPSILON
                              ? 1.0f
                              : (std::sin(pi_x) / pi_x) * (std::sin(0.5f * pi_x) / (0.5f * pi_x));
            weights[i] = static_cast<s16>(std::round(y * 32767.0f));
        }
        u8* upload = nullptr;
        const Object lut_buffer = CreateBuffer(device, sizeof(weights), true, &upload);
        if (!lut_buffer || upload == nullptr) {
            failed = true;
            return nullptr;
        }
        std::memcpy(upload, weights.data(), sizeof(weights));
        scheduler.CopyBufferToTexture(lut_buffer.Get(), 0, sizeof(weights), 0,
                                      lanczos_lut.Handle(), 0, 0, {}, {128, 1, 1});
        // The upload is read from the buffer by the copy: it must be done before the buffer goes.
        scheduler.Finish();
    }
    if (reset) {
        ClearTexture(accumulation[p], {0, 0, 0, 0});
        ClearTexture(frame_info, {-1, 1, 0, 0});
    }
    ClearTexture(spd_atomic, {0, 0, 0, 0});
    ClearTexture(spd_mips, {0, 0, 0, 0});
    // The standard-depth profile clears to 1.0f's bits, the reversed one to 0.
    const f64 depth_clear = profile == Profile::OcarinaLdr ? static_cast<f64>(0x3f800000U) : 0.0;
    ClearTexture(previous_nearest_depth, {depth_clear, 0, 0, 0});

    const auto tex = [](const Texture& texture) { return Resource{.texture = texture.Handle()}; };
    const auto mip = [&](u32 level) {
        return Resource{.texture =
                            spd_mips.levels[(std::min)(level, static_cast<u32>(spd_mips.levels.size() - 1))]
                                .Get()};
    };
    const Resource color{.texture = in.color};
    // The depth as a colour texture: Metal reads a depth texture as depth2d, which the program's
    // shared image type can't be.
    if (!scheduler.Convert(ConvertPass{.src = in.depth,
                                       .dst = Attachment{.texture = depth_color.Handle()},
                                       .dst_format = Fmt::R32Float,
                                       .width = render_extent[0],
                                       .height = render_extent[1],
                                       .kind = ConvertKind::DepthToColor})) {
        failed = true;
        return nullptr;
    }
    const Resource depth{.texture = depth_color.Handle()};
    const Resource motion{.texture = in.motion};
    const Resource exposure = tex(profile == Profile::OcarinaLdr ? unit_exposure : frame_info);
    const Resource reactive =
        in.reactive != nullptr ? Resource{.texture = in.reactive} : tex(default_reactive);
    const Resource transparency =
        in.transparency != nullptr ? Resource{.texture = in.transparency} : tex(default_reactive);
    const Resource current_luma = tex(luma[p]);
    const auto groups = [](s32 size, u32 tile) {
        return Common::DivCeil(static_cast<u32>(size), tile);
    };

    Dispatch(passes[PassPrepareInputs],
             std::array{motion, depth, color, tex(dilated_motion_vectors), tex(dilated_depth),
                        tex(previous_nearest_depth), tex(farthest_depth), current_luma, cb_fsr},
             groups(rw, 8), groups(rh, 8));
    Dispatch(passes[PassLumaPyramid],
             std::array{current_luma, tex(farthest_depth), tex(spd_atomic), tex(frame_info),
                        mip(0), mip(1), mip(2), mip(3), mip(4), mip(5), tex(farthest_depth_mip1),
                        cb_fsr, cb_spd},
             spd_groups_x, spd_groups_y);
    Dispatch(passes[PassShadingChangePyramid],
             std::array{current_luma, tex(luma[q]), tex(dilated_motion_vectors), exposure,
                        tex(spd_atomic), mip(0), mip(1), mip(2), mip(3), mip(4), mip(5), cb_fsr,
                        cb_spd},
             spd_groups_x, spd_groups_y);
    Dispatch(passes[PassShadingChange], std::array{tex(spd_mips), tex(shading_change), cb_fsr},
             groups(rw / 2, 8), groups(rh / 2, 8));
    Dispatch(passes[PassPrepareReactivity],
             std::array{tex(previous_nearest_depth), tex(dilated_motion_vectors),
                        tex(dilated_depth), reactive, transparency,
                        tex(accumulation[p]), tex(shading_change), current_luma, exposure,
                        tex(dilated_reactive_masks), tex(new_locks), tex(accumulation[q]), cb_fsr},
             groups(rw, 8), groups(rh, 8));
    Dispatch(passes[PassLumaInstability],
             std::array{exposure, tex(dilated_reactive_masks), tex(dilated_motion_vectors),
                        tex(frame_info), tex(luma_history[p]), tex(farthest_depth_mip1),
                        current_luma, tex(luma_history[q]), tex(farthest_depth), cb_fsr},
             groups(rw, 8), groups(rh, 8));
    Dispatch(passes[in.sharpen ? PassAccumulateSharpen : PassAccumulate],
             std::array{exposure, tex(dilated_reactive_masks), tex(dilated_motion_vectors),
                        tex(internal_upscaled[p]), tex(lanczos_lut), tex(farthest_depth_mip1),
                        current_luma, tex(farthest_depth), color, tex(internal_upscaled[q]),
                        tex(output), tex(new_locks), cb_fsr},
             groups(uw, 8), groups(uh, 8));
    if (in.sharpen) {
        Dispatch(passes[PassRcas],
                 std::array{exposure, tex(internal_upscaled[q]), tex(output), cb_fsr, cb_rcas},
                 groups(uw, 16), groups(uh, 16));
    }
    frame_parity = static_cast<u32>(q);
    return output.Handle();
}

} // namespace Metal
