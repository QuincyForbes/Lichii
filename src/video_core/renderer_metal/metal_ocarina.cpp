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
#include "common/settings.h"
#include "video_core/fsr.h"
#include "video_core/ocarina_upscale.h"
#include "video_core/host_shaders/camera_motion_vectors_comp_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_frag_spv.h"
#include "video_core/host_shaders/ocarina_actor_motion_vert_spv.h"
#include "video_core/host_shaders/ocarina_bilinear_upscale_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_composite_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_4k_reference_comp_spv.h"
#include "video_core/host_shaders/ocarina_dlss_bias_comp_spv.h"
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
/// EDEN_NX2_METAL_OOT_ACTOR_TOLERANCE overrides it (4294967295: no comparison).
u32 ActorDepthTolerance() {
    static const u32 tolerance = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_OOT_ACTOR_TOLERANCE");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : u32{16};
    }();
    return tolerance;
}
// The passes Vulkan's TryOcarinaBilinear takes: 2560x1440 (menus, gameplay), the gameplay one
// when the scene is under half the target (above 0.4 of it, and 0.4 and below: oot_resolution
// 40 % and 33 %) and the 0xe40f5efb5480e692 one.
[[maybe_unused]] constexpr std::array<u64, 5> BilinearShaders{
    0x4c1ae2fe58f30a97ULL, 0x9302bc0d6d1d9bc1ULL, 0x03bf0ffc7d2d0bedULL, 0x31c0acb7c5e87c89ULL,
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

/// Camera motion and fused input-preparation push constants: row-major reprojection matrix,
/// subpixel jitter and logical render extent (allowing padded backing textures without cropping).
struct MotionParameters {
    VideoCommon::OcarinaMatrix clip_to_previous_clip;
    std::array<f32, 2> jitter;
    std::array<u32, 2> extent;
};
static_assert(sizeof(MotionParameters) == 80);

constexpr std::string_view OcarinaMslSource = R"(
#include <metal_stdlib>
using namespace metal;

struct MotionParams {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
    float2 jitter;
    uint2 extent;
};

static inline float2 compute_camera_motion(depth2d<float, access::sample> scene_depth,
                                           constant MotionParams& params, uint2 gid) {
    const float2 size = float2(params.extent);
    const float depth = scene_depth.read(gid, 0);
    const float2 screen = (float2(gid) + 0.5f - params.jitter) / size;
    const float4 clip = float4(screen * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f),
                               depth * 2.0f - 1.0f, 1.0f);
    const float4 previous = float4(dot(params.row0, clip), dot(params.row1, clip),
                                   dot(params.row2, clip), dot(params.row3, clip));
    float2 pixels = float2(0.0f);
    if (depth >= 0.0f && depth <= 1.0f && previous.w > 1e-6f) {
        const float2 displacement =
            (previous.xy / previous.w - clip.xy) * float2(0.5f, -0.5f) * size;
        if (all(abs(displacement) <= float2(65504.0f))) {
            pixels = displacement;
        }
    }
    return pixels;
}

static inline float4 sanitize_ldr_color(float3 color) {
    const uint3 exponent = as_type<uint3>(color) & uint3(0x7f800000u);
    const float3 finite_color = select(float3(0.0f), color, exponent != uint3(0x7f800000u));
    return float4(clamp(finite_color, float3(0.0f), float3(1.0f)), 1.0f);
}

kernel void ocarina_camera_motion(
    depth2d<float, access::sample> scene_depth [[texture(0)]],
    texture2d<float, access::write> motion [[texture(1)]],
    constant MotionParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= motion.get_width() || gid.y >= motion.get_height()) {
        return;
    }
    if (gid.x >= params.extent.x || gid.y >= params.extent.y) {
        motion.write(float4(0.0f), gid);
        return;
    }
    motion.write(float4(compute_camera_motion(scene_depth, params, gid), 0.0f, 0.0f), gid);
}

kernel void ocarina_prepare_inputs(
    depth2d<float, access::sample> scene_depth [[texture(0)]],
    texture2d<float, access::sample> guest_color [[texture(1)]],
    texture2d<float, access::write> motion [[texture(2)]],
    texture2d<float, access::write> dlss_color [[texture(3)]],
    constant MotionParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= motion.get_width() || gid.y >= motion.get_height()) {
        return;
    }
    if (gid.x >= params.extent.x || gid.y >= params.extent.y) {
        const uint2 clamped = min(gid, max(params.extent, uint2(1u)) - 1u);
        motion.write(float4(0.0f), gid);
        dlss_color.write(sanitize_ldr_color(guest_color.read(clamped, 0).rgb), gid);
        return;
    }
    motion.write(float4(compute_camera_motion(scene_depth, params, gid), 0.0f, 0.0f), gid);
    dlss_color.write(sanitize_ldr_color(guest_color.read(gid, 0).rgb), gid);
}

kernel void ocarina_direct_output(
    texture2d<float, access::sample> dlss_color [[texture(0)]],
    texture2d<float, access::write> guest_color [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= guest_color.get_width() || gid.y >= guest_color.get_height()) {
        return;
    }
    const float3 color = max(dlss_color.read(gid, 0).rgb, float3(0.0f));
    guest_color.write(float4(color * color, 1.0f), gid);
}
)";

bool EnvironmentOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

Object MakePointClampSampler(const Device& device) {
    Tegra::Texture::TSCEntry point_tsc{};
    point_tsc.wrap_u.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    point_tsc.wrap_v.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    point_tsc.wrap_p.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    point_tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    point_tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    point_tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::None);
    return CreateSampler(device, point_tsc);
}

bool IsCameraCut(const VideoCommon::OcarinaMatrix& m) {
    const auto jumped = [&](f32 clip_z) {
        const f32 x = clip_z * m[2] + m[3];
        const f32 y = clip_z * m[6] + m[7];
        const f32 w = clip_z * m[14] + m[15];
        if (!std::isfinite(w) || w <= 0.2f || w >= 5.0f) {
            return true;
        }
        return std::abs(x / w) > 0.75f || std::abs(y / w) > 0.75f;
    };
    return jumped(0.95f) || jumped(1.0f);
}

/// One of the host compute shaders through the translator into `pass` (OcarinaTemporal::Pass).
template <typename PassT>
bool BuildPass(const Device& device, PassT& pass, std::span<const u32> code, std::string& error,
               const ShaderTranslate::Options& options = {}) {
    const ShaderTranslate::Result msl =
        ShaderTranslate::SpirvToMsl(code, ShaderTranslate::Stage::Compute, options);
    error = msl.error;
    if (msl.source.empty()) {
        return false;
    }
    pass.library = CreateLibrary(device, msl.source, error);
    pass.function = pass.library ? CreateFunction(pass.library.Get(), msl.entry_point) : Object{};
    pass.state =
        pass.function
            ? CreateComputePipelineState(device, pass.function.Get(), error, nullptr,
                                         GroupSize * GroupSize)
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

bool OcarinaUpscale::IsBilinearPass(u64 native_key, u64 shader_hash, const Shader::Info& info) {
    // As Vulkan's TryOcarinaBilinear.
    const auto pass = VideoCommon::FindOcarinaUpscalePass(native_key, shader_hash);
    return (pass == VideoCommon::OcarinaUpscalePass::Scaled ||
            pass == VideoCommon::OcarinaUpscalePass::ThreeQuarter) &&
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
        state = function ? CreateComputePipelineState(device, function.Get(), error, nullptr,
                                                      GroupSize * GroupSize)
                         : Object{};
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
                                  slot.binding == 0 ? color : target, false, slot.binding != 0);
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

CameraMotionPass::CameraMotionPass(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

CameraMotionPass::~CameraMotionPass() = default;

namespace {
/// camera_motion_vectors.comp's options: the depth (set 0, binding 0) is a depth texture.
ShaderTranslate::Options CameraMotionOptions() {
    ShaderTranslate::Options options{};
    options.depth_images = {{0, 0}};
    return options;
}

/// camera_motion_vectors.comp's push constants: the matrix, then the size, padded as MSL lays the
/// block out.
struct CameraMotionParameters {
    std::array<f32, 16> reprojection;
    std::array<s32, 2> size;
    std::array<s32, 2> padding;
};
static_assert(sizeof(CameraMotionParameters) == 80);
} // Anonymous namespace

std::string CameraMotionPass::Msl(u32 apple_family) {
    ShaderTranslate::Options options = CameraMotionOptions();
    options.ios = TranslateForIos;
    options.apple_family = apple_family;
    const ShaderTranslate::Result result = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(CAMERA_MOTION_VECTORS_COMP_SPV), ShaderTranslate::Stage::Compute,
        options);
    std::string text = result.source.empty() ? "error: " + result.error : result.source;
    for (const ShaderTranslate::ResourceSlot& slot : result.slots) {
        text += "\n// slot set " + std::to_string(slot.set) + " binding " +
                std::to_string(slot.binding) + " kind " +
                std::to_string(static_cast<int>(slot.kind)) + " index " +
                std::to_string(slot.index);
    }
    return text;
}

bool CameraMotionPass::Prepare() {
    if (state || failed) {
        return !failed;
    }
    failed = true;
    const ShaderTranslate::Result msl = ShaderTranslate::SpirvToMsl(
        std::span<const u32>(CAMERA_MOTION_VECTORS_COMP_SPV), ShaderTranslate::Stage::Compute,
        CameraMotionOptions());
    std::string error = msl.error;
    if (!msl.source.empty()) {
        library = CreateLibrary(device, msl.source, error);
        function = library ? CreateFunction(library.Get(), msl.entry_point) : Object{};
        state = function ? CreateComputePipelineState(device, function.Get(), error, nullptr,
                                                      GroupSize * GroupSize)
                         : Object{};
    }
    // The shader fetches texels (Vulkan binds a nearest sampler for it).
    sampler = CreateLinearClampSampler(device);
    if (!state || !sampler) {
        LOG_WARNING(Render_Metal, "Metal: camera motion vectors unavailable: {}", error);
        state = {};
        return false;
    }
    slots = msl.slots;
    failed = false;
    return true;
}

bool CameraMotionPass::Record(TextureHandle depth, std::array<u32, 2> extent,
                              const std::array<f32, 16>& reprojection) {
    if (depth == nullptr || extent[0] == 0 || extent[1] == 0 || !Prepare()) {
        return false;
    }
    if (extent != size || !motion) {
        motion = CreateTexture(device, {.kind = TextureKind::e2D,
                                        .format = Fmt::RG16Float,
                                        .width = extent[0],
                                        .height = extent[1],
                                        .shader_write = true,
                                        .pixel_format_view = false});
        size = extent;
        if (!motion) {
            return false;
        }
    }
    const CameraMotionParameters parameters{
        .reprojection = reprojection,
        .size = {static_cast<s32>(extent[0]), static_cast<s32>(extent[1])},
        .padding = {},
    };
    // camera_motion_vectors.comp: depth_image (binding 0), motion_image (binding 1).
    scheduler.BeginCompute(state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, &parameters,
                                    static_cast<u32>(sizeof(parameters)));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  slot.binding == 0 ? depth : motion.Get(), false,
                                  slot.binding != 0);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, sampler.Get());
            break;
        }
    }
    scheduler.Dispatch({(extent[0] + GroupSize - 1) / GroupSize,
                        (extent[1] + GroupSize - 1) / GroupSize, 1},
                       {GroupSize, GroupSize, 1});
    return true;
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

bool OcarinaTemporal::Present4kEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_OOT_4K");
        return value != nullptr && value[0] == '1';
    }();
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

VideoCommon::OcarinaUpscaleBackend OcarinaTemporal::ResolveBackend(const Device& device,
                                                                   bool metalfx_failed,
                                                                   bool fsr_failed) {
    using Backend = VideoCommon::OcarinaUpscaleBackend;
    if (!Enabled()) {
        return Backend::Native;
    }
    const bool metalfx_ok = !metalfx_failed && TemporalUpscaler::IsSupported(device);
    const bool fsr_ok = !fsr_failed;
    if (const char* const raw = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        raw != nullptr && raw[0] != '\0') {
        const std::string_view override{raw};
        if (override == "native") {
            return Backend::Native;
        }
        if (override == "fsr") {
            return fsr_ok ? Backend::Fsr : Backend::Native;
        }
        if (override == "dlss" || override == "metalfx") {
            return metalfx_ok ? Backend::Dlss : (fsr_ok ? Backend::Fsr : Backend::Native);
        }
    }
    switch (Settings::values.upscaler.GetValue()) {
    case Settings::Upscaler::Game:
        return Backend::Native;
    case Settings::Upscaler::MetalFx:
        return metalfx_ok ? Backend::Dlss : Backend::Native;
    case Settings::Upscaler::Fsr3:
        return fsr_ok ? Backend::Fsr : Backend::Native;
    case Settings::Upscaler::Fsr4:
        return fsr_ok ? Backend::Fsr : (metalfx_ok ? Backend::Dlss : Backend::Native);
    case Settings::Upscaler::Auto:
    case Settings::Upscaler::Dlss:
    default:
        return metalfx_ok ? Backend::Dlss : (fsr_ok ? Backend::Fsr : Backend::Native);
    }
}

bool OcarinaTemporal::IsTemporalPass(u64 native_key, u64 shader_hash, const Shader::Info& info) {
    return VideoCommon::FindOcarinaUpscalePass(native_key, shader_hash) !=
               VideoCommon::OcarinaUpscalePass::None &&
           SamplesIntoOneImage(info);
}

bool OcarinaTemporal::IsScaledPass(u64 shader_hash) {
    return std::ranges::find(TemporalShaders.begin(), TemporalShaders.begin() + 5, shader_hash) !=
           TemporalShaders.begin() + 5;
}

VideoCommon::OcarinaUpscaleBackend OcarinaTemporal::Backend() {
    if (failed) {
        return VideoCommon::OcarinaUpscaleBackend::Native;
    }
    if (!supported) {
        supported = TemporalUpscaler::IsSupported(device);
    }
    return ResolveBackend(device, metalfx_failed || !*supported, fsr3_failed);
}

bool OcarinaTemporal::Available() {
    return Backend() != VideoCommon::OcarinaUpscaleBackend::Native;
}

bool OcarinaTemporal::Prepare() {
    if (prepared || failed) {
        return !failed;
    }
    failed = true;
    std::string error;
    sampler = CreateLinearClampSampler(device);
    point_sampler = MakePointClampSampler(device);
    msl_library = CreateLibrary(device, OcarinaMslSource, error);
    const auto build_msl_pass = [&](Pass& pass, std::string_view entry,
                                    std::vector<ShaderTranslate::ResourceSlot> pass_slots) {
        pass.function = msl_library ? CreateFunction(msl_library.Get(), entry) : Object{};
        pass.state = pass.function
                         ? CreateComputePipelineState(device, pass.function.Get(), error, nullptr,
                                                      GroupSize * GroupSize)
                         : Object{};
        pass.slots = std::move(pass_slots);
        return static_cast<bool>(pass.state);
    };
    using ShaderTranslate::PushConstantSet;
    using ShaderTranslate::ResourceKind;
    if (!build_msl_pass(motion_pass, "ocarina_camera_motion",
                        {
                            {.set = PushConstantSet, .binding = 0, .kind = ResourceKind::Buffer, .index = 0},
                            {.set = 0, .binding = 0, .kind = ResourceKind::Texture, .index = 0},
                            {.set = 0, .binding = 1, .kind = ResourceKind::Texture, .index = 1},
                        }) ||
        !build_msl_pass(prepare_pass, "ocarina_prepare_inputs",
                        {
                            {.set = PushConstantSet, .binding = 0, .kind = ResourceKind::Buffer, .index = 0},
                            {.set = 0, .binding = 0, .kind = ResourceKind::Texture, .index = 0},
                            {.set = 0, .binding = 1, .kind = ResourceKind::Texture, .index = 1},
                            {.set = 0, .binding = 2, .kind = ResourceKind::Texture, .index = 2},
                            {.set = 0, .binding = 3, .kind = ResourceKind::Texture, .index = 3},
                        }) ||
        !build_msl_pass(direct_output_pass, "ocarina_direct_output",
                        {
                            {.set = 0, .binding = 0, .kind = ResourceKind::Texture, .index = 0},
                            {.set = 0, .binding = 1, .kind = ResourceKind::Texture, .index = 1},
                        }) ||
        !BuildPass(device, output_pass, OCARINA_DLSS_OUTPUT_COMP_SPV, error) || !sampler ||
        !point_sampler) {
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
    // The scene depth (binding 3) is a depth texture.
    options.depth_images = {{0, 3}};
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
    if (!point_sampler) {
        point_sampler = MakePointClampSampler(device);
    }
    if (!actor_state || !sampler || !point_sampler) {
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

std::array<std::string, 2> OcarinaTemporal::ActorMsl(u32 apple_family) {
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = apple_family;
    options.flip_vertex_y = false;
    const auto translate = [&](std::span<const u32> code, ShaderTranslate::Stage stage) {
        const ShaderTranslate::Result result = ShaderTranslate::SpirvToMsl(code, stage, options);
        std::string text = result.source.empty() ? "error: " + result.error : result.source;
        for (const ShaderTranslate::ResourceSlot& slot : result.slots) {
            text += "\n// slot set " + std::to_string(slot.set) + " binding " +
                    std::to_string(slot.binding) + " kind " +
                    std::to_string(static_cast<int>(slot.kind)) + " index " +
                    std::to_string(slot.index);
        }
        return text;
    };
    std::string vertex = translate(OCARINA_ACTOR_MOTION_VERT_SPV, ShaderTranslate::Stage::Vertex);
    options.depth_images = {{0, 3}};
    return {std::move(vertex),
            translate(OCARINA_ACTOR_MOTION_FRAG_SPV, ShaderTranslate::Stage::Fragment)};
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
    if (motion || color) {
        scheduler.Finish();
    }
    const auto make = [&](u32 format, bool render_target) {
        return CreateTexture(device, {.kind = TextureKind::e2D,
                                      .format = format,
                                      .width = extent[0],
                                      .height = extent[1],
                                      .render_target = render_target,
                                      .shader_write = true,
                                      .pixel_format_view = false});
    };
    // The actor overlay draws into the motion.
    motion = make(Fmt::RG16Float, true);
    color = make(Fmt::RGBA16Float, false);
    size = extent;
    return motion && color;
}

void OcarinaTemporal::Run(const Pass& pass, std::span<const TextureHandle> inputs,
                          std::span<const TextureHandle> outputs,
                          std::span<const u8> push_constants, std::array<u32, 2> extent,
                          bool linear_filter) {
    void* const bound_sampler =
        linear_filter || !point_sampler ? sampler.Get() : point_sampler.Get();
    // The host shaders: the sampled inputs at bindings 0.., the written images after them.
    scheduler.BeginCompute(pass.state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : pass.slots) {
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (slot.set == ShaderTranslate::PushConstantSet && !push_constants.empty()) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, push_constants.data(),
                                    static_cast<u32>(push_constants.size()));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture: {
            const bool is_write = slot.binding >= inputs.size();
            const size_t out_idx = is_write ? slot.binding - inputs.size() : 0;
            const TextureHandle tex =
                !is_write ? inputs[slot.binding]
                          : (out_idx < outputs.size() ? outputs[out_idx] : nullptr);
            scheduler.BindTexture(ShaderStage::Compute, slot.index, tex, false, is_write);
            break;
        }
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, bound_sampler);
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
        slot.current = CreateBuffer(device, bone_bytes, true, &slot.current_data, true);
        slot.previous = CreateBuffer(device, bone_bytes, true, &slot.previous_data, true);
    }
    const bool same_mesh = slot.used && slot.mesh_key == mesh.mesh_key &&
                           slot.vertex_bytes == mesh.vertices.size() &&
                           slot.index_count == mesh.indices.size();
    if (!same_mesh) {
        if (!slot.vertices || slot.vertex_bytes < mesh.vertices.size()) {
            slot.vertices =
                CreateBuffer(device, mesh.vertices.size(), true, &slot.vertex_data, true);
        }
        if (!slot.indices || slot.index_count < mesh.indices.size()) {
            slot.indices = CreateBuffer(device, mesh.indices.size() * sizeof(u16), true,
                                        &slot.index_data, true);
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
                                const ActorMotion& actor, std::array<f32, 2> jitter) {
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
        .jitter = jitter,
    };
    RenderTargetSet targets{};
    targets.color[0] = {.texture = motion.Get()};
    targets.width = extent[0];
    targets.height = extent[1];
    scheduler.BeginRendering(targets);
    scheduler.SetDrawState({
        .pipeline = actor_state.Get(),
        // Offset like the scene's jittered draws, so the replay covers the same samples.
        .viewport = {jitter[0], jitter[1], static_cast<f32>(extent[0]),
                     static_cast<f32>(extent[1]), 0.0f, 1.0f},
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
                                 0, false);
        } else if (resource.binding == 2) {
            scheduler.BindBytes(ShaderStage::Vertex, resource.index, &transforms,
                                static_cast<u32>(sizeof(transforms)));
        }
    }
    for (const ShaderTranslate::ResourceSlot& resource : actor_fragment_slots) {
        switch (resource.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (resource.set == ShaderTranslate::PushConstantSet) {
                const u32 tolerance = ActorDepthTolerance();
                scheduler.BindBytes(ShaderStage::Fragment, resource.index, &tolerance,
                                    static_cast<u32>(sizeof(tolerance)));
            } else if (resource.binding == 2) {
                scheduler.BindBytes(ShaderStage::Fragment, resource.index, &transforms,
                                    static_cast<u32>(sizeof(transforms)));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Fragment, resource.index, depth, false, false);
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Fragment, resource.index, point_sampler.Get());
            break;
        }
    }
    scheduler.DrawIndexed(*PrimitiveType(Maxwell::PrimitiveTopology::Triangles),
                          static_cast<u32>(mesh.indices.size()), 2, slot->indices.Get(), 0, 1, 0,
                          0);
    ++actor_frames;
    return true;
}

bool OcarinaTemporal::EncodeMotion(TextureHandle depth, std::array<u32, 2> extent,
                                   const VideoCommon::OcarinaMatrix& clip_to_previous_clip,
                                   const ActorMotion* actor, std::array<f32, 2> jitter) {
    if (depth == nullptr || extent[0] == 0 || extent[1] == 0 || !Prepare() || !Resize(extent)) {
        return false;
    }
    // The scene's jitter (none unless EDEN_NX2_DLSS_JITTER is set, as on Vulkan's MetalFX path).
    const MotionParameters parameters{
        .clip_to_previous_clip = clip_to_previous_clip,
        .jitter = jitter,
        .extent = extent,
    };
    Run(motion_pass, depth, motion.Get(),
        std::span(reinterpret_cast<const u8*>(&parameters), sizeof(parameters)), extent);
    // Without Link's pose (dialogue, cutscenes, first person) the motion is the camera's alone.
    if (actor != nullptr && ActorEnabled()) {
        DrawActor(depth, extent, *actor, jitter);
    }
    return true;
}

bool OcarinaTemporal::EnsureEffectsBias(std::array<u32, 2> bias_extent) {
    if (bias_extent[0] == 0 || bias_extent[1] == 0) {
        return false;
    }
    if (effects.bias_size == bias_extent && effects.bias) {
        return true;
    }
    if (effects.bias) {
        scheduler.Finish();
    }
    effects.bias = CreateTexture(device, {.kind = TextureKind::e2D,
                                          .format = Fmt::R16Float,
                                          .width = bias_extent[0],
                                          .height = bias_extent[1],
                                          .render_target = true,
                                          .shader_write = true,
                                          .pixel_format_view = false});
    effects.bias_size = effects.bias ? bias_extent : std::array<u32, 2>{};
    effects.bias_zeroed = false;
    return static_cast<bool>(effects.bias);
}

void OcarinaTemporal::ClearEffectsBias() {
    if (!effects.bias || effects.bias_size[0] == 0 || effects.bias_size[1] == 0) {
        return;
    }
    ClearPass clear{};
    clear.color[0] = {.texture = effects.bias.Get()};
    clear.clear_color[0] = true;
    clear.color_value[0] = {0.0, 0.0, 0.0, 0.0};
    clear.width = effects.bias_size[0];
    clear.height = effects.bias_size[1];
    scheduler.Clear(clear);
    effects.bias_zeroed = true;
}

TextureHandle OcarinaTemporal::EnsureUpscaled(std::array<u32, 2> output_size) {
    if (upscaled_rejected || output_size[0] == 0 || output_size[1] == 0) {
        return nullptr;
    }
    if (upscaled_size == output_size && upscaled) {
        return upscaled.Get();
    }
    if (upscaled) {
        scheduler.Finish();
    }
    upscaled = CreateTexture(device, {.kind = TextureKind::e2D,
                                      .format = Fmt::RG11B10Float,
                                      .width = output_size[0],
                                      .height = output_size[1],
                                      .render_target = true,
                                      .shader_write = true,
                                      .pixel_format_view = false});
    upscaled_size = upscaled ? output_size : std::array<u32, 2>{};
    if (!upscaled) {
        upscaled_rejected = true;
        return nullptr;
    }
    return upscaled.Get();
}

void OcarinaTemporal::SaveEffectsBase(TextureHandle scene, std::array<u32, 2> extent, u64 frame) {
    if (scene == nullptr || extent[0] == 0 || extent[1] == 0 || !PrepareEffects()) {
        return;
    }
    const u32 scene_format = TextureFormat(scene);
    if (scene_format == 0) {
        return;
    }
    const bool extent_changed = effects.extent != extent;
    if (!effects.base || extent_changed || effects.base_format != scene_format) {
        if (effects.base || (extent_changed && (effects.marks[0] || effects.marks[1]))) {
            scheduler.Finish();
        }
        const auto make = [&](u32 format, bool shader_write) {
            return CreateTexture(device, {.kind = TextureKind::e2D,
                                          .format = format,
                                          .width = extent[0],
                                          .height = extent[1],
                                          .shader_write = shader_write,
                                          .pixel_format_view = false});
        };
        effects.base = make(scene_format, false);
        if (extent_changed || !effects.marks[0] || !effects.marks[1]) {
            effects.marks = {make(Fmt::R8Unorm, true), make(Fmt::R8Unorm, true)};
            effects.marked_frame.reset();
            effects.previous_marked_frame.reset();
        }
        if (!effects.base || !effects.marks[0] || !effects.marks[1]) {
            effects.base = {};
            effects.marks = {};
            effects.extent = {};
            effects.base_format = 0;
            effects.base_frame.reset();
            return;
        }
        effects.extent = extent;
        effects.base_format = scene_format;
    }
    scheduler.CopyTexture(scene, 0, 0, {}, {extent[0], extent[1], 1}, effects.base.Get(), 0, 0, {});
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
        size[0] < extent[0] || size[1] < extent[1] || !EnsureEffectsBias(size)) {
        return false;
    }
    const bool has_previous = effects.previous_marked_frame == frame - 1;
    const u64 previous_mark = has_previous ? frame - 1 : frame;
    const std::array<TextureHandle, 3> inputs{effects.marks[frame % 2].Get(),
                                              effects.marks[previous_mark % 2].Get(),
                                              motion.Get()};
    const u32 control = has_previous ? 1U : 0U;
    Run(effects_bias_pass, inputs, effects.bias.Get(),
        std::span(reinterpret_cast<const u8*>(&control), sizeof(control)), size);
    effects.bias_zeroed = false;
    return true;
}

bool OcarinaTemporal::Record(const Frame& frame, const char*& reason) {
    using UpscaleBackend = VideoCommon::OcarinaUpscaleBackend;
    UpscaleBackend backend = Backend();
    if (backend == UpscaleBackend::Native) {
        reason = "Ocarina temporal upscaler unavailable";
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
    const f32 projection_y = std::hypot(frame.current[4], frame.current[5], frame.current[6]);
    if (backend == UpscaleBackend::Fsr && (!std::isfinite(projection_y) || projection_y < 0.01f)) {
        reason = "camera projection invalid";
        return false;
    }
    const std::array<u32, 2> output_size =
        frame.output_size[0] != 0 && frame.output_size[1] != 0 ? frame.output_size
                                                               : frame.target_size;
    std::optional<ActorMotion> actor;
    if (frame.actor != nullptr) {
        actor.emplace(ActorMotion{*frame.actor, frame.current, frame.previous});
    }
    const bool force_crop = force_crop_depth && crop_extent == frame.extent &&
                            crop_depth_size == frame.depth_size;
    const std::array<u32, 2> backing_size =
        !force_crop && backend == UpscaleBackend::Dlss &&
                frame.depth_size[0] >= frame.extent[0] && frame.depth_size[1] >= frame.extent[1]
            ? frame.depth_size
            : frame.extent;
    if (frame.depth == nullptr || frame.color == nullptr || frame.extent[0] == 0 ||
        frame.extent[1] == 0 || !Prepare() || !Resize(backing_size)) {
        reason = "motion pass not made";
        return false;
    }
    const MotionParameters motion_params{
        .clip_to_previous_clip = *transform,
        .jitter = frame.jitter,
        .extent = frame.extent,
    };
    const std::array<TextureHandle, 2> prep_inputs{frame.depth, frame.color};
    const std::array<TextureHandle, 2> prep_outputs{motion.Get(), color.Get()};
    Run(prepare_pass, prep_inputs, prep_outputs,
        std::span(reinterpret_cast<const u8*>(&motion_params), sizeof(motion_params)), size);
    if (actor && ActorEnabled()) {
        DrawActor(frame.depth, frame.extent, *actor, frame.jitter);
    }
    bool bias_encoded = EffectsEnabled() && EncodeEffectsBias(frame.frame, frame.extent);
    const auto now = std::chrono::steady_clock::now();
    const f32 elapsed =
        last_evaluation
            ? std::chrono::duration<f32, std::milli>(now - *last_evaluation).count()
            : 16.67f;
    const auto approx_equal = [](f32 a, f32 b) {
        return std::abs(a - b) <= (std::max)(0.001f, 1e-4f * (std::max)(std::abs(a), std::abs(b)));
    };
    const bool camera_cut =
        !std::ranges::equal(frame.previous, last_view_projection, approx_equal) ||
        IsCameraCut(*transform);
    const bool reset = !last_frame || frame.frame != *last_frame + 1 ||
                       last_render_size != frame.extent || last_output_size != output_size ||
                       last_backend != backend || elapsed > 250.0f || camera_cut;

    TextureHandle output = nullptr;
    if (backend == UpscaleBackend::Dlss) {
        const auto run_crop_depth = [&]() -> TextureHandle {
            if (cropped_depth_size != frame.extent || !cropped_depth) {
                if (cropped_depth) {
                    scheduler.Finish();
                }
                cropped_depth = CreateTexture(device, {.kind = TextureKind::e2D,
                                                       .format = Fmt::Depth32Float,
                                                       .width = frame.extent[0],
                                                       .height = frame.extent[1],
                                                       .render_target = true,
                                                       .pixel_format_view = false});
                cropped_depth_size = cropped_depth ? frame.extent : std::array<u32, 2>{};
            }
            if (!cropped_depth) {
                return nullptr;
            }
            scheduler.CopyTexture(frame.depth, 0, 0, {}, {frame.extent[0], frame.extent[1], 1},
                                  cropped_depth.Get(), 0, 0, {});
            return cropped_depth.Get();
        };
        TextureHandle depth_input = force_crop ? run_crop_depth() : frame.depth;
        if (depth_input == nullptr) {
            depth_input = frame.depth;
        }
        TextureHandle reactive_mask = nullptr;
        if (EffectsEnabled() && TemporalUpscaler::ReactiveMaskSupported(device) &&
            EnsureEffectsBias(size)) {
            if (!bias_encoded && !effects.bias_zeroed) {
                ClearEffectsBias();
            }
            reactive_mask = effects.bias.Get();
        }
        const TextureHandle direct_upscaled = EnsureUpscaled(output_size);
        // EDEN_NX2_DLSS_JITTER=2: y flipped.
        const std::array<f32, 2> dlss_jitter{
            frame.jitter[0],
            VideoCommon::OcarinaDlssJitterMode() == 2 ? -frame.jitter[1] : frame.jitter[1]};
        output = upscaler.Upscale(
            device, scheduler,
            {
                .color = color.Get(),
                .depth = depth_input,
                .motion = motion.Get(),
                .render_size = frame.extent,
                .output_size = output_size,
                .jitter = dlss_jitter,
                .motion_vector_scale = {1.0f, 1.0f},
                .reset = reset,
                .depth_reversed = false,
                .auto_exposure = false,
                .reactive = reactive_mask,
                .output = direct_upscaled,
            });
        if (output == nullptr && !upscaler.Failed() && !force_crop) {
            if (size != frame.extent && Resize(frame.extent)) {
                const std::array<TextureHandle, 2> fallback_outputs{motion.Get(), color.Get()};
                Run(prepare_pass, prep_inputs, fallback_outputs,
                    std::span(reinterpret_cast<const u8*>(&motion_params), sizeof(motion_params)),
                    size);
                if (actor && ActorEnabled()) {
                    DrawActor(frame.depth, frame.extent, *actor, frame.jitter);
                }
                bias_encoded = EffectsEnabled() && EncodeEffectsBias(frame.frame, frame.extent);
                reactive_mask = nullptr;
                if (EffectsEnabled() && TemporalUpscaler::ReactiveMaskSupported(device) &&
                    EnsureEffectsBias(size)) {
                    if (!bias_encoded && !effects.bias_zeroed) {
                        ClearEffectsBias();
                    }
                    reactive_mask = effects.bias.Get();
                }
            }
            if (size == frame.extent) {
                if (const TextureHandle cropped = run_crop_depth()) {
                    output = upscaler.Upscale(
                        device, scheduler,
                        {
                            .color = color.Get(),
                            .depth = cropped,
                            .motion = motion.Get(),
                            .render_size = frame.extent,
                            .output_size = output_size,
                            .jitter = dlss_jitter,
                            .motion_vector_scale = {1.0f, 1.0f},
                            .reset = reset,
                            .depth_reversed = false,
                            .auto_exposure = false,
                            .reactive = reactive_mask,
                            .output = direct_upscaled,
                        });
                    if (output != nullptr) {
                        force_crop_depth = true;
                        crop_extent = frame.extent;
                        crop_depth_size = frame.depth_size;
                    }
                }
            }
        }
        if (output != nullptr) {
            if (direct_upscaled != nullptr && output != direct_upscaled) {
                upscaled_rejected = true;
                upscaled = {};
                upscaled_size = {};
            }
            if (bias_encoded && reactive_mask != nullptr) {
                ++effects_frames;
            }
        } else {
            if (upscaler.Failed()) {
                metalfx_failed = true;
            }
            backend = ResolveBackend(device, true, fsr3_failed);
        }
    }
    if (output == nullptr && backend == UpscaleBackend::Fsr) {
        if (!std::isfinite(projection_y) || projection_y < 0.01f) {
            reason = "camera projection invalid";
            return false;
        }
        if (!fsr3) {
            fsr3 = std::make_unique<Fsr3Upscaler>(device, scheduler,
                                                  Fsr3Upscaler::Profile::OcarinaLdr);
        }
        const f32 sharpness = FSR::RcasSharpnessFromStops(
            static_cast<f32>(Settings::values.fsr_sharpening_slider.GetValue()) / 100.0f);
        const bool fsr_reset = reset || last_backend != UpscaleBackend::Fsr;
        const TextureHandle mask = bias_encoded ? effects.bias.Get() : nullptr;
        output = fsr3->Upscale({
            .color = color.Get(),
            .depth = frame.depth,
            .motion = motion.Get(),
            .reactive = mask,
            .transparency = mask,
            .render_size = frame.extent,
            .upscale_size = output_size,
            .jitter = frame.jitter,
            .motion_vector_scale = {1.0f, 1.0f},
            .frame_time_ms = elapsed,
            .reset = fsr_reset,
            .sharpen = sharpness > 0.0f,
            .sharpness = sharpness,
            .vertical_fov = 2.0f * std::atan(1.0f / projection_y),
        });
        if (output != nullptr) {
            if (bias_encoded) {
                ++effects_frames;
            }
        } else {
            fsr3_failed = true;
        }
    }
    if (output == nullptr) {
        if (Backend() == UpscaleBackend::Native) {
            failed = true;
        }
        reason = fsr3_failed ? "FSR 3.1 declined the frame" : "MetalFX declined the frame";
        return false;
    }

    last_upscaled = output;
    if (frame.target != nullptr) {
        const bool same_size = output_size == frame.target_size;
        Run(same_size ? direct_output_pass : output_pass, output, frame.target, {},
            frame.target_size, !same_size);
    }
    last_evaluation = now;
    last_frame = frame.frame;
    last_render_size = frame.extent;
    last_output_size = output_size;
    last_backend = backend;
    last_view_projection = frame.current;
    ++passes;
    return true;
}

void OcarinaTemporal::SaveSceneReference(TextureHandle guest_scene) {
    if (guest_scene == nullptr || composite_failed || !Prepare()) {
        return;
    }
    constexpr std::array<u32, 2> ReferenceExtent{2560, 1440};
    if (!reference_pass.state) {
        std::string error;
        if (!BuildPass(device, reference_pass, OCARINA_DLSS_4K_REFERENCE_COMP_SPV, error)) {
            LOG_WARNING(Render_Metal, "Metal: Ocarina 4K reference pass unavailable: {}", error);
            composite_failed = true;
            return;
        }
    }
    if (!native_scene) {
        native_scene = CreateTexture(device, {.kind = TextureKind::e2D,
                                              .format = Fmt::RGBA8Unorm,
                                              .width = ReferenceExtent[0],
                                              .height = ReferenceExtent[1],
                                              .shader_write = true,
                                              .pixel_format_view = false});
        if (!native_scene) {
            composite_failed = true;
            return;
        }
    }
    Run(reference_pass, guest_scene, native_scene.Get(), {}, ReferenceExtent);
}

TextureHandle OcarinaTemporal::Composite4k(TextureHandle native_composite, u64 frame) {
    if (composited_frame == frame && composite) {
        return composite.Get();
    }
    if (native_composite == nullptr || last_upscaled == nullptr || !native_scene ||
        composite_failed || !Prepare()) {
        return nullptr;
    }
    constexpr std::array<u32, 2> CompositeExtent{3840, 2160};
    if (!composite_pass.state) {
        std::string error;
        if (!BuildPass(device, composite_pass, OCARINA_DLSS_4K_COMPOSITE_COMP_SPV, error)) {
            LOG_WARNING(Render_Metal, "Metal: Ocarina 4K composite pass unavailable: {}", error);
            composite_failed = true;
            return nullptr;
        }
    }
    if (!composite) {
        composite = CreateTexture(device, {.kind = TextureKind::e2D,
                                           .format = Fmt::RGBA8Unorm,
                                           .width = CompositeExtent[0],
                                           .height = CompositeExtent[1],
                                           .shader_write = true,
                                           .pixel_format_view = false});
        if (!composite) {
            composite_failed = true;
            return nullptr;
        }
    }
    const std::array<TextureHandle, 3> inputs{last_upscaled, native_scene.Get(), native_composite};
    const u32 use_dlss = !failed ? 1U : 0U;
    Run(composite_pass, inputs, composite.Get(),
        std::span(reinterpret_cast<const u8*>(&use_dlss), sizeof(use_dlss)), CompositeExtent,
        true);
    composited_frame = frame;
    return composite.Get();
}

u64 OcarinaTemporal::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

std::array<u64, 2> OcarinaTemporal::TakeExtras() noexcept {
    return {std::exchange(actor_frames, 0), std::exchange(effects_frames, 0)};
}

} // namespace Metal
