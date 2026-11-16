// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/fsr.h"
#include "video_core/host_shaders/full_screen_triangle_vert_spv.h"
#include "video_core/host_shaders/fxaa_frag_spv.h"
#include "video_core/host_shaders/fxaa_vert_spv.h"
#include "video_core/host_shaders/present_area_frag_spv.h"
#include "video_core/host_shaders/present_bicubic_frag_spv.h"
#include "video_core/host_shaders/present_bspline_frag_spv.h"
#include "video_core/host_shaders/present_gaussian_frag_spv.h"
#include "video_core/host_shaders/present_lanczos_frag_spv.h"
#include "video_core/host_shaders/present_mitchell_frag_spv.h"
#include "video_core/host_shaders/present_mmpx_frag_spv.h"
#include "video_core/host_shaders/present_spline1_frag_spv.h"
#include "video_core/host_shaders/present_zero_tangent_frag_spv.h"
#include "video_core/host_shaders/sgsr1_shader_mobile_edge_direction_frag_spv.h"
#include "video_core/host_shaders/sgsr1_shader_mobile_frag_spv.h"
#include "video_core/host_shaders/sgsr1_shader_vert_spv.h"
#include "video_core/host_shaders/smaa_blending_weight_calculation_frag_spv.h"
#include "video_core/host_shaders/smaa_blending_weight_calculation_vert_spv.h"
#include "video_core/host_shaders/smaa_edge_detection_frag_spv.h"
#include "video_core/host_shaders/smaa_edge_detection_vert_spv.h"
#include "video_core/host_shaders/smaa_neighborhood_blending_frag_spv.h"
#include "video_core/host_shaders/smaa_neighborhood_blending_vert_spv.h"
#include "video_core/host_shaders/temporal_sharpen_frag_spv.h"
#include "video_core/host_shaders/temporal_upscale_frag_spv.h"
#include "video_core/host_shaders/vulkan_fidelityfx_fsr_easu_fp16_frag_spv.h"
#include "video_core/host_shaders/vulkan_fidelityfx_fsr_rcas_fp16_frag_spv.h"
#include "video_core/host_shaders/vulkan_fidelityfx_fsr_vert_spv.h"
#include "video_core/host_shaders/vulkan_present_scaleforce_fp16_frag_spv.h"
#include "video_core/host_shaders/vulkan_present_vert_spv.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_present.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/smaa_area_tex.h"
#include "video_core/smaa_search_tex.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

using Shader = PresentPasses::Shader;

/// Apple GPUs run half precision natively: the fp16 variants, as Vulkan picks them where the
/// device has shaderFloat16 (MoltenVK reports it).
std::span<const u32> Code(Shader shader) {
    switch (shader) {
    case Shader::PresentVertex:
        return VULKAN_PRESENT_VERT_SPV;
    case Shader::Bicubic:
        return PRESENT_BICUBIC_FRAG_SPV;
    case Shader::ZeroTangent:
        return PRESENT_ZERO_TANGENT_FRAG_SPV;
    case Shader::BSpline:
        return PRESENT_BSPLINE_FRAG_SPV;
    case Shader::Mitchell:
        return PRESENT_MITCHELL_FRAG_SPV;
    case Shader::Spline1:
        return PRESENT_SPLINE1_FRAG_SPV;
    case Shader::Gaussian:
        return PRESENT_GAUSSIAN_FRAG_SPV;
    case Shader::Lanczos:
        return PRESENT_LANCZOS_FRAG_SPV;
    case Shader::ScaleForce:
        return VULKAN_PRESENT_SCALEFORCE_FP16_FRAG_SPV;
    case Shader::Area:
        return PRESENT_AREA_FRAG_SPV;
    case Shader::Mmpx:
        return PRESENT_MMPX_FRAG_SPV;
    case Shader::FsrVertex:
        return VULKAN_FIDELITYFX_FSR_VERT_SPV;
    case Shader::Easu:
        return VULKAN_FIDELITYFX_FSR_EASU_FP16_FRAG_SPV;
    case Shader::Rcas:
        return VULKAN_FIDELITYFX_FSR_RCAS_FP16_FRAG_SPV;
    case Shader::SgsrVertex:
        return SGSR1_SHADER_VERT_SPV;
    case Shader::Sgsr:
        return SGSR1_SHADER_MOBILE_FRAG_SPV;
    case Shader::SgsrEdge:
        return SGSR1_SHADER_MOBILE_EDGE_DIRECTION_FRAG_SPV;
    case Shader::FxaaVertex:
        return FXAA_VERT_SPV;
    case Shader::Fxaa:
        return FXAA_FRAG_SPV;
    case Shader::SmaaEdgeVertex:
        return SMAA_EDGE_DETECTION_VERT_SPV;
    case Shader::SmaaEdge:
        return SMAA_EDGE_DETECTION_FRAG_SPV;
    case Shader::SmaaWeightVertex:
        return SMAA_BLENDING_WEIGHT_CALCULATION_VERT_SPV;
    case Shader::SmaaWeight:
        return SMAA_BLENDING_WEIGHT_CALCULATION_FRAG_SPV;
    case Shader::SmaaBlendVertex:
        return SMAA_NEIGHBORHOOD_BLENDING_VERT_SPV;
    case Shader::SmaaBlend:
        return SMAA_NEIGHBORHOOD_BLENDING_FRAG_SPV;
    case Shader::FullScreenVertex:
        return FULL_SCREEN_TRIANGLE_VERT_SPV;
    case Shader::TemporalUpscale:
        return TEMPORAL_UPSCALE_FRAG_SPV;
    case Shader::TemporalSharpen:
        return TEMPORAL_SHARPEN_FRAG_SPV;
    case Shader::Count:
        break;
    }
    return {};
}

bool IsVertex(Shader shader) {
    switch (shader) {
    case Shader::PresentVertex:
    case Shader::FsrVertex:
    case Shader::SgsrVertex:
    case Shader::FxaaVertex:
    case Shader::SmaaEdgeVertex:
    case Shader::SmaaWeightVertex:
    case Shader::SmaaBlendVertex:
    case Shader::FullScreenVertex:
        return true;
    default:
        return false;
    }
}

/// The window filter's fragment shader (Vulkan's BlitScreen::SetWindowAdaptPass on a device
/// without VK_EXT_filter_cubic, as MoltenVK is); nothing for the blit's nearest and bilinear.
std::optional<Shader> WindowShader(Settings::ScalingFilter filter) {
    using Filter = Settings::ScalingFilter;
    switch (filter) {
    case Filter::Bicubic:
        return Shader::Bicubic;
    case Filter::ZeroTangent:
        return Shader::ZeroTangent;
    case Filter::BSpline:
        return Shader::BSpline;
    case Filter::Mitchell:
        return Shader::Mitchell;
    case Filter::Spline1:
        return Shader::Spline1;
    case Filter::Gaussian:
        return Shader::Gaussian;
    case Filter::Lanczos:
        return Shader::Lanczos;
    case Filter::ScaleForce:
        return Shader::ScaleForce;
    case Filter::Area:
        return Shader::Area;
    case Filter::Mmpx:
        return Shader::Mmpx;
    default:
        return std::nullopt;
    }
}

/// vulkan_present.vert's push constants (PresentPushConstants).
struct WindowConstants {
    std::array<f32, 16> modelview;
    std::array<std::array<f32, 4>, 4> vertices;
};
static_assert(sizeof(WindowConstants) == 128);

template <typename T>
std::span<const u8> Bytes(const T& value) {
    return {reinterpret_cast<const u8*>(&value), sizeof(value)};
}

BlendTargetState BlendState(BlitBlend blend) {
    using Factor = Maxwell::Blend::Factor;
    using Equation = Maxwell::Blend::Equation;
    BlendTargetState state{};
    state.write_mask = 0xF;
    state.color_op = Equation::Add_D3D;
    state.alpha_op = Equation::Add_D3D;
    switch (blend) {
    case BlitBlend::None:
        state.color_source = Factor::One_D3D;
        state.alpha_source = Factor::One_D3D;
        state.color_dest = Factor::Zero_D3D;
        state.alpha_dest = Factor::Zero_D3D;
        break;
    case BlitBlend::Premultiplied:
        state.enable = 1;
        state.color_source = Factor::One_D3D;
        state.alpha_source = Factor::One_D3D;
        state.color_dest = Factor::OneMinusSourceAlpha_D3D;
        state.alpha_dest = Factor::OneMinusSourceAlpha_D3D;
        break;
    case BlitBlend::Coverage:
        state.enable = 1;
        state.color_source = Factor::SourceAlpha_D3D;
        state.alpha_source = Factor::One_D3D;
        state.color_dest = Factor::OneMinusSourceAlpha_D3D;
        state.alpha_dest = Factor::OneMinusSourceAlpha_D3D;
        break;
    }
    return state;
}

Object MakeSampler(const Device& device, bool linear) {
    if (linear) {
        return CreateLinearClampSampler(device);
    }
    Tegra::Texture::TSCEntry tsc{};
    tsc.wrap_u.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.wrap_v.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.wrap_p.Assign(Tegra::Texture::WrapMode::ClampToEdge);
    tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Nearest);
    tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::None);
    return CreateSampler(device, tsc);
}

} // Anonymous namespace

PresentPasses::PresentPasses(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {}

PresentPasses::~PresentPasses() = default;

bool PresentPasses::Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_PRESENT_FILTERS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

bool PresentPasses::HasWindowShader(Settings::ScalingFilter filter) {
    return WindowShader(filter).has_value();
}

const PresentPasses::Program* PresentPasses::Get(Shader shader) {
    Program& program = programs[static_cast<size_t>(shader)];
    if (program.function) {
        return &program;
    }
    if (program.failed) {
        return nullptr;
    }
    program.failed = true;
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = DeviceAppleFamily(device);
    const ShaderTranslate::Result msl = ShaderTranslate::SpirvToMsl(
        Code(shader),
        IsVertex(shader) ? ShaderTranslate::Stage::Vertex : ShaderTranslate::Stage::Fragment,
        options);
    std::string error = msl.error;
    if (!msl.source.empty()) {
        program.library = CreateLibrary(device, msl.source, error);
        program.function =
            program.library ? CreateFunction(program.library.Get(), msl.entry_point) : Object{};
    }
    if (!program.function) {
        LOG_WARNING(Render_Metal, "Metal: present shader {} unavailable: {}",
                    static_cast<int>(shader), error);
        return nullptr;
    }
    program.slots = msl.slots;
    program.failed = false;
    return &program;
}

bool PresentPasses::Prepare(Shader shader) {
    return Get(shader) != nullptr;
}

TextureHandle PresentPasses::Make(Target& target, std::array<u32, 2> size, u32 format) {
    if (!target.texture || target.size != size || target.format != format) {
        target.texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                                .format = format,
                                                .width = size[0],
                                                .height = size[1],
                                                .render_target = true});
        target.size = size;
        target.format = format;
    }
    return target.texture.Get();
}

bool PresentPasses::Draw(const Pass& pass) {
    const Program* const vertex = Get(pass.vertex);
    const Program* const fragment = Get(pass.fragment);
    if (vertex == nullptr || fragment == nullptr || pass.target.texture == nullptr ||
        pass.size[0] == 0 || pass.size[1] == 0) {
        return false;
    }
    Object& state = pipelines[{pass.vertex, pass.fragment, pass.format, pass.blend}];
    if (!state) {
        auto desc = Zeroed<RenderPipelineDesc>();
        desc.vertex_function = vertex->function.Get();
        desc.fragment_function = fragment->function.Get();
        desc.color_formats[0] = pass.format;
        desc.blend[0] = BlendState(pass.blend);
        desc.samples = 1;
        desc.topology_class = 2;
        desc.rasterize = 1;
        std::string error;
        state = CreateRenderPipelineState(device, desc, error);
        if (!state) {
            LOG_WARNING(Render_Metal, "Metal: present pass {}/{} not built: {}",
                        static_cast<int>(pass.vertex), static_cast<int>(pass.fragment), error);
            return false;
        }
    }
    Object& sampler = pass.nearest ? nearest : linear;
    if (!sampler) {
        sampler = MakeSampler(device, !pass.nearest);
        if (!sampler) {
            return false;
        }
    }
    if (!pass.keep_target) {
        ClearPass clear{};
        clear.color[0] = pass.target;
        clear.clear_color[0] = true;
        clear.color_value[0] = {0.0, 0.0, 0.0, 1.0};
        clear.width = pass.size[0];
        clear.height = pass.size[1];
        scheduler.Clear(clear);
    }
    RenderTargetSet targets{};
    targets.color[0] = pass.target;
    targets.width = pass.size[0];
    targets.height = pass.size[1];
    scheduler.BeginRendering(targets);
    scheduler.SetDrawState({
        .pipeline = state.Get(),
        .viewport = {0.0f, 0.0f, static_cast<f32>(pass.size[0]), static_cast<f32>(pass.size[1]),
                     0.0f, 1.0f},
        .scissor = {0, 0, pass.size[0], pass.size[1]},
        .cull_mode = 0, // MTLCullModeNone
    });
    const auto bind = [&](const Program& program, ShaderStage stage) {
        for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
            switch (slot.kind) {
            case ShaderTranslate::ResourceKind::Buffer:
                if (slot.set == ShaderTranslate::PushConstantSet && !pass.push_constants.empty()) {
                    scheduler.BindBytes(stage, slot.index, pass.push_constants.data(),
                                        static_cast<u32>(pass.push_constants.size()));
                }
                break;
            case ShaderTranslate::ResourceKind::Texture:
                if (slot.binding < pass.inputs.size()) {
                    scheduler.BindTexture(stage, slot.index, pass.inputs[slot.binding]);
                }
                break;
            case ShaderTranslate::ResourceKind::Sampler:
                scheduler.BindSampler(stage, slot.index, sampler.Get());
                break;
            }
        }
    };
    bind(*vertex, ShaderStage::Vertex);
    bind(*fragment, ShaderStage::Fragment);
    const Maxwell::PrimitiveTopology topology = pass.strip
                                                    ? Maxwell::PrimitiveTopology::TriangleStrip
                                                    : Maxwell::PrimitiveTopology::Triangles;
    scheduler.Draw(*PrimitiveType(topology), 0, pass.vertices, 1, 0);
    scheduler.EndRendering();
    ++passes;
    return true;
}

bool PresentPasses::PrepareSmaaTextures() {
    if (smaa_textures) {
        return *smaa_textures;
    }
    smaa_textures = false;
    smaa_area = CreateTexture(device, {.kind = TextureKind::e2D,
                                       .format = Fmt::RG8Unorm,
                                       .width = AREATEX_WIDTH,
                                       .height = AREATEX_HEIGHT});
    smaa_search = CreateTexture(device, {.kind = TextureKind::e2D,
                                         .format = Fmt::R8Unorm,
                                         .width = SEARCHTEX_WIDTH,
                                         .height = SEARCHTEX_HEIGHT});
    u8* area_bytes = nullptr;
    u8* search_bytes = nullptr;
    const Object area_upload = CreateBuffer(device, AREATEX_SIZE, true, &area_bytes);
    const Object search_upload = CreateBuffer(device, SEARCHTEX_SIZE, true, &search_bytes);
    if (!smaa_area || !smaa_search || area_bytes == nullptr || search_bytes == nullptr) {
        return false;
    }
    std::memcpy(area_bytes, areaTexBytes, AREATEX_SIZE);
    std::memcpy(search_bytes, searchTexBytes, SEARCHTEX_SIZE);
    scheduler.CopyBufferToTexture(area_upload.Get(), 0, AREATEX_PITCH, 0, smaa_area.Get(), 0, 0,
                                  {}, {AREATEX_WIDTH, AREATEX_HEIGHT, 1});
    scheduler.CopyBufferToTexture(search_upload.Get(), 0, SEARCHTEX_PITCH, 0, smaa_search.Get(), 0,
                                  0, {}, {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1});
    // The uploads must be done before the buffers go.
    scheduler.Finish();
    smaa_textures = true;
    return true;
}

PresentPasses::Source PresentPasses::AntiAlias(const Source& source,
                                               Settings::AntiAliasing anti_aliasing) {
    if (source.texture == nullptr || source.size[0] == 0 || source.size[1] == 0) {
        return source;
    }
    Source result = source;
    switch (anti_aliasing) {
    case Settings::AntiAliasing::Fxaa: {
        const TextureHandle output = Make(anti_aliased, source.size, Fmt::RGBA16Float);
        // fxaa.frag samples the input at bindings 0 and 1.
        const std::array<TextureHandle, 2> inputs{source.texture, source.texture};
        if (output != nullptr && Draw({.vertex = Shader::FxaaVertex,
                                       .fragment = Shader::Fxaa,
                                       .inputs = inputs,
                                       .target = {.texture = output},
                                       .format = Fmt::RGBA16Float,
                                       .size = source.size})) {
            result.texture = output;
        }
        return result;
    }
    case Settings::AntiAliasing::Smaa: {
        if (!PrepareSmaaTextures()) {
            return source;
        }
        const TextureHandle edges = Make(smaa_edges, source.size, Fmt::RG16Float);
        const TextureHandle weights = Make(smaa_weights, source.size, Fmt::RGBA16Float);
        const TextureHandle output = Make(anti_aliased, source.size, Fmt::RGBA16Float);
        if (edges == nullptr || weights == nullptr || output == nullptr) {
            return source;
        }
        // Each pass clears its target first, as Vulkan's render passes do.
        const std::array<TextureHandle, 1> edge_inputs{source.texture};
        const std::array<TextureHandle, 3> weight_inputs{edges, smaa_area.Get(),
                                                         smaa_search.Get()};
        const std::array<TextureHandle, 2> blend_inputs{source.texture, weights};
        const bool drawn = Draw({.vertex = Shader::SmaaEdgeVertex,
                                 .fragment = Shader::SmaaEdge,
                                 .inputs = edge_inputs,
                                 .target = {.texture = edges},
                                 .format = Fmt::RG16Float,
                                 .size = source.size}) &&
                           Draw({.vertex = Shader::SmaaWeightVertex,
                                 .fragment = Shader::SmaaWeight,
                                 .inputs = weight_inputs,
                                 .target = {.texture = weights},
                                 .format = Fmt::RGBA16Float,
                                 .size = source.size}) &&
                           Draw({.vertex = Shader::SmaaBlendVertex,
                                 .fragment = Shader::SmaaBlend,
                                 .inputs = blend_inputs,
                                 .target = {.texture = output},
                                 .format = Fmt::RGBA16Float,
                                 .size = source.size});
        if (drawn) {
            result.texture = output;
        }
        return result;
    }
    default:
        return source;
    }
}

PresentPasses::Source PresentPasses::Upscale(const Source& source,
                                             Settings::ScalingFilter filter,
                                             std::array<u32, 2> output) {
    using Filter = Settings::ScalingFilter;
    if (source.texture == nullptr || output[0] == 0 || output[1] == 0 || source.render[0] == 0 ||
        source.render[1] == 0) {
        return source;
    }
    const f32 input_width = static_cast<f32>(source.render[0]);
    const f32 input_height = static_cast<f32>(source.render[1]);
    const auto& crop = source.crop;
    if (filter == Filter::Fsr) {
        // Vulkan's FSR::Draw: EASU at the output size, then RCAS at the sharpness setting.
        const TextureHandle easu_output = Make(easu, output, Fmt::RGBA16Float);
        const TextureHandle rcas_output = Make(rcas, output, Fmt::RGBA16Float);
        if (easu_output == nullptr || rcas_output == nullptr) {
            return source;
        }
        std::array<u32, 16> easu_constants{};
        std::array<u32, 16> rcas_constants{};
        FSR::FsrEasuConOffset(easu_constants.data(), easu_constants.data() + 4,
                              easu_constants.data() + 8, easu_constants.data() + 12,
                              (crop[2] - crop[0]) * input_width,
                              (crop[3] - crop[1]) * input_height, input_width, input_height,
                              static_cast<f32>(output[0]), static_cast<f32>(output[1]),
                              crop[0] * input_width, crop[1] * input_height);
        FSR::FsrRcasCon(rcas_constants.data(),
                        static_cast<f32>(Settings::values.fsr_sharpening_slider.GetValue()) /
                            100.0f);
        const std::array<TextureHandle, 1> easu_inputs{source.texture};
        const std::array<TextureHandle, 1> rcas_inputs{easu_output};
        const bool drawn =
            Draw({.vertex = Shader::FsrVertex,
                  .fragment = Shader::Easu,
                  .inputs = easu_inputs,
                  .target = {.texture = easu_output},
                  .format = Fmt::RGBA16Float,
                  .size = output,
                  .push_constants = Bytes(easu_constants)}) &&
            Draw({.vertex = Shader::FsrVertex,
                  .fragment = Shader::Rcas,
                  .inputs = rcas_inputs,
                  .target = {.texture = rcas_output},
                  .format = Fmt::RGBA16Float,
                  .size = output,
                  .push_constants = Bytes(rcas_constants)});
        if (!drawn) {
            return source;
        }
        return {.texture = rcas_output, .size = output, .render = output, .crop = {0, 0, 1, 1}};
    }
    if (filter == Filter::Sgsr || filter == Filter::SgsrEdge) {
        // Vulkan's SGSR::Draw.
        const TextureHandle sgsr_output = Make(sgsr, output, Fmt::RGBA16Float);
        if (sgsr_output == nullptr) {
            return source;
        }
        const f32 crop_width = (crop[2] - crop[0]) * input_width;
        const f32 crop_height = (crop[3] - crop[1]) * input_height;
        constexpr f32 EdgeSharpnessMax = 2.0f;
        const f32 edge_sharpness =
            EdgeSharpnessMax -
            static_cast<f32>(Settings::values.fsr_sharpening_slider.GetValue()) / 200.0f;
        const std::array<f32, 9> constants{1.0f / input_width,
                                           1.0f / input_height,
                                           input_width,
                                           input_height,
                                           crop_width / input_width,
                                           crop_height / input_height,
                                           (std::min)(crop[0], crop[2]),
                                           (std::min)(crop[1], crop[3]),
                                           edge_sharpness};
        const std::array<TextureHandle, 1> inputs{source.texture};
        if (!Draw({.vertex = Shader::SgsrVertex,
                   .fragment = filter == Filter::SgsrEdge ? Shader::SgsrEdge : Shader::Sgsr,
                   .inputs = inputs,
                   .target = {.texture = sgsr_output},
                   .format = Fmt::RGBA16Float,
                   .size = output,
                   .push_constants = Bytes(constants)})) {
            return source;
        }
        return {.texture = sgsr_output, .size = output, .render = output, .crop = {0, 0, 1, 1}};
    }
    return source;
}

bool PresentPasses::DrawWindow(const Source& source, Settings::ScalingFilter filter,
                               Attachment target, u32 target_format,
                               std::array<u32, 2> target_size, std::array<s32, 4> screen,
                               bool keep_target, BlitBlend blend) {
    const std::optional<Shader> fragment = WindowShader(filter);
    if (!fragment || source.texture == nullptr) {
        return false;
    }
    // Vulkan's Layer::SetMatrixData and SetVertexData: an orthographic matrix over the window,
    // the screen's corners with the crop's texture coordinates, as a strip.
    const f32 width = static_cast<f32>(target_size[0]);
    const f32 height = static_cast<f32>(target_size[1]);
    const f32 x = static_cast<f32>(screen[0]);
    const f32 y = static_cast<f32>(screen[1]);
    const f32 w = static_cast<f32>(screen[2] - screen[0]);
    const f32 h = static_cast<f32>(screen[3] - screen[1]);
    const auto& crop = source.crop;
    const WindowConstants constants{
        .modelview = {2.0f / width, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f / height, 0.0f, 0.0f, 0.0f, 0.0f,
                      1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f},
        .vertices = {{{x, y, crop[0], crop[1]},
                      {x + w, y, crop[2], crop[1]},
                      {x, y + h, crop[0], crop[3]},
                      {x + w, y + h, crop[2], crop[3]}}},
    };
    const std::array<TextureHandle, 1> inputs{source.texture};
    return Draw({.vertex = Shader::PresentVertex,
                 .fragment = *fragment,
                 .inputs = inputs,
                 .nearest = filter == Settings::ScalingFilter::Mmpx,
                 .target = target,
                 .format = target_format,
                 .size = target_size,
                 .keep_target = keep_target,
                 .blend = blend,
                 .push_constants = Bytes(constants),
                 .vertices = 4,
                 .strip = true});
}

bool PresentPasses::TemporalFilter(const TemporalInputs& inputs) {
    if (inputs.color == nullptr || inputs.target.texture == nullptr || inputs.size[0] == 0 ||
        inputs.size[1] == 0 || (inputs.use_history && inputs.motion == nullptr)) {
        return false;
    }
    // At half-float precision, as on Vulkan: in the output's format a frame's small share of the
    // blend rounded away.
    bool reset = false;
    if (!history[0] || !history[1] || history_size != inputs.size) {
        for (Object& image : history) {
            image = CreateTexture(device, {.kind = TextureKind::e2D,
                                           .format = Fmt::RGBA16Float,
                                           .width = inputs.size[0],
                                           .height = inputs.size[1],
                                           .shader_write = true});
        }
        if (!history[0] || !history[1]) {
            history = {};
            return false;
        }
        history_size = inputs.size;
        history_current = 0;
        reset = true;
        LOG_INFO(Render_Metal, "Metal: temporal stand-in into {}x{}", inputs.size[0],
                 inputs.size[1]);
    }
    // Vulkan's weights (EDEN_NX2_TEMPORAL_WEIGHT, _KERNEL, _CLIP, _CLIP_MOTION, _KERNEL_MOTION).
    const auto tuned = [](const char* name, f32 fallback) {
        const char* const value = VulkanSwitch(name);
        return value != nullptr ? static_cast<f32>(std::strtod(value, nullptr)) : fallback;
    };
    static const f32 current_weight = tuned("EDEN_NX2_TEMPORAL_WEIGHT", 0.1f);
    static const f32 kernel_scale = tuned("EDEN_NX2_TEMPORAL_KERNEL", 3.0f);
    static const f32 clip_width = tuned("EDEN_NX2_TEMPORAL_CLIP", 1.5f);
    static const f32 clip_motion = tuned("EDEN_NX2_TEMPORAL_CLIP_MOTION", 1.0f);
    static const f32 kernel_motion = tuned("EDEN_NX2_TEMPORAL_KERNEL_MOTION", 1.0f);
    // full_screen_triangle.vert's scale and offset, then temporal_upscale.frag's constants.
    struct UpscaleConstants {
        std::array<f32, 2> tex_scale;
        std::array<f32, 2> tex_offset;
        std::array<f32, 2> jitter;
        std::array<f32, 2> motion_scale;
        f32 current_weight;
        f32 kernel_scale;
        f32 clip_width;
        f32 clip_motion;
        f32 kernel_motion;
        f32 padding;
    };
    static_assert(sizeof(UpscaleConstants) == 56);
    const UpscaleConstants upscale{
        .tex_scale = {1.0f, 1.0f},
        .tex_offset = {0.0f, 0.0f},
        .jitter = inputs.jitter,
        .motion_scale = inputs.motion_scale,
        .current_weight = !inputs.use_history ? -1.0f : reset ? 1.0f : current_weight,
        .kernel_scale = kernel_scale,
        .clip_width = clip_width,
        .clip_motion = clip_motion,
        .kernel_motion = kernel_motion,
        .padding = 0.0f,
    };
    const size_t read = history_current;
    const size_t write = read ^ 1;
    // A motion texture is bound without history too; the shader does not read it then.
    const std::array<TextureHandle, 4> upscale_inputs{
        inputs.color, inputs.motion != nullptr ? inputs.motion : inputs.color, history[read].Get(),
        history[write].Get()};
    if (!Draw({.vertex = Shader::FullScreenVertex,
               .fragment = Shader::TemporalUpscale,
               .inputs = upscale_inputs,
               .target = inputs.target,
               .format = inputs.format,
               .size = inputs.size,
               .keep_target = true,
               .push_constants = Bytes(upscale)})) {
        return false;
    }
    history_current = write;
    // RCAS set by the FSR sharpness slider (0 % = stored 200 is off), as Vulkan's;
    // EDEN_NX2_TEMPORAL_SHARPEN=<stops>|off overrides it. It reads the unsharpened history.
    static const char* const sharpen_override = VulkanSwitch("EDEN_NX2_TEMPORAL_SHARPEN");
    const int slider = Settings::values.fsr_sharpening_slider.GetValue();
    bool sharpen = slider < 200;
    f32 sharpness = static_cast<f32>(slider) / 100.0f;
    if (sharpen_override != nullptr) {
        sharpen = std::string_view{sharpen_override} != "off";
        sharpness = static_cast<f32>(std::strtod(sharpen_override, nullptr));
    }
    if (!sharpen) {
        return true;
    }
    struct SharpenConstants {
        std::array<f32, 2> tex_scale;
        std::array<f32, 2> tex_offset;
        std::array<u32, 4> rcas;
    };
    SharpenConstants constants{.tex_scale = {1.0f, 1.0f}, .tex_offset = {0.0f, 0.0f}, .rcas = {}};
    FSR::FsrRcasCon(constants.rcas.data(), sharpness);
    const std::array<TextureHandle, 1> sharpen_inputs{history[write].Get()};
    Draw({.vertex = Shader::FullScreenVertex,
          .fragment = Shader::TemporalSharpen,
          .inputs = sharpen_inputs,
          .nearest = true,
          .target = inputs.target,
          .format = inputs.format,
          .size = inputs.size,
          .keep_target = true,
          .push_constants = Bytes(constants)});
    return true;
}

u64 PresentPasses::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

} // namespace Metal
