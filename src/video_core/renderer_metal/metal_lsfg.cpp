// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2025 lsfg-vk
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_LSFG

#include <algorithm>
#include <cstring>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/logging.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_lsfg.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

namespace Shader = VideoCore::FrameGen::PerformanceShader;

/// Vulkan's LsfgConstants (48 bytes).
struct Constants {
    std::array<u32, 2> input_offset;
    u32 first_iter;
    u32 first_iter_s;
    u32 advanced_color_kind;
    u32 hdr_support;
    f32 resolution_inv_scale;
    f32 timestamp;
    f32 ui_threshold;
    std::array<u32, 3> padding;
};
static_assert(sizeof(Constants) == 48);

constexpr u32 FlowFormat = Fmt::R8Unorm;
constexpr u32 MotionFormat = Fmt::RGBA16Float;

constexpr f32 SlotTimestamp(size_t slot) {
    size_t count = 1;
    while (LsfgChain::GenerationSlot(count + 1, 0) <= slot) {
        ++count;
    }
    return static_cast<f32>(slot - LsfgChain::GenerationSlot(count, 0) + 1) /
           static_cast<f32>(count + 1);
}

std::array<u32, 2> Halve(std::array<u32, 2> size) {
    return {(size[0] + 1) >> 1, (size[1] + 1) >> 1};
}

std::array<u32, 3> LocalSize(std::span<const u32> words) {
    size_t offset = 5;
    while (offset < words.size()) {
        const u32 length = words[offset] >> 16;
        const u32 opcode = words[offset] & 0xffff;
        if (length == 0 || offset + length > words.size()) {
            break;
        }
        if (opcode == 16 && length >= 6 && words[offset + 2] == 17) {
            return {words[offset + 3], words[offset + 4], words[offset + 5]};
        }
        offset += length;
    }
    return {8, 8, 1};
}

Object MakeSampler(const Device& device, Tegra::Texture::WrapMode wrap, bool white) {
    Tegra::Texture::TSCEntry tsc{};
    tsc.wrap_u.Assign(wrap);
    tsc.wrap_v.Assign(wrap);
    tsc.wrap_p.Assign(wrap);
    tsc.mag_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.min_filter.Assign(Tegra::Texture::TextureFilter::Linear);
    tsc.mipmap_filter.Assign(Tegra::Texture::TextureMipmapFilter::Linear);
    tsc.border_color = white ? std::array<f32, 4>{1, 1, 1, 1} : std::array<f32, 4>{0, 0, 0, 0};
    return CreateSampler(device, tsc);
}

} // Anonymous namespace

LsfgChain::LsfgChain(const Device& device_, Scheduler& scheduler_,
                     const VideoCore::FrameGen::ShaderModules& shaders, std::array<u32, 2> extent,
                     u32 format, f32 flow_scale_)
    : device{device_}, scheduler{scheduler_}, modules{shaders}, flow_scale{flow_scale_} {
    using Tegra::Texture::WrapMode;
    linear = MakeSampler(device, WrapMode::Border, false);
    white_border = MakeSampler(device, WrapMode::Border, true);
    edge = MakeSampler(device, WrapMode::ClampToEdge, false);
    dummy = MakeImage({1, 1}, Fmt::RGBA8Unorm);
    for (Image& frame : frames) {
        frame = MakeImage({(std::max)(1U, extent[0]), (std::max)(1U, extent[1])}, format);
    }
    flow_extent = {(std::max)(1U, static_cast<u32>(static_cast<f32>(extent[0]) * flow_scale)),
                   (std::max)(1U, static_cast<u32>(static_cast<f32>(extent[1]) * flow_scale))};
    for (size_t i = 0; i < MipLevels; ++i) {
        flow[i] = MakeImage({flow_extent[0] >> i, flow_extent[1] >> i}, FlowFormat);
    }
    for (size_t i = 0; i < MipLevels; ++i) {
        Alpha& level = alpha[i];
        const std::array<u32, 2> half = Halve(flow[i].size);
        const std::array<u32, 2> quarter = Halve(half);
        level.temp1 = MakeImage(half, Fmt::RGBA8Unorm);
        level.temp2 = MakeImage(half, Fmt::RGBA8Unorm);
        for (size_t j = 0; j < 2; ++j) {
            level.temp3[j] = MakeImage(quarter, Fmt::RGBA8Unorm);
            for (size_t h = 0; h < HistorySlots; ++h) {
                level.out[h][j] = MakeImage(quarter, Fmt::RGBA8Unorm);
            }
        }
    }
    {
        const std::array<u32, 2> beta_extent = alpha[0].out[0][0].size;
        for (size_t i = 0; i < 2; ++i) {
            beta_temp1[i] = MakeImage(beta_extent, Fmt::RGBA8Unorm);
            beta_temp2[i] = MakeImage(beta_extent, Fmt::RGBA8Unorm);
        }
        for (size_t i = 0; i < BetaOutputs; ++i) {
            beta_out[i] = MakeImage({beta_extent[0] >> i, beta_extent[1] >> i}, FlowFormat);
        }
    }
    for (size_t i = 0; i < MipLevels; ++i) {
        const size_t level = MipLevels - 1 - i;
        const std::array<u32, 2> size = alpha[level].out[0][0].size;
        for (Image& image : gamma[i].temp1) {
            image = MakeImage(size, Fmt::RGBA8Unorm);
        }
        for (Image& image : gamma[i].temp2) {
            image = MakeImage(size, Fmt::RGBA8Unorm);
        }
        gamma[i].out = MakeImage(size, MotionFormat);
        if (i >= 4) {
            Delta& d = delta[i - 4];
            for (Image& image : d.temp1) {
                image = MakeImage(size, Fmt::RGBA8Unorm);
            }
            for (Image& image : d.temp2) {
                image = MakeImage(size, Fmt::RGBA8Unorm);
            }
            d.out1 = MakeImage(size, MotionFormat);
            d.out2 = MakeImage(size, MotionFormat);
        }
    }
    // Every program up front: a missing or untranslatable one ends the chain.
    for (const u32 id : {Shader::MIPMAPS, Shader::GENERATE}) {
        valid = valid && GetProgram(id) != nullptr;
    }
    for (const auto& group : {std::span<const u32>(Shader::ALPHA), std::span<const u32>(Shader::BETA),
                              std::span<const u32>(Shader::GAMMA),
                              std::span<const u32>(Shader::DELTA)}) {
        for (const u32 id : group) {
            valid = valid && GetProgram(id) != nullptr;
        }
    }
    valid = valid && linear && white_border && edge && dummy.texture;
    const auto made = [](const Image& image) { return static_cast<bool>(image.texture); };
    valid = valid && std::ranges::all_of(frames, made) && std::ranges::all_of(flow, made) &&
            std::ranges::all_of(beta_out, made);
}

LsfgChain::~LsfgChain() = default;

LsfgChain::Image LsfgChain::MakeImage(std::array<u32, 2> size, u32 format) {
    Image image;
    image.size = {(std::max)(1U, size[0]), (std::max)(1U, size[1])};
    image.texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                           .format = format,
                                           .width = image.size[0],
                                           .height = image.size[1],
                                           .render_target = true,
                                           .shader_write = true,
                                           .pixel_format_view = false});
    if (image.texture) {
        // Vulkan's images start undefined; these start black.
        ClearPass clear{};
        clear.color[0] = Attachment{.texture = image.texture.Get()};
        clear.clear_color[0] = true;
        clear.width = image.size[0];
        clear.height = image.size[1];
        scheduler.Clear(clear);
    }
    return image;
}

LsfgChain::Program* LsfgChain::GetProgram(u32 shader_id) {
    if (const auto it = programs.find(shader_id); it != programs.end()) {
        return it->second.state ? &it->second : nullptr;
    }
    Program& program = programs[shader_id];
    const auto code = modules.find(shader_id);
    if (code == modules.end()) {
        LOG_WARNING(Render_Metal, "LSFG: shader {} is not among the loaded modules", shader_id);
        return nullptr;
    }
    ShaderTranslate::Options options{};
    options.ios = TranslateForIos;
    options.apple_family = DeviceAppleFamily(device);
    const ShaderTranslate::Result msl =
        ShaderTranslate::SpirvToMsl(code->second, ShaderTranslate::Stage::Compute, options);
    std::string error = msl.error;
    if (!msl.source.empty()) {
        program.library = CreateLibrary(device, msl.source, error);
        program.function =
            program.library ? CreateFunction(program.library.Get(), msl.entry_point) : Object{};
        program.state =
            program.function ? CreateComputePipelineState(device, program.function.Get(), error)
                             : Object{};
    }
    if (!program.state) {
        LOG_WARNING(Render_Metal, "LSFG: shader {} not built: {}", shader_id, error);
        return nullptr;
    }
    program.slots = msl.slots;
    program.group = LocalSize(code->second);
    return &program;
}

LsfgChain::Bind LsfgChain::Texture(const Image* image) {
    Bind bind;
    bind.kind = Bind::Kind::Texture;
    bind.texture = image != nullptr ? image->H() : nullptr;
    return bind;
}

LsfgChain::Bind LsfgChain::Sampler(void* sampler) {
    Bind bind;
    bind.kind = Bind::Kind::Sampler;
    bind.sampler = sampler;
    return bind;
}

LsfgChain::Bind LsfgChain::Constants(f32 timestamp, bool first_iter, bool first_iter_s) const {
    Bind bind;
    bind.kind = Bind::Kind::Constants;
    const ::Metal::Constants constants{
        .input_offset = {0, 0},
        .first_iter = first_iter ? 1U : 0U,
        .first_iter_s = first_iter_s ? 1U : 0U,
        .advanced_color_kind = 0,
        .hdr_support = 0,
        .resolution_inv_scale = 1.0f / flow_scale,
        .timestamp = timestamp,
        .ui_threshold = 0.5f,
        .padding = {0, 0, 0},
    };
    std::memcpy(bind.constants.data(), &constants, sizeof(constants));
    return bind;
}

void LsfgChain::Run(u32 shader_id, std::vector<Bind> binds, std::array<u32, 2> size,
                    u32 tile_shift) {
    const Program* const program = GetProgram(shader_id);
    if (program == nullptr) {
        return;
    }
    scheduler.BeginCompute(program->state.Get());
    for (const ShaderTranslate::ResourceSlot& slot : program->slots) {
        if (slot.binding >= binds.size()) {
            continue;
        }
        const Bind& bind = binds[slot.binding];
        switch (slot.kind) {
        case ShaderTranslate::ResourceKind::Buffer:
            if (bind.kind == Bind::Kind::Constants) {
                scheduler.BindBytes(ShaderStage::Compute, slot.index, bind.constants.data(),
                                    static_cast<u32>(bind.constants.size()));
            }
            break;
        case ShaderTranslate::ResourceKind::Texture:
            scheduler.BindTexture(ShaderStage::Compute, slot.index,
                                  bind.texture != nullptr ? bind.texture : dummy.H());
            break;
        case ShaderTranslate::ResourceKind::Sampler:
            scheduler.BindSampler(ShaderStage::Compute, slot.index, bind.sampler);
            break;
        }
    }
    const u32 mask = (1U << tile_shift) - 1;
    scheduler.Dispatch({(size[0] + mask) >> tile_shift, (size[1] + mask) >> tile_shift, 1},
                       program->group);
}

void LsfgChain::DispatchShared(u64 frame_count) {
    const size_t slot = frame_count % 2;
    {
        std::vector<Bind> binds{Constants(0.0f, false, false), Sampler(linear.Get()),
                                Texture(frames[slot])};
        for (const Image& image : flow) {
            binds.push_back(Texture(image));
        }
        Run(Shader::MIPMAPS, std::move(binds), flow_extent, 6);
    }
    const size_t history = frame_count % HistorySlots;
    for (size_t stage = 0; stage < 4; ++stage) {
        for (Alpha& level : alpha) {
            const Image& input = flow[&level - alpha.data()];
            std::vector<Bind> binds{Sampler(linear.Get())};
            std::array<u32, 2> size = level.temp1.size;
            switch (stage) {
            case 0:
                binds.push_back(Texture(input));
                binds.push_back(Texture(level.temp1));
                break;
            case 1:
                binds.push_back(Texture(level.temp1));
                binds.push_back(Texture(level.temp2));
                break;
            case 2:
                binds.push_back(Texture(level.temp2));
                binds.push_back(Texture(level.temp3[0]));
                binds.push_back(Texture(level.temp3[1]));
                break;
            default:
                size = level.temp3[0].size;
                binds.push_back(Texture(level.temp3[0]));
                binds.push_back(Texture(level.temp3[1]));
                binds.push_back(Texture(level.out[history][0]));
                binds.push_back(Texture(level.out[history][1]));
                break;
            }
            Run(Shader::ALPHA[stage], std::move(binds), size, 3);
        }
    }
    // Beta over alpha level 0's history.
    const History& inputs = alpha[0].out;
    const std::array<u32, 2> extent = beta_temp1[0].size;
    {
        std::vector<Bind> binds{Sampler(white_border.Get())};
        for (const size_t index : {(history + 1) % HistorySlots, (history + 2) % HistorySlots,
                                   history % HistorySlots}) {
            binds.push_back(Texture(inputs[index][0]));
            binds.push_back(Texture(inputs[index][1]));
        }
        binds.push_back(Texture(beta_temp1[0]));
        binds.push_back(Texture(beta_temp1[1]));
        Run(Shader::BETA[0], std::move(binds), extent, 3);
    }
    const auto step = [&](size_t stage, const Pair& from, const Pair& to) {
        Run(Shader::BETA[stage],
            {Sampler(linear.Get()), Texture(from[0]), Texture(from[1]), Texture(to[0]),
             Texture(to[1])},
            extent, 3);
    };
    step(1, beta_temp1, beta_temp2);
    step(2, beta_temp2, beta_temp1);
    step(3, beta_temp1, beta_temp2);
    {
        std::vector<Bind> binds{Constants(0.5f, false, false), Sampler(linear.Get()),
                                Texture(beta_temp2[0]), Texture(beta_temp2[1])};
        for (const Image& image : beta_out) {
            binds.push_back(Texture(image));
        }
        Run(Shader::BETA[4], std::move(binds), extent, 5);
    }
}

void LsfgChain::GammaDispatch(size_t index, u64 frame_count, size_t slot) {
    Gamma& g = gamma[index];
    const size_t level = MipLevels - 1 - index;
    const History& inputs = alpha[level].out;
    const Image& flow_input = beta_out[(std::min)(level, BetaOutputs - 1)];
    const Image* const previous = index == 0 ? nullptr : &gamma[index - 1].out;
    const std::array<u32, 2> extent = g.temp1[0].size;
    const size_t history = frame_count % HistorySlots;
    const f32 timestamp = SlotTimestamp(slot);
    {
        std::vector<Bind> binds{Constants(timestamp, previous == nullptr, false),
                                Sampler(white_border.Get()), Sampler(edge.Get())};
        for (const size_t at : {(history + 2) % HistorySlots, history % HistorySlots}) {
            binds.push_back(Texture(inputs[at][0]));
            binds.push_back(Texture(inputs[at][1]));
        }
        binds.push_back(Texture(previous));
        for (const Image& image : g.temp1) {
            binds.push_back(Texture(image));
        }
        Run(Shader::GAMMA[0], std::move(binds), extent, 3);
    }
    Run(Shader::GAMMA[1],
        {Sampler(linear.Get()), Texture(g.temp1[0]), Texture(g.temp1[1]), Texture(g.temp1[2]),
         Texture(g.temp2[0]), Texture(g.temp2[1])},
        extent, 3);
    Run(Shader::GAMMA[2],
        {Sampler(linear.Get()), Texture(g.temp2[0]), Texture(g.temp2[1]), Texture(g.temp1[0]),
         Texture(g.temp1[1])},
        extent, 3);
    Run(Shader::GAMMA[3],
        {Sampler(linear.Get()), Texture(g.temp1[0]), Texture(g.temp1[1]), Texture(g.temp2[0]),
         Texture(g.temp2[1])},
        extent, 3);
    Run(Shader::GAMMA[4],
        {Constants(timestamp, previous == nullptr, false), Sampler(linear.Get()),
         Sampler(edge.Get()), Texture(g.temp2[0]), Texture(g.temp2[1]), Texture(previous),
         Texture(flow_input), Texture(g.out)},
        extent, 3);
}

void LsfgChain::DeltaDispatch(size_t index, u64 frame_count, size_t slot) {
    // Delta instance `index` runs at gamma level index + 4.
    Delta& d = delta[index];
    const size_t gamma_index = index + 4;
    const size_t level = MipLevels - 1 - gamma_index;
    const History& inputs = alpha[level].out;
    const Image& flow_input = beta_out[level];
    const Image* const previous_gamma = index == 0 ? nullptr : &gamma[gamma_index - 1].out;
    const Image* const previous1 = index == 0 ? nullptr : &delta[index - 1].out1;
    const Image* const previous2 = index == 0 ? nullptr : &delta[index - 1].out2;
    const std::array<u32, 2> extent = d.temp1[0].size;
    const size_t history = frame_count % HistorySlots;
    const f32 timestamp = SlotTimestamp(slot);
    const auto first = [&](u32 shader, const Image* extra, bool six) {
        std::vector<Bind> binds{Constants(timestamp, false, previous_gamma == nullptr),
                                Sampler(white_border.Get()), Sampler(edge.Get())};
        for (const size_t at : {(history + 2) % HistorySlots, history % HistorySlots}) {
            binds.push_back(Texture(inputs[at][0]));
            binds.push_back(Texture(inputs[at][1]));
        }
        binds.push_back(Texture(previous_gamma));
        if (six) {
            binds.push_back(Texture(extra));
            binds.push_back(Texture(d.temp2[0]));
        } else {
            for (const Image& image : d.temp1) {
                binds.push_back(Texture(image));
            }
        }
        Run(shader, std::move(binds), extent, 3);
    };
    first(Shader::DELTA[0], nullptr, false);
    Run(Shader::DELTA[1],
        {Sampler(linear.Get()), Texture(d.temp1[0]), Texture(d.temp1[1]), Texture(d.temp1[2]),
         Texture(d.temp2[0]), Texture(d.temp2[1])},
        extent, 3);
    Run(Shader::DELTA[2],
        {Sampler(linear.Get()), Texture(d.temp2[0]), Texture(d.temp2[1]), Texture(d.temp1[0]),
         Texture(d.temp1[1])},
        extent, 3);
    Run(Shader::DELTA[3],
        {Sampler(linear.Get()), Texture(d.temp1[0]), Texture(d.temp1[1]), Texture(d.temp2[0]),
         Texture(d.temp2[1])},
        extent, 3);
    Run(Shader::DELTA[4],
        {Constants(timestamp, false, previous_gamma == nullptr), Sampler(linear.Get()),
         Sampler(edge.Get()), Texture(d.temp2[0]), Texture(d.temp2[1]), Texture(previous_gamma),
         Texture(flow_input), Texture(d.out1)},
        extent, 3);
    first(Shader::DELTA[5], previous1, true);
    Run(Shader::DELTA[6], {Sampler(linear.Get()), Texture(d.temp2[0]), Texture(d.temp1[0])},
        extent, 3);
    Run(Shader::DELTA[7], {Sampler(linear.Get()), Texture(d.temp1[0]), Texture(d.temp2[0])},
        extent, 3);
    Run(Shader::DELTA[8], {Sampler(linear.Get()), Texture(d.temp2[0]), Texture(d.temp1[0])},
        extent, 3);
    Run(Shader::DELTA[9],
        {Constants(timestamp, false, previous_gamma == nullptr), Sampler(linear.Get()),
         Sampler(edge.Get()), Texture(d.temp1[0]), Texture(previous2), Texture(d.out2)},
        extent, 3);
}

void LsfgChain::DispatchGeneration(u64 frame_count, size_t generation_count, size_t generation,
                                   TextureHandle target, std::array<u32, 2> size) {
    const size_t slot = GenerationSlot(generation_count, generation);
    for (size_t i = 0; i < MipLevels; ++i) {
        GammaDispatch(i, frame_count, slot);
        if (i >= 4) {
            DeltaDispatch(i - 4, frame_count, slot);
        }
    }
    const size_t set = frame_count % 2;
    Bind output;
    output.kind = Bind::Kind::Texture;
    output.texture = target;
    Run(Shader::GENERATE,
        {Constants(SlotTimestamp(slot), false, false), Sampler(linear.Get()), Sampler(edge.Get()),
         Texture(frames[1 - set]), Texture(frames[set]), Texture(gamma[MipLevels - 1].out),
         Texture(delta[DeltaInstances - 1].out1), Texture(delta[DeltaInstances - 1].out2), output},
        size, 4);
}

} // namespace Metal

#endif // HAS_LSFG
