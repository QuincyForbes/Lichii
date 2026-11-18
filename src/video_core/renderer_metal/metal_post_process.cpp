// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#ifdef HAS_RESHADE

#include <algorithm>
#include <cstring>
#include <random>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/post_processing/fx_chain.h"
#include "video_core/post_processing/fx_compile.h"
#include "video_core/post_processing/fx_effect.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_post_process.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/textures/texture.h"

namespace Metal {

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

constexpr u32 BACKBUFFER_FORMAT = Fmt::RGBA16Float;
constexpr size_t NO_TEXTURE = ~size_t{0};
/// What setBytes takes: effects with more uniform data are skipped.
constexpr u32 MAX_UNIFORM_BYTES = 4096;

u32 ToMetalFormat(reshadefx::texture_format format) {
    switch (format) {
    case reshadefx::texture_format::r8:
        return Fmt::R8Unorm;
    case reshadefx::texture_format::r16f:
        return Fmt::R16Float;
    case reshadefx::texture_format::r32f:
        return Fmt::R32Float;
    case reshadefx::texture_format::rg8:
        return Fmt::RG8Unorm;
    case reshadefx::texture_format::rg16:
        return Fmt::RG16Unorm;
    case reshadefx::texture_format::rg16f:
        return Fmt::RG16Float;
    case reshadefx::texture_format::rg32f:
        return Fmt::RG32Float;
    case reshadefx::texture_format::rgba8:
        return Fmt::RGBA8Unorm;
    case reshadefx::texture_format::rgba16:
        return Fmt::RGBA16Unorm;
    case reshadefx::texture_format::rgba16f:
        return Fmt::RGBA16Float;
    case reshadefx::texture_format::rgba32f:
        return Fmt::RGBA32Float;
    case reshadefx::texture_format::rgb10a2:
        return Fmt::RGB10A2Unorm;
    default:
        return Fmt::RGBA8Unorm;
    }
}

Tegra::Texture::WrapMode ToWrap(reshadefx::texture_address_mode mode) {
    using Tegra::Texture::WrapMode;
    switch (mode) {
    case reshadefx::texture_address_mode::wrap:
        return WrapMode::Wrap;
    case reshadefx::texture_address_mode::mirror:
        return WrapMode::Mirror;
    case reshadefx::texture_address_mode::border:
        return WrapMode::Border;
    case reshadefx::texture_address_mode::clamp:
    default:
        return WrapMode::ClampToEdge;
    }
}

Maxwell::Blend::Factor ToFactor(reshadefx::blend_factor factor) {
    using Factor = Maxwell::Blend::Factor;
    switch (factor) {
    case reshadefx::blend_factor::zero:
        return Factor::Zero_D3D;
    case reshadefx::blend_factor::source_color:
        return Factor::SourceColor_D3D;
    case reshadefx::blend_factor::source_alpha:
        return Factor::SourceAlpha_D3D;
    case reshadefx::blend_factor::one_minus_source_color:
        return Factor::OneMinusSourceColor_D3D;
    case reshadefx::blend_factor::one_minus_source_alpha:
        return Factor::OneMinusSourceAlpha_D3D;
    case reshadefx::blend_factor::dest_color:
        return Factor::DestColor_D3D;
    case reshadefx::blend_factor::dest_alpha:
        return Factor::DestAlpha_D3D;
    case reshadefx::blend_factor::one_minus_dest_color:
        return Factor::OneMinusDestColor_D3D;
    case reshadefx::blend_factor::one_minus_dest_alpha:
        return Factor::OneMinusDestAlpha_D3D;
    case reshadefx::blend_factor::one:
    default:
        return Factor::One_D3D;
    }
}

Maxwell::Blend::Equation ToEquation(reshadefx::blend_op op) {
    using Equation = Maxwell::Blend::Equation;
    switch (op) {
    case reshadefx::blend_op::subtract:
        return Equation::Subtract_D3D;
    case reshadefx::blend_op::reverse_subtract:
        return Equation::ReverseSubtract_D3D;
    case reshadefx::blend_op::min:
        return Equation::Min_D3D;
    case reshadefx::blend_op::max:
        return Equation::Max_D3D;
    case reshadefx::blend_op::add:
    default:
        return Equation::Add_D3D;
    }
}

/// The Maxwell topology the pass draws (MTLPrimitiveType through PrimitiveType) and the pipeline's
/// topology class (0 points, 1 lines, 2 triangles).
std::pair<Maxwell::PrimitiveTopology, u32> ToTopology(reshadefx::primitive_topology topology) {
    using Topology = Maxwell::PrimitiveTopology;
    switch (topology) {
    case reshadefx::primitive_topology::point_list:
        return {Topology::Points, 0};
    case reshadefx::primitive_topology::line_list:
        return {Topology::Lines, 1};
    case reshadefx::primitive_topology::line_strip:
        return {Topology::LineStrip, 1};
    case reshadefx::primitive_topology::triangle_strip:
        return {Topology::TriangleStrip, 2};
    case reshadefx::primitive_topology::triangle_list:
    default:
        return {Topology::Triangles, 2};
    }
}

Object MakeSampler(const Device& device, const reshadefx::sampler& sampler) {
    using namespace Tegra::Texture;
    TSCEntry tsc{};
    tsc.wrap_u.Assign(ToWrap(sampler.address_u));
    tsc.wrap_v.Assign(ToWrap(sampler.address_v));
    tsc.wrap_p.Assign(ToWrap(sampler.address_w));
    // The filter's bits as Vulkan's chain reads them: 0x10 minification, 0x04 magnification.
    const u32 filter = static_cast<u32>(sampler.filter);
    tsc.min_filter.Assign((filter & 0x10) != 0 ? TextureFilter::Linear : TextureFilter::Nearest);
    tsc.mag_filter.Assign((filter & 0x04) != 0 ? TextureFilter::Linear : TextureFilter::Nearest);
    tsc.mipmap_filter.Assign(TextureMipmapFilter::None);
    return CreateSampler(device, tsc);
}

} // Anonymous namespace

PostProcess::PostProcess(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_} {
    start = previous = std::chrono::steady_clock::now();
}

PostProcess::~PostProcess() = default;

bool PostProcess::Wanted() {
    return Settings::values.post_shader_enabled.GetValue() &&
           VideoCore::FxChain::Instance().Size() != 0;
}

u64 PostProcess::TakePasses() noexcept {
    return std::exchange(passes, 0);
}

bool PostProcess::BuildEffects() {
    const auto snapshot = VideoCore::FxChain::Instance().Snapshot();
    generation = snapshot.generation;
    effects.clear();
    const auto root = VideoCore::GetFxRootDirectory();
    for (size_t entry_index = 0; entry_index < snapshot.entries.size(); ++entry_index) {
        const auto& entry = snapshot.entries[entry_index];
        const auto path = root / entry.file;
        const auto compiled = VideoCore::CompileFxEffect(path, extent[0], extent[1], 8);
        if (!compiled.Succeeded()) {
            LOG_ERROR(Render_Metal, "Post-processing effect '{}' failed to compile:\n{}",
                      entry.file, compiled.error);
            continue;
        }
        const auto& module = compiled.module;
        const auto technique = std::find_if(
            module.techniques.begin(), module.techniques.end(),
            [&](const reshadefx::technique& t) { return t.name == entry.technique; });
        if (technique == module.techniques.end()) {
            LOG_ERROR(Render_Metal, "Effect '{}' has no technique '{}'", entry.file,
                      entry.technique);
            continue;
        }
        if (module.total_uniform_size > MAX_UNIFORM_BYTES) {
            LOG_WARNING(Render_Metal, "Effect '{}' has {} bytes of uniforms; skipped", entry.file,
                        module.total_uniform_size);
            continue;
        }

        Effect effect;
        effect.entry_index = entry_index;
        effect.file = entry.file;
        effect.uniform_size = module.total_uniform_size;

        for (const auto& texture : module.textures) {
            Texture out;
            out.name = texture.unique_name;
            out.size = {texture.width, texture.height};
            out.format = ToMetalFormat(texture.format);
            if (texture.semantic == "COLOR") {
                out.is_backbuffer = true;
            } else if (texture.semantic != "DEPTH") {
                out.texture = CreateTexture(device, {.kind = TextureKind::e2D,
                                                     .format = out.format,
                                                     .width = (std::max)(out.size[0], 1U),
                                                     .height = (std::max)(out.size[1], 1U),
                                                     .render_target = true});
            }
            effect.textures.push_back(std::move(out));
        }
        for (const auto& sampler : module.samplers) {
            Sampler out;
            out.texture_index = NO_TEXTURE;
            for (size_t i = 0; i < effect.textures.size(); ++i) {
                if (effect.textures[i].name == sampler.texture_name) {
                    out.texture_index = i;
                    break;
                }
            }
            out.sampler = MakeSampler(device, sampler);
            effect.samplers.push_back(std::move(out));
        }
        for (const auto& uniform : module.uniforms) {
            UniformWrite write;
            write.name = uniform.name;
            write.offset = uniform.offset;
            write.components = (std::min)(uniform.type.components(), 4U);
            if (uniform.type.is_boolean()) {
                write.kind = UniformKind::Boolean;
            } else if (uniform.type.is_integral()) {
                write.kind = UniformKind::Integer;
            }
            for (const auto& annotation : uniform.annotations) {
                if (annotation.name != "source") {
                    continue;
                }
                const std::string& source = annotation.value.string_data;
                if (source == "frametime") {
                    write.source = UniformSource::FrameTime;
                } else if (source == "framecount") {
                    write.source = UniformSource::FrameCount;
                } else if (source == "timer") {
                    write.source = UniformSource::Timer;
                } else if (source == "random") {
                    write.source = UniformSource::Random;
                } else if (source == "pingpong") {
                    write.source = UniformSource::PingPong;
                }
            }
            if (uniform.has_initializer_value) {
                for (u32 i = 0; i < write.components; ++i) {
                    write.fallback[i] = write.kind == UniformKind::Floating
                                            ? uniform.initializer_value.as_float[i]
                                            : static_cast<f32>(uniform.initializer_value.as_int[i]);
                }
            }
            write.args = {0.0f, 1.0f, 1.0f, 0.0f};
            for (const auto& annotation : uniform.annotations) {
                if (annotation.name == "min" && annotation.type.is_floating_point()) {
                    write.args[0] = annotation.value.as_float[0];
                }
                if (annotation.name == "max" && annotation.type.is_floating_point()) {
                    write.args[1] = annotation.value.as_float[0];
                }
                if (annotation.name == "step" && annotation.type.is_floating_point()) {
                    write.args[2] = annotation.value.as_float[0];
                }
            }
            write.state = write.args[0];
            effect.uniforms.push_back(std::move(write));
        }

        ShaderTranslate::Options options{};
        options.ios = TranslateForIos;
        options.apple_family = DeviceAppleFamily(device);
        const auto translate = [&](const std::string& name, ShaderTranslate::Stage stage,
                                   Program& program) {
            const auto words = compiled.entry_points.find(name);
            if (words == compiled.entry_points.end()) {
                return false;
            }
            const ShaderTranslate::Result msl =
                ShaderTranslate::SpirvToMsl(words->second, stage, options);
            std::string error = msl.error;
            if (!msl.source.empty()) {
                program.library = CreateLibrary(device, msl.source, error);
                program.function = program.library
                                       ? CreateFunction(program.library.Get(), msl.entry_point)
                                       : Object{};
            }
            if (!program.function) {
                LOG_WARNING(Render_Metal, "Effect '{}' entry point '{}' not built: {}",
                            entry.file, name, error);
                return false;
            }
            program.slots = msl.slots;
            return true;
        };

        for (const auto& pass : technique->passes) {
            Pass out;
            out.vertices = pass.num_vertices;
            out.clear = pass.clear_render_targets != 0;
            out.target_texture = NO_TEXTURE;
            out.size = extent;
            const std::string& target = pass.render_target_names[0];
            if (target.empty()) {
                out.writes_backbuffer = true;
            } else {
                for (size_t i = 0; i < effect.textures.size(); ++i) {
                    if (effect.textures[i].name == target) {
                        out.target_texture = i;
                        out.size = effect.textures[i].size;
                        break;
                    }
                }
                if (out.target_texture == NO_TEXTURE ||
                    !effect.textures[out.target_texture].texture) {
                    LOG_WARNING(Render_Metal, "Effect '{}' pass targets unknown texture '{}'",
                                entry.file, target);
                    out.target_texture = NO_TEXTURE;
                    out.writes_backbuffer = true;
                }
            }
            if (pass.viewport_width != 0 && pass.viewport_height != 0) {
                out.size = {pass.viewport_width, pass.viewport_height};
            }
            out.target_format = out.writes_backbuffer
                                    ? BACKBUFFER_FORMAT
                                    : effect.textures[out.target_texture].format;
            for (const auto& binding : pass.sampler_bindings) {
                out.sampler_bindings.emplace_back(binding.entry_point_binding, binding.index);
            }
            const auto [topology, topology_class] = ToTopology(pass.topology);
            options.drop_point_size = topology_class != 0;
            if (!translate(pass.vs_entry_point, ShaderTranslate::Stage::Vertex, out.vertex) ||
                !translate(pass.ps_entry_point, ShaderTranslate::Stage::Fragment, out.fragment)) {
                continue;
            }
            out.topology = *PrimitiveType(topology);
            auto desc = Zeroed<RenderPipelineDesc>();
            desc.vertex_function = out.vertex.function.Get();
            desc.fragment_function = out.fragment.function.Get();
            desc.color_formats[0] = out.target_format;
            BlendTargetState& blend = desc.blend[0];
            blend.enable = pass.blend_enable[0] ? 1 : 0;
            blend.write_mask = pass.render_target_write_mask[0] & 0xF;
            blend.color_op = ToEquation(pass.color_blend_op[0]);
            blend.alpha_op = ToEquation(pass.alpha_blend_op[0]);
            blend.color_source = ToFactor(pass.source_color_blend_factor[0]);
            blend.color_dest = ToFactor(pass.dest_color_blend_factor[0]);
            blend.alpha_source = ToFactor(pass.source_alpha_blend_factor[0]);
            blend.alpha_dest = ToFactor(pass.dest_alpha_blend_factor[0]);
            desc.samples = 1;
            desc.topology_class = topology_class;
            desc.rasterize = 1;
            std::string error;
            out.pipeline = CreateRenderPipelineState(device, desc, error);
            if (!out.pipeline) {
                LOG_WARNING(Render_Metal, "Effect '{}' pass pipeline not built: {}", entry.file,
                            error);
                continue;
            }
            effect.passes.push_back(std::move(out));
        }
        if (effect.passes.empty()) {
            LOG_WARNING(Render_Metal, "Effect '{}' technique '{}' has no passes", entry.file,
                        entry.technique);
            continue;
        }
        LOG_INFO(Render_Metal, "Post-processing effect '{}' ({}): {} passes", entry.file,
                 entry.technique, effect.passes.size());
        effects.push_back(std::move(effect));
    }
    return true;
}

void PostProcess::UpdateUniforms(Effect& effect, std::vector<u8>& out, f32 delta_seconds) {
    static thread_local std::mt19937 rng{std::random_device{}()};
    out.assign((std::max)(effect.uniform_size, 16U), 0);
    const f32 elapsed =
        std::chrono::duration<f32>(std::chrono::steady_clock::now() - start).count();
    const auto overrides = VideoCore::FxChain::Instance().EntryValues(effect.entry_index);
    for (auto& uniform : effect.uniforms) {
        std::array<f32, 4> value = uniform.fallback;
        if (const auto found = overrides.find(uniform.name); found != overrides.end()) {
            value = found->second;
        }
        switch (uniform.source) {
        case UniformSource::FrameTime:
            value[0] = delta_seconds * 1000.0f;
            break;
        case UniformSource::FrameCount:
            value[0] = static_cast<f32>(frame_count);
            break;
        case UniformSource::Timer:
            value[0] = elapsed * 1000.0f;
            break;
        case UniformSource::Random: {
            const int low = static_cast<int>(uniform.args[0]);
            int high = static_cast<int>(uniform.args[1]);
            if (high <= low) {
                high = low + 1;
            }
            std::uniform_int_distribution<int> dist(low, high);
            value[0] = static_cast<f32>(dist(rng));
            break;
        }
        case UniformSource::PingPong: {
            const f32 min_value = uniform.args[0];
            f32 max_value = uniform.args[1];
            if (max_value <= min_value) {
                max_value = min_value + 1.0f;
            }
            f32 step = uniform.args[2];
            if (step == 0.0f) {
                step = 1.0f;
            }
            uniform.state += uniform.direction * step * delta_seconds;
            if (uniform.state >= max_value) {
                uniform.state = max_value;
                uniform.direction = -1.0f;
            }
            if (uniform.state <= min_value) {
                uniform.state = min_value;
                uniform.direction = 1.0f;
            }
            value[0] = uniform.state;
            value[1] = uniform.direction;
            break;
        }
        case UniformSource::Value:
            break;
        }
        for (u32 i = 0; i < uniform.components; ++i) {
            const size_t offset = uniform.offset + i * sizeof(u32);
            if (offset + sizeof(u32) > out.size()) {
                break;
            }
            if (uniform.kind == UniformKind::Floating) {
                std::memcpy(out.data() + offset, &value[i], sizeof(f32));
            } else {
                const s32 element = static_cast<s32>(value[i]);
                std::memcpy(out.data() + offset, &element, sizeof(s32));
            }
        }
    }
}

bool PostProcess::DrawPass(Effect& effect, Pass& pass, TextureHandle backbuffer,
                           TextureHandle target, const std::vector<u8>& uniforms) {
    if (target == nullptr || pass.size[0] == 0 || pass.size[1] == 0) {
        return false;
    }
    if (pass.clear) {
        ClearPass clear{};
        clear.color[0] = Attachment{.texture = target};
        clear.clear_color[0] = true;
        clear.color_value[0] = {0.0, 0.0, 0.0, 0.0};
        clear.width = pass.size[0];
        clear.height = pass.size[1];
        scheduler.Clear(clear);
    }
    RenderTargetSet targets{};
    targets.color[0] = Attachment{.texture = target};
    targets.width = pass.size[0];
    targets.height = pass.size[1];
    scheduler.BeginRendering(targets);
    scheduler.SetDrawState({
        .pipeline = pass.pipeline.Get(),
        .viewport = {0.0f, 0.0f, static_cast<f32>(pass.size[0]), static_cast<f32>(pass.size[1]),
                     0.0f, 1.0f},
        .scissor = {0, 0, pass.size[0], pass.size[1]},
        .cull_mode = 0, // MTLCullModeNone
    });
    const auto bind = [&](const Program& program, ShaderStage stage) {
        for (const ShaderTranslate::ResourceSlot& slot : program.slots) {
            switch (slot.kind) {
            case ShaderTranslate::ResourceKind::Buffer:
                scheduler.BindBytes(stage, slot.index, uniforms.data(),
                                    static_cast<u32>(uniforms.size()));
                break;
            case ShaderTranslate::ResourceKind::Texture:
            case ShaderTranslate::ResourceKind::Sampler: {
                const bool is_texture = slot.kind == ShaderTranslate::ResourceKind::Texture;
                TextureHandle texture = fallback.Get();
                void* sampler = fallback_sampler.Get();
                const auto binding = std::ranges::find_if(
                    pass.sampler_bindings, [&](const auto& b) { return b.first == slot.binding; });
                if (binding != pass.sampler_bindings.end() &&
                    binding->second < effect.samplers.size()) {
                    const Sampler& entry = effect.samplers[binding->second];
                    if (entry.sampler) {
                        sampler = entry.sampler.Get();
                    }
                    if (entry.texture_index != NO_TEXTURE) {
                        const Texture& source = effect.textures[entry.texture_index];
                        if (source.is_backbuffer) {
                            texture = backbuffer;
                        } else if (source.texture) {
                            texture = source.texture.Get();
                        }
                    }
                }
                if (is_texture) {
                    scheduler.BindTexture(stage, slot.index, texture);
                } else {
                    scheduler.BindSampler(stage, slot.index, sampler);
                }
                break;
            }
            }
        }
    };
    bind(pass.vertex, ShaderStage::Vertex);
    bind(pass.fragment, ShaderStage::Fragment);
    scheduler.Draw(pass.topology, 0, pass.vertices, 1, 0);
    scheduler.EndRendering();
    ++passes;
    return true;
}

TextureHandle PostProcess::Apply(TextureHandle source, std::array<u32, 2> size) {
    if (!Wanted() || source == nullptr || size[0] == 0 || size[1] == 0) {
        return source;
    }
    const u64 chain_generation = VideoCore::FxChain::Instance().Snapshot().generation;
    if (!built || generation != chain_generation || extent != size) {
        extent = size;
        built = true;
        frames = {};
        fallback = {};
        effects.clear();
        for (Object& frame : frames) {
            frame = CreateTexture(device, {.kind = TextureKind::e2D,
                                           .format = BACKBUFFER_FORMAT,
                                           .width = size[0],
                                           .height = size[1],
                                           .render_target = true});
        }
        fallback = CreateTexture(device, {.kind = TextureKind::e2D,
                                          .format = Fmt::RGBA8Unorm,
                                          .width = 1,
                                          .height = 1,
                                          .render_target = true});
        fallback_sampler = CreateLinearClampSampler(device);
        if (!frames[0] || !frames[1] || !fallback || !fallback_sampler) {
            effects.clear();
            frames = {};
            return source;
        }
        // Targets start black, as Vulkan clears them once.
        const auto clear_texture = [&](TextureHandle texture, std::array<u32, 2> texture_size) {
            ClearPass clear{};
            clear.color[0] = Attachment{.texture = texture};
            clear.clear_color[0] = true;
            clear.color_value[0] = {0.0, 0.0, 0.0, 0.0};
            clear.width = texture_size[0];
            clear.height = texture_size[1];
            scheduler.Clear(clear);
        };
        clear_texture(fallback.Get(), {1, 1});
        for (const Object& frame : frames) {
            clear_texture(frame.Get(), size);
        }
        BuildEffects();
        for (Effect& effect : effects) {
            for (Texture& texture : effect.textures) {
                if (texture.texture) {
                    clear_texture(texture.texture.Get(), texture.size);
                }
            }
        }
    }
    if (effects.empty()) {
        return source;
    }
    const auto now = std::chrono::steady_clock::now();
    const f32 delta_seconds = std::chrono::duration<f32>(now - previous).count();
    previous = now;
    ++frame_count;

    TextureHandle current = source;
    u32 slot = 0;
    std::vector<u8> uniforms;
    for (Effect& effect : effects) {
        UpdateUniforms(effect, uniforms, delta_seconds);
        for (Pass& pass : effect.passes) {
            TextureHandle target;
            if (pass.writes_backbuffer) {
                target = frames[(slot + 1) % 2].Get();
            } else {
                target = effect.textures[pass.target_texture].texture.Get();
            }
            if (!DrawPass(effect, pass, current, target, uniforms)) {
                continue;
            }
            if (pass.writes_backbuffer) {
                slot = (slot + 1) % 2;
                current = frames[slot].Get();
            }
        }
    }
    return current;
}

} // namespace Metal

#endif // HAS_RESHADE
