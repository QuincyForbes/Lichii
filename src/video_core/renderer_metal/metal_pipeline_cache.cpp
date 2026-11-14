// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/shader_environment.h"

namespace Metal {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;
using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;

namespace {

[[nodiscard]] Shader::InputTopology InputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::InputTopology::Points;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::InputTopology::Lines;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return Shader::InputTopology::LinesAdjacency;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        return Shader::InputTopology::TrianglesAdjacency;
    default:
        return Shader::InputTopology::Triangles;
    }
}

/// Logs one of the first 16 refusals of a kind.
void LogRefusal(std::atomic<u32>& count, u64 hash, std::string_view reason) {
    if (count.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Metal, "Metal: pipeline {:016x} not made: {}", hash, reason);
    }
}

} // Anonymous namespace

size_t GraphicsPipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

size_t ComputePipelineKey::Hash() const noexcept {
    return static_cast<size_t>(
        Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this));
}

GraphicsPipeline::GraphicsPipeline(const GraphicsPipelineKey& key_, StageProgram vertex_,
                                   StageProgram fragment_)
    : key{key_}, vertex{std::move(vertex_)}, fragment{std::move(fragment_)} {}

bool GraphicsPipeline::BuildFunctions(const Device& device) {
    for (StageProgram* const stage : {&vertex, &fragment}) {
        if (stage->function || stage->msl.empty()) {
            continue;
        }
        if (stage->build_failed) {
            return false;
        }
        std::string error;
        stage->library = CreateLibrary(device, stage->msl, error);
        stage->function = stage->library ? CreateFunction(stage->library.Get(), stage->entry_point)
                                         : Object{};
        if (!stage->function) {
            stage->build_failed = true;
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: {:016x} stage MSL failed to build: {}",
                          key.unique_hashes[stage == &vertex ? 1 : 5], error);
            }
            return false;
        }
    }
    return true;
}

ComputePipeline::ComputePipeline(const ComputePipelineKey& key_, StageProgram program_)
    : key{key_}, program{std::move(program_)} {}

void* ComputePipeline::State(const Device& device) {
    if (state || failed) {
        return state.Get();
    }
    std::string error;
    program.library = CreateLibrary(device, program.msl, error);
    program.function =
        program.library ? CreateFunction(program.library.Get(), program.entry_point) : Object{};
    if (program.function) {
        state = CreateComputePipelineState(device, program.function.Get(), error);
    }
    if (!state) {
        failed = true;
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Metal, "Metal: compute {:016x} failed to build: {}", key.unique_hash,
                      error);
        }
    }
    return state.Get();
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device_)
    : VideoCommon::ShaderCache{device_memory}, device{device_} {
    // What Apple GPUs run through SPIRV-Cross (MoltenVK's defaults for Apple7+).
    profile = Shader::Profile{
        .supported_spirv = 0x00010300,
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = false,
        .support_int8 = true,
        .support_uniform_and_storage_buffer_8bit = true,
        .support_storage_buffer_8bit = true,
        .support_int16 = true,
        .support_uniform_and_storage_buffer_16bit = true,
        .support_storage_buffer_16bit = true,
        .support_int64 = true,
        .support_vertex_instance_id = false,
        .support_float_controls = false,
        .support_vote = true,
        .supported_subgroup_stages = ~0U,
        .support_viewport_index_layer_non_geometry = true,
        .support_viewport_mask = false,
        .support_typeless_image_loads = false,
        .support_demote_to_helper_invocation = true,
        .support_int64_atomics = false,
        .support_float32_atomic_add = true,
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = false,
        .support_native_ndc = false,
        .support_scaled_attributes = false,
        .support_multi_viewport = true,
        .support_geometry_streams = false,
        .warp_size_potentially_larger_than_guest = false,
        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .has_broken_spirv_clamp = false,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .ignore_nan_fp_comparisons = false,
        .max_subgroup_size = 32,
        .min_ssbo_alignment = 16,
        .max_user_clip_distances = 8,
    };
    constexpr u32 Limit = Shader::HostTranslateInfo::DEFAULT_DESCRIPTOR_LIMIT;
    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = 16,
        .max_per_stage_descriptor_sampled_images = Limit,
        .max_per_stage_resources = Limit,
        .max_per_stage_descriptor_storage_buffers = Limit,
        .max_descriptor_set_samplers = Limit,
        .max_descriptor_set_uniform_buffers = Limit,
        .max_descriptor_set_uniform_buffers_dynamic = Limit,
        .max_descriptor_set_storage_buffers = Limit,
        .max_descriptor_set_storage_buffers_dynamic = Limit,
        .max_descriptor_set_sampled_images = Limit,
        .max_descriptor_set_storage_images = Limit,
        .max_descriptor_set_input_attachements = Limit,
        .support_float64 = false,
        .support_float16 = true,
        .support_int64 = true,
        .needs_demote_reorder = false,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = true,
        .support_geometry_shader_passthrough = false,
        .support_conditional_barrier = false,
    };
    host_info.ApplyDescriptorLimitPolicy();
#if defined(__APPLE__) && TARGET_OS_IPHONE
    translate_options.ios = true;
#endif
}

PipelineCache::~PipelineCache() = default;

std::array<u64, 3> PipelineCache::TakeCounts() noexcept {
    return {std::exchange(graphics_made, 0), std::exchange(compute_made, 0),
            std::exchange(refused, 0)};
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    const auto& regs = maxwell3d->regs;
    graphics_key.topology = static_cast<u32>(maxwell3d->draw_manager.draw_state.topology);
    graphics_key.early_z = regs.mandated_early_z != 0 ? 1 : 0;
    if (current_pipeline != nullptr && current_pipeline->Key() == graphics_key) {
        return current_pipeline;
    }
    const auto [it, is_new] = graphics_cache.try_emplace(graphics_key);
    if (is_new) {
        it->second = CreateGraphicsPipeline();
        it->second ? ++graphics_made : ++refused;
    }
    current_pipeline = it->second.get();
    return current_pipeline;
}

std::optional<StageProgram> PipelineCache::Translate(Shader::IR::Program& program,
                                                     ShaderTranslate::Stage stage,
                                                     Shader::Backend::Bindings& bindings,
                                                     const Shader::RuntimeInfo& runtime_info) {
    ConvertLegacyToGeneric(program, runtime_info);
    const std::vector<u32> spirv = EmitSPIRV(profile, runtime_info, program, bindings);
    ShaderTranslate::Result result = ShaderTranslate::SpirvToMsl(spirv, stage, translate_options);
    if (result.source.empty()) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Metal, "Metal: SPIR-V -> MSL failed: {}", result.error);
        }
        return std::nullopt;
    }
    return StageProgram{
        .stage = stage,
        .msl = std::move(result.source),
        .entry_point = std::move(result.entry_point),
        .slots = std::move(result.slots),
        .info = program.info,
    };
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() try {
    const GraphicsPipelineKey& key = graphics_key;
    const u64 hash = key.Hash();
    static std::atomic<u32> unsupported_logged{};
    // Metal draws with a vertex and a fragment stage; tessellation and geometry need emulation.
    for (const size_t index : {2, 3, 4}) {
        if (key.unique_hashes[index] != 0) {
            LogRefusal(unsupported_logged, hash, "tessellation or geometry stage");
            return nullptr;
        }
    }
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, key.unique_hashes);
    pools.ReleaseContents();

    const bool uses_vertex_a = key.unique_hashes[0] != 0;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    size_t env_index = 0;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env = *environments.Span()[env_index++];
        if (env.ShaderStage() == Shader::Stage::Task || env.ShaderStage() == Shader::Stage::Mesh) {
            LogRefusal(unsupported_logged, hash, "task or mesh stage (milestone 4)");
            return nullptr;
        }
        const u32 cfg_offset = static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            auto program_vb = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
            programs[index] = MergeDualVertexPrograms(programs[0], program_vb, env, host_info);
        }
    }
    if (key.unique_hashes[1] == 0) {
        LogRefusal(unsupported_logged, hash, "no vertex program");
        return nullptr;
    }

    Shader::Backend::Bindings bindings;
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    vertex_info.input_topology = InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology));
    auto vertex = Translate(programs[1], ShaderTranslate::Stage::Vertex, bindings, vertex_info);
    if (!vertex) {
        return nullptr;
    }
    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = programs[1].info.stores;
        fragment_info.previous_stage_legacy_stores_mapping =
            programs[1].info.legacy_stores_mapping;
        fragment_info.force_early_z = key.early_z != 0;
        auto translated =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!translated) {
            return nullptr;
        }
        fragment = std::move(*translated);
    }
    return std::make_unique<GraphicsPipeline>(key, std::move(*vertex), std::move(fragment));
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Metal, "Metal: {}", exception.what());
    return nullptr;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {
    const VideoCommon::ShaderInfo* const shader = ComputeShader();
    if (shader == nullptr) {
        return nullptr;
    }
    const auto& qmd = kepler_compute->launch_description;
    const ComputePipelineKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [it, is_new] = compute_cache.try_emplace(key);
    if (is_new) {
        it->second = CreateComputePipeline(key, shader);
        it->second ? ++compute_made : ++refused;
    }
    return it->second.get();
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineKey& key, const VideoCommon::ShaderInfo* shader) try {
    const GPUVAddr program_base = kepler_compute->regs.code_loc.Address();
    const auto& qmd = kepler_compute->launch_description;
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    pools.ReleaseContents();
    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};
    auto program = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
    const std::vector<u32> spirv = EmitSPIRV(profile, program);
    ShaderTranslate::Result result =
        ShaderTranslate::SpirvToMsl(spirv, ShaderTranslate::Stage::Compute, translate_options);
    if (result.source.empty()) {
        static std::atomic<u32> logged{};
        LogRefusal(logged, key.unique_hash, result.error);
        return nullptr;
    }
    return std::make_unique<ComputePipeline>(key, StageProgram{
                                                      .stage = ShaderTranslate::Stage::Compute,
                                                      .msl = std::move(result.source),
                                                      .entry_point = std::move(result.entry_point),
                                                      .slots = std::move(result.slots),
                                                      .info = program.info,
                                                  });
} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Metal, "Metal: {}", exception.what());
    return nullptr;
}

} // namespace Metal
