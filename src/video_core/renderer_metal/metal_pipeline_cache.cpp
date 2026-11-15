// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
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
                                   StageProgram fragment_, std::optional<MeshEmulation> mesh_)
    : key{key_}, vertex{std::move(vertex_)}, fragment{std::move(fragment_)},
      mesh{std::move(mesh_)} {}

void* GraphicsPipeline::KernelState(const Device& device, MeshKernel& kernel) {
    if (kernel.state || kernel.failed) {
        return kernel.state.Get();
    }
    StageProgram& program = kernel.program;
    std::string error;
    program.library = CreateLibrary(device, program.msl, error);
    program.function =
        program.library ? CreateFunction(program.library.Get(), program.entry_point) : Object{};
    if (program.function) {
        kernel.state = CreateComputePipelineState(device, program.function.Get(), error, archive);
    }
    if (!kernel.state) {
        kernel.failed = true;
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Metal, "Metal: mesh emulation kernel of {:016x} failed to build: {}",
                      key.unique_hashes[1], error);
        }
    }
    return kernel.state.Get();
}

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

void* GraphicsPipeline::RenderState(const Device& device, RenderPipelineDesc& desc) {
    desc.vertex_function = vertex.function.Get();
    desc.fragment_function = fragment.function.Get();
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(&desc), sizeof(desc));
    const auto [it, is_new] = render_states.try_emplace(hash);
    if (is_new) {
        std::string error;
        it->second = CreateRenderPipelineState(device, desc, error, archive);
        if (!it->second) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: render pipeline {:016x} refused: {}",
                          key.unique_hashes[1], error);
            }
        }
    }
    return it->second.Get();
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
        state = CreateComputePipelineState(device, program.function.Get(), error, archive);
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

void PipelineCache::OpenArchive(u64 title_id) {
    if (title_id == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    archive.Open(device, shader_dir / fmt::format("{:016x}", title_id) / "metal_pipelines.bin");
}

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
    graphics_key.attribute_types = Vulkan::FixedPipelineState::DynamicAttributeTypes(regs);
    if (current_pipeline != nullptr && current_pipeline->Key() == graphics_key) {
        return current_pipeline;
    }
    const auto [it, is_new] = graphics_cache.try_emplace(graphics_key);
    if (is_new) {
        it->second = CreateGraphicsPipeline();
        it->second ? ++graphics_made : ++refused;
        if (it->second) {
            it->second->SetArchive(&archive);
        }
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
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, key.unique_hashes);
    pools.ReleaseContents();

    const bool uses_vertex_a = key.unique_hashes[0] != 0;
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    size_t env_index = 0;
    bool task = false;
    size_t mesh_index = 0;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env = *environments.Span()[env_index++];
        const Shader::Stage stage = env.ShaderStage();
        if (stage == Shader::Stage::Task && index == 1) {
            task = true;
        } else if (stage == Shader::Stage::Mesh && (index == 1 || task) && mesh_index == 0) {
            mesh_index = index;
        } else if (stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) {
            LogRefusal(unsupported_logged, hash, "task or mesh program out of place");
            return nullptr;
        } else if (index >= 2 && index <= 4) {
            // Metal draws with a vertex and a fragment stage; tessellation and geometry need
            // emulation.
            LogRefusal(unsupported_logged, hash, "tessellation or geometry stage");
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
    if (task && mesh_index == 0) {
        LogRefusal(unsupported_logged, hash, "task program without a mesh program");
        return nullptr;
    }
    if (mesh_index != 0) {
        return CreateMeshPipeline(programs, mesh_index);
    }

    Shader::Backend::Bindings bindings;
    Shader::RuntimeInfo vertex_info;
    vertex_info.previous_stage_stores.mask.set();
    vertex_info.input_topology = InputTopology(static_cast<Maxwell::PrimitiveTopology>(key.topology));
    for (size_t attribute = 0; attribute < vertex_info.generic_input_types.size(); ++attribute) {
        constexpr std::array types{Shader::AttributeType::Disabled, Shader::AttributeType::Float,
                                   Shader::AttributeType::SignedInt,
                                   Shader::AttributeType::UnsignedInt};
        vertex_info.generic_input_types[attribute] = types[(key.attribute_types >> (attribute * 2)) & 3];
    }
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

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateMeshPipeline(
    std::span<Shader::IR::Program> programs, size_t mesh_index) {
    const GraphicsPipelineKey& key = graphics_key;
    const u64 hash = key.Hash();
    static std::atomic<u32> refused_logged{};
    const bool has_task = mesh_index != 1;
    Shader::IR::Program& mesh = programs[mesh_index];
    // As Vulkan's emulation takes them: points, lines or triangles, before a pixel program only.
    const u32 topology = mesh.mesh.topology;
    if ((topology != 0 && topology != 1 && topology != 4) ||
        !Shader::Backend::SPIRV::MeshEmulationTakesPrimitiveLines(mesh.mesh) ||
        mesh.mesh.vertex_lines.empty() || mesh.mesh.max_vertices == 0 ||
        mesh.mesh.max_primitives == 0) {
        LogRefusal(refused_logged, hash, "mesh program the emulation can't take");
        return nullptr;
    }
    // The output buffer's binding follows every program's (unified bindings).
    u32 output_binding = 0;
    for (size_t index = 1; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        const Shader::Info& info = programs[index].info;
        output_binding += Shader::NumDescriptors(info.constant_buffer_descriptors) +
                          Shader::NumDescriptors(info.storage_buffers_descriptors) +
                          Shader::NumDescriptors(info.texture_buffer_descriptors) +
                          Shader::NumDescriptors(info.image_buffer_descriptors) +
                          Shader::NumDescriptors(info.texture_descriptors) +
                          Shader::NumDescriptors(info.image_descriptors);
    }
    const u32 payload_stride = has_task ? (std::max)(programs[1].mesh.payload_words, 2U) : 0U;
    Shader::Backend::Bindings bindings;
    Shader::RuntimeInfo kernel_info;
    kernel_info.previous_stage_stores.mask.set();
    kernel_info.mesh_as_compute = true;
    kernel_info.mesh_output_binding = output_binding;
    if (has_task) {
        kernel_info.mesh_task_payload_binding = output_binding + 1;
        kernel_info.mesh_task_offsets_binding = output_binding + 2;
        kernel_info.mesh_task_args_binding = output_binding + 3;
        kernel_info.mesh_task_payload_stride = payload_stride;
    }
    MeshEmulation emulation{
        .has_task = has_task,
        .mesh_stage = static_cast<u32>(mesh_index - 1),
        .payload_stride = payload_stride,
        .output_binding = output_binding,
        .block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh.mesh),
        .vertices_per_group = mesh.mesh.max_primitives * mesh.mesh.PrimitiveVertices(),
        .topology = topology,
    };
    // Task, then mesh, then pixel program: the bindings count up in that order.
    const std::array<size_t, 2> kernel_programs{1, mesh_index};
    for (size_t kernel = 0; kernel < (has_task ? 2U : 1U); ++kernel) {
        const size_t index = kernel_programs[kernel];
        auto translated =
            Translate(programs[index], ShaderTranslate::Stage::Compute, bindings, kernel_info);
        if (!translated) {
            return nullptr;
        }
        MeshKernel& target = kernel == 0 ? emulation.first : emulation.mesh;
        target.program = std::move(*translated);
        target.workgroup_size = programs[index].workgroup_size;
    }
    if (has_task) {
        ShaderTranslate::Result distribution = ShaderTranslate::SpirvToMsl(
            Shader::Backend::SPIRV::EmitMeshEmulationDistribution(profile),
            ShaderTranslate::Stage::Compute, translate_options);
        if (distribution.source.empty()) {
            LogRefusal(refused_logged, hash, distribution.error);
            return nullptr;
        }
        emulation.distribution.program = StageProgram{
            .stage = ShaderTranslate::Stage::Compute,
            .msl = std::move(distribution.source),
            .entry_point = std::move(distribution.entry_point),
            .slots = std::move(distribution.slots),
        };
        emulation.distribution.workgroup_size = {1, 1, 1};
    }
    const std::vector<u32> vertex_spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
        profile, mesh.mesh, output_binding, has_task ? output_binding + 3 : ~0U, false);
    ShaderTranslate::Result vertex =
        ShaderTranslate::SpirvToMsl(vertex_spirv, ShaderTranslate::Stage::Vertex, translate_options);
    if (vertex.source.empty()) {
        LogRefusal(refused_logged, hash, vertex.error);
        return nullptr;
    }
    StageProgram fragment{};
    if (key.unique_hashes[5] != 0) {
        Shader::RuntimeInfo fragment_info;
        fragment_info.previous_stage_stores = mesh.info.stores;
        fragment_info.previous_stage_legacy_stores_mapping = mesh.info.legacy_stores_mapping;
        for (const u32 line : mesh.mesh.primitive_lines) {
            const auto attribute = static_cast<Shader::IR::Attribute>(line);
            if (Shader::IR::IsGeneric(attribute)) {
                fragment_info.per_primitive_generics.set(
                    Shader::IR::GenericAttributeIndex(attribute));
            }
        }
        fragment_info.mesh_emulation_inputs = true;
        fragment_info.force_early_z = key.early_z != 0;
        auto translated =
            Translate(programs[5], ShaderTranslate::Stage::Fragment, bindings, fragment_info);
        if (!translated) {
            return nullptr;
        }
        fragment = std::move(*translated);
    }
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
        LOG_INFO(Render_Metal,
                 "Metal: mesh pipeline {:016x} (mesh {:016x}, task {}) emulated: {} vertices x {} "
                 "lines, {} primitives per workgroup, {} output words",
                 hash, key.unique_hashes[mesh_index], has_task, mesh.mesh.max_vertices,
                 mesh.mesh.vertex_lines.size(), mesh.mesh.max_primitives, emulation.block_words);
    }
    StageProgram vertex_program{
        .stage = ShaderTranslate::Stage::Vertex,
        .msl = std::move(vertex.source),
        .entry_point = std::move(vertex.entry_point),
        .slots = std::move(vertex.slots),
    };
    return std::make_unique<GraphicsPipeline>(key, std::move(vertex_program), std::move(fragment),
                                              std::move(emulation));
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
        if (it->second) {
            it->second->SetArchive(&archive);
        }
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
