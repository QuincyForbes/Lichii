// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <utility>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/shader_environment.h"

namespace Metal {

using VideoCommon::ComputeEnvironment;

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

} // Anonymous namespace

std::string_view PipelineCache::NoteRefusal(u64 key_hash) {
    const auto [it, is_new] = refusal_causes.try_emplace(key_hash);
    if (is_new) {
        const std::string_view cause = translator.Cause();
        it->second = cause.empty() ? "pipeline not made" : cause;
    }
    return it->second;
}

GraphicsPipeline::GraphicsPipeline(const GraphicsPipelineKey& key_, StageProgram vertex_,
                                   StageProgram fragment_, std::optional<MeshEmulation> mesh_,
                                   std::optional<Tessellation> tessellation_)
    : key{key_}, vertex{std::move(vertex_)}, fragment{std::move(fragment_)},
      mesh{std::move(mesh_)}, tessellation{std::move(tessellation_)} {}

void* GraphicsPipeline::VertexKernelState(const Device& device, size_t variant,
                                          const RenderPipelineDesc& layout) {
    if (!tessellation || variant >= Tessellation::IndexSizes.size()) {
        return nullptr;
    }
    StageProgram& program = tessellation->vertex[variant];
    if (program.build_failed) {
        return nullptr;
    }
    std::string error;
    if (!program.function) {
        program.library = CreateLibrary(device, program.msl, error);
        program.function = program.library
                               ? CreateFunction(program.library.Get(), program.entry_point)
                               : Object{};
        if (!program.function) {
            program.build_failed = true;
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: vertex kernel of {:016x} failed to build: {}",
                          key.unique_hashes[1], error);
            }
            return nullptr;
        }
    }
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(layout.attributes.data()),
                                        sizeof(layout.attributes)) ^
                     Common::CityHash64(reinterpret_cast<const char*>(layout.streams.data()),
                                        sizeof(layout.streams)) ^
                     (u64{variant} << 60);
    const auto [it, is_new] = vertex_kernel_states.try_emplace(hash);
    if (is_new) {
        it->second = CreateStageInComputePipelineState(
            device, program.function.Get(), layout, Tessellation::IndexSizes[variant],
            program.tessellation_slots.index, error, archive);
        if (!it->second) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: vertex kernel state of {:016x} refused: {}; {}",
                          key.unique_hashes[1], error, Describe(layout));
            }
        }
    }
    return it->second.Get();
}

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
        // The boot trace: what the first pipelines ask of Metal, written before Metal sees it.
        static std::atomic<u32> traced{};
        if (BootTraceEnabled() && traced.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_INFO(Render_Metal, "Metal: render pipeline {:016x} fragment {:016x}: {}",
                     key.unique_hashes[1], key.unique_hashes[5], Describe(desc));
        }
        std::string error;
        it->second = CreateRenderPipelineState(device, desc, error, archive);
        if (!it->second) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal,
                            "Metal: render pipeline {:016x} refused: {}; fragment {:016x}, {}",
                            key.unique_hashes[1], error, key.unique_hashes[5], Describe(desc));
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
    : VideoCommon::ShaderCache{device_memory}, device{device_}, translator{TranslateForIos} {}

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
        refusal = "shader programs not in guest memory";
        return nullptr;
    }
    PipelineTranslator::RefreshKey(graphics_key, maxwell3d->regs,
                                   maxwell3d->draw_manager.draw_state.topology);
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
    if (current_pipeline == nullptr) {
        refusal = NoteRefusal(graphics_key.Hash());
    }
    return current_pipeline;
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {
    const VideoCommon::ShaderInfo* const shader = ComputeShader();
    if (shader == nullptr) {
        refusal = "shader programs not in guest memory";
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
    if (!it->second) {
        refusal = NoteRefusal(key.Hash());
    }
    return it->second.get();
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    std::optional<TranslatedPipeline> translated =
        translator.Graphics(graphics_key, environments.Span());
    if (!translated) {
        return nullptr;
    }
    return std::make_unique<GraphicsPipeline>(
        graphics_key, std::move(translated->vertex), std::move(translated->fragment),
        std::move(translated->mesh), std::move(translated->tessellation));
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineKey& key, const VideoCommon::ShaderInfo* shader) {
    const GPUVAddr program_base = kepler_compute->regs.code_loc.Address();
    const auto& qmd = kepler_compute->launch_description;
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    std::optional<StageProgram> program = translator.Compute(key.unique_hash, env);
    if (!program) {
        return nullptr;
    }
    return std::make_unique<ComputePipeline>(key, std::move(*program));
}

} // namespace Metal
