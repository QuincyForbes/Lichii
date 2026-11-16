// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>

#include <boost/container/static_vector.hpp>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "common/cityhash.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "common/thread_worker.h"
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

constexpr size_t MaxPrograms = 6;

/// The MSL of a built function is not read again (EDEN_NX2_METAL_PIPELINE_FILE=0 keeps it).
void ReleaseSource(StageProgram& program) {
    if (PipelineFiles::Mode() != 0) {
        std::string{}.swap(program.msl);
    }
}

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
                                   std::optional<Tessellation> tessellation_,
                                   std::optional<NativeMesh> native_mesh_)
    : key{key_}, vertex{std::move(vertex_)}, fragment{std::move(fragment_)},
      mesh{std::move(mesh_)}, tessellation{std::move(tessellation_)},
      native_mesh{std::move(native_mesh_)} {
    if (!fragment.entry_point.empty()) {
        for (size_t index = 0; index < fragment.info.interpolation.size(); ++index) {
            flat_inputs = flat_inputs ||
                          (fragment.info.loads.Generic(index) &&
                           fragment.info.interpolation[index] == Shader::Interpolation::Flat);
        }
    }
}

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
        ReleaseSource(program);
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
        ReleaseSource(program);
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

void GraphicsPipeline::FailNative(std::string_view reason) {
    if (!native_mesh || native_mesh->failed) {
        return;
    }
    native_mesh->failed = true;
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
        // Warning: the iOS log shows Render.Metal from there.
        LOG_WARNING(Render_Metal, "Metal: mesh pipeline {:016x} not drawn natively: {}; {}",
                    key.Hash(), reason,
                    mesh ? "drawn through the emulation" : "the emulation can't take it either");
    }
}

bool GraphicsPipeline::BuildNativeFunctions(const Device& device) {
    for (StageProgram* const stage :
         {&native_mesh->object, &native_mesh->mesh, &native_mesh->fragment}) {
        if (stage->function || stage->entry_point.empty()) {
            continue;
        }
        std::string error;
        stage->library = CreateLibrary(device, stage->msl, error);
        stage->function = stage->library ? CreateFunction(stage->library.Get(), stage->entry_point)
                                         : Object{};
        if (!stage->function) {
            FailNative(fmt::format("{} function not built: {}",
                                   stage == &native_mesh->object ? "object"
                                   : stage == &native_mesh->mesh ? "mesh"
                                                                 : "fragment",
                                   error));
            return false;
        }
        ReleaseSource(*stage);
    }
    return true;
}

bool GraphicsPipeline::BuildFunctions(const Device& device) {
    if (Native() != nullptr && BuildNativeFunctions(device)) {
        return true;
    }
    if (native_mesh && !mesh) {
        // Refused natively and not emulated.
        return false;
    }
    for (StageProgram* const stage : {&vertex, &fragment}) {
        if (stage->function || stage->entry_point.empty()) {
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
        ReleaseSource(*stage);
    }
    return true;
}

void* GraphicsPipeline::RenderState(const Device& device, RenderPipelineDesc& desc) {
    NativeMesh* const native = Native();
    if (native != nullptr) {
        desc.vertex_function = native->mesh.function.Get();
        desc.fragment_function = native->fragment.function.Get();
    } else {
        desc.vertex_function = vertex.function.Get();
        desc.fragment_function = fragment.function.Get();
    }
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(&desc), sizeof(desc));
    const auto [it, is_new] = render_states.try_emplace(hash);
    if (is_new && native != nullptr) {
        std::string error;
        it->second = CreateMeshRenderPipelineState(
            device, desc,
            MeshFunctions{
                .object_function = native->object.function.Get(),
                .mesh_function = native->mesh.function.Get(),
                .fragment_function = native->fragment.function.Get(),
                .payload_bytes = native->payload_bytes,
                .object_threads = native->object_threads,
                .mesh_threads = native->mesh_threads,
            },
            error, archive);
        if (!it->second) {
            render_states.erase(it);
            FailNative(fmt::format("mesh render pipeline refused: {}; {}", error, Describe(desc)));
            return nullptr;
        }
        if (recorder != nullptr) {
            recorder->RecordState(key, desc);
        }
    } else if (is_new) {
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
        } else if (recorder != nullptr) {
            // Only states Metal made are kept: boot never asks for one it has not seen made.
            recorder->RecordState(key, desc);
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
        ReleaseSource(program);
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
    : VideoCommon::ShaderCache{device_memory}, device{device_},
      translator{TranslateForIos, DeviceAppleFamily(device_), DeviceSupportsMeshShaders(device_)} {
    // A suspended app may be killed without notice (iOS): the archive is written when it leaves
    // the foreground, and the used list at the GPU thread's next frame.
    background_listener = Common::PlatformEvents::AddBackgroundListener([this] {
        save_requested.store(true, std::memory_order_relaxed);
        for (int wait = 0; wait < 25 && save_requested.load(std::memory_order_relaxed); ++wait) {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
        }
        files.Wait();
        archive.Save();
    });
}

PipelineCache::~PipelineCache() {
    Common::PlatformEvents::RemoveBackgroundListener(background_listener);
    files.SaveUsed();
}

void PipelineCache::OpenArchive(u64 title_id) {
    if (title_id == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    archive.Open(device, shader_dir / fmt::format("{:016x}", title_id) / "metal_pipelines.bin");
}

void PipelineCache::TickFrame() {
    // 0: with the archive, once a minute.
    static const std::chrono::seconds interval = [] {
        const char* const value = std::getenv("EDEN_NX2_USED_PIPELINES_SAVE_S");
        const long long seconds = value != nullptr ? (std::max)(0LL, std::atoll(value)) : 10LL;
        return std::chrono::seconds{seconds != 0 ? seconds : 60LL};
    }();
    const auto now = std::chrono::steady_clock::now();
    const bool requested = save_requested.load(std::memory_order_relaxed);
    if (requested || (files.Dirty() && now - used_saved >= interval)) {
        files.SaveUsed();
        used_saved = now;
        save_requested.store(false, std::memory_order_relaxed);
    }
    if (now - archive_saved > std::chrono::minutes{1}) {
        archive_saved = now;
        archive.Save();
    }
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    OpenArchive(title_id);
    if (title_id == 0 || PipelineFiles::Mode() == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    files.Open(shader_dir / fmt::format("{:016x}", title_id));
    if (!files.IsOpen()) {
        return;
    }
    // A boot build that never finished (Metal killed the process in it) is not tried again at
    // once: this run reads the records only, the next one builds again.
    const bool build = files.BeginBoot();
    if (!build) {
        LOG_WARNING(Render_Metal, "Metal: the last boot build of pipelines did not finish; "
                                  "nothing is built ahead in this run");
    }
    std::unordered_map<GraphicsPipelineKey, std::vector<RenderPipelineDesc>> states;
    if (build) {
        for (const PipelineFiles::StateRecord& record : files.LoadStates()) {
            states[record.key].push_back(record.desc);
        }
    }
    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        size_t graphics{};
        size_t compute{};
        size_t render_states{};
        size_t refused{};
        bool has_loaded{};
    } state;
    size_t unlisted = 0;
    const size_t num_workers = (std::max)(std::thread::hardware_concurrency(), 2U) - 1;
    const u32 apple_family = DeviceAppleFamily(device);
    const bool mesh_shaders = DeviceSupportsMeshShaders(device);
    const auto start = std::chrono::steady_clock::now();
    {
        Common::StatefulThreadWorker<PipelineTranslator> workers{
            build ? num_workers : 0, "MetalPipelines", [apple_family, mesh_shaders] {
                return PipelineTranslator{TranslateForIos, apple_family, mesh_shaders};
            }};
        const auto finish = [&state, &callback] {
            ++state.built;
            if (state.has_loaded && callback) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        };
        const auto load_compute = [&](std::ifstream& file, VideoCommon::FileEnvironment env) {
            ComputePipelineKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            if (!disk_compute.insert(key).second || !build) {
                return;
            }
            {
                std::scoped_lock lock{state.mutex};
                ++state.total;
            }
            workers.QueueWork([this, key, env_ = std::move(env), &state, &finish,
                               &stop_loading](PipelineTranslator* worker) mutable {
                if (stop_loading.stop_requested()) {
                    return;
                }
                std::unique_ptr<ComputePipeline> pipeline;
                InAutoreleasePool([&] {
                    std::optional<StageProgram> program = worker->Compute(key.unique_hash, env_);
                    if (!program) {
                        return;
                    }
                    pipeline = std::make_unique<ComputePipeline>(key, std::move(*program));
                    pipeline->SetArchive(&archive);
                    static_cast<void>(pipeline->State(device));
                });
                std::scoped_lock lock{state.mutex};
                if (pipeline) {
                    compute_cache.insert_or_assign(key, std::move(pipeline));
                    ++state.compute;
                } else {
                    ++state.refused;
                }
                finish();
            });
        };
        const auto load_graphics = [&](std::ifstream& file,
                                       std::vector<VideoCommon::FileEnvironment> envs) {
            GraphicsPipelineKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            if (!disk_graphics.insert(key).second || !build || envs.size() > MaxPrograms) {
                return;
            }
            if (!files.BuildAtBoot(key.Hash())) {
                ++unlisted;
                return;
            }
            const auto found = states.find(key);
            const std::vector<RenderPipelineDesc>* const descs =
                found != states.end() ? &found->second : nullptr;
            {
                std::scoped_lock lock{state.mutex};
                ++state.total;
            }
            workers.QueueWork([this, key, envs_ = std::move(envs), descs, &state, &finish,
                               &stop_loading](PipelineTranslator* worker) mutable {
                if (stop_loading.stop_requested()) {
                    return;
                }
                std::unique_ptr<GraphicsPipeline> pipeline;
                size_t made = 0;
                InAutoreleasePool([&] {
                    boost::container::static_vector<Shader::Environment*, MaxPrograms> env_ptrs;
                    for (VideoCommon::FileEnvironment& env : envs_) {
                        env_ptrs.push_back(&env);
                    }
                    std::optional<TranslatedPipeline> translated =
                        worker->Graphics(key, {env_ptrs.data(), env_ptrs.size()});
                    if (!translated) {
                        return;
                    }
                    pipeline = std::make_unique<GraphicsPipeline>(
                        key, std::move(translated->vertex), std::move(translated->fragment),
                        std::move(translated->mesh), std::move(translated->tessellation),
                        std::move(translated->native_mesh));
                    pipeline->SetArchive(&archive);
                    if (!pipeline->BuildFunctions(device)) {
                        return;
                    }
                    MeshEmulation* const mesh = pipeline->Mesh();
                    if (mesh != nullptr && pipeline->Native() == nullptr) {
                        static_cast<void>(pipeline->KernelState(device, mesh->first));
                        if (mesh->has_task) {
                            static_cast<void>(pipeline->KernelState(device, mesh->mesh));
                            static_cast<void>(pipeline->KernelState(device, mesh->distribution));
                        }
                    }
                    if (Tessellation* const tess = pipeline->Tess()) {
                        static_cast<void>(pipeline->KernelState(device, tess->control));
                    }
                    if (descs == nullptr) {
                        return;
                    }
                    for (RenderPipelineDesc desc : *descs) {
                        made += pipeline->RenderState(device, desc) != nullptr ? 1 : 0;
                    }
                });
                std::scoped_lock lock{state.mutex};
                if (pipeline) {
                    // The states made from here on are new ones: they go to the states file.
                    pipeline->SetRecorder(&files);
                    graphics_cache.insert_or_assign(key, std::move(pipeline));
                    ++state.graphics;
                    state.render_states += made;
                } else {
                    ++state.refused;
                }
                finish();
            });
        };
        VideoCommon::LoadPipelines(stop_loading, files.Records(), PipelineFiles::RecordsVersion,
                                   load_compute, load_graphics);
        {
            std::scoped_lock lock{state.mutex};
            if (callback) {
                callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
            }
            state.has_loaded = true;
        }
        workers.WaitForRequests(stop_loading);
    }
    if (!build) {
        return;
    }
    files.EndBoot();
    const PipelineArchive::Counts counts = archive.Totals();
    LOG_INFO(Render_Metal,
             "Metal: boot pipelines: {} of {} built ({} graphics with {} render states, {} "
             "compute, {} not made) in {:.1f} s on {} threads; {} recorded but not on the used "
             "list; archive {} hits, {} misses",
             state.graphics + state.compute, state.total, state.graphics, state.render_states,
             state.compute, state.refused,
             std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
             num_workers, unlisted, counts.hits, counts.misses);
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
            it->second->SetRecorder(&files);
        }
    }
    current_pipeline = it->second.get();
    if (current_pipeline == nullptr) {
        refusal = NoteRefusal(graphics_key.Hash());
    } else if (!current_pipeline->MarkUsed()) {
        files.NoteUsed(graphics_key.Hash());
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
    } else if (!it->second->MarkUsed()) {
        files.NoteUsed(key.Hash());
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
    if (files.IsOpen() && disk_graphics.insert(graphics_key).second) {
        files.Queue([this, key = graphics_key, envs = environments.envs] {
            boost::container::static_vector<const VideoCommon::GenericEnvironment*, MaxPrograms>
                env_ptrs;
            for (size_t index = 0; index < MaxPrograms; ++index) {
                if (key.unique_hashes[index] != 0) {
                    env_ptrs.push_back(&envs[index]);
                }
            }
            VideoCommon::SerializePipeline(key, env_ptrs, files.Records(),
                                           PipelineFiles::RecordsVersion);
        });
    }
    return std::make_unique<GraphicsPipeline>(
        graphics_key, std::move(translated->vertex), std::move(translated->fragment),
        std::move(translated->mesh), std::move(translated->tessellation),
        std::move(translated->native_mesh));
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
    if (files.IsOpen() && disk_compute.insert(key).second) {
        files.Queue([this, key, stored = std::move(env)] {
            VideoCommon::SerializePipeline(
                key, std::array<const VideoCommon::GenericEnvironment*, 1>{&stored},
                files.Records(), PipelineFiles::RecordsVersion);
        });
    }
    return std::make_unique<ComputePipeline>(key, std::move(*program));
}

} // namespace Metal
