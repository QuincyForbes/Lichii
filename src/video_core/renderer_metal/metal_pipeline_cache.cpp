// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
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
#include "video_core/async_translate.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/renderer_metal/metal_work_policy.h"
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

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

} // Anonymous namespace

u32 AsyncPipelineThreads() {
    static const u32 threads = [] {
        // Vulkan's switches apply too: EDEN_NX2_ASYNC_TRANSLATE other than 1 translates on the
        // draw's thread; EDEN_NX2_ASYNC_TRANSLATE_THREADS sets the count unless the Metal one does.
        if (VideoCommon::AsyncTranslateMode() != 1) {
            return 0U;
        }
        const u32 automatic = (std::max)(std::thread::hardware_concurrency(), 2U) - 1;
        const char* value = std::getenv("EDEN_NX2_METAL_ASYNC_PIPELINES");
        if (value == nullptr || value[0] == '\0') {
            value = std::getenv("EDEN_NX2_ASYNC_TRANSLATE_THREADS");
        }
        if (value == nullptr || value[0] == '\0') {
            return automatic;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed <= 0 ? 0U : (std::min)(static_cast<u32>(parsed), 16U);
    }();
    return threads;
}

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

std::optional<MeshFunctions> GraphicsPipeline::NativeFunctions() const noexcept {
    if (!native_mesh || native_mesh->failed) {
        return std::nullopt;
    }
    return MeshFunctions{
        .object_function = native_mesh->object.function.Get(),
        .mesh_function = native_mesh->mesh.function.Get(),
        .fragment_function = native_mesh->fragment.function.Get(),
        .payload_bytes = native_mesh->payload_bytes,
        .object_threads = native_mesh->object_threads,
        .mesh_threads = native_mesh->mesh_threads,
    };
}

u64 GraphicsPipeline::StateHash(RenderPipelineDesc& desc) const noexcept {
    if (const std::optional<MeshFunctions> native = NativeFunctions()) {
        desc.vertex_function = native->mesh_function;
        desc.fragment_function = native->fragment_function;
    } else {
        desc.vertex_function = vertex.function.Get();
        desc.fragment_function = fragment.function.Get();
    }
    return Common::CityHash64(reinterpret_cast<const char*>(&desc), sizeof(desc));
}

std::optional<void*> GraphicsPipeline::FindState(u64 hash) const {
    const auto it = render_states.find(hash);
    if (it == render_states.end()) {
        return std::nullopt;
    }
    return it->second.Get();
}

void* GraphicsPipeline::RenderState(const Device& device, RenderPipelineDesc& desc) {
    const u64 hash = StateHash(desc);
    if (const std::optional<void*> found = FindState(hash)) {
        return *found;
    }
    const std::optional<MeshFunctions> native = NativeFunctions();
    std::string error;
    if (native) {
        Object state = CreateMeshRenderPipelineState(device, desc, *native, error, archive);
        return AdoptState(hash, desc, std::move(state), error, true);
    }
    // The boot trace: what the first pipelines ask of Metal, written before Metal sees it.
    static std::atomic<u32> traced{};
    if (BootTraceEnabled() && traced.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_INFO(Render_Metal, "Metal: render pipeline {:016x} fragment {:016x}: {}",
                 key.unique_hashes[1], key.unique_hashes[5], Describe(desc));
    }
    Object state = CreateRenderPipelineState(device, desc, error, archive);
    return AdoptState(hash, desc, std::move(state), error, false);
}

void* GraphicsPipeline::AdoptState(u64 hash, const RenderPipelineDesc& desc, Object state,
                                   std::string_view error, bool mesh_state) {
    if (mesh_state) {
        // A refused mesh state is not kept: the pipeline's draws go through the emulation.
        if (!state) {
            FailNative(fmt::format("mesh render pipeline refused: {}; {}", error, Describe(desc)));
            return nullptr;
        }
        if (Native() == nullptr) {
            // Refused natively while this state was being built.
            return nullptr;
        }
    }
    if (!state) {
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
    Object& slot = render_states[hash];
    slot = std::move(state);
    return slot.Get();
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
    if (const u32 threads = AsyncPipelineThreads(); threads != 0) {
        // Background QoS: a draw that waits for a build runs it itself or lends it its own.
        const u32 apple_family = DeviceAppleFamily(device);
        builders = std::make_unique<Common::StatefulThreadWorker<PipelineTranslator>>(
            threads, "MetalPipelineBuilder",
            [apple_family] { return PipelineTranslator{TranslateForIos, apple_family}; },
            Common::ThreadPlacement::Background);
        LOG_INFO(Render_Metal, "Metal: pipelines first seen in a run are built on {} threads "
                               "(EDEN_NX2_METAL_ASYNC_PIPELINES)",
                 threads);
    }
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
    ++frame_count;
    EvictUnusedPipelines();
}

void PipelineCache::EvictUnusedPipelines() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_PIPELINE_EVICT");
        return value == nullptr || value[0] != '0';
    }();
    if (!enabled) {
        return;
    }
    // As Vulkan on Apple: building a pipeline again is a Metal compile mid-game, so none go
    // unless the OS reports memory pressure, ahead of a jetsam kill.
    const u32 pressure = Common::PlatformEvents::MemoryPressureGeneration();
    if (pressure == evict_pressure_generation) {
        return;
    }
    evict_pressure_generation = pressure;
    // Vulkan's EDEN_NX2_PIPELINE_EVICT_LEFT_MIB (the iOS device profile sets it) and _MAX, read
    // under EDEN_NX2_METAL_VULKAN_SWITCHES: only near the limit, fewer, older ones.
    static const u64 left_limit = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_PIPELINE_EVICT_LEFT_MIB");
        return value != nullptr ? u64{std::strtoull(value, nullptr, 10)} << 20 : u64{0};
    }();
    static const size_t evict_max = [] {
        const char* const value = VulkanSwitch("EDEN_NX2_PIPELINE_EVICT_MAX");
        return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10))
                                : size_t{256};
    }();
    bool critical = false;
    if (left_limit != 0) {
        const u64 left = Common::PlatformEvents::AvailableProcessMemory();
        if (left != 0 && left >= left_limit) {
            ++pressure_evictions_skipped;
            return;
        }
        critical = true;
    }
    std::vector<WorkPolicy::PipelineUse> uses;
    std::vector<GraphicsPipelineKey> keys;
    uses.reserve(graphics_cache.size());
    keys.reserve(graphics_cache.size());
    // A pipeline whose render state is still building stays: the build is kept by its address.
    std::unordered_set<const GraphicsPipeline*> building_states;
    for (const auto& [state_key, build] : state_builds) {
        building_states.insert(state_key.first);
    }
    for (const auto& [key, pipeline] : graphics_cache) {
        if (pipeline == nullptr) {
            continue;
        }
        uses.push_back({.last_use = pipeline->LastUse(),
                        .used = pipeline->WasUsed(),
                        .current = pipeline.get() == current_pipeline ||
                                   building_states.contains(pipeline.get())});
        keys.push_back(key);
    }
    const std::vector<size_t> evicted =
        WorkPolicy::PressureEvictions(uses, frame_count, critical, evict_max);
    if (evicted.empty()) {
        return;
    }
    for (const size_t index : evicted) {
        graphics_cache.erase(keys[index]);
    }
    LOG_INFO(Render_Metal,
             "Metal: evicted {} graphics pipelines on memory pressure, {} left{}", evicted.size(),
             graphics_cache.size(),
             critical ? fmt::format(" ({} MiB left to the process; {} pressure signals before "
                                    "passed with more than {} MiB left)",
                                    Common::PlatformEvents::AvailableProcessMemory() >> 20,
                                    pressure_evictions_skipped, left_limit >> 20)
                      : std::string{});
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
    building = false;
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        refusal = "shader programs not in guest memory";
        return nullptr;
    }
    PipelineTranslator::RefreshKey(graphics_key, maxwell3d->regs, DrawTopology(*maxwell3d));
    if (current_pipeline != nullptr && current_pipeline->Key() == graphics_key) {
        current_pipeline->SetLastUse(frame_count);
        return current_pipeline;
    }
    auto it = graphics_cache.find(graphics_key);
    if (it == graphics_cache.end()) {
        std::unique_ptr<GraphicsPipeline> made;
        if (builders) {
            bool pending = false;
            made = BuildInBackground(pending);
            if (pending) {
                current_pipeline = nullptr;
                building = true;
                refusal = "pipeline still building";
                return nullptr;
            }
        } else {
            made = CreateGraphicsPipeline();
        }
        made ? ++graphics_made : ++refused;
        if (made) {
            made->SetArchive(&archive);
            made->SetRecorder(&files);
        }
        it = graphics_cache.emplace(graphics_key, std::move(made)).first;
    }
    current_pipeline = it->second.get();
    if (current_pipeline == nullptr) {
        refusal = NoteRefusal(graphics_key.Hash());
    } else {
        current_pipeline->SetLastUse(frame_count);
        if (!current_pipeline->MarkUsed()) {
            files.NoteUsed(graphics_key.Hash());
        }
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

void PipelineCache::StoreRecord(const GraphicsPipelineKey& key,
                                const std::array<VideoCommon::GraphicsEnvironment, 6>& envs) {
    if (!files.IsOpen() || !disk_graphics.insert(key).second) {
        return;
    }
    files.Queue([this, key, envs] {
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

void PipelineCache::Hold(PipelineBuildJob& job) {
    const auto begin = std::chrono::steady_clock::now();
    bool boosted = false;
    const bool here = job.Wait(translator, boosted);
    const u64 waited = NanosecondsSince(begin);
    ++build_stats.held;
    build_stats.held_here += here ? 1 : 0;
    build_stats.boosted += boosted ? 1 : 0;
    build_stats.held_ns += waited;
    build_stats.held_max_ns = (std::max)(build_stats.held_max_ns, waited);
}

std::shared_ptr<PipelineCache::GraphicsBuild> PipelineCache::QueueGraphicsBuild() {
    auto build = std::make_shared<GraphicsBuild>();
    build->key = graphics_key;
    build->queued = std::chrono::steady_clock::now();
    {
        // Snapshots: what the translation reads of the engine and the constant buffers is
        // copied now, as EDEN_NX2_ASYNC_TRANSLATE does on Vulkan.
        GraphicsEnvironments environments;
        GetGraphicsEnvironments(environments, graphics_key.unique_hashes, true);
        build->envs = environments.envs;
    }
    GraphicsBuild* const raw = build.get();
    build->job = std::make_shared<PipelineBuildJob>([this, raw](PipelineTranslator& worker) {
        InAutoreleasePool([&] {
            boost::container::static_vector<Shader::Environment*, MaxPrograms> env_ptrs;
            for (size_t index = 0; index < MaxPrograms; ++index) {
                if (raw->key.unique_hashes[index] != 0) {
                    env_ptrs.push_back(&raw->envs[index]);
                }
            }
            std::optional<TranslatedPipeline> translated =
                worker.Graphics(raw->key, {env_ptrs.data(), env_ptrs.size()});
            if (!translated) {
                raw->cause = worker.Cause();
                return;
            }
            auto pipeline = std::make_unique<GraphicsPipeline>(
                raw->key, std::move(translated->vertex), std::move(translated->fragment),
                std::move(translated->mesh), std::move(translated->tessellation),
                std::move(translated->native_mesh));
            pipeline->SetArchive(&archive);
            if (pipeline->BuildFunctions(device)) {
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
            }
            raw->pipeline = std::move(pipeline);
        });
    });
    builders->QueueWork([build](PipelineTranslator* worker) { build->job->Run(*worker); });
    graphics_builds.insert_or_assign(graphics_key, build);
    ++build_stats.queued;
    return build;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::AdoptBuild(GraphicsBuild& build, bool& retry) {
    retry = false;
    for (size_t index = 0; index < MaxPrograms; ++index) {
        // A texture descriptor the translation read changed since the snapshot.
        if (build.key.unique_hashes[index] != 0 && !build.envs[index].TextureReadsStillMatch()) {
            retry = true;
            return nullptr;
        }
    }
    ++build_stats.adopted;
    const u64 ready = NanosecondsSince(build.queued);
    build_stats.ready_ns += ready;
    build_stats.ready_max_ns = (std::max)(build_stats.ready_max_ns, ready);
    if (!build.pipeline) {
        refusal_causes.try_emplace(build.key.Hash(),
                                   build.cause.empty() ? "pipeline not made" : build.cause);
        return nullptr;
    }
    StoreRecord(build.key, build.envs);
    return std::move(build.pipeline);
}

std::unique_ptr<GraphicsPipeline> PipelineCache::BuildInBackground(bool& pending) {
    pending = false;
    std::shared_ptr<GraphicsBuild> build;
    if (const auto found = graphics_builds.find(graphics_key); found != graphics_builds.end()) {
        build = found->second;
    } else {
        build = QueueGraphicsBuild();
    }
    if (!build->job->IsDone()) {
        if (MayBeSkipped()) {
            ++build_stats.skipped;
            pending = true;
            return nullptr;
        }
        Hold(*build->job);
    }
    graphics_builds.erase(graphics_key);
    bool retry = false;
    std::unique_ptr<GraphicsPipeline> pipeline = AdoptBuild(*build, retry);
    if (!retry) {
        return pipeline;
    }
    ++build_stats.stale;
    if (MayBeSkipped()) {
        // Again from what the guest holds now.
        static_cast<void>(QueueGraphicsBuild());
        ++build_stats.skipped;
        pending = true;
        return nullptr;
    }
    return CreateGraphicsPipeline();
}

void* PipelineCache::RenderState(GraphicsPipeline& pipeline, RenderPipelineDesc& desc,
                                 bool& building_state) {
    building_state = false;
    if (!builders) {
        return pipeline.RenderState(device, desc);
    }
    const u64 hash = pipeline.StateHash(desc);
    if (const std::optional<void*> found = pipeline.FindState(hash)) {
        return *found;
    }
    const std::pair<const GraphicsPipeline*, u64> state_key{&pipeline, hash};
    auto it = state_builds.find(state_key);
    if (it == state_builds.end()) {
        if (!MayBeSkipped()) {
            // Nothing queued and the draw waits: built here as before.
            return pipeline.RenderState(device, desc);
        }
        auto build = std::make_shared<StateBuild>();
        build->desc = desc;
        build->native = pipeline.NativeFunctions();
        StateBuild* const raw = build.get();
        build->job = std::make_shared<PipelineBuildJob>([this, raw](PipelineTranslator&) {
            InAutoreleasePool([&] {
                raw->state = raw->native ? CreateMeshRenderPipelineState(device, raw->desc,
                                                                         *raw->native, raw->error,
                                                                         &archive)
                                         : CreateRenderPipelineState(device, raw->desc,
                                                                     raw->error, &archive);
            });
        });
        builders->QueueWork([build](PipelineTranslator* worker) { build->job->Run(*worker); });
        it = state_builds.emplace(state_key, std::move(build)).first;
        ++build_stats.states_queued;
    }
    if (!it->second->job->IsDone()) {
        if (MayBeSkipped()) {
            ++build_stats.skipped;
            building_state = true;
            return nullptr;
        }
        Hold(*it->second->job);
    }
    const std::shared_ptr<StateBuild> build = std::move(it->second);
    state_builds.erase(it);
    return pipeline.AdoptState(hash, build->desc, std::move(build->state), build->error,
                               build->native.has_value());
}

std::string PipelineCache::TakeBuildLine() {
    const BuildStats stats = std::exchange(build_stats, BuildStats{});
    if (stats.queued + stats.states_queued + stats.held + stats.skipped == 0) {
        return {};
    }
    const auto ms = [](u64 ns) { return static_cast<f64>(ns) / 1e6; };
    return fmt::format(
        "{} pipelines and {} render states queued, {} adopted (ready after {:.0f} ms on average, "
        "{:.0f} max), {} stale; draws skipped {}, held {} for {:.0f} ms, max {:.0f} ({} built by "
        "the waiting thread, {} builders lent its QoS); {} building",
        stats.queued, stats.states_queued, stats.adopted,
        stats.adopted != 0 ? ms(stats.ready_ns) / static_cast<f64>(stats.adopted) : 0.0,
        ms(stats.ready_max_ns), stats.stale, stats.skipped, stats.held, ms(stats.held_ns),
        ms(stats.held_max_ns), stats.held_here, stats.boosted,
        graphics_builds.size() + state_builds.size());
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    std::optional<TranslatedPipeline> translated =
        translator.Graphics(graphics_key, environments.Span());
    if (!translated) {
        return nullptr;
    }
    StoreRecord(graphics_key, environments.envs);
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
