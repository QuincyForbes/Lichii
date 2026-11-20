// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include <boost/container/static_vector.hpp>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#ifndef _WIN32
#include <pthread.h>
#endif

#include "common/cityhash.h"
#include "common/fs/file.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/platform_events.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread_worker.h"
#include "video_core/async_translate.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_metal/metal_bind_memo.h"
#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_pipeline_cache.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_switches.h"
#include "video_core/renderer_metal/metal_work_policy.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_warm/shader_warm_builder.h"
#include "video_core/shader_warm/warm_records.h"

namespace Metal {

using VideoCommon::ComputeEnvironment;

namespace {

#if defined(__APPLE__) && TARGET_OS_IPHONE
constexpr bool TranslateForIos = true;
#else
constexpr bool TranslateForIos = false;
#endif

constexpr size_t MaxPrograms = 6;

/// The MSL of a built function is not read again (EDEN_NX2_METAL_PIPELINE_FILE=0 keeps it),
/// but for the reducing variant's, until it is built.
void ReleaseSource(StageProgram& program) {
    if (PipelineFiles::Mode() != 0) {
        if (!program.reduction_variants) {
            std::string{}.swap(program.msl);
        }
        std::string{}.swap(program.msl_without_reduction);
    }
}

/// Builds the program's library; where Metal refuses the MSL with the sampler reduction rewrite
/// (a textual rewrite that some expressions defeat), from the MSL without it: the program then
/// samples with Metal's own reduction instead of being skipped. `error` keeps the first error.
/// EDEN_NX2_METAL_DUMP_FAILED=<folder>: the MSL of each program Metal refuses goes there
/// (failed-N.metal, the error at its end), for looking at what the translation produced.
[[nodiscard]] bool DumpRefusedProgramEnabled() {
    static const bool enabled = [] {
        const char* const folder = std::getenv("EDEN_NX2_METAL_DUMP_FAILED");
        return folder != nullptr && folder[0] != '\0';
    }();
    return enabled;
}

void DumpRefusedProgram(const std::string& msl, const std::string& error) {
    if (!DumpRefusedProgramEnabled()) {
        return;
    }
    static const char* const folder = std::getenv("EDEN_NX2_METAL_DUMP_FAILED");
    static std::atomic<u32> count{};
    const u32 index = count.fetch_add(1, std::memory_order_relaxed);
    if (index >= 64) {
        return;
    }
    std::ofstream file(std::filesystem::path{folder} / fmt::format("failed-{}.metal", index));
    file << msl << "\n// Metal refused it: " << error << '\n';
}

struct SharedLibraryEntry {
    std::mutex mutex;
    std::condition_variable done_cv;
    bool done = false;
    Object library;
    std::string error;
};

struct SharedLibraryState {
    std::mutex mutex;
    std::unordered_map<u64, std::shared_ptr<SharedLibraryEntry>> cache;
};

SharedLibraryState& GetSharedLibraryState() {
    static SharedLibraryState state;
    return state;
}

void ClearSharedLibraryCache() {
    SharedLibraryState& state = GetSharedLibraryState();
    std::vector<std::shared_ptr<SharedLibraryEntry>> evicted;
    {
        std::scoped_lock lock{state.mutex};
        evicted.reserve(state.cache.size());
        std::erase_if(state.cache, [&](auto& item) {
            if (!item.second) {
                return true;
            }
            std::scoped_lock entry_lock{item.second->mutex};
            if (!item.second->done) {
                return false;
            }
            evicted.push_back(std::move(item.second));
            return true;
        });
    }
}

/// Libraries by MSL text, shared by every pipeline using the same program (a fragment program
/// is often part of dozens of pipelines) so the slow source compile runs once; concurrent
/// builders of the same text wait for the first one.
Object SharedLibrary(const Device& device, const std::string& source, std::string& error) {
    SharedLibraryState& state = GetSharedLibraryState();
    const u64 device_bits =
        static_cast<u64>(reinterpret_cast<std::uintptr_t>(DeviceHandle(device)));
    const u64 key = Common::CityHash64(source.data(), source.size()) ^ (u64{source.size()} << 40) ^
                    (device_bits * 0x9E3779B97F4A7C15ULL);
    std::shared_ptr<SharedLibraryEntry> entry;
    std::vector<std::shared_ptr<SharedLibraryEntry>> evicted;
    bool owner = false;
    {
        std::scoped_lock lock{state.mutex};
        const auto found = state.cache.find(key);
        if (found != state.cache.end() && found->second) {
            entry = found->second;
        } else {
            if (state.cache.size() >= 512) {
                evicted.reserve(state.cache.size());
                // Bound the cache; pipelines keep their own references.
                std::erase_if(state.cache, [&](auto& item) {
                    if (!item.second) {
                        return true;
                    }
                    std::scoped_lock entry_lock{item.second->mutex};
                    if (!item.second->done) {
                        return false;
                    }
                    evicted.push_back(std::move(item.second));
                    return true;
                });
            }
            entry = std::make_shared<SharedLibraryEntry>();
            state.cache[key] = entry;
            owner = true;
        }
    }
    evicted.clear();
    if (owner) {
        std::string build_error;
        Object library = CreateLibrary(device, source, build_error);
        {
            std::scoped_lock lock{entry->mutex};
            entry->library = std::move(library);
            entry->error = build_error;
            entry->done = true;
        }
        entry->done_cv.notify_all();
    }
    std::unique_lock lock{entry->mutex};
    entry->done_cv.wait(lock, [&] { return entry->done; });
    if (!entry->library) {
        error = entry->error;
        return {};
    }
    return Object{RetainObject(entry->library.Get())};
}

/// `plain_first` (the programs that may take a reducing variant): with
/// EDEN_NX2_METAL_REDUCTION_VARIANTS, a program with the rewrite is built from the MSL without
/// it, and keeps `msl` for its reducing variant.
void BuildProgramLibrary(const Device& device, StageProgram& program, std::string& error,
                         bool plain_first = false) {
    if (plain_first && ReductionVariantsEnabled() && program.reduction_slot != ~0U &&
        !program.msl.empty() && !program.msl_without_reduction.empty()) {
        program.library = SharedLibrary(device, program.msl_without_reduction, error);
        if (program.library) {
            program.reduction_variants = true;
            return;
        }
        // As before: the rewritten text first, then the plain one.
        error.clear();
    }
    program.library = SharedLibrary(device, program.msl, error);
    if (!program.library) {
        DumpRefusedProgram(program.msl, error);
    }
    if (program.library || program.msl_without_reduction.empty()) {
        return;
    }
    std::string retry_error;
    program.library = SharedLibrary(device, program.msl_without_reduction, retry_error);
    if (program.library) {
        program.reduction_slot = ~0U;
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Metal,
                        "Metal: program built without its sampler reduction rewrite ({})", error);
        }
    }
}

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

} // Anonymous namespace

bool ReductionVariantsEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_REDUCTION_VARIANTS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

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
                                   std::optional<NativeMesh> native_mesh_,
                                   std::optional<XfbCapture> xfb_,
                                   std::optional<GeometryCompute> geometry_)
    : key{key_}, vertex{std::move(vertex_)}, fragment{std::move(fragment_)},
      mesh{std::move(mesh_)}, tessellation{std::move(tessellation_)},
      native_mesh{std::move(native_mesh_)}, xfb{std::move(xfb_)}, geometry{std::move(geometry_)} {
    const StageProgram& frag =
        !fragment.entry_point.empty()
            ? fragment
            : (native_mesh && !native_mesh->fragment.entry_point.empty() ? native_mesh->fragment
                                                                         : fragment);
    const Shader::Info* const pre_info =
        native_mesh                              ? &native_mesh->mesh.info
        : (tessellation && tessellation->domain) ? &tessellation->domain->evaluate.program.info
        : !vertex.entry_point.empty()            ? &vertex.info
                                                 : nullptr;
    if (!frag.entry_point.empty()) {
        for (size_t index = 0; index < frag.info.interpolation.size(); ++index) {
            flat_inputs = flat_inputs ||
                          (frag.info.loads.Generic(index) &&
                           frag.info.interpolation[index] == Shader::Interpolation::Flat &&
                           (pre_info == nullptr || pre_info->stores.Generic(index)));
        }
    }

    const auto build_stage_kinds = [](StageProgram& stage) {
        stage.binding_kinds.clear();
        if (!stage.entry_point.empty()) {
            BindMemo::AppendKinds(stage.info, 0, stage.binding_kinds);
        }
    };
    const auto append_stage_kinds = [](const StageProgram& stage, std::vector<u32>& out) {
        if (!stage.entry_point.empty()) {
            out.insert(out.end(), stage.binding_kinds.begin(), stage.binding_kinds.end());
        }
    };
    build_stage_kinds(vertex);
    build_stage_kinds(fragment);
    append_stage_kinds(vertex, default_binding_kinds);
    append_stage_kinds(fragment, default_binding_kinds);
    if (native_mesh) {
        build_stage_kinds(native_mesh->object);
        build_stage_kinds(native_mesh->mesh);
        build_stage_kinds(native_mesh->fragment);
        append_stage_kinds(native_mesh->object, native_binding_kinds);
        append_stage_kinds(native_mesh->mesh, native_binding_kinds);
        append_stage_kinds(native_mesh->fragment, native_binding_kinds);
    }
    if (mesh) {
        build_stage_kinds(mesh->first.program);
        build_stage_kinds(mesh->mesh.program);
        append_stage_kinds(mesh->first.program, mesh_emulation_binding_kinds);
        if (mesh->has_task) {
            append_stage_kinds(mesh->mesh.program, mesh_emulation_binding_kinds);
        }
        append_stage_kinds(fragment, mesh_emulation_binding_kinds);
    }
    if (tessellation) {
        for (StageProgram& v : tessellation->vertex) {
            build_stage_kinds(v);
        }
        build_stage_kinds(tessellation->control.program);
        if (tessellation->domain) {
            build_stage_kinds(tessellation->domain->evaluate.program);
        }
    }
    if (geometry) {
        for (StageProgram& c : geometry->capture) {
            build_stage_kinds(c);
        }
        build_stage_kinds(geometry->kernel.program);
    }
}

VertexProgramUse GraphicsPipeline::GetVertexUse() noexcept {
    if (!cached_vertex_use) {
        if (vertex.entry_point.empty()) {
            return {};
        }
        const VertexProgramUse computed = VertexUse(vertex);
        if (vertex.function) {
            cached_vertex_use = computed;
        }
        return computed;
    }
    return *cached_vertex_use;
}

void* GraphicsPipeline::StageInKernelState(const Device& device, const StageProgram& program,
                                           u32 kind_tag, size_t variant, u32 index_size,
                                           const RenderPipelineDesc& layout,
                                           std::string_view label) {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(layout.attributes.data()),
                                        sizeof(layout.attributes)) ^
                     Common::CityHash64(reinterpret_cast<const char*>(layout.streams.data()),
                                        sizeof(layout.streams)) ^
                     (u64{variant} << 60) ^ (u64{kind_tag} << 58);
    const auto [it, is_new] = vertex_kernel_states.try_emplace(hash);
    if (is_new) {
        std::string error;
        it->second = CreateStageInComputePipelineState(
            device, program.function.Get(), layout, index_size, program.tessellation_slots.index,
            error, archive);
        if (!it->second) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: {} of {:016x} refused: {}; {}", label,
                          key.unique_hashes[1], error, Describe(layout));
            }
        }
    }
    return it->second.Get();
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
        BuildProgramLibrary(device, program, error);
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
    return StageInKernelState(device, program, 0, variant, Tessellation::IndexSizes[variant],
                              layout, "vertex kernel state");
}

std::pair<void*, void*> GraphicsPipeline::XfbKernelStates(const Device& device, size_t variant,
                                                          const RenderPipelineDesc& layout) {
    if (!xfb || variant >= XfbCapture::IndexSizes.size()) {
        return {};
    }
    StageProgram& program = xfb->capture[variant];
    if (program.build_failed) {
        return {};
    }
    std::string error;
    if (!program.function) {
        BuildProgramLibrary(device, program, error);
        const Object gather =
            program.library ? CreateFunction(program.library.Get(), std::string{Xfb::GatherEntry})
                            : Object{};
        program.function = program.library
                               ? CreateFunction(program.library.Get(), program.entry_point)
                               : Object{};
        if (program.function && gather) {
            xfb->gather[variant] = CreateComputePipelineState(device, gather.Get(), error, archive);
        }
        if (!program.function || !xfb->gather[variant]) {
            program.build_failed = true;
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: transform feedback kernels of {:016x} failed to build: {}",
                          key.unique_hashes[1], error);
            }
            return {};
        }
        ReleaseSource(program);
    }
    return {StageInKernelState(device, program, 1, variant, XfbCapture::IndexSizes[variant], layout,
                               "transform feedback kernel state"),
            xfb->gather[variant].Get()};
}

std::pair<void*, void*> GraphicsPipeline::GeometryKernelStates(const Device& device,
                                                               size_t variant,
                                                               const RenderPipelineDesc& layout) {
    if (!geometry || variant >= GeometryCompute::IndexSizes.size()) {
        return {};
    }
    StageProgram& program = geometry->capture[variant];
    if (program.build_failed) {
        return {};
    }
    std::string error;
    if (!program.function) {
        BuildProgramLibrary(device, program, error);
        const Object records =
            program.library
                ? CreateFunction(program.library.Get(), std::string{Geometry::RecordsEntry})
                : Object{};
        program.function = program.library
                               ? CreateFunction(program.library.Get(), program.entry_point)
                               : Object{};
        if (program.function && records) {
            geometry->records[variant] =
                CreateComputePipelineState(device, records.Get(), error, archive);
        }
        if (!program.function || !geometry->records[variant]) {
            program.build_failed = true;
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: geometry capture kernels of {:016x} failed to build: {}",
                          key.unique_hashes[1], error);
            }
            return {};
        }
        ReleaseSource(program);
    }
    return {StageInKernelState(device, program, 2, variant, GeometryCompute::IndexSizes[variant],
                               layout, "geometry capture kernel state"),
            geometry->records[variant].Get()};
}

void* GraphicsPipeline::GeometryXfbState(const Device& device, u32 stream) {
    if (!geometry || !geometry->kernel.program.library || stream >= Geometry::MaxStreams) {
        return nullptr;
    }
    Object& state = geometry->xfb_states[stream];
    if (!state) {
        std::string error;
        const Object function = CreateFunction(geometry->kernel.program.library.Get(),
                                               Geometry::XfbEntryOf(stream));
        if (function) {
            state = CreateComputePipelineState(device, function.Get(), error, archive);
        }
        if (!state) {
            LOG_ERROR(Render_Metal,
                      "Metal: geometry transform feedback kernel of {:016x} stream {} failed: {}",
                      key.unique_hashes[4], stream, error);
        }
    }
    return state.Get();
}

void* GraphicsPipeline::KernelState(const Device& device, MeshKernel& kernel) {
    if (kernel.program.entry_point.empty()) {
        // A tessellation pipeline without a control program.
        return nullptr;
    }
    if (kernel.state || kernel.failed) {
        return kernel.state.Get();
    }
    StageProgram& program = kernel.program;
    std::string error;
    BuildProgramLibrary(device, program, error);
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
    cached_render_state.state = nullptr;
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
        BuildProgramLibrary(device, *stage, error, stage == &native_mesh->fragment);
        stage->function = stage->library ? CreateFunction(stage->library.Get(), stage->entry_point)
                                         : Object{};
        if (!stage->function) {
            stage->reduction_variants = false;
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
    StageProgram* const expansion = xfb && xfb->expand ? &xfb->render : nullptr;
    for (StageProgram* const stage : {&vertex, &fragment, expansion}) {
        if (stage == nullptr || stage->function || stage->entry_point.empty()) {
            continue;
        }
        if (stage->build_failed) {
            if (stage == expansion) {
                continue;
            }
            return false;
        }
        std::string error;
        BuildProgramLibrary(device, *stage, error, stage != expansion);
        stage->function = stage->library ? CreateFunction(stage->library.Get(), stage->entry_point)
                                         : Object{};
        if (!stage->function) {
            stage->reduction_variants = false;
            stage->build_failed = true;
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_ERROR(Render_Metal, "Metal: {:016x} stage MSL failed to build: {}",
                          key.unique_hashes[stage == &vertex ? 1 : 5], error);
            }
            if (stage == expansion) {
                // The draws stay plain (lines 1 pixel wide, triangles as they are).
                continue;
            }
            return false;
        }
        ReleaseSource(*stage);
    }
    return true;
}

std::optional<MeshFunctions> GraphicsPipeline::NativeFunctions(u32 reducing) const noexcept {
    if (!native_mesh || native_mesh->failed) {
        return std::nullopt;
    }
    const StageProgram& frag = native_mesh->fragment;
    const void* const fragment_function =
        (reducing & ReducingFragment) != 0 && frag.reducing_function ? frag.reducing_function.Get()
                                                                     : frag.function.Get();
    return MeshFunctions{
        .object_function = native_mesh->object.function.Get(),
        .mesh_function = native_mesh->mesh.function.Get(),
        .fragment_function = fragment_function,
        .payload_bytes = native_mesh->payload_bytes,
        .object_threads = native_mesh->object_threads,
        .mesh_threads = native_mesh->mesh_threads,
    };
}

u64 GraphicsPipeline::StateHash(RenderPipelineDesc& desc, u32 reducing) const noexcept {
    if (const std::optional<MeshFunctions> native = NativeFunctions(reducing)) {
        desc.vertex_function = native->mesh_function;
        desc.fragment_function = native->fragment_function;
    } else {
        // A reducing function asked for and not built: the plain one.
        const auto pick = [](const StageProgram& program, bool want_reducing) {
            return want_reducing && program.reducing_function ? program.reducing_function.Get()
                                                              : program.function.Get();
        };
        desc.vertex_function = desc.expand != 0 && xfb && xfb->expand
                                   ? xfb->render.function.Get()
                                   : pick(vertex, (reducing & ReducingVertex) != 0);
        desc.fragment_function = pick(fragment, (reducing & ReducingFragment) != 0);
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

void* GraphicsPipeline::RenderState(const Device& device, RenderPipelineDesc& desc,
                                    u32 reducing) {
    const u64 hash = StateHash(desc, reducing);
    if (const std::optional<void*> found = FindState(hash)) {
        return *found;
    }
    const std::optional<MeshFunctions> native = NativeFunctions(reducing);
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
    : key{key_}, program{std::move(program_)} {
    BindMemo::AppendKinds(program.info, BindMemo::Kind::ComputeBit, program.binding_kinds);
}

void* ComputePipeline::State(const Device& device) {
    if (state || failed) {
        return state.Get();
    }
    std::string error;
    BuildProgramLibrary(device, program, error, true);
    program.function =
        program.library ? CreateFunction(program.library.Get(), program.entry_point) : Object{};
    if (program.function) {
        state = CreateComputePipelineState(device, program.function.Get(), error, archive);
        ReleaseSource(program);
    }
    program.function.Reset();
    program.library.Reset();
    if (!state) {
        program.reduction_variants = false;
        failed = true;
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_ERROR(Render_Metal, "Metal: compute {:016x} failed to build: {}", key.unique_hash,
                      error);
        }
    }
    return state.Get();
}

void* ComputePipeline::ReducingState(const Device& device) {
    if (reducing_state || !state || !program.reduction_variants || program.reducing_failed) {
        return reducing_state.Get();
    }
    std::string error;
    program.reducing_library = SharedLibrary(device, program.msl, error);
    program.reducing_function =
        program.reducing_library
            ? CreateFunction(program.reducing_library.Get(), program.entry_point)
            : Object{};
    if (program.reducing_function) {
        reducing_state =
            CreateComputePipelineState(device, program.reducing_function.Get(), error, archive);
    }
    program.reducing_function.Reset();
    program.reducing_library.Reset();
    if (!reducing_state) {
        program.reducing_failed = true;
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Metal,
                        "Metal: compute {:016x} with its sampler reduction rewrite refused ({}); "
                        "dispatched without it",
                        key.unique_hash, error);
        }
    }
    if (PipelineFiles::Mode() != 0) {
        std::string{}.swap(program.msl);
    }
    return reducing_state.Get();
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory,
                             const Device& device_)
    : VideoCommon::ShaderCache{device_memory}, device{device_},
      translator{TranslateForIos, DeviceAppleFamily(device_), DeviceSupportsMeshShaders(device_),
                 &stage_cache, &msl_cache} {
    // A suspended app may be killed without notice (iOS): the archive is written when it leaves
    // the foreground, and the used list at the GPU thread's next frame.
    background_listener = Common::PlatformEvents::AddBackgroundListener([this] {
        save_requested.store(true, std::memory_order_relaxed);
        if (frame_count.load(std::memory_order_relaxed) != 0) {
            for (int wait = 0; wait < 25 && save_requested.load(std::memory_order_relaxed);
                 ++wait) {
                std::this_thread::sleep_for(std::chrono::milliseconds{10});
            }
        }
        std::scoped_lock lock{lifecycle_mutex};
        if (files.IsOpen()) {
            files.Wait();
        }
        msl_cache.Save();
        if (archive.IsOpen()) {
            archive.Save();
        }
    });
    if (const u32 threads = AsyncPipelineThreads(); threads != 0) {
        // Background QoS: a draw that waits for a build runs it itself or lends it its own.
        const u32 apple_family = DeviceAppleFamily(device);
        const bool supports_mesh_shaders = DeviceSupportsMeshShaders(device);
        VideoCommon::StageCache* const shared_stage_cache = &stage_cache;
        StageMslCache* const shared_msl_cache = &msl_cache;
        builders = std::make_unique<Common::StatefulThreadWorker<PipelineTranslator>>(
            threads, "MetalPipelineBuilder",
            [apple_family, supports_mesh_shaders, shared_stage_cache, shared_msl_cache] {
                return PipelineTranslator{TranslateForIos, apple_family, supports_mesh_shaders,
                                          shared_stage_cache, shared_msl_cache};
            },
            Common::ThreadPlacement::Background);
        LOG_INFO(Render_Metal, "Metal: pipelines first seen in a run are built on {} threads "
                               "(EDEN_NX2_METAL_ASYNC_PIPELINES)",
                 threads);
    }
}

PipelineCache::~PipelineCache() {
    Common::PlatformEvents::RemoveBackgroundListener(background_listener);
    builders.reset();
    ClearSharedLibraryCache();
    files.SaveUsed();
    msl_cache.Save();
}

namespace {
[[nodiscard]] bool GameplayArchiveReadOnly() {
    static const bool read_only = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ARCHIVE");
        // Populate during live gameplay only when explicitly requested with EDEN_NX2_METAL_ARCHIVE=1.
        return value == nullptr || value[0] != '1';
    }();
    return read_only;
}
} // Anonymous namespace

void PipelineCache::OpenArchive(u64 title_id) {
    if (title_id == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
    std::scoped_lock lock{lifecycle_mutex};
    msl_cache.Open(title_dir / "metal_stages.v2.bin", VideoCommon::ShaderWarm::EmissionStamp());
    archive.Open(device, title_dir / "metal_pipelines.bin", GameplayArchiveReadOnly());
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
        msl_cache.Save();
        archive.Save();
    }
    frame_count.fetch_add(1, std::memory_order_relaxed);
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
    for (const auto& [program, build] : reducing_builds) {
        building_states.insert(build->pipeline);
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
    const std::vector<size_t> evicted = WorkPolicy::PressureEvictions(
        uses, frame_count.load(std::memory_order_relaxed), critical, evict_max);
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

namespace {
[[nodiscard]] u64 MetalWarmVariant(u32 apple_family, bool mesh_shaders) {
    const auto& resolution = Settings::values.resolution_info;
    const std::array<u64, 8> inputs{
        VideoCommon::ShaderWarm::EmissionStamp(),
        u64{apple_family},
        mesh_shaders ? 1ULL : 0ULL,
        TranslateForIos ? 1ULL : 0ULL,
        ReductionVariantsEnabled() ? 1ULL : 0ULL,
        resolution.active ? 1ULL : 0ULL,
        resolution.up_scale,
        resolution.down_shift,
    };
    return Common::CityHash64(reinterpret_cast<const char*>(inputs.data()), sizeof(inputs));
}

/// What a Metal precache marker records in place of a Vulkan driver: "MTL", the Apple GPU
/// family and the emission/settings variant. Vulkan's markers never match it, so switching
/// backends precaches once more for the other one.
[[nodiscard]] std::optional<VideoCommon::ShaderWarm::WarmStamp> MetalWarmStamp(
    const std::filesystem::path& warm_file, u64 entries, u32 apple_family, bool mesh_shaders) {
    return VideoCommon::ShaderWarm::CurrentWarmStamp(
        warm_file, entries, 0x4D544C00U, apple_family, VideoCommon::ShaderWarm::WarmCacheVersion,
        MetalWarmVariant(apple_family, mesh_shaders));
}

struct WarmGraphicsRecordKey {
    std::array<u64, 6> unique_hashes;
    Vulkan::FixedPipelineState state;
};
static_assert(sizeof(WarmGraphicsRecordKey) == VideoCommon::ShaderWarm::GraphicsKeySize);

[[nodiscard]] std::pair<GraphicsPipelineKey, size_t> MetalKeyFromWarmKey(
    const WarmGraphicsRecordKey& raw) {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    const size_t slot = raw.unique_hashes[1] != 0 ? 1 : (raw.unique_hashes[5] != 0 ? 5 : 0);
    Regs regs{};
    const Vulkan::FixedPipelineState& state = raw.state;
    regs.mandated_early_z = state.early_z.Value();
    regs.depth_mode = state.ndc_minus_one_to_one != 0 ? Regs::DepthMode::MinusOneToOne
                                                      : Regs::DepthMode::ZeroToOne;
    regs.window_origin.mode.Assign(state.y_negate != 0 ? Regs::WindowOrigin::Mode::LowerLeft
                                                       : Regs::WindowOrigin::Mode::UpperLeft);
    if (state.attachment0_dual_source_blend != 0) {
        regs.blend.enable[0] = 1;
        regs.blend.color_source = Regs::Blend::Factor::Source1Color_GL;
    }
    regs.alpha_test_enabled = 1;
    regs.alpha_test_func =
        Vulkan::FixedPipelineState::UnpackComparisonOp(state.alpha_test_func.Value());
    regs.alpha_test_ref = std::bit_cast<f32>(state.alpha_test_ref);
    for (size_t i = 0; i < state.color_formats.size(); ++i) {
        regs.rt[i].format = static_cast<Tegra::RenderTargetFormat>(state.color_formats[i]);
    }
    regs.point_size = std::bit_cast<f32>(state.point_size);
    if (state.dynamic_state.logic_op_enable != 0) {
        regs.logic_op.enable = 1;
        regs.logic_op.op =
            Vulkan::FixedPipelineState::UnpackLogicOp(state.dynamic_state.logic_op.Value());
    }
    if (state.viewport_swizzles[0] != 0) {
        regs.viewport_scale_offset_enabled = 1;
        regs.viewport_transform[0].swizzle.raw = state.viewport_swizzles[0];
    }
    GraphicsPipelineKey key{};
    key.unique_hashes = raw.unique_hashes;
    PipelineTranslator::RefreshKey(key, regs, PointFillTopology(regs, state.topology.Value()));
    if (slot == 1) {
        if (state.dynamic_vertex_input != 0) {
            key.attribute_types = state.attribute_types;
        } else {
            u64 types = 0;
            for (size_t i = 0; i < state.attributes.size(); ++i) {
                const auto& attr = state.attributes[i];
                u64 type = 0;
                if (attr.enabled != 0) {
                    switch (attr.Type()) {
                    case Regs::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
                        break;
                    case Regs::VertexAttribute::Type::SInt:
                        type = 2;
                        break;
                    case Regs::VertexAttribute::Type::UInt:
                        type = 3;
                        break;
                    default:
                        type = 1;
                        break;
                    }
                }
                types |= type << (i * 2);
            }
            key.attribute_types =
                types != 0 ? types : VideoCommon::ShaderWarm::DefaultAttributeTypes;
        }
    }
    return {key, slot};
}

struct WarmGraphicsResult {
    size_t entries{};
    size_t built{};
    size_t skipped{};
    size_t failed{};
    bool cancelled{};
    bool read_error{};
};

/// POSIX pthread wrapper with a 16 MiB stack so `Common::RunOnLargeStack` inside `SpirvToMsl`
/// executes inline on warm worker threads with zero per-shader `pthread_create` calls.
class LargeStackThread {
public:
    LargeStackThread() = default;

    template <typename Fn>
    explicit LargeStackThread(Fn&& fn) {
#ifndef _WIN32
        auto heap_fn = std::make_unique<std::function<void()>>(std::forward<Fn>(fn));
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 16 * 1024 * 1024);
        const int rc = pthread_create(
            &handle, &attr,
            +[](void* arg) -> void* {
                std::unique_ptr<std::function<void()>> task{
                    static_cast<std::function<void()>*>(arg)};
                (*task)();
                return nullptr;
            },
            heap_fn.get());
        pthread_attr_destroy(&attr);
        if (rc == 0) {
            heap_fn.release();
            joinable = true;
        } else {
            fallback = std::thread(std::move(*heap_fn));
        }
#else
        fallback = std::thread(std::forward<Fn>(fn));
#endif
    }

    LargeStackThread(const LargeStackThread&) = delete;
    LargeStackThread& operator=(const LargeStackThread&) = delete;

    LargeStackThread(LargeStackThread&& other) noexcept
        : fallback{std::move(other.fallback)}
#ifndef _WIN32
          ,
          handle{other.handle}, joinable{std::exchange(other.joinable, false)}
#endif
    {
    }

    ~LargeStackThread() {
        Join();
    }

    void Join() {
#ifndef _WIN32
        if (joinable) {
            pthread_join(handle, nullptr);
            joinable = false;
        }
#endif
        if (fallback.joinable()) {
            fallback.join();
        }
    }

private:
    std::thread fallback;
#ifndef _WIN32
    pthread_t handle{};
    bool joinable{};
#endif
};

/// Bounds concurrent asynchronous Metal XPC compilations so CPU translation threads never block
/// inside `_dispatch_sync_f_slow` while keeping `MTLCompilerService` saturated.
class AsyncCompileGate {
public:
    static constexpr size_t MaxInFlight = 48;

    void Acquire() {
        std::unique_lock lock{mutex};
        cv.wait(lock, [this] { return in_flight < MaxInFlight; });
        ++in_flight;
    }

    void Release() {
        {
            std::scoped_lock lock{mutex};
            --in_flight;
        }
        cv.notify_all();
    }

    void WaitAll() {
        std::unique_lock lock{mutex};
        cv.wait(lock, [this] { return in_flight == 0; });
    }

    [[nodiscard]] size_t InFlight() {
        std::scoped_lock lock{mutex};
        return in_flight;
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    size_t in_flight{};
};

[[nodiscard]] u64 WarmLedgerVariant(u32 apple_family) {
    static const bool full_fast_math = [] {
        const char* const v = std::getenv("EDEN_NX2_METAL_FAST_MATH");
        return v != nullptr && v[0] == '1';
    }();
    static const bool safe_math = [] {
        const char* const v = std::getenv("EDEN_NX2_METAL_SAFE_MATH");
        return v != nullptr && v[0] == '1';
    }();
    const std::array<u64, 4> inputs{
        u64{apple_family},
        TranslateForIos ? 1ULL : 0ULL,
        full_fast_math ? 1ULL : 0ULL,
        safe_math ? 1ULL : 0ULL,
    };
    return Common::CityHash64(reinterpret_cast<const char*>(inputs.data()), sizeof(inputs));
}

[[nodiscard]] u64 HashMslText(std::string_view msl) noexcept {
    return Common::CityHash64(msl.data(), msl.size()) ^ (u64{msl.size()} << 40);
}

/// LSB 0: MTLLibrary-only warm entry; LSB 1: compute MTLLibrary + MTLBinaryArchive entry.
[[nodiscard]] u64 MakeGraphicsLedgerKey(std::string_view msl, u64 variant) noexcept {
    return (HashMslText(msl) ^ variant) & ~1ULL;
}

[[nodiscard]] u64 MakeComputeLedgerKey(std::string_view msl, std::string_view entry_point,
                                       u64 variant) noexcept {
    const u64 entry_hash = Common::CityHash64(entry_point.data(), entry_point.size());
    return ((HashMslText(msl) ^ (entry_hash * 0x9E3779B97F4A7C15ULL)) ^ variant) | 1ULL;
}

/// Incremental append-safe ledger (<ShaderDir>/<title_id>/metal_warm_ledger.v1.bin) of MSL hashes
/// already compiled into macOS's Metal shader cache (and `metal_pipelines.bin` for compute).
class WarmLedger {
public:
    void Open(const std::filesystem::path& file_path, bool drop_compute_entries) {
        std::scoped_lock save_lock{save_mutex};
        std::scoped_lock lock{mutex};
        path = file_path;
        completed.clear();
        in_flight.clear();
        pending.clear();
        file_initialized = false;
        needs_rewrite = false;

        Common::FS::IOFile in{path, Common::FS::FileAccessMode::Read,
                              Common::FS::FileType::BinaryFile,
                              Common::FS::FileShareFlag::ShareReadWrite};
        if (!in.IsOpen()) {
            return;
        }
        LedgerHeader hdr{};
        if (!in.ReadObject(hdr) || hdr.magic != LedgerMagic || hdr.version != LedgerVersion) {
            return;
        }
        const u64 file_size = in.GetSize();
        if (file_size < sizeof(LedgerHeader)) {
            return;
        }
        const u64 payload_bytes = file_size - sizeof(LedgerHeader);
        const size_t count = static_cast<size_t>(payload_bytes / sizeof(u64));
        std::vector<u64> stored(count);
        if (count != 0 && in.ReadSpan<u64>(stored) != count) {
            return;
        }
        in.Close();
        if (payload_bytes % sizeof(u64) != 0) {
            std::error_code resize_ec;
            std::filesystem::resize_file(path, sizeof(LedgerHeader) + count * sizeof(u64),
                                         resize_ec);
        }
        completed.reserve(count);
        for (const u64 key : stored) {
            if (drop_compute_entries && (key & 1ULL) != 0) {
                needs_rewrite = true;
                continue;
            }
            completed.insert(key);
        }
        file_initialized = !needs_rewrite;
    }

    /// Returns true if the caller should compile `key`, or false if it was already compiled or is
    /// currently in flight.
    [[nodiscard]] bool TryBegin(u64 key) {
        std::scoped_lock lock{mutex};
        if (completed.contains(key) || in_flight.contains(key)) {
            return false;
        }
        in_flight.insert(key);
        return true;
    }

    void Complete(u64 key, bool succeeded) {
        std::scoped_lock lock{mutex};
        in_flight.erase(key);
        if (succeeded && completed.insert(key).second) {
            pending.push_back(key);
        }
    }

    [[nodiscard]] size_t PendingSize() const {
        std::scoped_lock lock{mutex};
        return pending.size();
    }

    void Flush() {
        std::scoped_lock save_lock{save_mutex};
        std::filesystem::path target_path;
        bool rewrite = false;
        std::vector<u64> batch;
        std::vector<u64> all_keys;
        {
            std::scoped_lock lock{mutex};
            if (path.empty() || (pending.empty() && !needs_rewrite)) {
                return;
            }
            target_path = path;
            rewrite = !file_initialized || needs_rewrite;
            if (rewrite) {
                all_keys.assign(completed.begin(), completed.end());
            } else {
                batch = pending;
            }
        }

        std::error_code ec;
        std::filesystem::create_directories(target_path.parent_path(), ec);
        if (rewrite) {
            std::filesystem::path tmp = target_path;
            tmp += ".tmp";
            {
                Common::FS::IOFile out{tmp, Common::FS::FileAccessMode::Write,
                                       Common::FS::FileType::BinaryFile,
                                       Common::FS::FileShareFlag::ShareReadWrite};
                const LedgerHeader hdr{.magic = LedgerMagic, .version = LedgerVersion, .reserved = 0};
                if (!out.IsOpen() || !out.WriteObject(hdr) ||
                    (!all_keys.empty() && out.WriteSpan<u64>(all_keys) != all_keys.size()) ||
                    !out.Flush()) {
                    out.Close();
                    std::filesystem::remove(tmp, ec);
                    return;
                }
            }
            std::filesystem::rename(tmp, target_path, ec);
            if (ec) {
                std::filesystem::remove(tmp, ec);
                return;
            }
            std::scoped_lock lock{mutex};
            file_initialized = true;
            needs_rewrite = false;
            pending.clear();
        } else if (!batch.empty()) {
            Common::FS::IOFile out{target_path, Common::FS::FileAccessMode::ReadWrite,
                                   Common::FS::FileType::BinaryFile,
                                   Common::FS::FileShareFlag::ShareReadWrite};
            const u64 prev_size = out.GetSize();
            if (!out.IsOpen() || !out.Seek(static_cast<s64>(prev_size))) {
                return;
            }
            if (out.WriteSpan<u64>(batch) != batch.size() || !out.Flush()) {
                static_cast<void>(out.SetSize(prev_size));
                return;
            }
            std::scoped_lock lock{mutex};
            if (pending.size() >= batch.size()) {
                pending.erase(pending.begin(),
                              pending.begin() + static_cast<std::ptrdiff_t>(batch.size()));
            } else {
                pending.clear();
            }
        }
    }

private:
    static constexpr std::array<char, 8> LedgerMagic{'M', 'T', 'L', 'W', 'L', 'D', '0', '1'};
    static constexpr u32 LedgerVersion = 1;

    struct LedgerHeader {
        std::array<char, 8> magic;
        u32 version;
        u32 reserved;
    };
    static_assert(sizeof(LedgerHeader) == 16);

    mutable std::mutex save_mutex;
    mutable std::mutex mutex;
    std::filesystem::path path;
    bool file_initialized{};
    bool needs_rewrite{};
    std::unordered_set<u64> completed;
    std::unordered_set<u64> in_flight;
    std::vector<u64> pending;
};

void SplitWarmStageMsl(StageProgram& program, std::string& primary_msl, std::string& fallback_msl) {
    if (ReductionVariantsEnabled() && program.reduction_slot != ~0U && !program.msl.empty() &&
        !program.msl_without_reduction.empty()) {
        primary_msl = std::move(program.msl_without_reduction);
        fallback_msl = std::move(program.msl);
    } else {
        primary_msl = std::move(program.msl);
        fallback_msl = std::move(program.msl_without_reduction);
    }
}

void SubmitWarmGraphicsAsync(const Device& device, StageProgram&& program, u64 ledger_variant,
                             WarmLedger& ledger, AsyncCompileGate& gate,
                             std::atomic<size_t>& built, std::atomic<size_t>& skipped,
                             std::atomic<size_t>& failed) {
    std::string primary_msl;
    std::string fallback_msl;
    SplitWarmStageMsl(program, primary_msl, fallback_msl);
    program = StageProgram{};
    const u64 ledger_key = MakeGraphicsLedgerKey(primary_msl, ledger_variant);
    if (!ledger.TryBegin(ledger_key)) {
        skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::string dump_copy = DumpRefusedProgramEnabled() ? primary_msl : std::string{};
    gate.Acquire();
    CreateLibraryAsync(
        device, primary_msl,
        [&device, &ledger, &gate, &built, &failed, ledger_key,
         fallback = std::move(fallback_msl),
         dump = std::move(dump_copy)](Object library, std::string error) mutable {
            if (library) {
                library.Reset();
                ledger.Complete(ledger_key, true);
                built.fetch_add(1, std::memory_order_relaxed);
                gate.Release();
                return;
            }
            if (!dump.empty()) {
                DumpRefusedProgram(dump, error);
                std::string{}.swap(dump);
            }
            if (!fallback.empty()) {
                CreateLibraryAsync(device, fallback,
                                   [&ledger, &gate, &built, &failed,
                                    ledger_key](Object retry_lib, std::string) {
                                       const bool ok = static_cast<bool>(retry_lib);
                                       retry_lib.Reset();
                                       ledger.Complete(ledger_key, ok);
                                       (ok ? built : failed).fetch_add(1, std::memory_order_relaxed);
                                       gate.Release();
                                   });
                return;
            }
            ledger.Complete(ledger_key, false);
            failed.fetch_add(1, std::memory_order_relaxed);
            gate.Release();
        });
}

void SubmitWarmComputeAsync(const Device& device, StageProgram&& program,
                            const ComputePipelineKey& key, PipelineArchive& archive,
                            u64 ledger_variant, WarmLedger& ledger, AsyncCompileGate& gate,
                            std::atomic<size_t>& built, std::atomic<size_t>& skipped,
                            std::atomic<size_t>& failed) {
    std::string primary_msl;
    std::string fallback_msl;
    SplitWarmStageMsl(program, primary_msl, fallback_msl);
    std::string entry_point = std::move(program.entry_point);
    const u32 workgroup_threads =
        (std::max)(1U, key.workgroup_size[0] * key.workgroup_size[1] * key.workgroup_size[2]);
    const u64 ledger_key = MakeComputeLedgerKey(primary_msl, entry_point, ledger_variant);
    if (!ledger.TryBegin(ledger_key)) {
        skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    std::string dump_copy = DumpRefusedProgramEnabled() ? primary_msl : std::string{};
    gate.Acquire();
    CreateLibraryAsync(
        device, primary_msl,
        [&device, &archive, &ledger, &gate, &built, &failed, ledger_key, workgroup_threads,
         entry = std::move(entry_point), fallback = std::move(fallback_msl),
         dump = std::move(dump_copy)](Object library, std::string error) mutable {
            auto finish_pso = [&device, &archive, &ledger, &gate, &built, &failed, ledger_key,
                               workgroup_threads](Object lib, const std::string& ep) {
                const void* const lib_handle = lib.Get();
                auto lib_keep = std::make_shared<Object>(std::move(lib));
                CreateComputeStateAsync(
                    device, lib_handle, ep, workgroup_threads, &archive,
                    [&ledger, &gate, &built, &failed, ledger_key,
                     lib_keep = std::move(lib_keep)](Object state, std::string) mutable {
                        const bool ok = static_cast<bool>(state);
                        state.Reset();
                        lib_keep.reset();
                        ledger.Complete(ledger_key, ok);
                        (ok ? built : failed).fetch_add(1, std::memory_order_relaxed);
                        gate.Release();
                    });
            };
            if (library) {
                finish_pso(std::move(library), entry);
                return;
            }
            if (!dump.empty()) {
                DumpRefusedProgram(dump, error);
                std::string{}.swap(dump);
            }
            if (!fallback.empty()) {
                CreateLibraryAsync(
                    device, fallback,
                    [finish_pso, entry = std::move(entry), &ledger, &gate, &failed,
                     ledger_key](Object retry_lib, std::string) mutable {
                        if (retry_lib) {
                            finish_pso(std::move(retry_lib), entry);
                            return;
                        }
                        ledger.Complete(ledger_key, false);
                        failed.fetch_add(1, std::memory_order_relaxed);
                        gate.Release();
                    });
                return;
            }
            ledger.Complete(ledger_key, false);
            failed.fetch_add(1, std::memory_order_relaxed);
            gate.Release();
        });
}

WarmGraphicsResult WarmGraphicsStages(
    const std::filesystem::path& warm_file, const Device& device, u32 apple_family,
    bool mesh_shaders, size_t num_workers, VideoCommon::StageCache* shared_stage_cache,
    StageMslCache& msl_cache, WarmLedger& ledger, std::stop_token stop, bool check_skip_button,
    const std::function<void(size_t done, size_t total, size_t built_count, size_t failed_count,
                             size_t skipped_count, u64 elapsed_ms)>& on_tick) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    WarmGraphicsResult out{};
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    struct Reader {
        std::mutex mutex;
        std::ifstream file;
        std::streampos end{};
        bool failed{};
        size_t read{};
        std::streamoff position{};
        std::unordered_set<u64> seen_keys;
    } reader;
    try {
        reader.file.open(warm_file, std::ios::binary | std::ios::ate);
        reader.file.exceptions(std::ifstream::failbit);
        reader.end = reader.file.tellg();
        reader.file.seekg(0, std::ios::beg);
        std::array<char, 8> magic{};
        u32 version{};
        reader.file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (magic != expected_magic || version != ShaderWarm::WarmCacheVersion) {
            LOG_WARNING(Render_Metal, "Metal shader precache: {} is not a version {} file", path,
                        ShaderWarm::WarmCacheVersion);
            out.read_error = true;
            return out;
        }
        reader.position = reader.file.tellg();
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Metal, "Metal shader precache: opening {} failed: {}", path, e.what());
        out.read_error = true;
        return out;
    }

    const size_t known_total = static_cast<size_t>(ShaderWarm::ReadWarmCount(warm_file));
    const auto begin = std::chrono::steady_clock::now();
    const size_t workers_count = (std::max)(num_workers, size_t{1});
    const u64 ledger_variant = WarmLedgerVariant(apple_family);
    std::atomic<size_t> built{};
    std::atomic<size_t> skipped{};
    std::atomic<size_t> failed{};
    std::atomic<bool> cancel{false};
    std::atomic<size_t> running{workers_count};
    AsyncCompileGate gate;
    LOG_INFO(Render_Metal,
             "Metal shader precache: {} graphics programs ({} MB) to warm on {} threads",
             known_total, static_cast<long long>(reader.end) / (1024 * 1024), workers_count);
    {
        std::vector<LargeStackThread> threads;
        threads.reserve(workers_count);
        for (size_t i = 0; i < workers_count; ++i) {
            threads.emplace_back([&] {
                static const VideoCommon::GuessMissing guess{};
                PipelineTranslator translator{TranslateForIos, apple_family, mesh_shaders,
                                              shared_stage_cache, &msl_cache};
                while (!cancel.load(std::memory_order_relaxed)) {
                    VideoCommon::FileEnvironment env;
                    WarmGraphicsRecordKey raw_key{};
                    bool duplicate = false;
                    GraphicsPipelineKey metal_key{};
                    size_t slot = 0;
                    {
                        std::scoped_lock lock{reader.mutex};
                        if (reader.failed || reader.file.tellg() == reader.end) {
                            break;
                        }
                        try {
                            u32 num_envs{};
                            reader.file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                            if (num_envs != 1) {
                                throw std::runtime_error("unexpected entry");
                            }
                            env.Deserialize(reader.file);
                            reader.file.read(reinterpret_cast<char*>(&raw_key), sizeof(raw_key));
                            ++reader.read;
                            reader.position = reader.file.tellg();
                            std::tie(metal_key, slot) = MetalKeyFromWarmKey(raw_key);
                            if (slot != 0) {
                                duplicate = !reader.seen_keys.insert(metal_key.Hash()).second;
                            }
                        } catch (const std::exception& e) {
                            LOG_WARNING(Render_Metal,
                                        "Metal shader precache: graphics entry {} unreadable ({})",
                                        reader.read + 1, e.what());
                            reader.failed = true;
                            break;
                        }
                    }
                    if (slot == 0) {
                        failed.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    if (duplicate) {
                        skipped.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    env.SetGuessMissing(&guess);
                    std::optional<StageProgram> program =
                        translator.WarmSingleStage(metal_key, slot, env);
                    if (!program) {
                        failed.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    SubmitWarmGraphicsAsync(device, std::move(*program), ledger_variant, ledger,
                                            gate, built, skipped, failed);
                }
                running.fetch_sub(1, std::memory_order_relaxed);
            });
        }
        while (running.load(std::memory_order_relaxed) != 0 || gate.InFlight() != 0) {
            if (!out.cancelled &&
                (stop.stop_requested() || (check_skip_button && ShaderWarm::TakeSkipWarm()))) {
                cancel.store(true, std::memory_order_relaxed);
                out.cancelled = true;
                break;
            }
            if (msl_cache.PendingSize() >= 128) {
                msl_cache.Flush();
            }
            if (ledger.PendingSize() >= 256) {
                msl_cache.Flush();
                ledger.Flush();
            }
            if (on_tick) {
                const size_t b = built.load(std::memory_order_relaxed);
                const size_t s = skipped.load(std::memory_order_relaxed);
                const size_t f = failed.load(std::memory_order_relaxed);
                const size_t done = b + s + f;
                size_t total = known_total;
                if (total == 0) {
                    std::scoped_lock lock{reader.mutex};
                    const std::streamoff header = 12;
                    const std::streamoff end = reader.end;
                    if (reader.position > header) {
                        total = static_cast<size_t>(static_cast<double>(reader.read) *
                                                    static_cast<double>(end - header) /
                                                    static_cast<double>(reader.position - header));
                    }
                }
                const u64 elapsed_ms = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - begin)
                        .count());
                on_tick(done, (std::max)(total, done), b, f, s, elapsed_ms);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{50});
        }
        for (LargeStackThread& thread : threads) {
            thread.Join();
        }
        gate.WaitAll();
    }
    msl_cache.Flush();
    ledger.Flush();
    out.entries = reader.read;
    out.built = built.load(std::memory_order_relaxed);
    out.skipped = skipped.load(std::memory_order_relaxed);
    out.failed = failed.load(std::memory_order_relaxed);
    out.read_error = !out.cancelled && (reader.failed || reader.end != std::streampos{reader.position});
    return out;
}
} // Anonymous namespace

std::optional<VideoCommon::ShaderWarm::WarmStamp> PipelineCache::QueuePrecacheFile(
    u64 title_id, u32 apple_family,
    const std::function<void(const ComputePipelineKey&, VideoCommon::FileEnvironment)>& queue,
    bool only_if_unmarked) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const std::filesystem::path warm_file = ShaderWarm::ComputeWarmPath(title_id);
    const std::string path = Common::FS::PathToUTF8String(warm_file);
    std::error_code ec;
    if (!std::filesystem::exists(warm_file, ec)) {
        if (only_if_unmarked) {
            LOG_INFO(Render_Metal, "Metal shader precache: no precache file at {} (Build Shader "
                                   "Precache in the game's menu makes one)",
                     path);
        }
        return std::nullopt;
    }
    const bool mesh_shaders = DeviceSupportsMeshShaders(device);
    const auto current = MetalWarmStamp(warm_file, 0, apple_family, mesh_shaders);
    const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
    const bool caches_present =
        (marker && marker->entries == 0) ||
        ((!archive.IsOpen() || archive.HasEntries()) && msl_cache.Size() != 0);
    const bool marked = current && marker && marker->Matches(*current) && caches_present;
    if (only_if_unmarked && marked) {
        LOG_INFO(Render_Metal, "Metal shader precache: compute programs already built");
        return std::nullopt;
    }
    if (!only_if_unmarked && !marked) {
        return std::nullopt;
    }
    // Its own reader, as Vulkan's: the disk cache loader deletes a file it rejects.
    size_t entries = 0;
    size_t queued = 0;
    std::unordered_set<ComputePipelineKey> seen_warm;
    try {
        std::ifstream file(warm_file, std::ios::binary | std::ios::ate);
        file.exceptions(std::ifstream::failbit);
        const auto end = file.tellg();
        file.seekg(0, std::ios::beg);
        std::array<char, 8> magic;
        u32 version{};
        file.read(magic.data(), magic.size())
            .read(reinterpret_cast<char*>(&version), sizeof(version));
        constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c', 'h'};
        if (magic != expected_magic || version != ShaderWarm::WarmCacheVersion) {
            LOG_WARNING(Render_Metal,
                        "Metal shader precache: {} is not a version {} file; build it again",
                        path, ShaderWarm::WarmCacheVersion);
            return std::nullopt;
        }
        static const VideoCommon::GuessMissing guess{};
        while (file.tellg() != end) {
            u32 num_envs{};
            file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
            VideoCommon::FileEnvironment env;
            env.Deserialize(file);
            if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                LOG_WARNING(Render_Metal, "Metal shader precache: unexpected entry in {}", path);
                return std::nullopt;
            }
            ComputePipelineKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            ++entries;
            // Programs in the pipeline records were queued already.
            if (disk_compute.contains(key) || !seen_warm.insert(key).second) {
                continue;
            }
            // The precache file records no reads: guesses stand in for them.
            env.SetGuessMissing(&guess);
            queue(key, std::move(env));
            ++queued;
        }
    } catch (const std::exception& e) {
        LOG_WARNING(Render_Metal, "Metal shader precache: reading {} failed: {}", path, e.what());
        return std::nullopt;
    }
    LOG_INFO(Render_Metal, "Metal shader precache: {} compute programs in {}, {} queued{}",
             entries, path, queued, only_if_unmarked ? " to precache" : " from warm archive");
    return MetalWarmStamp(warm_file, entries, apple_family, mesh_shaders);
}

void PipelineCache::RunWarmPrecache(u64 title_id, u32 apple_family, bool mesh_shaders,
                                    size_t num_workers, std::stop_token stop_loading,
                                    const VideoCore::DiskResourceLoadCallback& callback) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
    const bool restore_read_only = archive.IsReadOnly();
    archive.SetReadOnly(false);

    WarmLedger ledger;
    ledger.Open(title_dir / "metal_warm_ledger.v1.bin",
                archive.IsOpen() && !archive.HasEntries());

    const auto restore_archive = [this, &ledger, restore_read_only] {
        msl_cache.Flush();
        archive.Save();
        ledger.Flush();
        archive.SetReadOnly(restore_read_only);
    };

    struct ComputeJob {
        ComputePipelineKey key;
        VideoCommon::FileEnvironment env;
    };
    std::vector<ComputeJob> compute_jobs;
    const auto compute_stamp = QueuePrecacheFile(
        title_id, apple_family,
        [&](const ComputePipelineKey& key, VideoCommon::FileEnvironment env) {
            compute_jobs.push_back({key, std::move(env)});
        },
        true);

    bool skipped_warm = false;
    if (!compute_jobs.empty()) {
        const auto start = std::chrono::steady_clock::now();
        const size_t total = compute_jobs.size();
        const size_t workers_count = (std::max)(num_workers, size_t{1});
        const u64 ledger_variant = WarmLedgerVariant(apple_family);
        std::atomic<size_t> next_index{0};
        std::atomic<size_t> built{0};
        std::atomic<size_t> skipped{0};
        std::atomic<size_t> failed{0};
        std::atomic<bool> cancel{false};
        std::atomic<size_t> running{workers_count};
        AsyncCompileGate gate;
        ShaderWarm::SetBuildStatus({.total = total, .threads = workers_count, .running = true});
        if (callback) {
            callback(VideoCore::LoadCallbackStage::Warm, 0, total);
        }
        {
            std::vector<LargeStackThread> threads;
            threads.reserve(workers_count);
            for (size_t w = 0; w < workers_count; ++w) {
                threads.emplace_back([&] {
                    PipelineTranslator worker{TranslateForIos, apple_family, mesh_shaders,
                                              &stage_cache, &msl_cache};
                    for (;;) {
                        if (cancel.load(std::memory_order_relaxed)) {
                            break;
                        }
                        const size_t idx = next_index.fetch_add(1, std::memory_order_relaxed);
                        if (idx >= total) {
                            break;
                        }
                        ComputeJob& job = compute_jobs[idx];
                        std::optional<StageProgram> program =
                            worker.Compute(job.key.unique_hash, job.env);
                        if (!program) {
                            failed.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        SubmitWarmComputeAsync(device, std::move(*program), job.key, archive,
                                               ledger_variant, ledger, gate, built, skipped,
                                               failed);
                    }
                    running.fetch_sub(1, std::memory_order_relaxed);
                });
            }
            while (running.load(std::memory_order_relaxed) != 0 || gate.InFlight() != 0) {
                if (!skipped_warm && (stop_loading.stop_requested() || ShaderWarm::TakeSkipWarm())) {
                    cancel.store(true, std::memory_order_relaxed);
                    skipped_warm = true;
                    break;
                }
                if (msl_cache.PendingSize() >= 128) {
                    msl_cache.Flush();
                }
                if (ledger.PendingSize() >= 256) {
                    msl_cache.Flush();
                    archive.Save();
                    ledger.Flush();
                }
                const size_t b = built.load(std::memory_order_relaxed);
                const size_t s = skipped.load(std::memory_order_relaxed);
                const size_t f = failed.load(std::memory_order_relaxed);
                const size_t done = b + s + f;
                const u64 elapsed_ms = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count());
                ShaderWarm::SetBuildStatus({
                    .total = total,
                    .done = done,
                    .built = b,
                    .failed = f,
                    .skipped = s,
                    .threads = workers_count,
                    .elapsed_ms = elapsed_ms,
                    .running = true,
                });
                if (callback) {
                    callback(VideoCore::LoadCallbackStage::Warm, done, total);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds{50});
            }
            for (LargeStackThread& thread : threads) {
                thread.Join();
            }
            gate.WaitAll();
        }
        const size_t b = built.load(std::memory_order_relaxed);
        const size_t s = skipped.load(std::memory_order_relaxed);
        const size_t f = failed.load(std::memory_order_relaxed);
        const u64 elapsed_ms = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        ShaderWarm::SetBuildStatus({
            .total = total,
            .done = b + s + f,
            .built = b,
            .failed = f,
            .skipped = s,
            .threads = workers_count,
            .elapsed_ms = elapsed_ms,
            .running = false,
        });
        restore_archive();
        archive.SetReadOnly(false);
        if (!skipped_warm && !stop_loading.stop_requested() && compute_stamp) {
            static_cast<void>(ShaderWarm::WriteWarmMarker(ShaderWarm::ComputeWarmPath(title_id),
                                                          *compute_stamp));
        }
        LOG_INFO(Render_Metal,
                 "Metal shader precache: compute {} ({} built, {} skipped, {} failed in {:.1f} s)",
                 stop_loading.stop_requested() ? "stopped" : (skipped_warm ? "skipped" : "done"),
                 b, s, f, elapsed_ms / 1000.0);
    } else if (compute_stamp && !stop_loading.stop_requested()) {
        // Every entry in warm-compute.bin was already built in the session records (or 0 entries).
        restore_archive();
        archive.SetReadOnly(false);
        static_cast<void>(
            ShaderWarm::WriteWarmMarker(ShaderWarm::ComputeWarmPath(title_id), *compute_stamp));
    }

    std::error_code ec;
    const std::filesystem::path graphics_warm = ShaderWarm::GraphicsWarmPath(title_id);
    if (!skipped_warm && !stop_loading.stop_requested() &&
        std::filesystem::exists(graphics_warm, ec)) {
        const auto current = MetalWarmStamp(graphics_warm, 0, apple_family, mesh_shaders);
        const auto marker = ShaderWarm::ReadWarmMarker(graphics_warm);
        if (current && marker && marker->Matches(*current) &&
            (marker->entries == 0 || msl_cache.Size() != 0)) {
            LOG_INFO(Render_Metal, "Metal shader precache: graphics stages already built");
        } else {
            const WarmGraphicsResult graphics_res = WarmGraphicsStages(
                graphics_warm, device, apple_family, mesh_shaders, num_workers, &stage_cache,
                msl_cache, ledger, stop_loading, true,
                [&](size_t done, size_t total, size_t b, size_t f, size_t s, u64 elapsed_ms) {
                    ShaderWarm::SetBuildStatus({
                        .total = total,
                        .done = done,
                        .built = b,
                        .failed = f,
                        .skipped = s,
                        .threads = num_workers,
                        .elapsed_ms = elapsed_ms,
                        .running = true,
                    });
                    if (callback) {
                        callback(VideoCore::LoadCallbackStage::Warm, done, total);
                    }
                });
            ShaderWarm::SetBuildStatus({
                .total = graphics_res.entries,
                .done = graphics_res.built + graphics_res.failed + graphics_res.skipped,
                .built = graphics_res.built,
                .failed = graphics_res.failed,
                .skipped = graphics_res.skipped,
                .threads = num_workers,
                .running = false,
            });
            if (!graphics_res.cancelled && !graphics_res.read_error &&
                !stop_loading.stop_requested()) {
                msl_cache.Save();
                if (const auto done_stamp =
                        MetalWarmStamp(graphics_warm, graphics_res.entries, apple_family,
                                       mesh_shaders)) {
                    static_cast<void>(ShaderWarm::WriteWarmMarker(graphics_warm, *done_stamp));
                }
            }
            LOG_INFO(Render_Metal,
                     "Metal shader precache: graphics stages {} ({} built, {} skipped, {} failed)",
                     stop_loading.stop_requested()
                         ? "stopped"
                         : (graphics_res.cancelled ? "skipped" : "done"),
                     graphics_res.built, graphics_res.skipped, graphics_res.failed);
        }
    }

    restore_archive();
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    OpenArchive(title_id);
    if (title_id == 0 || PipelineFiles::Mode() == 0) {
        return;
    }
    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    {
        std::scoped_lock lock{lifecycle_mutex};
        files.Open(shader_dir / fmt::format("{:016x}", title_id));
        if (!files.IsOpen()) {
            return;
        }
    }
    // A boot build that never finished (Metal killed the process in it) is not tried again at
    // once: this run reads the records only, the next one builds again.
    const bool build = files.BeginBoot();
    if (!build) {
        LOG_WARNING(Render_Metal, "Metal: the last boot build of pipelines did not finish; "
                                  "nothing is built ahead in this run");
    }
    // Shader precache (precache_shaders Always, Ask answered yes, or a Build Shaders launch):
    // every recorded pipeline is built, not only those of the recent sessions, and the compute &
    // graphics programs found in the game files too.
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    const bool precache =
        build && (ShaderWarm::PrecacheBeforeStart(Settings::values.precache_shaders.GetValue(),
                                                  ShaderWarm::TakeWarmBeforeStart()) ||
                  ShaderWarm::BuildOnlyRequested());
    const bool restore_read_only = archive.IsReadOnly();
    if (precache) {
        archive.SetReadOnly(false);
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
        VideoCommon::StageCache* const shared_stage_cache = &stage_cache;
        StageMslCache* const shared_msl_cache = &msl_cache;
        Common::StatefulThreadWorker<PipelineTranslator> workers{
            build ? num_workers : 0, "MetalPipelines",
            [apple_family, mesh_shaders, shared_stage_cache, shared_msl_cache] {
                return PipelineTranslator{TranslateForIos, apple_family, mesh_shaders,
                                          shared_stage_cache, shared_msl_cache};
            }};
        const auto finish = [&state, &callback] {
            ++state.built;
            if (state.has_loaded && callback) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        };
        const auto queue_compute = [&](const ComputePipelineKey& key,
                                       VideoCommon::FileEnvironment env) {
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
        const auto load_compute = [&](std::ifstream& file, VideoCommon::FileEnvironment env) {
            ComputePipelineKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            if (!disk_compute.insert(key).second || !build) {
                return;
            }
            if (!precache && !files.BuildAtBoot(key.Hash())) {
                ++unlisted;
                return;
            }
            queue_compute(key, std::move(env));
        };
        const auto load_graphics = [&](std::ifstream& file,
                                       std::vector<VideoCommon::FileEnvironment> envs) {
            GraphicsPipelineKey key;
            file.read(reinterpret_cast<char*>(&key), sizeof(key));
            if (!disk_graphics.insert(key).second || !build || envs.size() > MaxPrograms) {
                return;
            }
            if (!precache && !files.BuildAtBoot(key.Hash())) {
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
                        std::move(translated->native_mesh), std::move(translated->xfb),
                        std::move(translated->geometry));
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
    ClearSharedLibraryCache();
    if (!build) {
        return;
    }
    if (precache && !stop_loading.stop_requested()) {
        RunWarmPrecache(title_id, apple_family, mesh_shaders, num_workers, stop_loading, callback);
    } else {
        msl_cache.Save();
    }
    ClearSharedLibraryCache();
    archive.SetReadOnly(restore_read_only);
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
    if (maxwell3d != last_rasterize_channel ||
        maxwell3d->regs.rasterize_enable != last_rasterize_enable) {
        maxwell3d->dirty.flags[VideoCommon::Dirty::Shaders] = true;
        last_rasterize_channel = maxwell3d;
        last_rasterize_enable = maxwell3d->regs.rasterize_enable;
    }
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        refusal = "shader programs not in guest memory";
        return nullptr;
    }
    PipelineTranslator::RefreshKey(graphics_key, maxwell3d->regs, DrawTopology(*maxwell3d));
    const u64 current_frame = frame_count.load(std::memory_order_relaxed);
    if (current_pipeline != nullptr && current_pipeline->Key() == graphics_key) {
        current_pipeline->SetLastUse(current_frame);
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
        current_pipeline->SetLastUse(current_frame);
        if (!current_pipeline->MarkUsed()) {
            files.NoteUsed(graphics_key.Hash());
        }
    }
    return current_pipeline;
}

ComputePipeline* PipelineCache::CurrentComputePipeline(u64* out_native_key) {
    if (out_native_key != nullptr) {
        *out_native_key = 0;
    }
    const VideoCommon::ShaderInfo* const shader = ComputeShader();
    if (shader == nullptr) {
        refusal = "shader programs not in guest memory";
        return nullptr;
    }
    if (out_native_key != nullptr && shader->lowered_native) {
        *out_native_key = shader->lowered_native->native_program_key;
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
                std::move(translated->native_mesh), std::move(translated->xfb),
                std::move(translated->geometry));
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
                                 bool& building_state, u32 reducing) {
    building_state = false;
    if (!builders) {
        return pipeline.RenderState(device, desc, reducing);
    }
    const u64 hash = pipeline.StateHash(desc, reducing);
    if (const std::optional<void*> found = pipeline.FindState(hash)) {
        return *found;
    }
    const std::pair<const GraphicsPipeline*, u64> state_key{&pipeline, hash};
    auto it = state_builds.find(state_key);
    if (it == state_builds.end()) {
        if (!MayBeSkipped()) {
            // Nothing queued and the draw waits: built here as before.
            return pipeline.RenderState(device, desc, reducing);
        }
        auto build = std::make_shared<StateBuild>();
        build->desc = desc;
        build->native = pipeline.NativeFunctions(reducing);
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

void PipelineCache::RunReducingBuild(ReducingBuild& build) const {
    build.library = SharedLibrary(device, build.msl, build.error);
    build.function = build.library ? CreateFunction(build.library.Get(), build.entry_point)
                                   : Object{};
    if (!build.library) {
        DumpRefusedProgram(build.msl, build.error);
    }
    std::string{}.swap(build.msl);
}

void* PipelineCache::AdoptReducing(StageProgram& program, ReducingBuild& build) {
    if (build.function) {
        program.reducing_library = std::move(build.library);
        program.reducing_function = std::move(build.function);
        ++reducing_built;
        return program.reducing_function.Get();
    }
    program.reducing_failed = true;
    ++reducing_refused;
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
        LOG_WARNING(Render_Metal,
                    "Metal: {:016x} with its sampler reduction rewrite refused ({}); drawn "
                    "without it",
                    build.program_hash, build.error);
    }
    return nullptr;
}

void* PipelineCache::ReducingFunction(GraphicsPipeline& pipeline, StageProgram& program,
                                      bool& building_variant) {
    building_variant = false;
    if (program.reducing_function) {
        return program.reducing_function.Get();
    }
    if (!program.reduction_variants || program.reducing_failed) {
        return nullptr;
    }
    const auto make_build = [&] {
        auto build = std::make_shared<ReducingBuild>();
        build->pipeline = &pipeline;
        // The plain function was built: the rewritten MSL is not read again but here.
        build->msl = PipelineFiles::Mode() != 0 ? std::move(program.msl) : program.msl;
        build->entry_point = program.entry_point;
        build->program_hash =
            pipeline.Key().unique_hashes[&program == &pipeline.Vertex() ? 1 : 5];
        return build;
    };
    auto it = reducing_builds.find(&program);
    if (it == reducing_builds.end()) {
        if (!builders || !MayBeSkipped()) {
            // Nothing queued and the draw waits: built here.
            const std::shared_ptr<ReducingBuild> build = make_build();
            InAutoreleasePool([&] { RunReducingBuild(*build); });
            return AdoptReducing(program, *build);
        }
        std::shared_ptr<ReducingBuild> build = make_build();
        ReducingBuild* const raw = build.get();
        build->job = std::make_shared<PipelineBuildJob>([this, raw](PipelineTranslator&) {
            InAutoreleasePool([&] { RunReducingBuild(*raw); });
        });
        builders->QueueWork([build](PipelineTranslator* worker) { build->job->Run(*worker); });
        it = reducing_builds.emplace(&program, std::move(build)).first;
        ++reducing_queued;
    }
    if (!it->second->job->IsDone()) {
        if (MayBeSkipped()) {
            ++build_stats.skipped;
            building_variant = true;
            return nullptr;
        }
        Hold(*it->second->job);
    }
    const std::shared_ptr<ReducingBuild> build = std::move(it->second);
    reducing_builds.erase(it);
    return AdoptReducing(program, *build);
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
        std::move(translated->native_mesh), std::move(translated->xfb),
                std::move(translated->geometry));
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

bool MetalWarmPending(u64 title_id) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    if (title_id == 0) {
        return false;
    }
    struct CachedDeviceCaps {
        bool available{};
        u32 apple_family{};
        bool mesh_shaders{};
    };
    static const CachedDeviceCaps caps = [] {
        auto dev = CreateOfflineDevice();
        if (!dev) {
            return CachedDeviceCaps{};
        }
        return CachedDeviceCaps{
            .available = true,
            .apple_family = DeviceAppleFamily(*dev),
            .mesh_shaders = DeviceSupportsMeshShaders(*dev),
        };
    }();
    if (!caps.available) {
        return false;
    }

    const u64 prev_program_id = Settings::GetCurrentProgramID();
    const auto prev_resolution_info = Settings::values.resolution_info;
    const bool switched_program = prev_program_id != title_id;
    if (switched_program) {
        Settings::SetCurrentProgramID(title_id);
        Settings::UpdateRescalingInfo();
    }
    SCOPE_EXIT {
        if (switched_program) {
            Settings::SetCurrentProgramID(prev_program_id);
            Settings::values.resolution_info = prev_resolution_info;
        }
    };

    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
    const auto stages_path = title_dir / "metal_stages.v2.bin";
    const auto pipelines_path = title_dir / "metal_pipelines.bin";
    static const bool archive_enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_ARCHIVE");
        return value == nullptr || value[0] != '0';
    }();

    std::optional<bool> stages_valid;
    const auto has_stages = [&] {
        if (!stages_valid) {
            stages_valid = StageMslCache::HasValidEntries(stages_path, ShaderWarm::EmissionStamp());
        }
        return *stages_valid;
    };

    const auto file_pending = [&](const std::filesystem::path& warm_file, bool need_archive) {
        if (ShaderWarm::WarmFilesStale(warm_file).has_value()) {
            return true;
        }
        const auto current =
            MetalWarmStamp(warm_file, 0, caps.apple_family, caps.mesh_shaders);
        const auto marker = ShaderWarm::ReadWarmMarker(warm_file);
        if (!current || !marker || !marker->Matches(*current)) {
            return true;
        }
        const u64 entries =
            marker->entries != 0 ? marker->entries : ShaderWarm::ReadWarmCount(warm_file);
        if (entries > 0) {
            if (!has_stages()) {
                return true;
            }
            if (need_archive && archive_enabled) {
                std::error_code ec;
                const auto sz = std::filesystem::file_size(pipelines_path, ec);
                if (ec || sz == 0) {
                    return true;
                }
            }
        }
        return false;
    };

    if (file_pending(ShaderWarm::ComputeWarmPath(title_id), true)) {
        return true;
    }
    if (ShaderWarm::GraphicsPrecacheRequested() &&
        file_pending(ShaderWarm::GraphicsWarmPath(title_id), false)) {
        return true;
    }
    return false;
}

OfflineWarmSummary CompileWarmShadersOffline(
    u64 title_id, std::stop_token stop,
    const std::function<void(size_t done, size_t total, std::string_view stage)>& on_progress) {
    namespace ShaderWarm = VideoCommon::ShaderWarm;
    OfflineWarmSummary summary{};
    if (title_id == 0) {
        summary.error = "Invalid title ID";
        return summary;
    }
    auto device_holder = CreateOfflineDevice();
    if (!device_holder) {
        summary.error = "No Metal device available";
        return summary;
    }

    const u64 prev_program_id = Settings::GetCurrentProgramID();
    const auto prev_resolution_info = Settings::values.resolution_info;
    Settings::SetCurrentProgramID(title_id);
    Settings::UpdateRescalingInfo();
    SCOPE_EXIT {
        Settings::SetCurrentProgramID(prev_program_id);
        Settings::values.resolution_info = prev_resolution_info;
    };

    const Device& device = *device_holder;
    const u32 apple_family = DeviceAppleFamily(device);
    const bool mesh_shaders = DeviceSupportsMeshShaders(device);
    const size_t num_workers = (std::max)(std::thread::hardware_concurrency(), 2U) - 1;

    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
    StageMslCache msl_cache;
    msl_cache.Open(title_dir / "metal_stages.v2.bin", ShaderWarm::EmissionStamp());
    PipelineArchive archive;
    static_cast<void>(archive.Open(device, title_dir / "metal_pipelines.bin", false));
    WarmLedger ledger;
    ledger.Open(title_dir / "metal_warm_ledger.v1.bin",
                archive.IsOpen() && !archive.HasEntries());

    std::error_code ec;
    const std::filesystem::path compute_warm = ShaderWarm::ComputeWarmPath(title_id);
    if (std::filesystem::exists(compute_warm, ec) && !stop.stop_requested()) {
        const auto current = MetalWarmStamp(compute_warm, 0, apple_family, mesh_shaders);
        const auto marker = ShaderWarm::ReadWarmMarker(compute_warm);
        const bool caches_present =
            (marker && marker->entries == 0) ||
            ((!archive.IsOpen() || archive.HasEntries()) && msl_cache.Size() != 0);
        if (current && marker && marker->Matches(*current) && caches_present) {
            summary.skipped += marker->entries;
        } else {
            struct ComputeJob {
                ComputePipelineKey key;
                VideoCommon::FileEnvironment env;
            };
            std::vector<ComputeJob> jobs;
            std::unordered_set<ComputePipelineKey> seen;
            bool read_ok = false;
            size_t entries = 0;
            try {
                std::ifstream file(compute_warm, std::ios::binary | std::ios::ate);
                file.exceptions(std::ifstream::failbit);
                const auto end = file.tellg();
                file.seekg(0, std::ios::beg);
                std::array<char, 8> magic{};
                u32 version{};
                file.read(magic.data(), magic.size())
                    .read(reinterpret_cast<char*>(&version), sizeof(version));
                constexpr std::array<char, 8> expected_magic{'y', 'u', 'z', 'u', 'c', 'a', 'c',
                                                             'h'};
                if (magic == expected_magic && version == ShaderWarm::WarmCacheVersion) {
                    static const VideoCommon::GuessMissing guess{};
                    while (file.tellg() != end) {
                        u32 num_envs{};
                        file.read(reinterpret_cast<char*>(&num_envs), sizeof(num_envs));
                        VideoCommon::FileEnvironment env;
                        env.Deserialize(file);
                        if (num_envs != 1 || env.ShaderStage() != Shader::Stage::Compute) {
                            throw std::runtime_error("unexpected compute warm entry");
                        }
                        ComputePipelineKey key{};
                        file.read(reinterpret_cast<char*>(&key), sizeof(key));
                        ++entries;
                        if (!seen.insert(key).second) {
                            ++summary.skipped;
                            continue;
                        }
                        env.SetGuessMissing(&guess);
                        jobs.push_back({key, std::move(env)});
                    }
                    read_ok = true;
                }
            } catch (const std::exception& e) {
                summary.error = e.what();
                return summary;
            }
            if (read_ok && !jobs.empty()) {
                const size_t total = jobs.size();
                const size_t workers_count = (std::max)(num_workers, size_t{1});
                const u64 ledger_variant = WarmLedgerVariant(apple_family);
                std::atomic<size_t> next_index{0};
                std::atomic<size_t> built{0};
                std::atomic<size_t> skipped{0};
                std::atomic<size_t> failed{0};
                std::atomic<size_t> running{workers_count};
                AsyncCompileGate gate;
                if (on_progress) {
                    on_progress(0, total, "Compiling Metal compute shaders");
                }
                {
                    std::vector<LargeStackThread> threads;
                    threads.reserve(workers_count);
                    for (size_t w = 0; w < workers_count; ++w) {
                        threads.emplace_back([&] {
                            PipelineTranslator worker{TranslateForIos, apple_family, mesh_shaders,
                                                      nullptr, &msl_cache};
                            for (;;) {
                                if (stop.stop_requested()) {
                                    break;
                                }
                                const size_t idx =
                                    next_index.fetch_add(1, std::memory_order_relaxed);
                                if (idx >= total) {
                                    break;
                                }
                                ComputeJob& job = jobs[idx];
                                std::optional<StageProgram> program =
                                    worker.Compute(job.key.unique_hash, job.env);
                                if (!program) {
                                    failed.fetch_add(1, std::memory_order_relaxed);
                                    continue;
                                }
                                SubmitWarmComputeAsync(device, std::move(*program), job.key,
                                                       archive, ledger_variant, ledger, gate,
                                                       built, skipped, failed);
                            }
                            running.fetch_sub(1, std::memory_order_relaxed);
                        });
                    }
                    while (running.load(std::memory_order_relaxed) != 0 || gate.InFlight() != 0) {
                        if (stop.stop_requested()) {
                            break;
                        }
                        if (msl_cache.PendingSize() >= 128) {
                            msl_cache.Flush();
                        }
                        if (ledger.PendingSize() >= 256) {
                            msl_cache.Flush();
                            archive.Save();
                            ledger.Flush();
                        }
                        if (on_progress) {
                            const size_t done = built.load(std::memory_order_relaxed) +
                                                skipped.load(std::memory_order_relaxed) +
                                                failed.load(std::memory_order_relaxed);
                            on_progress(done, total, "Compiling Metal compute shaders");
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds{50});
                    }
                    for (LargeStackThread& thread : threads) {
                        thread.Join();
                    }
                    gate.WaitAll();
                }
                summary.built += built.load(std::memory_order_relaxed);
                summary.skipped += skipped.load(std::memory_order_relaxed);
                summary.failed += failed.load(std::memory_order_relaxed);
                msl_cache.Flush();
                archive.Save();
                ledger.Flush();
                if (!stop.stop_requested()) {
                    if (const auto done_stamp =
                            MetalWarmStamp(compute_warm, entries, apple_family, mesh_shaders)) {
                        static_cast<void>(ShaderWarm::WriteWarmMarker(compute_warm, *done_stamp));
                    }
                }
            } else if (read_ok && !stop.stop_requested()) {
                if (const auto done_stamp =
                        MetalWarmStamp(compute_warm, entries, apple_family, mesh_shaders)) {
                    static_cast<void>(ShaderWarm::WriteWarmMarker(compute_warm, *done_stamp));
                }
            }
        }
    }

    const std::filesystem::path graphics_warm = ShaderWarm::GraphicsWarmPath(title_id);
    if (std::filesystem::exists(graphics_warm, ec) && !stop.stop_requested()) {
        const auto current = MetalWarmStamp(graphics_warm, 0, apple_family, mesh_shaders);
        const auto marker = ShaderWarm::ReadWarmMarker(graphics_warm);
        if (current && marker && marker->Matches(*current) &&
            (marker->entries == 0 || msl_cache.Size() != 0)) {
            summary.skipped += marker->entries;
        } else {
            const WarmGraphicsResult graphics_res = WarmGraphicsStages(
                graphics_warm, device, apple_family, mesh_shaders, num_workers, nullptr,
                msl_cache, ledger, stop, false,
                [&](size_t done, size_t total, size_t, size_t, size_t, u64) {
                    if (on_progress) {
                        on_progress(done, total, "Compiling Metal graphics stages");
                    }
                });
            summary.built += graphics_res.built;
            summary.skipped += graphics_res.skipped;
            summary.failed += graphics_res.failed;
            msl_cache.Flush();
            ledger.Flush();
            if (!graphics_res.cancelled && !graphics_res.read_error && !stop.stop_requested()) {
                if (const auto done_stamp = MetalWarmStamp(graphics_warm, graphics_res.entries,
                                                           apple_family, mesh_shaders)) {
                    static_cast<void>(ShaderWarm::WriteWarmMarker(graphics_warm, *done_stamp));
                }
            }
        }
    }

    msl_cache.Flush();
    archive.Save();
    ledger.Flush();
    summary.ok = !stop.stop_requested();
    if (!summary.ok && summary.error.empty()) {
        summary.error = "Stopped";
    }
    return summary;
}

} // namespace Metal
