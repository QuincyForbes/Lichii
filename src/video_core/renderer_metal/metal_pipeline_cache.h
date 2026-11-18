// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs translated to MSL (metal_pipeline_translate) and their Metal pipeline states,
// cached per pipeline key. Plain C++; Metal objects are made by metal_resources.h.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "common/polyfill_thread.h"
#include "common/thread_worker.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_build_job.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_pipeline_files.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"

namespace Metal {

class Device;

/// EDEN_NX2_METAL_ASYNC_PIPELINES: the builder threads that translate and build pipelines first
/// seen in a run (default logical CPUs - 1); 0 builds them on the draw's thread as before.
[[nodiscard]] u32 AsyncPipelineThreads();

/// A pipeline build handed to the builder threads (metal_build_job.h).
using PipelineBuildJob = BuildJob<PipelineTranslator>;

class GraphicsPipeline {
public:
    explicit GraphicsPipeline(const GraphicsPipelineKey& key, StageProgram vertex,
                              StageProgram fragment,
                              std::optional<MeshEmulation> mesh = std::nullopt,
                              std::optional<Tessellation> tessellation = std::nullopt,
                              std::optional<NativeMesh> native_mesh = std::nullopt,
                              std::optional<XfbCapture> xfb = std::nullopt,
                              std::optional<GeometryCompute> geometry = std::nullopt);

    /// The Metal mesh functions of a task/mesh pipeline while Metal takes them, else null (the
    /// pipeline draws through Mesh(), when it has one).
    [[nodiscard]] NativeMesh* Native() noexcept {
        return native_mesh && !native_mesh->failed ? &*native_mesh : nullptr;
    }

    /// A task/mesh pipeline, drawn natively or emulated.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return mesh.has_value() || native_mesh.has_value();
    }

    /// Native() is null from now on: its functions or a render state were refused by Metal.
    void FailNative(std::string_view reason);

    /// The tessellation stages of a tessellation pipeline, or null.
    [[nodiscard]] Tessellation* Tess() noexcept {
        return tessellation ? &*tessellation : nullptr;
    }

    /// The vertex kernel's MTLComputePipelineState for an index size variant and the draw's
    /// vertex layout (layout's attributes and streams), made on first use; null when it fails.
    [[nodiscard]] void* VertexKernelState(const Device& device, size_t variant,
                                          const RenderPipelineDesc& layout);

    /// The transform feedback capture of a pipeline whose key has a layout, or null.
    [[nodiscard]] XfbCapture* Xfb() noexcept {
        return xfb && !xfb->expand ? &*xfb : nullptr;
    }

    /// The line and triangle expansion of a pipeline that has one (its capture kernels are
    /// XfbKernelStates' with no gathering), or null.
    [[nodiscard]] XfbCapture* Expand() noexcept {
        return xfb && xfb->expand && !xfb->render.build_failed ? &*xfb : nullptr;
    }

    /// The capture kernel's MTLComputePipelineState for an index size variant and the draw's
    /// vertex layout, and the gather kernel's; both null when either fails.
    [[nodiscard]] std::pair<void*, void*> XfbKernelStates(const Device& device, size_t variant,
                                                          const RenderPipelineDesc& layout);

    /// The compute path of a pipeline with a geometry program, or null.
    [[nodiscard]] GeometryCompute* Geo() noexcept {
        return geometry ? &*geometry : nullptr;
    }

    /// The vertex capture kernel's state for an index size variant and the draw's vertex layout,
    /// and the records kernel's (appended to its MSL); both null when either fails.
    [[nodiscard]] std::pair<void*, void*> GeometryKernelStates(const Device& device, size_t variant,
                                                               const RenderPipelineDesc& layout);

    /// The mesh emulation of a task/mesh pipeline, or null.
    [[nodiscard]] MeshEmulation* Mesh() noexcept {
        return mesh ? &*mesh : nullptr;
    }

    /// A mesh emulation kernel's MTLComputePipelineState, made on first use; null when it fails.
    [[nodiscard]] void* KernelState(const Device& device, MeshKernel& kernel);

    [[nodiscard]] const GraphicsPipelineKey& Key() const noexcept {
        return key;
    }

    [[nodiscard]] StageProgram& Vertex() noexcept {
        return vertex;
    }

    [[nodiscard]] StageProgram& Fragment() noexcept {
        return fragment;
    }

    /// Whether the pixel program reads an input without interpolation (the provoking vertex's).
    [[nodiscard]] bool HasFlatInputs() const noexcept {
        return flat_inputs;
    }

    /// Builds the stages' functions (Native()'s while it has them); false (logged once) when Metal
    /// rejects one.
    bool BuildFunctions(const Device& device);

    /// The archive pipeline states are looked up in and added to (PipelineCache's).
    void SetArchive(PipelineArchive* archive_) noexcept {
        archive = archive_;
    }

    /// Where the render states Metal makes are written for the next boot, or null.
    void SetRecorder(PipelineFiles* recorder_) noexcept {
        recorder = recorder_;
    }

    /// Marks the pipeline as used by this run; whether it was already.
    bool MarkUsed() noexcept {
        return std::exchange(used, true);
    }

    [[nodiscard]] bool WasUsed() const noexcept {
        return used;
    }

    /// The frame of the last draw that took it (pipeline eviction).
    void SetLastUse(u64 frame) noexcept {
        last_use = frame;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use;
    }

    /// The MTLRenderPipelineState for these targets, blending and vertex layout (desc's function
    /// fields are filled in here), made on first use; null when Metal refuses it. While Native()
    /// is there, its mesh pipeline state; when Metal refuses that, Native() becomes null.
    [[nodiscard]] void* RenderState(const Device& device, RenderPipelineDesc& desc);

    /// Fills desc's function fields; the hash render states are kept by.
    [[nodiscard]] u64 StateHash(RenderPipelineDesc& desc) const noexcept;

    /// The state made for `hash`, if one was asked for (null when Metal refused it).
    [[nodiscard]] std::optional<void*> FindState(u64 hash) const;

    /// Keeps a state made elsewhere (a builder thread) for `hash`, logged and recorded as one made
    /// here; returns it. A refused `mesh_state` is not kept and makes Native() null.
    void* AdoptState(u64 hash, const RenderPipelineDesc& desc, Object state,
                     std::string_view error, bool mesh_state);

    /// Native()'s functions for a mesh pipeline state, none without Native().
    [[nodiscard]] std::optional<MeshFunctions> NativeFunctions() const noexcept;

private:
    bool BuildNativeFunctions(const Device& device);

    GraphicsPipelineKey key;
    StageProgram vertex;
    StageProgram fragment;
    std::optional<MeshEmulation> mesh;
    std::optional<Tessellation> tessellation;
    std::optional<NativeMesh> native_mesh;
    std::optional<XfbCapture> xfb;
    std::optional<GeometryCompute> geometry;
    PipelineArchive* archive{};
    PipelineFiles* recorder{};
    bool used{};
    bool flat_inputs{};
    u64 last_use{};
    ::Common::unordered_map<u64, Object> render_states;
    /// Vertex kernel states by variant and vertex layout.
    ::Common::unordered_map<u64, Object> vertex_kernel_states;
};

class ComputePipeline {
public:
    explicit ComputePipeline(const ComputePipelineKey& key, StageProgram program);

    [[nodiscard]] StageProgram& Program() noexcept {
        return program;
    }

    [[nodiscard]] const ComputePipelineKey& Key() const noexcept {
        return key;
    }

    void SetArchive(PipelineArchive* archive_) noexcept {
        archive = archive_;
    }

    bool MarkUsed() noexcept {
        return std::exchange(used, true);
    }

    /// The MTLComputePipelineState, made on first use; null (logged once) when it fails.
    [[nodiscard]] void* State(const Device& device);

private:
    ComputePipelineKey key;
    StageProgram program;
    Object state;
    PipelineArchive* archive{};
    bool failed{};
    bool used{};
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory, const Device& device);
    ~PipelineCache();

    /// The pipeline of the current draw, or null when it can't be made (unsupported stages,
    /// translation errors).
    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    /// The programs of the last graphics lookup (the feature census).
    [[nodiscard]] const std::array<u64, 6>& LookupHashes() const noexcept {
        return graphics_key.unique_hashes;
    }

    /// Opens the game's pipeline archive (EDEN_NX2_METAL_ARCHIVE=0: none).
    void OpenArchive(u64 title_id);

    /// Opens the archive and builds the pipelines earlier runs recorded, with the render states
    /// they were drawn with, on worker threads (EDEN_NX2_METAL_PIPELINE_FILE=0: archive only).
    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    /// Once a frame: the used pipelines list is written within EDEN_NX2_USED_PIPELINES_SAVE_S
    /// (default 10) of a pipeline's first use and when the app leaves the foreground, the archive
    /// once a minute and then.
    void TickFrame();

    [[nodiscard]] PipelineArchive& Archive() noexcept {
        return archive;
    }

    /// Pipelines made and refused since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 3> TakeCounts() noexcept;

    /// Why the last CurrentGraphicsPipeline or CurrentComputePipeline call gave no pipeline.
    [[nodiscard]] std::string_view Refusal() const noexcept {
        return refusal;
    }

    /// Whether the last CurrentGraphicsPipeline call gave no pipeline because it is still being
    /// built and the draw was skipped.
    [[nodiscard]] bool Building() const noexcept {
        return building;
    }

    /// Asked once per lookup of a pipeline or render state that is not built yet: whether the draw
    /// may be skipped instead of waiting (the rasterizer's DrawMayBeSkipped). Unset: never.
    void SetSkipCheck(std::function<bool()> check) {
        skip_check = std::move(check);
    }

    /// The pipeline's render state for `desc`, built on the builder threads when it is new
    /// (EDEN_NX2_METAL_ASYNC_PIPELINES): null and `building` while a draw that may be skipped
    /// waits for it, else built or waited for here. Null without `building`: Metal refused it.
    [[nodiscard]] void* RenderState(GraphicsPipeline& pipeline, RenderPipelineDesc& desc,
                                    bool& building);

    /// The 10 s "Perf Metal pipelines:" line, empty when nothing was built or waited for.
    [[nodiscard]] std::string TakeBuildLine();

private:
    struct GraphicsBuild {
        GraphicsPipelineKey key;
        std::array<VideoCommon::GraphicsEnvironment, 6> envs;
        std::unique_ptr<GraphicsPipeline> pipeline;
        std::string cause;
        std::chrono::steady_clock::time_point queued;
        std::shared_ptr<PipelineBuildJob> job;
    };

    struct StateBuild {
        RenderPipelineDesc desc;
        std::optional<MeshFunctions> native; ///< A mesh pipeline state
        Object state;
        std::string error;
        std::shared_ptr<PipelineBuildJob> job;
    };

    struct BuildStats {
        u64 queued{};
        u64 states_queued{};
        u64 adopted{};
        u64 stale{};
        u64 skipped{};
        u64 held{};
        u64 held_here{};
        u64 boosted{};
        u64 held_ns{};
        u64 held_max_ns{};
        u64 ready_ns{};
        u64 ready_max_ns{};
    };

    /// Keeps the cause of the pipeline just refused for its key, and returns the key's cause.
    [[nodiscard]] std::string_view NoteRefusal(u64 key_hash);

    /// The skip check's answer for this lookup.
    [[nodiscard]] bool MayBeSkipped() const {
        return skip_check && skip_check();
    }

    /// Waits for a job as a held draw does, counted in the build line.
    void Hold(PipelineBuildJob& job);

    /// The current key's pipeline through the builder threads (async lookups).
    [[nodiscard]] std::unique_ptr<GraphicsPipeline> BuildInBackground(bool& pending);

    /// A finished build of the current key as the cache's pipeline; null with `retry` when its
    /// texture reads no longer match the guest's descriptors.
    [[nodiscard]] std::unique_ptr<GraphicsPipeline> AdoptBuild(GraphicsBuild& build, bool& retry);

    /// Queues the translation and build of the current key on the builder threads.
    std::shared_ptr<GraphicsBuild> QueueGraphicsBuild();

    /// The records file entry of a pipeline made from `envs` (once per key).
    void StoreRecord(const GraphicsPipelineKey& key,
                     const std::array<VideoCommon::GraphicsEnvironment, 6>& envs);

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// Vulkan's EvictUnusedPipelines on Apple, from TickFrame (EDEN_NX2_METAL_PIPELINE_EVICT):
    /// on a memory pressure signal, graphics pipelines no draw took for a while are destroyed
    /// (their command buffers keep what they reference); EDEN_NX2_PIPELINE_EVICT_LEFT_MIB and
    /// _MAX as on Vulkan.
    void EvictUnusedPipelines();

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineKey& key,
                                                           const VideoCommon::ShaderInfo* shader);

    const Device& device;

    PipelineTranslator translator;
    GraphicsPipelineKey graphics_key{};
    GraphicsPipeline* current_pipeline{};
    ::Common::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    ::Common::unordered_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;

    PipelineArchive archive;
    PipelineFiles files;
    std::chrono::steady_clock::time_point archive_saved{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point used_saved{std::chrono::steady_clock::now()};
    std::size_t background_listener{};
    /// The app is leaving the foreground: TickFrame writes the used list at once.
    std::atomic_bool save_requested{};
    /// The pipelines the records file holds.
    std::unordered_set<GraphicsPipelineKey> disk_graphics;
    std::unordered_set<ComputePipelineKey> disk_compute;

    /// By key hash; the texts stay where they are (refusal points into them).
    std::map<u64, std::string> refusal_causes;
    std::string_view refusal;

    u64 graphics_made{};
    u64 compute_made{};
    u64 refused{};

    /// Frames ended (TickFrame), and the memory pressure signal eviction last answered.
    u64 frame_count{};
    u32 evict_pressure_generation{};
    u64 pressure_evictions_skipped{};
    std::function<bool()> skip_check;
    bool building{};
    BuildStats build_stats;
    /// Builds on the builder threads by key, until a lookup adopts them (draw's thread only).
    ::Common::unordered_map<GraphicsPipelineKey, std::shared_ptr<GraphicsBuild>> graphics_builds;
    /// Render states on the builder threads by pipeline and state hash (draw's thread only).
    std::map<std::pair<const GraphicsPipeline*, u64>, std::shared_ptr<StateBuild>> state_builds;
    /// Last: stopped before the pipelines and the archive its tasks use go.
    std::unique_ptr<::Common::StatefulThreadWorker<PipelineTranslator>> builders;
};

} // namespace Metal
