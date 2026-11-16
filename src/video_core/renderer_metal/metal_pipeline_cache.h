// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs translated to MSL (metal_pipeline_translate) and their Metal pipeline states,
// cached per pipeline key. Plain C++; Metal objects are made by metal_resources.h.

#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
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
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_pipeline_files.h"
#include "video_core/renderer_metal/metal_pipeline_translate.h"
#include "video_core/shader_cache.h"

namespace Metal {

class Device;

class GraphicsPipeline {
public:
    explicit GraphicsPipeline(const GraphicsPipelineKey& key, StageProgram vertex,
                              StageProgram fragment,
                              std::optional<MeshEmulation> mesh = std::nullopt,
                              std::optional<Tessellation> tessellation = std::nullopt);

    /// The tessellation stages of a tessellation pipeline, or null.
    [[nodiscard]] Tessellation* Tess() noexcept {
        return tessellation ? &*tessellation : nullptr;
    }

    /// The vertex kernel's MTLComputePipelineState for an index size variant and the draw's
    /// vertex layout (layout's attributes and streams), made on first use; null when it fails.
    [[nodiscard]] void* VertexKernelState(const Device& device, size_t variant,
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

    /// Builds the stages' functions; false (logged once) when Metal rejects one.
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

    /// The MTLRenderPipelineState for these targets, blending and vertex layout (desc's function
    /// fields are filled in here), made on first use; null when Metal refuses it.
    [[nodiscard]] void* RenderState(const Device& device, RenderPipelineDesc& desc);

private:
    GraphicsPipelineKey key;
    StageProgram vertex;
    StageProgram fragment;
    std::optional<MeshEmulation> mesh;
    std::optional<Tessellation> tessellation;
    PipelineArchive* archive{};
    PipelineFiles* recorder{};
    bool used{};
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

    /// Writes the archive and the used pipelines list when they changed.
    void SaveFiles();

    [[nodiscard]] PipelineArchive& Archive() noexcept {
        return archive;
    }

    /// Pipelines made and refused since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 3> TakeCounts() noexcept;

    /// Why the last CurrentGraphicsPipeline or CurrentComputePipeline call gave no pipeline.
    [[nodiscard]] std::string_view Refusal() const noexcept {
        return refusal;
    }

private:
    /// Keeps the cause of the pipeline just refused for its key, and returns the key's cause.
    [[nodiscard]] std::string_view NoteRefusal(u64 key_hash);

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

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
    /// The pipelines the records file holds.
    std::unordered_set<GraphicsPipelineKey> disk_graphics;
    std::unordered_set<ComputePipelineKey> disk_compute;

    /// By key hash; the texts stay where they are (refusal points into them).
    std::map<u64, std::string> refusal_causes;
    std::string_view refusal;

    u64 graphics_made{};
    u64 compute_made{};
    u64 refused{};
};

} // namespace Metal
