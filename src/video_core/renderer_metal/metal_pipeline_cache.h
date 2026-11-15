// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs -> shader recompiler IR -> SPIR-V -> MSL (metal_shader_translate), cached per
// pipeline key. Plain C++; Metal objects are made by metal_resources.h.

#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/container/unordered_map.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/shader_cache.h"

namespace Metal {

class Device;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

struct GraphicsPipelineKey {
    std::array<u64, 6> unique_hashes{};
    /// Two bits per generic attribute (FixedPipelineState::DynamicAttributeTypes).
    u64 attribute_types{};
    u32 topology{}; ///< Maxwell PrimitiveTopology of the draw
    u32 early_z{};
    /// Tessellation pipelines: domain | spacing << 2 | output primitives << 4 (Maxwell's).
    u32 tessellation{};
    /// Fixed state the programs are translated with, as Vulkan's MakeRuntimeInfo takes it from
    /// FixedPipelineState (EDEN_NX2_METAL_FIXED_STATE=0: all three stay zero). Bit 0: depth
    /// -1..1, bit 1: lower left window origin, bit 2: dual source blending on target 0, bits
    /// 4-7: alpha test (0 none, else Shader::CompareFunction + 1), bits 8-23: two bits per
    /// colour target (Shader::AttributeType of what the pixel program writes).
    u32 state{};
    u32 alpha_test_ref{}; ///< float bits, zero unless the test compares with it
    u32 point_size{};     ///< float bits, zero unless points are drawn

    bool operator==(const GraphicsPipelineKey&) const noexcept = default;

    [[nodiscard]] size_t Hash() const noexcept;
};

struct ComputePipelineKey {
    u64 unique_hash{};
    u32 shared_memory_size{};
    std::array<u32, 3> workgroup_size{};

    bool operator==(const ComputePipelineKey&) const noexcept = default;

    [[nodiscard]] size_t Hash() const noexcept;
};

} // namespace Metal

template <>
struct std::hash<Metal::GraphicsPipelineKey> {
    size_t operator()(const Metal::GraphicsPipelineKey& key) const noexcept {
        return key.Hash();
    }
};

template <>
struct std::hash<Metal::ComputePipelineKey> {
    size_t operator()(const Metal::ComputePipelineKey& key) const noexcept {
        return key.Hash();
    }
};

namespace Metal {

/// One translated stage: its MSL, the function, the slots its resources got, and what it uses.
struct StageProgram {
    ShaderTranslate::Stage stage{};
    std::string msl;
    std::string entry_point;
    std::vector<ShaderTranslate::ResourceSlot> slots;
    Shader::Info info;
    /// Tessellation stages: what the stage writes for the next one, and its own buffer slots.
    std::vector<ShaderTranslate::InterfaceVariable> outputs;
    ShaderTranslate::TessellationSlots tessellation_slots{};
    /// The MTLLibrary and MTLFunction, made on first use (BuildFunctions).
    Object library;
    Object function;
    bool build_failed{};
};

/// A compute kernel of the mesh emulation.
struct MeshKernel {
    StageProgram program;
    Object state;
    bool failed{};
    std::array<u32, 3> workgroup_size{};
};

/// A task/mesh pipeline drawn through Vulkan's mesh emulation: the mesh program runs as a compute
/// kernel (RuntimeInfo::mesh_as_compute) writing block_words words per workgroup to an output
/// buffer, which the pipeline's vertex shader (EmitMeshEmulationVertex) draws as a list. With a
/// task program, its kernel runs first, the distribution pass places the mesh workgroups the
/// tasks asked for in the command buffer's output arena, and both the mesh dispatch and the draw
/// are indirect.
struct MeshEmulation {
    /// The task program's kernel with a task program, else the mesh program's; its resources
    /// are the buffer cache's stage 0.
    MeshKernel first;
    /// With a task program: the mesh program's kernel (buffer cache stage mesh_stage) and the
    /// distribution pass (Shader::Backend::SPIRV::EmitMeshEmulationDistribution).
    MeshKernel mesh;
    MeshKernel distribution;
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{}; ///< payload words per task workgroup
    /// The output buffer's binding; with a task program the payload, offsets and arguments
    /// buffers follow it.
    u32 output_binding{};
    u32 block_words{};
    u32 vertices_per_group{};
    /// Maxwell PrimitiveTopology of the list: Points, Lines or Triangles.
    u32 topology{};
};

/// A tessellation pipeline as MoltenVK runs one: the vertex program as a compute kernel writing
/// its outputs to a buffer, the control program as a kernel writing control points, patch
/// constants and tessellation factors, and the evaluation program as the post-tessellation
/// vertex function (the pipeline's Vertex()) drawing the patches from those buffers.
struct Tessellation {
    /// Index sizes of the vertex kernel variants (VertexIndex reads the draw's indices).
    static constexpr std::array<u32, 3> IndexSizes{0, 2, 4};

    /// The vertex kernel by IndexSizes variant; its resources are the buffer cache's stage 0.
    std::array<StageProgram, 3> vertex;
    /// The control program's kernel; its resources are the buffer cache's stage 1.
    MeshKernel control;
    ShaderTranslate::TessellationModes modes{};
    /// Upper bounds of the bytes per element of the buffers between the stages.
    u32 vertex_stride{};
    u32 control_point_stride{};
    u32 patch_stride{};
};

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

    /// The MTLComputePipelineState, made on first use; null (logged once) when it fails.
    [[nodiscard]] void* State(const Device& device);

private:
    ComputePipelineKey key;
    StageProgram program;
    Object state;
    PipelineArchive* archive{};
    bool failed{};
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
    /// Notes why the pipeline being made is not, and logs one of the first 16 of `count`.
    void Refuse(std::atomic<u32>& count, u64 hash, std::string_view cause,
                std::string_view detail = {});

    /// Keeps the cause of the pipeline just refused for its key, and returns the key's cause.
    [[nodiscard]] std::string_view NoteRefusal(u64 key_hash);

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// A pipeline whose program 1 is a mesh program, or a task program with the mesh program at
    /// `mesh_index`, through the emulation.
    std::unique_ptr<GraphicsPipeline> CreateMeshPipeline(std::span<Shader::IR::Program> programs,
                                                         size_t mesh_index);

    /// A pipeline with tessellation control (2) and evaluation (3) programs.
    std::unique_ptr<GraphicsPipeline> CreateTessellationPipeline(
        std::span<Shader::IR::Program> programs);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineKey& key,
                                                           const VideoCommon::ShaderInfo* shader);

    [[nodiscard]] std::optional<StageProgram> Translate(Shader::IR::Program& program,
                                                        ShaderTranslate::Stage stage,
                                                        Shader::Backend::Bindings& bindings,
                                                        const Shader::RuntimeInfo& runtime_info,
                                                        const ShaderTranslate::Options* options = nullptr);

    const Device& device;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    ShaderTranslate::Options translate_options;

    ShaderPools pools;
    GraphicsPipelineKey graphics_key{};
    GraphicsPipeline* current_pipeline{};
    ::Common::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    ::Common::unordered_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;

    PipelineArchive archive;

    std::string creation_cause;
    /// By key hash; the texts stay where they are (refusal points into them).
    std::map<u64, std::string> refusal_causes;
    std::string_view refusal;

    u64 graphics_made{};
    u64 compute_made{};
    u64 refused{};
};

} // namespace Metal
