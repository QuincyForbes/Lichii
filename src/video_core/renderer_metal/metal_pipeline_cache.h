// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Guest programs -> shader recompiler IR -> SPIR-V -> MSL (metal_shader_translate), cached per
// pipeline key. Plain C++; Metal objects are made by metal_resources.h.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <string>
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

class GraphicsPipeline {
public:
    explicit GraphicsPipeline(const GraphicsPipelineKey& key, StageProgram vertex,
                              StageProgram fragment,
                              std::optional<MeshEmulation> mesh = std::nullopt);

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

    /// The MTLRenderPipelineState for these targets, blending and vertex layout (desc's function
    /// fields are filled in here), made on first use; null when Metal refuses it.
    [[nodiscard]] void* RenderState(const Device& device, RenderPipelineDesc& desc);

private:
    GraphicsPipelineKey key;
    StageProgram vertex;
    StageProgram fragment;
    std::optional<MeshEmulation> mesh;
    ::Common::unordered_map<u64, Object> render_states;
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

    /// The MTLComputePipelineState, made on first use; null (logged once) when it fails.
    [[nodiscard]] void* State(const Device& device);

private:
    ComputePipelineKey key;
    StageProgram program;
    Object state;
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

    /// Pipelines made and refused since the last call (the renderer's 10 s report).
    [[nodiscard]] std::array<u64, 3> TakeCounts() noexcept;

private:
    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    /// A pipeline whose program 1 is a mesh program, or a task program with the mesh program at
    /// `mesh_index`, through the emulation.
    std::unique_ptr<GraphicsPipeline> CreateMeshPipeline(std::span<Shader::IR::Program> programs,
                                                         size_t mesh_index);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineKey& key,
                                                           const VideoCommon::ShaderInfo* shader);

    [[nodiscard]] std::optional<StageProgram> Translate(Shader::IR::Program& program,
                                                        ShaderTranslate::Stage stage,
                                                        Shader::Backend::Bindings& bindings,
                                                        const Shader::RuntimeInfo& runtime_info);

    const Device& device;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    ShaderTranslate::Options translate_options;

    ShaderPools pools;
    GraphicsPipelineKey graphics_key{};
    GraphicsPipeline* current_pipeline{};
    ::Common::unordered_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;
    ::Common::unordered_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_cache;

    u64 graphics_made{};
    u64 compute_made{};
    u64 refused{};
};

} // namespace Metal
