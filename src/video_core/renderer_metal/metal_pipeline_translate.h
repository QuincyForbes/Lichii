// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Pipeline keys and what they translate to: guest programs -> shader recompiler IR -> SPIR-V ->
// MSL (metal_shader_translate). The part of the pipeline cache that needs neither a device nor a
// running game, so the offline replay (metal_replay.cpp) runs a game's stored pipelines through
// the code that draws them.

#pragma once

#include <array>
#include <atomic>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal {

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

/// A task/mesh pipeline as a Metal 3 mesh pipeline (EDEN_NX2_METAL_MESH_NATIVE): the task program
/// as an object function, the mesh program as a mesh function, drawn with drawMeshThreadgroups.
/// The pipeline keeps its emulation (MeshEmulation) for when Metal refuses this one.
struct NativeMesh {
    StageProgram object; ///< Empty without a task program
    StageProgram mesh;
    /// The pixel program as it reads the mesh function's outputs (per-primitive inputs).
    StageProgram fragment;
    /// The mesh program's buffer cache stage (the task program's is 0).
    u32 mesh_stage{};
    u32 payload_bytes{};  ///< MTLMeshRenderPipelineDescriptor.payloadMemoryLength
    u32 object_threads{1};
    u32 mesh_threads{1};
    /// Maxwell PrimitiveTopology of the mesh function's primitives: Points, Lines or Triangles.
    u32 topology{};
    /// Threadgroup memory the mesh function needs, as Vulkan estimates it for MoltenVK.
    u32 threadgroup_bytes{};
    /// Metal refused the functions or the pipeline: draws go through the emulation.
    bool failed{};
};

/// EDEN_NX2_METAL_MESH_NATIVE (default on; 0: every task/mesh pipeline takes the emulation).
[[nodiscard]] bool NativeMeshEnabled();

/// EDEN_NX2_METAL_MESH_PER_PRIMITIVE=1: a mesh program that writes per-primitive generics is built
/// natively too (default: emulated, as Vulkan on MoltenVK does since IPA 150).
[[nodiscard]] bool NativeMeshPerPrimitive();

/// Metal's limits for mesh pipelines on Apple GPUs, as the native path checks them.
struct NativeMeshLimits {
    static constexpr u32 MaxVertices = 256;
    static constexpr u32 MaxPrimitives = 512;
    static constexpr u32 MaxPayloadBytes = 16384;
    static constexpr u32 MaxThreads = 1024;
    static constexpr u32 MaxThreadgroupBytes = 32768;
    /// Scalar outputs per vertex a fragment function can take.
    static constexpr u32 MaxVertexComponents = 124;
};

/// Why a mesh program (after the task program, if any) can't be a Metal mesh function, or nothing.
[[nodiscard]] std::optional<std::string> NativeMeshLimit(const Shader::IR::Program* task,
                                                         const Shader::IR::Program& mesh);

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

/// What a graphics key's programs translate to (GraphicsPipeline's stages).
struct TranslatedPipeline {
    StageProgram vertex;
    StageProgram fragment;
    std::optional<MeshEmulation> mesh;
    std::optional<Tessellation> tessellation;
    /// A task/mesh pipeline Metal draws with mesh functions; `mesh` is its fallback, when the
    /// emulation can take it.
    std::optional<NativeMesh> native_mesh;
};

class PipelineTranslator {
public:
    /// `ios`: MSL for iOS (ShaderTranslate::Options::ios). `apple_family`: the GPU's
    /// MTLGPUFamilyApple number, 0 when it has none (Options::apple_family). `native_mesh`: the
    /// device has Metal 3 mesh pipelines (and NativeMeshEnabled()).
    explicit PipelineTranslator(bool ios, u32 apple_family, bool native_mesh = false);

    /// Fills everything of the key but its programs from the registers and the draw's topology.
    static void RefreshKey(GraphicsPipelineKey& key, const Tegra::Engines::Maxwell3D::Regs& regs,
                           Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology);

    /// The key's pipeline from its programs' environments (one per program of the key, in slot
    /// order); nothing when it can't be made, and Cause() says why.
    [[nodiscard]] std::optional<TranslatedPipeline> Graphics(
        const GraphicsPipelineKey& key, std::span<Shader::Environment* const> envs);

    [[nodiscard]] std::optional<StageProgram> Compute(u64 unique_hash, Shader::Environment& env);

    /// Why the last Graphics or Compute call gave nothing, and what the failing step said.
    [[nodiscard]] std::string_view Cause() const noexcept {
        return cause;
    }

    [[nodiscard]] std::string_view Detail() const noexcept {
        return detail;
    }

    /// Why the last Graphics call's task/mesh pipeline has no NativeMesh (empty when it has one,
    /// or when it is no mesh pipeline).
    [[nodiscard]] std::string_view NativeMeshCause() const noexcept {
        return native_cause;
    }

private:
    /// The key's programs from their environments; false (Cause() set) when they can't be.
    bool TranslatePrograms(const GraphicsPipelineKey& key,
                           std::span<Shader::Environment* const> envs,
                           std::span<Shader::IR::Program> programs, size_t& mesh_index);

    /// The task and mesh programs (mesh at `mesh_index`) as Metal object and mesh functions, and
    /// the pixel program reading them; nothing (NativeMeshCause() set) when Metal's mesh stage
    /// can't take them.
    std::optional<NativeMesh> NativeMeshPipeline(const GraphicsPipelineKey& key,
                                                 std::span<Shader::IR::Program> programs,
                                                 size_t mesh_index);

    /// Notes why the pipeline being made is not, and logs one of the first 16 of `count`.
    void Refuse(std::atomic<u32>& count, u64 hash, std::string_view cause_,
                std::string_view detail_ = {});

    /// A pipeline whose program 1 is a mesh program, or a task program with the mesh program at
    /// `mesh_index`, through the emulation.
    std::optional<TranslatedPipeline> Mesh(const GraphicsPipelineKey& key,
                                           std::span<Shader::IR::Program> programs,
                                           size_t mesh_index);

    /// A pipeline with tessellation control (2) and evaluation (3) programs.
    std::optional<TranslatedPipeline> Tessellated(const GraphicsPipelineKey& key,
                                                  std::span<Shader::IR::Program> programs);

    [[nodiscard]] std::optional<StageProgram> Translate(
        Shader::IR::Program& program, ShaderTranslate::Stage stage,
        Shader::Backend::Bindings& bindings, const Shader::RuntimeInfo& runtime_info,
        const ShaderTranslate::Options* options = nullptr);

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    ShaderTranslate::Options translate_options;
    ShaderPools pools;
    std::string cause;
    std::string detail;
    std::string native_cause;
    bool native_mesh{};
};

} // namespace Metal
