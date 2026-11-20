// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Pipeline keys and what they translate to: guest programs -> shader recompiler IR -> SPIR-V ->
// MSL (metal_shader_translate). The part of the pipeline cache that needs neither a device nor a
// running game, so the offline replay (metal_replay.cpp) runs a game's stored pipelines through
// the code that draws them.

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
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
#include "video_core/renderer_metal/metal_geometry.h"
#include "video_core/renderer_metal/metal_object.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_metal/metal_xfb.h"
#include "video_core/stage_cache.h"

namespace Common::FS {
class IOFile;
}

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

/// GraphicsPipelineKey::tessellation: where the patch control points of a pipeline without a
/// control program are (minus one, five bits).
inline constexpr u32 TessPatchShift = 8;

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
    /// Raster state the programs are patched for (metal_raster_patch.h). Bits 0-3: the logic
    /// operation (VkLogicOp) when `logic_formats` is not zero; bits 8-19: viewport 0's swizzle
    /// when the vertex program does it.
    u32 raster{};
    u32 logic_formats{}; ///< 4 bits per colour target: RasterPatch::LogicFormat
    /// RasterExpand bits: lines as quads, stipple, smooth lines, conservative triangles.
    u32 expand{};
    u32 expand_unused{};
    /// What the draw captures into the guest's transform feedback buffers (zero: nothing).
    Xfb::Layout xfb;

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
    /// ShaderTranslate::Result::source_without_reduction: built when `msl` is refused.
    std::string msl_without_reduction;
    std::string entry_point;
    std::vector<ShaderTranslate::ResourceSlot> slots;
    Shader::Info info;
    /// Tessellation stages: what the stage writes for the next one, and its own buffer slots.
    std::vector<ShaderTranslate::InterfaceVariable> outputs;
    ShaderTranslate::TessellationSlots tessellation_slots{};
    /// Buffer slot of the sampler reduction mask (Options::sampler_reduction), ~0 without.
    u32 reduction_slot{~0U};
    /// Buffer slot of the table of spilled buffers' addresses (Options::spill_buffers), ~0 without.
    u32 spill_slot{~0U};
    /// Buffer slot of the table of spilled samplers' resource IDs, ~0 without.
    u32 sampler_spill_slot{~0U};
    /// Buffer slot of the table of spilled texture and sampler arrays, ~0 without; its entries.
    u32 array_spill_slot{~0U};
    u32 array_spill_entries{};
    /// Precomputed Metal::BindMemo kinds for this stage's descriptors.
    std::vector<u32> binding_kinds;
    /// The MTLLibrary and MTLFunction, made on first use (BuildFunctions).
    Object library;
    Object function;
    bool build_failed{};
    /// EDEN_NX2_METAL_REDUCTION_VARIANTS: `function` was built from msl_without_reduction (no
    /// sampler reduction rewrite) and `msl` is kept; the function with the rewrite is built from
    /// it for the first draw whose samplers reduce in a way Metal's sampler states don't
    /// (PipelineCache::ReducingFunction, ComputePipeline::ReducingState).
    bool reduction_variants{};
    bool reducing_failed{};
    Object reducing_library;
    Object reducing_function;
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

/// EDEN_NX2_METAL_MESH_NATIVE / EDEN_NX2_METAL_NATIVE_MESH (default on; 0: emulated).
[[nodiscard]] bool NativeMeshEnabled();

/// EDEN_NX2_METAL_MESH_PER_PRIMITIVE (default on; 0: mesh programs with per-primitive generics
/// fall back to compute emulation).
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
    /// Isolines, and the point mode of triangles and quads, which Metal's tessellator can't draw
    /// (ShaderTranslate::EmulateDomain): the kernels evaluating the domain points, and the
    /// pipeline's Vertex() reading what they wrote. The factors are those the control program
    /// writes, in the layout Metal's quad patches have for isolines.
    struct Domain {
        ShaderTranslate::DomainEmulation emulation;
        MeshKernel count;
        MeshKernel evaluate; ///< The evaluation program's resources are the buffer cache's stage 2
    };
    std::optional<Domain> domain;
    /// No control program (its kernel is empty): the vertex outputs are the control points, and
    /// the factors are written from the registers (`modes.output_vertices` is their count).
    bool factors_from_registers{};
};

/// A pipeline that captures its vertex outputs (transform feedback, metal_xfb.h): the vertex
/// program as a stage-in kernel once per index size, each with the gather kernel appended to its
/// MSL (the vertex kernel's resources are the buffer cache's stage 0, as the pipeline's vertex
/// program's).
struct XfbCapture {
    static constexpr std::array<u32, 3> IndexSizes{0, 2, 4};

    std::array<StageProgram, 3> capture;
    /// The gather kernels' MTLComputePipelineStates by variant.
    std::array<Object, 3> gather;
    /// Bytes per captured vertex between the kernels (an upper bound of the output struct).
    u32 vertex_stride{};
    std::array<u32, Xfb::MaxBuffers> stride{};
    /// The pipeline expands lines and triangles (RasterExpand): `capture` are its vertex kernels
    /// (no varyings gathered), `render` the generated vertex function that pulls their output.
    bool expand{};
    StageProgram render;
};

/// A pipeline with a geometry program, drawn through metal_geometry.h's compute path: the vertex
/// program's capture kernels (with the records kernel appended), the geometry kernel (its extra
/// storage buffers, the input records, the output records and the index regions, are the last
/// buffer slots of its program), and the plan of the buffers' layouts. The pipeline's own vertex
/// program is the reader.
struct GeometryCompute {
    static constexpr std::array<u32, 3> IndexSizes{0, 2, 4};

    Geometry::Plan plan;
    std::array<StageProgram, 3> capture;
    std::array<Object, 3> records;
    MeshKernel kernel;
    /// The buffer slots (ShaderStage::Compute) of the geometry kernel's extra buffers.
    std::array<u32, Geometry::ExtraBuffers> extra_slots{};
    /// Bytes per captured vertex between the capture and records kernels (an upper bound).
    u32 vertex_stride{};
    /// Transform feedback of the geometry output: the strides of the buffers it captures into
    /// (all zero without), the kernel is Geometry::XfbEntry in the geometry kernel's MSL.
    std::array<u32, Xfb::MaxBuffers> xfb_stride{};
    /// Which streams (bit s) have buffers; the kernel of stream s is Geometry::XfbEntryOf(s).
    u32 xfb_streams{};
    std::array<Object, Geometry::MaxStreams> xfb_states;
    /// A geometry program after tessellation: the stages before it are in the pipeline's
    /// `tessellation` (its domain emulation lists triangles, lines or points), and the records
    /// kernel reads the evaluated vertices of that emulation (`tess_records`, `geo_records` in
    /// its MSL) instead of a vertex capture kernel's.
    bool from_tess{};
    MeshKernel tess_records;
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
    /// A vertex pipeline whose key has a transform feedback layout.
    std::optional<XfbCapture> xfb;
    /// A pipeline with a geometry program (its `vertex` is the reader).
    std::optional<GeometryCompute> geometry;
};

/// Persistent zstd-compressed, header-indexed disk cache of SPIR-V -> MSL translations
/// (<ShaderDir>/<title_id>/metal_stages.v2.bin). Thread-safe across translation, boot and offline
/// warm workers.
class StageMslCache {
public:
    StageMslCache() = default;

    void Open(const std::filesystem::path& file_path, u64 build_stamp);
    void Flush();
    void Save() {
        Flush();
    }

    [[nodiscard]] std::optional<ShaderTranslate::Result> Find(const u128& key) const;
    void Insert(const u128& key, const ShaderTranslate::Result& result);

    [[nodiscard]] size_t Size() const;
    [[nodiscard]] size_t PendingSize() const;

    [[nodiscard]] static bool HasValidEntries(const std::filesystem::path& file_path,
                                              u64 build_stamp);

private:
    struct DiskSlice {
        u64 offset{};
        u32 comp_size{};
        u32 uncomp_size{};
    };

    struct KeyHasher {
        size_t operator()(const u128& k) const noexcept {
            return static_cast<size_t>(k[0] ^ (k[1] * 0x9E3779B97F4A7C15ULL));
        }
    };

    mutable std::mutex save_mutex;
    mutable std::shared_mutex mutex;
    std::filesystem::path path;
    u64 stamp{};
    u64 valid_file_size{};
    bool open{};
    std::shared_ptr<::Common::FS::IOFile> read_file;
    std::unordered_map<u128, DiskSlice, KeyHasher> disk_index;
    std::unordered_map<u128, std::shared_ptr<const ShaderTranslate::Result>, KeyHasher>
        pending_entries;
};

class PipelineTranslator {
public:
    /// `ios`: MSL for iOS (ShaderTranslate::Options::ios). `apple_family`: the GPU's
    /// MTLGPUFamilyApple number, 0 when it has none (Options::apple_family). `native_mesh`: the
    /// device has Metal 3 mesh pipelines (and NativeMeshEnabled()).
    explicit PipelineTranslator(bool ios, u32 apple_family, bool native_mesh = false,
                                VideoCommon::StageCache* stage_cache = nullptr,
                                StageMslCache* msl_cache = nullptr);

    void SetCaches(VideoCommon::StageCache* stage_cache_, StageMslCache* msl_cache_) noexcept {
        stage_cache = stage_cache_;
        msl_cache = msl_cache_;
    }

    /// Fills everything of the key but its programs from the registers and the draw's topology.
    /// Programs EDEN_NX2_OOT_VISUAL_WORKAROUND edited so far (the offline replay's report).
    [[nodiscard]] static u64 VisualWorkaroundEdits() noexcept;

    static void RefreshKey(GraphicsPipelineKey& key, const Tegra::Engines::Maxwell3D::Regs& regs,
                           Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology);

    /// The key's pipeline from its programs' environments (one per program of the key, in slot
    /// order); nothing when it can't be made, and Cause() says why.
    [[nodiscard]] std::optional<TranslatedPipeline> Graphics(
        const GraphicsPipelineKey& key, std::span<Shader::Environment* const> envs);

    [[nodiscard]] std::optional<StageProgram> Compute(u64 unique_hash, Shader::Environment& env);

    /// Single-stage vertex (slot 1) or fragment (slot 5) translation for `warm-graphics.bin`,
    /// populating StageCache and StageMslCache (`metal_stages.v2.bin`).
    [[nodiscard]] std::optional<StageProgram> WarmSingleStage(const GraphicsPipelineKey& key,
                                                              size_t slot,
                                                              Shader::Environment& env);

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

public:
    /// A guest geometry passthrough program that only sets the layer or the viewport index from a
    /// vertex output it passes on (what Maxwell's passthrough does for cube and cascade targets),
    /// folded into the vertex program: that program sets it from the value it stored there.
    /// False, nothing changed, for anything else (the geometry stage is refused then).
    /// A pipeline with a geometry program (4) through the compute path; `programs` are the key's
    /// translated programs (the smoke test makes them by hand).
    [[nodiscard]] std::optional<TranslatedPipeline> GeometryPipeline(
        const GraphicsPipelineKey& key, std::span<Shader::IR::Program> programs);

    /// A transformed geometry program (Geometry::Transform) as its compute kernel.
    [[nodiscard]] std::optional<StageProgram> GeometryKernel(Shader::IR::Program& program);

    [[nodiscard]] static bool FoldGeometryPassthrough(Shader::IR::Program& vertex,
                                                      const Shader::IR::Program& geometry);

private:
    /// A pipeline with tessellation control (2) and evaluation (3) programs.
    /// `feeds_geometry`: a geometry program follows (4): the domain is always emulated, as a list
    /// of triangles, lines or points, no pixel program is made, the evaluation's outputs stay
    /// those a geometry program reads (no flip, no depth conversion: the reader does them), and
    /// `feed_bindings` is where the next stage's bindings continue.
    std::optional<TranslatedPipeline> Tessellated(const GraphicsPipelineKey& key,
                                                  std::span<Shader::IR::Program> programs,
                                                  bool feeds_geometry = false);
    Shader::Backend::Bindings feed_bindings;

    /// The raster state of `key` done in the program's MSL (logic operation, viewport swizzle).
    void PatchRasterState(std::string& msl, ShaderTranslate::Stage stage,
                          const GraphicsPipelineKey& key);

    [[nodiscard]] std::optional<StageProgram> Translate(
        Shader::IR::Program& program, ShaderTranslate::Stage stage,
        Shader::Backend::Bindings& bindings, const Shader::RuntimeInfo& runtime_info,
        const ShaderTranslate::Options* options = nullptr);

    [[nodiscard]] std::optional<StageProgram> TranslateCached(
        size_t slot, Shader::IR::Program& program, ShaderTranslate::Stage stage,
        Shader::Backend::Bindings& bindings, const Shader::RuntimeInfo& runtime_info,
        const ShaderTranslate::Options* options = nullptr);

    struct StageReuse {
        using Cache = VideoCommon::StageCache;
        std::array<std::shared_ptr<const Cache::Translation>, 6> reused{};
        std::array<std::shared_ptr<Cache::Translation>, 6> recorded{};
        std::vector<std::pair<std::shared_ptr<Cache::Translation>,
                              std::shared_ptr<const Cache::Emitted>>>
            new_translations;
    } stage_reuse;

    /// The key Graphics is translating: its raster state patches the MSL Translate makes.
    const GraphicsPipelineKey* patch_key{};
    /// The pipeline being translated is expanded: its pixel program reads the line coordinate.
    bool expand_active{};
    /// A stage's SPIR-V as MSL; nothing (Cause() set) when SPIRV-Cross fails.
    [[nodiscard]] std::optional<StageProgram> FromSpirv(std::span<const u32> spirv,
                                                        const Shader::IR::Program& program,
                                                        ShaderTranslate::Stage stage,
                                                        const ShaderTranslate::Options& options);

    /// The capture kernels of a vertex pipeline whose key has a transform feedback layout.
    [[nodiscard]] std::optional<XfbCapture> XfbPrograms(const GraphicsPipelineKey& key,
                                                        std::span<const u32> spirv,
                                                        const Shader::IR::Program& vertex);

    VideoCommon::StageCache* stage_cache{};
    StageMslCache* msl_cache{};
    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;
    ShaderTranslate::Options translate_options;
    ShaderPools pools;
    std::string cause;
    std::string detail;
    std::string native_cause;
    bool native_mesh{};
    /// The device has Metal 3 mesh pipelines (whatever EDEN_NX2_METAL_MESH_NATIVE says).
    bool native_mesh_capable{};
    /// TranslatePrograms folded the key's geometry program into the vertex program.
    bool folded_geometry{};
};

} // namespace Metal
