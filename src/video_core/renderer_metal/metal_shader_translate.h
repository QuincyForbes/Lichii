// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPIR-V (from the shader recompiler) to Metal Shading Language with SPIRV-Cross, with the
// renderer's own binding layout. Plain C++ (no Metal needed).

#pragma once

#include <span>
#include <string>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Metal::ShaderTranslate {

enum class Stage : u8 {
    Vertex,
    Fragment,
    Compute,
    /// Tessellation, as MoltenVK runs it: the vertex program as a compute kernel writing its
    /// outputs to a buffer, the control program as a kernel over several patches per workgroup
    /// writing control points, patch constants and tessellation factors, and the evaluation
    /// program as the post-tessellation vertex function reading them from buffers.
    VertexForTessellation,
    TessellationControl,
    TessellationEvaluation,
    /// Metal 3 mesh pipelines: a task program as an object function, a mesh program as a mesh
    /// function (SPV_EXT_mesh_shader through SPIRV-Cross).
    Object,
    Mesh,
};

enum class ResourceKind : u8 {
    Buffer,  ///< Uniform or storage buffer: [[buffer(n)]]
    Texture, ///< Sampled, separate or storage image, texel buffer: [[texture(n)]]
    Sampler, ///< [[sampler(n)]]
};

/// ResourceSlot::set of the push constant block (Shader::Backend::SPIRV::RescalingLayout or
/// RenderAreaLayout, set with BindBytes).
inline constexpr u32 PushConstantSet = ~0U;

/// Where a SPIR-V (set, binding) went. A combined image sampler gets a texture and a sampler
/// entry with the same set and binding.
struct ResourceSlot {
    u32 set{};
    u32 binding{};
    ResourceKind kind{};
    u32 index{};
    u32 count{1}; ///< Slots from index on (descriptor arrays)
    /// Spilled buffers (Options::spill_buffers): the entry of the table of addresses the program
    /// reads it through; spilled samplers: the entry of the sampler table; spilled texture and
    /// sampler arrays (in_array_table): the first entry in the array table. ~0 for the others.
    u32 spill{~0U};
    /// A texture or sampler array read from the table at Result::array_spill_slot.
    bool in_array_table{};
};

/// A stage output the next tessellation stage reads from a buffer: a location, or a builtin.
struct InterfaceVariable {
    static constexpr u32 NoBuiltin = ~0U;

    u32 location{};
    u32 builtin{NoBuiltin}; ///< spv::BuiltIn
    u32 vecsize{4};
    bool per_patch{};
    /// The member's name in the MSL output struct of a vertex kernel (transform feedback reads
    /// it by name).
    std::string name;
};

/// Execution modes the tessellation control and evaluation programs share; each declares only
/// some of them, and SPIRV-Cross needs all in both.
struct TessellationModes {
    enum class Primitive : u8 { Triangles, Quads, Isolines };
    enum class Spacing : u8 { Equal, FractionalEven, FractionalOdd };

    Primitive primitive{};
    Spacing spacing{};
    bool clockwise{};
    bool point_mode{};
    u32 output_vertices{}; ///< Control points the control program writes per patch
};

/// The buffer slots of the tessellation stages' own buffers, after their resources (~0 unused).
struct TessellationSlots {
    u32 output{~0U};          ///< Vertex outputs or control points written
    u32 patch_output{~0U};    ///< Patch constants written (control)
    u32 tess_factors{~0U};    ///< MTL*TessellationFactorsHalf per patch (control)
    u32 input{~0U};           ///< Vertex outputs or control points read
    u32 patch_input{~0U};     ///< Patch constants read (evaluation)
    u32 indirect_params{~0U}; ///< Input control points per patch, patch count (control)
    u32 index{~0U};           ///< The draw's indices, for VertexIndex (vertex kernel)
};

struct Options {
    bool ios{};
    /// MTLGPUFamilyApple number of the GPU the MSL is for, 0 when it is not an Apple GPU.
    u32 apple_family{};
    u32 msl_major{3};
    u32 msl_minor{0};
    /// Vertex buffers take buffer slots from this one down (stage_in), so resources stay below.
    u32 first_vertex_buffer_slot{30};
    /// Vertex stage: Vulkan's clip space has y down, Metal's y up; flipping in the shader keeps
    /// the viewports Vulkan's.
    bool flip_vertex_y{true};
    /// Every float add, subtract and multiply through fma (SPIRV-Cross invariant_float_math), so
    /// two programs computing a position the same way get the same bits whatever the compiler
    /// fuses or reorders. Off unless set here or EDEN_NX2_METAL_INVARIANT_POSITION=1.
    bool invariant_float_math{};
    /// Tessellation stages: what the stage before writes (its Result::outputs), so the buffer
    /// between them has one layout on both sides.
    std::vector<InterfaceVariable> inputs;
    TessellationModes tessellation{};
    /// Vertex kernel of an indexed draw: its index size (2 or 4), 0 for non-indexed.
    u32 vertex_index_size{};
    /// The stage writes its point size. Metal accepts one only from draws of points (its
    /// validation aborts otherwise), so a program drawing anything else is translated without
    /// (MoltenVK's enable_point_size_builtin).
    bool point_size_builtin{true};
    /// Sampled 2D textures of vertex, fragment and compute programs are read through
    /// spvReducedSample when the sampler asks for a min or max reduction (Metal's own
    /// MTLSamplerReductionMode is ignored on some devices): the bits of each sampler in a buffer
    /// at Result::reduction_slot (SamplerReductionBits).
    bool sampler_reduction{};
    /// A vertex program with many buffers keeps the first SpillCutoff of them in buffer slots and
    /// reads the others through a table of their addresses (one buffer at Result::spill_slot, 8
    /// bytes each, in ResourceSlot::spill order), so the vertex streams find slots left. The
    /// buffers are made resident by the caller (Scheduler::UseBuffer).
    bool spill_buffers{};
    /// (set, binding) of sampled images that are bound to depth textures: they become
    /// depth2d<float>, the type Metal declares depth formats with (as the blit shaders do).
    /// A shared image type changes for all its users.
    std::vector<std::pair<u32, u32>> depth_images;
    /// Vertex stage: no point size output (Metal refuses one in a triangle or line pipeline;
    /// ReShade's vertex programs write 1).
    bool drop_point_size{};
    /// Every expression in a temporary (SPIRV-Cross force_temporary). SpirvToMsl sets it itself
    /// for a program whose inlined expressions nest past what Metal parses.
    bool force_temporary{};
};

struct Result {
    std::string source;      ///< MSL; empty on failure
    /// The MSL before the sampler reduction rewrite, when one was made: the rewrite is textual,
    /// and where Metal refuses its output the program builds from this instead (no reduction).
    std::string source_without_reduction;
    std::string entry_point; ///< Function name in the MSL
    std::vector<ResourceSlot> slots;
    /// Tessellation stages: what this stage writes for the next one.
    std::vector<InterfaceVariable> outputs;
    TessellationSlots tessellation_slots{};
    /// The 64-bit mask of sampler reduction bits (four per MSL sampler slot) the program reads,
    /// ~0 when no sample was rewritten.
    u32 reduction_slot{~0U};
    /// The table of spilled buffers' addresses (Options::spill_buffers), ~0 without spills.
    u32 spill_slot{~0U};
    /// The table of spilled samplers (past Metal's 16 sampler slots): one MTLResourceID (8 bytes)
    /// each, in ResourceSlot::spill order of the sampler slots; ~0 without spills.
    u32 sampler_spill_slot{~0U};
    /// The table of spilled texture and sampler arrays (past Metal's 128 textures or 16 samplers,
    /// e.g. a bindless array of 1020): MTLResourceIDs, 8 bytes each, ~0 without.
    u32 array_spill_slot{~0U};
    /// Its size in entries.
    u32 array_spill_entries{};
    std::string error;
};

/// An evaluation program drawn as lines or points, which Metal's tessellator can't: isolines, and
/// the point mode of triangle and quad patches. The program's post-tessellation vertex function
/// becomes a function the kernels call for every domain point (generated here from the patch's
/// tessellation factors), the results go to a buffer, and a vertex function that reads it draws
/// them as lines (a segment is two vertices) or points. The kernels' resources are the program's
/// own at the same slots, then these buffers.
struct DomainEmulation {
    std::string source;         ///< MSL with the kernels and the vertex function
    std::string count_entry;    ///< One thread: every patch's first vertex and the draw's size
    std::string evaluate_entry; ///< A threadgroup of DomainThreads per patch
    std::string vertex_entry;   ///< Vertex function (buffer 0: the evaluated vertices)
    u32 vertex_stride{};        ///< Bytes of an evaluated vertex
    u32 max_vertices_per_patch{};
    bool lines{};               ///< Else points
    /// The patch's triangles as a list of corners (three vertices a triangle, evaluated from the
    /// corners' domain points): what a geometry program reads. Else lines or points as above.
    bool triangles{};
    u32 factors_slot{};         ///< Patch factors (the Metal layout of the domain, half floats)
    u32 params_slot{};          ///< DomainParams
    u32 offsets_slot{};         ///< A u32 for each patch and one more: where its vertices start
    u32 args_slot{};            ///< Indirect arguments of the draw (4 words), written by the count
    u32 output_slot{};          ///< The evaluated vertices
    /// The program's own buffer of tessellation levels (it reads gl_TessLevel*), ~0 when unused.
    u32 level_slot{~0U};
    std::string error;
};

struct DomainParams {
    u32 patches;
    u32 capacity; ///< Evaluated vertices the output holds, all patches together
    u32 reserved[2];
};

/// Threads of the evaluation kernel's threadgroup.
inline constexpr u32 DomainThreads = 64;

/// Turns the MSL of an evaluation program (Stage::TessellationEvaluation, translated for the
/// domain, isolines as quads) into DomainEmulation. `modes` are its domain, spacing and point
/// mode; lines are drawn for isolines unless the point mode is on.
[[nodiscard]] DomainEmulation EmulateDomain(const std::string& evaluation_msl,
                                            const std::string& entry_point,
                                            const TessellationModes& modes,
                                            bool triangle_list = false);

/// Resources are numbered per kind in (set, binding) order, which is the order the recompiler
/// emits them and the buffer cache appends their bindings, so the pipeline binds in sequence.
[[nodiscard]] Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options);

/// Buffers a vertex program may keep in slots when it spills (Options::spill_buffers).
inline constexpr u32 SpillCutoff = 12;
/// Buffers from which a vertex program spills.
inline constexpr u32 SpillThreshold = 14;

/// EDEN_NX2_METAL_SAMPLER_REDUCTION (default on): Options::sampler_reduction for the guest's programs.
[[nodiscard]] bool SamplerReductionEnabled();

/// EDEN_NX2_METAL_MSL_DEVICE: the MSL uses what Options::apple_family has (default on).
[[nodiscard]] bool DeviceMslEnabled();

} // namespace Metal::ShaderTranslate
