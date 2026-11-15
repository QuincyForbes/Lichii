// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPIR-V (from the shader recompiler) to Metal Shading Language with SPIRV-Cross, with the
// renderer's own binding layout. Plain C++ (no Metal needed).

#pragma once

#include <span>
#include <string>
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
};

/// A stage output the next tessellation stage reads from a buffer: a location, or a builtin.
struct InterfaceVariable {
    static constexpr u32 NoBuiltin = ~0U;

    u32 location{};
    u32 builtin{NoBuiltin}; ///< spv::BuiltIn
    u32 vecsize{4};
    bool per_patch{};
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
    /// Tessellation stages: what the stage before writes (its Result::outputs), so the buffer
    /// between them has one layout on both sides.
    std::vector<InterfaceVariable> inputs;
    TessellationModes tessellation{};
    /// Vertex kernel of an indexed draw: its index size (2 or 4), 0 for non-indexed.
    u32 vertex_index_size{};
};

struct Result {
    std::string source;      ///< MSL; empty on failure
    std::string entry_point; ///< Function name in the MSL
    std::vector<ResourceSlot> slots;
    /// Tessellation stages: what this stage writes for the next one.
    std::vector<InterfaceVariable> outputs;
    TessellationSlots tessellation_slots{};
    std::string error;
};

/// Resources are numbered per kind in (set, binding) order, which is the order the recompiler
/// emits them and the buffer cache appends their bindings, so the pipeline binds in sequence.
[[nodiscard]] Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options);

/// EDEN_NX2_METAL_MSL_DEVICE: the MSL uses what Options::apple_family has (default on).
[[nodiscard]] bool DeviceMslEnabled();

} // namespace Metal::ShaderTranslate
