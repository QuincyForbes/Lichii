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
};

enum class ResourceKind : u8 {
    Buffer,  ///< Uniform or storage buffer: [[buffer(n)]]
    Texture, ///< Sampled, separate or storage image, texel buffer: [[texture(n)]]
    Sampler, ///< [[sampler(n)]]
};

/// Where a SPIR-V (set, binding) went. A combined image sampler gets a texture and a sampler
/// entry with the same set and binding.
struct ResourceSlot {
    u32 set{};
    u32 binding{};
    ResourceKind kind{};
    u32 index{};
    u32 count{1}; ///< Slots from index on (descriptor arrays)
};

struct Options {
    bool ios{};
    u32 msl_major{3};
    u32 msl_minor{0};
    /// Vertex buffers take buffer slots from this one down (stage_in), so resources stay below.
    u32 first_vertex_buffer_slot{30};
    /// Vertex stage: Vulkan's clip space has y down, Metal's y up; flipping in the shader keeps
    /// the viewports Vulkan's.
    bool flip_vertex_y{true};
};

struct Result {
    std::string source;      ///< MSL; empty on failure
    std::string entry_point; ///< Function name in the MSL
    std::vector<ResourceSlot> slots;
    std::string error;
};

/// Resources are numbered per kind in (set, binding) order, which is the order the recompiler
/// emits them and the buffer cache appends their bindings, so the pipeline binds in sequence.
[[nodiscard]] Result SpirvToMsl(std::span<const u32> spirv, Stage stage, const Options& options);

} // namespace Metal::ShaderTranslate
