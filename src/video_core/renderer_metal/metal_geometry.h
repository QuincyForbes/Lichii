// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Geometry programs for the Metal renderer, which has no geometry stage: nxnative's
// geometry-to-compute (github.com/seedds/nxnative, MIT License, Copyright (c) 2026 seedds:
// docs/geometry-metal-research.md, the vendored Ryujinx GeometryToCompute transform, and the
// ordered compaction of dm/gfx/msl/geo_pack.metal, ported below), on this recompiler's IR:
//
//  1. the vertex program runs as a stage-in kernel (the transform feedback capture kernel) and a
//     kernel appended to its MSL copies the attributes the geometry program reads into records;
//  2. the geometry program, rewritten to a compute program (Transform), runs one thread per input
//     primitive: it reads its vertices' records, keeps the vertex it is building in local memory,
//     and EmitVertex / EndPrimitive write the vertex record and an index (a restart marker for a
//     strip's end) into the thread's own region, so no atomic cursor decides the order;
//  3. the compaction kernels turn the regions' strips into one ordered list and the indirect
//     arguments of its draw (the order of the guest's strips, the winding and the last vertex of
//     each primitive kept);
//  4. a reader vertex function draws that list from the records with the pixel program.
//
// Plain C++.

#pragma once

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace Metal::Geometry {

/// Threads (input primitives) per threadgroup of the geometry kernel.
inline constexpr u32 GroupSize = 64;
/// The geometry kernel's extra storage buffers, after the program's own.
inline constexpr u32 ExtraBuffers = 3;
/// Most bytes one draw's records and indices take (a bigger one is refused).
inline constexpr u64 MaxScratchBytes = u64{192} << 20;

/// What the rewritten program reads and writes, and how its regions are laid out.
struct Plan {
    /// Records (vertices) of an input primitive, and how far its first vertex moves from one
    /// primitive to the next in the draw (the primitive's vertices for a list, 1 for a strip).
    u32 input_vertices{};
    u32 input_step{};
    /// Invocations of the program a primitive (the workgroup grid's second dimension).
    u32 invocations{1};
    /// The attribute words (Shader::IR::Attribute) of an input record, and of an output record.
    std::vector<u32> input_words;
    std::vector<u32> output_words;
    /// Declared output vertices a primitive; index slots a primitive (a vertex or a restart each).
    u32 max_vertices{};
    u32 index_slots{};
    /// Compaction topology: 1 points, 6 lines, 3 triangles (what the strips are cut into).
    u32 output_topology{};

    [[nodiscard]] u32 InputStride() const noexcept {
        return static_cast<u32>(input_words.size());
    }
    [[nodiscard]] u32 OutputStride() const noexcept {
        return static_cast<u32>(output_words.size());
    }
    /// Words of a primitive's index region: its count, then the slots.
    [[nodiscard]] u32 IndexStride() const noexcept {
        return index_slots + 1;
    }
    /// Input primitives of `count` vertices of one instance.
    [[nodiscard]] u32 Primitives(u32 count) const noexcept {
        return count < input_vertices ? 0U : (count - input_vertices) / input_step + 1;
    }
};

/// The records of an input primitive for a draw topology: `vertices` of them, the first moving
/// `step` a primitive. Triangle strips with adjacency (their vertex order alternates) are not
/// taken.
[[nodiscard]] inline bool InputTopology(Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology,
                                        u32& vertices, u32& step) {
    using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    switch (topology) {
    case Topology::Points:
        vertices = step = 1;
        return true;
    case Topology::Lines:
        vertices = step = 2;
        return true;
    case Topology::LineStrip:
        vertices = 2;
        step = 1;
        return true;
    case Topology::Triangles:
        vertices = step = 3;
        return true;
    case Topology::TriangleStrip:
        vertices = 3;
        step = 1;
        return true;
    case Topology::LinesAdjacency:
        vertices = step = 4;
        return true;
    case Topology::LineStripAdjacency:
        vertices = 4;
        step = 1;
        return true;
    case Topology::TrianglesAdjacency:
        vertices = step = 6;
        return true;
    default:
        return false;
    }
}

/// Rewrites a geometry program (points, lines, triangles or their adjacency in, points, line
/// strips or triangle strips out, position, point size, layer, viewport index, clip distances and
/// generic attributes; output streams past 0 are dropped) into a compute
/// program in place; its extra storage buffers (input records, output records, indices) are the
/// last descriptors. Nothing is changed and `why` says so when it can't.
[[nodiscard]] std::optional<Plan> Transform(Shader::IR::Program& program, u32 input_vertices,
                                            u32 input_step, std::string& why);

/// The kernel appended to the MSL of the vertex capture kernel `entry`: record r (vertex r % k of
/// primitive r / k of the draw, instances after one another) takes the attribute words
/// (plan.input_words) of captured vertex instance * count + primitive * step + r % k. Buffers: 0
/// the vertex outputs, 1 the records, 2 RecordsParams.
inline constexpr std::string_view RecordsEntry = "geo_records";
inline constexpr u32 RecordsSourceSlot = 0;
inline constexpr u32 RecordsOutputSlot = 1;
inline constexpr u32 RecordsCountSlot = 2;
struct RecordsParams {
    u32 records;    ///< Records to write
    u32 vertices;   ///< k
    u32 step;
    u32 count;      ///< Vertices of one instance
    u32 primitives; ///< Primitives of one instance
};
[[nodiscard]] std::string RecordsSource(
    const Plan& plan, std::string_view entry,
    std::span<const ShaderTranslate::InterfaceVariable> outputs,
    const std::map<Shader::IR::Attribute, Shader::IR::Attribute>& legacy_mapping);

/// The reader vertex function: vertex `vertex_id` is record vertex_id of buffer 0; its outputs
/// are what the pixel program reads (`generics`, the generic attributes the geometry program
/// writes). `convert_depth`: the guest's depth range is -1..1; `flip_y`: Vulkan's clip space.
inline constexpr std::string_view ReaderEntry = "geo_reader";
[[nodiscard]] std::string ReaderSource(const Plan& plan, bool convert_depth, bool flip_y,
                                       float default_point_size);

/// The compaction kernels (nxnative's geo_pack_count, geo_pack_scan_legacy and geo_pack_write):
/// buffers 0 the index regions, 1 the list, 2 the draw's indirect arguments, 3 the per-primitive
/// counts (a primitive's offset after the scan), 4 the parameters (GeometryParams).
inline constexpr std::string_view PackSource = R"(#include <metal_stdlib>
using namespace metal;
// Ported from nxnative dm/gfx/msl/geo_pack.metal (MIT License, Copyright (c) 2026 seedds).
struct GeometryParams { uint groups, stride, topology, capacity; };
static bool geo_room(constant GeometryParams& j, uint o, uint k, uint n) {
    return o <= j.capacity && k <= j.capacity - o && n <= j.capacity - o - k;
}
static uint geo_group(device const uint* src, device uint* dst, constant GeometryParams& j, uint g,
                      bool w, uint o) {
    uint a = 0, b = 0, n = 0, k = 0;
    const uint count = min(src[g * j.stride], j.stride - 1);
    for (uint i = 0; i < count; i++) {
        const uint v = src[g * j.stride + 1 + i];
        if (v == 0xffffffffu) { n = 0; continue; }
        if (j.topology == 1) {
            if (w && geo_room(j, o, k, 1)) dst[o + k] = v;
            k += 1;
        } else if (j.topology == 6) {
            if (n) {
                if (w && geo_room(j, o, k, 2)) { dst[o + k] = b; dst[o + k + 1] = v; }
                k += 2;
            }
        } else if (n >= 2) {
            if (w && geo_room(j, o, k, 3)) {
                dst[o + k] = (n & 1) ? b : a;
                dst[o + k + 1] = (n & 1) ? a : b;
                dst[o + k + 2] = v;
            }
            k += 3;
        }
        a = b; b = v; n++;
    }
    return k;
}
kernel void geo_count(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                      device uint* args [[buffer(2)]], device uint* cnt [[buffer(3)]],
                      constant GeometryParams& j [[buffer(4)]], uint g [[thread_position_in_grid]]) {
    if (g < j.groups) cnt[g] = geo_group(src, dst, j, g, false, 0);
}
kernel void geo_scan(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                     device uint* args [[buffer(2)]], device uint* cnt [[buffer(3)]],
                     constant GeometryParams& j [[buffer(4)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    ulong s = 0;
    for (uint g = 0; g < j.groups; g++) { uint c = cnt[g]; cnt[g] = uint(min(s, ulong(0xffffffffu))); s += c; }
    const uint w = j.topology == 1 ? 1u : j.topology == 6 ? 2u : 3u;
    const uint n = s <= j.capacity ? uint(s) : j.capacity - j.capacity % w;
    args[0] = n; args[1] = 1; args[2] = 0; args[3] = 0; args[4] = 0;
}
kernel void geo_write(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                      device uint* args [[buffer(2)]], device uint* cnt [[buffer(3)]],
                      constant GeometryParams& j [[buffer(4)]], uint g [[thread_position_in_grid]]) {
    if (g < j.groups) geo_group(src, dst, j, g, true, cnt[g]);
}
)";

struct GeometryParams {
    u32 groups;
    u32 stride;
    u32 topology;
    u32 capacity;
};

} // namespace Metal::Geometry
