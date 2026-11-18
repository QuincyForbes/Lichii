// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Transform feedback for the Metal renderer, which has none: a draw whose pipeline captures runs
// its vertex program a second time as a stage-in kernel writing the vertex outputs to a buffer
// (the tessellation vertex kernel), then a gather kernel copies the varyings the guest asked for
// into its transform feedback buffers at the byte offsets the capture has reached. The
// capture-state arithmetic (per buffer offsets, the capacity clamp, whole primitives only) is
// nxnative's tf_prepare (dm/gfx/msl/tf.metal, dm/gfx/nvn/nvn_feedback.inc), kept on the CPU as its
// g_tf_cpu state is while it is known. Plain C++.
//
// nxnative (https://github.com/seedds/nxnative), MIT License, Copyright (c) 2026 seedds: the
// capture-state arithmetic of CapturedVertices is ported from it under that license.

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/transform_feedback.h"

namespace Metal::Xfb {

inline constexpr u32 MaxBuffers = 4;
inline constexpr size_t LayoutWords = sizeof(VideoCommon::TransformFeedbackState) / sizeof(u32);
static_assert(sizeof(VideoCommon::TransformFeedbackState) % sizeof(u32) == 0);

/// Buffer slots of the gather kernel: the captured vertices, the four capture buffers, the count.
inline constexpr u32 GatherSourceSlot = 0;
inline constexpr u32 GatherBufferSlot = 1;
inline constexpr u32 GatherCountSlot = 5;
inline constexpr u32 GatherGroupSize = 64;
inline constexpr std::string_view GatherEntry = "xfb_gather";

/// The most vertices one draw captures (the buffer between the two kernels is that many output
/// structs).
inline constexpr u32 MaxCapturedVertices = 1U << 20;

/// The guest's transform feedback layout as a pipeline key holds it: the varyings of the buffers
/// that are enabled, all zero when nothing is captured.
struct Layout {
    std::array<u32, LayoutWords> words{};

    bool operator==(const Layout&) const noexcept = default;

    [[nodiscard]] VideoCommon::TransformFeedbackState State() const noexcept {
        VideoCommon::TransformFeedbackState state;
        std::memcpy(&state, words.data(), sizeof(state));
        return state;
    }

    [[nodiscard]] u32 Stride(u32 buffer) const noexcept {
        return words[buffer * 3 + 2];
    }

    [[nodiscard]] bool Active() const noexcept {
        for (u32 buffer = 0; buffer < MaxBuffers; ++buffer) {
            if (Stride(buffer) != 0) {
                return true;
            }
        }
        return false;
    }
};

/// What the draw captures into the four buffers (a buffer that is disabled, unbound or without a
/// layout has no stride), or nothing (`Active()` false) when transform feedback is off or the
/// topology is one the capture can't take (it takes lists: points, lines, triangles).
[[nodiscard]] inline Layout MakeLayout(const Tegra::Engines::Maxwell3D::Regs& regs,
                                       Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology,
                                       bool geometry = false) {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    Layout layout;
    // A geometry program's capture is of its output, whatever the draw's topology.
    if (regs.transform_feedback_enabled == 0 ||
        (!geometry && topology != Regs::PrimitiveTopology::Points &&
         topology != Regs::PrimitiveTopology::Lines &&
         topology != Regs::PrimitiveTopology::Triangles)) {
        return layout;
    }
    VideoCommon::TransformFeedbackState state{};
    for (u32 buffer = 0; buffer < MaxBuffers; ++buffer) {
        const auto& binding = regs.transform_feedback.buffers[buffer];
        const auto& control = regs.transform_feedback.controls[buffer];
        if (binding.enable == 0 || binding.size <= 0 || control.stride == 0 ||
            control.varying_count == 0) {
            continue;
        }
        state.layouts[buffer] = {.stream = control.stream,
                                 .varying_count = control.varying_count,
                                 .stride = control.stride};
        state.varyings[buffer] = regs.stream_out_layout[buffer];
    }
    std::memcpy(layout.words.data(), &state, sizeof(state));
    return layout;
}

/// One varying in the capture buffers: `components` floats from the attribute word `attribute`
/// (Shader::IR::Attribute) onwards at byte `offset` of every vertex of `buffer`.
struct Varying {
    u32 buffer{};
    u32 offset{};
    u32 components{};
    u32 attribute{};
};

[[nodiscard]] inline std::vector<Varying> Varyings(const Layout& layout) {
    const auto [varyings, count] = VideoCommon::MakeTransformFeedbackVaryings(layout.State());
    std::vector<Varying> result;
    for (u32 attribute = 0; attribute < count; ++attribute) {
        const Shader::TransformFeedbackVarying& varying = varyings[attribute];
        if (varying.components == 0 || varying.stride == 0 || varying.buffer >= MaxBuffers) {
            continue;
        }
        result.push_back({varying.buffer, varying.offset, varying.components, attribute});
    }
    return result;
}

/// The gather kernel, appended to the MSL of the capture kernel `entry` (its output struct is
/// `<entry>_out`, one per captured vertex): vertex n's varyings go to byte n * stride of each
/// buffer. A component the vertex program did not write is zero.
[[nodiscard]] inline std::string GatherSource(
    const Layout& layout, std::string_view entry,
    std::span<const ShaderTranslate::InterfaceVariable> outputs,
    const std::map<Shader::IR::Attribute, Shader::IR::Attribute>& legacy_mapping) {
    using Variable = ShaderTranslate::InterfaceVariable;
    // spv::BuiltIn
    constexpr u32 BuiltinPosition = 0;
    constexpr u32 BuiltinPointSize = 1;
    const auto find = [&](u32 location, u32 builtin) -> const Variable* {
        for (const Variable& variable : outputs) {
            if (variable.per_patch || variable.name.empty()) {
                continue;
            }
            if (builtin == Variable::NoBuiltin ? variable.builtin == Variable::NoBuiltin &&
                                                     variable.location == location
                                               : variable.builtin == builtin) {
                return &variable;
            }
        }
        return nullptr;
    };
    std::string body;
    for (const Varying& varying : Varyings(layout)) {
        for (u32 component = 0; component < varying.components; ++component) {
            u32 word = varying.attribute + component;
            const auto mapped = legacy_mapping.find(static_cast<Shader::IR::Attribute>(word));
            if (mapped != legacy_mapping.end()) {
                word = static_cast<u32>(mapped->second);
            }
            const Variable* variable = nullptr;
            u32 element = 0;
            constexpr u32 PointSize = static_cast<u32>(Shader::IR::Attribute::PointSize);
            constexpr u32 PositionX = static_cast<u32>(Shader::IR::Attribute::PositionX);
            constexpr u32 Generic0X = static_cast<u32>(Shader::IR::Attribute::Generic0X);
            if (word == PointSize) {
                variable = find(0, BuiltinPointSize);
            } else if (word >= PositionX && word < PositionX + 4) {
                variable = find(0, BuiltinPosition);
                element = word - PositionX;
            } else if (word >= Generic0X && word < Generic0X + 32 * 4) {
                variable = find((word - Generic0X) / 4, Variable::NoBuiltin);
                element = (word - Generic0X) % 4;
            }
            std::string value = "0.0f";
            if (variable != nullptr && element < variable->vecsize) {
                static constexpr std::string_view Swizzle = "xyzw";
                value = variable->vecsize == 1
                            ? fmt::format("float(v.{})", variable->name)
                            : fmt::format("float(v.{}.{})", variable->name, Swizzle[element]);
            }
            body += fmt::format("    *(device float*)(dst{} + ulong(n) * {}u + {}u) = {};\n",
                                varying.buffer, layout.Stride(varying.buffer),
                                varying.offset + component * 4, value);
        }
    }
    return fmt::format(R"(
kernel void xfb_gather(device const {0}_out* src [[buffer({1})]],
                       device uchar* dst0 [[buffer({2})]], device uchar* dst1 [[buffer({3})]],
                       device uchar* dst2 [[buffer({4})]], device uchar* dst3 [[buffer({5})]],
                       constant uint& count [[buffer({6})]],
                       uint n [[thread_position_in_grid]]) {{
    if (n >= count) {{
        return;
    }}
    device const {0}_out& v = src[n];
{7}}}
)",
                       entry, GatherSourceSlot, GatherBufferSlot, GatherBufferSlot + 1,
                       GatherBufferSlot + 2, GatherBufferSlot + 3, GatherCountSlot, body);
}

/// Vertices of the capture of a draw of `vertices` (a vertex count times the instances) into
/// buffers of `capacity` bytes with `written` already in them (a stride 0 buffer is not captured),
/// whole primitives only: nxnative's tf_prepare.
[[nodiscard]] inline u32 CapturedVertices(u32 vertices, u32 primitive_vertices,
                                          const std::array<u32, MaxBuffers>& stride,
                                          const std::array<u32, MaxBuffers>& capacity,
                                          const std::array<u32, MaxBuffers>& written) {
    u32 count = vertices;
    for (u32 buffer = 0; buffer < MaxBuffers; ++buffer) {
        if (stride[buffer] != 0) {
            const u32 room = capacity[buffer] - (std::min)(capacity[buffer], written[buffer]);
            count = (std::min)(count, room / stride[buffer]);
        }
    }
    return count - count % primitive_vertices;
}

/// Primitives the vertices of a draw with `topology` make (the transform feedback queries and the
/// primitives generated counter), as Vulkan's query cache converts a byte count.
[[nodiscard]] inline u64 PrimitiveCount(Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology topology,
                                        u64 vertices, u32 patch_vertices) {
    using Topology = Tegra::Engines::Maxwell3D::Regs::PrimitiveTopology;
    const auto minus = [&](u64 amount) { return vertices > amount ? vertices - amount : 0; };
    switch (topology) {
    case Topology::Points:
        return vertices;
    case Topology::Lines:
        return vertices / 2;
    case Topology::LineLoop:
        return vertices > 1 ? vertices : 0;
    case Topology::LineStrip:
        return minus(1);
    case Topology::LinesAdjacency:
        return vertices / 4;
    case Topology::LineStripAdjacency:
        return minus(3);
    case Topology::Triangles:
        return vertices / 3;
    case Topology::TrianglesAdjacency:
        return vertices / 6;
    case Topology::TriangleFan:
    case Topology::TriangleStrip:
        return minus(2);
    case Topology::TriangleStripAdjacency:
        return vertices > 4 ? (vertices - 4) / 2 : 0;
    case Topology::Quads:
        return vertices / 4;
    case Topology::QuadStrip:
        return vertices > 2 ? (vertices - 2) / 2 : 0;
    case Topology::Polygon:
        return vertices >= 3 ? 1 : 0;
    case Topology::Patches:
        return vertices / (std::max)(patch_vertices, 1U);
    }
    return vertices;
}

} // namespace Metal::Xfb
