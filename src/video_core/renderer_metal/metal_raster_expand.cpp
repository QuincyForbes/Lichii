// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// See metal_raster_expand.h. The wide line's offset (half the width across the minor axis at each
// end, the ends on the major axis) is nxnative's dm_wl_offset (dm/gfx/tools/dmsh_pull.py, MIT
// License, Copyright (c) 2026 seedds).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_metal/metal_raster_expand.h"

namespace Metal::RasterExpand {

namespace {

using Topology = Maxwell::PrimitiveTopology;

struct Member {
    std::string type;
    std::string name;
    std::string attributes; ///< The `[[...]]` groups
    std::string array;      ///< `[N]` or empty
};

[[nodiscard]] std::string Trim(std::string_view text) {
    size_t begin = 0;
    size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t' || text[begin] == '\r')) {
        ++begin;
    }
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\r' ||
                           text[end - 1] == ';')) {
        --end;
    }
    return std::string{text.substr(begin, end - begin)};
}

/// The members of `struct <name>` in `msl`, one per line; empty when it is not found or a line is
/// not `type name [[attributes]] [N];`.
[[nodiscard]] std::vector<Member> ParseStruct(const std::string& msl, const std::string& name,
                                              size_t* body_begin = nullptr) {
    std::vector<Member> members;
    const std::string needle = "struct " + name + "\n";
    const size_t at = msl.find(needle);
    if (at == std::string::npos) {
        return members;
    }
    const size_t open = msl.find('{', at);
    const size_t close = msl.find("};", open);
    if (open == std::string::npos || close == std::string::npos) {
        return members;
    }
    if (body_begin != nullptr) {
        *body_begin = open + 1;
    }
    size_t pos = open + 1;
    while (pos < close) {
        size_t line_end = msl.find('\n', pos);
        line_end = (std::min)(line_end == std::string::npos ? close : line_end, close);
        std::string line = Trim(std::string_view{msl}.substr(pos, line_end - pos));
        pos = line_end + 1;
        if (line.empty()) {
            continue;
        }
        Member member;
        const size_t type_end = line.find(' ');
        if (type_end == std::string::npos) {
            return {};
        }
        member.type = line.substr(0, type_end);
        size_t at_name = type_end + 1;
        size_t name_end = at_name;
        while (name_end < line.size() &&
               (std::isalnum(static_cast<unsigned char>(line[name_end])) || line[name_end] == '_')) {
            ++name_end;
        }
        member.name = line.substr(at_name, name_end - at_name);
        std::string rest = Trim(line.substr(name_end));
        while (!rest.empty()) {
            if (rest.starts_with("[[")) {
                const size_t end = rest.find("]]");
                if (end == std::string::npos) {
                    return {};
                }
                member.attributes += rest.substr(0, end + 2);
                rest = Trim(rest.substr(end + 2));
            } else if (rest[0] == '[') {
                member.array = rest;
                rest.clear();
            } else {
                return {};
            }
        }
        if (member.name.empty()) {
            return {};
        }
        members.push_back(std::move(member));
    }
    return members;
}

[[nodiscard]] bool IsFloating(const Member& member) {
    return (member.type.starts_with("float") || member.type.starts_with("half")) &&
           member.attributes.find("flat") == std::string::npos;
}

[[nodiscard]] u32 ArraySize(const Member& member) {
    return member.array.empty() ? 0U : static_cast<u32>(std::atoi(member.array.c_str() + 1));
}

/// `out.name = expression;` for a member, element by element for an array. `value` holds the
/// expression with `%` for the member's access, e.g. "v.%".
[[nodiscard]] std::string Assign(const Member& member, const std::string& value_pattern) {
    const auto access = [&](const std::string& suffix) {
        std::string value = value_pattern;
        std::string replaced;
        for (const char c : value) {
            if (c == '%') {
                replaced += member.name + suffix;
            } else {
                replaced += c;
            }
        }
        return fmt::format("out.{}{} = {};\n", member.name, suffix, replaced);
    };
    if (const u32 size = ArraySize(member); size != 0) {
        std::string text;
        for (u32 index = 0; index < size; ++index) {
            text += "    " + access(fmt::format("[{}]", index));
        }
        return text;
    }
    return "    " + access("");
}

[[nodiscard]] std::string SwizzleOf(const std::string& variable, u32 swizzle) {
    static constexpr std::string_view Axes = "xyzw";
    std::string components;
    for (u32 index = 0; index < 4; ++index) {
        const u32 source = (swizzle >> (index * 4)) & 7;
        components += fmt::format("{}{}{}.{}", index != 0 ? ", " : "", (source & 1) != 0 ? "-" : "",
                                  variable, Axes[source >> 1]);
    }
    return fmt::format("float4({})", components);
}

} // Anonymous namespace

bool Enabled() noexcept {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_METAL_RASTER_EXPAND");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

u32 PrimitiveOf(Topology topology) noexcept {
    switch (topology) {
    case Topology::Lines:
        return PrimLines;
    case Topology::LineStrip:
        return PrimLineStrip;
    case Topology::LineLoop:
        return PrimLineLoop;
    case Topology::Triangles:
        return PrimTriangles;
    case Topology::TriangleStrip:
        return PrimTriangleStrip;
    case Topology::TriangleFan:
        return PrimTriangleFan;
    default:
        return ~0U;
    }
}

f32 LineWidth(const Maxwell& regs) noexcept {
    const bool smooth = regs.line_anti_alias_enable != 0;
    const f32 width = smooth ? regs.line_width_smooth : regs.line_width_aliased;
    // OpenGL rounds the width of a line that is not anti-aliased to a whole number of pixels.
    return (std::min)(smooth ? width : std::round(width), 64.0f);
}

u32 BitsOf(const Maxwell& regs, Topology topology) noexcept {
    const u32 prim = PrimitiveOf(topology);
    if (!Enabled() || prim == ~0U) {
        return 0;
    }
    u32 bits = 0;
    if (prim <= PrimLineLoop) {
        const bool smooth = regs.line_anti_alias_enable != 0;
        const bool stipple = regs.line_stipple_enable != 0;
        if (LineWidth(regs) >= 2.0f || smooth || stipple) {
            bits |= Wide | (stipple ? Stipple : 0U) | (smooth ? Smooth : 0U);
        }
    } else if (regs.conservative_raster_enable != 0) {
        bits |= Conservative;
    }
    return bits;
}

u32 PrimitivesOf(u32 prim, u32 count) noexcept {
    switch (prim) {
    case PrimLines:
        return count / 2;
    case PrimLineStrip:
        return count >= 2 ? count - 1 : 0;
    case PrimLineLoop:
        return count >= 2 ? count : 0;
    case PrimTriangles:
        return count / 3;
    default:
        return count >= 3 ? count - 2 : 0;
    }
}

u32 VerticesOf(u32 prim, u32 count) noexcept {
    return PrimitivesOf(prim, count) * (prim <= PrimLineLoop ? 6U : 3U);
}

std::string Generate(const std::string& vertex_msl, const std::string& kernel_msl, u32 bits,
                     u32 prim, u32 swizzle) {
    size_t body_begin = 0;
    const std::vector<Member> outputs = ParseStruct(vertex_msl, "main0_out", &body_begin);
    const std::vector<Member> kernel = ParseStruct(kernel_msl, "main0_out");
    if (outputs.empty() || kernel.empty()) {
        return {};
    }
    const auto position = std::ranges::find_if(outputs, [](const Member& member) {
        return member.attributes.find("[[position") != std::string::npos;
    });
    const auto in_kernel = [&](const std::string& name) {
        return std::ranges::any_of(kernel, [&](const Member& member) { return member.name == name; });
    };
    if (position == outputs.end() || !in_kernel(position->name)) {
        return {};
    }
    const bool lines = (bits & Wide) != 0;
    const bool coordinate = (bits & (Stipple | Smooth)) != 0;
    const bool flip = vertex_msl.find("Invert Y-axis for Metal") != std::string::npos;
    static_cast<void>(prim);

    std::string source = vertex_msl;
    if (coordinate) {
        source.insert(body_begin,
                      "\n    float4 rasterlc [[user(rasterlc)]];\n    float rasterlw [[user(rasterlw)]];");
    }
    std::string text = "\n\nstruct KernelOut\n{\n";
    for (const Member& member : kernel) {
        text += fmt::format("    {} {}{};\n", member.type, member.name, member.array);
    }
    text += "};\n\nstruct RasterExpandParams\n{\n    float2 half_viewport;\n    float width;\n"
            "    uint prim;\n    uint count;\n    float factor;\n    uint pattern;\n    uint pad;\n};\n\n";
    text += fmt::format(
        "vertex main0_out {}(device const KernelOut* src [[buffer({})]],\n"
        "                    constant RasterExpandParams& P [[buffer({})]],\n"
        "                    uint vid [[vertex_id]], uint iid [[instance_id]])\n{{\n",
        Entry, SourceSlot, ParamsSlot);
    const std::string position_name = position->name;
    const auto clip = [&](const std::string& variable) {
        const std::string access = variable + "." + position_name;
        return swizzle != 0 ? SwizzleOf(access, swizzle) : access;
    };
    // The position the pixel program's viewport takes: the y flip the kernel (which captures the
    // guest's clip space) did not do.
    const std::string finish_position = flip ? "    pos.y = -pos.y;\n" : "";
    if (lines) {
        text += "    const uint seg = vid / 6u;\n    const uint corner = vid % 6u;\n"
                "    uint ia = seg;\n    uint ib = seg + 1u;\n"
                "    if (P.prim == 0u) {\n        ia = seg * 2u;\n        ib = ia + 1u;\n    } "
                "else if (P.prim == 2u) {\n        ib = seg + 1u == P.count ? 0u : seg + 1u;\n    }\n"
                "    const uint base = iid * P.count;\n"
                "    const bool far_end = ((0x32u >> corner) & 1u) != 0u;\n"
                "    const bool plus = ((0x2Cu >> corner) & 1u) != 0u;\n"
                "    device const KernelOut& a = src[base + ia];\n"
                "    device const KernelOut& b = src[base + ib];\n"
                "    device const KernelOut& v = far_end ? b : a;\n"
                "    main0_out out = {};\n";
        for (const Member& member : outputs) {
            if (&member != &*position && in_kernel(member.name)) {
                text += Assign(member, "v.%");
            }
        }
        text += fmt::format(
            "    const float4 pa = {};\n    const float4 pb = {};\n    float4 pos = far_end ? pb : pa;\n"
            "    if (pa.w <= 0.0f || pb.w <= 0.0f) {{\n"
            "        out.{} = float4(0.0f, 0.0f, 0.0f, 1.0f);\n        return out;\n    }}\n"
            "    const float2 d = (pb.xy / pb.w - pa.xy / pa.w) * P.half_viewport;\n"
            "    const float side = plus ? 0.5f : -0.5f;\n"
            "    const float2 o = abs(d.x) >= abs(d.y) ? float2(0.0f, side * P.width)\n"
            "                                          : float2(side * P.width, 0.0f);\n"
            "    pos.xy += o / P.half_viewport * pos.w;\n{}    out.{} = pos;\n",
            clip("a"), clip("b"), position_name, finish_position, position_name);
        if (coordinate) {
            text += "    out.rasterlc = float4(far_end ? length(d) : 0.0f, plus ? 1.0f : -1.0f,\n"
                    "                          float(P.pattern), P.factor);\n"
                    "    out.rasterlw = P.width;\n";
        }
        text += "    return out;\n}\n";
    } else {
        text += "    const uint tri = vid / 3u;\n    const uint corner = vid % 3u;\n"
                "    uint i0 = tri * 3u;\n    uint i1 = i0 + 1u;\n    uint i2 = i0 + 2u;\n"
                "    if (P.prim == 4u) {\n        i0 = tri + (tri & 1u);\n        i1 = tri + 1u - (tri & 1u);\n"
                "        i2 = tri + 2u;\n    } else if (P.prim == 5u) {\n        i0 = 0u;\n"
                "        i1 = tri + 1u;\n        i2 = tri + 2u;\n    }\n"
                "    const uint base = iid * P.count;\n"
                "    device const KernelOut& v0 = src[base + i0];\n"
                "    device const KernelOut& v1 = src[base + i1];\n"
                "    device const KernelOut& v2 = src[base + i2];\n"
                "    main0_out out = {};\n";
        text += fmt::format(
            "    const float4 c0 = {};\n    const float4 c1 = {};\n    const float4 c2 = {};\n"
            "    const float4 own = corner == 0u ? c0 : (corner == 1u ? c1 : c2);\n"
            "    float3 lambda = float3(corner == 0u, corner == 1u, corner == 2u);\n"
            "    float4 pos = own;\n"
            "    if (c0.w > 0.0f && c1.w > 0.0f && c2.w > 0.0f) {{\n"
            "        const float2 s0 = c0.xy / c0.w * P.half_viewport;\n"
            "        const float2 s1 = c1.xy / c1.w * P.half_viewport;\n"
            "        const float2 s2 = c2.xy / c2.w * P.half_viewport;\n"
            "        const float area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);\n"
            "        if (abs(area) > 1e-4f) {{\n"
            "            const float sign = area > 0.0f ? 1.0f : -1.0f;\n"
            "            const float2 sp = corner == 0u ? s0 : (corner == 1u ? s1 : s2);\n"
            "            const float2 sa = corner == 0u ? s2 : (corner == 1u ? s0 : s1);\n"
            "            const float2 sn = corner == 0u ? s1 : (corner == 1u ? s2 : s0);\n"
            "            const float2 ea = sp - sa;\n"
            "            const float2 en = sn - sp;\n"
            "            const float2 na = normalize(float2(ea.y, -ea.x)) * sign;\n"
            "            const float2 nn = normalize(float2(en.y, -en.x)) * sign;\n"
            "            const float da = 0.5f * (abs(na.x) + abs(na.y));\n"
            "            const float dn = 0.5f * (abs(nn.x) + abs(nn.y));\n"
            "            const float ca = dot(na, sp) + da;\n"
            "            const float cn = dot(nn, sp) + dn;\n"
            "            const float det = na.x * nn.y - na.y * nn.x;\n"
            "            float2 x = sp + na * da;\n"
            "            if (abs(det) > 1e-3f) {{\n"
            "                x = float2(ca * nn.y - cn * na.y, na.x * cn - nn.x * ca) / det;\n"
            "            }}\n"
            "            const float2 grow = x - sp;\n"
            "            const float reach = length(grow);\n"
            "            if (reach > 8.0f) {{\n                x = sp + grow * (8.0f / reach);\n            }}\n"
            "            const float m0 = ((s1.x - x.x) * (s2.y - x.y) - (s1.y - x.y) * (s2.x - x.x)) / area;\n"
            "            const float m1 = ((s2.x - x.x) * (s0.y - x.y) - (s2.y - x.y) * (s0.x - x.x)) / area;\n"
            "            const float3 mu = float3(m0, m1, 1.0f - m0 - m1);\n"
            "            const float3 l = mu / float3(c0.w, c1.w, c2.w);\n"
            "            lambda = l / (l.x + l.y + l.z);\n"
            "            pos = lambda.x * c0 + lambda.y * c1 + lambda.z * c2;\n"
            "        }}\n"
            "    }}\n",
            clip("v0"), clip("v1"), clip("v2"));
        for (const Member& member : outputs) {
            if (&member == &*position || !in_kernel(member.name)) {
                continue;
            }
            text += IsFloating(member)
                        ? Assign(member, "lambda.x * v0.% + lambda.y * v1.% + lambda.z * v2.%")
                        : Assign(member, "v0.%");
        }
        text += fmt::format("{}    out.{} = pos;\n    return out;\n}}\n", finish_position,
                            position_name);
    }
    return source + text;
}

namespace {

struct FragmentEntry {
    std::string output;
    size_t start{};        ///< "fragment"
    size_t params_begin{}; ///< After the parenthesis
    size_t params_end{};   ///< The closing parenthesis
    size_t body{};         ///< The opening brace
};

[[nodiscard]] bool LocateFragment(const std::string& msl, FragmentEntry& entry) {
    entry.start = msl.rfind("fragment ");
    if (entry.start == std::string::npos) {
        return false;
    }
    size_t pos = entry.start + 9;
    const auto word = [&]() {
        while (pos < msl.size() && (msl[pos] == ' ' || msl[pos] == '\n')) {
            ++pos;
        }
        const size_t start = pos;
        while (pos < msl.size() &&
               (std::isalnum(static_cast<unsigned char>(msl[pos])) || msl[pos] == '_')) {
            ++pos;
        }
        return msl.substr(start, pos - start);
    };
    entry.output = word();
    const std::string name = word();
    while (pos < msl.size() && msl[pos] == ' ') {
        ++pos;
    }
    if (entry.output.empty() || name.empty() || pos >= msl.size() || msl[pos] != '(') {
        return false;
    }
    entry.params_begin = pos + 1;
    int depth = 0;
    for (; pos < msl.size(); ++pos) {
        if (msl[pos] == '(') {
            ++depth;
        } else if (msl[pos] == ')' && --depth == 0) {
            break;
        }
    }
    entry.params_end = pos;
    entry.body = pos < msl.size() ? msl.find('{', pos) : std::string::npos;
    return entry.body != std::string::npos;
}

[[nodiscard]] bool IsWordChar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

} // Anonymous namespace

bool PatchFragment(std::string& msl, u32 bits) {
    if ((bits & (Stipple | Smooth)) == 0) {
        return true;
    }
    FragmentEntry entry;
    if (!LocateFragment(msl, entry)) {
        return false;
    }
    const std::string members =
        "\n    float4 rasterlc [[user(rasterlc)]];\n    float rasterlw [[user(rasterlw)]];";
    std::string input = "raster_input";
    bool made_struct = true;
    const size_t stage_in = msl.find("[[stage_in]]", entry.params_begin);
    if (stage_in != std::string::npos && stage_in < entry.params_end) {
        // The stage-in parameter `<type> <name> [[stage_in]]`: its struct gets the members.
        size_t end = stage_in;
        while (end > entry.params_begin && msl[end - 1] == ' ') {
            --end;
        }
        size_t start = end;
        while (start > entry.params_begin && IsWordChar(msl[start - 1])) {
            --start;
        }
        input = msl.substr(start, end - start);
        size_t type_end = start;
        while (type_end > entry.params_begin && msl[type_end - 1] == ' ') {
            --type_end;
        }
        size_t type_start = type_end;
        while (type_start > entry.params_begin && IsWordChar(msl[type_start - 1])) {
            --type_start;
        }
        const std::string type = msl.substr(type_start, type_end - type_start);
        const size_t struct_at = msl.find("struct " + type + "\n");
        const size_t open = struct_at == std::string::npos ? struct_at : msl.find('{', struct_at);
        if (open == std::string::npos) {
            return false;
        }
        msl.insert(open + 1, members);
        made_struct = false;
    }
    if (!LocateFragment(msl, entry)) {
        return false;
    }
    // The output variable and the colour members of its struct.
    const std::string output_needle = entry.output + " ";
    const size_t variable_at = msl.find(output_needle, entry.body);
    if (variable_at == std::string::npos) {
        return false;
    }
    std::string output_variable;
    for (size_t p = variable_at + output_needle.size(); p < msl.size() && IsWordChar(msl[p]); ++p) {
        output_variable += msl[p];
    }
    std::string coverage;
    if ((bits & Smooth) != 0) {
        // Coverage across the line: its half width minus the distance from the centre, plus half
        // a pixel, on the alpha of every float colour output.
        const std::vector<Member> outputs = ParseStruct(msl, entry.output);
        for (const Member& member : outputs) {
            if (member.type == "float4" && member.attributes.find("[[color(") != std::string::npos &&
                member.attributes.find("index(") == std::string::npos) {
                coverage += fmt::format("        {0}.{1}.w *= saturate(0.5f * {2}.rasterlw * (1.0f - "
                                        "abs({2}.rasterlc.y)) + 0.5f);\n",
                                        output_variable, member.name, input);
            }
        }
    }
    const std::string return_needle = "return " + output_variable + ";";
    std::vector<size_t> returns;
    for (size_t at = msl.find(return_needle, entry.body); at != std::string::npos;
         at = msl.find(return_needle, at + return_needle.size())) {
        returns.push_back(at);
    }
    if (output_variable.empty() || returns.empty()) {
        return false;
    }
    if (!coverage.empty()) {
        for (auto it = returns.rbegin(); it != returns.rend(); ++it) {
            msl.replace(*it, return_needle.size(),
                        "{\n" + coverage + "        " + return_needle + "\n    }");
        }
    }
    if ((bits & Stipple) != 0) {
        msl.insert(entry.body + 1,
                   fmt::format("\n    if ((((uint({0}.rasterlc.z)) >> ((uint(floor({0}.rasterlc.x / "
                               "max({0}.rasterlc.w, 1.0f)))) & 15u)) & 1u) == 0u) {{\n"
                               "        discard_fragment();\n    }}",
                               input));
    }
    if (made_struct) {
        bool empty = true;
        for (size_t p = entry.params_begin; p < entry.params_end; ++p) {
            empty = empty && (msl[p] == ' ' || msl[p] == '\n');
        }
        msl.insert(entry.params_end,
                   std::string{empty ? "" : ", "} + "raster_in raster_input [[stage_in]]");
        msl.insert(entry.start, "struct raster_in\n{" + members.substr(0) + "\n};\n\n");
    }
    return true;
}

} // namespace Metal::RasterExpand
