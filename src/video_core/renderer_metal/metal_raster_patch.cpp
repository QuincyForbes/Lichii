// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// See metal_raster_patch.h. The logic operation's framebuffer fetch and its quantization of
// normalized targets follow nxnative's dm/gfx/tools/dmsh_logic.py (MIT).

#include <cstdlib>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "video_core/renderer_metal/metal_raster_patch.h"
#include "video_core/surface.h"

namespace Metal::RasterPatch {

namespace {

[[nodiscard]] bool EnvironmentOn(const char* name) noexcept {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

[[nodiscard]] bool IsIdentifier(char c) noexcept {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

/// The identifier starting at `pos`, and the position after it.
[[nodiscard]] std::string_view Identifier(const std::string& text, size_t& pos) {
    const size_t start = pos;
    while (pos < text.size() && IsIdentifier(text[pos])) {
        ++pos;
    }
    return std::string_view{text}.substr(start, pos - start);
}

void SkipSpace(const std::string& text, size_t& pos) {
    while (pos < text.size() &&
           (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\t' || text[pos] == '\r')) {
        ++pos;
    }
}

/// The entry point of a `stage` function ("fragment", "vertex"): the output struct it returns, its
/// name, where its parameter list starts (after the parenthesis) and ends (the parenthesis).
struct Entry {
    std::string output;
    std::string name;
    size_t parameters_begin{};
    size_t parameters_end{};
    size_t body_begin{};
};

[[nodiscard]] bool FindEntry(const std::string& msl, std::string_view stage, Entry& entry) {
    // The last `<stage> <struct> <name>(` that starts a line or follows an attribute (an early
    // fragment tests attribute comes first on the line).
    const std::string needle = fmt::format("{} ", stage);
    for (size_t at = msl.rfind(needle); at != std::string::npos;
         at = at == 0 ? std::string::npos : msl.rfind(needle, at - 1)) {
        if (at != 0 && msl[at - 1] != '\n' && msl[at - 1] != ' ') {
            continue;
        }
        size_t pos = at + needle.size();
        SkipSpace(msl, pos);
        entry.output = std::string{Identifier(msl, pos)};
        SkipSpace(msl, pos);
        entry.name = std::string{Identifier(msl, pos)};
        SkipSpace(msl, pos);
        if (entry.output.empty() || entry.name.empty() || pos >= msl.size() || msl[pos] != '(') {
            continue;
        }
        entry.parameters_begin = pos + 1;
        int depth = 0;
        for (; pos < msl.size(); ++pos) {
            if (msl[pos] == '(') {
                ++depth;
            } else if (msl[pos] == ')' && --depth == 0) {
                break;
            }
        }
        if (pos >= msl.size()) {
            continue;
        }
        entry.parameters_end = pos;
        entry.body_begin = msl.find('{', pos);
        return entry.body_begin != std::string::npos;
    }
    return false;
}

/// The variable the entry point returns: `main0_out out = {};`.
[[nodiscard]] std::string OutputVariable(const std::string& msl, const Entry& entry) {
    const std::string needle = entry.output + " ";
    size_t at = msl.find(needle, entry.body_begin);
    if (at == std::string::npos) {
        return {};
    }
    size_t pos = at + needle.size();
    return std::string{Identifier(msl, pos)};
}

/// `return <variable>;` positions in the entry's body, last first.
[[nodiscard]] std::vector<size_t> Returns(const std::string& msl, const Entry& entry,
                                          const std::string& variable) {
    std::vector<size_t> found;
    const std::string needle = "return " + variable + ";";
    for (size_t at = msl.find(needle, entry.body_begin); at != std::string::npos;
         at = msl.find(needle, at + needle.size())) {
        found.push_back(at);
    }
    return {found.rbegin(), found.rend()};
}

struct OutputMember {
    std::string type;
    std::string name;
    u32 target;
};

/// The colour outputs of the output struct (dual source ones are left out).
[[nodiscard]] std::vector<OutputMember> ColorMembers(const std::string& msl,
                                                     const std::string& output) {
    std::vector<OutputMember> members;
    const std::string needle = "struct " + output;
    const size_t begin = msl.find(needle);
    if (begin == std::string::npos) {
        return members;
    }
    const size_t end = msl.find("};", begin);
    if (end == std::string::npos) {
        return members;
    }
    size_t pos = msl.find('{', begin);
    while (pos != std::string::npos && pos < end) {
        const size_t line_end = msl.find('\n', pos + 1);
        const std::string_view line = std::string_view{msl}.substr(
            pos + 1, (line_end == std::string::npos ? end : (std::min)(line_end, end)) - pos - 1);
        pos = line_end;
        const size_t color = line.find("[[color(");
        if (color == std::string_view::npos || line.find("index(") != std::string_view::npos) {
            continue;
        }
        size_t at = 0;
        while (at < line.size() && line[at] == ' ') {
            ++at;
        }
        const size_t type_end = line.find(' ', at);
        if (type_end == std::string_view::npos || type_end > color) {
            continue;
        }
        size_t name_at = type_end;
        while (name_at < line.size() && line[name_at] == ' ') {
            ++name_at;
        }
        size_t name_end = name_at;
        while (name_end < line.size() && IsIdentifier(line[name_end])) {
            ++name_end;
        }
        members.push_back({std::string{line.substr(at, type_end - at)},
                           std::string{line.substr(name_at, name_end - name_at)},
                           static_cast<u32>(std::atoi(std::string{line.substr(color + 8)}.c_str()))});
    }
    return members;
}

[[nodiscard]] const char* LogicExpression(u32 op) {
    switch (op) {
    case 0:
        return "uint4(0u)";
    case 1:
        return "s & d";
    case 2:
        return "s & ~d";
    case 3:
        return "s";
    case 4:
        return "~s & d";
    case 5:
        return "d";
    case 6:
        return "s ^ d";
    case 7:
        return "s | d";
    case 8:
        return "~(s | d)";
    case 9:
        return "~(s ^ d)";
    case 10:
        return "~d";
    case 11:
        return "s | ~d";
    case 12:
        return "~s";
    case 13:
        return "~s | d";
    case 14:
        return "~(s & d)";
    default:
        return "~uint4(0u)";
    }
}

/// Bits per channel of a format.
[[nodiscard]] const char* LogicBits(LogicFormat format) {
    switch (format) {
    case LogicFormat::Unorm8:
    case LogicFormat::Snorm8:
    case LogicFormat::Uint8:
    case LogicFormat::Sint8:
        return "uint4(8u)";
    case LogicFormat::Unorm16:
    case LogicFormat::Snorm16:
    case LogicFormat::Uint16:
    case LogicFormat::Sint16:
        return "uint4(16u)";
    case LogicFormat::Uint32:
    case LogicFormat::Sint32:
        return "uint4(32u)";
    case LogicFormat::Unorm10_2:
    case LogicFormat::Uint10_2:
        return "uint4(10u, 10u, 10u, 2u)";
    case LogicFormat::Unorm565:
        return "uint4(5u, 6u, 5u, 0u)";
    case LogicFormat::Unorm5551:
        return "uint4(5u, 5u, 5u, 1u)";
    case LogicFormat::Unorm4444:
        return "uint4(4u)";
    default:
        return "";
    }
}

[[nodiscard]] bool IsUint(LogicFormat format) {
    return format >= LogicFormat::Uint8 && format <= LogicFormat::Uint10_2;
}

[[nodiscard]] bool IsSint(LogicFormat format) {
    return format >= LogicFormat::Sint8;
}

} // Anonymous namespace

LogicFormat LogicFormatOf(Tegra::RenderTargetFormat format) noexcept {
    using Format = Tegra::RenderTargetFormat;
    switch (format) {
    case Format::A8R8G8B8_UNORM:
    case Format::A8B8G8R8_UNORM:
    case Format::X8R8G8B8_UNORM:
    case Format::X8B8G8R8_UNORM:
    case Format::R8G8_UNORM:
    case Format::R8_UNORM:
        return LogicFormat::Unorm8;
    case Format::A8B8G8R8_SNORM:
    case Format::R8G8_SNORM:
    case Format::R8_SNORM:
        return LogicFormat::Snorm8;
    case Format::R16G16B16A16_UNORM:
    case Format::R16G16_UNORM:
    case Format::R16_UNORM:
        return LogicFormat::Unorm16;
    case Format::R16G16B16A16_SNORM:
    case Format::R16G16_SNORM:
    case Format::R16_SNORM:
        return LogicFormat::Snorm16;
    case Format::A2B10G10R10_UNORM:
    case Format::A2R10G10B10_UNORM:
        return LogicFormat::Unorm10_2;
    case Format::R5G6B5_UNORM:
        return LogicFormat::Unorm565;
    case Format::A1R5G5B5_UNORM:
    case Format::X1R5G5B5_UNORM:
        return LogicFormat::Unorm5551;
    case Format::A8B8G8R8_UINT:
    case Format::R8G8_UINT:
    case Format::R8_UINT:
        return LogicFormat::Uint8;
    case Format::R16G16B16A16_UINT:
    case Format::R16G16_UINT:
    case Format::R16_UINT:
        return LogicFormat::Uint16;
    case Format::R32G32B32A32_UINT:
    case Format::R32G32B32X32_UINT:
    case Format::R32G32_UINT:
    case Format::R32_UINT:
        return LogicFormat::Uint32;
    case Format::A2B10G10R10_UINT:
        return LogicFormat::Uint10_2;
    case Format::A8B8G8R8_SINT:
    case Format::R8G8_SINT:
    case Format::R8_SINT:
        return LogicFormat::Sint8;
    case Format::R16G16B16A16_SINT:
    case Format::R16G16_SINT:
    case Format::R16_SINT:
        return LogicFormat::Sint16;
    case Format::R32G32B32A32_SINT:
    case Format::R32G32B32X32_SINT:
    case Format::R32G32_SINT:
    case Format::R32_SINT:
        return LogicFormat::Sint32;
    default:
        return LogicFormat::None;
    }
    return LogicFormat::None;
}

bool LogicOpEnabled() noexcept {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_LOGIC_OP");
    return enabled;
}

LogicState LogicStateOf(const Maxwell& regs) noexcept {
    LogicState state{};
    const u32 op = static_cast<u32>(regs.logic_op.op);
    if (regs.logic_op.enable == 0 || op < 0x1500 || op >= 0x1510 || !LogicOpEnabled()) {
        return state;
    }
    for (size_t target = 0; target < Maxwell::NumRenderTargets; ++target) {
        state.formats |= static_cast<u32>(LogicFormatOf(regs.rt[target].format)) << (target * 4);
    }
    if (state.formats != 0) {
        state.op = op - 0x1500;
    }
    return state;
}

bool ApplyLogicOp(std::string& msl, const LogicState& state) {
    Entry entry;
    if (!state.Active() || !FindEntry(msl, "fragment", entry)) {
        return false;
    }
    const std::string variable = OutputVariable(msl, entry);
    const std::vector<OutputMember> members = ColorMembers(msl, entry.output);
    const std::vector<size_t> returns = Returns(msl, entry, variable);
    if (variable.empty() || members.empty() || returns.empty()) {
        return false;
    }
    std::string parameters;
    std::string apply;
    std::string used_formats;
    for (const OutputMember& member : members) {
        if (member.target >= Maxwell::NumRenderTargets) {
            continue;
        }
        const LogicFormat format = state.Format(member.target);
        const bool uint_format = IsUint(format);
        const bool sint_format = IsSint(format);
        const bool normalized = format != LogicFormat::None && !uint_format && !sint_format;
        if (format == LogicFormat::None || (normalized && member.type != "float4") ||
            (uint_format && member.type != "uint4") || (sint_format && member.type != "int4")) {
            continue;
        }
        const std::string dst = fmt::format("lop_dst{}", member.target);
        parameters += fmt::format(",\n    {} {} [[color({})]]", member.type, dst, member.target);
        const std::string value = fmt::format("{}.{}", variable, member.name);
        const char* const function = uint_format   ? "lop_uint"
                                     : sint_format ? "lop_sint"
                                     : (format == LogicFormat::Snorm8 ||
                                        format == LogicFormat::Snorm16)
                                         ? "lop_snorm"
                                         : "lop_unorm";
        apply += fmt::format("        {} = {}({}, {}, {});\n", value, function, value, dst,
                             LogicBits(format));
    }
    if (apply.empty()) {
        return false;
    }
    // Back to front, so the positions found stay where they are.
    const std::string replacement =
        fmt::format("{{\n{}        return {};\n    }}", apply, variable);
    for (const size_t at : returns) {
        msl.replace(at, std::string_view{"return ;"}.size() + variable.size(), replacement);
    }
    // An empty parameter list takes the first parameter without the comma.
    size_t end = entry.parameters_end;
    bool empty = true;
    for (size_t pos = entry.parameters_begin; pos < end; ++pos) {
        if (msl[pos] != ' ' && msl[pos] != '\n') {
            empty = false;
            break;
        }
    }
    if (empty) {
        parameters.erase(0, 1);
    }
    msl.insert(entry.parameters_end, parameters);
    // The helpers go before the output struct, which every use of them follows.
    const std::string helpers = fmt::format(
        "static inline uint4 lop_bits(uint4 s, uint4 d) {{ return {}; }}\n"
        "static inline uint4 lop_mask(uint4 bits) {{\n"
        "    return select((uint4(1u) << bits) - uint4(1u), uint4(~0u), bits >= uint4(32u));\n"
        "}}\n"
        "static inline uint4 lop_uint(uint4 s, uint4 d, uint4 bits) {{\n"
        "    return lop_bits(s, d) & lop_mask(bits);\n"
        "}}\n"
        "static inline int4 lop_sint(int4 s, int4 d, uint4 bits) {{\n"
        "    const uint4 shift = uint4(32u) - min(max(bits, uint4(1u)), uint4(32u));\n"
        "    const uint4 r = lop_bits(as_type<uint4>(s), as_type<uint4>(d)) & lop_mask(bits);\n"
        "    return int4(r << shift) >> int4(shift);\n"
        "}}\n"
        "static inline float4 lop_unorm(float4 s, float4 d, uint4 bits) {{\n"
        "    const uint4 mask = lop_mask(bits);\n"
        "    const float4 scale = float4(max(mask, uint4(1u)));\n"
        "    const uint4 a = uint4(floor(saturate(s) * scale + 0.5));\n"
        "    const uint4 b = uint4(floor(saturate(d) * scale + 0.5));\n"
        "    return select(s, float4(lop_bits(a, b) & mask) / scale, bits > uint4(0u));\n"
        "}}\n"
        "static inline float4 lop_snorm(float4 s, float4 d, uint4 bits) {{\n"
        "    const uint4 mask = lop_mask(bits);\n"
        "    const float4 scale = float4(max(mask >> 1u, uint4(1u)));\n"
        "    const uint4 a = as_type<uint4>(int4(floor(clamp(s, -1.0, 1.0) * scale + 0.5))) & mask;\n"
        "    const uint4 b = as_type<uint4>(int4(floor(clamp(d, -1.0, 1.0) * scale + 0.5))) & mask;\n"
        "    const uint4 shift = uint4(32u) - max(bits, uint4(1u));\n"
        "    const int4 r = int4(lop_bits(a, b) << shift) >> int4(shift);\n"
        "    return select(s, max(float4(r) / scale, float4(-1.0)), bits > uint4(0u));\n"
        "}}\n\n",
        LogicExpression(state.op));
    const size_t at = msl.find("struct " + entry.output);
    if (at == std::string::npos) {
        return false;
    }
    msl.insert(at, helpers);
    return true;
}

bool ViewportSwizzleEnabled() noexcept {
    static const bool enabled = EnvironmentOn("EDEN_NX2_METAL_VIEWPORT_SWIZZLE");
    return enabled;
}

bool SwizzleInProgram(u32 swizzle) noexcept {
    swizzle &= 0x7777;
    return swizzle != IdentitySwizzle && swizzle != NegatedYSwizzle;
}

bool ApplyViewportSwizzle(std::string& msl, u32 swizzle) {
    Entry entry;
    if (!FindEntry(msl, "vertex", entry)) {
        return false;
    }
    const std::string variable = OutputVariable(msl, entry);
    if (variable.empty()) {
        return false;
    }
    static constexpr std::string_view Axes = "xyzw";
    std::string components;
    for (u32 index = 0; index < 4; ++index) {
        const u32 source = (swizzle >> (index * 4)) & 7;
        components += fmt::format("{}{}lsw.{}", index != 0 ? ", " : "", (source & 1) != 0 ? "-" : "",
                                  Axes[source >> 1]);
    }
    const std::string position = variable + ".gl_Position";
    const std::string statement =
        fmt::format("    {{\n        const float4 lsw = {};\n        {} = float4({});\n    }}\n",
                    position, position, components);
    // Before the y flip SPIRV-Cross ends the position with, else before the returns.
    const std::string flip = fmt::format("    {}.y = -({}.y);", position, position);
    const size_t flip_at = msl.find(flip, entry.body_begin);
    if (flip_at != std::string::npos) {
        msl.insert(flip_at, statement);
        return true;
    }
    const std::vector<size_t> returns = Returns(msl, entry, variable);
    if (returns.empty() || msl.find(position, entry.body_begin) == std::string::npos) {
        return false;
    }
    for (const size_t at : returns) {
        msl.insert(at, statement);
    }
    return true;
}

} // namespace Metal::RasterPatch
