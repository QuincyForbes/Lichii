// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Metal answers some descriptors with an NSError and others with an assertion that aborts the
// process, also in release builds (asking a vertex descriptor for its 32nd buffer layout is one:
// Ocarina of Time's first draw has vertex stream 31 bound). Vulkan takes the guest's state as it
// is, so here it is brought into Metal's terms first, and what is left over is refused in words.

#include <algorithm>
#include <cstdlib>

#include <fmt/format.h>

#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

namespace {

using Type = Maxwell::VertexAttribute::Type;
using Size = Maxwell::VertexAttribute::Size;

[[nodiscard]] bool EnvironmentSwitch(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

[[nodiscard]] const char* TypeName(Type type) {
    switch (type) {
    case Type::SNorm:
        return "snorm";
    case Type::UNorm:
        return "unorm";
    case Type::SInt:
        return "sint";
    case Type::UInt:
        return "uint";
    case Type::UScaled:
        return "uscaled";
    case Type::SScaled:
        return "sscaled";
    case Type::Float:
        return "float";
    default:
        return "unused";
    }
}

[[nodiscard]] const char* SizeName(Size size) {
    switch (size) {
    case Size::Size_R32_G32_B32_A32:
        return "r32g32b32a32";
    case Size::Size_R32_G32_B32:
        return "r32g32b32";
    case Size::Size_R16_G16_B16_A16:
        return "r16g16b16a16";
    case Size::Size_R32_G32:
        return "r32g32";
    case Size::Size_R16_G16_B16:
        return "r16g16b16";
    case Size::Size_R8_G8_B8_A8:
        return "r8g8b8a8";
    case Size::Size_R16_G16:
        return "r16g16";
    case Size::Size_R32:
        return "r32";
    case Size::Size_R8_G8_B8:
        return "r8g8b8";
    case Size::Size_R8_G8:
        return "r8g8";
    case Size::Size_R16:
        return "r16";
    case Size::Size_R8:
        return "r8";
    case Size::Size_A2_B10_G10_R10:
        return "a2b10g10r10";
    case Size::Size_B10_G11_R11:
        return "b10g11r11";
    case Size::Size_G8_R8:
        return "g8r8";
    case Size::Size_X8_B8_G8_R8:
        return "x8b8g8r8";
    case Size::Size_A8:
        return "a8";
    default:
        return "invalid";
    }
}

[[nodiscard]] bool HasDepth(u32 format) {
    const FormatClass format_class = ClassOf(format);
    return format_class == FormatClass::Depth || format_class == FormatClass::DepthStencil;
}

[[nodiscard]] bool HasStencil(u32 format) {
    const FormatClass format_class = ClassOf(format);
    return format_class == FormatClass::Stencil || format_class == FormatClass::DepthStencil;
}

} // Anonymous namespace

bool PipelineCheckEnabled() {
    static const bool enabled = EnvironmentSwitch("EDEN_NX2_METAL_PIPELINE_CHECK");
    return enabled;
}

bool VertexUnpackEnabled() {
    static const bool enabled = EnvironmentSwitch("EDEN_NX2_METAL_VERTEX_UNPACK");
    return enabled;
}

const char* VertexUnpackKernelSource() noexcept {
    // Maxwell's A2B10G10R10: red in the low ten bits, alpha in the top two.
    return R"(
#include <metal_stdlib>
using namespace metal;

struct VertexUnpackParams {
    uint count;
    uint stride;
    uint is_signed;
    uint offset;
};

kernel void unpack_1010102(device const uchar* source [[buffer(0)]],
                           device ushort4* elements [[buffer(1)]],
                           constant VertexUnpackParams& params [[buffer(2)]],
                           uint element [[thread_position_in_grid]]) {
    if (element >= params.count) {
        return;
    }
    const uint at = params.offset + element * params.stride;
    const uint value = uint(source[at]) | (uint(source[at + 1]) << 8) |
                       (uint(source[at + 2]) << 16) | (uint(source[at + 3]) << 24);
    const uint4 fields = uint4(value & 0x3FFu, (value >> 10) & 0x3FFu, (value >> 20) & 0x3FFu,
                               value >> 30);
    if (params.is_signed != 0) {
        const uint4 widths = uint4(22u, 22u, 22u, 30u);
        elements[element] = ushort4(short4(int4(fields << widths) >> int4(widths)));
    } else {
        elements[element] = ushort4(fields);
    }
}
)";
}

bool VertexGatherEnabled() {
    static const bool enabled = EnvironmentSwitch("EDEN_NX2_METAL_VERTEX_GATHER");
    return enabled;
}

const char* VertexGatherKernelSource() noexcept {
    return R"(
#include <metal_stdlib>
using namespace metal;

struct VertexGatherParams {
    uint count;
    uint stride;
    uint offset;
    uint bytes;
    uint gathered_stride;
    uint gathered_offset;
};

kernel void gather_vertices(device const uchar* source [[buffer(0)]],
                            device uchar* gathered [[buffer(1)]],
                            constant VertexGatherParams& params [[buffer(2)]],
                            uint element [[thread_position_in_grid]]) {
    if (element >= params.count) {
        return;
    }
    const uint from = params.offset + element * params.stride;
    const uint to = params.gathered_offset + element * params.gathered_stride;
    for (uint byte = 0; byte < params.bytes; ++byte) {
        gathered[to + byte] = source[from + byte];
    }
}
)";
}

bool BootTraceEnabled() {
    static const bool enabled = EnvironmentSwitch("EDEN_NX2_METAL_ABORT_TRACE");
    return enabled;
}

// metal_graphics.mm's VertexFormat has a format for exactly the pairs with a size here.
u32 VertexFormatBytes(Type type, Size size) {
    if (type == Type::Float) {
        switch (size) {
        case Size::Size_R16:
            return 2;
        case Size::Size_R16_G16:
        case Size::Size_R32:
        case Size::Size_B10_G11_R11:
            return 4;
        case Size::Size_R16_G16_B16:
            return 6;
        case Size::Size_R16_G16_B16_A16:
        case Size::Size_R32_G32:
            return 8;
        case Size::Size_R32_G32_B32:
            return 12;
        case Size::Size_R32_G32_B32_A32:
            return 16;
        default:
            return 0;
        }
    }
    switch (size) {
    case Size::Size_R8:
    case Size::Size_A8:
        return 1;
    case Size::Size_R8_G8:
    case Size::Size_G8_R8:
    case Size::Size_R16:
        return 2;
    case Size::Size_R8_G8_B8:
        return 3;
    case Size::Size_R8_G8_B8_A8:
    case Size::Size_X8_B8_G8_R8:
    case Size::Size_R16_G16:
    case Size::Size_R32:
    case Size::Size_A2_B10_G10_R10:
        return 4;
    case Size::Size_R16_G16_B16:
        return 6;
    case Size::Size_R16_G16_B16_A16:
    case Size::Size_R32_G32:
        return 8;
    case Size::Size_R32_G32_B32:
        return 12;
    case Size::Size_R32_G32_B32_A32:
        return 16;
    default:
        return 0;
    }
}

bool SanitizeVertexLayout(RenderPipelineDesc& desc, u32 loaded, u32 resource_slots,
                          VertexSources& sources, std::string& reason) {
    // Entries nothing reads stay zero: equal layouts hash alike.
    auto attributes = Zeroed<decltype(desc.attributes)>();
    auto streams = Zeroed<decltype(desc.streams)>();
    VertexSources found{};
    for (u32 index = 0; index < desc.attributes.size(); ++index) {
        const VertexAttributeState& input = desc.attributes[index];
        const u32 bytes = VertexFormatBytes(input.type, input.size);
        // Vulkan declares an attribute when the guest enables it and the vertex program loads
        // it (vk_graphics_pipeline.cpp); nothing else reaches MoltenVK's vertex descriptor.
        if (input.enabled == 0 || ((loaded >> index) & 1) == 0 || bytes == 0) {
            continue;
        }
        if (index >= MaxVertexSlots) {
            reason = fmt::format("the vertex function reads attribute {}, Metal has {}", index,
                                 MaxVertexSlots);
            return false;
        }
        // An attribute of a stream without a buffer reads zeros, as Vulkan's null buffer gives.
        VertexSource source{};
        VertexStreamState layout{.enabled = 1, .stride = 16, .divisor = ConstantStep};
        u32 offset = 0;
        Size size = input.size;
        const bool bound =
            input.stream < desc.streams.size() && desc.streams[input.stream].enabled != 0;
        const bool packed_integer =
            VertexUnpackEnabled() && input.size == Size::Size_A2_B10_G10_R10 &&
            input.type != Type::UNorm && input.type != Type::SNorm && input.type != Type::Float;
        if (bound && packed_integer) {
            // Metal has 10:10:10:2 only normalized: a kernel widens the attribute into a buffer
            // of its own, which the program reads as the same integers (VertexSource::unpack).
            const VertexStreamState& stream = desc.streams[input.stream];
            const bool constant = stream.stride == 0 || stream.divisor == ConstantStep;
            source.stream = input.stream;
            source.shift = input.offset;
            source.unpack = input.type == Type::SInt || input.type == Type::SScaled ? 2U : 1U;
            layout.stride = 8;
            layout.divisor = constant ? ConstantStep : stream.divisor;
            size = Size::Size_R16_G16_B16_A16;
        } else if (bound) {
            const VertexStreamState& stream = desc.streams[input.stream];
            source.stream = input.stream;
            if (stream.stride == 0 || stream.divisor == ConstantStep) {
                // One element for every vertex: Metal wants a stride all the same, and the
                // attribute inside it.
                source.shift = input.offset & ~15U;
                offset = input.offset & 15U;
                layout.stride = 32;
            } else {
                layout.stride = stream.stride;
                layout.divisor = stream.divisor;
                offset = input.offset;
                if (offset != 0 && offset + bytes > stream.stride) {
                    // Vulkan lets an offset pass the stride, Metal doesn't: the attribute moves
                    // into the stride and reads a copy of the stream bound that much further on
                    // (MoltenVK's translated vertex bindings).
                    offset %= stream.stride;
                    if (offset + bytes > stream.stride) {
                        offset = 0;
                    }
                    source.shift = input.offset - offset;
                }
            }
        }
        u32 slot = 0;
        while (slot < found.count && found.streams[slot] != source) {
            ++slot;
        }
        // The last free slot goes to the gathered buffer when more streams follow; an attribute
        // of a per-vertex stream past it is copied there.
        const bool gatherable = VertexGatherEnabled() && bound && source.unpack == 0 &&
                                layout.divisor == 0 && layout.stride != 0;
        if (slot == found.count && gatherable &&
            (found.gather_slot != ~0U || found.count + resource_slots + 1 == MaxVertexSlots)) {
            if (found.gather_slot == ~0U) {
                found.gather_slot = found.count++;
                found.streams[found.gather_slot] = {.stream = VertexSource::Gathered};
            }
            const u32 gathered_offset = found.gather_stride;
            found.gathered[found.gathered_count++] = {
                .stream = input.stream,
                .offset = input.offset,
                .bytes = bytes,
                .gathered_offset = gathered_offset,
            };
            found.gather_stride += (bytes + 3) & ~3U;
            streams[found.gather_slot] = {.enabled = 1, .stride = found.gather_stride, .divisor = 0};
            attributes[index] = {1, found.gather_slot, gathered_offset, input.type, size};
            continue;
        }
        if (slot == found.count) {
            // Stream n sits at buffer slot 30 - n, the function's own buffers at 0 up.
            if (found.count + resource_slots >= MaxVertexSlots) {
                reason = fmt::format("more than {} vertex streams next to {} shader buffers, "
                                     "Metal has {} buffer slots",
                                     found.count, resource_slots, MaxVertexSlots);
                return false;
            }
            found.streams[slot] = source;
            streams[slot] = layout;
            ++found.count;
        }
        attributes[index] = {1, slot, offset, input.type, size};
    }
    desc.attributes = attributes;
    desc.streams = streams;
    sources = found;
    return true;
}

std::string CheckVertexLayout(const RenderPipelineDesc& desc) {
    for (u32 index = 0; index < desc.streams.size(); ++index) {
        if (desc.streams[index].enabled != 0 && index >= MaxVertexSlots) {
            return fmt::format("vertex stream {} is bound, Metal has {} buffer layouts", index,
                               MaxVertexSlots);
        }
    }
    for (u32 index = 0; index < desc.attributes.size(); ++index) {
        const VertexAttributeState& attribute = desc.attributes[index];
        const u32 bytes = VertexFormatBytes(attribute.type, attribute.size);
        if (attribute.enabled == 0 || bytes == 0) {
            continue;
        }
        if (index >= MaxVertexSlots) {
            return fmt::format("vertex attribute {} is enabled, Metal has {}", index,
                               MaxVertexSlots);
        }
        if (attribute.stream >= MaxVertexSlots) {
            return fmt::format("vertex attribute {} reads stream {}, Metal has {} buffer layouts",
                               index, attribute.stream, MaxVertexSlots);
        }
        // An attribute past its stream's stride builds (metal-smoke's descriptor probe).
        if (desc.streams[attribute.stream].enabled == 0) {
            return fmt::format("vertex attribute {} reads stream {}, which has no buffer", index,
                               attribute.stream);
        }
    }
    return {};
}

std::string CheckPipeline(const RenderPipelineDesc& desc, const PipelineLimits& limits) {
    if (desc.vertex_function == nullptr) {
        return "no vertex function";
    }
    const u32 samples = desc.samples != 0 ? desc.samples : 1;
    if (samples >= 32 || ((limits.sample_counts >> samples) & 1) == 0) {
        return fmt::format("{} samples per pixel, the device takes mask {:#x}", samples,
                           limits.sample_counts);
    }
    for (u32 index = 0; index < desc.color_formats.size(); ++index) {
        const u32 format = desc.color_formats[index];
        if (format != 0 && (HasDepth(format) || HasStencil(format))) {
            return fmt::format("colour target {} has depth or stencil format {}", index, format);
        }
    }
    if (desc.depth_format != 0 && !HasDepth(desc.depth_format)) {
        return fmt::format("depth format {} has no depth", desc.depth_format);
    }
    if (desc.stencil_format != 0 && !HasStencil(desc.stencil_format)) {
        return fmt::format("stencil format {} has no stencil", desc.stencil_format);
    }
    // Depth alone next to stencil alone builds; a format holding both has to be both.
    if (desc.depth_format != 0 && desc.stencil_format != 0 &&
        desc.depth_format != desc.stencil_format &&
        (HasStencil(desc.depth_format) || HasDepth(desc.stencil_format))) {
        return fmt::format("depth format {} next to stencil format {}", desc.depth_format,
                           desc.stencil_format);
    }
    return CheckVertexLayout(desc);
}

std::string CheckTexture(const TextureDesc& desc, const PipelineLimits& limits) {
    // Every Apple GPU since the A9 and every Mac one: 16384 texels a side, 2048 for 3D, 2048
    // layers.
    static constexpr u32 MaxSide = 16384;
    static constexpr u32 MaxVolumeSide = 2048;
    static constexpr u32 MaxLayers = 2048;
    if (desc.format == 0) {
        return "no pixel format";
    }
    const bool multisample =
        desc.kind == TextureKind::e2DMultisample || desc.kind == TextureKind::e2DMultisampleArray;
    const u32 samples = (std::max)(desc.samples, 1U);
    if (samples >= 32 || ((limits.sample_counts >> samples) & 1) == 0) {
        return fmt::format("{} samples per pixel, the device takes mask {:#x}", samples,
                           limits.sample_counts);
    }
    if (multisample != (samples > 1)) {
        return fmt::format("{} samples per pixel in a texture that is {}multisampled", samples,
                           multisample ? "" : "not ");
    }
    const u32 side = desc.kind == TextureKind::e3D ? MaxVolumeSide : MaxSide;
    const u32 depth = desc.kind == TextureKind::e3D ? desc.depth : 1;
    if (desc.width > side || desc.height > side || depth > side) {
        return fmt::format("{}x{}x{} texels, Metal has {} a side", desc.width, desc.height, depth,
                           side);
    }
    const bool line = desc.kind == TextureKind::e1D || desc.kind == TextureKind::e1DArray;
    if ((line || desc.kind == TextureKind::e3D) &&
        (HasDepth(desc.format) || HasStencil(desc.format))) {
        return fmt::format("depth or stencil format {} in a {} texture", desc.format,
                           line ? "1D" : "3D");
    }
    if (line && (desc.height > 1 || desc.levels > 1)) {
        return fmt::format("a 1D texture {} texels high with {} levels, Metal has one of each",
                           desc.height, desc.levels);
    }
    const bool cube = desc.kind == TextureKind::Cube || desc.kind == TextureKind::CubeArray;
    if (cube && desc.width != desc.height) {
        return fmt::format("cube faces of {}x{} texels", desc.width, desc.height);
    }
    if ((cube ? desc.layers / 6 : desc.layers) > MaxLayers) {
        return fmt::format("{} layers, Metal has {}", desc.layers, MaxLayers);
    }
    u32 levels = 1;
    for (u32 size = (std::max)({desc.width, desc.height, depth}); size > 1; size >>= 1) {
        ++levels;
    }
    if (desc.levels > (multisample ? 1 : levels)) {
        return fmt::format("{} levels, {}x{}x{} texels have {}", desc.levels, desc.width,
                           desc.height, depth, multisample ? 1 : levels);
    }
    return {};
}

std::string CheckTextureView(TextureKind base, TextureKind view, u32 layers) {
    // Views stay among 1D, among 2D and cube, among multisampled and among 3D textures.
    const auto family = [](TextureKind kind) {
        switch (kind) {
        case TextureKind::e1D:
        case TextureKind::e1DArray:
            return 0;
        case TextureKind::e2DMultisample:
        case TextureKind::e2DMultisampleArray:
            return 2;
        case TextureKind::e3D:
            return 3;
        default:
            return 1;
        }
    };
    if (family(base) != family(view)) {
        return fmt::format("kind {} seen as kind {}", static_cast<u32>(base),
                           static_cast<u32>(view));
    }
    u32 wanted = layers;
    switch (view) {
    case TextureKind::e1D:
    case TextureKind::e2D:
    case TextureKind::e2DMultisample:
    case TextureKind::e3D:
        wanted = 1;
        break;
    case TextureKind::Cube:
        wanted = 6;
        break;
    case TextureKind::CubeArray:
        wanted = layers - layers % 6;
        break;
    default:
        break;
    }
    if (layers != wanted || layers == 0) {
        return fmt::format("{} slices in a view of kind {}", layers, static_cast<u32>(view));
    }
    return {};
}

std::string Describe(const RenderPipelineDesc& desc) {
    std::string text = "colour";
    bool any = false;
    for (u32 index = 0; index < desc.color_formats.size(); ++index) {
        if (desc.color_formats[index] == 0) {
            continue;
        }
        any = true;
        const BlendTargetState& blend = desc.blend[index];
        text += fmt::format(" {}:format {} mask {:x}", index, desc.color_formats[index],
                            blend.write_mask);
        if (blend.enable != 0) {
            text += fmt::format(" blend {:#x}/{:#x} {:#x},{:#x}/{:#x},{:#x}",
                                static_cast<u32>(blend.color_op), static_cast<u32>(blend.alpha_op),
                                static_cast<u32>(blend.color_source),
                                static_cast<u32>(blend.color_dest),
                                static_cast<u32>(blend.alpha_source),
                                static_cast<u32>(blend.alpha_dest));
        }
    }
    if (!any) {
        text += " none";
    }
    text += fmt::format("; depth {} stencil {}; samples {}; topology {}; rasterize {}; "
                        "alpha to coverage {}; attributes",
                        desc.depth_format, desc.stencil_format, desc.samples, desc.topology_class,
                        desc.rasterize, desc.alpha_to_coverage);
    any = false;
    for (u32 index = 0; index < desc.attributes.size(); ++index) {
        const VertexAttributeState& attribute = desc.attributes[index];
        if (attribute.enabled == 0) {
            continue;
        }
        any = true;
        text += fmt::format(" {}:{} {} +{} stream {}", index, TypeName(attribute.type),
                            SizeName(attribute.size), attribute.offset, attribute.stream);
    }
    if (!any) {
        text += " none";
    }
    text += "; streams";
    any = false;
    for (u32 index = 0; index < desc.streams.size(); ++index) {
        const VertexStreamState& stream = desc.streams[index];
        if (stream.enabled == 0) {
            continue;
        }
        any = true;
        text += fmt::format(" {}:stride {}", index, stream.stride);
        if (stream.stride == 0 || stream.divisor == ConstantStep) {
            text += " constant";
        } else if (stream.divisor != 0) {
            text += fmt::format(" per {} instances", stream.divisor);
        }
    }
    if (!any) {
        text += " none";
    }
    if (desc.tessellation != 0) {
        text += fmt::format("; tessellation partition {} winding {}", desc.tess_partition,
                            desc.tess_winding);
    }
    return text;
}

} // namespace Metal
