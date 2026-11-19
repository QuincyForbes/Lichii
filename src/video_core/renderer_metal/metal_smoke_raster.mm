// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke's raster state cases (metal-coverage.md section B): the logic operations and
// viewport swizzles done in the translated programs (metal_raster_patch.h, on SPIRV-Cross's own
// output for GLSL programs), triangles drawn as points, the primitive restart off, and the depth
// bounds test where the GPU has it. Each is drawn on the runner's device, read back and compared.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <fmt/format.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "raster_line_frag_spv.h"
#include "raster_line_vert_spv.h"
#include "raster_smoke_frag_spv.h"
#include "raster_smoke_sint_frag_spv.h"
#include "raster_smoke_uint_frag_spv.h"
#include "raster_smoke_vert_spv.h"
#include "video_core/renderer_metal/metal_draw_desc.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_raster_expand.h"
#include "video_core/renderer_metal/metal_raster_patch.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_metal/metal_smoke_raster.h"

namespace SmokeRaster {

namespace {

namespace Patch = Metal::RasterPatch;
namespace Translate = Metal::ShaderTranslate;

constexpr uint32_t Size = 32;

struct Context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

/// A render target of `format`, cleared to `clear` (colour attachment 0 only).
struct Target {
    id<MTLTexture> texture;
};

Target MakeTarget(const Context& context, MTLPixelFormat format) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:Size
                                                                                   height:Size
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    return {[context.device newTextureWithDescriptor:desc]};
}

/// One render pass: `draw` encodes into it after the target is cleared (or loaded, `clear` null).
template <typename Draw>
void Pass(const Context& context, const Target& target, const MTLClearColor* clear,
          id<MTLTexture> depth, const Draw& draw) {
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target.texture;
    pass.colorAttachments[0].loadAction = clear != nullptr ? MTLLoadActionClear : MTLLoadActionLoad;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    if (clear != nullptr) {
        pass.colorAttachments[0].clearColor = *clear;
    }
    if (depth != nil) {
        pass.depthAttachment.texture = depth;
        pass.depthAttachment.loadAction = MTLLoadActionLoad;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
    }
    id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    draw(encoder);
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
}

std::vector<uint8_t> Read(const Target& target, uint32_t bytes_per_pixel) {
    std::vector<uint8_t> bytes(size_t{Size} * Size * bytes_per_pixel);
    [target.texture getBytes:bytes.data()
                 bytesPerRow:Size * bytes_per_pixel
                  fromRegion:MTLRegionMake2D(0, 0, Size, Size)
                 mipmapLevel:0];
    return bytes;
}

struct Library {
    id<MTLLibrary> library;
    std::string error;
};

Library Compile(const Context& context, const std::string& source) {
    NSError* error = nil;
    id<MTLLibrary> library = [context.device newLibraryWithSource:@(source.c_str())
                                                          options:[MTLCompileOptions new]
                                                            error:&error];
    return {library, error != nil ? std::string{error.localizedDescription.UTF8String}
                                  : std::string{}};
}

uint32_t Apply(uint32_t op, uint32_t s, uint32_t d) {
    switch (op) {
    case 0:
        return 0;
    case 1:
        return s & d;
    case 2:
        return s & ~d;
    case 3:
        return s;
    case 4:
        return ~s & d;
    case 5:
        return d;
    case 6:
        return s ^ d;
    case 7:
        return s | d;
    case 8:
        return ~(s | d);
    case 9:
        return ~(s ^ d);
    case 10:
        return ~d;
    case 11:
        return s | ~d;
    case 12:
        return ~s;
    case 13:
        return ~s | d;
    case 14:
        return ~(s & d);
    default:
        return ~0U;
    }
}

id<MTLRenderPipelineState> Pipeline(const Context& context, const Library& vertex,
                                    const std::string& vertex_entry, const Library& fragment,
                                    const std::string& fragment_entry, MTLPixelFormat format,
                                    MTLPrimitiveTopologyClass topology, std::string& error) {
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    desc.vertexFunction = [vertex.library newFunctionWithName:@(vertex_entry.c_str())];
    desc.fragmentFunction = [fragment.library newFunctionWithName:@(fragment_entry.c_str())];
    desc.colorAttachments[0].pixelFormat = format;
    desc.inputPrimitiveTopology = topology;
    NSError* ns_error = nil;
    id<MTLRenderPipelineState> state =
        [context.device newRenderPipelineStateWithDescriptor:desc error:&ns_error];
    if (state == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "no state";
    }
    return state;
}

/// The logic operation on `format` targets: the cleared target's own bytes are the destination,
/// the pixel program's colour the source, the result what the CPU gets by the same operation on the
/// stored bits. `integer`: the pixel program writes integers.
struct LogicCase {
    const char* name;
    MTLPixelFormat format;
    Patch::LogicFormat logic;
    bool integer;
    bool is_signed;
    MTLClearColor clear;
    std::vector<uint32_t> ops;
};

/// The channels' stored value as an unsigned number of `bits` (the bytes of a pixel as the
/// format lays them out), and back.
struct Layout {
    std::array<uint32_t, 4> bits;
    std::array<uint32_t, 4> shift;
};

Layout LayoutOf(Patch::LogicFormat format) {
    switch (format) {
    case Patch::LogicFormat::Unorm10_2:
        return {{10, 10, 10, 2}, {0, 10, 20, 30}};
    default:
        return {{8, 8, 8, 8}, {0, 8, 16, 24}};
    }
}

int RunLogicOps(const Context& context) {
    const std::span<const uint32_t> vertex_spirv{RASTER_SMOKE_VERT_SPV};
    const Translate::Options options;
    const Translate::Result vertex_msl =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::Vertex, options);
    const Translate::Result float_msl = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_SMOKE_FRAG_SPV}, Translate::Stage::Fragment, options);
    const Translate::Result uint_msl = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_SMOKE_UINT_FRAG_SPV}, Translate::Stage::Fragment, options);
    const Translate::Result sint_msl = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_SMOKE_SINT_FRAG_SPV}, Translate::Stage::Fragment, options);
    if (vertex_msl.source.empty() || float_msl.source.empty() || uint_msl.source.empty() ||
        sint_msl.source.empty()) {
        std::printf("metal-smoke: FAIL raster state: the translation failed: %s%s%s%s\n",
                    vertex_msl.error.c_str(), float_msl.error.c_str(), uint_msl.error.c_str(),
                    sint_msl.error.c_str());
        return 1;
    }
    const Library vertex = Compile(context, vertex_msl.source);
    if (vertex.library == nil) {
        std::printf("metal-smoke: FAIL raster state: vertex program: %s\n", vertex.error.c_str());
        return 1;
    }
    std::vector<uint32_t> all_ops(16);
    for (uint32_t op = 0; op < 16; ++op) {
        all_ops[op] = op;
    }
    const std::vector<LogicCase> cases{
        {"RGBA8 unorm", MTLPixelFormatRGBA8Unorm, Patch::LogicFormat::Unorm8, false, false,
         MTLClearColorMake(200.0 / 255, 100.0 / 255, 50.0 / 255, 1.0), all_ops},
        {"RGB10A2 unorm", MTLPixelFormatRGB10A2Unorm, Patch::LogicFormat::Unorm10_2, false, false,
         MTLClearColorMake(700.0 / 1023, 90.0 / 1023, 300.0 / 1023, 1.0), {6, 1, 8}},
        {"RGBA8 uint", MTLPixelFormatRGBA8Uint, Patch::LogicFormat::Uint8, true, false,
         MTLClearColorMake(0xC8, 0x64, 0x32, 0xFF), all_ops},
        {"RGBA8 sint", MTLPixelFormatRGBA8Sint, Patch::LogicFormat::Sint8, true, true,
         MTLClearColorMake(-56, 100, 50, -1), all_ops},
    };
    uint32_t wrong = 0;
    std::string report;
    for (const LogicCase& test : cases) {
        const Target target = MakeTarget(context, test.format);
        for (const uint32_t op : test.ops) {
            Patch::LogicState state{};
            state.op = op;
            state.formats = static_cast<uint32_t>(test.logic);
            const Translate::Result& base =
                test.integer ? (test.is_signed ? sint_msl : uint_msl) : float_msl;
            std::string fragment_source = base.source;
            if (!Patch::ApplyLogicOp(fragment_source, state)) {
                std::printf("metal-smoke: FAIL logic op %s %u: the program was not patched\n%s\n",
                            test.name, op, fragment_source.c_str());
                return 1;
            }
            const Library fragment = Compile(context, fragment_source);
            if (fragment.library == nil) {
                std::printf("metal-smoke: FAIL logic op %s %u: the patched program does not "
                            "build: %s\n%s\n",
                            test.name, op, fragment.error.c_str(), fragment_source.c_str());
                return 1;
            }
            std::string error;
            id<MTLRenderPipelineState> pipeline =
                Pipeline(context, vertex, vertex_msl.entry_point, fragment,
                         base.entry_point, test.format,
                         MTLPrimitiveTopologyClassTriangle, error);
            if (pipeline == nil) {
                std::printf("metal-smoke: FAIL logic op %s %u: pipeline: %s\n", test.name, op,
                            error.c_str());
                return 1;
            }
            Pass(context, target, &test.clear, nil, [](id<MTLRenderCommandEncoder>) {});
            const std::vector<uint8_t> before = Read(target, 4);
            Pass(context, target, nullptr, nil, [&](id<MTLRenderCommandEncoder> encoder) {
                [encoder setRenderPipelineState:pipeline];
                [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            });
            const std::vector<uint8_t> after = Read(target, 4);
            const size_t left = (size_t{Size} / 2 * Size + 4) * 4;
            const size_t right = (size_t{Size} / 2 * Size + Size - 4) * 4;
            uint32_t destination = 0;
            uint32_t result = 0;
            std::memcpy(&destination, before.data() + left, 4);
            std::memcpy(&result, after.data() + left, 4);
            uint32_t expected = 0;
            const Layout layout = LayoutOf(test.logic);
            if (test.integer) {
                const std::array<uint32_t, 4> source{0xA5, 0x3C, 0x0F, 0xF0};
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    const uint32_t d = (destination >> layout.shift[channel]) & 0xFF;
                    expected |= (Apply(op, source[channel], d) & 0xFF) << layout.shift[channel];
                }
            } else {
                // The colour (0.2, 0.4, 0.6, 0.8) as the target stores it.
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    const uint32_t bits = layout.bits[channel];
                    const uint32_t mask = (1U << bits) - 1;
                    const float values[4]{0.2f, 0.4f, 0.6f, 0.8f};
                    const uint32_t s = static_cast<uint32_t>(std::lrint(values[channel] * mask));
                    const uint32_t d = (destination >> layout.shift[channel]) & mask;
                    expected |= (Apply(op, s, d) & mask) << layout.shift[channel];
                }
            }
            const bool untouched = std::memcmp(before.data() + right, after.data() + right, 4) == 0;
            const bool ok = result == expected && untouched;
            wrong += ok ? 0 : 1;
            if (!ok) {
                char line[128];
                std::snprintf(line, sizeof(line), " %s op %u: dst %08x got %08x want %08x%s;",
                              test.name, op, destination, result, expected,
                              untouched ? "" : " (right half changed)");
                report += line;
            }
        }
    }
    std::printf("metal-smoke: logic operations in the pixel program (RGBA8, RGB10A2, RGBA8 uint and sint; "
                "16 operations on all but the second, 3 on it)%s: %s\n",
                report.c_str(), wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

/// The vertex program's rectangle (the left half of the target, x -1..0, y -1..1 as the guest
/// writes it) with a swizzle: negated x lands it on the right half, x and y swapped on the upper
/// half (the guest's y down).
int RunViewportSwizzle(const Context& context) {
    const Translate::Options options;
    const Translate::Result vertex_msl = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_SMOKE_VERT_SPV}, Translate::Stage::Vertex, options);
    const Translate::Result fragment_msl = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_SMOKE_FRAG_SPV}, Translate::Stage::Fragment, options);
    const Library fragment = Compile(context, fragment_msl.source);
    if (vertex_msl.source.empty() || fragment.library == nil) {
        std::printf("metal-smoke: FAIL viewport swizzle: translation: %s%s\n",
                    vertex_msl.error.c_str(), fragment.error.c_str());
        return 1;
    }
    struct Case {
        const char* name;
        uint32_t swizzle;
        /// The quarters of the target the rectangle covers: top left, top right, bottom left,
        /// bottom right.
        std::array<bool, 4> covers;
    };
    constexpr uint32_t Quarter = Size / 4;
    const std::vector<Case> cases{
        {"identity", Patch::IdentitySwizzle, {true, false, true, false}},
        {"-x", 0x6421, {false, true, false, true}},
        {"x and y swapped", 0x6402, {true, true, false, false}},
        {"y and -x", 0x6412, {false, false, true, true}},
    };
    // The identity and the negated y alone are the viewport's (the program is left as it is).
    uint32_t wrong = Patch::SwizzleInProgram(Patch::IdentitySwizzle) ||
                             Patch::SwizzleInProgram(Patch::NegatedYSwizzle) ||
                             !Patch::SwizzleInProgram(0x6421)
                         ? 1
                         : 0;
    std::string report;
    const Target target = MakeTarget(context, MTLPixelFormatRGBA8Unorm);
    const MTLClearColor black = MTLClearColorMake(0, 0, 0, 1);
    for (const Case& test : cases) {
        std::string source = vertex_msl.source;
        if (!Patch::ApplyViewportSwizzle(source, test.swizzle)) {
            std::printf("metal-smoke: FAIL viewport swizzle %s: the program was not patched\n%s\n",
                        test.name, source.c_str());
            return 1;
        }
        const Library vertex = Compile(context, source);
        if (vertex.library == nil) {
            std::printf("metal-smoke: FAIL viewport swizzle %s: does not build: %s\n%s\n",
                        test.name, vertex.error.c_str(), source.c_str());
            return 1;
        }
        std::string error;
        id<MTLRenderPipelineState> pipeline =
            Pipeline(context, vertex, vertex_msl.entry_point, fragment, fragment_msl.entry_point,
                     MTLPixelFormatRGBA8Unorm, MTLPrimitiveTopologyClassTriangle, error);
        if (pipeline == nil) {
            std::printf("metal-smoke: FAIL viewport swizzle %s: pipeline: %s\n", test.name,
                        error.c_str());
            return 1;
        }
        Pass(context, target, &black, nil, [&](id<MTLRenderCommandEncoder> encoder) {
            [encoder setRenderPipelineState:pipeline];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
        });
        const std::vector<uint8_t> pixels = Read(target, 4);
        std::array<bool, 4> got{};
        for (uint32_t quarter = 0; quarter < 4; ++quarter) {
            const uint32_t x = (quarter % 2 == 0 ? 1 : 3) * Quarter;
            const uint32_t y = (quarter < 2 ? 1 : 3) * Quarter;
            got[quarter] = pixels[(size_t{y} * Size + x) * 4] != 0;
        }
        const bool ok = got == test.covers;
        wrong += ok ? 0 : 1;
        char line[96];
        std::snprintf(line, sizeof(line), " %s: TL%d TR%d BL%d BR%d%s", test.name, got[0], got[1],
                      got[2], got[3], ok ? "" : " WRONG");
        report += line;
    }
    std::printf("metal-smoke: viewport swizzles in the vertex program:%s: %s\n", report.c_str(),
                wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

const char* PointSource = R"(
#include <metal_stdlib>
using namespace metal;
struct Out { float4 position [[position]]; float size [[point_size]]; };
vertex Out point_vertex(const device float2* positions [[buffer(0)]], uint id [[vertex_id]]) {
    Out out;
    out.position = float4(positions[id], 0.0, 1.0);
    out.size = 5.0;
    return out;
}
fragment float4 point_fragment() { return float4(1.0, 1.0, 1.0, 1.0); }
)";

/// Triangles whose polygon mode is point are drawn as their vertices (PointFillTopology): the
/// vertex stream of a triangle drawn as points covers the corners and not the inside.
int RunPointFill(const Context& context) {
    using Maxwell = Metal::Maxwell;
    auto regs = std::make_unique<Maxwell>();
    std::memset(static_cast<void*>(regs.get()), 0, sizeof(Maxwell));
    regs->polygon_mode_front = Maxwell::PolygonMode::Point;
    regs->polygon_mode_back = Maxwell::PolygonMode::Point;
    regs->gl_cull_test_enabled = 0;
    const auto mapped = Metal::PointFillTopology(*regs, Maxwell::PrimitiveTopology::Triangles);
    const auto strip = Metal::PointFillTopology(*regs, Maxwell::PrimitiveTopology::TriangleStrip);
    const auto lines = Metal::PointFillTopology(*regs, Maxwell::PrimitiveTopology::Lines);
    regs->polygon_mode_back = Maxwell::PolygonMode::Fill;
    const auto mixed = Metal::PointFillTopology(*regs, Maxwell::PrimitiveTopology::Triangles);
    const bool topology_ok = mapped == Maxwell::PrimitiveTopology::Points &&
                             strip == Maxwell::PrimitiveTopology::Points &&
                             lines == Maxwell::PrimitiveTopology::Lines &&
                             mixed == Maxwell::PrimitiveTopology::Triangles;

    const Library library = Compile(context, PointSource);
    if (library.library == nil) {
        std::printf("metal-smoke: FAIL point fill: %s\n", library.error.c_str());
        return 1;
    }
    std::string error;
    id<MTLRenderPipelineState> pipeline =
        Pipeline(context, library, "point_vertex", library, "point_fragment",
                 MTLPixelFormatRGBA8Unorm, MTLPrimitiveTopologyClassPoint, error);
    if (pipeline == nil) {
        std::printf("metal-smoke: FAIL point fill: pipeline: %s\n", error.c_str());
        return 1;
    }
    // One triangle spanning the target's lower-left half; its corners and its inside.
    const std::array<float, 6> positions{-0.75f, -0.75f, 0.75f, -0.75f, -0.75f, 0.75f};
    id<MTLBuffer> buffer = [context.device newBufferWithBytes:positions.data()
                                                       length:sizeof(positions)
                                                      options:MTLResourceStorageModeShared];
    const Target target = MakeTarget(context, MTLPixelFormatRGBA8Unorm);
    const MTLClearColor black = MTLClearColorMake(0, 0, 0, 1);
    Pass(context, target, &black, nil, [&](id<MTLRenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        [encoder setVertexBuffer:buffer offset:0 atIndex:0];
        [encoder drawPrimitives:static_cast<MTLPrimitiveType>(
                                    *Metal::PrimitiveType(mapped))
                    vertexStart:0
                    vertexCount:3];
    });
    const std::vector<uint8_t> pixels = Read(target, 4);
    const auto covered = [&](float x, float y) {
        // NDC (y up) to the texture's texel.
        const uint32_t px = static_cast<uint32_t>((x * 0.5f + 0.5f) * Size);
        const uint32_t py = static_cast<uint32_t>((0.5f - y * 0.5f) * Size);
        return pixels[(size_t{py} * Size + px) * 4] != 0;
    };
    const bool corners = covered(-0.75f, -0.75f) && covered(0.75f, -0.75f) && covered(-0.75f, 0.75f);
    const bool inside = covered(-0.3f, -0.3f);
    const bool ok = topology_ok && corners && !inside;
    std::printf("metal-smoke: polygon mode point (triangles drawn as their vertices): topology %s, "
                "corners %s, inside %s: %s\n",
                topology_ok ? "ok" : "WRONG", corners ? "drawn" : "MISSING",
                inside ? "FILLED" : "empty", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

const char* StripSource = R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 strip_vertex(const device float2* positions [[buffer(0)]], uint id [[vertex_id]]) {
    return float4(positions[id], 0.0, 1.0);
}
fragment float4 strip_fragment() { return float4(1.0, 1.0, 1.0, 1.0); }
)";

/// A 16-bit strip whose third index is 0xFFFF with the restart off: drawn as it is Metal restarts
/// at it (nothing is drawn); widened to 32 bits with the restart off (IndexConvert::Convert, as the
/// rasterizer does for RestartOffWidens) the triangle is drawn.
int RunRestartOff(const Context& context) {
    namespace IndexConvert = Metal::IndexConvert;
    using Topology = IndexConvert::Topology;
    constexpr uint32_t Vertices = 0x10000;
    std::vector<float> positions(size_t{Vertices} * 2, 0.0f);
    positions[0] = -1.0f;
    positions[1] = -1.0f;
    positions[2] = 1.0f;
    positions[3] = -1.0f;
    positions[(size_t{Vertices} - 1) * 2] = -1.0f;
    positions[(size_t{Vertices} - 1) * 2 + 1] = 1.0f;
    id<MTLBuffer> vertex_buffer = [context.device newBufferWithBytes:positions.data()
                                                              length:positions.size() * sizeof(float)
                                                             options:MTLResourceStorageModeShared];
    const std::array<uint16_t, 3> guest{0, 1, 0xFFFF};
    std::array<uint32_t, 3> widened{};
    std::span<const uint8_t> source{reinterpret_cast<const uint8_t*>(guest.data()),
                                    sizeof(guest)};
    const uint32_t written =
        IndexConvert::Convert(Topology::TriangleStrip, source, 2, 0, 3, false, widened);
    const uint32_t restarted =
        IndexConvert::Convert(Topology::TriangleStrip, source, 2, 0, 3, true, widened);
    const bool strips = IndexConvert::RestartsStrips(Topology::TriangleStrip) &&
                        IndexConvert::RestartsStrips(Topology::LineStrip) &&
                        !IndexConvert::RestartsStrips(Topology::Triangles);
    IndexConvert::Convert(Topology::TriangleStrip, source, 2, 0, 3, false, widened);
    const bool kept = written == 3 && restarted == 3 && widened[2] == 0xFFFF;
    const Library library = Compile(context, StripSource);
    std::string error;
    id<MTLRenderPipelineState> pipeline =
        library.library != nil
            ? Pipeline(context, library, "strip_vertex", library, "strip_fragment",
                       MTLPixelFormatRGBA8Unorm, MTLPrimitiveTopologyClassTriangle, error)
            : nil;
    if (pipeline == nil) {
        std::printf("metal-smoke: FAIL restart off: %s%s\n", library.error.c_str(), error.c_str());
        return 1;
    }
    const Target target = MakeTarget(context, MTLPixelFormatRGBA8Unorm);
    const MTLClearColor black = MTLClearColorMake(0, 0, 0, 1);
    const auto lower_left_painted = [&]() {
        return Read(target, 4)[(size_t{Size - 8} * Size + 8) * 4] != 0;
    };
    id<MTLBuffer> native_indices = [context.device newBufferWithBytes:guest.data()
                                                               length:sizeof(guest)
                                                              options:MTLResourceStorageModeShared];
    Pass(context, target, &black, nil, [&](id<MTLRenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        [encoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip
                            indexCount:3
                             indexType:MTLIndexTypeUInt16
                           indexBuffer:native_indices
                     indexBufferOffset:0];
    });
    const bool native_restarts = !lower_left_painted();
    id<MTLBuffer> wide_indices = [context.device newBufferWithBytes:widened.data()
                                                             length:sizeof(widened)
                                                            options:MTLResourceStorageModeShared];
    Pass(context, target, &black, nil, [&](id<MTLRenderCommandEncoder> encoder) {
        [encoder setRenderPipelineState:pipeline];
        [encoder setVertexBuffer:vertex_buffer offset:0 atIndex:0];
        [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangleStrip
                            indexCount:3
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:wide_indices
                     indexBufferOffset:0];
    });
    const bool widened_drawn = lower_left_painted();
    const bool ok = strips && kept && native_restarts && widened_drawn;
    std::printf("metal-smoke: primitive restart off (16-bit strip with 0xFFFF as a vertex): Metal "
                "restarts at it %s, widened to 32 bits the triangle is %s, rewrite keeps the "
                "index %s: %s\n",
                native_restarts ? "(as known)" : "NOT", widened_drawn ? "drawn" : "MISSING",
                kept && strips ? "yes" : "NO", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

/// The depth bounds test (Apple10 GPUs: setDepthTestMinBound:maxBound:): a triangle at depth
/// 0.5 over a target whose stored depth is 0.5 is drawn with bounds around it and not with bounds
/// above it. Skipped on GPUs without the test.
int RunDepthBounds(const Context& context) {
#if defined(__MAC_26_0) || defined(__IPHONE_26_0)
    if (@available(macOS 26.0, iOS 26.0, *)) {
        if (![context.device supportsFamily:MTLGPUFamilyApple10]) {
            std::printf("metal-smoke: depth bounds: SKIP (the GPU is not an Apple10 family one; "
                        "the test is not applied there)\n");
            return 0;
        }
        const Library library = Compile(context, StripSource);
        std::string error;
        MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
        desc.vertexFunction = [library.library newFunctionWithName:@"strip_vertex"];
        desc.fragmentFunction = [library.library newFunctionWithName:@"strip_fragment"];
        desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        desc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        NSError* ns_error = nil;
        id<MTLRenderPipelineState> pipeline =
            [context.device newRenderPipelineStateWithDescriptor:desc error:&ns_error];
        MTLDepthStencilDescriptor* depth_desc = [MTLDepthStencilDescriptor new];
        depth_desc.depthCompareFunction = MTLCompareFunctionAlways;
        id<MTLDepthStencilState> depth_state = [context.device newDepthStencilStateWithDescriptor:depth_desc];
        MTLTextureDescriptor* texture_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                               width:Size
                                                              height:Size
                                                           mipmapped:NO];
        texture_desc.usage = MTLTextureUsageRenderTarget;
        texture_desc.storageMode = MTLStorageModePrivate;
        id<MTLTexture> depth = [context.device newTextureWithDescriptor:texture_desc];
        if (pipeline == nil || depth == nil) {
            std::printf("metal-smoke: FAIL depth bounds: %s\n", ns_error.localizedDescription.UTF8String);
            return 1;
        }
        const std::array<float, 6> positions{-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f};
        id<MTLBuffer> buffer = [context.device newBufferWithBytes:positions.data()
                                                           length:sizeof(positions)
                                                          options:MTLResourceStorageModeShared];
        const Target target = MakeTarget(context, MTLPixelFormatRGBA8Unorm);
        const MTLClearColor black = MTLClearColorMake(0, 0, 0, 1);
        bool results[2]{};
        const std::array<std::array<float, 2>, 2> bounds{{{0.4f, 0.6f}, {0.6f, 1.0f}}};
        for (size_t index = 0; index < 2; ++index) {
            MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = target.texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].clearColor = black;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.depthAttachment.texture = depth;
            pass.depthAttachment.loadAction = MTLLoadActionClear;
            pass.depthAttachment.clearDepth = 0.5;
            pass.depthAttachment.storeAction = MTLStoreActionStore;
            id<MTLCommandBuffer> command_buffer = [context.queue commandBuffer];
            id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
            [encoder setRenderPipelineState:pipeline];
            [encoder setDepthStencilState:depth_state];
            [encoder setDepthTestMinBound:bounds[index][0] maxBound:bounds[index][1]];
            [encoder setVertexBuffer:buffer offset:0 atIndex:0];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            results[index] = Read(target, 4)[(size_t{Size - 8} * Size + 8) * 4] != 0;
        }
        const bool ok = results[0] && !results[1];
        std::printf("metal-smoke: depth bounds (Apple10): stored depth 0.5 inside [0.4, 0.6] drawn "
                    "%s, outside [0.6, 1] %s: %s\n",
                    results[0] ? "yes" : "NO", results[1] ? "DRAWN" : "discarded",
                    ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
#else
    (void)context;
#endif
    std::printf("metal-smoke: depth bounds: SKIP (the OS or SDK is older than 26)\n");
    return 0;
}


/// Lines and triangles expanded from the vertex program's outputs (RasterExpand): the program
/// runs as a stage-in kernel writing every vertex's outputs, the generated vertex function pulls
/// the ends of each primitive from them. Each case draws one primitive into a 32x32 target and
/// checks pixels.
struct ExpandCase {
    const char* name;
    std::vector<float> positions; ///< NDC x, y per vertex (the guest's y down)
    uint32_t bits;
    uint32_t prim;
    float width;
    float factor;
    uint32_t pattern;
    /// Reads the target: returns "" when right, else what was wrong.
    std::string (*check)(const std::vector<uint8_t>&);
};

uint8_t Pixel(const std::vector<uint8_t>& pixels, uint32_t x, uint32_t y, uint32_t channel) {
    return pixels[(size_t{y} * Size + x) * 4 + channel];
}

std::string CheckWide(const std::vector<uint8_t>& p) {
    // A line at row 16 (y 0) from x 4 to 28, 6 pixels wide: rows 13 to 18, its ends on its axis.
    std::string wrong;
    const auto expect = [&](uint32_t x, uint32_t y, bool drawn) {
        if ((Pixel(p, x, y, 1) != 0) != drawn) {
            wrong += fmt::format(" ({},{}) {}", x, y, drawn ? "empty" : "drawn");
        }
    };
    expect(16, 13, true);
    expect(16, 18, true);
    expect(16, 12, false);
    expect(16, 19, false);
    expect(5, 15, true);
    expect(3, 15, false);
    expect(28, 15, false);
    return wrong;
}

std::string CheckStipple(const std::vector<uint8_t>& p) {
    // Pattern 0x00FF, factor 1: along the line (from x 4) pixels 0-7 on, 8-15 off, 16-23 on.
    std::string wrong;
    const auto expect = [&](uint32_t x, bool drawn) {
        if ((Pixel(p, x, 15, 1) != 0 || Pixel(p, x, 16, 1) != 0) != drawn) {
            wrong += fmt::format(" x {} {}", x, drawn ? "empty" : "drawn");
        }
    };
    expect(6, true);
    expect(14, false);
    expect(22, true);
    return wrong;
}

std::string CheckSmooth(const std::vector<uint8_t>& p) {
    // A line 2 pixels wide centred 0.25 below the row boundary 16: row 16 (distance 0.25) has full
    // coverage, row 15 (distance 0.75) three quarters.
    const uint32_t full = Pixel(p, 16, 16, 3);
    const uint32_t edge = Pixel(p, 16, 15, 3);
    if (full < 250 || edge < 170 || edge > 215) {
        return fmt::format(" alpha {} and {} (want 255 and about 191)", full, edge);
    }
    return {};
}

std::string CheckConservative(const std::vector<uint8_t>& p) {
    // A triangle inside one pixel, covering no pixel centre: the pixel is drawn, the others not.
    std::string wrong;
    if (Pixel(p, 14, 14, 1) == 0) {
        wrong += " (14,14) empty";
    }
    if (Pixel(p, 17, 17, 1) != 0 || Pixel(p, 10, 10, 1) != 0) {
        wrong += " far pixel drawn";
    }
    return wrong;
}

int RunRasterExpand(const Context& context) {
    namespace Expand = Metal::RasterExpand;
    const Translate::Options options;
    const std::span<const uint32_t> vertex_spirv{RASTER_LINE_VERT_SPV};
    const Translate::Result normal =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::Vertex, options);
    const Translate::Result kernel =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::VertexForTessellation, options);
    const Translate::Result fragment = Translate::SpirvToMsl(
        std::span<const uint32_t>{RASTER_LINE_FRAG_SPV}, Translate::Stage::Fragment, options);
    if (normal.source.empty() || kernel.source.empty() || fragment.source.empty()) {
        std::printf("metal-smoke: FAIL raster expansion: the translation failed: %s%s%s\n",
                    normal.error.c_str(), kernel.error.c_str(), fragment.error.c_str());
        return 1;
    }
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    std::string error;
    const Metal::Object kernel_library = Metal::CreateLibrary(device, kernel.source, error);
    const Metal::Object kernel_function =
        kernel_library ? Metal::CreateFunction(kernel_library.Get(), kernel.entry_point)
                       : Metal::Object{};
    auto layout = Metal::Zeroed<Metal::RenderPipelineDesc>();
    layout.attributes[0] = {1, 0, 0, Metal::Maxwell::VertexAttribute::Type::Float,
                            Metal::Maxwell::VertexAttribute::Size::Size_R32_G32};
    layout.streams[0] = {1, 8, 0};
    const Metal::Object kernel_state =
        kernel_function ? Metal::CreateStageInComputePipelineState(
                              device, kernel_function.Get(), layout, 0,
                              kernel.tessellation_slots.index, error)
                        : Metal::Object{};
    if (!kernel_state) {
        std::printf("metal-smoke: FAIL raster expansion: capture kernel: %s\n%s\n", error.c_str(),
                    kernel.source.c_str());
        return 1;
    }
    const std::vector<ExpandCase> cases{
        {"wide line", {-0.75f, 0.0f, 0.75f, 0.0f}, Expand::Wide, Expand::PrimLines, 6.0f, 1.0f,
         0xFFFF, CheckWide},
        {"stippled line", {-0.75f, 0.0f, 0.75f, 0.0f}, Expand::Wide | Expand::Stipple,
         Expand::PrimLines, 2.0f, 1.0f, 0x00FF, CheckStipple},
        {"smooth line", {-0.75f, 0.015625f, 0.75f, 0.015625f}, Expand::Wide | Expand::Smooth,
         Expand::PrimLines, 2.0f, 1.0f, 0xFFFF, CheckSmooth},
        {"conservative triangle",
         {14.1f / 16 - 1, 14.1f / 16 - 1, 14.4f / 16 - 1, 14.1f / 16 - 1, 14.1f / 16 - 1,
          14.4f / 16 - 1},
         Expand::Conservative, Expand::PrimTriangles, 1.0f, 1.0f, 0xFFFF, CheckConservative},
    };
    uint32_t wrong = 0;
    std::string report;
    for (const ExpandCase& test : cases) {
        std::string fragment_source = fragment.source;
        const std::string generated =
            Expand::Generate(normal.source, kernel.source, test.bits, test.prim, 0);
        if (generated.empty() || !Expand::PatchFragment(fragment_source, test.bits)) {
            std::printf("metal-smoke: FAIL raster expansion %s: not generated (%s)\n%s\n", test.name,
                        generated.empty() ? "vertex function" : "pixel program",
                        normal.source.c_str());
            return 1;
        }
        const Library vertex_library = Compile(context, generated);
        const Library fragment_library = Compile(context, fragment_source);
        if (vertex_library.library == nil || fragment_library.library == nil) {
            std::printf("metal-smoke: FAIL raster expansion %s: does not build: %s%s\n%s\n%s\n",
                        test.name, vertex_library.error.c_str(), fragment_library.error.c_str(),
                        generated.c_str(), fragment_source.c_str());
            return 1;
        }
        std::string pipeline_error;
        id<MTLRenderPipelineState> pipeline =
            Pipeline(context, vertex_library, Expand::Entry, fragment_library,
                     fragment.entry_point, MTLPixelFormatRGBA8Unorm,
                     MTLPrimitiveTopologyClassTriangle, pipeline_error);
        if (pipeline == nil) {
            std::printf("metal-smoke: FAIL raster expansion %s: pipeline: %s\n%s\n%s\n", test.name,
                        pipeline_error.c_str(), generated.c_str(), fragment_source.c_str());
            return 1;
        }
        const uint32_t count = static_cast<uint32_t>(test.positions.size() / 2);
        uint8_t* position_data = nullptr;
        const Metal::Object position_buffer =
            Metal::CreateBuffer(device, test.positions.size() * sizeof(float), true, &position_data);
        std::memcpy(position_data, test.positions.data(), test.positions.size() * sizeof(float));
        uint8_t* output_data = nullptr;
        const Metal::Object output_buffer =
            Metal::CreateBuffer(device, size_t{count} * 512, true, &output_data);
        scheduler.BeginCompute(kernel_state.Get());
        scheduler.BindBuffer(Metal::ShaderStage::Compute, Metal::VertexBufferSlot(0),
                             position_buffer.Get(), 0);
        scheduler.BindBuffer(Metal::ShaderStage::Compute, kernel.tessellation_slots.output,
                             output_buffer.Get(), 0);
        scheduler.DispatchStageIn(0, count, 0, 1);
        scheduler.Finish();
        Expand::Params params{};
        params.half_viewport[0] = params.half_viewport[1] = Size / 2.0f;
        params.width = test.width;
        params.prim = test.prim;
        params.count = count;
        params.factor = test.factor;
        params.pattern = test.pattern;
        const uint32_t vertices = Expand::VerticesOf(test.prim, count);
        const Target target = MakeTarget(context, MTLPixelFormatRGBA8Unorm);
        const MTLClearColor black = MTLClearColorMake(0, 0, 0, 0);
        Pass(context, target, &black, nil, [&](id<MTLRenderCommandEncoder> encoder) {
            [encoder setRenderPipelineState:pipeline];
            [encoder setCullMode:MTLCullModeNone];
            [encoder setVertexBuffer:(__bridge id<MTLBuffer>)output_buffer.Get()
                              offset:0
                             atIndex:Expand::SourceSlot];
            [encoder setVertexBytes:&params length:sizeof(params) atIndex:Expand::ParamsSlot];
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:vertices];
        });
        const std::string problem = test.check(Read(target, 4));
        wrong += problem.empty() ? 0 : 1;
        report += fmt::format(" {}{}{};", test.name, problem.empty() ? "" : " WRONG:", problem);
    }
    std::printf("metal-smoke: raster expansion from the vertex kernel's outputs:%s %s\n",
                report.c_str(), wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

} // Anonymous namespace

int RunAll() {
    Context context{MTLCreateSystemDefaultDevice(), nil};
    if (context.device == nil) {
        return 2;
    }
    context.queue = [context.device newCommandQueue];
    int result = 0;
    for (const auto test : {RunLogicOps, RunViewportSwizzle, RunPointFill, RunRestartOff,
                            RunDepthBounds, RunRasterExpand}) {
        const int outcome = test(context);
        result = result != 0 ? result : outcome;
    }
    return result;
}

} // namespace SmokeRaster
