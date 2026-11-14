// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: an offscreen check of the Metal paths the renderer uses, for machines
// without a window (CI): a clear and a draw in raw Metal, then the renderer's own pieces (SPIR-V
// -> MSL, the scheduler's render passes and compute, render pipeline states from Maxwell state,
// rewritten quad indices, occlusion counting, texel buffer views), each read back and compared.
// Exit 0 passed, 1 failed, 2 no Metal device.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>

#include "texel_buffer_smoke_comp_spv.h"
#include "translate_smoke_comp_spv.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"

namespace {

constexpr uint32_t Size = 64;

constexpr const char* ShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut smoke_vertex(uint id [[vertex_id]]) {
    // One triangle covering the whole target.
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    VertexOut out;
    out.position = float4(positions[id], 0.0, 1.0);
    return out;
}

fragment float4 smoke_fragment(VertexOut in [[stage_in]], constant float4& colour [[buffer(0)]]) {
    return colour;
}
)";

struct Texel {
    uint8_t r, g, b, a;
};

bool Near(Texel texel, std::array<uint8_t, 4> expected) {
    const auto close = [](uint8_t a, uint8_t b) { return (a > b ? a - b : b - a) <= 1; };
    return close(texel.r, expected[0]) && close(texel.g, expected[1]) &&
           close(texel.b, expected[2]) && close(texel.a, expected[3]);
}

int Run() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        std::printf("metal-smoke: no Metal device\n");
        return 2;
    }
    std::printf("metal-smoke: device %s, Apple9 %d, Mac2 %d, Metal3 %d\n", device.name.UTF8String,
                [device supportsFamily:MTLGPUFamilyApple9],
                [device supportsFamily:MTLGPUFamilyMac2],
                [device supportsFamily:MTLGPUFamilyMetal3]);

    NSError* error = nil;
    MTLCompileOptions* options = [MTLCompileOptions new];
    id<MTLLibrary> library = [device newLibraryWithSource:@(ShaderSource)
                                                  options:options
                                                    error:&error];
    if (library == nil) {
        std::printf("metal-smoke: FAIL shader compile: %s\n",
                    error.localizedDescription.UTF8String);
        return 1;
    }
    MTLRenderPipelineDescriptor* pipeline_desc = [MTLRenderPipelineDescriptor new];
    pipeline_desc.vertexFunction = [library newFunctionWithName:@"smoke_vertex"];
    pipeline_desc.fragmentFunction = [library newFunctionWithName:@"smoke_fragment"];
    pipeline_desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pipeline =
        [device newRenderPipelineStateWithDescriptor:pipeline_desc error:&error];
    if (pipeline == nil) {
        std::printf("metal-smoke: FAIL pipeline: %s\n", error.localizedDescription.UTF8String);
        return 1;
    }

    MTLTextureDescriptor* texture_desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                           width:Size
                                                          height:Size
                                                       mipmapped:NO];
    texture_desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    texture_desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> target = [device newTextureWithDescriptor:texture_desc];
    id<MTLBuffer> readback = [device newBufferWithLength:Size * Size * 4
                                                 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];

    // Clear to red, then draw green into the right half only (scissor).
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 0.0, 0.0, 1.0);
    id<MTLRenderCommandEncoder> encoder = [command_buffer renderCommandEncoderWithDescriptor:pass];
    [encoder setRenderPipelineState:pipeline];
    [encoder setScissorRect:MTLScissorRect{Size / 2, 0, Size / 2, Size}];
    const float green[4]{0.0f, 1.0f, 0.0f, 1.0f};
    [encoder setFragmentBytes:green length:sizeof(green) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];

    id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];
    [blit copyFromTexture:target
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(Size, Size, 1)
                    toBuffer:readback
           destinationOffset:0
      destinationBytesPerRow:Size * 4
    destinationBytesPerImage:Size * Size * 4];
    [blit endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        std::printf("metal-smoke: FAIL command buffer: %s\n",
                    command_buffer.error.localizedDescription.UTF8String);
        return 1;
    }

    const auto* texels = static_cast<const Texel*>(readback.contents);
    const Texel left = texels[(Size / 2) * Size + 4];
    const Texel right = texels[(Size / 2) * Size + Size - 4];
    const bool passed = Near(left, {255, 0, 0, 255}) && Near(right, {0, 255, 0, 255});
    std::printf("metal-smoke: left %u,%u,%u,%u right %u,%u,%u,%u: %s (GPU %.3f ms)\n", left.r,
                left.g, left.b, left.a, right.r, right.g, right.b, right.a,
                passed ? "PASS" : "FAIL",
                (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0);
    return passed ? 0 : 1;
}

/// The renderer's shader path: SPIR-V through metal_shader_translate, built by Metal, run.
int RunTranslate() {
    namespace Translate = Metal::ShaderTranslate;
    const Translate::Result msl = Translate::SpirvToMsl(
        std::span<const uint32_t>(TRANSLATE_SMOKE_SPV), Translate::Stage::Compute, {});
    if (msl.source.empty()) {
        std::printf("metal-smoke: FAIL SPIR-V -> MSL: %s\n", msl.error.c_str());
        return 1;
    }
    uint32_t params_slot = UINT32_MAX;
    uint32_t data_slot = UINT32_MAX;
    for (const Translate::ResourceSlot& slot : msl.slots) {
        if (slot.kind == Translate::ResourceKind::Buffer) {
            (slot.binding == 0 ? params_slot : data_slot) = slot.index;
        }
    }
    if (params_slot == UINT32_MAX || data_slot == UINT32_MAX) {
        std::printf("metal-smoke: FAIL translated shader has no buffer slots\n");
        return 1;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:@(msl.source.c_str())
                                                  options:[MTLCompileOptions new]
                                                    error:&error];
    if (library == nil) {
        std::printf("metal-smoke: FAIL translated MSL does not build: %s\n%s\n",
                    error.localizedDescription.UTF8String, msl.source.c_str());
        return 1;
    }
    id<MTLFunction> function = [library newFunctionWithName:@(msl.entry_point.c_str())];
    id<MTLComputePipelineState> pipeline =
        function != nil ? [device newComputePipelineStateWithFunction:function error:&error]
                        : nil;
    if (pipeline == nil) {
        std::printf("metal-smoke: FAIL compute pipeline (entry %s): %s\n", msl.entry_point.c_str(),
                    error != nil ? error.localizedDescription.UTF8String : "no function");
        return 1;
    }
    constexpr uint32_t Count = 256;
    id<MTLBuffer> data = [device newBufferWithLength:Count * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    const uint32_t params[4]{7, 3, 0, 0};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:params length:sizeof(params) atIndex:params_slot];
    [encoder setBuffer:data offset:0 atIndex:data_slot];
    [encoder dispatchThreadgroups:MTLSizeMake(Count / 64, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const auto* values = static_cast<const uint32_t*>(data.contents);
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < Count; ++index) {
        wrong += values[index] != index * 7 + 3 ? 1 : 0;
    }
    std::printf("metal-smoke: SPIR-V -> MSL compute (%zu bytes of MSL, entry %s, buffers %u/%u): "
                "%u of %u wrong: %s\n",
                msl.source.size(), msl.entry_point.c_str(), params_slot, data_slot, wrong, Count,
                wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

constexpr const char* QuadSource = R"(
#include <metal_stdlib>
using namespace metal;

struct QuadIn {
    float2 position [[attribute(0)]];
};

struct QuadOut {
    float4 position [[position]];
};

vertex QuadOut quad_vertex(QuadIn in [[stage_in]]) {
    QuadOut out;
    out.position = float4(in.position, 0.0, 1.0);
    return out;
}

fragment float4 quad_fragment() {
    return float4(0.0, 1.0, 0.0, 1.0);
}
)";

/// A Maxwell quad with 8-bit indices: IndexConvert, a render pipeline state built from Maxwell
/// state (vertex stream at the renderer's slot), the scheduler's render pass, occlusion counting.
int RunRendererDraw() {
    using Maxwell = Metal::Maxwell;
    Metal::Device device;
    Metal::Scheduler scheduler(device);

    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex = library ? Metal::CreateFunction(library.Get(), "quad_vertex")
                                         : Metal::Object{};
    const Metal::Object fragment =
        library ? Metal::CreateFunction(library.Get(), "quad_fragment") : Metal::Object{};
    if (!vertex || !fragment) {
        std::printf("metal-smoke: FAIL quad shaders: %s\n", error.c_str());
        return 1;
    }
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = vertex.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    desc.blend[0].write_mask = 0xF;
    desc.samples = 1;
    desc.topology_class = 2;
    desc.rasterize = 1;
    desc.attributes[0] = {1, 0, 0, Maxwell::VertexAttribute::Type::Float,
                          Maxwell::VertexAttribute::Size::Size_R32_G32};
    desc.streams[0] = {1, 8, 0};
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL render pipeline from Maxwell state: %s\n", error.c_str());
        return 1;
    }

    // The left half of the target, as one quad of four vertices.
    uint8_t* vertices = nullptr;
    const Metal::Object vertex_buffer = Metal::CreateBuffer(device, 32, true, &vertices);
    const float positions[8]{-1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, -1.0f, 1.0f};
    std::memcpy(vertices, positions, sizeof(positions));
    const std::array<uint8_t, 4> guest_indices{0, 1, 2, 3};
    const auto topology = Maxwell::PrimitiveTopology::Quads;
    const uint32_t count = Metal::IndexConvert::ConvertedCount(topology, 4);
    uint8_t* index_data = nullptr;
    const Metal::Object index_buffer =
        Metal::CreateBuffer(device, count * sizeof(uint32_t), true, &index_data);
    Metal::IndexConvert::Convert(topology, guest_indices, 1, 0, 4, false,
                                 std::span(reinterpret_cast<uint32_t*>(index_data), count));

    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm),
                 .width = Size,
                 .height = Size,
                 .render_target = true});
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer = Metal::CreateBuffer(device, Size * Size * 4, true, &readback);

    Metal::ClearPass clear{};
    clear.color[0] = {.texture = target.Get()};
    clear.clear_color[0] = true;
    clear.color_value[0] = {1.0, 0.0, 0.0, 1.0};
    clear.width = Size;
    clear.height = Size;
    scheduler.Clear(clear);

    Metal::RenderTargetSet targets{};
    targets.color[0] = {.texture = target.Get()};
    targets.width = Size;
    targets.height = Size;
    scheduler.BeginRendering(targets);
    Metal::DrawState state{};
    state.pipeline = pipeline.Get();
    state.depth_stencil = depth_stencil.Get();
    state.viewport = {0.0f, 0.0f, float(Size), float(Size), 0.0f, 1.0f};
    state.scissor = {0, 0, Size, Size};
    scheduler.SetDrawState(state);
    scheduler.BindVertexBuffer(Metal::VertexBufferSlot(0), vertex_buffer.Get(), 0);
    scheduler.VisibilityResults()[0] = 0;
    scheduler.SetVisibility(0);
    const auto primitive =
        Metal::PrimitiveType(Metal::IndexConvert::ConvertedTopology(topology));
    scheduler.DrawIndexed(*primitive, count, 4, index_buffer.Get(), 0, 1, 0, 0);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel left = texels[(Size / 2) * Size + 4];
    const Texel right = texels[(Size / 2) * Size + Size - 4];
    const uint64_t samples = scheduler.VisibilityResults()[0];
    const bool drawn = Near(left, {0, 255, 0, 255}) && Near(right, {255, 0, 0, 255});
    const bool counted = samples == uint64_t{Size} * Size / 2;
    std::printf("metal-smoke: renderer quad (8-bit indices -> %u rewritten): left %u,%u,%u right "
                "%u,%u,%u: %s; occlusion %llu samples (expected %u): %s\n",
                count, left.r, left.g, left.b, right.r, right.g, right.b,
                drawn ? "PASS" : "FAIL", static_cast<unsigned long long>(samples),
                Size * Size / 2, counted ? "PASS" : "FAIL");
    return drawn && counted ? 0 : 1;
}

/// A texel buffer read: GLSL samplerBuffer -> SPIR-V -> translator (native texture_buffer), the
/// renderer's texture buffer view at an aligned offset, the scheduler's compute encoder.
int RunTexelBuffer() {
    namespace Translate = Metal::ShaderTranslate;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Translate::Result msl = Translate::SpirvToMsl(
        std::span<const uint32_t>(TEXEL_BUFFER_SMOKE_SPV), Translate::Stage::Compute, {});
    if (msl.source.empty()) {
        std::printf("metal-smoke: FAIL texel buffer SPIR-V -> MSL: %s\n", msl.error.c_str());
        return 1;
    }
    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, msl.source, error);
    const Metal::Object function =
        library ? Metal::CreateFunction(library.Get(), msl.entry_point) : Metal::Object{};
    const Metal::Object state =
        function ? Metal::CreateComputePipelineState(device, function.Get(), error)
                 : Metal::Object{};
    if (!state) {
        std::printf("metal-smoke: FAIL texel buffer pipeline: %s\n%s\n", error.c_str(),
                    msl.source.c_str());
        return 1;
    }
    uint32_t texture_slot = UINT32_MAX;
    uint32_t data_slot = UINT32_MAX;
    for (const Translate::ResourceSlot& slot : msl.slots) {
        if (slot.kind == Translate::ResourceKind::Texture && slot.binding == 0) {
            texture_slot = slot.index;
        } else if (slot.kind == Translate::ResourceKind::Buffer && slot.binding == 1) {
            data_slot = slot.index;
        }
    }
    constexpr uint32_t Count = 64;
    const uint32_t format = static_cast<uint32_t>(MTLPixelFormatR32Float);
    const uint32_t offset = Metal::TextureBufferAlignment(device, format);
    uint8_t* source_data = nullptr;
    const Metal::Object source =
        Metal::CreateBuffer(device, offset + Count * sizeof(float), true, &source_data);
    for (uint32_t index = 0; index < Count; ++index) {
        const float value = static_cast<float>(index) + 0.5f;
        std::memcpy(source_data + offset + index * sizeof(float), &value, sizeof(value));
    }
    const Metal::Object view =
        Metal::CreateTextureBufferView(source.Get(), offset, Count, 4, format, false);
    uint8_t* out_data = nullptr;
    const Metal::Object out = Metal::CreateBuffer(device, Count * sizeof(float), true, &out_data);
    if (!view || texture_slot == UINT32_MAX || data_slot == UINT32_MAX) {
        std::printf("metal-smoke: FAIL texel buffer view (offset %u) or slots\n", offset);
        return 1;
    }
    scheduler.BeginCompute(state.Get());
    scheduler.BindTexture(Metal::ShaderStage::Compute, texture_slot, view.Get());
    scheduler.BindBuffer(Metal::ShaderStage::Compute, data_slot, out.Get(), 0);
    scheduler.Dispatch({1, 1, 1}, {Count, 1, 1});
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < Count; ++index) {
        float value;
        std::memcpy(&value, out_data + index * sizeof(float), sizeof(value));
        wrong += value != (static_cast<float>(index) + 0.5f) * 2.0f ? 1 : 0;
    }
    std::printf("metal-smoke: texel buffer (view at offset %u, texture slot %u): %u of %u wrong: "
                "%s\n",
                offset, texture_slot, wrong, Count, wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

} // Anonymous namespace
int main() {
    @autoreleasepool {
        const int draw = Run();
        if (draw == 2) {
            return 2;
        }
        int result = draw;
        for (const auto test : {RunTranslate, RunRendererDraw, RunTexelBuffer}) {
            const int outcome = test();
            result = result != 0 ? result : outcome;
        }
        return result;
    }
}
