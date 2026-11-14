// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: an offscreen check of the Metal paths the renderer uses, for machines
// without a window (CI). Clears a texture, draws a triangle with a shader compiled at run time,
// copies the texture to a shared buffer and compares texels. Exit 0 passed, 1 failed, 2 no
// Metal device.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <span>

#include "translate_smoke_comp_spv.h"
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

} // Anonymous namespace

int main() {
    @autoreleasepool {
        const int draw = Run();
        if (draw == 2) {
            return 2;
        }
        const int translate = RunTranslate();
        return draw != 0 ? draw : translate;
    }
}
