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

} // Anonymous namespace

int main() {
    @autoreleasepool {
        return Run();
    }
}
