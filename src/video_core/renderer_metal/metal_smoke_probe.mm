// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Metal answers some descriptors with an NSError and others with an assertion that aborts the
// process. Each suspect here is built in a child (this binary run again with --probe <name>), so
// an abort is a line of the report with Metal's own words, and the tests go on. The report holds
// for the machine it ran on: the checks inside Metal.framework's descriptor classes are the same
// code on every Apple system, a GPU driver's are not.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iterator>
#include <string>
#include <utility>
#include <vector>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_smoke_probe.h"

extern char** environ;

namespace SmokeProbe {

namespace {

constexpr const char* Source = R"(
#include <metal_stdlib>
using namespace metal;

struct ProbeIn {
    float2 position [[attribute(0)]];
};

struct ProbeOut {
    float4 position [[position]];
};

vertex ProbeOut probe_vertex(ProbeIn in [[stage_in]]) {
    ProbeOut out;
    out.position = float4(in.position, 0.0, 1.0);
    return out;
}

vertex ProbeOut probe_vertex_plain(uint id [[vertex_id]]) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    ProbeOut out;
    out.position = float4(positions[id], 0.5, 1.0);
    return out;
}

fragment float4 probe_fragment() {
    return float4(0.0, 1.0, 0.0, 1.0);
}

fragment uint4 probe_fragment_uint() {
    return uint4(1, 2, 3, 4);
}

struct ProbeTwo {
    float4 first [[color(0)]];
    uint4 second [[color(1)]];
};

fragment ProbeTwo probe_fragment_two() {
    ProbeTwo out;
    out.first = float4(0.0, 1.0, 0.0, 1.0);
    out.second = uint4(1, 2, 3, 4);
    return out;
}
)";

struct Context {
    id<MTLDevice> device;
    id<MTLFunction> vertex;
    id<MTLFunction> vertex_plain;
    id<MTLFunction> fragment;
    id<MTLFunction> fragment_uint;
    id<MTLFunction> fragment_two;
};

/// Where the renderer puts vertex stream 0.
constexpr NSUInteger Slot = Metal::VertexBufferSlot(0);

constexpr NSUInteger TargetSize = 64;

/// The descriptor every suspect starts from: one float2 attribute of one 8-byte stream, one
/// RGBA8 target.
MTLRenderPipelineDescriptor* Quad(const Context& context) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = context.vertex;
    descriptor.fragmentFunction = context.fragment;
    descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    MTLVertexDescriptor* vertex = [MTLVertexDescriptor vertexDescriptor];
    vertex.attributes[0].format = MTLVertexFormatFloat2;
    vertex.attributes[0].offset = 0;
    vertex.attributes[0].bufferIndex = Slot;
    vertex.layouts[Slot].stride = 8;
    vertex.layouts[Slot].stepFunction = MTLVertexStepFunctionPerVertex;
    descriptor.vertexDescriptor = vertex;
    return descriptor;
}

/// The same without vertex streams or targets, as a draw that only counts samples has it.
MTLRenderPipelineDescriptor* Targetless(const Context& context) {
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = context.vertex_plain;
    descriptor.fragmentFunction = context.fragment;
    return descriptor;
}

std::string ErrorText(NSError* error) {
    return std::string("error: ") +
           (error != nil ? error.localizedDescription.UTF8String : "none given");
}

std::string Build(const Context& context, MTLRenderPipelineDescriptor* descriptor) {
    NSError* error = nil;
    const id<MTLRenderPipelineState> state =
        [context.device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    return state != nil ? "built" : ErrorText(error);
}

enum class Pass {
    Colour,   ///< One 64x64 RGBA8 target
    Sized,    ///< No attachments, 64x64 by renderTargetWidth and Height
    Sampled,  ///< The same with defaultRasterSampleCount 1
    Unsized,  ///< No attachments, no size
};

/// Builds the pipeline, draws three vertices with it and reports the samples that passed, when
/// they are counted.
std::string Draw(const Context& context, MTLRenderPipelineDescriptor* descriptor, Pass kind,
                 const std::function<void(id<MTLRenderCommandEncoder>)>& setup,
                 bool counted = true,
                 const std::function<void(MTLRenderPassDescriptor*)>& prepare = {}) {
    NSError* error = nil;
    const id<MTLRenderPipelineState> state =
        [context.device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (state == nil) {
        return ErrorText(error);
    }
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    if (kind == Pass::Colour) {
        MTLTextureDescriptor* target =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                               width:TargetSize
                                                              height:TargetSize
                                                           mipmapped:NO];
        target.usage = MTLTextureUsageRenderTarget;
        target.storageMode = MTLStorageModePrivate;
        pass.colorAttachments[0].texture = [context.device newTextureWithDescriptor:target];
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    } else if (kind != Pass::Unsized) {
        pass.renderTargetWidth = TargetSize;
        pass.renderTargetHeight = TargetSize;
        if (kind == Pass::Sampled) {
            pass.defaultRasterSampleCount = 1;
        }
    }
    const id<MTLBuffer> visibility =
        [context.device newBufferWithLength:sizeof(uint64_t) options:MTLResourceStorageModeShared];
    if (counted) {
        pass.visibilityResultBuffer = visibility;
    }
    if (prepare) {
        prepare(pass);
    }
    const id<MTLCommandQueue> queue = [context.device newCommandQueue];
    const id<MTLCommandBuffer> commands = [queue commandBuffer];
    const id<MTLRenderCommandEncoder> encoder =
        [commands renderCommandEncoderWithDescriptor:pass];
    if (encoder == nil) {
        return "error: no render command encoder";
    }
    [encoder setRenderPipelineState:state];
    if (counted) {
        [encoder setVisibilityResultMode:MTLVisibilityResultModeCounting offset:0];
    }
    if (setup) {
        setup(encoder);
    }
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) {
        return std::string("error: command buffer: ") +
               (commands.error != nil ? commands.error.localizedDescription.UTF8String : "?");
    }
    if (!counted) {
        return "built and drawn, samples not counted";
    }
    uint64_t samples = 0;
    std::memcpy(&samples, visibility.contents, sizeof(samples));
    return "built and drawn, " + std::to_string(samples) + " samples passed";
}

std::string Samples(const Context& context, NSUInteger samples) {
    MTLRenderPipelineDescriptor* descriptor = Quad(context);
    descriptor.rasterSampleCount = samples;
    return Build(context, descriptor) + " (supportsTextureSampleCount " +
           ([context.device supportsTextureSampleCount:samples] ? "yes)" : "no)");
}

std::string Texture(const Context& context, NSUInteger width, NSUInteger levels,
                    NSUInteger samples) {
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = samples > 1 ? MTLTextureType2DMultisample : MTLTextureType2D;
    descriptor.pixelFormat = MTLPixelFormatRGBA8Unorm;
    descriptor.width = width;
    descriptor.height = 16;
    descriptor.mipmapLevelCount = levels;
    descriptor.sampleCount = samples;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModePrivate;
    return [context.device newTextureWithDescriptor:descriptor] != nil ? "built"
                                                                       : "error: no texture";
}

id<MTLTexture> Shape(const Context& context, MTLTextureType type, MTLPixelFormat format,
                     NSUInteger width, NSUInteger height, NSUInteger depth, NSUInteger layers,
                     NSUInteger levels, MTLTextureUsage usage) {
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = type;
    descriptor.pixelFormat = format;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.depth = depth;
    descriptor.arrayLength = layers;
    descriptor.mipmapLevelCount = levels;
    descriptor.sampleCount = type == MTLTextureType2DMultisample ? 4 : 1;
    descriptor.usage = usage;
    descriptor.storageMode = MTLStorageModePrivate;
    return [context.device newTextureWithDescriptor:descriptor];
}

std::string Shaped(const Context& context, MTLTextureType type, MTLPixelFormat format,
                   NSUInteger width, NSUInteger height, NSUInteger depth, NSUInteger layers,
                   NSUInteger levels = 1, MTLTextureUsage usage = MTLTextureUsageShaderRead) {
    return Shape(context, type, format, width, height, depth, layers, levels, usage) != nil
               ? "built"
               : "error: no texture";
}

/// A view of a 16x16 RGBA8 texture of 6 layers and 5 levels (1D: 16x1 with one level;
/// multisampled and 3D: one layer and level), made with the usage the renderer's textures have.
std::string View(const Context& context, MTLTextureType base_type, MTLTextureType type,
                 MTLPixelFormat format, NSRange levels, NSRange slices) {
    const bool flat = base_type == MTLTextureType2DMultisample || base_type == MTLTextureType3D;
    const bool line = base_type == MTLTextureType1DArray;
    const id<MTLTexture> base =
        Shape(context, base_type, MTLPixelFormatRGBA8Unorm, 16, line ? 1 : 16,
              base_type == MTLTextureType3D ? 16 : 1, flat ? 1 : 6, flat || line ? 1 : 5,
              MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
    if (base == nil) {
        return "error: no texture";
    }
    return [base newTextureViewWithPixelFormat:format
                                   textureType:type
                                        levels:levels
                                        slices:slices] != nil
               ? "built"
               : "error: no view";
}

/// A 2D view in format `to` of a 16x16 texture in format `from`.
std::string Cast(const Context& context, MTLPixelFormat from, MTLPixelFormat to) {
    if (from >= MTLPixelFormatBC1_RGBA && from <= MTLPixelFormatBC7_RGBAUnorm_sRGB &&
        !context.device.supportsBCTextureCompression) {
        return "error: no BC textures on this device";
    }
    const id<MTLTexture> base = Shape(context, MTLTextureType2D, from, 16, 16, 1, 1, 1,
                                      MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
    if (base == nil) {
        return "error: no texture";
    }
    return [base newTextureViewWithPixelFormat:to
                                   textureType:MTLTextureType2D
                                        levels:NSMakeRange(0, 1)
                                        slices:NSMakeRange(0, 1)] != nil
               ? "built"
               : "error: no view";
}

std::string Sampler(const Context& context, NSUInteger anisotropy, float min_lod, float max_lod) {
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.maxAnisotropy = anisotropy;
    descriptor.lodMinClamp = min_lod;
    descriptor.lodMaxClamp = max_lod;
    return [context.device newSamplerStateWithDescriptor:descriptor] != nil ? "built"
                                                                            : "error: no sampler";
}

struct Suspect {
    const char* name;
    std::string (*run)(const Context&);
};

// The first is the control: it has to build.
const Suspect Suspects[] = {
    {"control", [](const Context& context) { return Build(context, Quad(context)); }},
    // The vertex descriptor's arrays. Stream 31 at the renderer's slot 30 - 31 is the first.
    {"layout-index-of-stream-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Metal::VertexBufferSlot(31)].stride = 16;
         return Build(context, descriptor);
     }},
    {"layout-index-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[31].stride = 16;
         return Build(context, descriptor);
     }},
    {"attribute-index-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[31].format = MTLVertexFormatFloat2;
         descriptor.vertexDescriptor.attributes[31].bufferIndex = Slot;
         return Build(context, descriptor);
     }},
    {"attribute-buffer-index-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].bufferIndex = 31;
         return Build(context, descriptor);
     }},
    {"colour-attachment-index-8",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[8].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Build(context, descriptor);
     }},
    // The vertex layout.
    {"attribute-without-layout",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Slot].stride = 0;
         return Build(context, descriptor);
     }},
    {"unread-attribute-without-layout",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[5].format = MTLVertexFormatFloat4;
         descriptor.vertexDescriptor.attributes[5].bufferIndex = Slot - 3;
         return Build(context, descriptor);
     }},
    {"offset-past-stride",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].offset = 8;
         return Build(context, descriptor);
     }},
    {"attribute-straddles-stride",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].offset = 4;
         return Build(context, descriptor);
     }},
    {"stride-not-multiple-of-4",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Slot].stride = 10;
         return Build(context, descriptor);
     }},
    {"offset-not-multiple-of-4",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].offset = 2;
         descriptor.vertexDescriptor.layouts[Slot].stride = 12;
         return Build(context, descriptor);
     }},
    {"constant-step-with-rate-1",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Slot].stepFunction = MTLVertexStepFunctionConstant;
         descriptor.vertexDescriptor.layouts[Slot].stepRate = 1;
         return Build(context, descriptor);
     }},
    {"constant-step-with-rate-0",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Slot].stepFunction = MTLVertexStepFunctionConstant;
         descriptor.vertexDescriptor.layouts[Slot].stepRate = 0;
         return Build(context, descriptor);
     }},
    {"per-instance-step-with-rate-0",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.layouts[Slot].stepFunction = MTLVertexStepFunctionPerInstance;
         descriptor.vertexDescriptor.layouts[Slot].stepRate = 0;
         return Build(context, descriptor);
     }},
    {"attribute-the-function-reads-missing",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor = [MTLVertexDescriptor vertexDescriptor];
         return Build(context, descriptor);
     }},
    // The same by way of the descriptor's getter, as the other suspects change theirs: has to
    // give the answer above.
    {"attribute-the-function-reads-removed",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].format = MTLVertexFormatInvalid;
         return Build(context, descriptor);
     }},
    {"attribute-format-uint-for-float-input",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexDescriptor.attributes[0].format = MTLVertexFormatUInt2;
         return Build(context, descriptor);
     }},
    // Targets.
    {"no-attachments",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
         return Build(context, descriptor);
     }},
    {"no-attachments-no-rasterization",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
         descriptor.rasterizationEnabled = NO;
         return Build(context, descriptor);
     }},
    {"depth-format-as-stencil",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float;
         return Build(context, descriptor);
     }},
    {"stencil-format-as-depth",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatStencil8;
         return Build(context, descriptor);
     }},
    {"depth32-next-to-stencil8",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
         descriptor.stencilAttachmentPixelFormat = MTLPixelFormatStencil8;
         return Build(context, descriptor);
     }},
    {"depth32-next-to-depth32stencil8",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
         descriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
         return Build(context, descriptor);
     }},
    {"depth32stencil8-as-both",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
         descriptor.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
         return Build(context, descriptor);
     }},
    {"colour-format-as-depth",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatRGBA8Unorm;
         return Build(context, descriptor);
     }},
    {"depth-format-as-colour",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatDepth32Float;
         return Build(context, descriptor);
     }},
    {"samples-2", [](const Context& context) { return Samples(context, 2); }},
    {"samples-3", [](const Context& context) { return Samples(context, 3); }},
    {"samples-4", [](const Context& context) { return Samples(context, 4); }},
    {"samples-8", [](const Context& context) { return Samples(context, 8); }},
    {"samples-16", [](const Context& context) { return Samples(context, 16); }},
    {"blending-on-rgba8uint",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.fragmentFunction = context.fragment_uint;
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Uint;
         descriptor.colorAttachments[0].blendingEnabled = YES;
         return Build(context, descriptor);
     }},
    {"blending-on-rgba32float",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
         descriptor.colorAttachments[0].blendingEnabled = YES;
         return Build(context, descriptor);
     }},
    {"float-fragment-on-rgba8uint",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Uint;
         return Build(context, descriptor);
     }},
    {"alpha-to-coverage-on-rgba8uint",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.fragmentFunction = context.fragment_uint;
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Uint;
         descriptor.alphaToCoverageEnabled = YES;
         return Build(context, descriptor);
     }},
    {"alpha-to-coverage-without-targets",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatInvalid;
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
         descriptor.alphaToCoverageEnabled = YES;
         return Build(context, descriptor);
     }},
    {"alpha-to-coverage-with-rgba8uint-second-target",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.fragmentFunction = context.fragment_two;
         descriptor.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA8Uint;
         descriptor.alphaToCoverageEnabled = YES;
         return Build(context, descriptor);
     }},
    // Functions.
    {"no-vertex-function",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexFunction = nil;
         return Build(context, descriptor);
     }},
    {"no-fragment-function",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.fragmentFunction = nil;
         return Build(context, descriptor);
     }},
    {"fragment-function-as-vertex",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Quad(context);
         descriptor.vertexFunction = context.fragment;
         return Build(context, descriptor);
     }},
    // Textures and samplers.
    {"texture-zero-width", [](const Context& context) { return Texture(context, 0, 1, 1); }},
    {"texture-32768-wide", [](const Context& context) { return Texture(context, 32768, 1, 1); }},
    {"texture-too-many-levels", [](const Context& context) { return Texture(context, 16, 9, 1); }},
    {"texture-zero-levels", [](const Context& context) { return Texture(context, 16, 0, 1); }},
    {"texture-3-samples", [](const Context& context) { return Texture(context, 16, 1, 3); }},
    {"texture-4-samples", [](const Context& context) { return Texture(context, 16, 1, 4); }},
    {"texture-multisample-with-1-sample",
     [](const Context& context) {
         MTLTextureDescriptor* descriptor =
             [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                width:16
                                                               height:16
                                                            mipmapped:NO];
         descriptor.textureType = MTLTextureType2DMultisample;
         descriptor.usage = MTLTextureUsageRenderTarget;
         descriptor.storageMode = MTLStorageModePrivate;
         return std::string([context.device newTextureWithDescriptor:descriptor] != nil
                                ? "built"
                                : "error: no texture");
     }},
    {"texture-2d-with-4-samples",
     [](const Context& context) {
         MTLTextureDescriptor* descriptor =
             [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                width:16
                                                               height:16
                                                            mipmapped:NO];
         descriptor.sampleCount = 4;
         descriptor.usage = MTLTextureUsageRenderTarget;
         descriptor.storageMode = MTLStorageModePrivate;
         return std::string([context.device newTextureWithDescriptor:descriptor] != nil
                                ? "built"
                                : "error: no texture");
     }},
    {"texture-cube-not-square",
     [](const Context& context) {
         return Shaped(context, MTLTextureTypeCube, MTLPixelFormatRGBA8Unorm, 16, 8, 1, 1);
     }},
    {"texture-array-2048-layers",
     [](const Context& context) {
         return Shaped(context, MTLTextureType2DArray, MTLPixelFormatRGBA8Unorm, 4, 4, 1, 2048);
     }},
    {"texture-array-4096-layers",
     [](const Context& context) {
         return Shaped(context, MTLTextureType2DArray, MTLPixelFormatRGBA8Unorm, 4, 4, 1, 4096);
     }},
    {"texture-3d-4096-deep",
     [](const Context& context) {
         return Shaped(context, MTLTextureType3D, MTLPixelFormatRGBA8Unorm, 4, 4, 4096, 1);
     }},
    {"texture-3d-depth-format",
     [](const Context& context) {
         return Shaped(context, MTLTextureType3D, MTLPixelFormatDepth32Float, 4, 4, 4, 1);
     }},
    // 1D textures, which the texture cache makes for the guest's 1D images, as render targets
    // when their format can be one.
    {"texture-1d-array",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 1, 1, 2);
     }},
    {"texture-1d-array-as-render-target",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 1, 1, 2, 1,
                       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
     }},
    {"texture-1d-array-with-pixel-format-view",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 1, 1, 2, 1,
                       MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
     }},
    {"texture-1d-array-with-shader-write",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 1, 1, 2, 1,
                       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
     }},
    {"texture-1d-array-with-5-levels",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 1, 1, 2, 5);
     }},
    {"texture-1d-array-depth-format",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatDepth32Float, 16, 1, 1, 2);
     }},
    {"texture-1d-array-2-texels-high",
     [](const Context& context) {
         return Shaped(context, MTLTextureType1DArray, MTLPixelFormatRGBA8Unorm, 16, 2, 1, 2);
     }},
    {"texture-2d-array-depth-with-shader-write",
     [](const Context& context) {
         return Shaped(context, MTLTextureType2DArray, MTLPixelFormatDepth32Float, 16, 16, 1, 2, 1,
                       MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
     }},
    // Views, of the type the shader asks for.
    {"view-2d-of-2d-array",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 5), NSMakeRange(1, 1));
     }},
    {"view-cube-of-2d-array",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureTypeCube, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 5), NSMakeRange(0, 6));
     }},
    {"view-cube-of-5-slices",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureTypeCube, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 5), NSMakeRange(0, 5));
     }},
    {"view-2d-of-2-slices",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 5), NSMakeRange(0, 2));
     }},
    {"view-1d-of-2d-array",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType1D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-2d-of-1d-array",
     [](const Context& context) {
         return View(context, MTLTextureType1DArray, MTLTextureType2D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-1d-of-1d-array",
     [](const Context& context) {
         return View(context, MTLTextureType1DArray, MTLTextureType1D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-2d-of-3d",
     [](const Context& context) {
         return View(context, MTLTextureType3D, MTLTextureType2D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-2d-of-multisample",
     [](const Context& context) {
         return View(context, MTLTextureType2DMultisample, MTLTextureType2D,
                     MTLPixelFormatRGBA8Unorm, NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-levels-past-the-end",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatRGBA8Unorm,
                     NSMakeRange(4, 2), NSMakeRange(0, 1));
     }},
    {"view-slices-past-the-end",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2DArray,
                     MTLPixelFormatRGBA8Unorm, NSMakeRange(0, 1), NSMakeRange(5, 2));
     }},
    {"view-rgba8-as-rgba8uint",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatRGBA8Uint,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-rgba8-as-r32float",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatR32Float,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-rgba8-as-rg8",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatRG8Unorm,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-rgba8-as-depth32float",
     [](const Context& context) {
         return View(context, MTLTextureType2DArray, MTLTextureType2D, MTLPixelFormatDepth32Float,
                     NSMakeRange(0, 1), NSMakeRange(0, 1));
     }},
    {"view-2d-array-of-2d",
     [](const Context& context) {
         const id<MTLTexture> base =
             Shape(context, MTLTextureType2D, MTLPixelFormatRGBA8Unorm, 16, 16, 1, 1, 1,
                   MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView);
         return [base newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                        textureType:MTLTextureType2DArray
                                             levels:NSMakeRange(0, 1)
                                             slices:NSMakeRange(0, 1)] != nil
                    ? std::string{"built"}
                    : std::string{"error: no view"};
     }},
    // Views in another format: which casts Metal takes.
    {"view-rgba8-as-bgra8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA8Unorm, MTLPixelFormatBGRA8Unorm);
     }},
    {"view-rgba8-as-rgba8-srgb",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB);
     }},
    {"view-rgba8-as-rgb10a2",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGB10A2Unorm);
     }},
    {"view-rgba8-as-rg11b10float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA8Unorm, MTLPixelFormatRG11B10Float);
     }},
    {"view-r16-as-rg8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatR16Unorm, MTLPixelFormatRG8Unorm);
     }},
    {"view-r16-as-r8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatR16Unorm, MTLPixelFormatR8Unorm);
     }},
    {"view-rgba16float-as-rg32float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA16Float, MTLPixelFormatRG32Float);
     }},
    {"view-rgba16float-as-rgba32float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatRGBA16Float, MTLPixelFormatRGBA32Float);
     }},
    {"view-r32float-as-depth32float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatR32Float, MTLPixelFormatDepth32Float);
     }},
    {"view-depth32float-as-r32float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatDepth32Float, MTLPixelFormatR32Float);
     }},
    {"view-depth32float-as-depth16",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatDepth32Float, MTLPixelFormatDepth16Unorm);
     }},
    {"view-depth32float-stencil8-as-x32-stencil8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatX32_Stencil8);
     }},
    {"view-depth32float-stencil8-as-depth32float",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatDepth32Float);
     }},
    {"view-depth32float-stencil8-as-stencil8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatDepth32Float_Stencil8, MTLPixelFormatStencil8);
     }},
    {"view-stencil8-as-r8uint",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatStencil8, MTLPixelFormatR8Uint);
     }},
    {"view-b5g6r5-as-r16",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatB5G6R5Unorm, MTLPixelFormatR16Unorm);
     }},
    {"view-bc1-as-bc1-srgb",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatBC1_RGBA, MTLPixelFormatBC1_RGBA_sRGB);
     }},
    {"view-bc1-as-bc4",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatBC1_RGBA, MTLPixelFormatBC4_RUnorm);
     }},
    {"view-bc1-as-rgba8",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatBC1_RGBA, MTLPixelFormatRGBA8Unorm);
     }},
    {"view-bc1-as-rgba16",
     [](const Context& context) {
         return Cast(context, MTLPixelFormatBC1_RGBA, MTLPixelFormatRGBA16Unorm);
     }},
    {"sampler-anisotropy-0", [](const Context& context) { return Sampler(context, 0, 0, 16); }},
    {"sampler-anisotropy-17", [](const Context& context) { return Sampler(context, 17, 0, 16); }},
    {"sampler-min-lod-above-max",
     [](const Context& context) { return Sampler(context, 1, 8, 2); }},
    // Passes and draws.
    {"draw-without-attachments",
     [](const Context& context) { return Draw(context, Targetless(context), Pass::Sized, {}); }},
    {"draw-without-attachments-with-default-samples-1",
     [](const Context& context) {
         return Draw(context, Targetless(context), Pass::Sampled, {});
     }},
    {"draw-without-attachments-uncounted",
     [](const Context& context) {
         return Draw(context, Targetless(context), Pass::Sized, {}, false);
     }},
    {"draw-without-attachments-with-default-samples-1-uncounted",
     [](const Context& context) {
         return Draw(context, Targetless(context), Pass::Sampled, {}, false);
     }},
    {"draw-without-attachments-or-fragment-function",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.fragmentFunction = nil;
         return Draw(context, descriptor, Pass::Sized, {});
     }},
    {"pass-without-attachments-or-size",
     [](const Context& context) {
         return Draw(context, Targetless(context), Pass::Unsized, {});
     }},
    {"pipeline-with-target-in-pass-without",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Sized, {});
     }},
    {"pipeline-without-target-in-pass-with",
     [](const Context& context) { return Draw(context, Targetless(context), Pass::Colour, {}); }},
    {"pass-colour-64-with-depth-32",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
         const id<MTLTexture> depth =
             Shape(context, MTLTextureType2D, MTLPixelFormatDepth32Float, TargetSize / 2,
                   TargetSize / 2, 1, 1, 1, MTLTextureUsageRenderTarget);
         return Draw(context, descriptor, Pass::Colour, {}, true,
                     [depth](MTLRenderPassDescriptor* pass) {
                         pass.depthAttachment.texture = depth;
                         pass.depthAttachment.loadAction = MTLLoadActionClear;
                         pass.depthAttachment.clearDepth = 1.0;
                     });
     }},
    {"pass-with-depth-the-pipeline-does-not-name",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         const id<MTLTexture> depth =
             Shape(context, MTLTextureType2D, MTLPixelFormatDepth32Float, TargetSize, TargetSize,
                   1, 1, 1, MTLTextureUsageRenderTarget);
         return Draw(context, descriptor, Pass::Colour, {}, true,
                     [depth](MTLRenderPassDescriptor* pass) {
                         pass.depthAttachment.texture = depth;
                         pass.depthAttachment.loadAction = MTLLoadActionClear;
                         pass.depthAttachment.clearDepth = 1.0;
                     });
     }},
    {"pass-with-level-and-slice-past-the-texture",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, {}, true,
                     [](MTLRenderPassDescriptor* pass) {
                         pass.colorAttachments[0].level = 3;
                         pass.colorAttachments[0].slice = 2;
                     });
     }},
    {"scissor-past-the-target",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, [](id<MTLRenderCommandEncoder> encoder) {
             [encoder setScissorRect:MTLScissorRect{0, 0, TargetSize + 1, TargetSize}];
         });
     }},
    {"scissor-empty",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, [](id<MTLRenderCommandEncoder> encoder) {
             [encoder setScissorRect:MTLScissorRect{0, 0, 0, 0}];
         });
     }},
    {"viewport-empty",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, [](id<MTLRenderCommandEncoder> encoder) {
             [encoder setViewport:MTLViewport{0.0, 0.0, 0.0, 0.0, 0.0, 1.0}];
         });
     }},
    {"viewport-negative-height",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, [](id<MTLRenderCommandEncoder> encoder) {
             [encoder setViewport:MTLViewport{0.0, 64.0, 64.0, -64.0, 0.0, 1.0}];
         });
     }},
    {"viewport-depth-range-reversed",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         return Draw(context, descriptor, Pass::Colour, [](id<MTLRenderCommandEncoder> encoder) {
             [encoder setViewport:MTLViewport{0.0, 0.0, 64.0, 64.0, 1.0, 0.0}];
         });
     }},
    {"vertex-buffer-slot-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         const id<MTLBuffer> buffer =
             [context.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
         return Draw(context, descriptor, Pass::Colour,
                     [buffer](id<MTLRenderCommandEncoder> encoder) {
                         [encoder setVertexBuffer:buffer offset:0 atIndex:31];
                     });
     }},
    {"vertex-buffer-offset-past-the-end",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         const id<MTLBuffer> buffer =
             [context.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
         return Draw(context, descriptor, Pass::Colour,
                     [buffer](id<MTLRenderCommandEncoder> encoder) {
                         [encoder setVertexBuffer:buffer offset:128 atIndex:Slot];
                     });
     }},
    {"stage-in-draw-without-its-buffer",
     [](const Context& context) { return Draw(context, Quad(context), Pass::Colour, {}); }},
    {"fragment-buffer-slot-31",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         const id<MTLBuffer> buffer =
             [context.device newBufferWithLength:64 options:MTLResourceStorageModeShared];
         return Draw(context, descriptor, Pass::Colour,
                     [buffer](id<MTLRenderCommandEncoder> encoder) {
                         [encoder setFragmentBuffer:buffer offset:0 atIndex:31];
                     });
     }},
    {"fragment-texture-slot-128",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         MTLTextureDescriptor* texture =
             [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                width:16
                                                               height:16
                                                            mipmapped:NO];
         const id<MTLTexture> bound = [context.device newTextureWithDescriptor:texture];
         return Draw(context, descriptor, Pass::Colour,
                     [bound](id<MTLRenderCommandEncoder> encoder) {
                         [encoder setFragmentTexture:bound atIndex:128];
                     });
     }},
    {"fragment-sampler-slot-16",
     [](const Context& context) {
         MTLRenderPipelineDescriptor* descriptor = Targetless(context);
         descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
         const id<MTLSamplerState> sampler =
             [context.device newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]];
         return Draw(context, descriptor, Pass::Colour,
                     [sampler](id<MTLRenderCommandEncoder> encoder) {
                         [encoder setFragmentSamplerState:sampler atIndex:16];
                     });
     }},
};

enum class Verdict {
    Built,
    Error,
    Aborted,
    NoChild,
};

struct Outcome {
    Verdict verdict;
    std::string text;
};

/// Line breaks and control characters become spaces, at most `limit` bytes of the end stay.
std::string Flatten(std::string text, size_t limit) {
    for (char& character : text) {
        if (static_cast<unsigned char>(character) < 0x20) {
            character = ' ';
        }
    }
    while (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
    return text.size() > limit ? "..." + text.substr(text.size() - limit) : text;
}

constexpr const char* ResultPrefix = "probe-result: ";

/// Runs one suspect in a child, with Metal's validation layer loaded into it or, as the app
/// runs, without.
Outcome RunSuspect(char* path, const Suspect& suspect, bool validation) {
    int out[2];
    if (pipe(out) != 0) {
        return {Verdict::NoChild, "no pipe"};
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, out[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, out[0]);
    posix_spawn_file_actions_addclose(&actions, out[1]);
    char probe[] = "--probe";
    std::string name = suspect.name;
    char* const arguments[] = {path, probe, name.data(), nullptr};
    char layer[] = "MTL_DEBUG_LAYER=1";
    std::vector<char*> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        environment.push_back(*entry);
    }
    if (validation) {
        environment.push_back(layer);
    }
    environment.push_back(nullptr);
    pid_t child = 0;
    const int spawned =
        posix_spawn(&child, path, &actions, nullptr, arguments, environment.data());
    posix_spawn_file_actions_destroy(&actions);
    close(out[1]);
    if (spawned != 0) {
        close(out[0]);
        return {Verdict::NoChild, std::strerror(spawned)};
    }
    std::string output;
    char chunk[1024];
    for (ssize_t count; (count = read(out[0], chunk, sizeof(chunk))) > 0;) {
        output.append(chunk, static_cast<size_t>(count));
    }
    close(out[0]);
    int status = 0;
    waitpid(child, &status, 0);
    const size_t result = output.rfind(ResultPrefix);
    if (WIFSIGNALED(status)) {
        return {Verdict::Aborted,
                "ABORTED (signal " + std::to_string(WTERMSIG(status)) + "): " +
                    Flatten(output, 600)};
    }
    if (result == std::string::npos) {
        return {Verdict::Error, "exit " + std::to_string(WEXITSTATUS(status)) +
                                    " without a result: " + Flatten(output, 600)};
    }
    std::string text = Flatten(output.substr(result + std::strlen(ResultPrefix)), 600);
    const Verdict verdict = text.starts_with("built") ? Verdict::Built : Verdict::Error;
    return {verdict, std::move(text)};
}

} // Anonymous namespace

int RunChild(const char* name) {
    const auto suspect = std::ranges::find_if(
        Suspects, [name](const Suspect& entry) { return std::strcmp(entry.name, name) == 0; });
    if (suspect == std::end(Suspects)) {
        std::printf("%sunknown suspect\n", ResultPrefix);
        return 4;
    }
    // A suspect that hangs ends as a signal in the report.
    alarm(60);
    Context context;
    context.device = MTLCreateSystemDefaultDevice();
    NSError* error = nil;
    const id<MTLLibrary> library = [context.device newLibraryWithSource:@(Source)
                                                                options:[MTLCompileOptions new]
                                                                  error:&error];
    if (library == nil) {
        std::printf("%sno shader library: %s\n", ResultPrefix,
                    error != nil ? error.localizedDescription.UTF8String : "no device");
        return 4;
    }
    context.vertex = [library newFunctionWithName:@"probe_vertex"];
    context.vertex_plain = [library newFunctionWithName:@"probe_vertex_plain"];
    context.fragment = [library newFunctionWithName:@"probe_fragment"];
    context.fragment_uint = [library newFunctionWithName:@"probe_fragment_uint"];
    context.fragment_two = [library newFunctionWithName:@"probe_fragment_two"];
    const std::string result = suspect->run(context);
    std::printf("%s%s\n", ResultPrefix, result.c_str());
    std::fflush(stdout);
    return 0;
}

int RunAll() {
    char path[4096];
    uint32_t path_size = sizeof(path);
    if (_NSGetExecutablePath(path, &path_size) != 0) {
        std::printf("metal-smoke: FAIL probe: no executable path\n");
        return 1;
    }
    uint32_t built = 0;
    uint32_t errors = 0;
    uint32_t aborted = 0;
    uint32_t invalid = 0;
    bool works = true;
    for (const Suspect& suspect : Suspects) {
        const Outcome outcome = RunSuspect(path, suspect, false);
        if (outcome.verdict == Verdict::NoChild) {
            std::printf("metal-smoke: FAIL probe %s: no child (%s)\n", suspect.name,
                        outcome.text.c_str());
            return 1;
        }
        ++(outcome.verdict == Verdict::Built     ? built
           : outcome.verdict == Verdict::Aborted ? aborted
                                                 : errors);
        if (&suspect == &Suspects[0] && outcome.verdict != Verdict::Built) {
            works = false;
        }
        std::printf("metal-smoke: probe %s: %s\n", suspect.name, outcome.text.c_str());
        if (outcome.verdict == Verdict::Aborted) {
            continue;
        }
        // What Metal took may still be against its rules: the validation layer, which the app
        // doesn't run with, says so. What the GPU's driver does with it then is its own matter.
        // An error may be one the layer names a rule for as well.
        const Outcome checked = RunSuspect(path, suspect, true);
        if (outcome.verdict == Verdict::Built && checked.verdict != Verdict::Built) {
            ++invalid;
        }
        if (checked.verdict != outcome.verdict || checked.text != outcome.text) {
            std::printf("metal-smoke: probe %s with MTL_DEBUG_LAYER=1: %s\n", suspect.name,
                        checked.text.c_str());
        }
    }
    std::printf("metal-smoke: descriptor probe, %zu suspects: %u built (%u of them against the "
                "validation layer), %u errors, %u aborted the process: %s\n",
                std::size(Suspects), built, invalid, errors, aborted, works ? "PASS" : "FAIL");
    return works ? 0 : 1;
}

} // namespace SmokeProbe
