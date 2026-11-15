// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// lichii-metal-smoke: an offscreen check of the Metal paths the renderer uses, for machines
// without a window (CI): a clear and a draw in raw Metal, then the renderer's own pieces (SPIR-V
// -> MSL, the scheduler's render passes and compute, render pipeline states from Maxwell state,
// rewritten quad indices, occlusion counting, texel buffer views), each read back and compared,
// and last the descriptor probe (metal_smoke_probe.mm; --probe <name> is its child's side).
// Exit 0 passed, 1 failed, 2 no Metal device.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/shader_info.h"
#include "tess_smoke_tesc_spv.h"
#include "tess_smoke_tese_spv.h"
#include "tess_smoke_vert_spv.h"
#include "texel_buffer_smoke_comp_spv.h"
#include "translate_smoke_comp_spv.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_metal/metal_smoke_probe.h"

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
        std::span<const uint32_t>(TRANSLATE_SMOKE_COMP_SPV), Translate::Stage::Compute, {});
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

    // The right half: six vertices drawn from indirect arguments in the guest's layout
    // (vertex count, instance count, first vertex, first instance).
    uint8_t* right_vertices = nullptr;
    const Metal::Object right_buffer = Metal::CreateBuffer(device, 48, true, &right_vertices);
    const float right_positions[12]{0.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f,
                                    0.0f, -1.0f, 1.0f, 1.0f,  0.0f, 1.0f};
    std::memcpy(right_vertices, right_positions, sizeof(right_positions));
    uint8_t* arguments = nullptr;
    const Metal::Object argument_buffer = Metal::CreateBuffer(device, 16, true, &arguments);
    const uint32_t draw_arguments[4]{6, 1, 0, 0};
    std::memcpy(arguments, draw_arguments, sizeof(draw_arguments));
    scheduler.BindVertexBuffer(Metal::VertexBufferSlot(0), right_buffer.Get(), 0);
    scheduler.VisibilityResults()[1] = 0;
    scheduler.SetVisibility(1);
    scheduler.DrawIndirect(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), argument_buffer.Get(), 0);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel left = texels[(Size / 2) * Size + 4];
    const Texel right = texels[(Size / 2) * Size + Size - 4];
    const uint64_t samples = scheduler.VisibilityResults()[0];
    const uint64_t indirect_samples = scheduler.VisibilityResults()[1];
    const bool drawn = Near(left, {0, 255, 0, 255}) && Near(right, {0, 255, 0, 255});
    const bool counted = samples == uint64_t{Size} * Size / 2 &&
                         indirect_samples == uint64_t{Size} * Size / 2;
    std::printf("metal-smoke: renderer quad (8-bit indices -> %u rewritten) + indirect draw: left "
                "%u,%u,%u right %u,%u,%u: %s; occlusion %llu + %llu samples (expected %u each): "
                "%s\n",
                count, left.r, left.g, left.b, right.r, right.g, right.b,
                drawn ? "PASS" : "FAIL", static_cast<unsigned long long>(samples),
                static_cast<unsigned long long>(indirect_samples), Size * Size / 2,
                counted ? "PASS" : "FAIL");
    return drawn && counted ? 0 : 1;
}

/// A texel buffer read: GLSL samplerBuffer -> SPIR-V -> translator (native texture_buffer), the
/// renderer's texture buffer view at an aligned offset, the scheduler's compute encoder.
int RunTexelBuffer() {
    namespace Translate = Metal::ShaderTranslate;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Translate::Result msl = Translate::SpirvToMsl(
        std::span<const uint32_t>(TEXEL_BUFFER_SMOKE_COMP_SPV), Translate::Stage::Compute, {});
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

constexpr const char* MeshColorSource = R"(
#include <metal_stdlib>
using namespace metal;

struct MeshColorIn {
    float4 color [[user(locn0)]];
};

fragment float4 mesh_color_fragment(MeshColorIn in [[stage_in]]) {
    return in.color;
}
)";

/// Mesh emulation's draw side: two output blocks in the layout of EmitMeshEmulationVertex (four
/// vertices with a position and generic 0 each, three triangles of which the last is unused),
/// drawn by its vertex shader through the translator; the left half green, the right half blue.
int RunMeshEmulation() {
    using Shader::IR::Attribute;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Shader::MeshInfo mesh{};
    mesh.max_vertices = 4;
    mesh.max_primitives = 3;
    mesh.topology = 4;
    for (const Attribute attribute :
         {Attribute::PositionX, Attribute::PositionY, Attribute::PositionZ, Attribute::PositionW,
          Attribute::Generic0X, Attribute::Generic0Y, Attribute::Generic0Z, Attribute::Generic0W}) {
        mesh.vertex_lines.push_back(static_cast<uint32_t>(attribute));
    }
    const Shader::Profile profile{.supported_spirv = 0x00010300,
                                  .unified_descriptor_binding = true};
    constexpr uint32_t OutputBinding = 3;
    const std::vector<uint32_t> spirv = Shader::Backend::SPIRV::EmitMeshEmulationVertex(
        profile, mesh, OutputBinding, ~0U, false);
    namespace Translate = Metal::ShaderTranslate;
    const Translate::Result msl =
        Translate::SpirvToMsl(spirv, Translate::Stage::Vertex, Translate::Options{});
    if (msl.source.empty()) {
        std::printf("metal-smoke: FAIL mesh emulation vertex SPIR-V -> MSL: %s\n",
                    msl.error.c_str());
        return 1;
    }
    uint32_t output_slot = UINT32_MAX;
    for (const Translate::ResourceSlot& slot : msl.slots) {
        if (slot.kind == Translate::ResourceKind::Buffer && slot.binding == OutputBinding) {
            output_slot = slot.index;
        }
    }
    std::string error;
    const Metal::Object vertex_library = Metal::CreateLibrary(device, msl.source, error);
    const Metal::Object vertex = vertex_library
                                     ? Metal::CreateFunction(vertex_library.Get(), msl.entry_point)
                                     : Metal::Object{};
    const Metal::Object fragment_library = Metal::CreateLibrary(device, MeshColorSource, error);
    const Metal::Object fragment =
        fragment_library ? Metal::CreateFunction(fragment_library.Get(), "mesh_color_fragment")
                         : Metal::Object{};
    if (!vertex || !fragment || output_slot == UINT32_MAX) {
        std::printf("metal-smoke: FAIL mesh emulation shaders (output slot %u): %s\n%s\n",
                    output_slot, error.c_str(), msl.source.c_str());
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
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL mesh emulation render pipeline: %s\n", error.c_str());
        return 1;
    }

    // Per block: vertex v line l at v * 8 + l, then the triangles' indices.
    const uint32_t block_words = Shader::Backend::SPIRV::MeshEmulationBlockWords(mesh);
    constexpr uint32_t Groups = 2;
    uint8_t* output_data = nullptr;
    const Metal::Object output =
        Metal::CreateBuffer(device, Groups * block_words * sizeof(uint32_t), true, &output_data);
    auto* const words = reinterpret_cast<uint32_t*>(output_data);
    std::memset(words, 0, Groups * block_words * sizeof(uint32_t));
    const std::array<std::array<float, 4>, Groups> colors{{{0, 1, 0, 1}, {0, 0, 1, 1}}};
    for (uint32_t group = 0; group < Groups; ++group) {
        uint32_t* const block = words + group * block_words;
        const float left = group == 0 ? -1.0f : 0.0f;
        const std::array<std::array<float, 2>, 4> corners{
            {{left, -1.0f}, {left + 1.0f, -1.0f}, {left + 1.0f, 1.0f}, {left, 1.0f}}};
        for (uint32_t index = 0; index < 4; ++index) {
            const std::array<float, 8> lines{corners[index][0], corners[index][1], 0.0f, 1.0f,
                                             colors[group][0],  colors[group][1],  colors[group][2],
                                             colors[group][3]};
            std::memcpy(block + index * 8, lines.data(), sizeof(lines));
        }
        const std::array<uint32_t, 9> triangles{0, 1, 2, 0, 2, 3, 0, 0, 0};
        std::memcpy(block + 4 * 8, triangles.data(), sizeof(triangles));
    }

    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm),
                 .width = Size,
                 .height = Size,
                 .render_target = true});
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Size * Size * 4, true, &readback);
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
    scheduler.BindBuffer(Metal::ShaderStage::Vertex, output_slot, output.Get(), 0);
    const uint32_t vertices = Groups * mesh.max_primitives * mesh.PrimitiveVertices();
    scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, vertices, 1, 0);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel left = texels[(Size / 2) * Size + 4];
    const Texel right = texels[(Size / 2) * Size + Size - 4];
    const bool drawn = Near(left, {0, 255, 0, 255}) && Near(right, {0, 0, 255, 255});
    std::printf("metal-smoke: mesh emulation draw (%u blocks of %u words, %u vertices, output "
                "slot %u): left %u,%u,%u right %u,%u,%u: %s\n",
                Groups, block_words, vertices, output_slot, left.r, left.g, left.b, right.r,
                right.g, right.b, drawn ? "PASS" : "FAIL");
    return drawn ? 0 : 1;
}

constexpr const char* GroupMarkSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void mark_groups(device uint* marks [[buffer(0)]],
                        uint group [[threadgroup_position_in_grid]]) {
    marks[group] = 1;
}
)";

/// Mesh emulation with a task program: the distribution pass through the translator, twice on
/// one arena counter (the second limited by the arena's room), then an indirect dispatch from
/// the arguments it wrote.
int RunMeshDistribution() {
    namespace SPIRV = Shader::Backend::SPIRV;
    namespace Translate = Metal::ShaderTranslate;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Shader::Profile profile{.supported_spirv = 0x00010300,
                                  .unified_descriptor_binding = true};
    const Translate::Result msl = Translate::SpirvToMsl(
        SPIRV::EmitMeshEmulationDistribution(profile), Translate::Stage::Compute, {});
    std::string error;
    const Metal::Object library =
        msl.source.empty() ? Metal::Object{} : Metal::CreateLibrary(device, msl.source, error);
    const Metal::Object function =
        library ? Metal::CreateFunction(library.Get(), msl.entry_point) : Metal::Object{};
    const Metal::Object state =
        function ? Metal::CreateComputePipelineState(device, function.Get(), error)
                 : Metal::Object{};
    const Metal::Object mark_library = Metal::CreateLibrary(device, GroupMarkSource, error);
    const Metal::Object mark_function =
        mark_library ? Metal::CreateFunction(mark_library.Get(), "mark_groups") : Metal::Object{};
    const Metal::Object mark_state =
        mark_function ? Metal::CreateComputePipelineState(device, mark_function.Get(), error)
                      : Metal::Object{};
    if (!state || !mark_state) {
        std::printf("metal-smoke: FAIL distribution pipeline: %s %s\n%s\n", msl.error.c_str(),
                    error.c_str(), msl.source.c_str());
        return 1;
    }
    constexpr uint32_t PayloadWords = 4;
    constexpr uint32_t BlockWords = 10;
    constexpr uint32_t VerticesPerGroup = 6;
    const auto buffer = [&](uint32_t words, uint8_t** contents) {
        Metal::Object object = Metal::CreateBuffer(device, words * 4, true, contents);
        std::memset(*contents, 0, words * 4);
        return object;
    };
    struct Draw {
        std::vector<uint32_t> requests;
        uint8_t* payload{};
        uint8_t* offsets{};
        uint8_t* args{};
        Metal::Object payload_buffer;
        Metal::Object offsets_buffer;
        Metal::Object args_buffer;
    };
    std::array<Draw, 2> draws{};
    draws[0].requests = {3, 2};
    draws[1].requests = {4};
    uint8_t* counter = nullptr;
    const Metal::Object counter_buffer = buffer(1, &counter);
    uint8_t* marks = nullptr;
    const Metal::Object marks_buffer = buffer(64, &marks);
    // Room for seven blocks: the second draw gets two of the four it asks for.
    constexpr uint32_t CapacityWords = 7 * BlockWords;
    scheduler.BeginCompute(state.Get());
    for (Draw& draw : draws) {
        const uint32_t tasks = static_cast<uint32_t>(draw.requests.size());
        draw.payload_buffer = buffer(tasks * PayloadWords, &draw.payload);
        draw.offsets_buffer = buffer(tasks, &draw.offsets);
        draw.args_buffer = buffer(SPIRV::MESH_EMULATION_ARGS_WORDS, &draw.args);
        for (uint32_t task = 0; task < tasks; ++task) {
            std::memcpy(draw.payload + (task * PayloadWords + 1) * 4, &draw.requests[task], 4);
        }
        const SPIRV::MeshEmulationDistributionConstants constants{
            .tasks = tasks,
            .payload_words = PayloadWords,
            .capacity_words = CapacityWords,
            .block_words = BlockWords,
            .vertices_per_group = VerticesPerGroup,
            .max_groups = 65535,
        };
        const std::array<void*, 4> buffers{draw.payload_buffer.Get(), draw.offsets_buffer.Get(),
                                           draw.args_buffer.Get(), counter_buffer.Get()};
        for (const Translate::ResourceSlot& slot : msl.slots) {
            if (slot.kind != Translate::ResourceKind::Buffer) {
                continue;
            }
            if (slot.set == Translate::PushConstantSet) {
                scheduler.BindBytes(Metal::ShaderStage::Compute, slot.index, &constants,
                                    sizeof(constants));
            } else if (slot.binding < buffers.size()) {
                scheduler.BindBuffer(Metal::ShaderStage::Compute, slot.index,
                                     buffers[slot.binding], 0);
            }
        }
        scheduler.Dispatch({1, 1, 1}, {1, 1, 1});
    }
    // The first draw's mesh dispatch, from its arguments.
    scheduler.BeginCompute(mark_state.Get());
    scheduler.BindBuffer(Metal::ShaderStage::Compute, 0, marks_buffer.Get(), 0);
    scheduler.DispatchIndirect(draws[0].args_buffer.Get(), 0, {1, 1, 1});
    scheduler.Finish();

    const auto word = [](const uint8_t* data, uint32_t index) {
        uint32_t value;
        std::memcpy(&value, data + index * 4, 4);
        return value;
    };
    const uint32_t draw_words = SPIRV::MESH_EMULATION_DRAW_ARGS_OFFSET / 4;
    uint32_t marked = 0;
    for (uint32_t group = 0; group < 64; ++group) {
        marked += word(marks, group);
    }
    const bool first = word(draws[0].args, 0) == 5 && word(draws[0].args, draw_words) == 30 &&
                       word(draws[0].args, SPIRV::MESH_EMULATION_ARGS_BASE) == 0 &&
                       word(draws[0].offsets, 0) == 0 && word(draws[0].offsets, 1) == 3;
    const bool second = word(draws[1].args, 0) == 2 &&
                        word(draws[1].args, SPIRV::MESH_EMULATION_ARGS_REQUESTED) == 4 &&
                        word(draws[1].args, draw_words) == 12 &&
                        word(draws[1].args, SPIRV::MESH_EMULATION_ARGS_BASE) == 5 * BlockWords;
    const bool counted = word(counter, 0) == 9 * BlockWords && marked == 5;
    std::printf("metal-smoke: mesh distribution: draw 1 %u groups (%u vertices, base %u, offsets "
                "%u %u), draw 2 %u of %u groups (base %u), counter %u words, %u groups "
                "dispatched indirectly: %s\n",
                word(draws[0].args, 0), word(draws[0].args, draw_words),
                word(draws[0].args, SPIRV::MESH_EMULATION_ARGS_BASE), word(draws[0].offsets, 0),
                word(draws[0].offsets, 1), word(draws[1].args, 0),
                word(draws[1].args, SPIRV::MESH_EMULATION_ARGS_REQUESTED),
                word(draws[1].args, SPIRV::MESH_EMULATION_ARGS_BASE), word(counter, 0), marked,
                first && second && counted ? "PASS" : "FAIL");
    return first && second && counted ? 0 : 1;
}

/// The GPU index rewrite (indices the GPU wrote) against IndexConvert's CPU rewrite: quads,
/// quad strips, fans, line loops and lists, with 8-, 16- and 32-bit indices and restarts, read
/// from an odd byte offset.
int RunIndexRewrite() {
    namespace IndexConvert = Metal::IndexConvert;
    using Topology = IndexConvert::Topology;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    std::string error;
    const Metal::Object library =
        Metal::CreateLibrary(device, IndexConvert::RewriteKernelSource(), error);
    const Metal::Object function =
        library ? Metal::CreateFunction(library.Get(), "rewrite_indices") : Metal::Object{};
    const Metal::Object state =
        function ? Metal::CreateComputePipelineState(device, function.Get(), error)
                 : Metal::Object{};
    if (!state) {
        std::printf("metal-smoke: FAIL index rewrite kernel: %s\n", error.c_str());
        return 1;
    }
    constexpr uint32_t Count = 23;
    constexpr uint32_t SourceOffset = 3;
    uint8_t* source = nullptr;
    const Metal::Object source_buffer =
        Metal::CreateBuffer(device, SourceOffset + Count * 4, true, &source);
    struct Case {
        Topology topology;
        uint32_t index_size;
        bool restart;
        Metal::Object output;
        uint8_t* contents{};
        uint32_t converted{};
    };
    std::vector<Case> cases;
    for (const Topology topology : {Topology::Quads, Topology::QuadStrip, Topology::TriangleFan,
                                    Topology::LineLoop, Topology::TriangleStrip}) {
        for (const uint32_t index_size : {1U, 2U, 4U}) {
            cases.push_back({topology, index_size, topology == Topology::TriangleStrip});
        }
    }
    // One source for all: bytes that make different values at each size, with restart values.
    for (uint32_t byte = 0; byte < Count * 4; ++byte) {
        source[SourceOffset + byte] = static_cast<uint8_t>(byte * 37 + 5);
    }
    std::memset(source + SourceOffset + 4 * 2, 0xFF, 4);
    scheduler.BeginCompute(state.Get());
    for (Case& test : cases) {
        test.converted = IndexConvert::ConvertedCount(test.topology, Count);
        test.output = Metal::CreateBuffer(device, test.converted * 4, true, &test.contents);
        const IndexConvert::RewriteParams params{
            .mode = IndexConvert::RewriteMode(test.topology),
            .index_size = test.index_size,
            .count = Count,
            .restart = test.restart ? 1U : 0U,
            .converted = test.converted,
            .source_offset = SourceOffset,
        };
        scheduler.BindBuffer(Metal::ShaderStage::Compute, 0, source_buffer.Get(), 0);
        scheduler.BindBuffer(Metal::ShaderStage::Compute, 1, test.output.Get(), 0);
        scheduler.BindBytes(Metal::ShaderStage::Compute, 2, &params, sizeof(params));
        scheduler.Dispatch({(test.converted + 63) / 64, 1, 1}, {64, 1, 1});
    }
    scheduler.Finish();
    uint32_t wrong = 0;
    uint32_t total = 0;
    for (const Case& test : cases) {
        std::vector<uint32_t> expected(test.converted);
        IndexConvert::Convert(test.topology,
                              std::span<const uint8_t>(source + SourceOffset, Count * 4),
                              test.index_size, 0, Count, test.restart, expected);
        for (uint32_t index = 0; index < test.converted; ++index) {
            uint32_t value;
            std::memcpy(&value, test.contents + index * 4, 4);
            wrong += value != expected[index] ? 1 : 0;
        }
        total += test.converted;
    }
    std::printf("metal-smoke: index rewrite kernel vs CPU (%zu cases, %u indices): %u wrong: %s\n",
                cases.size(), total, wrong, wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

/// A packed RG11B10 float channel (`mantissa_bits` 6 or 5, 5-bit exponent, no sign).
float PackedFloat(uint32_t bits, uint32_t mantissa_bits) {
    const uint32_t mantissa = bits & ((1U << mantissa_bits) - 1);
    const int exponent = static_cast<int>(bits >> mantissa_bits);
    const float fraction = static_cast<float>(mantissa) / static_cast<float>(1U << mantissa_bits);
    return exponent == 0 ? std::ldexp(fraction, -14) : std::ldexp(1.0f + fraction, exponent - 15);
}

/// Ocarina's bilinear upscale (ocarina_bilinear_upscale.comp through the translator) from a
/// 16x16 scene of which 8x8 is the content into a 32x32 RG11B10Float storage target (the format
/// the game's pass writes): the squared colour everywhere.
int RunOcarinaBilinear() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t SceneSize = 16;
    constexpr uint32_t TargetSize = 32;
    const Metal::Object scene = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA32Float),
                 .width = SceneSize,
                 .height = SceneSize});
    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRG11B10Float),
                 .width = TargetSize,
                 .height = TargetSize,
                 .shader_write = true});
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer =
        Metal::CreateBuffer(device, SceneSize * SceneSize * 16, true, &upload);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, TargetSize * TargetSize * 4, true, &readback);
    if (!scene || !target || upload == nullptr || readback == nullptr) {
        std::printf("metal-smoke: FAIL Ocarina bilinear textures\n");
        return 1;
    }
    // The content (top-left 8x8) is (0.5, 0.25, 1); the rest would show if it were sampled.
    for (uint32_t y = 0; y < SceneSize; ++y) {
        for (uint32_t x = 0; x < SceneSize; ++x) {
            const bool content = x < SceneSize / 2 && y < SceneSize / 2;
            const std::array<float, 4> texel = content ? std::array{0.5f, 0.25f, 1.0f, 1.0f}
                                                       : std::array{4.0f, 4.0f, 4.0f, 1.0f};
            std::memcpy(upload + (y * SceneSize + x) * 16, texel.data(), 16);
        }
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, SceneSize * 16, 0, scene.Get(), 0, 0,
                                  {}, {SceneSize, SceneSize, 1});
    Metal::OcarinaUpscale ocarina(device, scheduler);
    const bool encoded = ocarina.Bilinear(scene.Get(), {SceneSize, SceneSize},
                                          {SceneSize / 2, SceneSize / 2}, target.Get(),
                                          {TargetSize, TargetSize});
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {TargetSize, TargetSize, 1},
                                  readback_buffer.Get(), 0, TargetSize * 4, 0);
    scheduler.Finish();
    if (!encoded) {
        std::printf("metal-smoke: FAIL Ocarina bilinear pass not encoded (see the log)\n");
        return 1;
    }
    // Texels well inside the content's footprint (the edge one blends with what follows it).
    uint32_t wrong = 0;
    std::array<float, 3> sample{};
    for (uint32_t y = 0; y < TargetSize - 2; ++y) {
        for (uint32_t x = 0; x < TargetSize - 2; ++x) {
            uint32_t packed;
            std::memcpy(&packed, readback + (y * TargetSize + x) * 4, 4);
            const std::array<float, 3> rgb{PackedFloat(packed & 0x7FF, 6),
                                           PackedFloat((packed >> 11) & 0x7FF, 6),
                                           PackedFloat(packed >> 22, 5)};
            const bool near = std::abs(rgb[0] - 0.25f) < 0.01f &&
                              std::abs(rgb[1] - 0.0625f) < 0.003f &&
                              std::abs(rgb[2] - 1.0f) < 0.04f;
            wrong += near ? 0 : 1;
            if (x == 4 && y == 4) {
                sample = rgb;
            }
        }
    }
    std::printf("metal-smoke: Ocarina bilinear upscale 8x8 of 16x16 -> 32x32 RG11B10: texel "
                "%.4f,%.4f,%.4f (expected 0.25,0.0625,1), %u of %u wrong: %s\n",
                sample[0], sample[1], sample[2], wrong, (TargetSize - 2) * (TargetSize - 2),
                wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

/// The pipeline archive: a compute and a render pipeline built into a new archive file (misses,
/// added), written, then the file opened again and both built once more (hits).
int RunPipelineArchive() {
    Metal::Device device;
    std::string error;
    const Metal::Object compute_library = Metal::CreateLibrary(device, GroupMarkSource, error);
    const Metal::Object compute =
        compute_library ? Metal::CreateFunction(compute_library.Get(), "mark_groups")
                        : Metal::Object{};
    const Metal::Object render_library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex =
        render_library ? Metal::CreateFunction(render_library.Get(), "quad_vertex")
                       : Metal::Object{};
    const Metal::Object fragment =
        render_library ? Metal::CreateFunction(render_library.Get(), "quad_fragment")
                       : Metal::Object{};
    if (!compute || !vertex || !fragment) {
        std::printf("metal-smoke: FAIL archive shaders: %s\n", error.c_str());
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
    desc.attributes[0] = {1, 0, 0, Metal::Maxwell::VertexAttribute::Type::Float,
                          Metal::Maxwell::VertexAttribute::Size::Size_R32_G32};
    desc.streams[0] = {1, 8, 0};
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "lichii-metal-smoke" / "metal_pipelines.bin";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    const auto build = [&](Metal::PipelineArchive& archive) {
        const bool compute_built =
            static_cast<bool>(Metal::CreateComputePipelineState(device, compute.Get(), error,
                                                                &archive));
        const bool render_built =
            static_cast<bool>(Metal::CreateRenderPipelineState(device, desc, error, &archive));
        return compute_built && render_built;
    };
    Metal::PipelineArchive::Counts first{};
    {
        Metal::PipelineArchive archive;
        if (!archive.Open(device, path)) {
            std::printf("metal-smoke: SKIP pipeline archive: no MTLBinaryArchive on this device\n");
            return 0;
        }
        if (!build(archive)) {
            std::printf("metal-smoke: FAIL pipeline archive: build failed: %s\n", error.c_str());
            return 1;
        }
        first = archive.Totals();
        if (first.added == 0) {
            std::printf("metal-smoke: SKIP pipeline archive: this device adds nothing to archives "
                        "(%llu not added)\n",
                        static_cast<unsigned long long>(first.add_failures));
            return 0;
        }
        archive.Save();
    }
    const bool written = std::filesystem::exists(path, ec);
    Metal::PipelineArchive reopened;
    const bool opened = reopened.Open(device, path);
    const bool built = opened && build(reopened);
    const Metal::PipelineArchive::Counts second = reopened.Totals();
    const bool passed = first.misses == 2 && first.added == 2 && written && built &&
                        second.hits == 2 && second.misses == 0;
    std::printf("metal-smoke: pipeline archive: first run %llu misses (%.2f ms), %llu added; "
                "file %s (%llu bytes); second run %llu hits (%.2f ms), %llu misses: %s\n",
                static_cast<unsigned long long>(first.misses), first.miss_ns / 1e6,
                static_cast<unsigned long long>(first.added), written ? "written" : "missing",
                static_cast<unsigned long long>(written ? std::filesystem::file_size(path, ec)
                                                        : 0),
                static_cast<unsigned long long>(second.hits), second.hit_ns / 1e6,
                static_cast<unsigned long long>(second.misses), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// The scheduler's depth blit (rescaling depth targets): a 16x16 Depth32Float holding 0.25 on
/// the left and 0.75 on the right, scaled into a 32x32 depth target and read back.
int RunDepthBlit() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Native = 16;
    constexpr uint32_t Scaled = 32;
    const uint32_t format = static_cast<uint32_t>(MTLPixelFormatDepth32Float);
    const auto make = [&](uint32_t size) {
        return Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                             .format = format,
                                             .width = size,
                                             .height = size,
                                             .render_target = true});
    };
    const Metal::Object native = make(Native);
    const Metal::Object scaled = make(Scaled);
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer = Metal::CreateBuffer(device, Native * Native * 4, true, &upload);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Scaled * Scaled * 4, true, &readback);
    if (!native || !scaled || upload == nullptr || readback == nullptr) {
        std::printf("metal-smoke: FAIL depth blit textures\n");
        return 1;
    }
    for (uint32_t y = 0; y < Native; ++y) {
        for (uint32_t x = 0; x < Native; ++x) {
            const float depth = x < Native / 2 ? 0.25f : 0.75f;
            std::memcpy(upload + (y * Native + x) * 4, &depth, 4);
        }
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, Native * 4, 0, native.Get(), 0, 0, {},
                                  {Native, Native, 1});
    const bool blitted = scheduler.Blit(Metal::BlitPass{
        .src = native.Get(),
        .src_rect = {0, 0, static_cast<int32_t>(Native), static_cast<int32_t>(Native)},
        .dst = {.texture = scaled.Get()},
        .dst_format = format,
        .dst_rect = {0, 0, static_cast<int32_t>(Scaled), static_cast<int32_t>(Scaled)},
        .dst_width = Scaled,
        .dst_height = Scaled,
        .linear = false,
        .keep_dst = false,
    });
    scheduler.CopyTextureToBuffer(scaled.Get(), 0, 0, {}, {Scaled, Scaled, 1},
                                  readback_buffer.Get(), 0, Scaled * 4, 0);
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t y = 0; y < Scaled; ++y) {
        for (uint32_t x = 0; x < Scaled; ++x) {
            float depth;
            std::memcpy(&depth, readback + (y * Scaled + x) * 4, 4);
            wrong += depth == (x < Scaled / 2 ? 0.25f : 0.75f) ? 0 : 1;
        }
    }
    const bool passed = blitted && wrong == 0;
    std::printf("metal-smoke: depth blit 16x16 -> 32x32 Depth32Float (%s): %u of %u wrong: %s\n",
                blitted ? "encoded" : "refused", wrong, Scaled * Scaled,
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Tessellation as the renderer runs it: GLSL vertex, control and evaluation programs through the
/// translator (stage outputs passed on, execution modes shared), the vertex program as a stage-in
/// kernel over a vertex buffer, the control kernel writing control points and factors, and the
/// evaluation program drawing the patch: a triangle over the top-left half, tessellation factors
/// 4 (half precision) in the factor buffer, and the samples it covers counted.
int RunTessellation() {
    namespace Translate = Metal::ShaderTranslate;
    using Maxwell = Metal::Maxwell;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Translate::Options options;
    const Translate::Result vertex = Translate::SpirvToMsl(
        std::span<const uint32_t>(TESS_SMOKE_VERT_SPV), Translate::Stage::VertexForTessellation,
        options);
    options.inputs = vertex.outputs;
    options.tessellation.output_vertices = 3;
    const Translate::Result control = Translate::SpirvToMsl(
        std::span<const uint32_t>(TESS_SMOKE_TESC_SPV), Translate::Stage::TessellationControl,
        options);
    options.inputs = control.outputs;
    const Translate::Result evaluation = Translate::SpirvToMsl(
        std::span<const uint32_t>(TESS_SMOKE_TESE_SPV), Translate::Stage::TessellationEvaluation,
        options);
    if (vertex.source.empty() || control.source.empty() || evaluation.source.empty()) {
        std::printf("metal-smoke: FAIL tessellation SPIR-V -> MSL: %s %s %s\n",
                    vertex.error.c_str(), control.error.c_str(), evaluation.error.c_str());
        return 1;
    }
    std::string error;
    const auto function = [&](const Translate::Result& result, Metal::Object& library) {
        library = Metal::CreateLibrary(device, result.source, error);
        return library ? Metal::CreateFunction(library.Get(), result.entry_point)
                       : Metal::Object{};
    };
    Metal::Object vertex_library, control_library, evaluation_library;
    const Metal::Object vertex_function = function(vertex, vertex_library);
    const Metal::Object control_function = function(control, control_library);
    const Metal::Object evaluation_function = function(evaluation, evaluation_library);
    const Metal::Object fragment_library = Metal::CreateLibrary(device, MeshColorSource, error);
    const Metal::Object fragment =
        fragment_library ? Metal::CreateFunction(fragment_library.Get(), "mesh_color_fragment")
                         : Metal::Object{};
    if (!vertex_function || !control_function || !evaluation_function || !fragment) {
        std::printf("metal-smoke: FAIL tessellation functions: %s\n%s\n%s\n%s\n", error.c_str(),
                    vertex.source.c_str(), control.source.c_str(), evaluation.source.c_str());
        return 1;
    }
    // The vertex layout: one float2 position per vertex in stream 0.
    auto layout = Metal::Zeroed<Metal::RenderPipelineDesc>();
    layout.attributes[0] = {1, 0, 0, Maxwell::VertexAttribute::Type::Float,
                            Maxwell::VertexAttribute::Size::Size_R32_G32};
    layout.streams[0] = {1, 8, 0};
    const Metal::Object vertex_kernel = Metal::CreateStageInComputePipelineState(
        device, vertex_function.Get(), layout, 0, vertex.tessellation_slots.index, error);
    const Metal::Object control_kernel =
        Metal::CreateComputePipelineState(device, control_function.Get(), error);
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = evaluation_function.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    desc.blend[0].write_mask = 0xF;
    desc.samples = 1;
    desc.rasterize = 1;
    desc.tessellation = 1;
    desc.tess_partition = static_cast<uint32_t>(MTLTessellationPartitionModeInteger);
    desc.tess_winding = static_cast<uint32_t>(MTLWindingClockwise);
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!vertex_kernel || !control_kernel || !pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL tessellation pipelines: %s\n", error.c_str());
        return 1;
    }

    const auto buffer = [&](uint32_t bytes, uint8_t** contents) {
        Metal::Object object = Metal::CreateBuffer(device, bytes, true, contents);
        std::memset(*contents, 0, bytes);
        return object;
    };
    uint8_t* positions = nullptr;
    const Metal::Object position_buffer = buffer(24, &positions);
    const float corners[6]{-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f};
    std::memcpy(positions, corners, sizeof(corners));
    uint8_t* vertices = nullptr;
    uint8_t* points = nullptr;
    uint8_t* patch_constants = nullptr;
    uint8_t* factors = nullptr;
    const Metal::Object vertex_out = buffer(3 * 64, &vertices);
    const Metal::Object points_out = buffer(32 * 64, &points);
    const Metal::Object patch_out = buffer(64, &patch_constants);
    const Metal::Object factors_out = buffer(64, &factors);
    const auto bind = [&](uint32_t slot, const Metal::Object& object) {
        if (slot != ~0U) {
            scheduler.BindBuffer(Metal::ShaderStage::Compute, slot, object.Get(), 0);
        }
    };
    scheduler.BeginCompute(vertex_kernel.Get());
    scheduler.BindBuffer(Metal::ShaderStage::Compute, Metal::VertexBufferSlot(0),
                         position_buffer.Get(), 0);
    bind(vertex.tessellation_slots.output, vertex_out);
    scheduler.DispatchStageIn(0, 3, 0, 1);
    scheduler.BeginCompute(control_kernel.Get());
    bind(control.tessellation_slots.input, vertex_out);
    bind(control.tessellation_slots.output, points_out);
    bind(control.tessellation_slots.patch_output, patch_out);
    bind(control.tessellation_slots.tess_factors, factors_out);
    const std::array<uint32_t, 2> params{3, 1};
    scheduler.BindBytes(Metal::ShaderStage::Compute, control.tessellation_slots.indirect_params,
                        params.data(), sizeof(params));
    scheduler.Dispatch({1, 1, 1}, {30, 1, 1});

    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm),
                 .width = Size,
                 .height = Size,
                 .render_target = true});
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer = buffer(Size * Size * 4, &readback);
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
    if (evaluation.tessellation_slots.input != ~0U) {
        scheduler.BindBuffer(Metal::ShaderStage::Vertex, evaluation.tessellation_slots.input,
                             points_out.Get(), 0);
    }
    if (evaluation.tessellation_slots.patch_input != ~0U) {
        scheduler.BindBuffer(Metal::ShaderStage::Vertex, evaluation.tessellation_slots.patch_input,
                             patch_out.Get(), 0);
    }
    scheduler.VisibilityResults()[2] = 0;
    scheduler.SetVisibility(2);
    scheduler.SetTessellationFactorBuffer(factors_out.Get(), 0);
    scheduler.DrawPatches(3, 1);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    // MTLTriangleTessellationFactorsHalf: three edge factors and the inside one, 4.0 each.
    std::array<uint16_t, 4> halves{};
    std::memcpy(halves.data(), factors, sizeof(halves));
    const bool factors_ok = std::ranges::all_of(halves, [](uint16_t half) { return half == 0x4400; });
    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel inside = texels[4 * Size + 4];
    const Texel outside = texels[(Size - 4) * Size + Size - 4];
    const bool drawn = Near(inside, {0, 255, 0, 255}) && Near(outside, {255, 0, 0, 255});
    const uint64_t samples = scheduler.VisibilityResults()[2];
    // Half the target and the diagonal's pixels whose centres it covers.
    const bool counted = samples >= Size * Size / 2 - Size && samples <= Size * Size / 2 + Size;
    const bool passed = factors_ok && drawn && counted;
    std::printf("metal-smoke: tessellation (vertex kernel -> control kernel -> patch draw): "
                "factors %04x %04x %04x %04x, inside %u,%u,%u outside %u,%u,%u, %llu samples "
                "(about %u): %s\n",
                halves[0], halves[1], halves[2], halves[3], inside.r, inside.g, inside.b,
                outside.r, outside.g, outside.b, static_cast<unsigned long long>(samples),
                Size * Size / 2, passed ? "PASS" : "FAIL");
    if (!passed) {
        std::printf("%s\n%s\n%s\n", vertex.source.c_str(), control.source.c_str(),
                    evaluation.source.c_str());
    }
    return passed ? 0 : 1;
}

/// The depth blit of a depth-stencil target (rescaling): a 16x16 Depth32Float_Stencil8 cleared
/// to stencil 0x5A, scaled to 32x32, then a full quad drawn with the stencil test "equal 0x5A"
/// (every sample passes) and "equal 0" (none do).
int RunDepthStencilBlit() {
    using Maxwell = Metal::Maxwell;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Native = 16;
    constexpr uint32_t Scaled = 32;
    const uint32_t format = static_cast<uint32_t>(MTLPixelFormatDepth32Float_Stencil8);
    const auto make = [&](uint32_t size, uint32_t texture_format) {
        return Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                             .format = texture_format,
                                             .width = size,
                                             .height = size,
                                             .render_target = true});
    };
    const Metal::Object native = make(Native, format);
    const Metal::Object scaled = make(Scaled, format);
    const Metal::Object color = make(Scaled, static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm));
    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex =
        library ? Metal::CreateFunction(library.Get(), "quad_vertex") : Metal::Object{};
    const Metal::Object fragment =
        library ? Metal::CreateFunction(library.Get(), "quad_fragment") : Metal::Object{};
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = vertex.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    desc.blend[0].write_mask = 0xF;
    desc.depth_format = format;
    desc.stencil_format = format;
    desc.samples = 1;
    desc.topology_class = 2;
    desc.rasterize = 1;
    desc.attributes[0] = {1, 0, 0, Maxwell::VertexAttribute::Type::Float,
                          Maxwell::VertexAttribute::Size::Size_R32_G32};
    desc.streams[0] = {1, 8, 0};
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    // The reference is the draw state's.
    const Metal::Object equal_state = [&] {
        auto depth_stencil = Metal::Zeroed<Metal::DepthStencilDesc>();
        depth_stencil.stencil_test = 1;
        depth_stencil.front = {Maxwell::StencilOp::Op::Keep_D3D, Maxwell::StencilOp::Op::Keep_D3D,
                               Maxwell::StencilOp::Op::Keep_D3D,
                               Maxwell::ComparisonOp::Equal_D3D, 0xFF, 0};
        depth_stencil.back = depth_stencil.front;
        return Metal::CreateDepthStencilState(device, depth_stencil);
    }();
    if (!native || !scaled || !color || !pipeline || !equal_state) {
        std::printf("metal-smoke: FAIL depth-stencil blit setup: %s\n", error.c_str());
        return 1;
    }
    uint8_t* vertices = nullptr;
    const Metal::Object vertex_buffer = Metal::CreateBuffer(device, 48, true, &vertices);
    const float positions[12]{-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    std::memcpy(vertices, positions, sizeof(positions));

    Metal::ClearPass clear{};
    clear.depth = {.texture = native.Get()};
    clear.stencil = {.texture = native.Get()};
    clear.clear_depth = true;
    clear.clear_stencil = true;
    clear.depth_value = 0.5;
    clear.stencil_value = 0x5A;
    clear.width = Native;
    clear.height = Native;
    scheduler.Clear(clear);
    const bool blitted = scheduler.Blit(Metal::BlitPass{
        .src = native.Get(),
        .src_rect = {0, 0, static_cast<int32_t>(Native), static_cast<int32_t>(Native)},
        .dst = {.texture = scaled.Get()},
        .dst_format = format,
        .dst_rect = {0, 0, static_cast<int32_t>(Scaled), static_cast<int32_t>(Scaled)},
        .dst_width = Scaled,
        .dst_height = Scaled,
        .linear = false,
        .keep_dst = false,
    });
    // Two draws counted apart in one pass: stencil reference 0x5A, then 0.
    std::array<uint64_t, 2> samples{};
    for (uint32_t pass = 0; pass < 2; ++pass) {
        Metal::RenderTargetSet targets{};
        targets.color[0] = {.texture = color.Get()};
        targets.depth = {.texture = scaled.Get()};
        targets.has_depth = true;
        targets.has_stencil = true;
        targets.width = Scaled;
        targets.height = Scaled;
        scheduler.BeginRendering(targets);
        Metal::DrawState state{};
        state.pipeline = pipeline.Get();
        state.depth_stencil = equal_state.Get();
        state.viewport = {0.0f, 0.0f, float(Scaled), float(Scaled), 0.0f, 1.0f};
        state.scissor = {0, 0, Scaled, Scaled};
        state.stencil_front_ref = pass == 0 ? 0x5A : 0;
        state.stencil_back_ref = state.stencil_front_ref;
        scheduler.SetDrawState(state);
        scheduler.BindVertexBuffer(Metal::VertexBufferSlot(0), vertex_buffer.Get(), 0);
        scheduler.VisibilityResults()[3 + pass] = 0;
        scheduler.SetVisibility(3 + pass);
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    }
    scheduler.Finish();
    samples = {scheduler.VisibilityResults()[3], scheduler.VisibilityResults()[4]};
    const bool passed = blitted && samples[0] == Scaled * Scaled && samples[1] == 0;
    std::printf("metal-smoke: depth-stencil blit 16x16 -> 32x32 (%s): stencil 0x5A passes %llu "
                "of %u samples, stencil 0 passes %llu: %s\n",
                blitted ? "encoded" : "refused", static_cast<unsigned long long>(samples[0]),
                Scaled * Scaled, static_cast<unsigned long long>(samples[1]),
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// The pipeline check (metal_pipeline_check.cpp). The guest's layout here has what aborted the
/// process in Ocarina of Time's first draw on the phone (vertex stream 31 has a buffer: the
/// renderer asked Metal's vertex descriptor for buffer layout 30 - 31), an attribute past its
/// stream's stride, one of a stream without a buffer and one the vertex function doesn't read. As
/// it stands it is refused in words; sanitised it builds and draws.
int RunPipelineCheck() {
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
        std::printf("metal-smoke: FAIL pipeline check shaders: %s\n", error.c_str());
        return 1;
    }
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    auto guest = Metal::Zeroed<Metal::RenderPipelineDesc>();
    guest.vertex_function = vertex.Get();
    guest.fragment_function = fragment.Get();
    guest.color_formats[0] = rgba8;
    guest.blend[0].write_mask = 0xF;
    guest.samples = 1;
    guest.topology_class = 2;
    guest.rasterize = 1;
    guest.attributes[0] = {1, 31, 16, Maxwell::VertexAttribute::Type::Float,
                           Maxwell::VertexAttribute::Size::Size_R32_G32};
    guest.attributes[3] = {1, 7, 0, Maxwell::VertexAttribute::Type::Float,
                           Maxwell::VertexAttribute::Size::Size_R32_G32_B32_A32};
    guest.attributes[9] = {1, 2, 0, Maxwell::VertexAttribute::Type::Float,
                           Maxwell::VertexAttribute::Size::Size_R32};
    guest.streams[2] = {1, 4, 0};
    guest.streams[31] = {1, 8, 0};
    // The vertex function loads generics 0 and 3 and has two buffers of its own.
    constexpr uint32_t Loaded = 0b1001;
    const Metal::PipelineLimits limits = Metal::DevicePipelineLimits(device);
    const std::string refusal = Metal::CheckPipeline(guest, limits);
    std::string build_error;
    const bool refused = !refusal.empty() &&
                         !Metal::CreateRenderPipelineState(device, guest, build_error) &&
                         !build_error.empty();

    Metal::RenderPipelineDesc desc = guest;
    Metal::VertexSources sources{};
    std::string reason;
    const bool sanitised = Metal::SanitizeVertexLayout(desc, Loaded, 2, sources, reason) &&
                           sources.count == 2 &&
                           sources.streams[0] == Metal::VertexSource{31, 16} &&
                           sources.streams[1] == Metal::VertexSource{} &&
                           desc.attributes[0].stream == 0 && desc.attributes[0].offset == 0 &&
                           desc.attributes[3].stream == 1 && desc.attributes[9].enabled == 0 &&
                           desc.streams[31].enabled == 0 &&
                           Metal::CheckPipeline(desc, limits).empty();
    // What the function doesn't read leaves the description alone: one pipeline for both.
    Metal::RenderPipelineDesc other = guest;
    other.attributes[9].offset = 2;
    other.streams[2].stride = 12;
    Metal::VertexSources other_sources{};
    const bool same = Metal::SanitizeVertexLayout(other, Loaded, 2, other_sources, reason) &&
                      std::memcmp(&other, &desc, sizeof(desc)) == 0;
    // No room next to 30 buffers of the function's own: refused, the description untouched.
    Metal::RenderPipelineDesc crowded = guest;
    reason.clear();
    const bool crowded_refused =
        !Metal::SanitizeVertexLayout(crowded, Loaded, 30, other_sources, reason) &&
        !reason.empty() && std::memcmp(&crowded, &guest, sizeof(guest)) == 0;
    Metal::RenderPipelineDesc odd = desc;
    odd.samples = 3;
    const bool samples_refused = !Metal::CheckPipeline(odd, limits).empty();
    odd = desc;
    odd.depth_format = rgba8;
    bool depth_refused = !Metal::CheckPipeline(odd, limits).empty();
    // Depth alone next to stencil alone is taken, a format with both next to another is not.
    odd = desc;
    odd.depth_format = static_cast<uint32_t>(MTLPixelFormatDepth32Float);
    odd.stencil_format = static_cast<uint32_t>(MTLPixelFormatStencil8);
    depth_refused = depth_refused && Metal::CheckPipeline(odd, limits).empty();
    odd.stencil_format = static_cast<uint32_t>(MTLPixelFormatDepth32Float_Stencil8);
    depth_refused = depth_refused && !Metal::CheckPipeline(odd, limits).empty();
    const bool textures_refused =
        !Metal::CreateTexture(device, {.format = rgba8, .width = 32768, .render_target = true}) &&
        !Metal::CreateTexture(device, {.format = rgba8, .width = 16, .levels = 9}) &&
        !Metal::CreateTexture(device, {.kind = Metal::TextureKind::Cube,
                                       .format = rgba8,
                                       .width = 16,
                                       .height = 8,
                                       .layers = 6});

    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL pipeline check: sanitised layout (%s) does not build: %s\n",
                    Metal::Describe(desc).c_str(), error.c_str());
        return 1;
    }
    // 16 bytes the attribute's offset skips, then two triangles over the whole target.
    uint8_t* vertices = nullptr;
    const Metal::Object vertex_buffer = Metal::CreateBuffer(device, 64, true, &vertices);
    const float positions[12]{-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    std::memset(vertices, 0, 64);
    std::memcpy(vertices + 16, positions, sizeof(positions));
    const Metal::Object zeros = Metal::CreateBuffer(device, 64, false, nullptr);
    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = rgba8,
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
    for (uint32_t slot = 0; slot < sources.count; ++slot) {
        const Metal::VertexSource& source = sources.streams[slot];
        const bool is_zeros = source.stream == Metal::VertexSource::Zeros;
        scheduler.BindVertexBuffer(Metal::VertexBufferSlot(slot),
                                   is_zeros ? zeros.Get() : vertex_buffer.Get(), source.shift);
    }
    scheduler.SetVisibility(std::nullopt);
    scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel first = texels[4 * Size + 4];
    const Texel last = texels[(Size - 4) * Size + Size - 4];
    const bool drawn = Near(first, {0, 255, 0, 255}) && Near(last, {0, 255, 0, 255});
    const bool passed = refused && sanitised && same && crowded_refused && samples_refused &&
                        depth_refused && textures_refused && drawn;
    std::printf("metal-smoke: pipeline check: stream 31 %s (%s); sanitised to %u streams %s, "
                "equal for unread state %d, crowded %d, 3 samples %d, colour as depth %d, "
                "textures %d; drawn %u,%u,%u and %u,%u,%u: %s\n",
                refused ? "refused" : "NOT refused", refusal.c_str(), sources.count,
                sanitised ? "as expected" : "NOT as expected", same, crowded_refused,
                samples_refused, depth_refused, textures_refused, first.r, first.g, first.b,
                last.r, last.g, last.b, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

} // Anonymous namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc == 3 && std::strcmp(argv[1], "--probe") == 0) {
            return SmokeProbe::RunChild(argv[2]);
        }
        const int draw = Run();
        if (draw == 2) {
            return 2;
        }
        int result = draw;
        for (const auto test : {RunTranslate, RunRendererDraw, RunTexelBuffer, RunMeshEmulation,
                                 RunMeshDistribution, RunIndexRewrite,
                                 RunOcarinaBilinear, RunPipelineArchive,
                                 RunDepthBlit, RunTessellation, RunDepthStencilBlit,
                                 RunPipelineCheck, SmokeProbe::RunAll}) {
            const int outcome = test();
            result = result != 0 ? result : outcome;
        }
        return result;
    }
}
