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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include <mach-o/dyld.h>
#include <spawn.h>
#include <sys/wait.h>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "common/settings_enums.h"
#include "shader_recompiler/shader_info.h"
#include "device_msl_smoke_frag_spv.h"
#include "device_msl_smoke_vert_spv.h"
#include "tess_smoke_tesc_spv.h"
#include "tess_smoke_tese_spv.h"
#include "tess_smoke_vert_spv.h"
#include "texel_buffer_smoke_comp_spv.h"
#include "texture_1d_smoke_comp_spv.h"
#include "translate_smoke_comp_spv.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_index_convert.h"
#include "video_core/renderer_metal/metal_ocarina.h"
#include "video_core/renderer_metal/metal_pipeline_files.h"
#include "video_core/renderer_metal/metal_present.h"
#include "video_core/renderer_metal/metal_resources.h"
#include "video_core/renderer_metal/metal_scheduler.h"
#include "video_core/renderer_metal/metal_shader_translate.h"
#include "video_core/renderer_metal/metal_smoke_probe.h"
#include "video_core/renderer_metal/metal_staging_buffer_pool.h"

extern char** environ;

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

/// A 1D image as the texture cache keeps it with the pipeline check: GLSL sampler1D -> SPIR-V ->
/// translator (1D as 2D), a 2D array texture a texel high with two levels, a 2D view of its
/// second layer. Then the views and the texture Metal would abort on, each refused.
int RunTexture1D() {
    namespace Translate = Metal::ShaderTranslate;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const Translate::Result msl = Translate::SpirvToMsl(
        std::span<const uint32_t>(TEXTURE_1D_SMOKE_COMP_SPV), Translate::Stage::Compute, {});
    if (msl.source.empty()) {
        std::printf("metal-smoke: FAIL 1D texture SPIR-V -> MSL: %s\n", msl.error.c_str());
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
        std::printf("metal-smoke: FAIL 1D texture pipeline: %s\n%s\n", error.c_str(),
                    msl.source.c_str());
        return 1;
    }
    uint32_t texture_slot = UINT32_MAX;
    uint32_t sampler_slot = UINT32_MAX;
    uint32_t data_slot = UINT32_MAX;
    for (const Translate::ResourceSlot& slot : msl.slots) {
        if (slot.kind == Translate::ResourceKind::Texture && slot.binding == 0) {
            texture_slot = slot.index;
        } else if (slot.kind == Translate::ResourceKind::Sampler && slot.binding == 0) {
            sampler_slot = slot.index;
        } else if (slot.kind == Translate::ResourceKind::Buffer && slot.binding == 1) {
            data_slot = slot.index;
        }
    }
    constexpr uint32_t Count = 64;
    const uint32_t format = static_cast<uint32_t>(MTLPixelFormatR32Float);
    const Metal::Object texture = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2DArray,
                 .format = format,
                 .width = Count,
                 .levels = 2,
                 .layers = 2});
    const Metal::Object view =
        texture ? Metal::CreateTextureView(texture.Get(), format, Metal::TextureKind::e2D, 0, 2, 1,
                                           1, Metal::IdentitySwizzle)
                : Metal::Object{};
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer =
        Metal::CreateBuffer(device, (Count + Count / 2) * sizeof(float), true, &upload);
    uint8_t* out_data = nullptr;
    const Metal::Object out = Metal::CreateBuffer(device, Count * sizeof(float), true, &out_data);
    if (!view || upload == nullptr || texture_slot == UINT32_MAX || data_slot == UINT32_MAX) {
        std::printf("metal-smoke: FAIL 1D texture, its view or slots\n%s\n", msl.source.c_str());
        return 1;
    }
    for (uint32_t index = 0; index < Count + Count / 2; ++index) {
        const float value = index < Count ? static_cast<float>(index) + 0.5f
                                          : 100.0f + static_cast<float>(index - Count);
        std::memcpy(upload + index * sizeof(float), &value, sizeof(value));
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, Count * 4, 0, texture.Get(), 0, 1, {},
                                  {Count, 1, 1});
    scheduler.CopyBufferToTexture(upload_buffer.Get(), Count * 4, Count / 2 * 4, 0, texture.Get(),
                                  1, 1, {}, {Count / 2, 1, 1});
    const Metal::Object sampler = Metal::Retain(
        [device.GetDevice() newSamplerStateWithDescriptor:[MTLSamplerDescriptor new]]);
    scheduler.BeginCompute(state.Get());
    scheduler.BindTexture(Metal::ShaderStage::Compute, texture_slot, view.Get());
    if (sampler_slot != UINT32_MAX) {
        scheduler.BindSampler(Metal::ShaderStage::Compute, sampler_slot, sampler.Get());
    }
    scheduler.BindBuffer(Metal::ShaderStage::Compute, data_slot, out.Get(), 0);
    scheduler.Dispatch({1, 1, 1}, {Count, 1, 1});
    scheduler.Finish();
    uint32_t wrong = 0;
    for (uint32_t index = 0; index < Count; ++index) {
        float value;
        std::memcpy(&value, out_data + index * sizeof(float), sizeof(value));
        const float expected = (static_cast<float>(index) + 0.5f) * 2.0f + 100.0f +
                               static_cast<float>(index / 2);
        wrong += value != expected ? 1 : 0;
    }
    const auto refused = [&](uint32_t view_format, Metal::TextureKind kind, uint32_t layers) {
        return !Metal::CreateTextureView(texture.Get(), view_format, kind, 0, 1, 0, layers,
                                         Metal::IdentitySwizzle);
    };
    const bool kinds_refused = refused(format, Metal::TextureKind::e1D, 1) &&
                               refused(format, Metal::TextureKind::e2D, 2) &&
                               refused(format, Metal::TextureKind::Cube, 2) &&
                               refused(format, Metal::TextureKind::e3D, 1) &&
                               refused(format, Metal::TextureKind::e2DMultisample, 1);
    const bool formats_refused =
        refused(static_cast<uint32_t>(MTLPixelFormatRG8Unorm), Metal::TextureKind::e2D, 1) &&
        refused(static_cast<uint32_t>(MTLPixelFormatBC1_RGBA), Metal::TextureKind::e2D, 1) &&
        refused(static_cast<uint32_t>(MTLPixelFormatDepth32Float), Metal::TextureKind::e2D, 1);
    // What Metal does take: every layer in another format of the same texel size.
    const bool cast = !refused(static_cast<uint32_t>(MTLPixelFormatR32Uint),
                               Metal::TextureKind::e2DArray, 2);
    const bool line_refused =
        !Metal::CreateTexture(device, {.kind = Metal::TextureKind::e1DArray,
                                       .format = format,
                                       .width = Count,
                                       .levels = 2,
                                       .layers = 2}) &&
        !Metal::CreateTexture(device, {.kind = Metal::TextureKind::e1DArray,
                                       .format = format,
                                       .width = Count,
                                       .height = 2});
    const bool passed = wrong == 0 && kinds_refused && formats_refused && cast && line_refused;
    std::printf("metal-smoke: 1D texture as 2D (texture slot %u, sampler slot %d): %u of %u "
                "wrong; views of other kinds refused %d, of other formats %d, R32Uint of "
                "R32Float built %d, 1D textures with levels or height refused %d: %s\n",
                texture_slot, static_cast<int>(sampler_slot), wrong, Count, kinds_refused,
                formats_refused, cast, line_refused, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
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
        bool provoking_last{};
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
    // Rewritten for the provoking vertex (EDEN_NX2_METAL_PROVOKING_LAST): strips without restarts.
    for (const Topology topology : {Topology::Triangles, Topology::TriangleStrip}) {
        for (const uint32_t index_size : {1U, 2U, 4U}) {
            cases.push_back({topology, index_size, false, true});
        }
    }
    // One source for all: bytes that make different values at each size, with restart values.
    for (uint32_t byte = 0; byte < Count * 4; ++byte) {
        source[SourceOffset + byte] = static_cast<uint8_t>(byte * 37 + 5);
    }
    std::memset(source + SourceOffset + 4 * 2, 0xFF, 4);
    scheduler.BeginCompute(state.Get());
    for (Case& test : cases) {
        test.converted = IndexConvert::ConvertedCount(test.topology, Count, test.provoking_last);
        test.output = Metal::CreateBuffer(device, test.converted * 4, true, &test.contents);
        const IndexConvert::RewriteParams params{
            .mode = IndexConvert::RewriteMode(test.topology, test.provoking_last),
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
                              test.index_size, 0, Count, test.restart, expected,
                              test.provoking_last);
        for (uint32_t index = 0; index < test.converted; ++index) {
            uint32_t value;
            std::memcpy(&value, test.contents + index * 4, 4);
            wrong += value != expected[index] ? 1 : 0;
        }
        total += test.converted;
    }
    // A strip with a restart, rewritten for the provoking vertex on the CPU: each part on its own,
    // every triangle starting with its last vertex and keeping its winding.
    const std::array<uint16_t, 9> strip{0, 1, 2, 3, 0xFFFF, 4, 5, 6, 7};
    const std::array<uint32_t, 12> strip_expected{2, 0, 1, 3, 2, 1, 6, 4, 5, 7, 6, 5};
    std::array<uint32_t, 21> strip_out{};
    const uint32_t strip_written = IndexConvert::Convert(
        Topology::TriangleStrip,
        std::span(reinterpret_cast<const uint8_t*>(strip.data()), sizeof(strip)), 2, 0,
        static_cast<uint32_t>(strip.size()), true, strip_out, true);
    const bool strip_passed =
        strip_written == strip_expected.size() &&
        std::equal(strip_expected.begin(), strip_expected.end(), strip_out.begin());
    const bool passed = wrong == 0 && strip_passed;
    std::printf("metal-smoke: index rewrite kernel vs CPU (%zu cases, %u indices): %u wrong; "
                "provoking vertex strip with a restart %u indices: %s: %s\n",
                cases.size(), total, wrong, strip_written, strip_passed ? "as expected" : "WRONG",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
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

/// The Vulkan present filters on Metal (EDEN_NX2_METAL_PRESENT_FILTERS): every shader
/// translated and built, then a 32x32 picture of four coloured quadrants through FXAA, SMAA and
/// each scaling filter to a 64x64 target. Each quadrant's centre keeps its colour: wrong
/// bindings, constants or a flipped picture show there.
int RunPresentFilters() {
    using Filter = Settings::ScalingFilter;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Metal::PresentPasses present(device, scheduler);
    uint32_t shaders_failed = 0;
    for (size_t shader = 0; shader < static_cast<size_t>(Metal::PresentPasses::Shader::Count);
         ++shader) {
        if (!present.Prepare(static_cast<Metal::PresentPasses::Shader>(shader))) {
            std::printf("metal-smoke: present shader %zu not built (see the log)\n", shader);
            ++shaders_failed;
        }
    }
    constexpr uint32_t Source = 32;
    constexpr uint32_t Target = 64;
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    const Metal::Object picture = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D, .format = rgba8, .width = Source, .height = Source});
    const Metal::Object target = Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                                               .format = rgba8,
                                                               .width = Target,
                                                               .height = Target,
                                                               .render_target = true});
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer = Metal::CreateBuffer(device, Source * Source * 4, true, &upload);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Target * Target * 4, true, &readback);
    if (!picture || !target || upload == nullptr || readback == nullptr) {
        std::printf("metal-smoke: FAIL present filter textures\n");
        return 1;
    }
    // Top left, top right, bottom left, bottom right.
    constexpr std::array<std::array<uint8_t, 4>, 4> Quadrants{{
        {200, 40, 40, 255},
        {40, 200, 40, 255},
        {40, 40, 200, 255},
        {200, 200, 200, 255},
    }};
    for (uint32_t y = 0; y < Source; ++y) {
        for (uint32_t x = 0; x < Source; ++x) {
            const size_t quadrant = (y >= Source / 2 ? 2 : 0) + (x >= Source / 2 ? 1 : 0);
            std::memcpy(upload + (y * Source + x) * 4, Quadrants[quadrant].data(), 4);
        }
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, Source * 4, 0, picture.Get(), 0, 0, {},
                                  {Source, Source, 1});
    scheduler.Finish();

    struct Case {
        const char* name;
        Settings::AntiAliasing anti_aliasing;
        Filter filter;
    };
    constexpr std::array Cases{
        Case{"FXAA", Settings::AntiAliasing::Fxaa, Filter::Bilinear},
        Case{"SMAA", Settings::AntiAliasing::Smaa, Filter::Bilinear},
        Case{"FSR", Settings::AntiAliasing::None, Filter::Fsr},
        Case{"SGSR", Settings::AntiAliasing::None, Filter::Sgsr},
        Case{"SGSR edge", Settings::AntiAliasing::None, Filter::SgsrEdge},
        Case{"bicubic", Settings::AntiAliasing::None, Filter::Bicubic},
        Case{"zero tangent", Settings::AntiAliasing::None, Filter::ZeroTangent},
        Case{"B-spline", Settings::AntiAliasing::None, Filter::BSpline},
        Case{"Mitchell", Settings::AntiAliasing::None, Filter::Mitchell},
        Case{"spline", Settings::AntiAliasing::None, Filter::Spline1},
        Case{"Gaussian", Settings::AntiAliasing::None, Filter::Gaussian},
        Case{"Lanczos", Settings::AntiAliasing::None, Filter::Lanczos},
        Case{"ScaleForce", Settings::AntiAliasing::None, Filter::ScaleForce},
        Case{"area", Settings::AntiAliasing::None, Filter::Area},
        Case{"MMPX", Settings::AntiAliasing::None, Filter::Mmpx},
        Case{"SMAA + FSR", Settings::AntiAliasing::Smaa, Filter::Fsr},
    };
    const auto close = [](uint8_t a, uint8_t b) { return (a > b ? a - b : b - a) <= 6; };
    uint32_t failed = 0;
    for (const Case& test : Cases) {
        (void)present.TakePasses();
        Metal::PresentPasses::Source source{.texture = picture.Get(),
                                            .size = {Source, Source},
                                            .render = {Source, Source}};
        source = present.AntiAlias(source, test.anti_aliasing);
        source = present.Upscale(source, test.filter, {Target, Target});
        const Metal::Attachment attachment{.texture = target.Get()};
        bool drawn = false;
        if (Metal::PresentPasses::HasWindowShader(test.filter)) {
            drawn = present.DrawWindow(source, test.filter, attachment, rgba8, {Target, Target},
                                       {0, 0, static_cast<int32_t>(Target),
                                        static_cast<int32_t>(Target)},
                                       false, Metal::BlitBlend::None);
        } else {
            const float width = static_cast<float>(source.size[0]);
            const float height = static_cast<float>(source.size[1]);
            drawn = scheduler.Blit({
                .src = source.texture,
                .src_rect = {static_cast<int32_t>(source.crop[0] * width),
                             static_cast<int32_t>(source.crop[1] * height),
                             static_cast<int32_t>(source.crop[2] * width),
                             static_cast<int32_t>(source.crop[3] * height)},
                .dst = attachment,
                .dst_format = rgba8,
                .dst_rect = {0, 0, static_cast<int32_t>(Target), static_cast<int32_t>(Target)},
                .dst_width = Target,
                .dst_height = Target,
                .linear = true,
                .keep_dst = false,
            });
        }
        const uint64_t passes = present.TakePasses();
        scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Target, Target, 1},
                                      readback_buffer.Get(), 0, Target * 4, 0);
        scheduler.Finish();
        // The anti-aliasing passes (1 or 3) and the scaler's (FSR 2, SGSR 1, a window filter 1).
        const uint64_t expected_passes =
            (test.anti_aliasing == Settings::AntiAliasing::Fxaa   ? 1
             : test.anti_aliasing == Settings::AntiAliasing::Smaa ? 3
                                                                  : 0) +
            (test.filter == Filter::Fsr                                   ? 2
             : test.filter == Filter::Sgsr || test.filter == Filter::SgsrEdge ? 1
             : Metal::PresentPasses::HasWindowShader(test.filter)         ? 1
                                                                          : 0);
        bool colours = true;
        std::string seen;
        for (size_t quadrant = 0; quadrant < 4; ++quadrant) {
            const uint32_t x = (quadrant & 1) != 0 ? Target * 3 / 4 : Target / 4;
            const uint32_t y = (quadrant & 2) != 0 ? Target * 3 / 4 : Target / 4;
            Texel texel;
            std::memcpy(&texel, readback + (y * Target + x) * 4, 4);
            const auto& want = Quadrants[quadrant];
            colours = colours && close(texel.r, want[0]) && close(texel.g, want[1]) &&
                      close(texel.b, want[2]) && close(texel.a, want[3]);
            seen += (quadrant != 0 ? " " : "") + std::to_string(texel.r) + "," +
                    std::to_string(texel.g) + "," + std::to_string(texel.b);
        }
        const bool passed = drawn && colours && passes == expected_passes;
        failed += passed ? 0 : 1;
        std::printf("metal-smoke: present %s: %llu passes (expected %llu), quadrant centres %s: "
                    "%s\n",
                    test.name, static_cast<unsigned long long>(passes),
                    static_cast<unsigned long long>(expected_passes), seen.c_str(),
                    passed ? "PASS" : "FAIL");
    }
    std::printf("metal-smoke: present filters: %u of %zu shaders not built, %u of %zu cases "
                "failed: %s\n",
                shaders_failed, static_cast<size_t>(Metal::PresentPasses::Shader::Count), failed,
                Cases.size(), shaders_failed == 0 && failed == 0 ? "PASS" : "FAIL");
    return shaders_failed == 0 && failed == 0 ? 0 : 1;
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
                                       .layers = 6}) &&
        !Metal::CreateTexture(device,
                              {.kind = Metal::TextureKind::e3D,
                               .format = static_cast<uint32_t>(MTLPixelFormatDepth32Float),
                               .width = 16,
                               .height = 16,
                               .depth = 16});

    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL pipeline check: sanitised layout (%s) does not build: %s\n",
                    Metal::Describe(desc).c_str(), error.c_str());
        return 1;
    }
    Metal::RenderPipelineDesc targetless = desc;
    targetless.color_formats[0] = 0;
    std::string bare_error;
    const Metal::Object bare = Metal::CreateRenderPipelineState(device, targetless, bare_error);
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
    // The same two triangles counted in a pass without attachments, as the game's occlusion
    // queries draw.
    if (bare) {
        Metal::RenderTargetSet nothing{};
        nothing.width = Size;
        nothing.height = Size;
        scheduler.BeginRendering(nothing);
        state.pipeline = bare.Get();
        scheduler.SetDrawState(state);
        for (uint32_t slot = 0; slot < sources.count; ++slot) {
            const Metal::VertexSource& source = sources.streams[slot];
            const bool is_zeros = source.stream == Metal::VertexSource::Zeros;
            scheduler.BindVertexBuffer(Metal::VertexBufferSlot(slot),
                                       is_zeros ? zeros.Get() : vertex_buffer.Get(), source.shift);
        }
        scheduler.VisibilityResults()[5] = 0;
        scheduler.SetVisibility(5);
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    }
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel first = texels[4 * Size + 4];
    const Texel last = texels[(Size - 4) * Size + Size - 4];
    const bool drawn = Near(first, {0, 255, 0, 255}) && Near(last, {0, 255, 0, 255});
    const uint64_t counted = bare ? scheduler.VisibilityResults()[5] : 0;
    const bool passed = refused && sanitised && same && crowded_refused && samples_refused &&
                        depth_refused && textures_refused && drawn && counted == Size * Size;
    std::printf("metal-smoke: pipeline check: stream 31 %s (%s); sanitised to %u streams %s, "
                "equal for unread state %d, crowded %d, 3 samples %d, colour as depth %d, "
                "textures %d; drawn %u,%u,%u and %u,%u,%u; without attachments %s, %llu of %u "
                "samples: %s\n",
                refused ? "refused" : "NOT refused", refusal.c_str(), sources.count,
                sanitised ? "as expected" : "NOT as expected", same, crowded_refused,
                samples_refused, depth_refused, textures_refused, first.r, first.g, first.b,
                last.r, last.g, last.b, bare ? "built" : bare_error.c_str(),
                static_cast<unsigned long long>(counted), Size * Size, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// EDEN_NX2_METAL_MSL_DEVICE: a vertex program that reads the base instance and a pixel program
/// with subgroup operations, translated for iOS without the GPU's family (the first is refused)
/// and with it, then built for this device and drawn: two instances from base instance 5.
/// LICHII_SMOKE_IOS_MSL names a directory the iOS translations are written to, for the iOS Metal
/// compiler (metal.yml).
int RunDeviceMsl() {
    namespace Translate = Metal::ShaderTranslate;
    const std::span<const uint32_t> vertex_spirv{DEVICE_MSL_SMOKE_VERT_SPV};
    const std::span<const uint32_t> fragment_spirv{DEVICE_MSL_SMOKE_FRAG_SPV};
    Translate::Options plain;
    plain.ios = true;
    const Translate::Result plain_vertex =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::Vertex, plain);
    const Translate::Result plain_fragment =
        Translate::SpirvToMsl(fragment_spirv, Translate::Stage::Fragment, plain);
    Translate::Options phone = plain;
    phone.apple_family = 10;
    const Translate::Result phone_vertex =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::Vertex, phone);
    const Translate::Result phone_fragment =
        Translate::SpirvToMsl(fragment_spirv, Translate::Stage::Fragment, phone);
    const auto has = [](const Translate::Result& msl, const char* word) {
        return msl.source.find(word) != std::string::npos;
    };
    const bool translated = plain_vertex.source.empty() && has(plain_fragment, "quad_shuffle") &&
                            has(phone_vertex, "base_instance") &&
                            has(phone_fragment, "simd_shuffle") &&
                            has(phone_fragment, "simd_ballot") &&
                            !has(phone_fragment, "quad_");
    if (const char* const directory = std::getenv("LICHII_SMOKE_IOS_MSL")) {
        const std::filesystem::path root{directory};
        std::error_code code;
        std::filesystem::create_directories(root, code);
        for (const auto& [name, msl] : {std::pair{"device_msl_vertex.metal", &phone_vertex},
                                        std::pair{"device_msl_fragment.metal", &phone_fragment}}) {
            if (std::FILE* const file = std::fopen((root / name).c_str(), "wb")) {
                std::fwrite(msl->source.data(), 1, msl->source.size(), file);
                std::fclose(file);
            }
        }
    }

    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Translate::Options here;
    here.apple_family = device.AppleFamily();
    const Translate::Result vertex_msl =
        Translate::SpirvToMsl(vertex_spirv, Translate::Stage::Vertex, here);
    const Translate::Result fragment_msl =
        Translate::SpirvToMsl(fragment_spirv, Translate::Stage::Fragment, here);
    std::string error = vertex_msl.error + fragment_msl.error;
    const Metal::Object vertex_library = Metal::CreateLibrary(device, vertex_msl.source, error);
    const Metal::Object fragment_library =
        Metal::CreateLibrary(device, fragment_msl.source, error);
    const Metal::Object vertex = vertex_library ? Metal::CreateFunction(vertex_library.Get(),
                                                                        vertex_msl.entry_point)
                                                : Metal::Object{};
    const Metal::Object fragment =
        fragment_library ? Metal::CreateFunction(fragment_library.Get(), fragment_msl.entry_point)
                         : Metal::Object{};
    if (!vertex || !fragment) {
        std::printf("metal-smoke: FAIL device MSL: the translation does not build: %s\n",
                    error.c_str());
        return 1;
    }
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = vertex.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = rgba8;
    desc.blend[0].write_mask = 0xF;
    desc.samples = 1;
    desc.topology_class = 2;
    desc.rasterize = 1;
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object depth_stencil =
        Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
    if (!pipeline || !depth_stencil) {
        std::printf("metal-smoke: FAIL device MSL: pipeline: %s\n", error.c_str());
        return 1;
    }
    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = rgba8,
                 .width = Size,
                 .height = Size,
                 .render_target = true});
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Size * Size * 4, true, &readback);

    Metal::ClearPass clear{};
    clear.color[0] = {.texture = target.Get()};
    clear.clear_color[0] = true;
    clear.color_value[0] = {0.0, 0.0, 1.0, 1.0};
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
    scheduler.SetVisibility(std::nullopt);
    scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangleStrip), 0, 4, 2, 5);
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Size, Size, 1}, readback_buffer.Get(),
                                  0, Size * 4, 0);
    scheduler.Finish();

    const auto* texels = reinterpret_cast<const Texel*>(readback);
    const Texel left = texels[(Size / 2) * Size + 4];
    const Texel right = texels[(Size / 2) * Size + Size - 4];
    const bool drawn = Near(left, {255, 0, 0, 255}) && Near(right, {0, 255, 0, 255});
    const bool passed = translated && drawn;
    std::printf("metal-smoke: device MSL: iOS without the GPU's family: vertex %s, pixel with "
                "quad groups %d; with Apple10: base instance %d, SIMD groups %d; here (Apple%u) "
                "left %u,%u,%u right %u,%u,%u: %s\n",
                plain_vertex.source.empty() ? plain_vertex.error.c_str() : "NOT refused",
                has(plain_fragment, "quad_shuffle"), has(phone_vertex, "base_instance"),
                has(phone_fragment, "simd_shuffle") && !has(phone_fragment, "quad_"),
                device.AppleFamily(), left.r, left.g, left.b, right.r, right.g, right.b,
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// A copy between a plain format and a block-compressed one, as the texture cache makes it
/// (ReinterpretImage): four RGBA16Uint texels through a buffer into an 8x8 BC4 texture as its
/// four blocks, each of one red value, then the blocks read back and the texture drawn.
int RunBlockCopy() {
    Metal::Device device;
    if (!device.SupportsBCTextures()) {
        std::printf("metal-smoke: block copy: no BC textures on this GPU: SKIP\n");
        return 0;
    }
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Blocks = 2;
    constexpr uint32_t Texels = Blocks * 4;
    constexpr uint32_t BlockBytes = 8;
    constexpr uint32_t BlockRow = Blocks * BlockBytes;
    constexpr std::array<uint8_t, 4> Reds{40, 80, 160, 240};
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    const Metal::Object plain = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA16Uint),
                 .width = Blocks,
                 .height = Blocks});
    const Metal::Object compressed = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatBC4_RUnorm),
                 .width = Texels,
                 .height = Texels});
    const Metal::Object target = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = rgba8,
                 .width = Texels,
                 .height = Texels,
                 .render_target = true});
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer =
        Metal::CreateBuffer(device, BlockRow * Blocks, true, &upload);
    const Metal::Object between = Metal::CreateBuffer(device, BlockRow * Blocks, false, nullptr);
    uint8_t* blocks = nullptr;
    const Metal::Object blocks_buffer =
        Metal::CreateBuffer(device, BlockRow * Blocks, true, &blocks);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Texels * Texels * 4, true, &readback);
    if (!plain || !compressed || !target || !between || upload == nullptr || blocks == nullptr ||
        readback == nullptr) {
        std::printf("metal-smoke: FAIL block copy textures\n");
        return 1;
    }
    // A BC4 block: two red values, then sixteen 3-bit indices (all 0: the first value).
    std::memset(upload, 0, BlockRow * Blocks);
    for (uint32_t block = 0; block < Blocks * Blocks; ++block) {
        upload[block * BlockBytes] = Reds[block];
        upload[block * BlockBytes + 1] = Reds[block];
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, BlockRow, 0, plain.Get(), 0, 0, {},
                                  {Blocks, Blocks, 1});
    scheduler.CopyTextureToBuffer(plain.Get(), 0, 0, {}, {Blocks, Blocks, 1}, between.Get(), 0,
                                  BlockRow, 0);
    scheduler.BreakBlits();
    scheduler.CopyBufferToTexture(between.Get(), 0, BlockRow, 0, compressed.Get(), 0, 0, {},
                                  {Texels, Texels, 1});
    scheduler.BreakBlits();
    scheduler.CopyTextureToBuffer(compressed.Get(), 0, 0, {}, {Texels, Texels, 1},
                                  blocks_buffer.Get(), 0, BlockRow, 0);
    const bool blitted = scheduler.Blit(Metal::BlitPass{
        .src = compressed.Get(),
        .src_rect = {0, 0, static_cast<int32_t>(Texels), static_cast<int32_t>(Texels)},
        .dst = {.texture = target.Get()},
        .dst_format = rgba8,
        .dst_rect = {0, 0, static_cast<int32_t>(Texels), static_cast<int32_t>(Texels)},
        .dst_width = Texels,
        .dst_height = Texels,
        .linear = false,
        .keep_dst = false,
    });
    scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {Texels, Texels, 1},
                                  readback_buffer.Get(), 0, Texels * 4, 0);
    scheduler.Finish();
    const bool same_blocks = std::memcmp(upload, blocks, BlockRow * Blocks) == 0;
    const auto* texels = reinterpret_cast<const Texel*>(readback);
    uint32_t wrong = 0;
    for (uint32_t y = 0; y < Texels; ++y) {
        for (uint32_t x = 0; x < Texels; ++x) {
            const uint8_t red = Reds[(y / 4) * Blocks + x / 4];
            wrong += Near(texels[y * Texels + x], {red, 0, 0, 255}) ? 0 : 1;
        }
    }
    const bool passed = blitted && same_blocks && wrong == 0;
    std::printf("metal-smoke: block copy 2x2 RGBA16Uint -> 8x8 BC4: blocks %s, %u of %u texels "
                "wrong: %s\n",
                same_blocks ? "kept" : "CHANGED", wrong, Texels * Texels,
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Clears a load action can't do, drawn (Scheduler::ClearRectangle): the middle 8x8 of 16x16
/// targets cleared in green only, in depth, in the stencil bits of mask 0x3C, and in integers.
int RunClearRect() {
    using Maxwell = Metal::Maxwell;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Side = 16;
    constexpr std::array<uint32_t, 4> Middle{4, 4, 8, 8};
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    const uint32_t packed = static_cast<uint32_t>(MTLPixelFormatDepth32Float_Stencil8);
    const auto make = [&](uint32_t format) {
        return Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                             .format = format,
                                             .width = Side,
                                             .height = Side,
                                             .render_target = true});
    };
    const Metal::Object color = make(rgba8);
    const Metal::Object integers = make(static_cast<uint32_t>(MTLPixelFormatRGBA8Uint));
    const Metal::Object depth = make(static_cast<uint32_t>(MTLPixelFormatDepth32Float));
    const Metal::Object depth_stencil = make(packed);
    const Metal::Object counted = make(rgba8);
    const auto readback = [&](uint8_t** data) {
        return Metal::CreateBuffer(device, Side * Side * 4, true, data);
    };
    uint8_t* color_data = nullptr;
    uint8_t* integer_data = nullptr;
    uint8_t* depth_data = nullptr;
    const Metal::Object color_buffer = readback(&color_data);
    const Metal::Object integer_buffer = readback(&integer_data);
    const Metal::Object depth_buffer = readback(&depth_data);
    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex =
        library ? Metal::CreateFunction(library.Get(), "quad_vertex") : Metal::Object{};
    const Metal::Object fragment =
        library ? Metal::CreateFunction(library.Get(), "quad_fragment") : Metal::Object{};
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = vertex.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = rgba8;
    desc.blend[0].write_mask = 0xF;
    desc.depth_format = packed;
    desc.stencil_format = packed;
    desc.samples = 1;
    desc.topology_class = 2;
    desc.rasterize = 1;
    desc.attributes[0] = {1, 0, 0, Maxwell::VertexAttribute::Type::Float,
                          Maxwell::VertexAttribute::Size::Size_R32_G32};
    desc.streams[0] = {1, 8, 0};
    const Metal::Object pipeline = Metal::CreateRenderPipelineState(device, desc, error);
    const Metal::Object equal_state = [&] {
        auto state = Metal::Zeroed<Metal::DepthStencilDesc>();
        state.stencil_test = 1;
        state.front = {Maxwell::StencilOp::Op::Keep_D3D, Maxwell::StencilOp::Op::Keep_D3D,
                       Maxwell::StencilOp::Op::Keep_D3D, Maxwell::ComparisonOp::Equal_D3D, 0xFF,
                       0};
        state.back = state.front;
        return Metal::CreateDepthStencilState(device, state);
    }();
    uint8_t* vertices = nullptr;
    const Metal::Object vertex_buffer = Metal::CreateBuffer(device, 48, true, &vertices);
    if (!color || !integers || !depth || !depth_stencil || !counted || !pipeline ||
        !equal_state || color_data == nullptr || integer_data == nullptr ||
        depth_data == nullptr || vertices == nullptr) {
        std::printf("metal-smoke: FAIL clear rectangle setup: %s\n", error.c_str());
        return 1;
    }
    const float positions[12]{-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};
    std::memcpy(vertices, positions, sizeof(positions));

    Metal::ClearPass whole{};
    whole.color[0] = {.texture = color.Get()};
    whole.clear_color[0] = true;
    whole.color_value[0] = {0.2, 0.4, 0.6, 1.0};
    whole.depth = {.texture = depth.Get()};
    whole.clear_depth = true;
    whole.depth_value = 1.0;
    whole.width = Side;
    whole.height = Side;
    scheduler.Clear(whole);
    Metal::ClearPass whole_stencil{};
    whole_stencil.depth = {.texture = depth_stencil.Get()};
    whole_stencil.stencil = {.texture = depth_stencil.Get()};
    whole_stencil.clear_depth = true;
    whole_stencil.clear_stencil = true;
    whole_stencil.depth_value = 1.0;
    whole_stencil.stencil_value = 0x0F;
    whole_stencil.width = Side;
    whole_stencil.height = Side;
    scheduler.Clear(whole_stencil);

    bool drawn = scheduler.ClearRectangle(Metal::ClearRect{
        .color = {.texture = color.Get()},
        .color_mask = 2,
        .color_value = {1.0, 1.0, 1.0, 0.0},
        .depth = {.texture = depth.Get()},
        .clear_depth = true,
        .depth_value = 0.5f,
        .rect = Middle,
        .width = Side,
        .height = Side,
    });
    // Stencil bits 0x3C of 0x0F become those of 0xF0: 0x33. The depth beside them stays.
    drawn = scheduler.ClearRectangle(Metal::ClearRect{
        .depth = {.texture = depth_stencil.Get()},
        .clear_stencil = true,
        .stencil_value = 0xF0,
        .stencil_mask = 0x3C,
        .rect = Middle,
        .width = Side,
        .height = Side,
    }) && drawn;
    drawn = scheduler.ClearRectangle(Metal::ClearRect{
        .color = {.texture = integers.Get()},
        .color_mask = 0xF,
        .color_value = {7.0, 8.0, 9.0, 10.0},
        .rect = {0, 0, Side, Side},
        .width = Side,
        .height = Side,
    }) && drawn;
    scheduler.CopyTextureToBuffer(color.Get(), 0, 0, {}, {Side, Side, 1}, color_buffer.Get(), 0,
                                  Side * 4, 0);
    scheduler.CopyTextureToBuffer(integers.Get(), 0, 0, {}, {Side, Side, 1},
                                  integer_buffer.Get(), 0, Side * 4, 0);
    scheduler.CopyTextureToBuffer(depth.Get(), 0, 0, {}, {Side, Side, 1}, depth_buffer.Get(), 0,
                                  Side * 4, 0);
    // The stencil is counted: a quad over the target at reference 0x33, then at 0x0F.
    for (uint32_t pass = 0; pass < 2; ++pass) {
        Metal::RenderTargetSet targets{};
        targets.color[0] = {.texture = counted.Get()};
        targets.depth = {.texture = depth_stencil.Get()};
        targets.has_depth = true;
        targets.has_stencil = true;
        targets.width = Side;
        targets.height = Side;
        scheduler.BeginRendering(targets);
        Metal::DrawState state{};
        state.pipeline = pipeline.Get();
        state.depth_stencil = equal_state.Get();
        state.viewport = {0.0f, 0.0f, float(Side), float(Side), 0.0f, 1.0f};
        state.scissor = {0, 0, Side, Side};
        state.stencil_front_ref = pass == 0 ? 0x33 : 0x0F;
        state.stencil_back_ref = state.stencil_front_ref;
        scheduler.SetDrawState(state);
        scheduler.BindVertexBuffer(Metal::VertexBufferSlot(0), vertex_buffer.Get(), 0);
        scheduler.VisibilityResults()[5 + pass] = 0;
        scheduler.SetVisibility(5 + pass);
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    }
    scheduler.Finish();
    const uint64_t inside = scheduler.VisibilityResults()[5];
    const uint64_t outside = scheduler.VisibilityResults()[6];
    const auto* texels = reinterpret_cast<const Texel*>(color_data);
    uint32_t wrong_color = 0;
    uint32_t wrong_integer = 0;
    uint32_t wrong_depth = 0;
    for (uint32_t y = 0; y < Side; ++y) {
        for (uint32_t x = 0; x < Side; ++x) {
            const bool middle = x >= Middle[0] && x < Middle[0] + Middle[2] && y >= Middle[1] &&
                                y < Middle[1] + Middle[3];
            const std::array<uint8_t, 4> expected{51, static_cast<uint8_t>(middle ? 255 : 102),
                                                  153, 255};
            wrong_color += Near(texels[y * Side + x], expected) ? 0 : 1;
            const uint8_t* const integer = integer_data + (y * Side + x) * 4;
            wrong_integer +=
                integer[0] == 7 && integer[1] == 8 && integer[2] == 9 && integer[3] == 10 ? 0 : 1;
            float value;
            std::memcpy(&value, depth_data + (y * Side + x) * 4, 4);
            wrong_depth += value == (middle ? 0.5f : 1.0f) ? 0 : 1;
        }
    }
    const uint64_t middle_samples = Middle[2] * Middle[3];
    const bool passed = drawn && wrong_color == 0 && wrong_integer == 0 && wrong_depth == 0 &&
                        inside == middle_samples && outside == Side * Side - middle_samples;
    std::printf("metal-smoke: clear rectangle 8x8 of 16x16 (%s): %u colour, %u integer, %u depth "
                "texels wrong, stencil 0x33 on %llu samples (of %llu), 0x0F on %llu: %s\n",
                drawn ? "drawn" : "refused", wrong_color, wrong_integer, wrong_depth,
                static_cast<unsigned long long>(inside),
                static_cast<unsigned long long>(middle_samples),
                static_cast<unsigned long long>(outside), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// A clear waiting for its render pass (EDEN_NX2_METAL_CLEAR_FOLD): the load action of a pass on
/// its target, recorded first when a copy reads the target or a pass on another target comes.
int RunClearFold() {
    using Maxwell = Metal::Maxwell;
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
    const auto make = [&] {
        return Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                             .format = rgba8,
                                             .width = Size,
                                             .height = Size,
                                             .render_target = true});
    };
    const Metal::Object first = make();
    const Metal::Object second = make();
    uint8_t* first_data = nullptr;
    uint8_t* second_data = nullptr;
    uint8_t* late_data = nullptr;
    const auto readback = [&](uint8_t** data) {
        return Metal::CreateBuffer(device, Size * Size * 4, true, data);
    };
    const Metal::Object first_buffer = readback(&first_data);
    const Metal::Object second_buffer = readback(&second_data);
    const Metal::Object late_buffer = readback(&late_data);
    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex =
        library ? Metal::CreateFunction(library.Get(), "quad_vertex") : Metal::Object{};
    const Metal::Object fragment =
        library ? Metal::CreateFunction(library.Get(), "quad_fragment") : Metal::Object{};
    auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
    desc.vertex_function = vertex.Get();
    desc.fragment_function = fragment.Get();
    desc.color_formats[0] = rgba8;
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
    uint8_t* vertices = nullptr;
    const Metal::Object vertex_buffer = Metal::CreateBuffer(device, 48, true, &vertices);
    if (!first || !second || !pipeline || !depth_stencil || first_data == nullptr ||
        second_data == nullptr || late_data == nullptr || vertices == nullptr) {
        std::printf("metal-smoke: FAIL clear fold setup: %s\n", error.c_str());
        return 1;
    }
    // The left half of the target.
    const float positions[12]{-1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1, 1};
    std::memcpy(vertices, positions, sizeof(positions));

    const auto clear = [&](const Metal::Object& texture, std::array<double, 4> value) {
        Metal::ClearPass pass{};
        pass.color[0] = {.texture = texture.Get()};
        pass.clear_color[0] = true;
        pass.color_value[0] = value;
        pass.width = Size;
        pass.height = Size;
        scheduler.Clear(pass);
    };
    const auto draw_left = [&](const Metal::Object& texture) {
        Metal::RenderTargetSet targets{};
        targets.color[0] = {.texture = texture.Get()};
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
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    };
    const auto read = [&](const Metal::Object& texture, const Metal::Object& buffer) {
        scheduler.CopyTextureToBuffer(texture.Get(), 0, 0, {}, {Size, Size, 1}, buffer.Get(), 0,
                                      Size * 4, 0);
    };
    (void)scheduler.TakeCounts();
    // Red as the load action of the pass that draws the left half green.
    clear(first, {1.0, 0.0, 0.0, 1.0});
    draw_left(first);
    // Blue read by a copy before any pass on the target.
    clear(second, {0.0, 0.0, 1.0, 1.0});
    read(second, second_buffer);
    read(first, first_buffer);
    // Two clears of one target are one, the later value; a pass on another target comes after it.
    clear(first, {1.0, 0.0, 0.0, 1.0});
    clear(first, {1.0, 1.0, 0.0, 1.0});
    draw_left(second);
    read(first, late_buffer);
    scheduler.Finish();
    const auto counts = scheduler.TakeCounts();
    const uint64_t passes = counts[3];
    const uint64_t folded = counts[4];

    const auto texel = [](const uint8_t* data, uint32_t x) {
        return reinterpret_cast<const Texel*>(data)[(Size / 2) * Size + x];
    };
    const bool pictures = Near(texel(first_data, 4), {0, 255, 0, 255}) &&
                          Near(texel(first_data, Size - 4), {255, 0, 0, 255}) &&
                          Near(texel(second_data, 4), {0, 0, 255, 255}) &&
                          Near(texel(second_data, Size - 4), {0, 0, 255, 255}) &&
                          Near(texel(late_data, 4), {255, 255, 0, 255}) &&
                          Near(texel(late_data, Size - 4), {255, 255, 0, 255});
    // Folding: the first clear rides its pass, the other two get one pass each (4 with the two
    // draws' passes); without it each of the four clears has a pass (6).
    const char* const value = std::getenv("EDEN_NX2_METAL_CLEAR_FOLD");
    const bool folding = value == nullptr || value[0] != '0';
    const bool counted = folding ? folded == 1 && passes == 4 : folded == 0 && passes == 6;
    std::printf("metal-smoke: clear fold (%s): pictures %s; %llu render passes, %llu starting "
                "with a clear: %s\n",
                folding ? "on" : "off", pictures ? "PASS" : "FAIL",
                static_cast<unsigned long long>(passes), static_cast<unsigned long long>(folded),
                counted ? "PASS" : "FAIL");
    return pictures && counted ? 0 : 1;
}

/// What the two-quad cases below share: a pipeline drawing QuadSource into RGBA8, a vertex
/// buffer of two quads (the left half of the target, then the right half) and read-backs.
struct QuadScene {
    explicit QuadScene(const Metal::Device& device) {
        using Maxwell = Metal::Maxwell;
        const uint32_t rgba8 = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
        for (size_t index = 0; index < textures.size(); ++index) {
            textures[index] = Metal::CreateTexture(device, {.kind = Metal::TextureKind::e2D,
                                                            .format = rgba8,
                                                            .width = Size,
                                                            .height = Size,
                                                            .render_target = true});
            readbacks[index] = Metal::CreateBuffer(device, Size * Size * 4, true, &pixels[index]);
        }
        library = Metal::CreateLibrary(device, QuadSource, error);
        if (library) {
            vertex = Metal::CreateFunction(library.Get(), "quad_vertex");
            fragment = Metal::CreateFunction(library.Get(), "quad_fragment");
        }
        auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
        desc.vertex_function = vertex.Get();
        desc.fragment_function = fragment.Get();
        desc.color_formats[0] = rgba8;
        desc.blend[0].write_mask = 0xF;
        desc.samples = 1;
        desc.topology_class = 2;
        desc.rasterize = 1;
        desc.attributes[0] = {1, 0, 0, Maxwell::VertexAttribute::Type::Float,
                              Maxwell::VertexAttribute::Size::Size_R32_G32};
        desc.streams[0] = {1, 8, 0};
        pipeline = Metal::CreateRenderPipelineState(device, desc, error);
        depth_stencil =
            Metal::CreateDepthStencilState(device, Metal::Zeroed<Metal::DepthStencilDesc>());
        vertex_buffer = Metal::CreateBuffer(device, 2 * QuadBytes, true, &vertices);
        staging = Metal::CreateBuffer(device, QuadBytes, true, &staged);
        valid = textures[0] && textures[1] && pipeline && depth_stencil && pixels[0] != nullptr &&
                pixels[1] != nullptr && vertices != nullptr && staged != nullptr;
        if (valid) {
            std::memset(vertices, 0, 2 * QuadBytes);
            std::memcpy(vertices, Left.data(), QuadBytes);
            std::memcpy(staged, Right.data(), QuadBytes);
        }
    }

    Metal::RenderTargetSet Targets(size_t index) const {
        Metal::RenderTargetSet targets{};
        targets.color[0] = {.texture = textures[index].Get()};
        targets.width = Size;
        targets.height = Size;
        return targets;
    }

    void Clear(Metal::Scheduler& scheduler, size_t index, std::array<double, 4> value) const {
        Metal::ClearPass pass{};
        pass.color[0] = {.texture = textures[index].Get()};
        pass.clear_color[0] = true;
        pass.color_value[0] = value;
        pass.width = Size;
        pass.height = Size;
        scheduler.Clear(pass);
    }

    /// One quad of the vertex buffer, within the upper `rows` rows of the target.
    void Draw(Metal::Scheduler& scheduler, size_t index, uint32_t quad, uint32_t rows) const {
        scheduler.BeginRendering(Targets(index));
        Metal::DrawState state{};
        state.pipeline = pipeline.Get();
        state.depth_stencil = depth_stencil.Get();
        state.viewport = {0.0f, 0.0f, float(Size), float(Size), 0.0f, 1.0f};
        state.scissor = {0, 0, Size, rows};
        scheduler.SetDrawState(state);
        scheduler.BindVertexBuffer(Metal::VertexBufferSlot(0), vertex_buffer.Get(),
                                   quad * QuadBytes);
        scheduler.Draw(static_cast<uint32_t>(MTLPrimitiveTypeTriangle), 0, 6, 1, 0);
    }

    void Read(Metal::Scheduler& scheduler, size_t index) const {
        scheduler.CopyTextureToBuffer(textures[index].Get(), 0, 0, {}, {Size, Size, 1},
                                      readbacks[index].Get(), 0, Size * 4, 0);
    }

    Texel At(size_t index, uint32_t x, uint32_t y) const {
        return reinterpret_cast<const Texel*>(pixels[index])[y * Size + x];
    }

    static constexpr uint32_t QuadBytes = 48;
    static constexpr std::array<float, 12> Left{-1, -1, 0, -1, 0, 1, -1, -1, 0, 1, -1, 1};
    static constexpr std::array<float, 12> Right{0, -1, 1, -1, 1, 1, 0, -1, 1, 1, 0, 1};

    std::array<Metal::Object, 2> textures;
    std::array<Metal::Object, 2> readbacks;
    std::array<uint8_t*, 2> pixels{};
    Metal::Object library;
    Metal::Object vertex;
    Metal::Object fragment;
    Metal::Object pipeline;
    Metal::Object depth_stencil;
    Metal::Object vertex_buffer;
    Metal::Object staging;
    uint8_t* vertices{};
    uint8_t* staged{};
    std::string error;
    bool valid{};
};

bool SwitchOn(const char* name) {
    const char* const value = std::getenv(name);
    return value == nullptr || value[0] != '0';
}

/// State and bindings an encoder already has are not set again (EDEN_NX2_METAL_STATE_CACHE):
/// draws of one pass differing in the scissor and the vertex buffer's offset, then the same
/// state in a new pass, which has nothing yet.
int RunStateCache() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const QuadScene scene(device);
    if (!scene.valid) {
        std::printf("metal-smoke: FAIL state cache setup: %s\n", scene.error.c_str());
        return 1;
    }
    std::memcpy(scene.vertices + QuadScene::QuadBytes, QuadScene::Right.data(),
                QuadScene::QuadBytes);
    (void)scheduler.TakeCounts();
    scene.Clear(scheduler, 0, {1.0, 0.0, 0.0, 1.0});
    scene.Draw(scheduler, 0, 0, Size);
    // The right half, upper rows only: another scissor, the same buffer further in.
    scene.Draw(scheduler, 0, 1, Size / 2);
    scene.Draw(scheduler, 0, 1, Size / 2);
    scene.Clear(scheduler, 1, {0.0, 0.0, 1.0, 1.0});
    scene.Draw(scheduler, 1, 1, Size / 2);
    scene.Read(scheduler, 0);
    scene.Read(scheduler, 1);
    scheduler.Finish();
    const auto counts = scheduler.TakeCounts();
    const uint64_t sets = counts[5];
    const uint64_t skips = counts[6];

    const uint32_t upper = Size / 4;
    const uint32_t lower = Size - Size / 4;
    const bool pictures = Near(scene.At(0, 4, upper), {0, 255, 0, 255}) &&
                          Near(scene.At(0, 4, lower), {0, 255, 0, 255}) &&
                          Near(scene.At(0, Size - 4, upper), {0, 255, 0, 255}) &&
                          Near(scene.At(0, Size - 4, lower), {255, 0, 0, 255}) &&
                          Near(scene.At(1, 4, upper), {0, 0, 255, 255}) &&
                          Near(scene.At(1, Size - 4, upper), {0, 255, 0, 255}) &&
                          Near(scene.At(1, Size - 4, lower), {0, 0, 255, 255});
    // A draw is ten pieces of state and one buffer. The first of a pass sets them all (11); the
    // second sets the scissor and the offset (2 set, 9 left); the third nothing (11 left).
    // Vulkan's EDEN_NX2_DYNSTATE_CACHE=0 sets the ten every draw, EDEN_NX2_VERTEX_INPUT_CACHE=0
    // the buffer (both read under EDEN_NX2_METAL_VULKAN_SWITCHES).
    const bool caching = SwitchOn("EDEN_NX2_METAL_STATE_CACHE");
    const bool vulkan = SwitchOn("EDEN_NX2_METAL_VULKAN_SWITCHES");
    const bool state_cached = caching && !(vulkan && !SwitchOn("EDEN_NX2_DYNSTATE_CACHE"));
    const bool buffers_cached = caching && !(vulkan && !SwitchOn("EDEN_NX2_VERTEX_INPUT_CACHE"));
    const uint64_t state_sets = state_cached ? 10 + 1 + 0 + 10 : 40;
    const uint64_t buffer_sets = buffers_cached ? 1 + 1 + 0 + 1 : 4;
    const bool counted = sets == state_sets + buffer_sets && skips == 44 - sets;
    std::printf("metal-smoke: state cache (%s, draw state %s, vertex buffers %s): pictures %s; "
                "%llu set, %llu left as they were: %s\n",
                caching ? "on" : "off", state_cached ? "cached" : "set every draw",
                buffers_cached ? "cached" : "bound every draw", pictures ? "PASS" : "FAIL",
                static_cast<unsigned long long>(sets), static_cast<unsigned long long>(skips),
                counted ? "PASS" : "FAIL");
    return pictures && counted ? 0 : 1;
}

/// Runs this program again with `--child <name>` and `variables` (NAME=VALUE) added to its
/// environment; the child's exit code, -1 when it could not run.
int RunChildWith(const char* name, std::initializer_list<const char*> variables) {
    char path[4096];
    uint32_t path_size = sizeof(path);
    if (_NSGetExecutablePath(path, &path_size) != 0) {
        return -1;
    }
    std::vector<std::string> owned(variables.begin(), variables.end());
    std::vector<char*> environment;
    for (char** entry = environ; *entry != nullptr; ++entry) {
        environment.push_back(*entry);
    }
    for (std::string& variable : owned) {
        environment.push_back(variable.data());
    }
    environment.push_back(nullptr);
    char child_flag[] = "--child";
    std::string child_name = name;
    char* const arguments[] = {path, child_flag, child_name.data(), nullptr};
    pid_t child = 0;
    if (posix_spawn(&child, path, nullptr, nullptr, arguments, environment.data()) != 0) {
        return -1;
    }
    int status = 0;
    waitpid(child, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/// Vulkan's switches the Metal renderer reads (EDEN_NX2_METAL_VULKAN_SWITCHES), each read once
/// per process: the state cache test in children with EDEN_NX2_DYNSTATE_CACHE=0,
/// EDEN_NX2_VERTEX_INPUT_CACHE=0, and both with the Vulkan switches ignored.
int RunVulkanSwitches() {
    const int draw_state = RunChildWith("state-cache", {"EDEN_NX2_DYNSTATE_CACHE=0"});
    const int vertex_input = RunChildWith("state-cache", {"EDEN_NX2_VERTEX_INPUT_CACHE=0"});
    const int ignored =
        RunChildWith("state-cache", {"EDEN_NX2_DYNSTATE_CACHE=0", "EDEN_NX2_VERTEX_INPUT_CACHE=0",
                                     "EDEN_NX2_METAL_VULKAN_SWITCHES=0"});
    const int fence_batch = RunChildWith(
        "fence-batch", {"EDEN_NX2_FENCE_FLUSH_US=10000000", "EDEN_NX2_METAL_FENCE_BATCH=2"});
    const bool passed = draw_state == 0 && vertex_input == 0 && ignored == 0 && fence_batch == 0;
    std::printf("metal-smoke: Vulkan switches in children: EDEN_NX2_DYNSTATE_CACHE=0 %d, "
                "EDEN_NX2_VERTEX_INPUT_CACHE=0 %d, ignored %d, EDEN_NX2_FENCE_FLUSH_US %d (0 "
                "each): %s\n",
                draw_state, vertex_input, ignored, fence_batch, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Uploads and kernels ahead of their segment (EDEN_NX2_METAL_AHEAD): a copy and a compute
/// dispatch recorded while a render pass is open leave it open and run before it; a pass that
/// wanted its own segment gets one (EDEN_NX2_METAL_AHEAD_SPLIT).
int RunAhead() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const QuadScene scene(device);
    constexpr uint32_t Marks = 16;
    uint8_t* marks = nullptr;
    uint8_t* late = nullptr;
    const Metal::Object mark_buffer = Metal::CreateBuffer(device, Marks * 4, true, &marks);
    const Metal::Object late_buffer =
        Metal::CreateBuffer(device, QuadScene::QuadBytes, true, &late);
    std::string error;
    const Metal::Object mark_library = Metal::CreateLibrary(device, GroupMarkSource, error);
    const Metal::Object mark_function =
        mark_library ? Metal::CreateFunction(mark_library.Get(), "mark_groups") : Metal::Object{};
    const Metal::Object mark_state =
        mark_function ? Metal::CreateComputePipelineState(device, mark_function.Get(), error)
                      : Metal::Object{};
    if (!scene.valid || !mark_state || marks == nullptr || late == nullptr) {
        std::printf("metal-smoke: FAIL ahead setup: %s %s\n", scene.error.c_str(), error.c_str());
        return 1;
    }
    std::memset(marks, 0, Marks * 4);
    std::memset(late, 0, QuadScene::QuadBytes);
    const Metal::RenderTargetSet first = scene.Targets(0);
    const Metal::RenderTargetSet second = scene.Targets(1);

    (void)scheduler.TakeCounts();
    scene.Draw(scheduler, 0, 0, Size);
    // The second quad's vertices arrive while the pass is open, as the buffer cache's uploads
    // do; then a kernel, as the index rewrite, mesh and tessellation kernels of a draw.
    scheduler.CopyBuffer(scene.staging.Get(), 0, scene.vertex_buffer.Get(), QuadScene::QuadBytes,
                         QuadScene::QuadBytes, true);
    const bool open_after_copy = scheduler.IsRenderingTo(first);
    const bool noted_ahead = scheduler.UsedInOrder(scene.vertex_buffer.Get());
    scheduler.BeginCompute(mark_state.Get(), scheduler.CanRecordAhead(first));
    scheduler.BindBuffer(Metal::ShaderStage::Compute, 0, mark_buffer.Get(), 0);
    scheduler.Dispatch({Marks, 1, 1}, {1, 1, 1});
    const bool open_after_kernel = scheduler.IsRenderingTo(first);
    scene.Draw(scheduler, 0, 1, Size);
    // The other target's pass asked for a segment of its own; a pass that did not gets none.
    scene.Clear(scheduler, 1, {0.0, 0.0, 1.0, 1.0});
    scheduler.WantSplitBefore(second);
    const uint64_t before_split = scheduler.SegmentSerial();
    scheduler.SplitBefore(second);
    const bool split = scheduler.SegmentSerial() != before_split;
    scene.Draw(scheduler, 1, 0, Size);
    scheduler.SplitBefore(second);
    scheduler.SplitBefore(first);
    // A copy in its place: the buffer takes no copy ahead for the rest of the segment.
    scheduler.CopyBuffer(scene.staging.Get(), 0, late_buffer.Get(), 0, QuadScene::QuadBytes);
    const bool noted_in_order = scheduler.UsedInOrder(late_buffer.Get());
    scene.Read(scheduler, 0);
    scene.Read(scheduler, 1);
    scheduler.Finish();
    const bool noted_after = scheduler.UsedInOrder(late_buffer.Get());
    const auto counts = scheduler.TakeCounts();
    const uint64_t command_buffers = counts[1];
    const uint64_t passes = counts[3];
    const uint64_t splits = counts[7];
    const uint64_t ahead_copies = counts[8];
    const uint64_t ahead_kernels = counts[9];

    uint32_t marked = 0;
    for (uint32_t index = 0; index < Marks; ++index) {
        uint32_t mark;
        std::memcpy(&mark, marks + index * 4, sizeof(mark));
        marked += mark == 1 ? 1 : 0;
    }
    const uint32_t row = Size / 2;
    const bool results = Near(scene.At(0, 4, row), {0, 255, 0, 255}) &&
                         Near(scene.At(0, Size - 4, row), {0, 255, 0, 255}) &&
                         Near(scene.At(1, 4, row), {0, 255, 0, 255}) &&
                         Near(scene.At(1, Size - 4, row), {0, 0, 255, 255}) && marked == Marks &&
                         std::memcmp(late, QuadScene::Right.data(), QuadScene::QuadBytes) == 0 &&
                         noted_in_order && !noted_after;
    // Ahead: the first target's pass stays one (2 with the other target's), the copy and the
    // kernel have a command buffer each, and so has the segment split off (4 with the last).
    // In order: the copy ends the pass, the second quad begins another (3), one command buffer.
    // A clear that does not wait for its pass (EDEN_NX2_METAL_CLEAR_FOLD=0) is one pass more.
    const bool ahead = SwitchOn("EDEN_NX2_METAL_AHEAD");
    const bool splitting = ahead && SwitchOn("EDEN_NX2_METAL_AHEAD_SPLIT");
    const uint64_t clear_passes = SwitchOn("EDEN_NX2_METAL_CLEAR_FOLD") ? 0 : 1;
    const bool counted =
        ahead ? open_after_copy && open_after_kernel && !noted_ahead &&
                    passes == 2 + clear_passes && ahead_copies == 1 && ahead_kernels == 1 &&
                    split == splitting && splits == (splitting ? 1U : 0U) &&
                    command_buffers == (splitting ? 4U : 3U)
              : !open_after_copy && !open_after_kernel && noted_ahead &&
                    passes == 3 + clear_passes && ahead_copies == 0 && ahead_kernels == 0 &&
                    !split && splits == 0 && command_buffers == 1;
    std::printf("metal-smoke: ahead of the segment (%s, split %s): results %s; %llu render "
                "passes, %llu copies and %llu kernels ahead, %llu segments split off, %llu "
                "command buffers: %s\n",
                ahead ? "on" : "off", splitting ? "on" : "off", results ? "PASS" : "FAIL",
                static_cast<unsigned long long>(passes),
                static_cast<unsigned long long>(ahead_copies),
                static_cast<unsigned long long>(ahead_kernels),
                static_cast<unsigned long long>(splits),
                static_cast<unsigned long long>(command_buffers), counted ? "PASS" : "FAIL");
    return results && counted ? 0 : 1;
}

/// A guest fence with nothing recorded since the last commit takes no command buffer and is
/// free once that commit is done (EDEN_NX2_METAL_FENCE_IDLE).
int RunFenceIdle() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const QuadScene scene(device);
    if (!scene.valid) {
        std::printf("metal-smoke: FAIL fence setup: %s\n", scene.error.c_str());
        return 1;
    }
    (void)scheduler.TakeCounts();
    const uint64_t start = scheduler.CurrentTick();
    // Before anything was recorded, after a draw, and twice with nothing new.
    const uint64_t first = scheduler.FenceTick();
    scene.Draw(scheduler, 0, 0, Size);
    const uint64_t working = scheduler.FenceTick();
    const bool committed = !scheduler.HasPendingWork() && scheduler.IsSubmitted(working);
    const uint64_t idle = scheduler.FenceTick();
    const uint64_t again = scheduler.FenceTick();
    scheduler.Wait(again);
    scheduler.WaitSubmitted(again);
    const bool all_free = scheduler.IsFree(first) && scheduler.IsFree(working) &&
                      scheduler.IsFree(idle) && scheduler.IsFree(again);
    const auto counts = scheduler.TakeCounts();
    const uint64_t command_buffers = counts[1];
    const uint64_t idle_fences = counts[10];
    scene.Read(scheduler, 0);
    scheduler.Finish();
    const bool results = committed && all_free &&
                         Near(scene.At(0, 4, Size / 2), {0, 255, 0, 255});
    const bool skipping = SwitchOn("EDEN_NX2_METAL_FENCE_IDLE");
    const bool counted =
        skipping ? first == start - 1 && working == start && idle == working && again == working &&
                       command_buffers == 1 && idle_fences == 3
                 : first == start && working == start + 1 && idle == start + 2 &&
                       again == start + 3 && command_buffers == 4 && idle_fences == 0;
    std::printf("metal-smoke: fences with nothing pending (%s): results %s; %llu command "
                "buffers for 4 fences, %llu without one: %s\n",
                skipping ? "on" : "off", results ? "PASS" : "FAIL",
                static_cast<unsigned long long>(command_buffers),
                static_cast<unsigned long long>(idle_fences), counted ? "PASS" : "FAIL");
    return results && counted ? 0 : 1;
}

/// Fences batched (EDEN_NX2_METAL_FENCE_BATCH with Vulkan's EDEN_NX2_FENCE_FLUSH_US, run in a
/// child with a 10 s window and =2 so it applies off the split's back): two fences after draws
/// within the window share the tick being recorded and commit nothing, the idle hook commits it.
int RunFenceBatch() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    const QuadScene scene(device);
    if (!scene.valid) {
        std::printf("metal-smoke: FAIL fence batch setup: %s\n", scene.error.c_str());
        return 1;
    }
    scheduler.Flush();
    (void)scheduler.TakeCounts();
    (void)scheduler.TakeDeferredFences();
    scene.Draw(scheduler, 0, 0, Size);
    const uint64_t first = scheduler.FenceTick();
    const bool first_pending = !scheduler.IsSubmitted(first) && scheduler.HasPendingWork();
    scene.Draw(scheduler, 0, 1, Size / 2);
    const uint64_t second = scheduler.FenceTick();
    const bool may_wait = scheduler.DeferredFlushMayWait();
    const bool idle_committed = scheduler.FlushDeferredFromIdle(false);
    const bool submitted = scheduler.IsSubmitted(second);
    const bool nothing_left = !scheduler.FlushDeferredFromIdle(false);
    scheduler.Wait(second);
    scheduler.WaitSubmitted(second);
    const auto counts = scheduler.TakeCounts();
    const auto [deferred, idle_commits] = scheduler.TakeDeferredFences();
    const bool passed = first == second && first_pending && may_wait && idle_committed &&
                        submitted && nothing_left && scheduler.IsFree(second) &&
                        counts[1] == 1 && deferred == 2 && idle_commits == 1;
    std::printf("metal-smoke: fence batch: ticks %llu/%llu, pending %d, may wait %d, idle commit "
                "%d, %llu command buffers, %llu fences left pending, %llu idle commits: %s\n",
                static_cast<unsigned long long>(first), static_cast<unsigned long long>(second),
                first_pending ? 1 : 0, may_wait ? 1 : 0, idle_committed ? 1 : 0,
                static_cast<unsigned long long>(counts[1]),
                static_cast<unsigned long long>(deferred),
                static_cast<unsigned long long>(idle_commits), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Staging buffers bounded (EDEN_NX2_METAL_STAGING_BOUND): a command buffer that took its mark's
/// worth asks to be committed, over the cap the buffers the GPU is done with go, a slot freed is
/// used again, and a deferred buffer stays through all of it.
int RunStagingBound() {
    const bool bounded = SwitchOn("EDEN_NX2_METAL_STAGING_BOUND");
    constexpr uint64_t MiB = 1 << 20;
    constexpr size_t Chunk = 512 << 10;
    // Marks of 1 and 2 MiB for this pool.
    setenv("EDEN_NX2_STAGING_SUBMIT_MIB", "1", 1);
    setenv("EDEN_NX2_STAGING_CAP_MIB", "2", 1);
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    Metal::StagingBufferPool pool(device, scheduler);
    unsetenv("EDEN_NX2_STAGING_SUBMIT_MIB");
    unsetenv("EDEN_NX2_STAGING_CAP_MIB");
    const auto finish = [&scheduler] {
        const uint64_t tick = scheduler.Flush();
        scheduler.Wait(tick);
        scheduler.WaitSubmitted(tick);
    };

    bool asked_early = false;
    for (uint32_t index = 0; index < 6; ++index) {
        const Metal::StagingBufferRef ref = pool.Request(Chunk);
        std::memset(ref.mapped_span.data(), static_cast<int>(index), 16);
        asked_early |= index == 0 && pool.WantsFlush();
    }
    const bool asked = pool.WantsFlush();
    const bool over = pool.OverCap();
    const uint64_t held_taken = pool.HeldBytes();
    finish();
    const bool asked_after = pool.WantsFlush();
    pool.Trim();
    const uint64_t held_trimmed = pool.HeldBytes();
    const uint64_t freed_trimmed = pool.TakeFreed();
    (void)pool.Request(Chunk);
    const uint64_t held_again = pool.HeldBytes();
    Metal::StagingBufferRef kept = pool.Request(4 * Chunk, true);
    finish();
    for (uint32_t tick = 0; tick < 10; ++tick) {
        pool.TickFrame();
    }
    const uint64_t held_ticked = pool.HeldBytes();
    const uint64_t freed_ticked = pool.TakeFreed();
    std::memset(kept.mapped_span.data(), 0x5A, kept.mapped_span.size());
    pool.FreeDeferred(kept);

    // EDEN_NX2_STREAM_BUFFER_MIB=8: a ring of 8 MiB, regions of 512 KiB; a larger slice is a
    // buffer of its own, a smaller one is in the ring.
    setenv("EDEN_NX2_STREAM_BUFFER_MIB", "8", 1);
    Metal::StagingBufferPool small_ring(device, scheduler);
    unsetenv("EDEN_NX2_STREAM_BUFFER_MIB");
    const Metal::StagingBufferPool default_ring(device, scheduler);
    const Metal::StagingBufferRef in_ring = small_ring.Stream(Chunk, 16);
    const Metal::StagingBufferRef own = small_ring.Stream(Chunk + 16, 16);
    constexpr size_t NotPooled = (std::numeric_limits<size_t>::max)();
    const bool stream_sized = small_ring.StreamSize() == 8 * MiB &&
                              default_ring.StreamSize() == 32 * MiB && in_ring.index == NotPooled &&
                              own.index != NotPooled;
    finish();

    const bool bound_passed =
        bounded ? !asked_early && asked && over && held_taken == 3 * MiB && !asked_after &&
                      held_trimmed == MiB && freed_trimmed == 4 && held_again == MiB &&
                      held_ticked == 2 * MiB && freed_ticked == 2
                : !asked_early && !asked && !over && held_taken == 3 * MiB && !asked_after &&
                      held_trimmed == 3 * MiB && freed_trimmed == 0 && held_again == 3 * MiB &&
                      held_ticked == 5 * MiB && freed_ticked == 0;
    const bool passed = bound_passed && stream_sized;
    std::printf("metal-smoke: staging bound (%s): 6 buffers of 512 KiB in one command buffer "
                "ask for a commit %d, over the cap %d; held %llu KiB, %llu after the trim (%llu "
                "freed), %llu with one more, %llu after 10 frames with 2 MiB deferred (%llu "
                "freed); stream ring %s for EDEN_NX2_STREAM_BUFFER_MIB: %s\n",
                bounded ? "on" : "off", asked ? 1 : 0, over ? 1 : 0,
                static_cast<unsigned long long>(held_taken >> 10),
                static_cast<unsigned long long>(held_trimmed >> 10),
                static_cast<unsigned long long>(freed_trimmed),
                static_cast<unsigned long long>(held_again >> 10),
                static_cast<unsigned long long>(held_ticked >> 10),
                static_cast<unsigned long long>(freed_ticked), stream_sized ? "sized" : "NOT sized",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Ocarina's temporal upscale (EDEN_NX2_METAL_OOT_TEMPORAL): the camera motion pass
/// (ocarina_camera_motion.comp through the translator) over a 16x16 Depth32Float scene at depth
/// 0.5 with a camera that moved a quarter of clip space right: 2 pixels of motion everywhere.
/// Where MetalFX temporal upscaling runs, the whole replacement of a flat scene into a 32x32
/// RG11B10Float target too: the squared colour everywhere.
int RunOcarinaTemporal() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t SceneSize = 16;
    constexpr uint32_t TargetSize = 32;
    const Metal::Object depth = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatDepth32Float),
                 .width = SceneSize,
                 .height = SceneSize,
                 .render_target = true});
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer =
        Metal::CreateBuffer(device, SceneSize * SceneSize * 16, true, &upload);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, TargetSize * TargetSize * 4, true, &readback);
    if (!depth || upload == nullptr || readback == nullptr) {
        std::printf("metal-smoke: FAIL Ocarina temporal textures\n");
        return 1;
    }
    for (uint32_t texel = 0; texel < SceneSize * SceneSize; ++texel) {
        const float value = 0.5f;
        std::memcpy(upload + texel * 4, &value, 4);
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, SceneSize * 4, 0, depth.Get(), 0, 0, {},
                                  {SceneSize, SceneSize, 1});
    // Row-major, as the shader's push constant block reads it: previous x = current x + 0.25.
    VideoCommon::OcarinaMatrix moved{1, 0, 0, 0.25f, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Metal::OcarinaTemporal temporal(device, scheduler);
    const bool encoded = temporal.EncodeMotion(depth.Get(), {SceneSize, SceneSize}, moved);
    if (encoded) {
        scheduler.CopyTextureToBuffer(temporal.Motion(), 0, 0, {}, {SceneSize, SceneSize, 1},
                                      readback_buffer.Get(), 0, SceneSize * 4, 0);
    }
    scheduler.Finish();
    if (!encoded) {
        std::printf("metal-smoke: FAIL Ocarina temporal motion pass not encoded (see the log)\n");
        return 1;
    }
    const auto half = [](uint16_t bits) {
        const float magnitude = PackedFloat(bits & 0x7FFFU, 10);
        return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
    };
    uint32_t motion_wrong = 0;
    std::array<float, 2> sample{};
    for (uint32_t texel = 0; texel < SceneSize * SceneSize; ++texel) {
        std::array<uint16_t, 2> bits{};
        std::memcpy(bits.data(), readback + texel * 4, 4);
        const std::array<float, 2> value{half(bits[0]), half(bits[1])};
        motion_wrong += std::abs(value[0] - 2.0f) < 0.01f && std::abs(value[1]) < 0.01f ? 0 : 1;
        if (texel == 0) {
            sample = value;
        }
    }
    std::string upscale = "MetalFX temporal not on this device";
    bool upscale_passed = true;
    if (Metal::TemporalUpscaler::IsSupported(device)) {
        const Metal::Object color = Metal::CreateTexture(
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
        for (uint32_t texel = 0; texel < SceneSize * SceneSize; ++texel) {
            const std::array<float, 4> value{0.5f, 0.25f, 1.0f, 1.0f};
            std::memcpy(upload + texel * 16, value.data(), 16);
        }
        scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, SceneSize * 16, 0, color.Get(), 0,
                                      0, {}, {SceneSize, SceneSize, 1});
        const VideoCommon::OcarinaMatrix identity{1, 0, 0, 0, 0, 1, 0, 0,
                                                  0, 0, 1, 0, 0, 0, 0, 1};
        const char* reason = nullptr;
        bool recorded = true;
        for (u64 frame = 1; frame <= 4 && recorded; ++frame) {
            recorded = temporal.Record({.color = color.Get(),
                                        .depth = depth.Get(),
                                        .target = target.Get(),
                                        .extent = {SceneSize, SceneSize},
                                        .target_size = {TargetSize, TargetSize},
                                        .current = identity,
                                        .previous = identity,
                                        .frame = frame},
                                       reason);
        }
        if (recorded) {
            scheduler.CopyTextureToBuffer(target.Get(), 0, 0, {}, {TargetSize, TargetSize, 1},
                                          readback_buffer.Get(), 0, TargetSize * 4, 0);
        }
        scheduler.Finish();
        uint32_t wrong = 0;
        for (uint32_t texel = 0; recorded && texel < TargetSize * TargetSize; ++texel) {
            uint32_t packed;
            std::memcpy(&packed, readback + texel * 4, 4);
            const float red = PackedFloat(packed & 0x7FF, 6);
            const float green = PackedFloat((packed >> 11) & 0x7FF, 6);
            wrong += std::abs(red - 0.25f) < 0.03f && std::abs(green - 0.0625f) < 0.01f ? 0 : 1;
        }
        upscale_passed = recorded && wrong == 0;
        upscale = recorded ? "4 frames through MetalFX into 32x32 RG11B10, " +
                                 std::to_string(wrong) + " of " +
                                 std::to_string(TargetSize * TargetSize) + " wrong"
                           : std::string{"declined: "} + (reason != nullptr ? reason : "?");
    }
    const bool passed = motion_wrong == 0 && upscale_passed;
    std::printf("metal-smoke: Ocarina temporal: motion %.3f,%.3f (expected 2,0), %u of %u wrong; "
                "%s: %s\n",
                sample[0], sample[1], motion_wrong, SceneSize * SceneSize, upscale.c_str(),
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// Ocarina's temporal inputs Vulkan adds over the camera motion, without MetalFX:
/// - Link's skinned motion (EDEN_NX2_METAL_OOT_ACTOR, ocarina_actor_motion.vert/.frag through
///   the translator): a triangle over a 16x16 scene at depth 0.5 whose previous pose was half
///   of clip space to the right, over a camera that did not move. Where the scene depth is 0.5
///   (the left half) his motion is 4 pixels right; where it is 0.25 the replay is hidden and the
///   camera's motion (none) stays.
/// - The effects mask (EDEN_NX2_METAL_OOT_EFFECTS): a scene saved at 0.2, then brightened to 1.0
///   in a 4x4 block; the reactive mask is 1 over the block and one pixel around it, 0 elsewhere.
int RunOcarinaActorEffects() {
    Metal::Device device;
    Metal::Scheduler scheduler(device);
    constexpr uint32_t Size = 16;
    const Metal::Object depth = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatDepth32Float),
                 .width = Size,
                 .height = Size,
                 .render_target = true});
    const Metal::Object scene = Metal::CreateTexture(
        device, {.kind = Metal::TextureKind::e2D,
                 .format = static_cast<uint32_t>(MTLPixelFormatRGBA16Float),
                 .width = Size,
                 .height = Size});
    uint8_t* upload = nullptr;
    const Metal::Object upload_buffer = Metal::CreateBuffer(device, Size * Size * 8, true, &upload);
    uint8_t* bright = nullptr;
    const Metal::Object bright_buffer = Metal::CreateBuffer(device, Size * Size * 8, true, &bright);
    uint8_t* readback = nullptr;
    const Metal::Object readback_buffer =
        Metal::CreateBuffer(device, Size * Size * 4, true, &readback);
    uint8_t* bias_readback = nullptr;
    const Metal::Object bias_buffer = Metal::CreateBuffer(device, Size * Size * 2, true,
                                                          &bias_readback);
    if (!depth || !scene || upload == nullptr || bright == nullptr || readback == nullptr ||
        bias_readback == nullptr) {
        std::printf("metal-smoke: FAIL Ocarina actor/effects textures\n");
        return 1;
    }
    for (uint32_t y = 0; y < Size; ++y) {
        for (uint32_t x = 0; x < Size; ++x) {
            const float value = x < Size / 2 ? 0.5f : 0.25f;
            std::memcpy(upload + (y * Size + x) * 4, &value, 4);
        }
    }
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, Size * 4, 0, depth.Get(), 0, 0, {},
                                  {Size, Size, 1});
    // One vertex layout of Link's: position, joints, extra joints, weights, extra weights.
    std::vector<uint8_t> vertices(3 * Metal::OcarinaActorMesh::Stride);
    const std::array<std::array<float, 3>, 3> positions{{{-1, -1, 0}, {3, -1, 0}, {-1, 3, 0}}};
    for (size_t vertex = 0; vertex < 3; ++vertex) {
        uint8_t* const base = vertices.data() + vertex * Metal::OcarinaActorMesh::Stride;
        std::memcpy(base, positions[vertex].data(), 12);
        base[40] = 255; // all of joint 0
    }
    const std::array<uint16_t, 3> indices{0, 1, 2};
    std::vector<float> current_bones(Metal::OcarinaActorMesh::Bones * 12);
    std::vector<float> previous_bones(current_bones.size());
    const std::array<float, 12> identity_bone{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    std::copy(identity_bone.begin(), identity_bone.end(), current_bones.begin());
    std::copy(identity_bone.begin(), identity_bone.end(), previous_bones.begin());
    previous_bones[3] = 0.5f;
    const Metal::OcarinaActorMesh mesh{
        .vertices = vertices,
        .indices = indices,
        .mesh_key = 1,
        .current_origin = {},
        .previous_origin = {},
        .current_bones = current_bones,
        .previous_bones = previous_bones,
    };
    const VideoCommon::OcarinaMatrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    Metal::OcarinaTemporal temporal(device, scheduler);
    const Metal::OcarinaTemporal::ActorMotion actor{mesh, identity, identity};
    const bool encoded = temporal.EncodeMotion(depth.Get(), {Size, Size}, identity, &actor);
    if (encoded) {
        scheduler.CopyTextureToBuffer(temporal.Motion(), 0, 0, {}, {Size, Size, 1},
                                      readback_buffer.Get(), 0, Size * 4, 0);
    }
    // The effects: the saved scene at 0.2, a 4x4 block at 1.0 after it.
    const auto half = [](float value) {
        // 0.2 and 1.0 only (the test's two values).
        return static_cast<uint16_t>(value == 1.0f ? 0x3C00 : 0x3266);
    };
    for (uint32_t texel = 0; texel < Size * Size; ++texel) {
        const uint16_t dim = half(0.2f);
        const std::array<uint16_t, 4> pixel{dim, dim, dim, 0x3C00};
        std::memcpy(upload + texel * 8, pixel.data(), 8);
        const uint32_t x = texel % Size;
        const uint32_t y = texel / Size;
        const bool lit = x >= 4 && x < 8 && y >= 4 && y < 8;
        const uint16_t value = lit ? half(1.0f) : dim;
        const std::array<uint16_t, 4> after{value, value, value, 0x3C00};
        std::memcpy(bright + texel * 8, after.data(), 8);
    }
    constexpr uint64_t Frame = 5;
    scheduler.CopyBufferToTexture(upload_buffer.Get(), 0, Size * 8, 0, scene.Get(), 0, 0, {},
                                  {Size, Size, 1});
    temporal.SaveEffectsBase(scene.Get(), {Size, Size}, Frame);
    scheduler.CopyBufferToTexture(bright_buffer.Get(), 0, Size * 8, 0, scene.Get(), 0, 0, {},
                                  {Size, Size, 1});
    temporal.MarkEffects(scene.Get(), Frame);
    const bool biased = temporal.EncodeEffectsBias(Frame, {Size, Size});
    if (biased) {
        scheduler.CopyTextureToBuffer(temporal.EffectsBias(), 0, 0, {}, {Size, Size, 1},
                                      bias_buffer.Get(), 0, Size * 2, 0);
    }
    scheduler.Finish();
    const auto to_float = [](uint16_t bits) {
        const float magnitude = PackedFloat(bits & 0x7FFFU, 10);
        return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
    };
    uint32_t actor_wrong = 0;
    uint32_t camera_wrong = 0;
    std::array<float, 2> left{};
    std::array<float, 2> right{};
    for (uint32_t texel = 0; encoded && texel < Size * Size; ++texel) {
        std::array<uint16_t, 2> bits{};
        std::memcpy(bits.data(), readback + texel * 4, 4);
        const std::array<float, 2> value{to_float(bits[0]), to_float(bits[1])};
        const bool is_left = texel % Size < Size / 2;
        const float expected = is_left ? 4.0f : 0.0f;
        const bool right_value =
            std::abs(value[0] - expected) < 0.05f && std::abs(value[1]) < 0.05f;
        (is_left ? actor_wrong : camera_wrong) += right_value ? 0 : 1;
        (is_left ? left : right) = value;
    }
    uint32_t bias_wrong = 0;
    for (uint32_t texel = 0; biased && texel < Size * Size; ++texel) {
        uint16_t bits;
        std::memcpy(&bits, bias_readback + texel * 2, 2);
        const uint32_t x = texel % Size;
        const uint32_t y = texel / Size;
        const bool near = x >= 3 && x < 9 && y >= 3 && y < 9;
        bias_wrong += std::abs(to_float(bits) - (near ? 1.0f : 0.0f)) < 0.01f ? 0 : 1;
    }
    const bool passed =
        encoded && biased && actor_wrong == 0 && camera_wrong == 0 && bias_wrong == 0;
    std::printf("metal-smoke: Ocarina actor/effects: %s; Link's pixels %.3f,%.3f (expected 4,0) "
                "%u of %u wrong, hidden pixels %.3f,%.3f (expected 0,0) %u wrong; effects mask "
                "%s, %u of %u wrong: %s\n",
                encoded ? "motion encoded" : "motion NOT encoded", left[0], left[1], actor_wrong,
                Size * Size / 2, right[0], right[1], camera_wrong,
                biased ? "made" : "NOT made", bias_wrong, Size * Size, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/// The pipeline files (EDEN_NX2_METAL_PIPELINE_FILE): render states and the used list written
/// and read back, the boot marker of a build that never finished, and render pipelines built on
/// several threads against one archive (the boot build), then again from the written archive.
int RunPipelineFiles() {
    Metal::Device device;
    std::string error;
    const Metal::Object library = Metal::CreateLibrary(device, QuadSource, error);
    const Metal::Object vertex =
        library ? Metal::CreateFunction(library.Get(), "quad_vertex") : Metal::Object{};
    const Metal::Object fragment =
        library ? Metal::CreateFunction(library.Get(), "quad_fragment") : Metal::Object{};
    if (!vertex || !fragment) {
        std::printf("metal-smoke: FAIL pipeline files shaders: %s\n", error.c_str());
        return 1;
    }
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "lichii-metal-smoke" / "pipeline-files";
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
    constexpr size_t Variants = 8;
    std::vector<Metal::RenderPipelineDesc> descs;
    for (size_t index = 0; index < Variants; ++index) {
        auto desc = Metal::Zeroed<Metal::RenderPipelineDesc>();
        desc.vertex_function = vertex.Get();
        desc.fragment_function = fragment.Get();
        desc.color_formats[0] = static_cast<uint32_t>(MTLPixelFormatRGBA8Unorm);
        desc.blend[0].write_mask = static_cast<uint32_t>(index + 1);
        desc.samples = 1;
        desc.topology_class = 2;
        desc.rasterize = 1;
        desc.attributes[0] = {1, 0, 0, Metal::Maxwell::VertexAttribute::Type::Float,
                              Metal::Maxwell::VertexAttribute::Size::Size_R32_G32};
        desc.streams[0] = {1, 8, 0};
        descs.push_back(desc);
    }
    Metal::GraphicsPipelineKey key{};
    key.unique_hashes[1] = 0x1111;
    key.unique_hashes[5] = 0x5555;
    size_t states_read = 0;
    bool functions_cleared = true;
    bool boot_marked = false;
    bool listed = false;
    bool unlisted = false;
    bool rebooted = false;
    {
        Metal::PipelineFiles files;
        files.Open(directory);
        if (!files.IsOpen()) {
            std::printf("metal-smoke: FAIL pipeline files: not opened\n");
            return 1;
        }
        boot_marked = files.BeginBoot();
        for (const Metal::RenderPipelineDesc& desc : descs) {
            files.RecordState(key, desc);
            files.RecordState(key, desc);
        }
        for (u64 hash = 1; hash <= Metal::PipelineFiles::MinimumUsed; ++hash) {
            files.NoteUsed(hash);
        }
        files.SaveUsed();
        files.Wait();
    }
    {
        Metal::PipelineFiles files;
        files.Open(directory);
        // The first boot was never ended: this one must not build.
        rebooted = !files.BeginBoot();
        const auto states = files.LoadStates();
        states_read = states.size();
        for (const auto& state : states) {
            functions_cleared = functions_cleared && state.desc.vertex_function == nullptr &&
                                state.desc.fragment_function == nullptr && state.key == key;
        }
        listed = files.BuildAtBoot(1) && files.BuildAtBoot(Metal::PipelineFiles::MinimumUsed);
        unlisted = !files.BuildAtBoot(Metal::PipelineFiles::MinimumUsed + 1);
    }
    const std::filesystem::path archive_path = directory / "metal_pipelines.bin";
    const auto build_all = [&](Metal::PipelineArchive& archive) {
        std::array<bool, Variants> built{};
        std::vector<std::thread> threads;
        for (size_t index = 0; index < Variants; ++index) {
            threads.emplace_back([&, index] {
                Metal::InAutoreleasePool([&] {
                    std::string thread_error;
                    built[index] = static_cast<bool>(Metal::CreateRenderPipelineState(
                        device, descs[index], thread_error, &archive));
                });
            });
        }
        for (std::thread& thread : threads) {
            thread.join();
        }
        return std::ranges::all_of(built, [](bool value) { return value; });
    };
    Metal::PipelineArchive::Counts first{};
    bool first_built = false;
    {
        Metal::PipelineArchive archive;
        if (!archive.Open(device, archive_path)) {
            std::printf("metal-smoke: SKIP pipeline files: no MTLBinaryArchive on this device\n");
            return 0;
        }
        first_built = build_all(archive);
        first = archive.Totals();
        archive.Save();
    }
    Metal::PipelineArchive reopened;
    const bool second_built = reopened.Open(device, archive_path) && build_all(reopened);
    const Metal::PipelineArchive::Counts second = reopened.Totals();
    const bool passed = boot_marked && rebooted && states_read == Variants && functions_cleared &&
                        listed && unlisted && first_built && second_built &&
                        first.misses == Variants && first.added == Variants &&
                        second.hits == Variants && second.misses == 0;
    std::printf("metal-smoke: pipeline files: %zu of %zu render states read back (written twice "
                "each), functions cleared %s; used list %s/%s; unfinished boot seen %s; %zu "
                "pipelines on %zu threads: first %llu misses, %llu added, second %llu hits, %llu "
                "misses: %s\n",
                states_read, Variants, functions_cleared ? "yes" : "no",
                listed ? "listed" : "missing", unlisted ? "others not" : "others too",
                rebooted ? "yes" : "no", Variants, Variants,
                static_cast<unsigned long long>(first.misses),
                static_cast<unsigned long long>(first.added),
                static_cast<unsigned long long>(second.hits),
                static_cast<unsigned long long>(second.misses), passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

} // Anonymous namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc == 3 && std::strcmp(argv[1], "--probe") == 0) {
            return SmokeProbe::RunChild(argv[2]);
        }
        if (argc == 3 && std::strcmp(argv[1], "--child") == 0) {
            if (std::strcmp(argv[2], "state-cache") == 0) {
                return RunStateCache();
            }
            if (std::strcmp(argv[2], "fence-batch") == 0) {
                return RunFenceBatch();
            }
            return 4;
        }
        const int draw = Run();
        if (draw == 2) {
            return 2;
        }
        int result = draw;
        for (const auto test : {RunTranslate, RunRendererDraw, RunTexelBuffer, RunTexture1D,
                                 RunMeshEmulation, RunMeshDistribution, RunIndexRewrite,
                                 RunOcarinaBilinear, RunPipelineArchive,
                                 RunDepthBlit, RunTessellation, RunDepthStencilBlit,
                                 RunPipelineCheck, RunDeviceMsl, RunBlockCopy, RunClearRect,
                                 RunClearFold, RunStateCache, RunAhead, RunFenceIdle,
                                 RunStagingBound, RunPipelineFiles, RunOcarinaTemporal,
                                 RunOcarinaActorEffects, RunVulkanSwitches, RunPresentFilters,
                                 SmokeProbe::RunAll}) {
            const int outcome = test();
            result = result != 0 ? result : outcome;
        }
        return result;
    }
}
