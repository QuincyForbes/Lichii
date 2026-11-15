// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>

#import <Metal/Metal.h>
#include <TargetConditionals.h>

#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

namespace {

using Type = Maxwell::VertexAttribute::Type;
using Size = Maxwell::VertexAttribute::Size;

// Maxwell -> Metal, after maxwell_to_vk.cpp's tables and MoltenVK's Vulkan -> Metal maps.
MTLBlendOperation BlendOperation(Maxwell::Blend::Equation equation) {
    switch (equation) {
    case Maxwell::Blend::Equation::Subtract_D3D:
    case Maxwell::Blend::Equation::Subtract_GL:
        return MTLBlendOperationSubtract;
    case Maxwell::Blend::Equation::ReverseSubtract_D3D:
    case Maxwell::Blend::Equation::ReverseSubtract_GL:
        return MTLBlendOperationReverseSubtract;
    case Maxwell::Blend::Equation::Min_D3D:
    case Maxwell::Blend::Equation::Min_GL:
        return MTLBlendOperationMin;
    case Maxwell::Blend::Equation::Max_D3D:
    case Maxwell::Blend::Equation::Max_GL:
        return MTLBlendOperationMax;
    default:
        return MTLBlendOperationAdd;
    }
}

MTLBlendFactor BlendFactor(Maxwell::Blend::Factor factor) {
    using F = Maxwell::Blend::Factor;
    switch (factor) {
    case F::Zero_D3D:
    case F::Zero_GL:
        return MTLBlendFactorZero;
    case F::One_D3D:
    case F::One_GL:
        return MTLBlendFactorOne;
    case F::SourceColor_D3D:
    case F::SourceColor_GL:
        return MTLBlendFactorSourceColor;
    case F::OneMinusSourceColor_D3D:
    case F::OneMinusSourceColor_GL:
        return MTLBlendFactorOneMinusSourceColor;
    case F::SourceAlpha_D3D:
    case F::SourceAlpha_GL:
        return MTLBlendFactorSourceAlpha;
    case F::OneMinusSourceAlpha_D3D:
    case F::OneMinusSourceAlpha_GL:
        return MTLBlendFactorOneMinusSourceAlpha;
    case F::DestAlpha_D3D:
    case F::DestAlpha_GL:
        return MTLBlendFactorDestinationAlpha;
    case F::OneMinusDestAlpha_D3D:
    case F::OneMinusDestAlpha_GL:
        return MTLBlendFactorOneMinusDestinationAlpha;
    case F::DestColor_D3D:
    case F::DestColor_GL:
        return MTLBlendFactorDestinationColor;
    case F::OneMinusDestColor_D3D:
    case F::OneMinusDestColor_GL:
        return MTLBlendFactorOneMinusDestinationColor;
    case F::SourceAlphaSaturate_D3D:
    case F::SourceAlphaSaturate_GL:
        return MTLBlendFactorSourceAlphaSaturated;
    case F::Source1Color_D3D:
    case F::Source1Color_GL:
        return MTLBlendFactorSource1Color;
    case F::OneMinusSource1Color_D3D:
    case F::OneMinusSource1Color_GL:
        return MTLBlendFactorOneMinusSource1Color;
    case F::Source1Alpha_D3D:
    case F::Source1Alpha_GL:
        return MTLBlendFactorSource1Alpha;
    case F::OneMinusSource1Alpha_D3D:
    case F::OneMinusSource1Alpha_GL:
        return MTLBlendFactorOneMinusSource1Alpha;
    case F::BlendFactor_D3D:
    case F::ConstantColor_GL:
        return MTLBlendFactorBlendColor;
    case F::OneMinusBlendFactor_D3D:
    case F::OneMinusConstantColor_GL:
        return MTLBlendFactorOneMinusBlendColor;
    case F::BothSourceAlpha_D3D:
    case F::ConstantAlpha_GL:
        return MTLBlendFactorBlendAlpha;
    case F::OneMinusBothSourceAlpha_D3D:
    case F::OneMinusConstantAlpha_GL:
        return MTLBlendFactorOneMinusBlendAlpha;
    }
    return MTLBlendFactorOne;
}

MTLCompareFunction CompareFunction(Maxwell::ComparisonOp op) {
    using C = Maxwell::ComparisonOp;
    switch (op) {
    case C::Never_D3D:
    case C::Never_GL:
        return MTLCompareFunctionNever;
    case C::Less_D3D:
    case C::Less_GL:
        return MTLCompareFunctionLess;
    case C::Equal_D3D:
    case C::Equal_GL:
        return MTLCompareFunctionEqual;
    case C::LessEqual_D3D:
    case C::LessEqual_GL:
        return MTLCompareFunctionLessEqual;
    case C::Greater_D3D:
    case C::Greater_GL:
        return MTLCompareFunctionGreater;
    case C::NotEqual_D3D:
    case C::NotEqual_GL:
        return MTLCompareFunctionNotEqual;
    case C::GreaterEqual_D3D:
    case C::GreaterEqual_GL:
        return MTLCompareFunctionGreaterEqual;
    case C::Always_D3D:
    case C::Always_GL:
        return MTLCompareFunctionAlways;
    }
    return MTLCompareFunctionAlways;
}

MTLStencilOperation StencilOperation(Maxwell::StencilOp::Op op) {
    using S = Maxwell::StencilOp::Op;
    switch (op) {
    case S::Zero_D3D:
    case S::Zero_GL:
        return MTLStencilOperationZero;
    case S::Replace_D3D:
    case S::Replace_GL:
        return MTLStencilOperationReplace;
    case S::IncrSaturate_D3D:
    case S::IncrSaturate_GL:
        return MTLStencilOperationIncrementClamp;
    case S::DecrSaturate_D3D:
    case S::DecrSaturate_GL:
        return MTLStencilOperationDecrementClamp;
    case S::Invert_D3D:
    case S::Invert_GL:
        return MTLStencilOperationInvert;
    case S::Incr_D3D:
    case S::Incr_GL:
        return MTLStencilOperationIncrementWrap;
    case S::Decr_D3D:
    case S::Decr_GL:
        return MTLStencilOperationDecrementWrap;
    default:
        return MTLStencilOperationKeep;
    }
}

/// Scaled types are read as integers and converted by the shader (support_scaled_attributes
/// is off), as on MoltenVK.
MTLVertexFormat VertexFormat(Type type, Size size) {
    const bool normalized = type == Type::UNorm || type == Type::SNorm;
    const bool is_signed = type == Type::SNorm || type == Type::SInt || type == Type::SScaled;
    if (type == Type::Float) {
        switch (size) {
        case Size::Size_R16: return MTLVertexFormatHalf;
        case Size::Size_R16_G16: return MTLVertexFormatHalf2;
        case Size::Size_R16_G16_B16: return MTLVertexFormatHalf3;
        case Size::Size_R16_G16_B16_A16: return MTLVertexFormatHalf4;
        case Size::Size_R32: return MTLVertexFormatFloat;
        case Size::Size_R32_G32: return MTLVertexFormatFloat2;
        case Size::Size_R32_G32_B32: return MTLVertexFormatFloat3;
        case Size::Size_R32_G32_B32_A32: return MTLVertexFormatFloat4;
        case Size::Size_B10_G11_R11: return MTLVertexFormatFloatRG11B10;
        default: return MTLVertexFormatInvalid;
        }
    }
    switch (size) {
    case Size::Size_R8:
    case Size::Size_A8:
        return normalized ? (is_signed ? MTLVertexFormatCharNormalized : MTLVertexFormatUCharNormalized)
                          : (is_signed ? MTLVertexFormatChar : MTLVertexFormatUChar);
    case Size::Size_R8_G8:
    case Size::Size_G8_R8:
        return normalized ? (is_signed ? MTLVertexFormatChar2Normalized : MTLVertexFormatUChar2Normalized)
                          : (is_signed ? MTLVertexFormatChar2 : MTLVertexFormatUChar2);
    case Size::Size_R8_G8_B8:
        return normalized ? (is_signed ? MTLVertexFormatChar3Normalized : MTLVertexFormatUChar3Normalized)
                          : (is_signed ? MTLVertexFormatChar3 : MTLVertexFormatUChar3);
    case Size::Size_R8_G8_B8_A8:
    case Size::Size_X8_B8_G8_R8:
        return normalized ? (is_signed ? MTLVertexFormatChar4Normalized : MTLVertexFormatUChar4Normalized)
                          : (is_signed ? MTLVertexFormatChar4 : MTLVertexFormatUChar4);
    case Size::Size_R16:
        return normalized ? (is_signed ? MTLVertexFormatShortNormalized : MTLVertexFormatUShortNormalized)
                          : (is_signed ? MTLVertexFormatShort : MTLVertexFormatUShort);
    case Size::Size_R16_G16:
        return normalized ? (is_signed ? MTLVertexFormatShort2Normalized : MTLVertexFormatUShort2Normalized)
                          : (is_signed ? MTLVertexFormatShort2 : MTLVertexFormatUShort2);
    case Size::Size_R16_G16_B16:
        return normalized ? (is_signed ? MTLVertexFormatShort3Normalized : MTLVertexFormatUShort3Normalized)
                          : (is_signed ? MTLVertexFormatShort3 : MTLVertexFormatUShort3);
    case Size::Size_R16_G16_B16_A16:
        return normalized ? (is_signed ? MTLVertexFormatShort4Normalized : MTLVertexFormatUShort4Normalized)
                          : (is_signed ? MTLVertexFormatShort4 : MTLVertexFormatUShort4);
    case Size::Size_R32:
        return is_signed ? MTLVertexFormatInt : MTLVertexFormatUInt;
    case Size::Size_R32_G32:
        return is_signed ? MTLVertexFormatInt2 : MTLVertexFormatUInt2;
    case Size::Size_R32_G32_B32:
        return is_signed ? MTLVertexFormatInt3 : MTLVertexFormatUInt3;
    case Size::Size_R32_G32_B32_A32:
        return is_signed ? MTLVertexFormatInt4 : MTLVertexFormatUInt4;
    case Size::Size_A2_B10_G10_R10:
        // Metal has 10:10:10:2 only normalized.
        return is_signed ? MTLVertexFormatInt1010102Normalized
                         : MTLVertexFormatUInt1010102Normalized;
    default:
        return MTLVertexFormatInvalid;
    }
}

} // Anonymous namespace

std::optional<u32> PrimitiveType(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return static_cast<u32>(MTLPrimitiveTypePoint);
    case Maxwell::PrimitiveTopology::Lines:
        return static_cast<u32>(MTLPrimitiveTypeLine);
    case Maxwell::PrimitiveTopology::LineStrip:
        return static_cast<u32>(MTLPrimitiveTypeLineStrip);
    case Maxwell::PrimitiveTopology::Triangles:
        return static_cast<u32>(MTLPrimitiveTypeTriangle);
    case Maxwell::PrimitiveTopology::TriangleStrip:
        return static_cast<u32>(MTLPrimitiveTypeTriangleStrip);
    default:
        return std::nullopt;
    }
}

u32 TopologyClass(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return 0;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return 1;
    default:
        return 2;
    }
}

PipelineLimits DevicePipelineLimits(const Device& device) {
    PipelineLimits limits{.sample_counts = 0};
    for (const u32 samples : {1U, 2U, 4U, 8U}) {
        if ([device.GetDevice() supportsTextureSampleCount:samples]) {
            limits.sample_counts |= 1U << samples;
        }
    }
    return limits;
}

Object CreateStageInComputePipelineState(const Device& device, const void* function,
                                         const RenderPipelineDesc& layout, u32 index_size,
                                         u32 index_slot, std::string& error,
                                         PipelineArchive* archive) {
    if (PipelineCheckEnabled()) {
        // The descriptor's arrays abort the process past their ends.
        error = CheckVertexLayout(layout);
        if (!error.empty()) {
            return {};
        }
    }
    // MTLAttributeFormat has MTLVertexFormat's values.
    MTLStageInputOutputDescriptor* stage_in = [MTLStageInputOutputDescriptor new];
    for (size_t index = 0; index < layout.attributes.size(); ++index) {
        const VertexAttributeState& attribute = layout.attributes[index];
        const MTLVertexFormat format =
            attribute.enabled != 0 ? VertexFormat(attribute.type, attribute.size)
                                   : MTLVertexFormatInvalid;
        if (format == MTLVertexFormatInvalid) {
            continue;
        }
        stage_in.attributes[index].format = static_cast<MTLAttributeFormat>(format);
        stage_in.attributes[index].offset = attribute.offset;
        stage_in.attributes[index].bufferIndex = VertexBufferSlot(attribute.stream);
    }
    for (size_t index = 0; index < layout.streams.size(); ++index) {
        const VertexStreamState& stream = layout.streams[index];
        if (stream.enabled == 0) {
            continue;
        }
        MTLBufferLayoutDescriptor* buffer =
            stage_in.layouts[VertexBufferSlot(static_cast<u32>(index))];
        if (stream.stride == 0 || stream.divisor == ConstantStep) {
            buffer.stepFunction = MTLStepFunctionConstant;
            buffer.stepRate = 0;
            buffer.stride = stream.stride != 0 ? stream.stride : 16;
        } else if (stream.divisor != 0) {
            buffer.stepFunction = MTLStepFunctionThreadPositionInGridY;
            buffer.stepRate = stream.divisor;
            buffer.stride = stream.stride;
        } else {
            buffer.stepFunction = index_size != 0 ? MTLStepFunctionThreadPositionInGridXIndexed
                                                  : MTLStepFunctionThreadPositionInGridX;
            buffer.stride = stream.stride;
        }
    }
    if (index_size != 0) {
        stage_in.indexType = index_size == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
        stage_in.indexBufferIndex = index_slot;
    }
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = (__bridge id<MTLFunction>)function;
    descriptor.stageInputDescriptor = stage_in;
    return CreateComputeState(device, descriptor, error, archive);
}

Object CreateRenderPipelineState(const Device& device, const RenderPipelineDesc& desc,
                                 std::string& error, PipelineArchive* archive) {
    if (PipelineCheckEnabled()) {
        // What Metal answers with an assertion instead of an error (metal_pipeline_check.cpp).
        error = CheckPipeline(desc, DevicePipelineLimits(device));
        if (!error.empty()) {
            return {};
        }
    }
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = (__bridge id<MTLFunction>)desc.vertex_function;
    descriptor.fragmentFunction = (__bridge id<MTLFunction>)desc.fragment_function;
    descriptor.rasterizationEnabled = desc.rasterize != 0 ? YES : NO;
    descriptor.rasterSampleCount = (std::max)(desc.samples, 1U);
    descriptor.alphaToCoverageEnabled = desc.alpha_to_coverage != 0 ? YES : NO;
    if (desc.tessellation != 0) {
        // The control kernel writes half-precision factors, one set per patch.
        descriptor.tessellationPartitionMode =
            static_cast<MTLTessellationPartitionMode>(desc.tess_partition);
        descriptor.tessellationOutputWindingOrder = static_cast<MTLWinding>(desc.tess_winding);
        descriptor.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
        descriptor.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
        descriptor.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
        descriptor.tessellationFactorScaleEnabled = NO;
#if TARGET_OS_IPHONE
        descriptor.maxTessellationFactor = 16;
#else
        descriptor.maxTessellationFactor = 64;
#endif
    } else {
        descriptor.inputPrimitiveTopology = desc.topology_class == 0
                                                ? MTLPrimitiveTopologyClassPoint
                                            : desc.topology_class == 1
                                                ? MTLPrimitiveTopologyClassLine
                                                : MTLPrimitiveTopologyClassTriangle;
    }
    for (size_t index = 0; index < desc.color_formats.size(); ++index) {
        MTLRenderPipelineColorAttachmentDescriptor* color = descriptor.colorAttachments[index];
        color.pixelFormat = static_cast<MTLPixelFormat>(desc.color_formats[index]);
        if (desc.color_formats[index] == 0) {
            continue;
        }
        const BlendTargetState& blend = desc.blend[index];
        MTLColorWriteMask mask = MTLColorWriteMaskNone;
        mask |= (blend.write_mask & 1) != 0 ? MTLColorWriteMaskRed : MTLColorWriteMaskNone;
        mask |= (blend.write_mask & 2) != 0 ? MTLColorWriteMaskGreen : MTLColorWriteMaskNone;
        mask |= (blend.write_mask & 4) != 0 ? MTLColorWriteMaskBlue : MTLColorWriteMaskNone;
        mask |= (blend.write_mask & 8) != 0 ? MTLColorWriteMaskAlpha : MTLColorWriteMaskNone;
        color.writeMask = mask;
        // Metal's descriptor check aborts the process for blending on an integer target; the
        // hardware ignores it there.
        const FormatClass format_class = ClassOf(desc.color_formats[index]);
        const bool blending = blend.enable != 0 && format_class != FormatClass::Uint &&
                              format_class != FormatClass::Sint;
        color.blendingEnabled = blending ? YES : NO;
        if (blending) {
            color.rgbBlendOperation = BlendOperation(blend.color_op);
            color.alphaBlendOperation = BlendOperation(blend.alpha_op);
            color.sourceRGBBlendFactor = BlendFactor(blend.color_source);
            color.destinationRGBBlendFactor = BlendFactor(blend.color_dest);
            color.sourceAlphaBlendFactor = BlendFactor(blend.alpha_source);
            color.destinationAlphaBlendFactor = BlendFactor(blend.alpha_dest);
        }
    }
    descriptor.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(desc.depth_format);
    descriptor.stencilAttachmentPixelFormat = static_cast<MTLPixelFormat>(desc.stencil_format);

    MTLVertexDescriptor* vertex = [MTLVertexDescriptor vertexDescriptor];
    for (size_t index = 0; index < desc.attributes.size(); ++index) {
        const VertexAttributeState& attribute = desc.attributes[index];
        if (attribute.enabled == 0) {
            continue;
        }
        const MTLVertexFormat format = VertexFormat(attribute.type, attribute.size);
        if (format == MTLVertexFormatInvalid) {
            continue;
        }
        vertex.attributes[index].format = format;
        vertex.attributes[index].offset = attribute.offset;
        vertex.attributes[index].bufferIndex = VertexBufferSlot(attribute.stream);
    }
    for (size_t index = 0; index < desc.streams.size(); ++index) {
        const VertexStreamState& stream = desc.streams[index];
        if (stream.enabled == 0) {
            continue;
        }
        MTLVertexBufferLayoutDescriptor* layout =
            vertex.layouts[VertexBufferSlot(static_cast<u32>(index))];
        if (stream.stride == 0 || stream.divisor == ConstantStep) {
            // One element for every vertex (Metal wants a non-zero stride otherwise).
            layout.stepFunction = MTLVertexStepFunctionConstant;
            layout.stepRate = 0;
            layout.stride = stream.stride != 0 ? stream.stride : 16;
        } else if (stream.divisor != 0) {
            layout.stepFunction = MTLVertexStepFunctionPerInstance;
            layout.stepRate = stream.divisor;
            layout.stride = stream.stride;
        } else {
            layout.stepFunction = MTLVertexStepFunctionPerVertex;
            layout.stride = stream.stride;
        }
    }
    descriptor.vertexDescriptor = vertex;

    const id<MTLBinaryArchive> binary =
        archive != nullptr ? (__bridge id<MTLBinaryArchive>)archive->Handle() : nil;
    const auto start = std::chrono::steady_clock::now();
    id<MTLRenderPipelineState> state = nil;
    if (binary != nil) {
        descriptor.binaryArchives = @[ binary ];
        state = [device.GetDevice() newRenderPipelineStateWithDescriptor:descriptor
                                                                 options:
                                                                     MTLPipelineOptionFailOnBinaryArchiveMiss
                                                              reflection:nil
                                                                   error:nil];
    }
    const bool hit = state != nil;
    NSError* ns_error = nil;
    if (!hit) {
        state = [device.GetDevice() newRenderPipelineStateWithDescriptor:descriptor
                                                                    error:&ns_error];
    }
    if (state == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
    }
    if (binary != nil && state != nil) {
        archive->NoteBuild(hit, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - start)
                                                     .count()));
        if (!hit) {
            archive->NoteAdded([binary addRenderPipelineFunctionsWithDescriptor:descriptor
                                                                          error:nil]);
        }
    }
    return Retain(state);
}

Object CreateDepthStencilState(const Device& device, const DepthStencilDesc& desc) {
    MTLDepthStencilDescriptor* descriptor = [MTLDepthStencilDescriptor new];
    descriptor.depthCompareFunction =
        desc.depth_test != 0 ? CompareFunction(desc.depth_func) : MTLCompareFunctionAlways;
    descriptor.depthWriteEnabled = desc.depth_test != 0 && desc.depth_write != 0 ? YES : NO;
    if (desc.stencil_test != 0) {
        const auto face = [](const StencilFaceState& state) {
            MTLStencilDescriptor* stencil = [MTLStencilDescriptor new];
            stencil.stencilFailureOperation = StencilOperation(state.fail);
            stencil.depthFailureOperation = StencilOperation(state.depth_fail);
            stencil.depthStencilPassOperation = StencilOperation(state.pass);
            stencil.stencilCompareFunction = CompareFunction(state.func);
            stencil.readMask = state.read_mask;
            stencil.writeMask = state.write_mask;
            return stencil;
        };
        descriptor.frontFaceStencil = face(desc.front);
        descriptor.backFaceStencil = face(desc.back);
    }
    return Retain([device.GetDevice() newDepthStencilStateWithDescriptor:descriptor]);
}

} // namespace Metal
