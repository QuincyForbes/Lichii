// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>

#include "common/logging.h"
#include "video_core/renderer_metal/metal_archive.h"
#include "video_core/renderer_metal/metal_bridge.h"
#include "video_core/renderer_metal/metal_device.h"
#include "video_core/renderer_metal/metal_formats.h"
#include "video_core/renderer_metal/metal_graphics.h"
#include "video_core/renderer_metal/metal_resources.h"

namespace Metal {

using VideoCore::Surface::PixelFormat;

namespace {

// metal_formats.h names Metal's formats by number.
#define X(name, value) static_assert(static_cast<u32>(MTLPixelFormat##name) == value);
METAL_PIXEL_FORMATS(X)
#undef X
#define X(name) static_assert(static_cast<u32>(MTLTextureSwizzle##name) == u32{Swizzle::name});
X(Zero) X(One) X(Red) X(Green) X(Blue) X(Alpha)
#undef X

MTLTextureType TextureType(TextureKind kind) {
    switch (kind) {
    case TextureKind::e1D: return MTLTextureType1D;
    case TextureKind::e1DArray: return MTLTextureType1DArray;
    case TextureKind::e2D: return MTLTextureType2D;
    case TextureKind::e2DArray: return MTLTextureType2DArray;
    case TextureKind::e2DMultisample: return MTLTextureType2DMultisample;
    case TextureKind::e2DMultisampleArray: return MTLTextureType2DMultisampleArray;
    case TextureKind::e3D: return MTLTextureType3D;
    case TextureKind::Cube: return MTLTextureTypeCube;
    case TextureKind::CubeArray: return MTLTextureTypeCubeArray;
    }
    return MTLTextureType2D;
}

TextureKind KindOf(MTLTextureType type) {
    switch (type) {
    case MTLTextureType1D: return TextureKind::e1D;
    case MTLTextureType1DArray: return TextureKind::e1DArray;
    case MTLTextureType2DArray: return TextureKind::e2DArray;
    case MTLTextureType2DMultisample: return TextureKind::e2DMultisample;
    case MTLTextureType2DMultisampleArray: return TextureKind::e2DMultisampleArray;
    case MTLTextureType3D: return TextureKind::e3D;
    case MTLTextureTypeCube: return TextureKind::Cube;
    case MTLTextureTypeCubeArray: return TextureKind::CubeArray;
    default: return TextureKind::e2D;
    }
}

/// Bytes of a texel of a plain colour format, 0 for the others.
u32 TexelBytes(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatA8Unorm:
    case MTLPixelFormatR8Unorm:
    case MTLPixelFormatR8Unorm_sRGB:
    case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint:
    case MTLPixelFormatR8Sint:
        return 1;
    case MTLPixelFormatR16Unorm:
    case MTLPixelFormatR16Snorm:
    case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint:
    case MTLPixelFormatR16Float:
    case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Unorm_sRGB:
    case MTLPixelFormatRG8Snorm:
    case MTLPixelFormatRG8Uint:
    case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm:
    case MTLPixelFormatA1BGR5Unorm:
    case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm:
        return 2;
    case MTLPixelFormatR32Uint:
    case MTLPixelFormatR32Sint:
    case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm:
    case MTLPixelFormatRG16Snorm:
    case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint:
    case MTLPixelFormatRG16Float:
    case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB:
    case MTLPixelFormatRGBA8Snorm:
    case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint:
    case MTLPixelFormatBGRA8Unorm:
    case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm:
    case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatBGR10A2Unorm:
    case MTLPixelFormatRG11B10Float:
    case MTLPixelFormatRGB9E5Float:
        return 4;
    case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm:
    case MTLPixelFormatRGBA16Snorm:
    case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint:
    case MTLPixelFormatRGBA16Float:
        return 8;
    case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGBA32Sint:
    case MTLPixelFormatRGBA32Float:
        return 16;
    default:
        return 0;
    }
}

/// MoltenVK's view classes, 0 for a format in none: plain colour formats by texel size, a
/// compressed format with its sRGB, signed and HDR variants.
u32 ViewClass(MTLPixelFormat format) {
    constexpr u32 Compressed = 0x100;
    if (const u32 bytes = TexelBytes(format)) {
        return bytes;
    }
    const u32 raw = static_cast<u32>(format);
    // Each of these is followed by its variant.
    if ((format >= MTLPixelFormatBC1_RGBA && format <= MTLPixelFormatBC7_RGBAUnorm_sRGB) ||
        (format >= MTLPixelFormatEAC_RGBA8 && format <= MTLPixelFormatETC2_RGB8A1_sRGB)) {
        return Compressed + (raw & ~1U);
    }
    if (format == MTLPixelFormatEAC_R11Unorm || format == MTLPixelFormatEAC_R11Snorm) {
        return Compressed + static_cast<u32>(MTLPixelFormatEAC_R11Unorm);
    }
    if (format == MTLPixelFormatEAC_RG11Unorm || format == MTLPixelFormatEAC_RG11Snorm) {
        return Compressed + static_cast<u32>(MTLPixelFormatEAC_RG11Unorm);
    }
    // ASTC's sRGB, LDR and HDR formats list the block sizes in one order.
    for (const MTLPixelFormat first : {MTLPixelFormatASTC_4x4_sRGB, MTLPixelFormatASTC_4x4_LDR,
                                       MTLPixelFormatASTC_4x4_HDR}) {
        const u32 last = static_cast<u32>(first) + static_cast<u32>(MTLPixelFormatASTC_12x12_LDR) -
                         static_cast<u32>(MTLPixelFormatASTC_4x4_LDR);
        if (raw >= static_cast<u32>(first) && raw <= last) {
            return Compressed + raw - static_cast<u32>(first) +
                   static_cast<u32>(MTLPixelFormatASTC_4x4_LDR);
        }
    }
    return 0;
}

/// Why Metal would reject by assertion a view in format `to` of a texture in format `from`, or
/// nil: it casts between colour formats of one texel size, between the variants of a compressed
/// format and to the stencil half of a depth-stencil format, nothing else.
const char* ViewFormatReason(MTLPixelFormat from, MTLPixelFormat to) {
    if (from == to) {
        return nullptr;
    }
    const auto depth_stencil = [](MTLPixelFormat format) {
        const FormatClass format_class = ClassOf(static_cast<u32>(format));
        return format_class == FormatClass::Depth || format_class == FormatClass::Stencil ||
               format_class == FormatClass::DepthStencil || format == MTLPixelFormatX32_Stencil8;
    };
    if (depth_stencil(from) || depth_stencil(to)) {
        return from == MTLPixelFormatDepth32Float_Stencil8 && to == MTLPixelFormatX32_Stencil8
                   ? nullptr
                   : "a depth or stencil format seen as another format";
    }
    const u32 from_class = ViewClass(from);
    const u32 to_class = ViewClass(to);
    if (from_class != 0 && to_class != 0 && from_class != to_class) {
        return "formats of different texel sizes or compressions";
    }
    return nullptr;
}

} // Anonymous namespace

u64 DeviceMemoryBudget(const Device& device) {
    return device.MemoryBudget();
}

u32 DeviceAppleFamily(const Device& device) {
    return device.AppleFamily();
}

bool IsConverted(const Device& device, PixelFormat format) {
    if (VideoCore::Surface::IsPixelFormatASTC(format)) {
        // Apple GPUs (every Apple GPU family) read ASTC; Intel and AMD Macs don't.
        return device.AppleFamily() == 0;
    }
    if (VideoCore::Surface::IsPixelFormatBCn(format)) {
        return !device.SupportsBCTextures();
    }
    return false;
}

FormatInfo SurfaceFormat(const Device&, PixelFormat format, bool converted) {
    return SurfaceFormat(format, converted);
}

u32 DrawableFormat() {
    return static_cast<u32>(MTLPixelFormatBGRA8Unorm);
}

Object CreateTexture(const Device& device, const TextureDesc& desc) {
    if (PipelineCheckEnabled()) {
        // Metal aborts the process for a texture descriptor it rejects.
        const std::string reason = CheckTexture(desc, DevicePipelineLimits(device));
        if (!reason.empty()) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture refused: {}; kind {} format {} "
                            "{}x{}x{} levels {} layers {} samples {}",
                            reason, static_cast<u32>(desc.kind), desc.format, desc.width,
                            desc.height, desc.depth, desc.levels, desc.layers, desc.samples);
            }
            return {};
        }
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor new];
    descriptor.textureType = TextureType(desc.kind);
    descriptor.pixelFormat = static_cast<MTLPixelFormat>(desc.format);
    descriptor.width = (std::max)(desc.width, 1U);
    descriptor.height = (std::max)(desc.height, 1U);
    descriptor.depth = (std::max)(desc.depth, 1U);
    descriptor.mipmapLevelCount = (std::max)(desc.levels, 1U);
    // Cube textures count faces in arrayLength / 6.
    const bool cube = desc.kind == TextureKind::Cube || desc.kind == TextureKind::CubeArray;
    descriptor.arrayLength = (std::max)(cube ? desc.layers / 6 : desc.layers, 1U);
    descriptor.sampleCount = (std::max)(desc.samples, 1U);
    descriptor.storageMode = MTLStorageModePrivate;
    // Format views reinterpret texture cache images (ImageView formats, copies between
    // compatible formats). It costs lossless compression on some GPUs; narrowing it to images
    // that need it is a later step (docs/nx2/metal-renderer.md).
    MTLTextureUsage usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (desc.render_target) {
        usage |= MTLTextureUsageRenderTarget;
    }
    if (desc.shader_write) {
        usage |= MTLTextureUsageShaderWrite;
    }
    descriptor.usage = usage;
    // Metal orders the encoders until the scheduler tracks hazards itself (untracked heaps).
    descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
    return Retain([device.GetDevice() newTextureWithDescriptor:descriptor]);
}

Object CreateTextureView(TextureHandle texture, u32 format, TextureKind kind, u32 base_level,
                         u32 levels, u32 base_layer, u32 layers, std::array<u8, 4> swizzle) {
    const id<MTLTexture> base = AsTexture(texture);
    if (base == nil) {
        return {};
    }
    if (PipelineCheckEnabled()) {
        // A range of levels or slices the texture doesn't have, a view of another family of
        // texture types and a format Metal doesn't cast to abort the process.
        const bool cube = base.textureType == MTLTextureTypeCube ||
                          base.textureType == MTLTextureTypeCubeArray;
        const u64 slices =
            base.textureType == MTLTextureType3D ? 1 : u64{base.arrayLength} * (cube ? 6 : 1);
        std::string reason;
        if (levels == 0 || layers == 0 || u64{base_level} + levels > base.mipmapLevelCount ||
            u64{base_layer} + layers > slices) {
            reason = "levels or slices past the texture";
        } else if (reason = CheckTextureView(KindOf(base.textureType), kind, layers);
                   reason.empty()) {
            if (const char* const cast = ViewFormatReason(
                    base.pixelFormat, static_cast<MTLPixelFormat>(format))) {
                reason = cast;
            }
        }
        if (!reason.empty()) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture view refused: {}; levels {}+{} of {}, "
                            "slices {}+{} of {}, kind {} of {}, format {} of {}",
                            reason, base_level, levels, static_cast<u32>(base.mipmapLevelCount),
                            base_layer, layers, slices, static_cast<u32>(kind),
                            static_cast<u32>(KindOf(base.textureType)), format,
                            static_cast<u32>(base.pixelFormat));
            }
            return {};
        }
    }
    const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
        static_cast<MTLTextureSwizzle>(swizzle[0]), static_cast<MTLTextureSwizzle>(swizzle[1]),
        static_cast<MTLTextureSwizzle>(swizzle[2]), static_cast<MTLTextureSwizzle>(swizzle[3]));
    return Retain([base newTextureViewWithPixelFormat:static_cast<MTLPixelFormat>(format)
                                          textureType:TextureType(kind)
                                               levels:NSMakeRange(base_level, levels)
                                               slices:NSMakeRange(base_layer, layers)
                                              swizzle:channels]);
}

u32 TextureBufferAlignment(const Device& device, u32 format) {
    return static_cast<u32>([device.GetDevice()
        minimumTextureBufferAlignmentForPixelFormat:static_cast<MTLPixelFormat>(format)]);
}

Object CreateTextureBufferView(BufferHandle buffer, u64 offset, u32 elements, u32 bytes_per_texel,
                               u32 format, bool writable) {
    const id<MTLBuffer> object = AsBuffer(buffer);
    if (object == nil || elements == 0) {
        return {};
    }
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (writable) {
        usage |= MTLTextureUsageShaderWrite;
    }
    MTLTextureDescriptor* descriptor = [MTLTextureDescriptor
        textureBufferDescriptorWithPixelFormat:static_cast<MTLPixelFormat>(format)
                                         width:elements
                               resourceOptions:object.resourceOptions
                                         usage:usage];
    return Retain([object newTextureWithDescriptor:descriptor
                                            offset:offset
                                       bytesPerRow:u64{elements} * bytes_per_texel]);
}

Object CreateBuffer(const Device& device, u64 size, bool shared, u8** contents) {
    const MTLResourceOptions options =
        shared ? (MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache)
               : MTLResourceStorageModePrivate;
    const id<MTLBuffer> buffer = [device.GetDevice() newBufferWithLength:(std::max)(size, u64{16})
                                                                 options:options];
    if (contents != nullptr) {
        *contents = buffer != nil && shared ? static_cast<u8*>(buffer.contents) : nullptr;
    }
    return Retain(buffer);
}

Object CreateLinearClampSampler(const Device& device) {
    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = MTLSamplerMinMagFilterLinear;
    descriptor.magFilter = MTLSamplerMinMagFilterLinear;
    descriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    descriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
}

Object CreateSampler(const Device& device, const Tegra::Texture::TSCEntry& tsc) {
    using Tegra::Texture::TextureFilter;
    using Tegra::Texture::TextureMipmapFilter;
    using Tegra::Texture::WrapMode;
    const auto filter = [](TextureFilter value) {
        return value == TextureFilter::Linear ? MTLSamplerMinMagFilterLinear
                                              : MTLSamplerMinMagFilterNearest;
    };
    const auto address = [](WrapMode mode) {
        switch (mode) {
        case WrapMode::Wrap: return MTLSamplerAddressModeRepeat;
        case WrapMode::Mirror: return MTLSamplerAddressModeMirrorRepeat;
        case WrapMode::ClampToEdge:
        case WrapMode::Clamp:
        case WrapMode::MirrorOnceClampOGL: return MTLSamplerAddressModeClampToEdge;
        case WrapMode::Border:
        case WrapMode::MirrorOnceBorder: return MTLSamplerAddressModeClampToBorderColor;
        case WrapMode::MirrorOnceClampToEdge: return MTLSamplerAddressModeMirrorClampToEdge;
        }
        return MTLSamplerAddressModeRepeat;
    };
    MTLSamplerDescriptor* descriptor = [MTLSamplerDescriptor new];
    descriptor.minFilter = filter(tsc.min_filter.Value());
    descriptor.magFilter = filter(tsc.mag_filter.Value());
    switch (tsc.mipmap_filter.Value()) {
    case TextureMipmapFilter::Nearest: descriptor.mipFilter = MTLSamplerMipFilterNearest; break;
    case TextureMipmapFilter::Linear: descriptor.mipFilter = MTLSamplerMipFilterLinear; break;
    default: descriptor.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
    }
    descriptor.sAddressMode = address(tsc.wrap_u.Value());
    descriptor.tAddressMode = address(tsc.wrap_v.Value());
    descriptor.rAddressMode = address(tsc.wrap_p.Value());
    // Metal has three border colours: the guest's nearest one.
    const auto border = tsc.BorderColor();
    if (border[3] < 0.5f) {
        descriptor.borderColor = MTLSamplerBorderColorTransparentBlack;
    } else if (border[0] + border[1] + border[2] > 1.5f) {
        descriptor.borderColor = MTLSamplerBorderColorOpaqueWhite;
    } else {
        descriptor.borderColor = MTLSamplerBorderColorOpaqueBlack;
    }
    if (tsc.depth_compare_enabled != 0) {
        // Tegra's DepthCompareFunc has MTLCompareFunction's values.
        descriptor.compareFunction = static_cast<MTLCompareFunction>(tsc.depth_compare_func.Value());
    }
    descriptor.lodMinClamp = tsc.MinLod();
    descriptor.lodMaxClamp = (std::max)(tsc.MaxLod(), tsc.MinLod());
    descriptor.maxAnisotropy =
        static_cast<NSUInteger>(std::clamp(tsc.MaxAnisotropy(), 1.0f, 16.0f));
    descriptor.normalizedCoordinates = YES;
    descriptor.supportArgumentBuffers = YES;
    return Retain([device.GetDevice() newSamplerStateWithDescriptor:descriptor]);
}

namespace {
[[nodiscard]] NSString* ToNSString(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}
} // Anonymous namespace

Object CreateLibrary(const Device& device, std::string_view source, std::string& error) {
    MTLCompileOptions* options = [MTLCompileOptions new];
    // The recompiler marks what must stay exact (NoContraction); the rest may use fast math.
    options.fastMathEnabled = YES;
    NSError* ns_error = nil;
    const id<MTLLibrary> library = [device.GetDevice() newLibraryWithSource:ToNSString(source)
                                                                    options:options
                                                                      error:&ns_error];
    if (library == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
    }
    return Retain(library);
}

Object CreateFunction(const void* library, std::string_view name) {
    const id<MTLLibrary> object = (__bridge id<MTLLibrary>)library;
    return object != nil ? Retain([object newFunctionWithName:ToNSString(name)]) : Object{};
}

Object CreateComputeState(const Device& device, MTLComputePipelineDescriptor* descriptor,
                          std::string& error, PipelineArchive* archive) {
    const id<MTLBinaryArchive> binary =
        archive != nullptr ? (__bridge id<MTLBinaryArchive>)archive->Handle() : nil;
    const auto start = std::chrono::steady_clock::now();
    id<MTLComputePipelineState> state = nil;
    if (binary != nil) {
        descriptor.binaryArchives = @[ binary ];
        state = [device.GetDevice()
            newComputePipelineStateWithDescriptor:descriptor
                                          options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                       reflection:nil
                                            error:nil];
    }
    const bool hit = state != nil;
    NSError* ns_error = nil;
    if (!hit) {
        state = [device.GetDevice() newComputePipelineStateWithDescriptor:descriptor
                                                                  options:MTLPipelineOptionNone
                                                               reflection:nil
                                                                    error:&ns_error];
    }
    if (state == nil) {
        error = ns_error != nil ? ns_error.localizedDescription.UTF8String : "unknown error";
        return {};
    }
    if (binary != nil) {
        archive->NoteBuild(hit, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                     std::chrono::steady_clock::now() - start)
                                                     .count()));
        if (!hit) {
            archive->NoteAdded([binary addComputePipelineFunctionsWithDescriptor:descriptor
                                                                           error:nil]);
        }
    }
    return Retain(state);
}

Object CreateComputePipelineState(const Device& device, const void* function, std::string& error,
                                  PipelineArchive* archive) {
    MTLComputePipelineDescriptor* descriptor = [MTLComputePipelineDescriptor new];
    descriptor.computeFunction = (__bridge id<MTLFunction>)function;
    return CreateComputeState(device, descriptor, error, archive);
}

u64 AllocatedSize(const void* resource) {
    const id<MTLResource> object = (__bridge id<MTLResource>)resource;
    return object != nil ? object.allocatedSize : 0;
}

void SetLabel(const void* resource, std::string_view label) {
    const id<MTLResource> object = (__bridge id<MTLResource>)resource;
    if (object != nil) {
        object.label = [[NSString alloc] initWithBytes:label.data()
                                                length:label.size()
                                              encoding:NSUTF8StringEncoding];
    }
}

} // namespace Metal
