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

} // Anonymous namespace

u64 DeviceMemoryBudget(const Device& device) {
    return device.MemoryBudget();
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
        // A range of levels or slices the texture doesn't have aborts the process.
        const bool cube = base.textureType == MTLTextureTypeCube ||
                          base.textureType == MTLTextureTypeCubeArray;
        const u64 slices =
            base.textureType == MTLTextureType3D ? 1 : u64{base.arrayLength} * (cube ? 6 : 1);
        if (levels == 0 || layers == 0 || u64{base_level} + levels > base.mipmapLevelCount ||
            u64{base_layer} + layers > slices) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Metal, "Metal: texture view refused: levels {}+{} of {}, "
                            "slices {}+{} of {}, kind {} format {}",
                            base_level, levels, static_cast<u32>(base.mipmapLevelCount),
                            base_layer, layers, slices, static_cast<u32>(kind), format);
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
